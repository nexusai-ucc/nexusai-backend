"""
Harness de evaluación RAG end-to-end — corre la metodología documentada en
investigacion/02-rag/evaluacion-rag.md contra el backend real (services/api).

Qué hace, en orden:
  1. Indexa el apunte de prueba (investigacion/02-rag/fixtures/apunte-bases-de-datos.md)
     en un curso dedicado, subiéndolo por el endpoint real POST /api/v1/documents
     (HMAC-firmado, igual que el plugin PHP) y esperando a que termine de indexar.
  2. Nivel 1 (retrieval): para cada pregunta con material real (categorías A y B
     del dataset), llama directamente a `retrieve_context()` — la misma función
     que usa /chat/messages — y calcula Recall@5, Precision@5 y MRR contra las
     `ground_truth_keywords` del dataset.
  3. Nivel 2 (generación / faithfulness): llama al endpoint real POST
     /api/v1/chat/messages para obtener la respuesta del LLM en producción, y
     un segundo LLM-as-judge (prompt tomado literal de evaluacion-rag.md) la
     clasifica FIEL / PARCIAL / ALUCINADO contra el contexto recuperado.
  4. Nivel 3 (fallback honesto): para las preguntas fuera del material
     (categoría C), llama al mismo endpoint y reusa
     `app.gaps.recorder.llm_indicated_no_answer()` — la función real que usa
     producción para decidir si el LLM admitió que no sabía — para medir si
     el fallback ocurrió.
  5. Imprime un reporte y lo escribe como JSON reproducible.

Uso (correr DENTRO del container de la API, donde million `app.*` es importable
y las env vars de .env ya están cargadas):

    docker exec nexusai-api python scripts/eval_rag.py \
        --material /app/eval/apunte-bases-de-datos.md \
        --dataset /app/eval/eval-dataset.json \
        --out /app/eval/reporte.json

Requiere que la API esté corriendo en localhost:8000 dentro del mismo container
(uvicorn) y que Postgres/Redis estén accesibles con las mismas env vars que usa
la API (DATABASE_URL, NEXUSAI_API_KEY, NEXUSAI_SHARED_SECRET, etc.).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import hmac as hmac_lib
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import get_session_factory, dispose_engine  # noqa: E402
from app.documents.retriever import retrieve_context  # noqa: E402
from app.gaps.recorder import llm_indicated_no_answer  # noqa: E402
from app.providers.embeddings import get_embedding_provider  # noqa: E402
from app.providers.llm import LLMProvider  # noqa: E402
from app.shared.config import get_settings  # noqa: E402
from sqlalchemy import delete  # noqa: E402
from app.db.models import Document  # noqa: E402


API_BASE_URL = "http://localhost:8000"

# Umbrales reales usados en producción (leídos del código, no inventados):
#   - chat/router.py: retrieve_context(top_k=5, min_similarity=0.3)
#   - gaps/recorder.py: WEAK_MATCH_THRESHOLD = 0.4 (umbral de "contexto débil")
#   - quiz/router.py: QUIZ_TOPIC_MIN_SIMILARITY = 0.5
#   - search/router.py: filtro semántico >= 0.32, score combinado >= 0.35
CHAT_TOP_K = 5
CHAT_MIN_SIMILARITY = 0.3

# El tier gratuito de Gemini limita a 5 requests/min al modelo primario
# (generativelanguage.googleapis.com/generate_content_free_tier_requests).
# Cada pregunta consume 2 llamadas al LLM (respuesta + judge), así que
# espaciamos las llamadas para no gatillar 429 y flapear al fallback.
LLM_CALL_PACING_SEC = 13.0

JUDGE_PROMPT_TEMPLATE = """
Sos un evaluador estricto. Dado el CONTEXTO y la RESPUESTA, decidí:

- FIEL: toda afirmación de la respuesta se puede verificar en el contexto.
- PARCIAL: la respuesta mezcla info del contexto con info que no está.
- ALUCINADO: la respuesta contiene afirmaciones que no están en el contexto.

Contexto: {context}
Respuesta: {answer}

Responde solo con una palabra: FIEL, PARCIAL, ALUCINADO.
""".strip()


def sign_request(body: bytes) -> dict[str, str]:
    """Firma HMAC de la request, igual que el plugin PHP (ver app/auth/hmac.py)."""
    settings = get_settings()
    timestamp = str(int(time.time()))
    nonce = str(uuid.uuid4())
    signed_string = (timestamp + nonce).encode("utf-8") + body
    signature = hmac_lib.new(
        settings.nexusai_shared_secret.encode("utf-8"),
        signed_string,
        hashlib.sha256,
    ).hexdigest()
    return {
        "Authorization": f"Bearer {settings.nexusai_api_key}",
        "X-Timestamp": timestamp,
        "X-Nonce": nonce,
        "X-Signature": signature,
        "Content-Type": "application/json",
    }


async def signed_post(client: httpx.AsyncClient, path: str, payload: dict[str, Any]) -> httpx.Response:
    body = json.dumps(payload).encode("utf-8")
    headers = sign_request(body)
    return await client.post(f"{API_BASE_URL}{path}", content=body, headers=headers)


async def signed_post_with_retry(
    client: httpx.AsyncClient,
    path: str,
    payload: dict[str, Any],
    *,
    max_attempts: int = 4,
    backoff_sec: float = 25.0,
) -> httpx.Response:
    """
    Reintenta en 503 (saturación transitoria del tier gratuito de Gemini,
    ver .env.example: "gemini-2.5-flash está saturado... 503 seguidos").
    No es parte del sistema bajo evaluación — /chat/messages ya reintenta
    internamente contra su propia cadena de fallback; esto es la evaluación
    dándole varias chances a la MISMA pregunta para conseguir una respuesta
    real que evaluar, en vez de contar un 503 transitorio como fallback
    honesto o como ALUCINADO.
    """
    resp = await signed_post(client, path, payload)
    attempt = 1
    while resp.status_code == 503 and attempt < max_attempts:
        await asyncio.sleep(backoff_sec)
        resp = await signed_post(client, path, payload)
        attempt += 1
    return resp


@dataclass
class RetrievalResult:
    question_id: str
    recall_at_5: float
    precision_at_5: float
    reciprocal_rank: float
    retrieved_filenames: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    question_id: str
    answer: str
    judge_verdict: Optional[str]
    context_used: str


@dataclass
class FallbackResult:
    question_id: str
    answer: str
    fallback_triggered: bool
    # False si el LLM contestó y no se registró como fallback honesto;
    # None si la pregunta ni siquiera obtuvo una respuesta real (error de
    # infraestructura — 503/429 tras agotar reintentos). Distinguir esto es
    # crítico: un 503 no es "el modelo no admitió que no sabía", es "el
    # modelo nunca llegó a responder". Contarlo como fallo de honestidad
    # infla artificialmente el % de alucinación.
    answered: bool = True


async def index_course_material(
    client: httpx.AsyncClient,
    course_id: int,
    uploader_id: int,
    material_path: Path,
) -> str:
    """Sube el apunte de prueba vía el endpoint real y espera a que se indexe."""
    content = material_path.read_bytes()
    content_b64 = base64.b64encode(content).decode("ascii")

    payload = {
        "filename": material_path.name,
        "mime_type": "text/markdown",
        "content_b64": content_b64,
        "course_id": course_id,
        "uploader_id": uploader_id,
    }
    resp = await signed_post(client, "/api/v1/documents", payload)
    if resp.status_code not in (200, 202):
        raise RuntimeError(f"Upload falló: {resp.status_code} {resp.text}")
    document_id = resp.json()["id"]

    # Poll hasta que termine de indexar (o falle).
    for _ in range(60):
        headers = sign_request(b"")
        headers.pop("Content-Type", None)
        status_resp = await client.get(
            f"{API_BASE_URL}/api/v1/documents/{document_id}",
            headers=headers,
        )
        if status_resp.status_code == 200:
            doc = status_resp.json()
            if doc["status"] == "indexed":
                return document_id
            if doc["status"] == "error":
                raise RuntimeError(f"Indexación falló: {doc.get('error_message')}")
        await asyncio.sleep(1)

    raise TimeoutError("El documento no terminó de indexar a tiempo")


async def cleanup_course(course_id: int) -> None:
    """Borra los documents/chunks del curso de evaluación (CASCADE)."""
    factory = get_session_factory()
    async with factory() as db:
        await db.execute(delete(Document).where(Document.course_id == course_id))
        await db.commit()


async def eval_retrieval(dataset: list[dict], course_id: int) -> list[RetrievalResult]:
    """Nivel 1 — retrieval. Llama directo a retrieve_context(), igual que /chat."""
    factory = get_session_factory()
    embeddings = get_embedding_provider()
    results: list[RetrievalResult] = []

    async with factory() as db:
        for item in dataset:
            if item["category"] == "C":
                continue  # sin material real, no aplica retrieval ground-truth

            chunks = await retrieve_context(
                question=item["question"],
                course_id=course_id,
                db=db,
                embeddings=embeddings,
                top_k=CHAT_TOP_K,
                min_similarity=CHAT_MIN_SIMILARITY,
            )

            keywords = [k.lower() for k in item["ground_truth_keywords"]]

            def is_correct(chunk) -> bool:
                content_lower = chunk.content.lower()
                return all(k in content_lower for k in keywords)

            hit_ranks = [rank for rank, c in enumerate(chunks, start=1) if is_correct(c)]
            recall = 1.0 if hit_ranks else 0.0
            precision = (
                sum(1 for c in chunks if is_correct(c)) / len(chunks) if chunks else 0.0
            )
            rr = 1.0 / hit_ranks[0] if hit_ranks else 0.0

            results.append(
                RetrievalResult(
                    question_id=item["id"],
                    recall_at_5=recall,
                    precision_at_5=precision,
                    reciprocal_rank=rr,
                    retrieved_filenames=[c.document_filename for c in chunks],
                )
            )

    return results


async def get_retrieved_context_text(item: dict, course_id: int) -> str:
    """Reconstruye el contexto que /chat/messages usó (misma llamada, mismos params)."""
    factory = get_session_factory()
    embeddings = get_embedding_provider()
    async with factory() as db:
        chunks = await retrieve_context(
            question=item["question"],
            course_id=course_id,
            db=db,
            embeddings=embeddings,
            top_k=CHAT_TOP_K,
            min_similarity=CHAT_MIN_SIMILARITY,
        )
    return "\n\n".join(c.content for c in chunks)


def get_judge_provider() -> LLMProvider:
    """
    LLM-as-judge separado del proveedor primario bajo evaluación.

    Nota importante (hallazgo de esta corrida): el LLM_MODEL primario
    (gemini-2.5-flash) tiene una cuota gratuita de 20 requests/DIA (no por
    minuto), y el fallback configurado en producción (LLM_FALLBACK_MODEL=
    llama-3.3-70b-versatile) está deprecado en Groq (404 model_not_found) —
    ver evaluacion-rag.md, sección "Hallazgos de infraestructura". Para no
    quemar la cuota diaria del sistema bajo evaluación con las llamadas del
    judge (que son tooling de la evaluación, no el sistema medido), el judge
    usa un modelo de Groq confirmado disponible con la misma API key de
    fallback, en vez de reusar get_llm_provider().
    """
    settings = get_settings()
    return LLMProvider(
        api_key=settings.llm_fallback_api_key,
        base_url=settings.llm_fallback_base_url,
        model="openai/gpt-oss-120b",
        # Este modelo de Groq no acepta reasoning_effort="none" (el default
        # de LLM_REASONING_EFFORT) — exige low/medium/high.
        reasoning_effort="low",
    )


async def eval_generation(
    client: httpx.AsyncClient,
    dataset: list[dict],
    course_id: int,
    user_id: int,
) -> list[GenerationResult]:
    """Nivel 2 — faithfulness. Pregunta real al chat, juzga con un segundo LLM."""
    judge_llm = get_judge_provider()
    results: list[GenerationResult] = []

    for item in dataset:
        if item["category"] == "C":
            continue

        payload = {"question": item["question"], "course_id": course_id, "user_id": user_id}
        await asyncio.sleep(LLM_CALL_PACING_SEC)
        resp = await signed_post_with_retry(client, "/api/v1/chat/messages", payload)
        if resp.status_code != 200:
            results.append(
                GenerationResult(item["id"], f"<error {resp.status_code}>", None, "")
            )
            continue
        answer = resp.json()["answer"]

        context_text = await get_retrieved_context_text(item, course_id)

        verdict = None
        if context_text:
            judge_prompt = JUDGE_PROMPT_TEMPLATE.format(context=context_text, answer=answer)
            judge_result = await judge_llm.chat_completion(
                [{"role": "user", "content": judge_prompt}],
                temperature=0.0,
            )
            verdict_raw = judge_result.text.strip().upper()
            for candidate in ("FIEL", "PARCIAL", "ALUCINADO"):
                if candidate in verdict_raw:
                    verdict = candidate
                    break

        results.append(GenerationResult(item["id"], answer, verdict, context_text))

    return results


async def eval_fallback(
    client: httpx.AsyncClient, dataset: list[dict], course_id: int, user_id: int
) -> list[FallbackResult]:
    """Nivel 3 — fallback honesto. Preguntas fuera del material (categoría C)."""
    results: list[FallbackResult] = []
    for item in dataset:
        if item["category"] != "C":
            continue
        payload = {"question": item["question"], "course_id": course_id, "user_id": user_id}
        await asyncio.sleep(LLM_CALL_PACING_SEC)
        resp = await signed_post_with_retry(client, "/api/v1/chat/messages", payload)
        if resp.status_code != 200:
            results.append(
                FallbackResult(item["id"], f"<error {resp.status_code}>", False, answered=False)
            )
            continue
        answer = resp.json()["answer"]
        results.append(FallbackResult(item["id"], answer, llm_indicated_no_answer(answer)))
    return results


def summarize(
    retrieval: list[RetrievalResult],
    generation: list[GenerationResult],
    fallback: list[FallbackResult],
) -> dict[str, Any]:
    n_r = len(retrieval)
    recall_at_5 = sum(r.recall_at_5 for r in retrieval) / n_r if n_r else None
    precision_at_5 = sum(r.precision_at_5 for r in retrieval) / n_r if n_r else None
    mrr = sum(r.reciprocal_rank for r in retrieval) / n_r if n_r else None

    judged = [g for g in generation if g.judge_verdict is not None]
    n_g = len(judged)
    fiel_rate = (
        sum(1 for g in judged if g.judge_verdict == "FIEL") / n_g if n_g else None
    )
    n_g_infra_errors = sum(1 for g in generation if g.answer.startswith("<error"))

    n_f = len(fallback)
    answered_f = [f for f in fallback if f.answered]
    n_f_answered = len(answered_f)
    n_f_infra_errors = n_f - n_f_answered
    fallback_rate = (
        sum(1 for f in answered_f if f.fallback_triggered) / n_f_answered
        if n_f_answered
        else None
    )

    return {
        "nivel_1_retrieval": {
            "n_preguntas": n_r,
            "recall_at_5": recall_at_5,
            "precision_at_5": precision_at_5,
            "mrr": mrr,
            "objetivo_recall_at_5": 0.85,
            "objetivo_precision_at_5": 0.50,
            "objetivo_mrr": 0.70,
        },
        "nivel_2_generacion": {
            "n_preguntas": len(generation),
            "n_preguntas_evaluadas_por_judge": n_g,
            "n_preguntas_error_infraestructura": n_g_infra_errors,
            "fiel_rate": fiel_rate,
            "objetivo_fiel_rate": 0.95,
            "detalle_verdicts": {g.question_id: g.judge_verdict for g in generation},
        },
        "nivel_3_fallback_honesto": {
            "n_preguntas": n_f,
            "n_preguntas_respondidas": n_f_answered,
            "n_preguntas_error_infraestructura": n_f_infra_errors,
            "fallback_rate": fallback_rate,
            "objetivo_fallback_rate": 0.90,
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación RAG end-to-end de NexusAI")
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--course-id", type=int, default=990001)
    parser.add_argument("--user-id", type=int, default=999999)
    parser.add_argument("--keep-data", action="store_true", help="No borrar el material indexado al terminar")
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text())["items"]

    async with httpx.AsyncClient(timeout=120.0) as client:
        print(f"[1/4] Indexando {args.material.name} en curso {args.course_id}...")
        await index_course_material(client, args.course_id, args.user_id, args.material)
        print("      Indexado OK.")

        print("[2/4] Nivel 1 — retrieval (Recall@5, Precision@5, MRR)...")
        retrieval_results = await eval_retrieval(dataset, args.course_id)

        print("[3/4] Nivel 2 — generación / faithfulness (LLM-as-judge)...")
        generation_results = await eval_generation(client, dataset, args.course_id, args.user_id)

        print("[4/4] Nivel 3 — fallback honesto (preguntas fuera del material)...")
        fallback_results = await eval_fallback(client, dataset, args.course_id, args.user_id)

    summary = summarize(retrieval_results, generation_results, fallback_results)

    report = {
        "meta": {
            "course_id": args.course_id,
            "dataset": str(args.dataset),
            "material": str(args.material),
            "chat_top_k": CHAT_TOP_K,
            "chat_min_similarity": CHAT_MIN_SIMILARITY,
        },
        "summary": summary,
        "detalle": {
            "retrieval": [r.__dict__ for r in retrieval_results],
            "generacion": [g.__dict__ for g in generation_results],
            "fallback": [f.__dict__ for f in fallback_results],
        },
    }

    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReporte escrito en {args.out}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if not args.keep_data:
        await cleanup_course(args.course_id)
        print(f"Datos de evaluación del curso {args.course_id} eliminados.")

    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
