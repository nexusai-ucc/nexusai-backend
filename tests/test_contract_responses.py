"""
Contrato de respuestas con el plugin (TPC-07, #579).

`NexusAI/contracts/responses/*.json` son respuestas de ejemplo del backend.
Este test las valida contra los modelos Pydantic REALES de cada endpoint y
exige que no tengan campos que el modelo no declara. Del otro lado,
`plugin/local/nexusai/tests/contract_test.php` las usa como respuesta del
doble de prueba y verifica que el web service que las consume las procesa.
Así el doble del plugin no puede desviarse de lo que el backend manda.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from app.chat.schemas import ChatResponse
from app.documents.router import DocumentOut
from app.quiz.router import QuizResponse
from app.search.router import SearchResponse

RESPONSES = Path(__file__).resolve().parents[3] / "contracts" / "responses"

pytestmark = pytest.mark.skipif(
    not RESPONSES.exists(), reason="contracts/ no está en este checkout"
)

MODELS: dict[str, type[BaseModel]] = {
    "chat_messages.json": ChatResponse,
    "search.json": SearchResponse,
    "quiz_generate.json": QuizResponse,
    "document.json": DocumentOut,
}


def _payload(name: str) -> dict:
    data = json.loads((RESPONSES / name).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _unknown_keys(model: type[BaseModel], data: dict, path: str = "") -> list[str]:
    """Claves del JSON que el modelo (o sus submodelos) no declara."""
    unknown = []
    for key, value in data.items():
        field = model.model_fields.get(key)
        if field is None:
            unknown.append(f"{path}{key}")
            continue
        sub = _submodel(field.annotation)
        if sub and isinstance(value, dict):
            unknown += _unknown_keys(sub, value, f"{path}{key}.")
        elif sub and isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    unknown += _unknown_keys(sub, item, f"{path}{key}[{i}].")
    return unknown


def _submodel(annotation) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in getattr(annotation, "__args__", ()) or ():
        found = _submodel(arg)
        if found:
            return found
    return None


def test_every_contract_file_has_a_model():
    """Criterio: no hay respuestas de contrato huérfanas, sin modelo que las valide."""
    # Act.
    files = sorted(p.name for p in RESPONSES.glob("*.json"))

    # Assert.
    assert files == sorted(MODELS)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_contract_response_matches_the_real_model(name):
    """Criterio: la respuesta de ejemplo valida contra el modelo real y no inventa campos."""
    # Arrange.
    model = MODELS[name]
    data = _payload(name)

    # Act.
    parsed = model.model_validate(data)
    unknown = _unknown_keys(model, data)

    # Assert.
    assert parsed is not None
    assert (
        unknown == []
    ), f"{name} trae campos que {model.__name__} no declara: {unknown}"
