"""
Lista y carga precios de modelos en `model_prices` (COST-01, issue #519).

Los precios van en USD por millón de tokens (o por minuto de audio). Nunca
se pisan: cada cambio de precio es una fila nueva con su fecha de vigencia,
así el costo ya registrado en llm_usage no cambia. La API cachea cada precio
10 minutos, así que un precio nuevo se nota en ese plazo.

Uso, dentro del contenedor de la API:

    python scripts/model_prices.py list
    python scripts/model_prices.py set --provider openai --model gpt-4o-mini \\
        --input 0.15 --output 0.60 --cached-input 0.075
    python scripts/model_prices.py set --provider groq --model whisper-large-v3 \\
        --audio-minute 0.00185 --valid-from 2026-10-01

`provider` tiene que coincidir con el que registra la API: google, openai,
groq, local o el host de la base_url (ver usage_ledger.provider_from_base_url).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.db.models import ModelPrice  # noqa: E402
from app.db.session import get_session_factory  # noqa: E402


async def list_prices() -> None:
    async with get_session_factory()() as session:
        rows = (
            await session.execute(
                select(ModelPrice).order_by(
                    ModelPrice.provider, ModelPrice.model, ModelPrice.valid_from
                )
            )
        ).scalars()
        print(
            "provider | model | entrada | salida | entrada cacheada | embeddings | minuto de audio | vigente desde"
        )
        for p in rows:
            print(
                f"{p.provider} | {p.model} | {p.input_per_mtok} | {p.output_per_mtok} | "
                f"{p.cached_input_per_mtok} | {p.embedding_per_mtok} | "
                f"{p.audio_per_minute} | {p.valid_from.isoformat()}"
            )


async def set_price(args: argparse.Namespace) -> None:
    valid_from = (
        datetime.fromisoformat(args.valid_from).replace(tzinfo=timezone.utc)
        if args.valid_from
        else datetime.now(timezone.utc)
    )
    price = ModelPrice(
        provider=args.provider,
        model=args.model,
        input_per_mtok=Decimal(args.input),
        output_per_mtok=Decimal(args.output),
        cached_input_per_mtok=(
            Decimal(args.cached_input) if args.cached_input is not None else None
        ),
        embedding_per_mtok=Decimal(args.embedding),
        audio_per_minute=Decimal(args.audio_minute),
        valid_from=valid_from,
    )
    async with get_session_factory()() as session:
        session.add(price)
        await session.commit()
    print(
        f"Precio cargado para {args.provider}/{args.model}, vigente desde {valid_from.isoformat()}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precios de modelos para el registro de consumo"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Lista todos los precios cargados")
    set_cmd = sub.add_parser("set", help="Carga un precio nuevo")
    set_cmd.add_argument("--provider", required=True)
    set_cmd.add_argument("--model", required=True)
    set_cmd.add_argument(
        "--input", default="0", help="USD por millón de tokens de entrada"
    )
    set_cmd.add_argument(
        "--output", default="0", help="USD por millón de tokens de salida"
    )
    set_cmd.add_argument(
        "--cached-input",
        default=None,
        help="USD por millón de tokens de entrada cacheados (sin valor: se cobran como entrada normal)",
    )
    set_cmd.add_argument(
        "--embedding", default="0", help="USD por millón de tokens de embeddings"
    )
    set_cmd.add_argument("--audio-minute", default="0", help="USD por minuto de audio")
    set_cmd.add_argument(
        "--valid-from",
        default=None,
        help="Fecha ISO desde la que rige (por defecto, ahora, en UTC)",
    )
    args = parser.parse_args()

    if args.command == "list":
        asyncio.run(list_prices())
    else:
        asyncio.run(set_price(args))


if __name__ == "__main__":
    main()
