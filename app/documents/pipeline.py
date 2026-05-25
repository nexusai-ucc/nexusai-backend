"""Document indexing pipeline.

The pipeline loads a `Document`, extracts text from the uploaded file, splits it
into overlapping chunks, embeds each chunk individually, and persists the chunk
rows back into the same database transaction flow used by the API.

Resiliencia (BACK-11):
  - Si un chunk individual falla al embeddear (después de los retries del
    EmbeddingProvider), se loguea y se continúa con el siguiente.
  - El chunk se persiste igualmente con embedding=NULL — el retriever
    lo ignora (filtra `Chunk.embedding.is_not(None)`), pero el contenido
    textual queda disponible para futuros re-intentos.
  - Solo si TODOS los chunks fallan se marca el documento como 'error'.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk as ChunkModel
from app.db.models import Document
from app.documents.chunker import chunk_text
from app.documents.extractor import extract_text
from app.providers.embeddings import EmbeddingProvider


async def index_document(
    document_id: UUID,
    file_bytes: bytes,
    db: AsyncSession,
    embeddings: EmbeddingProvider,
) -> Document:
    """Indexa un documento: extrae texto → chunking → embeddings → persist.

    Flujo:
      1. Carga el row Document de la DB.
      2. Marca como 'indexing' y commitea.
      3. Extrae texto del archivo (PDF/DOCX/TXT según mime_type del Document).
      4. Divide en chunks con overlap.
      5. Embeddea cada chunk. Si uno falla → loguea y guarda con embedding=NULL.
      6. Si TODOS los chunks fallaron → marca como 'error'.
      7. Si al menos uno tuvo éxito → marca como 'indexed'.
    """
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if document is None:
        raise ValueError(f"Document {document_id} not found")

    # CONT-04: si el documento ya tiene chunks (re-encola de un intento previo
    # parcialmente exitoso), salteamos la indexación completa.
    chunk_count_result = await db.execute(
        select(func.count(ChunkModel.id)).where(ChunkModel.document_id == document_id)
    )
    if chunk_count_result.scalar_one() > 0:
        document.status = "indexed"
        document.error_message = None
        await db.commit()
        await db.refresh(document)
        return document

    document.status = "indexing"
    document.error_message = None
    await db.commit()

    try:
        text = extract_text(file_bytes, document.mime_type)
    except Exception as exc:
        await db.rollback()
        result = await db.execute(select(Document).where(Document.id == document_id))
        document = result.scalar_one_or_none() or document
        document.status = "error"
        document.error_message = str(exc)
        await db.commit()
        raise

    try:
        chunks = chunk_text(text)
        chunk_rows = []
        failed_count = 0

        for chunk in chunks:
            try:
                embedding = await embeddings.embed(chunk.content)
            except Exception as exc:
                failed_count += 1
                print(
                    f"[NexusAI] Embedding failed for chunk {chunk.chunk_index} "
                    f"of document {document_id}: {type(exc).__name__}: {exc}. "
                    "Storing chunk without embedding.",
                    flush=True,
                )
                embedding = None

            chunk_rows.append(
                ChunkModel(
                    document_id=document.id,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    token_count=chunk.token_count,
                    embedding=embedding,
                )
            )

        if failed_count == len(chunks):
            raise RuntimeError(
                f"Todos los {len(chunks)} chunks fallaron al embeddear. "
                "Verificá la API key y cuota del proveedor de embeddings."
            )

        if failed_count > 0:
            print(
                f"[NexusAI] Document {document_id}: {failed_count}/{len(chunks)} chunks "
                "sin embedding. El documento se indexó parcialmente.",
                flush=True,
            )

        db.add_all(chunk_rows)
        document.status = "indexed"
        document.error_message = None
        await db.commit()

    except Exception as exc:
        await db.rollback()
        result = await db.execute(select(Document).where(Document.id == document_id))
        document = result.scalar_one_or_none() or document
        document.status = "error"
        document.error_message = str(exc)
        await db.commit()
        raise

    await db.refresh(document)
    return document
