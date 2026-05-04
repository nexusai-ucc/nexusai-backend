"""Document indexing pipeline.

The pipeline loads a `Document`, extracts text from the uploaded PDF, splits it
into overlapping chunks, embeds each chunk individually, and persists the chunk
rows back into the same database transaction flow used by the API.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
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
    """Index a PDF document into chunk rows with embeddings.

    The flow is:
      1. Load the document row.
      2. Mark it as indexing and commit that state.
      3. Extract text from the uploaded PDF bytes.
      4. Split the text into overlapping chunks.
      5. Embed each chunk one by one.
      6. Persist all chunk rows.
      7. Mark the document as indexed and commit.

    If extraction or indexing fails, the document is marked as error and the
    error message is committed before the exception is re-raised.
    """
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if document is None:
        raise ValueError(f"Document {document_id} not found")

    document.status = "indexing"
    document.error_message = None
    await db.commit()

    try:
        text = extract_text(file_bytes)
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
        for chunk in chunks:
            embedding = await embeddings.embed(chunk.content)
            chunk_rows.append(
                ChunkModel(
                    document_id=document.id,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    token_count=chunk.token_count,
                    embedding=embedding,
                )
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