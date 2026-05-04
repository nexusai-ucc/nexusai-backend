"""Text chunking utilities based on a sliding token window.

The strategy keeps a fixed-size window of tokens and advances by a stride that
leaves `overlap_tokens` between consecutive chunks. That overlap preserves local
context across chunk boundaries, which is important for RAG retrieval quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import tiktoken


@dataclass(frozen=True)
class Chunk:
    content: str
    chunk_index: int
    token_count: int


def chunk_text(text: str, max_tokens: int = 512, overlap_tokens: int = 64) -> List[Chunk]:
    """Split text into overlapping token chunks using cl100k_base tokenization."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be greater than or equal to 0")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("text must not be empty")

    encoding = tiktoken.get_encoding("cl100k_base")
    token_ids = encoding.encode(cleaned_text)

    if len(token_ids) <= max_tokens:
        return [
            Chunk(
                content=encoding.decode(token_ids).strip(),
                chunk_index=0,
                token_count=len(token_ids),
            )
        ]

    chunks: List[Chunk] = []
    step = max_tokens - overlap_tokens

    start = 0
    chunk_index = 0
    total_tokens = len(token_ids)

    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        chunk_tokens = token_ids[start:end]
        chunk_content = encoding.decode(chunk_tokens).strip()

        if chunk_content:
            chunks.append(
                Chunk(
                    content=chunk_content,
                    chunk_index=chunk_index,
                    token_count=len(chunk_tokens),
                )
            )

        chunk_index += 1
        if end >= total_tokens:
            break
        start += step

    return chunks