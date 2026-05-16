"""Pydantic schemas that define the contract between the Moodle PHP plugin and the NexusAI backend."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    session_id: Optional[UUID] = None


class MessageOut(BaseModel):
    id: UUID
    role: str
    content: str
    created_at: datetime
    # Token counts solo presentes en mensajes role='assistant'. NULL → None.
    token_count_prompt: Optional[int] = None
    token_count_completion: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class ChatResponse(BaseModel):
    session_id: UUID
    answer: str
    messages: List[MessageOut]
    # Tokens consumidos en ESTA respuesta (útiles para monitoreo en tiempo real).
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
