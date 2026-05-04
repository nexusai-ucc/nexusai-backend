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

    model_config = ConfigDict(from_attributes=True)


class ChatResponse(BaseModel):
    session_id: UUID
    answer: str
    messages: List[MessageOut]