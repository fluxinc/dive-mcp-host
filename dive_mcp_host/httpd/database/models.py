from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ResourceUsage(BaseModel):
    """Represents information about a language model's usage statistics."""

    model: str
    total_input_tokens: int
    total_output_tokens: int
    total_run_time: float


class QueryInput(BaseModel):
    """User input for a query with text, images and documents."""

    text: str | None
    images: list[str] | None
    documents: list[str] | None


class Chat(BaseModel):
    """Represents a chat conversation with its basic properties."""

    id: str
    title: str
    created_at: datetime = Field(alias="createdAt")
    user_id: str | None


class Role(StrEnum):
    """Role for Messages."""

    ASSISTANT = "assistant"
    USER = "user"


class NewMessage(BaseModel):
    """Represents a message within a chat conversation."""

    content: str
    role: Role
    chat_id: str = Field(alias="chatId")
    message_id: str = Field(alias="messageId")
    resource_usage: ResourceUsage | None = None
    files: str = "[]"


class Message(BaseModel):
    """Represents a message within a chat conversation."""

    id: int
    create_at: datetime = Field(alias="createdAt")
    content: str
    role: Role
    chat_id: str = Field(alias="chatId")
    message_id: str = Field(alias="messageId")
    resource_usage: ResourceUsage | None = None
    files: str = "[]"


class ChatMessage(BaseModel):
    """Combines a chat with its associated messages."""

    chat: Chat
    messages: list[Message]
