from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ResourceUsage(BaseModel):
    """Represents information about a language model's usage statistics."""

    model: str
    total_input_tokens: int
    total_output_tokens: int
    total_run_time: float


class Options(BaseModel):
    """Contains configuration options for user sessions and model usage."""

    user_access_token: str | None
    fingerprint: str | None
    llm_model: LLMModel | None = Field(alias="LLM_Model")


class NewMessage(BaseModel):
    """Represents a newly created message in a chat conversation."""

    role: str
    content: str
    created_at: datetime = Field(alias="createdAt")
    chat_id: str = Field(alias="chatId")
    message_id: str = Field(alias="messageId")
    files: list[str]
    id: int | None


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
    user_id: str


class Role(StrEnum):
    """Role for Messages."""

    ASSISTANT = "assistant"
    USER = "user"


class NewMessage(BaseModel):
    """Represents a message within a chat conversation."""

    id: str = Field(alias="messageId")
    content: str
    role: Role
    chat_id: str = Field(alias="chatId")
    resource_usage: ResourceUsage | None = None


class Message(BaseModel):
    """Represents a message within a chat conversation."""

    id: str = Field(alias="messageId")
    created_at: datetime = Field(alias="createdAt")
    content: str
    role: Role
    chat_id: str = Field(alias="chatId")
    resource_usage: ResourceUsage | None = None


class ChatMessage(BaseModel):
    """Combines a chat with its associated messages."""

    chat: Chat
    messages: list[Message]
