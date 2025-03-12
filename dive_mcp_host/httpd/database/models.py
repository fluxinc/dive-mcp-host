from datetime import datetime

from pydantic import BaseModel, Field


class ResourceUsage(BaseModel):
    """Represents information about a language model's usage statistics."""

    model: str
    total_input_tokens: int
    total_output_tokens: int
    total_run_time: int


# NOTE: Currently not used
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


class Message(BaseModel):
    """Represents a message within a chat conversation."""

    id: int
    create_at: datetime = Field(alias="createdAt")
    content: str
    role: str
    chat_id: str = Field(alias="chatId")
    message_id: str = Field(alias="messageId")
    resource_usage: ResourceUsage | None = None


class ChatMessage(BaseModel):
    """Combines a chat with its associated messages."""

    chat: Chat
    messages: list[Message]
