from typing import Generic, TypeVar

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .models import Chat, ChatMessage, ResultResponse

chat = APIRouter(prefix="/chat", tags=["chat"])

T = TypeVar("T")


class DataResult(ResultResponse, Generic[T]):
    """Generic result that extends ResultResponse with a data field."""

    data: T | None


@chat.get("/list")
async def list_chat() -> DataResult[list[Chat]]:
    """List all available chats.

    Returns:
        DataResult[list[Chat]]: List of available chats.
    """
    raise NotImplementedError


@chat.post("/")
async def create_chat(chat: Chat) -> StreamingResponse:
    """Create a new chat.

    Args:
        chat (Chat): The chat to create.
    """
    raise NotImplementedError


@chat.post("/edit")
async def edit_chat(chat: Chat) -> StreamingResponse:
    """Edit a chat.

    Args:
        chat (Chat): The chat to edit.
    """
    raise NotImplementedError


@chat.post("/retry")
async def retry_chat(chat: Chat) -> StreamingResponse:
    """Retry a chat.

    Args:
        chat (Chat): The chat to retry.
    """
    raise NotImplementedError


@chat.get("/{chat_id}")
async def get_chat(chat_id: str) -> DataResult[ChatMessage]:
    """Get a specific chat by ID with its messages.

    Args:
        chat_id (str): The ID of the chat to retrieve.

    Returns:
        DataResult[ChatMessage]: The chat and its messages.
    """
    raise NotImplementedError


@chat.delete("/{chat_id}")
async def delete_chat(chat_id: str) -> ResultResponse:
    """Delete a specific chat by ID.

    Args:
        chat_id (str): The ID of the chat to delete.

    Returns:
        ResultResponse: Result of the delete operation.
    """
    raise NotImplementedError


@chat.post("/{chat_id}/abort")
async def abort_chat(chat_id: str) -> ResultResponse:
    """Abort an ongoing chat operation.

    Args:
        chat_id (str): The ID of the chat to abort.

    Returns:
        ResultResponse: Result of the abort operation.
    """
    raise NotImplementedError
