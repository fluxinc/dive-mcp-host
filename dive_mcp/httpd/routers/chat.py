from typing import TypeVar

from fastapi import APIRouter

from .models import Chat, ChatMessage, ResultResponse

chat = APIRouter(prefix="/chat", tags=["chat"])

T = TypeVar("T")


class DataResult[T](ResultResponse):
    data: T | None


@chat.get("/list")
async def list_chat() -> DataResult[list[Chat]]:
    raise NotImplementedError


@chat.get("/{chat_id}")
async def get_chat(chat_id: str) -> DataResult[ChatMessage]:
    raise NotImplementedError


@chat.delete("/{chat_id}")
async def delete_chat(chat_id: str) -> ResultResponse:
    raise NotImplementedError


@chat.post("/{chat_id}/abort")
async def abort_chat(chat_id: str) -> ResultResponse:
    raise NotImplementedError
