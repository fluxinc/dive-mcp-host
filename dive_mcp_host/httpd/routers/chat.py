from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Annotated, TypeVar

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse

from dive_mcp_host.httpd.database.models import Chat, ChatMessage, QueryInput
from dive_mcp_host.httpd.dependencies import get_app, get_dive_user
from dive_mcp_host.httpd.routers.utils import ChatProcessor, event_stream
from dive_mcp_host.httpd.server import DiveHostAPI

from .models import (
    ResultResponse,
    UserInputError,
)

if TYPE_CHECKING:
    from dive_mcp_host.httpd.middlewares.general import DiveUser
    from dive_mcp_host.httpd.store import Store

chat = APIRouter(prefix="/chat", tags=["chat"])

T = TypeVar("T")


class DataResult[T](ResultResponse):
    """Generic result that extends ResultResponse with a data field."""

    data: T | None


@chat.get("/list")
async def list_chat(
    app: DiveHostAPI = Depends(get_app),
    dive_user: "DiveUser" = Depends(get_dive_user),
) -> DataResult[list[Chat]]:
    """List all available chats.

    Args:
        app (DiveHostAPI): The DiveHostAPI instance.
        dive_user (DiveUser): The DiveUser instance.

    Returns:
        DataResult[list[Chat]]: List of available chats.
    """
    async with app.db_sessionmaker() as session:
        chats = await app.msg_store(session).get_all_chats(dive_user["user_id"])
    return DataResult(success=True, message=None, data=chats)


@chat.post("")
async def create_chat(
    request: Request,
    chat_id: Annotated[str | None, Form(alias="chatId")] = None,
    message: Annotated[str | None, Form()] = None,
    files: Annotated[list[UploadFile] | None, File()] = None,
    filepaths: Annotated[list[str] | None, Form()] = None,
) -> StreamingResponse:
    """Create a new chat.

    Args:
        request (Request): The request object.
        chat_id (str | None): The ID of the chat to create.
        message (str | None): The message to send.
        files (list[UploadFile] | None): The files to upload.
        filepaths (list[str] | None): The file paths to upload.
    """
    store: Store = request.app.state.store

    if files is None:
        files = []

    if filepaths is None:
        filepaths = []

    images, documents = await store.upload_files(files, filepaths)

    async def process() -> AsyncGenerator[str, None]:
        query_input = QueryInput(text=message, images=images, documents=documents)
        processor = ChatProcessor(request.app.state, request.state)
        async for chunk in processor.handle_chat(chat_id, query_input, None):
            yield chunk

        yield "[Done]"

    return event_stream(process())


@chat.post("/edit")
async def edit_chat(
    request: Request,
    chat_id: Annotated[str | None, Form(alias="chatId")] = None,
    message_id: Annotated[str | None, Form(alias="messageId")] = None,
    content: Annotated[str | None, Form()] = None,
    files: Annotated[list[UploadFile] | None, File()] = None,
    filepaths: Annotated[list[str] | None, Form()] = None,
) -> StreamingResponse:
    """Edit a chat.

    Args:
        request (Request): The request object.
        chat_id (str | None): The ID of the chat to edit.
        message_id (str | None): The ID of the message to edit.
        content (str | None): The content to send.
        files (list[UploadFile] | None): The files to upload.
        filepaths (list[str] | None): The file paths to upload.
    """
    store: Store = request.app.state.store
    db: Database = request.app.state.db
    db_opts = request.state.get_kwargs("db_opts")

    if chat_id is None or message_id is None:
        raise UserInputError("Chat ID and Message ID are required")

    if files is None:
        files = []

    if filepaths is None:
        filepaths = []

    images, documents = await store.upload_files(files, filepaths)

    async def process() -> AsyncGenerator[str, None]:
        query_input = QueryInput(text=content, images=images, documents=documents)
        await db.update_message_content(message_id, query_input, **db_opts)
        next_ai_message = await db.get_next_ai_message(chat_id, message_id, **db_opts)
        # TODO: send query to LLM

        yield "[Done]"

    return event_stream(process())


@chat.post("/retry")
async def retry_chat(
    request: Request,
    chat_id: Annotated[str | None, Form(alias="chatId")] = None,
    message_id: Annotated[str | None, Form(alias="messageId")] = None,
) -> StreamingResponse:
    """Retry a chat.

    Args:
        request (Request): The request object.
        chat_id (str | None): The ID of the chat to retry.
        message_id (str | None): The ID of the message to retry.
    """
    store: Store = request.app.state.store
    db: Database = request.app.state.db
    db_opts = request.state.get_kwargs("db_opts")
    if chat_id is None or message_id is None:
        raise UserInputError("Chat ID and Message ID are required")

    async def content() -> AsyncGenerator[str, None]:
        # TODO: send query to LLM
        yield "[Done]"

    return event_stream(content())


@chat.get("/{chat_id}")
async def get_chat(
    chat_id: str,
    app: DiveHostAPI = Depends(get_app),
    dive_user: "DiveUser" = Depends(get_dive_user),
) -> DataResult[ChatMessage]:
    """Get a specific chat by ID with its messages.

    Args:
        chat_id (str): The ID of the chat to retrieve.
        app (DiveHostAPI): The DiveHostAPI instance.
        dive_user (DiveUser): The DiveUser instance.

    Returns:
        DataResult[ChatMessage]: The chat and its messages.
    """
    async with app.db_sessionmaker() as session:
        chat = await app.msg_store(session).get_chat_with_messages(
            chat_id=chat_id,
            user_id=dive_user["user_id"],
        )
    return DataResult(success=True, message=None, data=chat)


@chat.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    app: DiveHostAPI = Depends(get_app),
    dive_user: "DiveUser" = Depends(get_dive_user),
) -> ResultResponse:
    """Delete a specific chat by ID.

    Args:
        chat_id (str): The ID of the chat to delete.
        app (DiveHostAPI): The DiveHostAPI instance.
        dive_user (DiveUser): The DiveUser instance.

    Returns:
        ResultResponse: Result of the delete operation.
    """
    async with app.db_sessionmaker() as session:
        await app.msg_store(session).delete_chat(
            chat_id=chat_id,
            user_id=dive_user["user_id"],
        )
        await session.commit()
    return ResultResponse(success=True, message=None)


@chat.post("/{chat_id}/abort")
async def abort_chat(request: Request, chat_id: str) -> ResultResponse:
    """Abort an ongoing chat operation.

    Args:
        request (Request): The request object.
        chat_id (str): The ID of the chat to abort.

    Returns:
        ResultResponse: Result of the abort operation.
    """
    return ResultResponse(success=True, message="Chat abort signal sent successfully")
