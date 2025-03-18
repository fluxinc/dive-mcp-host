from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Annotated, TypeVar

from fastapi import APIRouter, BackgroundTasks, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse

from dive_mcp_host.httpd.database.models import Chat, ChatMessage, QueryInput
from dive_mcp_host.httpd.routers.models import (
    ResultResponse,
    UserInputError,
)
from dive_mcp_host.httpd.routers.utils import ChatProcessor, EventStreamContextManager

if TYPE_CHECKING:
    from dive_mcp_host.httpd.database.msg_store.abstract import AbstractMessageStore
    from dive_mcp_host.httpd.store import Store

chat = APIRouter(prefix="/chat", tags=["chat"])

T = TypeVar("T")


class DataResult[T](ResultResponse):
    """Generic result that extends ResultResponse with a data field."""

    data: T | None


@chat.get("/list")
async def list_chat(request: Request) -> DataResult[list[Chat]]:
    """List all available chats.

    Returns:
        DataResult[list[Chat]]: List of available chats.
    """
    db: AbstractMessageStore = request.app.state.db
    db_opts = request.state.get_kwargs("db_opts")
    chats = await db.get_all_chats(**db_opts)
    return DataResult(success=True, message=None, data=chats)


@chat.post("")
async def create_chat(  # noqa: PLR0913
    request: Request,
    background_tasks: BackgroundTasks,
    chat_id: Annotated[str | None, Form(alias="chatId")] = None,
    message: Annotated[str | None, Form()] = None,
    files: Annotated[list[UploadFile] | None, File()] = None,
    filepaths: Annotated[list[str] | None, Form()] = None,
) -> StreamingResponse:
    """Create a new chat.

    Args:
        request (Request): The request object.
        background_tasks (BackgroundTasks): The background tasks to run.
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

    stream = EventStreamContextManager()
    response = stream.get_response()

    async def process() -> None:
        async with stream:
            query_input = QueryInput(text=message, images=images, documents=documents)
            processor = ChatProcessor(request.app.state, request.state, stream)
            await processor.handle_chat(chat_id, query_input, None)

    stream.add_task(process)
    return response


@chat.post("/edit")
async def edit_chat(  # noqa: PLR0913
    request: Request,
    background_tasks: BackgroundTasks,
    chat_id: Annotated[str | None, Form(alias="chatId")] = None,
    message_id: Annotated[str | None, Form(alias="messageId")] = None,
    content: Annotated[str | None, Form()] = None,
    files: Annotated[list[UploadFile] | None, File()] = None,
    filepaths: Annotated[list[str] | None, Form()] = None,
) -> StreamingResponse:
    """Edit a chat.

    Args:
        request (Request): The request object.
        background_tasks (BackgroundTasks): The background tasks to run.
        chat_id (str | None): The ID of the chat to edit.
        message_id (str | None): The ID of the message to edit.
        content (str | None): The content to send.
        files (list[UploadFile] | None): The files to upload.
        filepaths (list[str] | None): The file paths to upload.
    """
    store: Store = request.app.state.store
    db: AbstractMessageStore = request.app.state.db
    db_opts = request.state.get_kwargs("db_opts")

    if chat_id is None or message_id is None:
        raise UserInputError("Chat ID and Message ID are required")

    if files is None:
        files = []

    if filepaths is None:
        filepaths = []

    images, documents = await store.upload_files(files, filepaths)

    stream = EventStreamContextManager()
    response = stream.get_response()

    async def process() -> None:
        async with stream:
            query_input = QueryInput(text=content, images=images, documents=documents)
            await db.update_message_content(message_id, query_input, **db_opts)
            next_ai_message = await db.get_next_ai_message(
                chat_id, message_id, **db_opts
            )
            # TODO: send query to LLM

    background_tasks.add_task(process)
    return response


@chat.post("/retry")
async def retry_chat(
    request: Request,
    background_tasks: BackgroundTasks,
    chat_id: Annotated[str | None, Form(alias="chatId")] = None,
    message_id: Annotated[str | None, Form(alias="messageId")] = None,
) -> StreamingResponse:
    """Retry a chat.

    Args:
        request (Request): The request object.
        background_tasks (BackgroundTasks): The background tasks to run.
        chat_id (str | None): The ID of the chat to retry.
        message_id (str | None): The ID of the message to retry.
    """
    store: Store = request.app.state.store
    db: AbstractMessageStore = request.app.state.db
    db_opts = request.state.get_kwargs("db_opts")
    if chat_id is None or message_id is None:
        raise UserInputError("Chat ID and Message ID are required")

    stream = EventStreamContextManager()
    response = stream.get_response()

    async def process() -> None:
        async with stream:
            # TODO: send query to LLM
            pass

    background_tasks.add_task(process)
    return response


@chat.get("/{chat_id}")
async def get_chat(request: Request, chat_id: str) -> DataResult[ChatMessage]:
    """Get a specific chat by ID with its messages.

    Args:
        request (Request): The request object.
        chat_id (str): The ID of the chat to retrieve.

    Returns:
        DataResult[ChatMessage]: The chat and its messages.
    """
    db: AbstractMessageStore = request.app.state.db
    db_opts = request.state.get_kwargs("db_opts")
    chat = await db.get_chat_with_messages(chat_id, **db_opts)
    return DataResult(success=True, message=None, data=chat)


@chat.delete("/{chat_id}")
async def delete_chat(request: Request, chat_id: str) -> ResultResponse:
    """Delete a specific chat by ID.

    Args:
        request (Request): The request object.
        chat_id (str): The ID of the chat to delete.

    Returns:
        ResultResponse: Result of the delete operation.
    """
    db: AbstractMessageStore = request.app.state.db
    db_opts = request.state.get_kwargs("db_opts")
    await db.delete_chat(chat_id, **db_opts)
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
