from typing import TYPE_CHECKING, Annotated, TypeVar, List, Dict, Any
import logging

from fastapi import APIRouter, Body, Depends, File, Form, Request, UploadFile, Query
from fastapi.responses import StreamingResponse

from dive_mcp_host.httpd.database.models import (
    Chat,
    ChatMessage,
    QueryInput,
)
from dive_mcp_host.httpd.dependencies import get_app, get_dive_user
from dive_mcp_host.httpd.routers.models import (
    ResultResponse,
    SortBy,
    UserInputError,
)
from dive_mcp_host.httpd.routers.utils import ChatProcessor, EventStreamContextManager
from dive_mcp_host.httpd.server import DiveHostAPI
from dive_mcp_host.mcpServer import MCPServerTracker

if TYPE_CHECKING:
    from dive_mcp_host.httpd.middlewares.general import DiveUser

chat = APIRouter(tags=["chat"])

T = TypeVar("T")

logger = logging.getLogger(__name__)
class DataResult[T](ResultResponse):
    """Generic result that extends ResultResponse with a data field."""

    data: T | None


class SourcesResponse(ResultResponse):
    """Response model for chat sources."""
    sources: List[Dict[str, str]]
    
    
class ToolCallsResponse(ResultResponse):
    """Response model for tool calls."""
    tool_calls: List[Dict[str, Any]]


class ToolResultsResponse(ResultResponse):
    """Response model for tool results."""
    tool_results: List[Dict[str, Any]]


@chat.get("/list")
async def list_chat(
    app: DiveHostAPI = Depends(get_app),
    dive_user: "DiveUser" = Depends(get_dive_user),
    sort_by: SortBy = SortBy.CHAT,
    session_id: Annotated[str | None, Query(alias="sessionId")] = None,
) -> DataResult[list[Chat]]:
    """List all available chats for a session.

    Args:
        app (DiveHostAPI): The DiveHostAPI instance.
        dive_user (DiveUser): The DiveUser instance.
        sort_by (SortBy):
            - 'chat': Sort by chat creation time.
            - 'msg': Sort by message creation time.
            default: 'chat'
        session_id (str | None): Session ID to filter chats.

    Returns:
        DataResult[list[Chat]]: List of available chats.
    """
    async with app.db_sessionmaker() as session:
        chats = await app.msg_store(session).get_all_chats(
            dive_user["user_id"],
            session_id=session_id,
            sort_by=sort_by,
        )
    return DataResult(success=True, message=None, data=chats)


@chat.post("")
async def create_chat(  # noqa: PLR0913
    request: Request,
    session_id: Annotated[str, Form(alias="sessionId")],
    app: DiveHostAPI = Depends(get_app),
    chat_id: Annotated[str | None, Form(alias="chatId")] = None,
    message: Annotated[str | None, Form()] = None,
    files: Annotated[list[UploadFile] | None, File()] = None,
    filepaths: Annotated[list[str] | None, Form()] = None,
) -> StreamingResponse:
    """Create a new chat.

    Args:
        request (Request): The request object.
        session_id (str): The session ID for the chat.
        app (DiveHostAPI): The DiveHostAPI instance.
        chat_id (str | None): The ID of the chat to create.
        message (str | None): The message to send.
        files (list[UploadFile] | None): The files to upload.
        filepaths (list[str] | None): The file paths to upload.
    """


    if files is None:
        files = []

    if filepaths is None:
        filepaths = []

    # If chat_id is provided, verify session_id matches
    if chat_id:
        dive_user = request.state.dive_user
        async with app.db_sessionmaker() as session:
            # Check if chat exists
            chat = await app.msg_store(session).get_chat_with_messages(
                chat_id=chat_id,
                user_id=dive_user["user_id"],
            )
            
            # If chat exists, verify session_id
            if chat and chat.chat.session_id and session_id != chat.chat.session_id:
                raise UserInputError("Session ID does not match the chat's session ID")

    images, documents = await app.store.upload_files(files, filepaths)

    stream = EventStreamContextManager()
    response = stream.get_response()
    query_input = QueryInput(text=message, images=images, documents=documents)

    async def process() -> None:
        async with stream:
            processor = ChatProcessor(app, request.state, stream)
            await processor.handle_chat(chat_id, query_input, None, session_id=session_id)

    stream.add_task(process)
    return response


@chat.post("/edit")
async def edit_chat(  # noqa: PLR0913
    request: Request,
    app: DiveHostAPI = Depends(get_app),
    chat_id: Annotated[str | None, Form(alias="chatId")] = None,
    message_id: Annotated[str | None, Form(alias="messageId")] = None,
    content: Annotated[str | None, Form()] = None,
    files: Annotated[list[UploadFile] | None, File()] = None,
    filepaths: Annotated[list[str] | None, Form()] = None,
    session_id: Annotated[str | None, Form(alias="sessionId")] = None,
) -> StreamingResponse:
    """Edit a message in a chat and query again.

    Args:
        request (Request): The request object.
        app (DiveHostAPI): The DiveHostAPI instance.
        chat_id (str | None): The ID of the chat to edit.
        message_id (str | None): The ID of the message to edit.
        content (str | None): The content to send.
        files (list[UploadFile] | None): The files to upload.
        filepaths (list[str] | None): The file paths to upload.
        session_id (str | None): Session ID for the chat.
    """
    if chat_id is None or message_id is None or session_id is None:
        raise UserInputError("Chat ID, Message ID and Session ID are required")

    if files is None:
        files = []

    if filepaths is None:
        filepaths = []

    # Verify session ID if provided
    dive_user = request.state.dive_user
    async with app.db_sessionmaker() as session:
        chat = await app.msg_store(session).get_chat_with_messages(
            chat_id=chat_id,
            user_id=dive_user["user_id"],
        )
        
        # If chat doesn't exist or user doesn't own it, raise error
        if chat is None:
            raise UserInputError("Chat not found")
            
        # If session_id doesn't match the chat's session_id, raise error
        if chat.chat.session_id and session_id != chat.chat.session_id:
            raise UserInputError("Session ID does not match the chat's session ID")

    images, documents = await app.store.upload_files(files, filepaths)

    stream = EventStreamContextManager()
    response = stream.get_response()
    query_input = QueryInput(text=content, images=images, documents=documents)

    async def process() -> None:
        async with stream:
            processor = ChatProcessor(app, request.state, stream)
            await processor.handle_chat(chat_id, query_input, message_id, session_id=session_id)

    stream.add_task(process)
    return response


@chat.post("/retry")
async def retry_chat(
    request: Request,
    app: DiveHostAPI = Depends(get_app),
    chat_id: Annotated[str | None, Body(alias="chatId", embed=True)] = None,
    message_id: Annotated[str | None, Body(alias="messageId", embed=True)] = None,
    session_id: Annotated[str | None, Body(alias="sessionId", embed=True)] = None,
) -> StreamingResponse:
    """Retry a chat.

    Args:
        request (Request): The request object.
        app (DiveHostAPI): The DiveHostAPI instance.
        chat_id (str | None): The ID of the chat to retry.
        message_id (str | None): The ID of the message to retry.
        session_id (str | None): Session ID for the chat.
    """
    if chat_id is None or message_id is None or session_id is None:
        raise UserInputError("Chat ID, Message ID and Session ID are required")

    # Verify session ID if provided
    dive_user = request.state.dive_user
    async with app.db_sessionmaker() as session:
        chat = await app.msg_store(session).get_chat_with_messages(
            chat_id=chat_id,
            user_id=dive_user["user_id"],
        )
        
        # If chat doesn't exist or user doesn't own it, raise error
        if chat is None:
            raise UserInputError("Chat not found")
            
        # If session_id doesn't match the chat's session_id, raise error
        if chat.chat.session_id and session_id != chat.chat.session_id:
            raise UserInputError("Session ID does not match the chat's session ID")

    stream = EventStreamContextManager()
    response = stream.get_response()

    async def process() -> None:
        async with stream:
            processor = ChatProcessor(app, request.state, stream)
            await processor.handle_chat(chat_id, None, message_id, session_id=session_id)

    stream.add_task(process)
    return response


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
    session_id: Annotated[str, Body(alias="sessionId", embed=True)],
    app: DiveHostAPI = Depends(get_app),
    dive_user: "DiveUser" = Depends(get_dive_user),
) -> ResultResponse:
    """Delete a specific chat by ID.

    Args:
        chat_id (str): The ID of the chat to delete.
        session_id (str): Session ID for the chat.
        app (DiveHostAPI): The DiveHostAPI instance.
        dive_user (DiveUser): The DiveUser instance.

    Returns:
        ResultResponse: Result of the delete operation.
    """
    
    async with app.db_sessionmaker() as session:
        # Get the chat first to verify session_id
        chat = await app.msg_store(session).get_chat_with_messages(
            chat_id=chat_id,
            user_id=dive_user["user_id"],
        )
        
        if chat is None:
            raise UserInputError("Chat not found or you don't have permission to delete it")
        
        # Check if session_id matches the chat's session_id
        if chat.chat.session_id != session_id:
            raise UserInputError("Session ID does not match the chat's session ID")
            
        # Delete the chat
        await app.msg_store(session).delete_chat(
            chat_id=chat_id,
            user_id=dive_user["user_id"],
            session_id=session_id,
        )
        
        # Explicitly commit the transaction - this should be handled by the context manager
        # but adding it to be sure
        await session.commit()
        
        return ResultResponse(success=True)


@chat.post("/{chat_id}/abort")
async def abort_chat(
    chat_id: str,
    app: DiveHostAPI = Depends(get_app),
) -> ResultResponse:
    """Abort an ongoing chat operation.

    Args:
        chat_id (str): The ID of the chat to abort.
        app (DiveHostAPI): The DiveHostAPI instance.

    Returns:
        ResultResponse: Result of the abort operation.
    """
    abort_controller = app.abort_controller
    ok = await abort_controller.abort(chat_id)
    if not ok:
        raise UserInputError("Chat not found")

    return ResultResponse(success=True, message="Chat abort signal sent successfully")


@chat.get("/{chat_id}/sources")
async def chat_sources(
    chat_id: str,
    app: DiveHostAPI = Depends(get_app),
    dive_user: Annotated[dict, Depends(get_dive_user)] = None,
) -> SourcesResponse:
    """Get the sources used in a chat session.
    
    Args:
        chat_id: The ID of the chat session.
        
    Returns:
        SourcesResponse: The sources used in the chat session.
    """
    # Get the sources for the chat
    tracker = MCPServerTracker.getInstance()
    sources = tracker.get_last_sources(chat_id) or []
    logger.info(f"Sources: {sources}")
    return SourcesResponse(
        success=True,
        message=None,
        sources=sources,
    )


@chat.get("/{chat_id}/tool_calls")
async def chat_tool_calls(
    chat_id: str,
    app: DiveHostAPI = Depends(get_app),
    dive_user: Annotated[dict, Depends(get_dive_user)] = None,
) -> ToolCallsResponse:
    """Get the tool calls used in a chat session.
    
    Args:
        chat_id: The ID of the chat session.
        
    Returns:
        ToolCallsResponse: The tool calls used in the chat session.
    """
    # Get the tool calls for the chat
    tracker = MCPServerTracker.getInstance()
    tool_calls = tracker.get_last_tool_calls(chat_id) or []
    
    return ToolCallsResponse(
        success=True,
        message=None,
        tool_calls=tool_calls,
    )


@chat.get("/{chat_id}/tool_results")
async def chat_tool_results(
    chat_id: str,
    app: DiveHostAPI = Depends(get_app),
    dive_user: Annotated[dict, Depends(get_dive_user)] = None,
) -> ToolResultsResponse:
    """Get the tool results used in a chat session.
    
    Args:
        chat_id: The ID of the chat session.
        
    Returns:
        ToolResultsResponse: The tool results used in the chat session.
    """
    # Get the tool results for the chat
    tracker = MCPServerTracker.getInstance()
    tool_results = tracker.get_last_tool_results(chat_id) or []
    
    return ToolResultsResponse(
        success=True,
        message=None,
        tool_results=tool_results,
    )
