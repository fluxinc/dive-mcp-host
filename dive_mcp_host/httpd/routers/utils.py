import asyncio
import json
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import TYPE_CHECKING, Any, Self
from uuid import uuid4

from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from pydantic import BaseModel
from starlette.datastructures import State

from dive_mcp_host.httpd.database.models import Message, NewMessage, QueryInput, Role
from dive_mcp_host.httpd.routers.models import (
    ChatInfoContent,
    MessageInfoContent,
    StreamMessage,
    TokenUsage,
)
from dive_mcp_host.httpd.server import DiveHostAPI
from dive_mcp_host.httpd.store.store import SUPPORTED_IMAGE_EXTENSIONS, Store

if TYPE_CHECKING:
    from langchain_core.messages.ai import AIMessageChunk

    from dive_mcp_host.host.host import DiveMcpHost
    from dive_mcp_host.httpd.middlewares.general import DiveUser


class EventStreamContextManager:
    """Context manager for event streaming."""

    task: asyncio.Task | None = None
    done: bool = False
    response: StreamingResponse | None = None

    def __init__(self) -> None:
        """Initialize the event stream context manager."""
        self.queue = asyncio.Queue()

    def __del__(self) -> None:
        """Delete the event stream context manager."""
        self.done = True
        asyncio.create_task(self.queue.put(None))  # noqa: RUF006

    def add_task(
        self, func: Callable[[], Coroutine[Any, Any, None]], *args: Any, **kwargs: Any
    ) -> None:
        """Add a task to the event stream."""
        self.task = asyncio.create_task(func(*args, **kwargs))

    async def __aenter__(self) -> Self:
        """Enter the context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Exit the context manager."""
        self.done = True
        await self.queue.put(None)  # Signal completion

    async def write(self, data: str | BaseModel) -> None:
        """Write data to the event stream.

        Args:
            data (str): The data to write to the stream.
        """
        if isinstance(data, BaseModel):
            data = data.model_dump_json()
        await self.queue.put(data)

    async def _generate(self) -> AsyncGenerator[str, None]:
        """Generate the event stream content."""
        while not self.done or not self.queue.empty():
            chunk = await self.queue.get()
            if chunk is None:  # End signal
                continue
            yield "data: " + chunk + "\n\n"

        yield "[Done]"

    def get_response(self) -> StreamingResponse:
        """Get the streaming response.

        Returns:
            StreamingResponse: The streaming response.
        """
        self.response = StreamingResponse(
            content=self._generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
        return self.response


class ChatError(Exception):
    """Chat error."""

    def __init__(self, message: str) -> None:
        """Initialize chat error."""
        self.message = message


class ChatProcessor:
    """Chat processor."""

    def __init__(
        self,
        app: DiveHostAPI,
        request_state: State,
        stream: EventStreamContextManager,
    ) -> None:
        """Initialize chat processor."""
        self.app = app
        self.request_state = request_state
        self.stream = stream
        self.store: Store = app.store
        self.dive_host: DiveMcpHost = app.dive_host["default"]

    async def handle_chat(
        self,
        chat_id: str | None,
        query_input: QueryInput,
        regenerate_message_id: str | None,
    ) -> str:
        """Handle chat."""
        if chat_id is None:
            chat_id = str(uuid4())

        title = "New Chat"
        title_await = None

        if query_input.text:
            title_await = self._generate_title(query_input.text)

        await self.stream.write(
            StreamMessage(
                type="chat_info",
                content=ChatInfoContent(id=chat_id, title=title),
            ).model_dump_json()
        )

        result, token_usage = await self._process_chat(chat_id, query_input)

        if title_await:
            title = await title_await

        dive_user: DiveUser = self.request_state.dive_user
        async with self.app.db_sessionmaker() as session:
            db = self.app.msg_store(session)
            if not await db.check_chat_exists(chat_id, dive_user["user_id"]):
                await db.create_chat(
                    chat_id, title, dive_user["user_id"], dive_user["user_type"]
                )

            user_message_id = str(uuid4())
            if regenerate_message_id:
                await db.delete_messages_after(chat_id, regenerate_message_id)
            else:
                files = (query_input.images or []) + (query_input.documents or [])
                await db.create_message(
                    NewMessage(
                        chatId=chat_id,
                        role=Role.USER,
                        messageId=user_message_id,
                        content=query_input.text or "",
                        files=json.dumps(files),
                    ),
                )

            assistant_message_id = str(uuid4())
            await db.create_message(
                NewMessage(
                    chatId=chat_id,
                    role=Role.ASSISTANT,
                    messageId=assistant_message_id,
                    content=result,
                ),
            )

            await session.commit()

        await self.stream.write(
            StreamMessage(
                type="message_info",
                content=MessageInfoContent(
                    userMessageId=user_message_id,
                    assistantMessageId=assistant_message_id,
                ),
            ).model_dump_json()
        )

        await self.stream.write(
            StreamMessage(
                type="chat_info",
                content=ChatInfoContent(id=chat_id, title=title),
            ).model_dump_json()
        )

        return result

    async def _process_chat(
        self,
        chat_id: str | None,
        query_input: str | QueryInput | None,
    ) -> tuple[str, TokenUsage]:
        """Process chat.

        Args:
            chat_id (str): The unique identifier of the chat.
            query_input (QueryInput): The input query containing text and/or files.
            history (list[Message]): List of previous messages in the chat.

        Returns:
            tuple[str, TokenUsage]: Assistant message ID and token usage statistics.
        """
        if chat_id:
            # TODO: abort controller
            ...

        final_response = ""
        token_usage = TokenUsage(
            totalInputTokens=0,
            totalOutputTokens=0,
            totalTokens=0,
        )

        messages = []

        # if retry input is empty
        if query_input:
            if isinstance(query_input, str):
                messages.append(HumanMessage(content=query_input))
            else:
                content = []

                if query_input.text:
                    content.append(
                        {
                            "type": "text",
                            "text": query_input.text,
                        }
                    )

                for image in query_input.images or []:
                    local_path = image
                    base64_image = await self.store.get_image(local_path)
                    content.append(
                        {
                            "type": "text",
                            "text": f"![Image]({base64_image})",
                        }
                    )
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image,
                            },
                        }
                    )

                for document in query_input.documents or []:
                    local_path = document
                    content.append(
                        {
                            "type": "text",
                            "text": f"![Document]({local_path})",
                        }
                    )

                messages.append(HumanMessage(content=content))

        dive_user: DiveUser = self.request_state.dive_user

        conversation = self.dive_host.conversation(
            thread_id=chat_id, user_id=dive_user["user_id"] or "default"
        )
        async with conversation:
            async for message_chunk, _ in conversation.query(  # type: ignore noqa: PGH003
                messages[0], stream_mode="messages"
            ):
                message_chunk: AIMessageChunk
                await self.stream.write(
                    StreamMessage(
                        type="text",
                        content=str(message_chunk.content),
                    )
                )

        return final_response, token_usage

    async def _generate_title(self, query: str) -> str:
        """Generate title."""
        return "New Chat"  # TODO: generate title

    async def _process_history_messages(
        self, history_messages: list[Message], history: list[BaseMessage]
    ) -> list[BaseMessage]:
        """Process history messages."""
        for message in history_messages:
            files: list[str] = json.loads(message.files)
            if not files:
                message_content = message.content.strip()
                if message.role == Role.USER:
                    history.append(HumanMessage(content=message_content))
                else:
                    history.append(AIMessage(content=message_content))
            else:
                content = []
                if message.content:
                    content.append(
                        {
                            "type": "text",
                            "text": message.content,
                        }
                    )

                for file_path in files:
                    local_path = file_path
                    if any(
                        local_path.endswith(suffix)
                        for suffix in SUPPORTED_IMAGE_EXTENSIONS
                    ):
                        base64_image = await self.store.get_image(local_path)

                        content.append(
                            {
                                "type": "text",
                                "text": f"![Image]({base64_image})",
                            }
                        )
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_image,
                                },
                            }
                        )
                    else:
                        content.append(
                            {
                                "type": "text",
                                "text": f"![Document]({local_path})",
                            }
                        )

                if message.role == Role.ASSISTANT:
                    history.append(AIMessage(content=content))
                else:
                    history.append(HumanMessage(content=content))

        return history
