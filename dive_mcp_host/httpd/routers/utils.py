import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine
from typing import TYPE_CHECKING, Any, Self, cast
from uuid import uuid4
from contextlib import suppress

from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage, ToolCall
from pydantic import BaseModel
from starlette.datastructures import State

from dive_mcp_host.httpd.database.models import (
    Message,
    NewMessage,
    QueryInput,
    ResourceUsage,
    Role,
)
from dive_mcp_host.httpd.routers.models import (
    ChatInfoContent,
    MessageInfoContent,
    StreamMessage,
    TokenUsage,
    ToolCallsContent,
    ToolResultContent,
)
from dive_mcp_host.httpd.server import DiveHostAPI
from dive_mcp_host.httpd.store.store import SUPPORTED_IMAGE_EXTENSIONS, Store
from dive_mcp_host.models.fake import FakeMessageToolModel

if TYPE_CHECKING:
    from dive_mcp_host.host.host import DiveMcpHost
    from dive_mcp_host.httpd.middlewares.general import DiveUser

title_prompt = """You are a title generator from the user input.
Your only task is to generate a short title based on the user input.
IMPORTANT:
- Output ONLY the title
- DO NOT try to answer or resolve the user input query.
- DO NOT try to use any tools to generate title
- NO explanations, quotes, or extra text
- NO punctuation at the end
- If the input is URL only, output the description of the URL, for example, "the URL of xxx website"
- If the input contains Traditional Chinese characters, use Traditional Chinese for the title.
- For all other languages, generate the title in the same language as the input."""


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
        if exc_val:
            import traceback

            print(traceback.format_exception(exc_type, exc_val, exc_tb))

        self.done = True
        await self.queue.put(None)  # Signal completion

    async def write(self, data: str | StreamMessage) -> None:
        """Write data to the event stream.

        Args:
            data (str): The data to write to the stream.
        """
        if isinstance(data, BaseModel):
            data = json.dumps({"message": data.model_dump(mode="json")})
        await self.queue.put(data)

    async def _generate(self) -> AsyncGenerator[str, None]:
        """Generate the event stream content."""
        while not self.done or not self.queue.empty():
            chunk = await self.queue.get()
            if chunk is None:  # End signal
                continue
            yield "data: " + chunk + "\n\n"

        yield "data: [Done]\n\n"

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
        query_input: str | QueryInput | None,
        regenerate_message_id: str | None,
    ) -> tuple[str, TokenUsage]:
        """Handle chat."""
        if chat_id is None:
            chat_id = str(uuid4())

        title = "New Chat"
        title_await = None

        if isinstance(query_input, QueryInput) and query_input.text:
            title_await = self._generate_title(query_input.text)

        await self.stream.write(
            StreamMessage(
                type="chat_info",
                content=ChatInfoContent(id=chat_id, title=title),
            )
        )

        user_message, ai_message = await self._process_chat(chat_id, query_input)
        assert user_message.id
        assert ai_message.id
        assert ai_message.usage_metadata
        result = str(ai_message.content)

        if title_await:
            title = await title_await

        dive_user: DiveUser = self.request_state.dive_user
        async with self.app.db_sessionmaker() as session:
            db = self.app.msg_store(session)
            if not await db.check_chat_exists(chat_id, dive_user["user_id"]):
                await db.create_chat(
                    chat_id, title, dive_user["user_id"], dive_user["user_type"]
                )

            if regenerate_message_id:
                await db.delete_messages_after(chat_id, regenerate_message_id)
            elif isinstance(query_input, QueryInput):
                files = (query_input.images or []) + (query_input.documents or [])
                await db.create_message(
                    NewMessage(
                        chatId=chat_id,
                        role=Role.USER,
                        messageId=user_message.id,
                        content=query_input.text or "",
                        files=json.dumps(files),
                    ),
                )

            await db.create_message(
                NewMessage(
                    chatId=chat_id,
                    role=Role.ASSISTANT,
                    messageId=ai_message.id,
                    content=result,
                    resource_usage=ResourceUsage(
                        model=ai_message.response_metadata["model"],
                        total_input_tokens=ai_message.usage_metadata["input_tokens"],
                        total_output_tokens=ai_message.usage_metadata["output_tokens"],
                        total_run_time=ai_message.response_metadata["total_duration"]
                        / (10**9),
                    ),
                ),
            )

            await session.commit()

        await self.stream.write(
            StreamMessage(
                type="message_info",
                content=MessageInfoContent(
                    userMessageId=user_message.id,
                    assistantMessageId=ai_message.id,
                ),
            )
        )

        await self.stream.write(
            StreamMessage(
                type="chat_info",
                content=ChatInfoContent(id=chat_id, title=title),
            )
        )

        token_usage = TokenUsage(
            totalInputTokens=ai_message.usage_metadata["input_tokens"],
            totalOutputTokens=ai_message.usage_metadata["output_tokens"],
            totalTokens=ai_message.usage_metadata["total_tokens"],
        )

        return result, token_usage

    async def _process_chat(
        self,
        chat_id: str | None,
        query_input: str | QueryInput | None,
    ) -> tuple[HumanMessage, AIMessage]:
        if chat_id:
            # TODO: abort controller
            ...

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
            thread_id=chat_id, user_id=dive_user.get("user_id") or "default"
        )
        async with conversation:
            response = conversation.query(messages, stream_mode=["messages", "values"])
            return await self._handle_response(response)

    async def _handle_response(
        self, response: AsyncIterator[dict[str, Any] | Any]
    ) -> tuple[HumanMessage, AIMessage]:
        user_message = None
        ai_message = None

        async for res_type, res_content in response:
            event_type = None
            content = None
            if res_type == "messages":
                message, _ = res_content
                if isinstance(message, AIMessage):
                    event_type = "tool_calls"
                    if calls := message.tool_calls:
                        content = [
                            ToolCallsContent(name=c["name"], arguments=c["args"])
                            for c in calls
                        ]
                    else:
                        event_type = "text"
                        content = str(message.content)
                if isinstance(message, AIMessageChunk):
                    event_type = "text"
                    content = str(message.content)
                elif isinstance(message, ToolMessage):
                    event_type = "tool_result"
                    result = message.content
                    with suppress(json.JSONDecodeError):
                        if isinstance(result, list):
                            result = [
                                json.loads(r) if isinstance(r, str) else r
                                for r in result
                            ]
                        else:
                            result = json.loads(result)
                    content = ToolResultContent(name=message.name or "", result=result)
                else:
                    # idk what is this
                    pass
            elif res_type == "values" and len(res_content["messages"]) >= 2:  # type: ignore
                user_message, ai_message = res_content["messages"][-2:]  # type: ignore
            else:
                pass

            if event_type and content:
                await self.stream.write(StreamMessage(type=event_type, content=content))

        return user_message, ai_message  # type: ignore

    async def _generate_title(self, query: str) -> str:
        """Generate title."""
        conversation = self.dive_host.conversation(
            tools=[],  # do not use tools
            system_prompt=title_prompt,
            volatile=True,
        )
        async with conversation:
            responses = [
                response
                async for response in conversation.query(query, stream_mode="updates")
            ]
            return responses[0]["agent"]["messages"][0].content
        return "New Chat"

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
