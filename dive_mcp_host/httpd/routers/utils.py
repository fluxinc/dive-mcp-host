import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine
from contextlib import AsyncExitStack, suppress
from typing import TYPE_CHECKING, Any, Self
from uuid import uuid4

from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolMessage
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
- For all other languages, generate the title in the same language as the input."""  # noqa: E501


logger = logging.getLogger(__name__)


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
        self, func: Callable[[], Coroutine[Any, Any, Any]], *args: Any, **kwargs: Any
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

            logger.error(traceback.format_exception(exc_type, exc_val, exc_tb))

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

        yield "data: [DONE]\n\n"

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

        start = time.time()
        user_message, ai_message = await self._process_chat(chat_id, query_input)
        end = time.time()
        duration = ai_message.response_metadata.get("total_duration")
        assert user_message.id
        assert ai_message.id
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
                        model=ai_message.response_metadata.get("model")
                        or ai_message.response_metadata.get("model_name")
                        or "",
                        total_input_tokens=ai_message.usage_metadata["input_tokens"]
                        if ai_message.usage_metadata
                        else 0,
                        total_output_tokens=ai_message.usage_metadata["output_tokens"]
                        if ai_message.usage_metadata
                        else 0,
                        total_run_time=(
                            duration / (10**9) if duration else int(end - start)
                        ),
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
            totalInputTokens=ai_message.usage_metadata["input_tokens"]
            if ai_message.usage_metadata
            else 0,
            totalOutputTokens=ai_message.usage_metadata["output_tokens"]
            if ai_message.usage_metadata
            else 0,
            totalTokens=ai_message.usage_metadata["total_tokens"]
            if ai_message.usage_metadata
            else 0,
        )

        return result, token_usage

    async def handle_chat_with_history(
        self,
        chat_id: str,
        query_input: BaseMessage | None,
        history: list[BaseMessage],
        tools: list | None = None,
    ) -> tuple[str, TokenUsage]:
        """Handle chat with history.

        Args:
            chat_id (str): The chat ID.
            query_input (BaseMessage | None): The query input.
            history (list[BaseMessage]): The history.
            tools (list | None): The tools.

        Returns:
            tuple[str, TokenUsage]: The result and token usage.
        """
        _, ai_message = await self._process_chat(chat_id, query_input, history, tools)
        assert ai_message.usage_metadata

        return str(ai_message.content), TokenUsage(
            totalInputTokens=ai_message.usage_metadata["input_tokens"],
            totalOutputTokens=ai_message.usage_metadata["output_tokens"],
            totalTokens=ai_message.usage_metadata["total_tokens"],
        )

    async def _process_chat(
        self,
        chat_id: str | None,
        query_input: str | QueryInput | BaseMessage | None,
        history: list[BaseMessage] | None = None,
        tools: list | None = None,
    ) -> tuple[HumanMessage, AIMessage]:
        messages = [*history] if history else []

        # if retry input is empty
        if query_input:
            if isinstance(query_input, str):
                messages.append(HumanMessage(content=query_input))
            elif isinstance(query_input, QueryInput):
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
            else:
                messages.append(query_input)

        dive_user: DiveUser = self.request_state.dive_user

        def _prompt_cb(_: Any) -> list[BaseMessage]:
            return messages

        prompt: Callable[..., list[BaseMessage]] | None = None
        if any(isinstance(m, SystemMessage) for m in messages):
            prompt = _prompt_cb

        conversation = self.dive_host.conversation(
            thread_id=chat_id,
            user_id=dive_user.get("user_id") or "default",
            tools=tools,
            system_prompt=prompt,
        )
        async with AsyncExitStack() as stack:
            if chat_id:
                await stack.enter_async_context(
                    self.app.abort_controller.abort_signal(chat_id, conversation.abort)
                )
            await stack.enter_async_context(conversation)
            response = conversation.query(messages, stream_mode=["messages", "values"])
            return await self._handle_response(response)

        raise RuntimeError("Unreachable")

    async def _handle_response(  # noqa: C901, PLR0912
        self, response: AsyncIterator[dict[str, Any] | Any]
    ) -> tuple[HumanMessage | Any, AIMessage | Any]:
        user_message = None
        ai_message = None
        latest_messages = []

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
                    logger.warning("Unknown message type: %s", message)
            elif res_type == "values" and len(res_content["messages"]) >= 2:  # type: ignore  # noqa: PLR2004
                latest_messages = res_content["messages"]  # type: ignore
            else:
                pass

            if event_type and content:
                await self.stream.write(StreamMessage(type=event_type, content=content))

        for message in latest_messages[::-1]:
            if user_message and ai_message:
                break
            if isinstance(message, HumanMessage):
                user_message = message
            elif isinstance(message, AIMessage):
                ai_message = message
        return user_message, ai_message

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
