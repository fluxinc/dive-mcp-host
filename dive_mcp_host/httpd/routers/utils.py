import asyncio
import json
from collections.abc import AsyncGenerator, Callable, Coroutine, Iterator
from typing import TYPE_CHECKING, Any, Awaitable, Self
from uuid import uuid4

from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel
from starlette.datastructures import State

from dive_mcp_host.httpd.database.models import Message, NewMessage, QueryInput, Role
from dive_mcp_host.httpd.routers.models import (
    ChatInfoContent,
    McpServerManager,
    MessageInfoContent,
    ModelManager,
    ModelType,
    StreamMessage,
    TokenUsage,
)
from dive_mcp_host.httpd.store.store import SUPPORTED_IMAGE_EXTENSIONS

if TYPE_CHECKING:
    from langchain_core.messages.ai import AIMessageChunk

    from dive_mcp_host.httpd.database.msg_store.abstract import AbstractMessageStore


class EventStreamContextManager:
    """Context manager for event streaming."""

    task: asyncio.Task | None = None
    done: bool = False
    response: StreamingResponse | None = None

    def __init__(self) -> None:
        """Initialize the event stream context manager."""
        self.queue = asyncio.Queue()

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
        app_state: State,
        request_state: State,
        stream: EventStreamContextManager,
    ) -> None:
        """Initialize chat processor."""
        self.app_state = app_state
        self.request_state = request_state
        self.stream = stream
        self.db: AbstractMessageStore = app_state.db
        self.mcp: McpServerManager = app_state.mcp

    async def handle_chat(
        self,
        chat_id: str | None,
        query_input: QueryInput,
        regenerate_message_id: str | None,
    ) -> str:
        """Handle chat."""
        db_opts = self.request_state.get_kwargs("db_opts")

        if chat_id is None:
            chat_id = str(uuid4())

        history: list[BaseMessage] = []
        title = "New Chat"
        title_await = None
        system_prompt = ""  # TODO: get system prompt

        if system_prompt:
            history.append(SystemMessage(content=system_prompt))

        message_history = await self.db.get_chat_with_messages(chat_id, **db_opts)

        if message_history:
            title = message_history.chat.title

            if regenerate_message_id:
                target_index = next(
                    (
                        i
                        for i, message in enumerate(message_history.messages)
                        if message.id == regenerate_message_id
                    ),
                    None,
                )
                if target_index is not None:
                    message_history.messages = message_history.messages[:target_index]

            # TODO: max N history messages

            history = await self.__process_history_messages(
                message_history.messages, history
            )
        elif query_input.text:
            # TODO: generate title
            model_manager: ModelManager = self.app_state.model_manager
            title_await = model_manager.generate_title(query_input.text)

        await self.stream.write(
            StreamMessage(
                type="chat_info",
                content=ChatInfoContent(id=chat_id, title=title),
            ).model_dump_json()
        )

        result, token_usage = await self.__process_chat(chat_id, query_input, history)

        if title_await:
            title = await title_await

        if not await self.db.check_chat_exists(chat_id, **db_opts):
            await self.db.create_chat(chat_id, title, **db_opts)

        user_message_id = str(uuid4())
        if regenerate_message_id:
            await self.db.delete_messages_after(
                chat_id, regenerate_message_id, **db_opts
            )
        else:
            files = (query_input.images or []) + (query_input.documents or [])
            await self.db.create_message(
                NewMessage(
                    chatId=chat_id,
                    role=Role.USER,
                    messageId=user_message_id,
                    content=query_input.text or "",
                    files=json.dumps(files),
                ),
                **db_opts,
            )

        assistant_message_id = str(uuid4())
        await self.db.create_message(
            NewMessage(
                chatId=chat_id,
                role=Role.ASSISTANT,
                messageId=assistant_message_id,
                content=result,
            ),
            **db_opts,
        )

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

    async def __process_chat(
        self,
        chat_id: str | None,
        query_input: str | QueryInput | None,
        history: list[BaseMessage],
    ) -> tuple[str, TokenUsage]:
        """Process chat.

        Args:
            chat_id (str): The unique identifier of the chat.
            query_input (QueryInput): The input query containing text and/or files.
            history (list[Message]): List of previous messages in the chat.

        Returns:
            tuple[str, TokenUsage]: Assistant message ID and token usage statistics.
        """
        tool_client_map = await self.mcp.get_tool_to_server_map()
        available_tools = await self.mcp.get_available_tools()
        model_manager: ModelManager = self.app_state.model_manager
        model = model_manager.get_model()
        current_model_settings = model_manager.current_model_settings

        if chat_id:
            # TODO: abort controller
            ...

        final_response = ""
        token_usage = TokenUsage(
            totalInputTokens=0,
            totalOutputTokens=0,
            totalTokens=0,
        )

        messages = history

        if not model:
            raise ChatError("Model not initialized")

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
                    base64_image = ""  # TODO: image to base64
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

        has_tool_calls = True
        tools = []  # TODO get tool definitions
        run_model = model.bind_tools(tools) if model_manager.enable_tools else model

        is_ollama = (
            current_model_settings.model_type == ModelType.OLLAMA
            if current_model_settings
            else False
        )
        is_deepseek = (
            current_model_settings.model_type == ModelType.DEEPSEEK
            if current_model_settings
            else False
        )
        is_mistral = (
            current_model_settings.model_type == ModelType.MISTRAL
            if current_model_settings
            else False
        )
        is_bedrock = (
            current_model_settings.model_type == ModelType.BEDROCK
            if current_model_settings
            else False
        )

        while has_tool_calls:
            # TODO: abort controller
            stream: Iterator[AIMessageChunk] = run_model.stream(messages)  # type: ignore  # noqa: PGH003
            current_content = ""
            tool_calls = []

            for chunk in stream:
                chunk: AIMessageChunk
                # TODO: calc token usage

                if chunk.content:
                    chunk_message = ""
                    if isinstance(chunk.content, list):
                        # compatible Anthropic response format
                        for item in chunk.content:
                            if isinstance(item, dict) and (
                                item.get("type") == "text"
                                or item.get("type") == "text_delta"
                            ):
                                chunk_message = item.get("text", "")
                                break
                    else:
                        chunk_message = chunk.content

                    current_content += chunk_message
                    await self.stream.write(
                        StreamMessage(
                            type="text",
                            content=chunk_message,
                        ).model_dump_json()
                    )

                is_tool_use = isinstance(chunk.content, list) and any(
                    (isinstance(x, dict) and x.get("type") == "tool_use")
                    for x in chunk.content
                )

                if chunk.tool_calls or chunk.tool_call_chunks or is_tool_use:
                    tool_call_chunks = chunk.tool_call_chunks or []

                    for chunks in tool_call_chunks:
                        index = chunks["index"]
                        if (
                            is_ollama
                            and index
                            and index > 0
                            and len(tool_calls) >= index
                        ):
                            ...

        raise NotImplementedError("Not implemented")

    async def __process_history_messages(
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
                        # TODO: image to base64
                        ...
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
