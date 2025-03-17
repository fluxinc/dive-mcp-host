from collections.abc import AsyncGenerator
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi.responses import StreamingResponse
from langchain_core.messages import SystemMessage
from starlette.datastructures import State

from dive_mcp_host.httpd.routers.models import (
    ChatInfoContent,
    LLMModel,
    McpServerManager,
    Message,
    MessageInfoContent,
    NewMessage,
    QueryInput,
    StreamMessage,
    TokenUsage,
)

if TYPE_CHECKING:
    from dive_mcp_host.httpd.database import Database


def event_stream(
    content: AsyncGenerator[str, None],
) -> StreamingResponse:
    """Event stream for chat.

    Args:
        content (AsyncGenerator[str, None]): The content to stream.
    """

    async def stream_content() -> AsyncGenerator[str, None]:
        async for chunk in content:
            yield "data: " + chunk + "\n\n"

    return StreamingResponse(
        content=stream_content(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


class ChatProcessor:
    """Chat processor."""

    def __init__(self, app_state: State, request_state: State) -> None:
        """Initialize chat processor."""
        self.app_state = app_state
        self.request_state = request_state
        self.db: Database = app_state.db
        self.mcp: McpServerManager = app_state.mcp

    async def handle_chat(
        self,
        chat_id: str | None,
        query_input: QueryInput,
        regenerate_message_id: str | None,
    ) -> AsyncGenerator[str, None]:
        """Handle chat."""
        db_opts = self.request_state.get_kwargs("db_opts")

        if chat_id is None:
            chat_id = str(uuid4())

        history = []
        title = "New Chat"
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
                    history = message_history.messages[:target_index]

            # TODO: process hostory
        elif query_input.text:
            # TODO: generate title
            ...

        yield StreamMessage(
            type="chat_info",
            content=ChatInfoContent(id=chat_id, title=title),
        ).model_dump_json()

        result, token_usage = await self.process_chat(chat_id, query_input, history)

        # TODO: handle title promise

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
                    role="user",
                    messageId=user_message_id,
                    content=query_input.text or "",
                    files=files,
                    createdAt=datetime.now().astimezone(),
                    id=None,
                ),
                **db_opts,
            )

        assistant_message_id = str(uuid4())
        db_opts["llm_model"] = LLMModel(
            model="gpt-4o-mini",  # TODO: model manager get model
            total_input_tokens=0,
            total_output_tokens=0,
            total_run_time=0,
        )
        await self.db.create_message(
            NewMessage(
                chatId=chat_id,
                role="assistant",
                messageId=assistant_message_id,
                content="",
                files=[],
                createdAt=datetime.now().astimezone(),
                id=None,
            ),
            **db_opts,
        )

        yield StreamMessage(
            type="message_info",
            content=MessageInfoContent(
                userMessageId=user_message_id,
                assistantMessageId=assistant_message_id,
            ),
        ).model_dump_json()

    async def process_chat(
        self,
        chat_id: str | None,
        query_input: QueryInput,
        history: list[Message],
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

        if chat_id:
            # TODO: abort controller
            ...

        final_response = ""
        raise NotImplementedError("Not implemented")
