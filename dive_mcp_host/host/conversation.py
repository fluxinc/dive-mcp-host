import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Any, Self, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import MessagesState
from langgraph.store.base import BaseStore
from langgraph.types import StreamMode

from dive_mcp_host.host.agents import AgentFactory, V
from dive_mcp_host.host.errors import (
    GraphNotCompiledError,
    MessageTypeError,
    ThreadNotFoundError,
)
from dive_mcp_host.host.helpers.context import ContextProtocol
from dive_mcp_host.host.prompt import default_system_prompt

logger = logging.getLogger(__name__)


class Conversation[STATE_TYPE: MessagesState](ContextProtocol):
    """A conversation with a language model."""

    def __init__(  # noqa: PLR0913, too many arguments
        self,
        model: BaseChatModel,
        agent_factory: AgentFactory[STATE_TYPE],
        *,
        system_prompt: str | Callable[[STATE_TYPE], list[BaseMessage]] | None = None,
        thread_id: str | None = None,
        user_id: str = "default",
        store: BaseStore | None = None,
        checkpointer: BaseCheckpointSaver[V] | None = None,
    ) -> None:
        """Initialize the conversation.

        Args:
            model: The language model to use for the conversation.
            agent_factory: The agent factory to use for the conversation.
            system_prompt: The system prompt to use for the conversation.
            thread_id: The ID of the thread.
            user_id: The user ID to use for the conversation.
            store: The store to use for the conversation.
            checkpointer: The langgraph checkpointer to use for the conversation.
        The agent_factory is called only once to compile the agent.
        """
        self._thread_id: str = thread_id if thread_id else uuid.uuid4().hex
        self._user_id: str = user_id
        self._store = store
        self._checkpointer = checkpointer
        self._model = model
        self._system_prompt = system_prompt
        self._agent: CompiledGraph | None = None
        self._agent_factory: AgentFactory[STATE_TYPE] = agent_factory
        self._abort_signal: asyncio.Event | None = None

    @property
    def active_agent(self) -> CompiledGraph:
        """The active agent of the conversation."""
        if self._agent is None:
            raise GraphNotCompiledError(self._thread_id)
        return self._agent

    @property
    def thread_id(self) -> str:
        """The thread ID of the conversation."""
        return self._thread_id

    def abort(self) -> None:
        """Abort the conversation."""
        if self._abort_signal is None:
            return
        self._abort_signal.set()

    async def _run_in_context(self) -> AsyncGenerator[Self, None]:
        if self._system_prompt is None:
            system_prompt = default_system_prompt()
        else:
            system_prompt = self._system_prompt
        if callable(system_prompt):
            prompt = system_prompt
        else:
            prompt = self._agent_factory.create_prompt(system_prompt=system_prompt)
        # we can do something to the prompt here.
        self._agent = self._agent_factory.create_agent(
            prompt=prompt,
            checkpointer=self._checkpointer,
            store=self._store,
        )
        try:
            yield self
        finally:
            self._agent = None

    async def _get_updates_for_resend(
        self,
        resend: list[BaseMessage],
        update: list[BaseMessage],
    ) -> list[BaseMessage]:
        if not self._checkpointer:
            return update
        resend_map = {msg.id: msg for msg in resend}
        to_update = [i for i in update if i.id not in resend_map]
        if state := await self.active_agent.aget_state(
            RunnableConfig(
                configurable={
                    "thread_id": self._thread_id,
                    "user_id": self._user_id,
                },
            )
        ):
            drop_after = False
            for msg in cast(MessagesState, state.values)["messages"]:
                assert msg.id is not None  # all messages from the agent have an ID
                if msg.id in resend_map:
                    drop_after = True
                elif drop_after:
                    to_update.append(RemoveMessage(msg.id))
            return to_update
        raise ThreadNotFoundError(self._thread_id)

    def query(
        self,
        query: str | HumanMessage | list[BaseMessage] | None,
        *,
        stream_mode: list[StreamMode] | StreamMode | None = "messages",
        modify: list[BaseMessage] | None = None,
        is_resend: bool = False,
    ) -> AsyncIterator[dict[str, Any] | Any]:
        """Query the conversation.

        Args:
            query: The query to ask the conversation. Can be a string, HumanMessage, or
                list of messages.
                For resending messages, pass the messages to resend here.
            stream_mode: The mode to stream the response.
            modify: Messages to modify in the conversation state. Used for modifying
                messages without resending, e.g. when confirming tool call parameters.
            is_resend: If True, indicates that query contains messages to resend. The
                messages in query and all subsequent messages in the state will be
                removed. Any messages in modify that appear in query will be ignored.

        Returns:
            An async generator of the response.

        Raises:
            MessageTypeError: If the messages to modify are invalid.
        """

        async def _stream_response() -> AsyncGenerator[dict[str, Any] | Any, None]:
            query_msgs = _convert_query_to_messages(query)
            if is_resend and query_msgs:
                if len(query_msgs) == 0 or not all(
                    isinstance(msg, BaseMessage) and msg.id for msg in query_msgs
                ):
                    raise MessageTypeError("Resending messages must has an ID")
                query_msgs += await self._get_updates_for_resend(
                    query_msgs, modify or []
                )
            elif modify:
                query_msgs = [*query_msgs, *modify]
            signal = asyncio.Event()
            self._abort_signal = signal
            if query_msgs:
                init_state = self._agent_factory.create_initial_state(query=query_msgs)
            else:
                init_state = None
            logger.debug("init_state: %s", query_msgs)
            config = self._agent_factory.create_config(
                user_id=self._user_id,
                thread_id=self._thread_id,
            )
            async for response in self.active_agent.astream(
                input=init_state,
                stream_mode=stream_mode,
                config=config,
            ):
                if signal.is_set():
                    break
                yield response

        return _stream_response()


def _convert_query_to_messages(
    query: str | HumanMessage | list[BaseMessage] | None,
) -> list[BaseMessage]:
    if isinstance(query, BaseMessage):
        return [query]
    if isinstance(query, str):
        return [HumanMessage(content=query)]
    if isinstance(query, list):
        return [
            i if isinstance(i, BaseMessage) else HumanMessage(content=i) for i in query
        ]
    return []
