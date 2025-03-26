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

from .errors import GraphNotCompiledError

if TYPE_CHECKING:
    from langgraph.graph.graph import CompiledGraph

from dive_mcp_host.host.agents import AgentFactory, V
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
        yield self
        self._agent = None

    def query(
        self,
        query: str | HumanMessage | list[BaseMessage],
        *,
        stream_mode: list[StreamMode] | StreamMode | None = "messages",
    ) -> AsyncIterator[dict[str, Any] | Any]:
        """Query the conversation.

        Args:
            query: The query to ask the conversation.
            stream_mode: The mode to stream the response.

        Returns:
            An async generator of the response.
        """

        async def _stream_response() -> AsyncGenerator[dict[str, Any] | Any, None]:
            if self._agent is None:
                raise GraphNotCompiledError(self._thread_id)
            signal = asyncio.Event()
            self._abort_signal = signal
            async for response in self._agent.astream(
                self._agent_factory.create_initial_state(
                    query=query,
                ),
                stream_mode=stream_mode,
                config=self._agent_factory.create_config(
                    user_id=self._user_id,
                    thread_id=self._thread_id,
                ),
            ):
                if signal.is_set():
                    break
                yield response

        return _stream_response()
