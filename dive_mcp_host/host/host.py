import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from copy import deepcopy
from typing import TYPE_CHECKING, Self

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import MessagesState
from langgraph.prebuilt.tool_node import ToolNode

from dive_mcp_host.host.agents import AgentFactory, get_chat_agent_factory
from dive_mcp_host.host.conf import HostConfig
from dive_mcp_host.host.conversation import Conversation
from dive_mcp_host.host.errors import ThreadNotFoundError
from dive_mcp_host.host.helpers.checkpointer import get_checkpointer
from dive_mcp_host.host.helpers.context import ContextProtocol
from dive_mcp_host.host.tools import McpServerInfo, ToolManager
from dive_mcp_host.models import load_model

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


logger = logging.getLogger(__name__)


class DiveMcpHost(ContextProtocol):
    """The Model Context Protocol (MCP) Host.

    The DiveMcpHost class provides an async context manager interface for managing
    and interacting with language models through the Model Context Protocol (MCP).
    It handles initialization and cleanup of model instances, manages server
    connections, and provides a unified interface for agent conversations.

    The MCP enables tools and models to communicate in a standardized way, allowing for
    consistent interaction patterns regardless of the underlying model implementation.

    Example:
        # Initialize host with configuration
        config = HostConfig(...)
        thread_id = ""
        async with DiveMcpHost(config) as host:
            # Send a message and get response
            async with host.conversation() as conversation:
                while query := input("Enter a message: "):
                    if query == "exit":
                        nonlocal thread_id
                        # save the thread_id for resume
                        thread_id = conversation.thread_id
                        break
                    async for response in await conversation.query(query):
                        print(response)
        ...
        # Resume conversation
        async with DiveMcpHost(config) as host:
            # pass the thread_id to resume the conversation
            async with host.conversation(thread_id=thread_id) as conversation:
                ...

    The host must be used as an async context manager to ensure proper resource
    management, including model initialization and cleanup.
    """

    def __init__(
        self,
        config: HostConfig,
    ) -> None:
        """Initialize the host.

        Args:
            config: The host configuration.
        """
        self._config = config
        self._model: BaseChatModel | None = None
        self._tools: Sequence[BaseTool] = []
        self._checkpointer: BaseCheckpointSaver[str] | None = None
        self._tool_manager: ToolManager = ToolManager(self._config.mcp_servers)
        self._exit_stack: AsyncExitStack | None = None

    async def _run_in_context(self) -> AsyncGenerator[Self, None]:
        async with AsyncExitStack() as stack:
            self._exit_stack = stack
            await self._init_models()
            if self._config.checkpointer:
                checkpointer = get_checkpointer(str(self._config.checkpointer.uri))
                self._checkpointer = await stack.enter_async_context(checkpointer)
                await self._checkpointer.setup()
            await stack.enter_async_context(self._tool_manager)
            try:
                self._tools = self._tool_manager.langchain_tools()
                yield self
            except Exception as e:
                raise e

    async def _init_models(self) -> None:
        if self._model:
            return
        model = load_model(
            self._config.llm.modelProvider,
            self._config.llm.model,
            **self._config.llm.to_load_model_kwargs(),
        )
        self._model = model

    def conversation[T: MessagesState](  # noqa: PLR0913 Is there a better way to do this?
        self,
        *,
        thread_id: str | None = None,
        user_id: str = "default",
        tools: Sequence[BaseTool] | None = None,
        get_agent_factory_method: Callable[
            [BaseChatModel, Sequence[BaseTool] | ToolNode],
            AgentFactory[T],
        ] = get_chat_agent_factory,
        system_prompt: str | Callable[[T], list[BaseMessage]] | None = None,
        volatile: bool = False,
    ) -> Conversation[T]:
        """Start or resume a conversation.

        Args:
            thread_id: The thread ID to use for the conversation.
            user_id: The user ID to use for the conversation.
            tools: The tools to use for the conversation.
            system_prompt: Use a custom system prompt for the conversation.
            get_agent_factory_method: The method to get the agent factory.
            volatile: if True, the conversation will not be saved.

        If the thread ID is not provided, a new thread will be created.
        Customize the agent factory to use a different model or tools.
        If the tools are not provided, the host will use the tools initialized in the
        host.
        """
        if self._model is None:
            raise RuntimeError("Model not initialized")
        if tools is None:
            tools = self._tool_manager.langchain_tools()
        agent_factory = get_agent_factory_method(
            self._model,
            tools,
        )
        return Conversation(
            model=self._model,
            agent_factory=agent_factory,
            system_prompt=system_prompt,
            thread_id=thread_id,
            user_id=user_id,
            checkpointer=None if volatile else self._checkpointer,
        )

    async def reload(
        self,
        new_config: HostConfig,
        reloader: Callable[[], Awaitable[None]],
    ) -> None:
        """Reload the host with a new configuration.

        Args:
            new_config: The new configuration.
            reloader: The reloader function.

        The reloader function is called when the host is ready to reload. This means
        all ongoing conversations have completed and no new queries are being processed.
        The reloader should handle stopping and restarting services as needed.
        Conversations can be resumed after reload by using the same thread_id.
        """
        # NOTE: Do Not restart MCP Servers when there is on-going query.
        raise NotImplementedError

    @property
    def config(self) -> HostConfig:
        """A copy of the current host configuration.

        Note: Do not modify the returned config. Use `reload` to change the config.
        """
        return deepcopy(self._config)

    @property
    def tools(self) -> Sequence[BaseTool]:
        """The ACTIVE tools to the host.

        This property is read-only. Call `reload` to change the tools.
        """
        return self._tools

    @property
    def mcp_server_info(self) -> dict[str, McpServerInfo]:
        """Get information about active MCP servers.

        Returns:
            A dictionary mapping server names to their capabilities and tools.
            The value will be None for any server that has not completed initialization.
        """
        return self._tool_manager.mcp_server_info

    @property
    def model(self) -> BaseChatModel:
        """The model of the host."""
        if self._model is None:
            raise RuntimeError("Model not initialized")
        return self._model

    async def get_messages(self, thread_id: str, user_id: str) -> list[BaseMessage]:
        """Get messages of a specific thread.

        Args:
            thread_id: The thread ID to retrieve messages for.
            user_id: The user ID to retrieve messages for.

        Returns:
            A list of messages.

        Raises:
            ThreadNotFoundError: If the thread is not found.
        """
        if self._checkpointer is None:
            return []

        if ckp := await self._checkpointer.aget(
            {
                "configurable": {
                    "thread_id": thread_id,
                    "user_id": user_id,
                }
            }
        ):
            return ckp["channel_values"].get("messages", [])
        raise ThreadNotFoundError(thread_id)
