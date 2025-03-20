"""Model for the MCP servers."""

from __future__ import annotations

import asyncio
import logging
import os
from asyncio import CancelledError
from collections.abc import AsyncGenerator
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    asynccontextmanager,
    suppress,
)
from enum import Enum, auto
from itertools import chain
from typing import TYPE_CHECKING, Any, Self

from langchain_core.tools import BaseTool, ToolException
from mcp import ClientSession, StdioServerParameters, stdio_client, types
from mcp.client.sse import sse_client
from pydantic import BaseModel, ConfigDict
from pydantic_core import to_json

from dive_mcp_host.host.conf import ServerConfig  # noqa: TC001 Pydantic Need this
from dive_mcp_host.host.helpers.context import ContextProtocol

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream


type ReadStreamType = MemoryObjectReceiveStream[types.JSONRPCMessage | Exception]
type WriteStreamType = MemoryObjectSendStream[types.JSONRPCMessage]
type StreamContextType = AbstractAsyncContextManager[
    tuple[ReadStreamType, WriteStreamType]
]


class McpServerInfo(BaseModel):
    """MCP server capability and tool list."""

    name: str
    """The name of the MCP server."""
    tools: list[types.Tool]
    """The tools provided by the MCP server."""
    initialize_result: types.InitializeResult | None
    """The result of the initialize method.

    initialize_result.capabilities: Server capabilities.
    initialize_result.instructions: Server instructions.
    """

    error: BaseException | None
    """The error that occurred of the MCP server."""

    client_status: ClientState
    """The status of the client: RUNNING, CLOSED, RESTARTING, or INIT."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ToolManager(ContextProtocol):
    """Manager for the MCP Servers.

    Example:
        config = HostConfig(...)
        async with ToolManager(config.mcp_servers) as tool_manager:
            tool_manager.tools()
    """

    def __init__(self, configs: dict[str, ServerConfig]) -> None:
        """Initialize the ToolManager."""
        self._configs = configs
        self._mcp_servers = dict[str, McpServer]()
        self._async_exit_stack = AsyncExitStack()

    def langchain_tools(
        self,
        tool_filter: Callable[[McpServer], bool] = lambda _: True,
    ) -> list[McpTool]:
        """Get the langchain tools for the MCP servers."""
        return list(
            chain.from_iterable(
                [i.get_tools() for i in self._mcp_servers.values() if tool_filter(i)],
            ),
        )

    async def _run_in_context(self) -> AsyncGenerator[Self, None]:
        """Get the langchain tools for the MCP servers."""
        # start tools
        self._mcp_servers = {
            name: McpServer(name=name, config=config)
            for name, config in self._configs.items()
        }

        # we can manipulate the stack to add or remove tools
        async with self._async_exit_stack:
            async with asyncio.TaskGroup() as tg:
                for tool in self._mcp_servers.values():
                    tg.create_task(self._async_exit_stack.enter_async_context(tool))
            yield self

    @property
    def mcp_server_info(self) -> dict[str, McpServerInfo]:
        """Get the MCP server capabilities and tools.

        Returns:
            A dictionary of MCP server name to server info.
            If the mcp server is not initialized, the value will be `None`.
        """
        return {name: i.server_info for name, i in self._mcp_servers.items()}


class ClientState(Enum):
    """The state of the client.

    States and transitions:
    """

    INIT = auto()
    RUNNING = auto()
    CLOSED = auto()
    RESTARTING = auto()
    FAILED = auto()


class McpServer(ContextProtocol):
    """McpServer Toolkit.

    A background task continuously monitors the client's state.
    If the client is closed, the task will restart the client.
    The task stops when the McpServer's status is CLOSED.
    """

    RETRY_LIMIT: int = 3

    def __init__(self, name: str, config: ServerConfig) -> None:
        """Initialize the McpToolKit."""
        self.name = name
        self.config = config
        self._cond = asyncio.Condition()
        """The condition variable to synchronize access to shared variables."""
        self._client_status: ClientState = ClientState.INIT
        self._task: asyncio.Task[Awaitable[None]] | None = None
        self._session: ClientSession | None = None
        self._tool_results: types.ListToolsResult | None = None
        self._initialize_result: types.InitializeResult | None = None
        self._session_count: int = 0
        self._exception: BaseException | None = None
        self._mcp_tools: list[McpTool] = []
        self._retries: int = 0

    async def _client_watcher(self) -> None:
        """Task watcher restart the client if it is closed."""
        while True:
            try:
                async with (
                    self._get_client() as streams,
                    ClientSession(*streams) as session,
                ):
                    self._initialize_result = await session.initialize()
                    tool_results = await session.list_tools()
                    mcp_tools = [
                        McpTool(
                            toolkit_name=self.name,
                            name=tool.name,
                            description=tool.description or "",
                            mcp_server=self,
                        )
                        for tool in tool_results.tools
                    ]
                    async with self._cond:
                        self._client_status = ClientState.RUNNING
                        self._session = session
                        self._tool_results = tool_results
                        self._mcp_tools = mcp_tools
                        self._exception = None
                        self._retries = 0
                        self._cond.notify_all()
                        await self._cond.wait_for(
                            lambda: self._client_status
                            in [ClientState.CLOSED, ClientState.RESTARTING],
                        )
                        if self._client_status == ClientState.CLOSED:  # type: ignore[unreachable]
                            break
            except* BaseException as e:  # noqa: BLE001
                logging.error("unknown error in client watcher: %s", e)
                self._exception = e
                self._retries += 1

            if self._retries > self.RETRY_LIMIT:
                logging.warning(
                    "client %s failed after %d retries",
                    self.name,
                    self._retries,
                )
                async with self._cond:
                    self._client_status = ClientState.FAILED
                    self._cond.notify_all()
                    return
            await asyncio.sleep(1)

    async def _run_in_context(self) -> AsyncGenerator[Self, None]:
        """Get the langchain tools for the MCP servers."""
        task = asyncio.create_task(self._client_watcher())
        async with self._cond:
            await self._cond.wait_for(
                lambda: self._client_status
                in [ClientState.RUNNING, ClientState.CLOSED, ClientState.FAILED]
            )
        try:
            yield self
        finally:
            async with self._cond:
                self._client_status = ClientState.CLOSED
                self._session = None
                self._cond.notify_all()
            await self._cond.wait_for(
                lambda: self._session_count == 0,
            )
        with suppress(Exception):
            await task

    def _get_client(
        self,
    ) -> StreamContextType:
        """Create a new client."""
        if self.config.url is not None:
            return sse_client(
                url=self.config.url,
            )

        env = os.environ.copy()
        env.update(self.config.env)
        return stdio_client(
            StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                env=env,
            ),
        )

    def get_tools(self) -> list[McpTool]:
        """Get the tools."""
        return self._mcp_tools

    def session(self) -> AbstractAsyncContextManager[ClientSession]:
        """Get the session.

        Only one session can exist at a time for a McpStdioServer instance.

        Returns:
            The context manager for the session.
        """

        @asynccontextmanager
        async def session_ctx() -> AsyncGenerator[ClientSession, None]:
            """Get the session."""
            async with self._cond:
                if self._session is None:
                    raise RuntimeError("Session not initialized")
                self._session_count += 1
            try:
                yield self._session
            except CancelledError:
                pass
            # What kinds of exceptions will be raised by a mcp client?
            except Exception as e:  # noqa: BLE001.
                async with self._cond:
                    self._exception = e
                    if self._client_status == ClientState.RUNNING:
                        self._client_status = ClientState.RESTARTING
                    self._session_count -= 1
                    self._cond.notify_all()
            else:
                async with self._cond:
                    self._session_count -= 1
                    self._cond.notify_all()

        return session_ctx()

    @property
    def server_info(self) -> McpServerInfo:
        """Get the server info."""
        return McpServerInfo(
            name=self.name,
            initialize_result=self._initialize_result,
            tools=self._tool_results.tools if self._tool_results is not None else [],
            client_status=self._client_status,
            error=self._exception,
        )


class McpTool(BaseTool):
    """A tool for the MCP."""

    toolkit_name: str
    description: str = ""
    mcp_server: McpServer

    def _run(self, **kwargs: dict[str, Any]) -> str:
        """Run the tool."""
        return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs: dict[str, Any]) -> str:
        """Run the tool."""
        async with self.mcp_server.session() as session:
            result = await session.call_tool(self.name, arguments=kwargs)  # type: ignore[arg-type]
        content = to_json(result.content).decode()
        if result.isError:
            raise ToolException(content)
        return content
