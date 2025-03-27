"""Model for the MCP servers."""

from __future__ import annotations

import asyncio
import logging
import os
import time
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
from json import JSONDecodeError
from json import loads as json_loads
from typing import TYPE_CHECKING, Any, Literal, Self

import httpx
from langchain_core.tools import BaseTool, ToolException
from mcp import ClientSession, StdioServerParameters, stdio_client, types
from mcp.client.sse import sse_client
from mcp.client.websocket import websocket_client
from pydantic import BaseModel, ConfigDict
from pydantic_core import to_json

from dive_mcp_host.host.conf import ServerConfig  # noqa: TC001 Pydantic Need this
from dive_mcp_host.host.errors import (
    InvalidMcpServerError,
    McpSessionClosedOrFailedError,
    McpSessionNotInitializedError,
)
from dive_mcp_host.host.helpers.context import ContextProtocol

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream


type ReadStreamType = MemoryObjectReceiveStream[types.JSONRPCMessage | Exception]
type WriteStreamType = MemoryObjectSendStream[types.JSONRPCMessage]
type StreamContextType = AbstractAsyncContextManager[
    tuple[ReadStreamType, WriteStreamType]
]


logger = logging.getLogger(__name__)


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
    KEEP_ALIVE_INTERVAL: float = 60
    RESTART_INTERVAL: float = 3

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
        self._last_active: float = 0

    async def _keep_alive(self, session: ClientSession) -> None:
        """Send keep-alive pings to maintain connection."""
        if not self.config.keep_alive:
            return

        while True:
            if self._client_status != ClientState.RUNNING:
                return
            try:
                await asyncio.sleep(self.config.keep_alive)
                await session.send_ping()
                self._last_active = time.time()
            except asyncio.CancelledError:
                return
            except BaseException as e:
                logger.error("Keep-alive error for %s: %s", self.name, e)
                async with self._cond:
                    self.__change_state(
                        ClientState.RESTARTING, [ClientState.RUNNING], e
                    )
                raise
            await asyncio.sleep(self.KEEP_ALIVE_INTERVAL)

    async def _client_watcher(self) -> None:
        """Client watcher task.

        Restart the client if need.
        Only this watcher can set the client status to RUNNING / FAILED.
        """
        keep_alive_task: asyncio.Task[None] | None = None
        should_break = False
        while True:
            try:
                async with (
                    self._get_client() as streams,
                    ClientSession(*streams) as session,
                ):
                    self._initialize_result = await session.initialize()
                    tool_results = await session.list_tools()
                    self._last_active = time.time()
                    mcp_tools = [
                        McpTool(
                            toolkit_name=self.name,
                            name=tool.name,
                            description=tool.description or "",
                            mcp_server=self,
                            kwargs_arg="kwargs" in tool.inputSchema,
                        )
                        for tool in tool_results.tools
                    ]
                    async with self._cond:
                        self._session = session
                        self._tool_results = tool_results
                        self._mcp_tools = mcp_tools
                        self._exception = None
                        self._retries = 0
                        self.__change_state(ClientState.RUNNING, None, None)

                        # Start keep-alive task if configured
                        keep_alive_task = (
                            asyncio.create_task(self._keep_alive(session))
                            if self.config.keep_alive
                            else None
                        )

                        await self._cond.wait_for(
                            lambda: self._client_status
                            in [ClientState.CLOSED, ClientState.RESTARTING],
                        )
            except* FileNotFoundError:
                self._exception = InvalidMcpServerError(self.name, "Command not found")
                should_break = True
            except* asyncio.CancelledError:
                should_break = True
            except* BaseException as e:  # noqa: BLE001
                self._exception = e

            self._retries += 1
            self._session = None
            if keep_alive_task:
                with suppress(Exception):
                    keep_alive_task.cancel()
                    await keep_alive_task
            if self._client_status == ClientState.CLOSED:
                return
            if self._retries >= self.RETRY_LIMIT or should_break:
                logger.warning(
                    "client %s failed after %d retries %s",
                    self.name,
                    self._retries,
                    self._exception,
                )
                async with self._cond:
                    if self._client_status != ClientState.CLOSED:
                        self.__change_state(ClientState.FAILED, None, False)
                return
            await asyncio.sleep(self.RESTART_INTERVAL)

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
                self._session = None
                self.__change_state(
                    ClientState.CLOSED, [ClientState.INIT, ClientState.RUNNING], False
                )
                await self._cond.wait_for(
                    lambda: self._session_count == 0,
                )
            with suppress(Exception):
                task.cancel()
                await task
                logger.debug("client watcher task cancelled: %s", self.name)

    def _get_client(
        self,
    ) -> StreamContextType:
        """Create a new client.

        Only called by the client watcher.
        """
        if self.config.command:
            env = os.environ.copy()
            env.update(self.config.env)
            if self.config.transport in ("stdio", None):
                return stdio_client(
                    StdioServerParameters(
                        command=self.config.command,
                        args=self.config.args,
                        env=env,
                    ),
                )
            if self.config.transport in ("sse", "websocket") and self.config.url:
                return local_mcp_net_server_client(
                    self.config,
                    self.config.command,
                    self.config.args,
                    env,
                )
            raise InvalidMcpServerError(
                self.name, "Only stdio is supported for command."
            )
        if self.config.url:
            if self.config.transport in ("sse", None):
                return sse_client(
                    url=self.config.url,
                )
            if self.config.transport == "websocket":
                return websocket_client(
                    url=self.config.url,
                )
            raise InvalidMcpServerError(
                self.name, "Only sse and websocket are supported for url."
            )
        raise InvalidMcpServerError(self.name, "No url or command provided.")

    def get_tools(self) -> list[McpTool]:
        """Get the tools."""
        return self._mcp_tools

    def __change_state(
        self,
        new_state: ClientState,
        orig_state: list[ClientState] | None,
        e: BaseException | None | Literal[False],
    ) -> None:
        """Change the client state.

        The caller have to acquire self._cond before calling this function.
        It only notify the condition variable if the state is changed.

        Args:
            new_state: The new state.
            orig_state: The original state.
              Change to new_state if orig_state is None
              or self._client_status == orig_state.
            e: The exception that occurred.
              If e is not False, set self._exception to e.
        """
        if orig_state is None or self._client_status in orig_state:
            if e is not False:
                self._exception = e
            self._client_status = new_state
            logger.debug(
                "client status changed, %s %s %s,",
                self.name,
                new_state,
                e,
            )
            self._cond.notify_all()

    async def _wait_for_session(self) -> ClientSession:
        """Only called by the session context manager."""
        for retried in range(self.RETRY_LIMIT):
            async with self._cond:
                if self._client_status in (ClientState.RESTARTING, ClientState.INIT):
                    logger.warning(
                        "wait for sessioned, %s %s",
                        self.name,
                        self._client_status,
                    )
                    await self._cond.wait_for(
                        lambda: self._client_status
                        in [
                            ClientState.CLOSED,
                            ClientState.RUNNING,
                            ClientState.FAILED,
                        ],
                    )
            if self._client_status in [ClientState.FAILED, ClientState.CLOSED]:
                raise McpSessionClosedOrFailedError(self.name, self._client_status.name)
            now = time.time()
            if (
                self._client_status == ClientState.RUNNING
                and self._session
                and (now - self._last_active > self.KEEP_ALIVE_INTERVAL)
            ):
                # check if the session is still active
                try:
                    async with asyncio.timeout(10):
                        await self._session.send_ping()
                        self._last_active = time.time()
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        "Keep-alive error for %s: %s",
                        self.name,
                        e,
                        extra={
                            "mcp_server": self.name,
                            "client_status": self._client_status,
                        },
                    )
                    async with self._cond:
                        self.__change_state(
                            ClientState.RESTARTING, [ClientState.RUNNING], e
                        )
            if self._client_status == ClientState.RUNNING and self._session:
                return self._session
            if retried < self.RETRY_LIMIT - 1:
                logger.warning(
                    "session not initialized, retrying, %s %s",
                    self.name,
                    self._client_status,
                    extra={
                        "mcp_server": self.name,
                        "client_status": self._client_status,
                    },
                )
                await asyncio.sleep(self.RESTART_INTERVAL)
        logger.error(
            "session not initialized, %s %s",
            self.name,
            self._client_status,
            extra={"mcp_server": self.name, "client_status": self._client_status},
        )
        raise McpSessionNotInitializedError(self.name)

    def session(self) -> AbstractAsyncContextManager[ClientSession]:
        """Get the session.

        Only one session can exist at a time for a McpStdioServer instance.

        Returns:
            The context manager for the session.
        """

        @asynccontextmanager
        async def session_ctx() -> AsyncGenerator[ClientSession, None]:
            """Get the session.

            If the session is inactive for a long time, ping it first
            to check if it is still active.
            """
            session = await self._wait_for_session()
            self._session_count += 1
            try:
                yield session
            # TODO: Not all exceptions require server restart
            # We should handle different exception types appropriately
            # For example, connection errors might need restart
            # while application-level errors might just need to be propagated
            except Exception as e:
                async with self._cond:
                    self._session_count -= 1
                    self.__change_state(
                        ClientState.RESTARTING, [ClientState.RUNNING], e
                    )
                logger.warning(
                    "mcp server %s failed, restarting, %s",
                    self.name,
                    e,
                    extra={
                        "mcp_server": self.name,
                        "client_status": self._client_status,
                    },
                )
                raise
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

    async def wait(self, states: list[ClientState]) -> bool:
        """Wait until the client is in the given state or in the failed or closed state.

        Returns:
            True if the client is in the given state.
        """
        async with self._cond:
            await self._cond.wait_for(
                lambda: self._client_status
                in [
                    *states,
                    ClientState.FAILED,
                    ClientState.CLOSED,
                ],
            )
            return self._client_status in states


class McpTool(BaseTool):
    """A tool for the MCP."""

    toolkit_name: str
    description: str = ""
    mcp_server: McpServer
    kwargs_arg: bool = False

    def _run(self, **kwargs: dict[str, Any]) -> str:
        """Run the tool."""
        return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs: dict[str, Any]) -> str:
        """Run the tool."""
        if not self.kwargs_arg and len(kwargs) == 1 and "kwargs" in kwargs:
            if isinstance(kwargs["kwargs"], str):
                with suppress(JSONDecodeError):
                    kwargs = json_loads(kwargs["kwargs"])
            else:
                kwargs = kwargs["kwargs"]
        async with self.mcp_server.session() as session:
            result = await session.call_tool(self.name, arguments=kwargs)
        content = to_json(result.content).decode()
        if result.isError:
            raise ToolException(content)
        return content


@asynccontextmanager
async def local_mcp_net_server_client(
    config: ServerConfig,
    command: str | None = None,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    max_connection_retries: int = 10,
) -> AsyncGenerator[tuple[ReadStreamType, WriteStreamType], None]:
    """Create a local MCP server client.

    Args:
        config: The configuration of the MCP server.
        command: The command to start the MCP server. if None, use config.command.
        args: The arguments to start the MCP server. if None, use config.args.
        env: The environment variables to start the MCP server. if None, use config.env.
        max_connection_retries: The maximum number of connection creaation.
    """
    command = command or config.command
    args = args or config.args
    env = env or config.env
    assert config.url is not None, "url is required"
    if config.transport == "sse":
        get_client = sse_client
    elif config.transport == "websocket":
        get_client = websocket_client
    if not (
        subprocess := await asyncio.create_subprocess_exec(
            command,
            *args,
            env=env,
        )
    ):
        raise RuntimeError("failed to start subprocess")
    retried = 0
    should_break = False
    # it tooks time to start the server, so we need to retry
    while retried < max_connection_retries and not should_break:
        await asyncio.sleep(0.3 if retried == 0 else 1)
        try:
            with suppress(TimeoutError, httpx.ConnectError):
                async with (
                    get_client(url=config.url) as streams,
                    ClientSession(*streams) as session,
                ):
                    async with asyncio.timeout(10):
                        await session.initialize()
                        break
        except* CancelledError:
            should_break = True
        if should_break:
            break
        retried += 1
    logger.info("connected to the server %s %s", config.name, retried)
    async with get_client(url=config.url) as streams:
        yield streams
    try:
        logger.debug("Terminating subprocess")
        subprocess.terminate()
        await asyncio.wait_for(subprocess.wait(), timeout=10)
    except TimeoutError:
        logger.info("Timeout to terminate mcp-server %s. Kill it.", config.name)
        subprocess.kill()
        await asyncio.wait_for(subprocess.wait(), timeout=10)
