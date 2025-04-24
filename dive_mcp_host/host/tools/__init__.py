"""Model for the MCP servers."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from enum import Enum, auto
from itertools import chain
from json import JSONDecodeError
from json import loads as json_loads
from typing import TYPE_CHECKING, Any, Literal, Self

import anyio
import httpx
from langchain_core.tools import BaseTool, ToolException
from mcp import ClientSession, McpError, StdioServerParameters, types
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
from dive_mcp_host.host.tools.stdio_server import stdio_client

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Mapping

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
        self._mcp_servers_task = dict[str, tuple[asyncio.Task, asyncio.Event]]()
        self._lock = asyncio.Lock()
        self._initialized_event = asyncio.Event()

        self._mcp_servers = {
            name: McpServer(name=name, config=config)
            for name, config in self._configs.items()
        }

    def langchain_tools(
        self,
        tool_filter: Callable[[McpServer], bool] = lambda _: True,
    ) -> list[McpTool]:
        """Get the langchain tools for the MCP servers."""
        return list(
            chain.from_iterable(
                [i.mcp_tools for i in self._mcp_servers.values() if tool_filter(i)],
            ),
        )

    async def _launch_tools(self, servers: Mapping[str, McpServer]) -> None:
        async def tool_process(
            server: McpServer, exit_signal: asyncio.Event, ready: asyncio.Event
        ) -> None:
            async with server:
                ready.set()
                await exit_signal.wait()
            logger.debug("Tool process %s exited", server.name)

        async def _launch_task(name: str, server: McpServer) -> None:
            event = asyncio.Event()
            ready = asyncio.Event()
            task = asyncio.create_task(tool_process(server, event, ready))
            await ready.wait()
            self._mcp_servers_task[name] = (task, event)

        async with self._lock, asyncio.TaskGroup() as tg:
            for name, server in servers.items():
                tg.create_task(_launch_task(name, server))

        self._initialized_event.set()

    async def _shutdown_tools(self, servers: Iterable[str]) -> None:
        async def _shutdown_task(name: str) -> None:
            task, event = self._mcp_servers_task.pop(name, (None, None))
            if not (task and event):
                logger.warning(
                    "task or event not found for %s. %s %s", name, task, event
                )
                return
            event.set()
            logger.debug("ToolManager shutting down %s", name)
            await task
            del self._mcp_servers[name]

        async with self._lock, asyncio.TaskGroup() as tg:
            for name in servers:
                tg.create_task(_shutdown_task(name))
        logger.debug("ToolManager shutdown complete")

    async def reload(
        self, new_configs: dict[str, ServerConfig], force: bool = False
    ) -> None:
        """Reload the MCP servers.

        Args:
            new_configs: The new MCP server configurations.
            force: If True, reload all MCP servers even if they are not changed.
        """
        logger.debug("Reloading MCP servers, force: %s", force)

        if not force:
            to_shutdown = set(self._configs.keys()) - set(new_configs.keys())
            to_launch = set(new_configs.keys()) - set(self._configs.keys())

            # check if the config has changed
            for key in set(self._configs) - to_shutdown:
                if self._configs[key] != new_configs[key]:
                    to_shutdown.add(key)
                    to_launch.add(key)
        else:
            to_shutdown = set(self._configs.keys())
            to_launch = set(new_configs.keys())

        self._configs = new_configs

        await self._shutdown_tools(to_shutdown)

        launch_servers = {}
        for l_key in to_launch:
            new_server = McpServer(name=l_key, config=new_configs[l_key])
            launch_servers[l_key] = new_server
            self._mcp_servers[l_key] = new_server
        await self._launch_tools(launch_servers)

    async def _run_in_context(self) -> AsyncGenerator[Self, None]:
        """Get the langchain tools for the MCP servers."""
        # we can manipulate the stack to add or remove tools
        launch_tools_task = asyncio.create_task(
            self._launch_tools(self._mcp_servers),
            name="init-launch-tools",
        )
        try:
            yield self
        finally:
            await self._shutdown_tools(list(self._mcp_servers.keys()))
            launch_tools_task.cancel()
            await launch_tools_task

    @property
    def mcp_server_info(self) -> dict[str, McpServerInfo]:
        """Get the MCP server capabilities and tools.

        Returns:
            A dictionary of MCP server name to server info.
            If the mcp server is not initialized, the value will be `None`.
        """
        return {name: i.server_info for name, i in self._mcp_servers.items()}

    @property
    def initialized_event(self) -> asyncio.Event:
        """Get the initialization event.

        Only useful on initial startup, not when reloading.
        """
        return self._initialized_event


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
                logger.debug(
                    "Keep-alive stopped for %s: client status is %s",
                    self.name,
                    self._client_status,
                )
                return
            try:
                await asyncio.sleep(self.config.keep_alive)
                await session.send_ping()
                self._last_active = time.time()
            except asyncio.CancelledError:
                logger.debug("Keep-alive cancelled for %s", self.name)
                return
            except BaseException as e:
                logger.error("Keep-alive error for %s: %s", self.name, e)
                async with self._cond:
                    self.__change_state(
                        ClientState.RESTARTING, [ClientState.RUNNING], e
                    )
                raise
            await asyncio.sleep(self.KEEP_ALIVE_INTERVAL)

    async def _client_watcher(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """Client watcher task.

        Restart the client if need.
        Only this watcher can set the client status to RUNNING / FAILED.
        """
        keep_alive_task: asyncio.Task[None] | None = None
        should_break = False
        while True:
            try:
                logger.debug("Attempting to initialize client %s", self.name)
                async with (
                    self._get_client() as streams,
                    ClientSession(*streams) as session,
                ):
                    async with asyncio.timeout(10):
                        # When using stdio, the initialize call may block indefinitely
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
                            args_schema=tool.inputSchema,
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
                        logger.debug(
                            "Client %s initialized successfully with %d tools",
                            self.name,
                            len(mcp_tools),
                        )

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
                        logger.debug(
                            "client watcher %s exited. status: %s",
                            self.name,
                            self._client_status,
                        )
            # cannot launch local sse server
            except* InvalidMcpServerError as e:
                self._exception = e
                should_break = True
            except* ProcessLookupError as eg:
                # this raised when a stdio process is exited
                # and the initialize call is timeout
                logger.warning(
                    "ProcessLookupError for %s: %s",
                    self.name,
                    eg.exceptions,
                )
                self._exception = InvalidMcpServerError(self.name, str(eg.exceptions))
                should_break = True
            except* (
                FileNotFoundError,
                PermissionError,
                McpError,
                httpx.ConnectError,
                httpx.InvalidURL,
                httpx.TooManyRedirects,
            ) as eg:
                logger.exception(
                    "Client initialization error for %s: %s",
                    self.name,
                    eg.exceptions,
                )
                self._exception = InvalidMcpServerError(self.name, str(eg.exceptions))
                should_break = True
            except* httpx.HTTPStatusError as eg:
                for e in eg.exceptions:
                    if (
                        isinstance(e, httpx.HTTPStatusError)
                        and e.response.status_code < 500  # noqa: PLR2004
                        and e.response.status_code != 429  # noqa: PLR2004
                    ):
                        should_break = True
                        break
                self._exception = InvalidMcpServerError(self.name, str(eg.exceptions))
            except* asyncio.CancelledError as e:
                should_break = True
                logger.debug("Client watcher cancelled for %s", self.name)
            except* BaseException as eg:
                self._exception = InvalidMcpServerError(self.name, str(eg.exceptions))
                logger.exception(
                    "Client initialization error for %s: %s",
                    self.name,
                    eg.exceptions,
                )

            self._retries += 1
            self._session = None
            if keep_alive_task:
                with suppress(Exception):
                    keep_alive_task.cancel()
                    await keep_alive_task
                    logger.debug("Keep-alive task awaited for %s", self.name)
            if self._client_status == ClientState.CLOSED:
                logger.info("Client %s closed, stopping watcher", self.name)
                return
            if self._retries >= self.RETRY_LIMIT or should_break:
                logger.warning(
                    "client for [%s] failed after %d retries %s",
                    self.name,
                    self._retries,
                    self._exception,
                )
                async with self._cond:
                    if self._client_status != ClientState.CLOSED:
                        self.__change_state(ClientState.FAILED, None, False)
                return
            logger.debug(
                "Retrying client initialization for %s (attempt %d/%d)",
                self.name,
                self._retries,
                self.RETRY_LIMIT,
            )
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
            logger.debug("mcp server shutting down %s", self.name)
            async with self._cond:
                self._session = None
                self.__change_state(
                    ClientState.CLOSED, [ClientState.INIT, ClientState.RUNNING], False
                )
                logger.debug(
                    "%s: wait all sessions to be closed. now is %s",
                    self.name,
                    self._session_count,
                )
            async with self._cond, asyncio.timeout(30):
                try:
                    await self._cond.wait_for(
                        lambda: self._session_count == 0,
                    )
                except TimeoutError:
                    logger.warning(
                        "Timeout to wait %d sessions to be closed",
                        self._session_count,
                    )
            with suppress(Exception):
                async with asyncio.timeout(10):
                    task.cancel()
                    await task
                logger.debug("MCP server %s exited", self.name)

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

    @property
    def mcp_tools(self) -> list[McpTool]:
        """Get the tools."""
        if self._client_status == ClientState.RUNNING:
            return self._mcp_tools
        return []

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
                    logger.debug(
                        "Waiting for session initialization, %s status: %s",
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
                logger.error(
                    "Session failed or closed for %s: %s",
                    self.name,
                    self._client_status,
                )
                raise McpSessionClosedOrFailedError(self.name, self._client_status.name)
            now = time.time()
            if (
                self._client_status == ClientState.RUNNING
                and self._session
                and (now - self._last_active > self.KEEP_ALIVE_INTERVAL)
            ):
                # check if the session is still active
                try:
                    logger.debug(
                        "Checking session health for %s (inactive for %.1f seconds)",
                        self.name,
                        now - self._last_active,
                    )
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
                    "Session not initialized, retrying, %s status: %s (attempt %d/%d)",
                    self.name,
                    self._client_status,
                    retried + 1,
                    self.RETRY_LIMIT,
                    extra={
                        "mcp_server": self.name,
                        "client_status": self._client_status,
                    },
                )
                await asyncio.sleep(self.RESTART_INTERVAL)
        logger.error(
            "Session not initialized after %d attempts, %s status: %s",
            self.RETRY_LIMIT,
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
            except (ToolException, McpError) as e:
                logger.error("Tool exception for %s: %s", self.name, e)
                raise
            except (
                httpx.HTTPError,
                httpx.StreamError,
                httpx.TimeoutException,
                httpx.TooManyRedirects,
                anyio.BrokenResourceError,
                anyio.ClosedResourceError,
                anyio.EndOfStream,
                Exception,  # Before we know the exception type
            ) as e:
                async with self._cond:
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
            finally:
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
        logger.debug(
            "Executing tool %s.%s with args: %s", self.toolkit_name, self.name, kwargs
        )
        async with self.mcp_server.session() as session:
            result = await session.call_tool(self.name, arguments=kwargs)
        
        # Get the raw content as JSON string
        content_json = to_json(result.content).decode()
        
        # Filter out sources if this is a query tool result
        # if self.name == "query":
        #     try:
        #         content_dict = json_loads(content_json)
        #         # Handle different result formats
        #         if isinstance(content_dict, dict) and isinstance(content_dict.get("content"), list):
        #             # Standard format: {content: [{type: 'text', text: '...'}]}
        #             # Filter out source info before sending to LLM
        #             content_dict["content"] = [
        #                 item for item in content_dict["content"]
        #                 if not (isinstance(item, dict) and
        #                         item.get("type") == "text" and
        #                         item.get("text") and
        #                         isinstance(item.get("text"), str) and
        #                         item["text"].startswith("<SOURCES>"))
        #             ]
        #             content_json = to_json(content_dict).decode()
        #         elif isinstance(content_dict, list):
        #             # Direct list format: [{type: 'text', text: '...'}]
        #             filtered_content = [
        #                 item for item in content_dict
        #                 if not (isinstance(item, dict) and
        #                         item.get("type") == "text" and
        #                         item.get("text") and
        #                         isinstance(item.get("text"), str) and
        #                         item["text"].startswith("<SOURCES>"))
        #             ]
        #             content_json = to_json(filtered_content).decode()
        #     except (JSONDecodeError, Exception) as e:
        #         logger.warning(f"Error filtering sources from query result: {e}")
        
        if result.isError:
            logger.error(
                "Tool execution failed for %s.%s: %s",
                self.toolkit_name,
                self.name,
                content_json,
            )
        logger.debug("Tool %s.%s executed successfully", self.toolkit_name, self.name)
        
        return content_json


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
    get_client = sse_client if config.transport == "sse" else websocket_client
    logger.debug("Starting local MCP server %s with command: %s", config.name, command)
    if not (
        subprocess := await asyncio.create_subprocess_exec(
            command,
            *args,
            env=env,
        )
    ):
        logger.error("Failed to start subprocess for %s", config.name)
        raise RuntimeError("failed to start subprocess")
    retried = 0
    # it tooks time to start the server, so we need to retry
    try:
        while retried < max_connection_retries:
            await asyncio.sleep(0.3 if retried == 0 else 1)
            logger.debug(
                "Attempting to connect to server %s (attempt %d/%d)",
                config.name,
                retried + 1,
                max_connection_retries,
            )
            with suppress(TimeoutError, httpx.HTTPError):
                async with (
                    get_client(url=config.url) as streams,
                    ClientSession(*streams) as session,
                ):
                    async with asyncio.timeout(10):
                        await session.initialize()
                        logger.info("Successfully connected to server %s", config.name)
                        break
            retried += 1
        else:
            raise InvalidMcpServerError(config.name)
        logger.info(
            "Connected to the server %s after %d attempts", config.name, retried
        )
        async with get_client(url=config.url) as streams:
            yield streams
    finally:
        with suppress(TimeoutError):
            logger.debug("Terminating subprocess for %s", config.name)
            subprocess.terminate()
            subprocess.send_signal(signal.SIGINT)
            await asyncio.wait_for(subprocess.wait(), timeout=10)
            subprocess = None
        if subprocess:
            logger.info("Timeout to terminate mcp-server %s. Kill it.", config.name)
            with suppress(TimeoutError):
                subprocess.kill()
                await asyncio.wait_for(subprocess.wait(), timeout=10)
