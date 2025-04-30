import asyncio
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from enum import StrEnum
from logging import INFO, getLogger
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from traceback import format_exception

from pydantic import BaseModel, Field

from dive_mcp_host.host.errors import LogBufferNotFoundError

logger = getLogger(__name__)


class LogEvent(StrEnum):
    """Log event type."""

    STATUS_CHANGE = "status_change"
    STDERR = "stderr"
    SESSION_ERROR = "session_error"

    # extra events for api
    STREAMING_ERROR = "streaming_error"


class LogMsg(BaseModel):
    """Structure of logs."""

    event: LogEvent
    body: str
    mcp_server_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LogBuffer:
    """Log buffer with limited size, supports adding listeners to watch new logs."""

    def __init__(self, size: int = 1000, name: str = "") -> None:
        """Initialize the log buffer."""
        self._size = size
        self._logs: list[LogMsg] = []
        self._listeners: list[Callable[[LogMsg], Coroutine[None, None, None]]] = []
        self._name = name

    @property
    def name(self) -> str:
        """Get the name of the buffer."""
        return self._name

    def _listener_wrapper(
        self,
        listener: Callable[[LogMsg], Coroutine[None, None, None]],
    ) -> Callable[[LogMsg], Coroutine[None, None, None]]:
        """Wrap the listener to handle exceptions."""

        async def _wrapper(msg: LogMsg) -> None:
            try:
                await listener(msg)
            except Exception:
                logger.exception("listener exception")

        return _wrapper

    async def push_session_error(
        self,
        inpt: BaseExceptionGroup | BaseException,
    ) -> None:
        """Push the session error to the log buffer."""
        msg = LogMsg(
            event=LogEvent.SESSION_ERROR,
            body="".join(format_exception(inpt)),
            mcp_server_name=self.name,
        )
        await self.push_log(msg)

    async def push_state_change(
        self,
        inpt: str,
    ) -> None:
        """Push the client status change to the log buffer."""
        msg = LogMsg(
            event=LogEvent.STATUS_CHANGE,
            body=inpt,
            mcp_server_name=self.name,
        )
        await self.push_log(msg)

    async def push_log(self, log: LogMsg) -> None:
        """Add a log to the buffer."""
        self._logs.append(log)
        if len(self._logs) > self._size:
            self._logs.pop(0)

        async with asyncio.TaskGroup() as group:
            for listener in self._listeners:
                group.create_task(listener(log))

    def get_logs(self) -> list[LogMsg]:
        """Get the logs."""
        return self._logs

    @asynccontextmanager
    async def add_listener(
        self, listener: Callable[[LogMsg], Coroutine[None, None, None]]
    ) -> AsyncIterator[None]:
        """Add a listener to the buffer.

        Reads all the logs in the buffer and listens for new logs.
        """
        for i in self._logs:
            await listener(i)
        _listener = self._listener_wrapper(listener)
        self._listeners.append(_listener)
        try:
            yield
        except Exception:
            logger.exception("listener error")
        finally:
            self._listeners.remove(_listener)


class LogProxy:
    """Proxy stderr logs."""

    def __init__(
        self,
        callback: Callable[[LogMsg], Coroutine[None, None, None]],
        mcp_server_name: str,
    ) -> None:
        """Initialize the proxy."""
        self._stderr = sys.stderr
        self._callback = callback
        self._mcp_server_name = mcp_server_name

    async def write(self, s: str) -> None:
        """Write logs."""
        msg = LogMsg(
            event=LogEvent.STDERR,
            body=s,
            mcp_server_name=self._mcp_server_name,
        )
        await self._callback(msg)
        self._stderr.write(s)

    async def flush(self) -> None:
        """Flush the logs."""
        self._stderr.flush()


class _LogFile:
    """Rotating log file by days."""

    def __init__(self, name: str, log_dir: Path, rotation_files: int = 5) -> None:
        self._name = f"{name}.log"
        self._path = log_dir / self._name
        self._logger = getLogger(self._name)
        self._logger.setLevel(INFO)
        self._logger.propagate = False
        handler = TimedRotatingFileHandler(
            self._path,
            when="D",
            interval=1,
            backupCount=rotation_files,
            encoding="utf-8",
        )
        self._logger.addHandler(handler)

    async def __call__(self, log: LogMsg) -> None:
        self._logger.info(log.model_dump_json())


class LogManager:
    """Log Manger for MCP Servers."""

    def __init__(self, log_dir: Path, rotation_files: int = 5) -> None:
        """Initialize the log manager.

        Args:
            log_dir: The directory to store the logs.
            rotation_files: The number of rotation files per mcp server.
        """
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._buffers: dict[str, LogBuffer] = {}
        self._rotation_files = rotation_files

    @asynccontextmanager
    async def register_buffer(self, buffer: LogBuffer) -> AsyncGenerator[None, None]:
        """Register a buffer to the log manager.

        The manager will write log to files.
        """
        self._buffers[buffer.name] = buffer
        log_file = _LogFile(buffer.name, self._log_dir, self._rotation_files)
        async with buffer.add_listener(log_file):
            try:
                yield
            except Exception:
                logger.exception("register buffer error")
            finally:
                self._buffers.pop(buffer.name)

    @asynccontextmanager
    async def listen_log(
        self,
        name: str,
        listener: Callable[[LogMsg], Coroutine[None, None, None]],
    ) -> AsyncGenerator[None, None]:
        """Listen to logs from a specific MCP server."""
        buffer = self._buffers.get(name)
        if buffer is None:
            raise LogBufferNotFoundError(name)
        async with buffer.add_listener(listener):
            yield
