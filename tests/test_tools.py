import asyncio
import json
import random
import secrets
import signal
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import pytest
import pytest_asyncio
from langchain_core.messages import ToolCall, ToolMessage

from dive_mcp_host.host.tools import ServerConfig, ToolManager


@pytest.fixture
def echo_tool_stdio_config() -> dict[str, ServerConfig]:  # noqa: D103
    return {
        "echo": ServerConfig(
            name="echo",
            command="python3",
            args=[
                "-m",
                "dive_mcp_host.host.tools.echo",
                "--transport=stdio",
            ],
        ),
    }


@pytest_asyncio.fixture
@asynccontextmanager
async def echo_tool_sse_server(
    unused_tcp_port_factory: Callable[[], int],
) -> AsyncGenerator[tuple[int, dict[str, ServerConfig]], None]:
    """Start the echo tool SSE server."""
    port = unused_tcp_port_factory()
    proc = await asyncio.create_subprocess_exec(
        "python3",
        "-m",
        "dive_mcp_host.host.tools.echo",
        "--transport=sse",
        "--host=localhost",
        f"--port={port}",
    )
    yield port, {"echo": ServerConfig(name="echo", url=f"http://localhost:{port}/sse")}
    proc.send_signal(signal.SIGKILL)
    await proc.wait()


@pytest.mark.asyncio
async def test_tool_manager_sse(
    echo_tool_sse_server: AbstractAsyncContextManager[
        tuple[int, dict[str, ServerConfig]]
    ],
) -> None:
    """Test the tool manager."""
    async with (
        echo_tool_sse_server as (port, configs),
        ToolManager(configs) as tool_manager,
    ):
        tools = tool_manager.tools()
        assert sorted([i.name for i in tools]) == ["echo", "ignore"]
        for tool in tools:
            result = await tool.ainvoke(
                ToolCall(
                    name=tool.name,
                    id="123",
                    args={"message": "Hello, world!"},
                    type="tool_call",
                ),
            )
            assert isinstance(result, ToolMessage)
            if tool.name == "echo":
                assert json.loads(str(result.content))[0]["text"] == "Hello, world!"
            else:
                assert json.loads(str(result.content)) == []


@pytest.mark.asyncio
async def test_tool_manager_stdio(
    echo_tool_stdio_config: dict[str, ServerConfig],
) -> None:
    """Test the tool manager."""
    async with ToolManager(echo_tool_stdio_config) as tool_manager:
        tools = tool_manager.tools()
        assert sorted([i.name for i in tools]) == ["echo", "ignore"]
        for tool in tools:
            result = await tool.ainvoke(
                ToolCall(
                    name=tool.name,
                    id="123",
                    args={"message": "Hello, world!"},
                    type="tool_call",
                ),
            )
            assert isinstance(result, ToolMessage)
            if tool.name == "echo":
                assert json.loads(str(result.content))[0]["text"] == "Hello, world!"
            else:
                assert json.loads(str(result.content)) == []


@pytest.mark.asyncio
async def test_stdio_parallel(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test that stdio tools can execute in parallel.

    This test is to ensure that the tool manager can handle multiple requests
    simultaneously and respond correctly.
    """
    async with ToolManager(echo_tool_stdio_config) as tool_manager:
        tools = tool_manager.tools()
        echo_tool = None
        ignore_tool = None
        for tool in tools:
            if tool.name == "echo":
                echo_tool = tool
            elif tool.name == "ignore":
                ignore_tool = tool
        assert echo_tool is not None
        assert ignore_tool is not None

        random_message = secrets.token_hex(2048)

        async def test_echo():
            return await echo_tool.ainvoke(
                ToolCall(
                    name=echo_tool.name,
                    id=str(random.randint(1, 1000000)),  # noqa: S311
                    args={"message": random_message},
                    type="tool_call",
                ),
            )

        async def test_ignore():
            return await ignore_tool.ainvoke(
                ToolCall(
                    name=ignore_tool.name,
                    id=str(random.randint(1, 1000000)),  # noqa: S311
                    args={"message": random_message},
                    type="tool_call",
                ),
            )

        n_tasks = 30
        async with asyncio.TaskGroup() as tg:
            echo_tasks = [tg.create_task(test_echo()) for _ in range(n_tasks)]
            ignore_tasks = [tg.create_task(test_ignore()) for _ in range(n_tasks)]
        echo_results = await asyncio.gather(*echo_tasks)
        ignore_results = await asyncio.gather(*ignore_tasks)
        assert len(echo_results) == n_tasks
        assert len(ignore_results) == n_tasks
        for result in echo_results:
            assert json.loads(str(result.content))[0]["text"] == random_message
        for result in ignore_results:
            assert json.loads(str(result.content)) == []

@pytest.mark.asyncio
async def test_tool_manager_massive_tools(
    echo_tool_stdio_config: dict[str, ServerConfig],
) -> None:
    """Test the tool manager."""
    echo_config = echo_tool_stdio_config["echo"]
    more_tools = 10
    for i in range(more_tools):
        echo_tool_stdio_config[f"echo_{i}"] = echo_config.model_copy(
            update={"name": f"echo_{i}"},
        )
    async with ToolManager(echo_tool_stdio_config) as tool_manager:
        tools = tool_manager.tools()
        assert len(tools) == 2 * (more_tools + 1)
