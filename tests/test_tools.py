import asyncio
import json
import random
import secrets
import signal
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from os import environ
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest
import pytest_asyncio
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)

from dive_mcp_host.host.conf import HostConfig, LLMConfig
from dive_mcp_host.host.host import DiveMcpHost
from dive_mcp_host.host.tools import (
    ClientState,
    McpServer,
    McpServerInfo,
    ServerConfig,
    ToolManager,
)

if TYPE_CHECKING:
    from dive_mcp_host.models.fake import FakeMessageToolModel


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
            transport="stdio",
        ),
    }


@pytest.fixture
def echo_tool_local_sse_config(
    unused_tcp_port_factory: Callable[[], int],
) -> dict[str, ServerConfig]:
    """Echo Local SSE server configuration."""
    port = unused_tcp_port_factory()
    return {
        "echo": ServerConfig(
            name="echo",
            command="python3",
            args=[
                "-m",
                "dive_mcp_host.host.tools.echo",
                "--transport=sse",
                "--host=localhost",
                f"--port={port}",
            ],
            transport="sse",
            url=f"http://localhost:{port}/sse",
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
    yield (
        port,
        {
            "echo": ServerConfig(
                name="echo", url=f"http://localhost:{port}/sse", transport="sse"
            )
        },
    )
    proc.send_signal(signal.SIGKILL)
    await proc.wait()


@pytest.fixture
def no_such_file_mcp_server() -> dict[str, ServerConfig]:
    """MCP server that does not exist."""
    return {
        "no_such_file": ServerConfig(
            name="no_such_file",
            command="no_such_file",
            transport="stdio",
        ),
        "sse": ServerConfig(
            name="sse_server",
            url="http://localhost:2/sse",
            transport="sse",
        ),
    }


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
        tools = tool_manager.langchain_tools()
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
        tools = tool_manager.langchain_tools()
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
        tools = tool_manager.langchain_tools()
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
    """Test starting the tool manager with a large number of tools."""
    echo_config = echo_tool_stdio_config["echo"]
    more_tools = 10
    for i in range(more_tools):
        echo_tool_stdio_config[f"echo_{i}"] = echo_config.model_copy(
            update={"name": f"echo_{i}"},
        )
    async with ToolManager(echo_tool_stdio_config) as tool_manager:
        tools = tool_manager.langchain_tools()
        assert len(tools) == 2 * (more_tools + 1)


@pytest.mark.asyncio
async def test_mcp_tool_exception_handling(
    echo_tool_stdio_config: dict[str, ServerConfig],
):
    """Test the exception handling of the MCP tool.

    This test verifies that:
    1. When a tool call fails, the exception is properly propagated to the caller
    2. The SSE connection is automatically reconnected after failure
    3. Subsequent tool calls succeed after the connection is restored
    """
    async with McpServer("echo", echo_tool_stdio_config["echo"]) as server:
        server.RESTART_INTERVAL = 0.1
        tools = server.get_tools()
        session = server._session
        with patch("mcp.ClientSession.call_tool") as mocked:
            mocked.side_effect = RuntimeError("test")
            with pytest.raises(RuntimeError, match="test"):
                await tools[0].ainvoke(
                    ToolCall(
                        name=tools[0].name,
                        id="123",
                        args={"xxxx": "Hello, world!"},
                        type="tool_call",
                    ),
                )
            assert mocked.call_count == 1
        assert server._client_status in [
            ClientState.RESTARTING,
            ClientState.RUNNING,
        ]
        await server.wait([ClientState.RUNNING])
        assert server._session != session
        await tools[0].ainvoke(
            ToolCall(
                name=tools[0].name,
                id="123",
                args={"message": "Hello, world!"},
                type="tool_call",
            ),
        )


@pytest.mark.asyncio
async def test_tool_manager_local_sse(
    echo_tool_local_sse_config: dict[str, ServerConfig],
) -> None:
    """Test the tool manager."""
    import logging

    async with ToolManager(echo_tool_local_sse_config) as tool_manager:
        tools = tool_manager.langchain_tools()
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
            logging.info("Tool %s tested", tool.name)


@pytest.mark.asyncio
async def test_host_with_tools(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test the host context initialization."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            modelProvider="dive",
        ),
        mcp_servers=echo_tool_stdio_config,
    )
    async with DiveMcpHost(config) as mcp_host:
        fake_responses = [
            AIMessage(
                content="Call echo tool",
                tool_calls=[
                    ToolCall(
                        name="echo",
                        args={"message": "Hello, world!"},
                        id="123",
                        type="tool_call",
                    ),
                ],
            ),
        ]
        cast("FakeMessageToolModel", mcp_host._model).responses = fake_responses
        async with mcp_host.conversation() as conversation:
            responses = [
                response
                async for response in conversation.query(
                    HumanMessage(content="Hello, world!"),
                    stream_mode=["messages"],
                )
            ]
            assert len(responses) == len(fake_responses) + 1  # plus one tool message
            # need more understanding of the response structure
            tool_message = responses[-1][1][0]  # type: ignore
            assert isinstance(tool_message, ToolMessage)
            assert tool_message.name == "echo"
            assert json.loads(str(tool_message.content))[0]["text"] == "Hello, world!"


@pytest.mark.asyncio
async def test_mcp_server_info(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test the host context initialization."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            modelProvider="dive",
        ),
        mcp_servers=echo_tool_stdio_config,
    )
    import dive_mcp_host.host.tools.echo as echo_tool

    async with DiveMcpHost(config) as mcp_host:
        assert list(mcp_host.mcp_server_info.keys()) == ["echo"]
        assert isinstance(mcp_host.mcp_server_info["echo"], McpServerInfo)
        assert mcp_host.mcp_server_info["echo"].initialize_result is not None
        assert mcp_host.mcp_server_info["echo"].initialize_result.capabilities
        assert (
            mcp_host.mcp_server_info["echo"].initialize_result.instructions
            == echo_tool.Instructions
        )


@pytest.mark.asyncio
async def test_mcp_server_info_no_such_file(
    no_such_file_mcp_server: dict[str, ServerConfig],
) -> None:
    """Test the host context initialization."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            modelProvider="dive",
        ),
        mcp_servers=no_such_file_mcp_server,
    )
    async with DiveMcpHost(config) as mcp_host:
        assert list(mcp_host.mcp_server_info.keys()) == [
            "no_such_file",
            "sse",
        ]
        assert mcp_host.mcp_server_info["no_such_file"] is not None
        assert mcp_host.mcp_server_info["no_such_file"].initialize_result is None
        assert mcp_host.mcp_server_info["no_such_file"].error is not None
        assert (
            mcp_host.mcp_server_info["no_such_file"].client_status == ClientState.FAILED
        )
        assert mcp_host.mcp_server_info["sse"] is not None
        assert mcp_host.mcp_server_info["sse"].initialize_result is None
        assert mcp_host.mcp_server_info["sse"].error is not None
        assert mcp_host.mcp_server_info["sse"].client_status == ClientState.FAILED


@pytest.mark.asyncio
async def test_host_ollama(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test the host context initialization."""
    if (base_url := environ.get("OLLAMA_URL")) and (
        olama_model := environ.get("OLLAMA_MODEL")
    ):
        # quen2.5:14b
        config = HostConfig(
            llm=LLMConfig(
                model=olama_model,
                modelProvider="ollama",
                configuration={"base_url": base_url},
            ),
            mcp_servers=echo_tool_stdio_config,
        )
    else:
        pytest.skip(
            "need environment variable OLLAMA_URL and OLLAMA_MODEL to run this test"
        )

    async with (
        DiveMcpHost(config) as mcp_host,
        mcp_host.conversation(
            # system_prompt=system_prompt(""),
        ) as conversation,
    ):
        # r = await conversation.invoke("test mcp tool echo with 'hello'")
        async for response in conversation.query(
            HumanMessage(content="test mcp tool to echo message 'helloXXX'."),
            stream_mode=["updates"],
        ):
            response = cast("tuple[str, dict[str, dict[str, BaseMessage]]]", response)
            if msg_dict := response[1].get("tools"):
                contents = list[str]()
                for msg in msg_dict.get("messages", []):
                    if isinstance(msg, ToolMessage):
                        # XXX the content type is complex.
                        if isinstance(msg.content, str):
                            rep = json.loads(msg.content)
                        else:
                            rep = msg.content
                        for r in rep:
                            assert r["type"] == "text"  # type: ignore[index]
                            contents.append(r["text"])  # type: ignore[index]
                assert any("helloXXX" in c for c in contents)


@pytest.mark.asyncio
async def test_tool_kwargs(
    echo_tool_stdio_config: dict[str, ServerConfig],
) -> None:
    """Some LLM set the tool call argument in kwargs."""
    async with ToolManager(echo_tool_stdio_config) as tool_manager:
        tools = tool_manager.langchain_tools()
        assert sorted([i.name for i in tools]) == ["echo", "ignore"]
        for tool in tools:
            result = await tool.ainvoke(
                ToolCall(
                    name=tool.name,
                    id="123",
                    args={"kwargs": {"message": "Hello, world!"}},
                    type="tool_call",
                ),
            )
            assert isinstance(result, ToolMessage)
            if tool.name == "echo":
                assert json.loads(str(result.content))[0]["text"] == "Hello, world!"
            else:
                assert json.loads(str(result.content)) == []

        for tool in tools:
            result = await tool.ainvoke(
                ToolCall(
                    name=tool.name,
                    id="123",
                    args={"kwargs": """{"message": "Hello, world!"}"""},
                    type="tool_call",
                ),
            )
            assert isinstance(result, ToolMessage)
            if tool.name == "echo":
                assert json.loads(str(result.content))[0]["text"] == "Hello, world!"
            else:
                assert json.loads(str(result.content)) == []
