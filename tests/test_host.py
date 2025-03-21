import json
from typing import cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from pydantic import AnyUrl

from dive_mcp_host.host.conf import CheckpointerConfig, HostConfig, LLMConfig
from dive_mcp_host.host.host import DiveMcpHost
from dive_mcp_host.host.tools import ServerConfig
from dive_mcp_host.models.fake import FakeMessageToolModel, default_responses
from tests.helper import SQLITE_URI


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


@pytest.mark.asyncio
async def test_host_context() -> None:
    """Test the host context initialization."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            modelProvider="dive",
        ),
        mcp_servers={},
    )
    espect_responses = default_responses()
    # prompt = ChatPromptTemplate.from_messages(
    #     [("system", "You are a helpful assistant."), ("placeholder", "{messages}")],
    # )
    async with DiveMcpHost(config) as mcp_host:
        conversation = mcp_host.conversation()
        async with conversation:
            responses = [
                response["agent"]["messages"][0]
                async for response in conversation.query(
                    "Hello, world!",
                    stream_mode=None,
                )
            ]
            for res, expect in zip(responses, espect_responses, strict=True):
                assert res.content == expect.content  # type: ignore[attr-defined]
        conversation = mcp_host.conversation()
        async with conversation:
            responses = [
                response["agent"]["messages"][0]
                async for response in conversation.query(
                    HumanMessage(content="Hello, world!"),
                    stream_mode=None,
                )
            ]
            for res, expect in zip(responses, espect_responses, strict=True):
                assert res.content == expect.content  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_query_two_messages() -> None:
    """Test that the query method can handle two or more messages."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            modelProvider="dive",
        ),
        mcp_servers={},
    )
    async with DiveMcpHost(config) as mcp_host, mcp_host.conversation() as conversation:
        responses = [
            response
            async for response in conversation.query(
                [
                    HumanMessage(content="Attachment"),
                    HumanMessage(content="Hello, world!"),
                ],
                stream_mode=["values"],
            )
        ]
        for i in responses:
            human_messages = [
                i
                for i in i[1]["messages"]  # type: ignore[index]
                if isinstance(i, HumanMessage)
            ]
            assert len(human_messages) == 2
            assert human_messages[0].content == "Attachment"
            assert human_messages[1].content == "Hello, world!"


@pytest.mark.asyncio
async def test_get_messages(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test the get_messages."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            modelProvider="dive",
        ),
        mcp_servers=echo_tool_stdio_config,
        checkpointer=CheckpointerConfig(uri=AnyUrl(SQLITE_URI)),
    )

    async with DiveMcpHost(config) as mcp_host:
        fake_responses = [
            AIMessage(
                content="Call echo tool",
                tool_calls=[
                    ToolCall(
                        name="echo",
                        args={"message": "Hello, world! 許個願望吧"},
                        id="123",
                        type="tool_call",
                    ),
                ],
            ),
        ]
        cast("FakeMessageToolModel", mcp_host._model).responses = fake_responses  # noqa: SLF001
        conversation = mcp_host.conversation()
        async with conversation:
            async for _ in conversation.query(
                HumanMessage(content="Hello, world! 許個願望吧"),
                stream_mode=["messages"],
            ):
                pass

            thread_id = conversation.thread_id
            messages = await mcp_host.get_messages(thread_id)
            assert len(messages) > 0

            human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
            assert any(msg.content == "Hello, world! 許個願望吧" for msg in human_messages)

            ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
            assert any(msg.content == "Call echo tool" for msg in ai_messages)

            tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
            assert tool_messages[0].name == "echo"
            assert (
                json.loads(str(tool_messages[0].content))[0]["text"]
                == "Hello, world! 許個願望吧"
            )

            empty_messages = await mcp_host.get_messages("non_existent_thread_id")
            assert len(empty_messages) == 0
