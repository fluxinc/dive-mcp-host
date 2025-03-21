from typing import Any, cast
from unittest.mock import MagicMock
import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from dive_mcp_host.host.conf import HostConfig, LLMConfig
from dive_mcp_host.host.host import DiveMcpHost
from dive_mcp_host.models.fake import FakeMessageToolModel, default_responses


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
async def test_callable_system_prompt() -> None:
    """Test that the system prompt can be a callable."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            modelProvider="dive",
        ),
        mcp_servers={},
    )
    msgs = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Line 1!"),
    ]

    mock_system_prompt = MagicMock(return_value=msgs)

    async with (
        DiveMcpHost(config) as mcp_host,
        mcp_host.conversation(
            system_prompt=mock_system_prompt, volatile=True
        ) as conversation,
    ):
        assert mcp_host._model is not None
        model = cast(FakeMessageToolModel, mcp_host._model)
        async for _ in conversation.query(
            msgs,
        ):
            ...
        assert len(model.query_history) == 2
        assert model.query_history[0].content == "You are a helpful assistant."
        assert model.query_history[1].content == "Line 1!"

        assert mock_system_prompt.call_count == 1
        msgs = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Line 2!"),
        ]
        model.query_history = []
        mock_system_prompt.reset_mock()
        mock_system_prompt.return_value = msgs

        async for _ in conversation.query(
            msgs,
        ):
            ...
        assert len(model.query_history) == 2
        assert model.query_history[0].content == "You are a helpful assistant."
        assert model.query_history[1].content == "Line 2!"
        assert mock_system_prompt.call_count == 1
