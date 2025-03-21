import pytest
from langchain_core.messages import HumanMessage
from pydantic import AnyUrl

from dive_mcp_host.host.conf import CheckpointerConfig, HostConfig, LLMConfig
from dive_mcp_host.host.host import DiveMcpHost
from dive_mcp_host.models.fake import default_responses
from tests.helper import SQLITE_URI


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
async def test_get_messages() -> None:
    """Test the get_messages."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            modelProvider="dive",
        ),
        mcp_servers={},
        checkpointer=CheckpointerConfig(
            uri= AnyUrl(SQLITE_URI)
        ),
    )

    async with DiveMcpHost(config) as mcp_host:
        conversation = mcp_host.conversation()
        async with conversation:
            query_message = "Hello, world!"
            async for _ in conversation.query(query_message, stream_mode=None):
                pass

            thread_id = conversation.thread_id
            messages = await mcp_host.get_messages(thread_id)
            assert len(messages) > 0

            human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
            assert any(msg.content == query_message for msg in human_messages)

            empty_messages = await mcp_host.get_messages("non_existent_thread_id")
            assert len(empty_messages) == 0
