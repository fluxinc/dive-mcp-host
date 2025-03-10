import pytest

from dive_mcp.host.conf import HostConfig, LLMConfig
from dive_mcp.host.host import DiveMcpHost
from dive_mcp.models.fake import default_responses


@pytest.mark.asyncio
async def test_host_context() -> None:
    """Test the host context initialization."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            provider="dive",
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
            # assert len(responses) == len(espect_responses)
            for res, expect in zip(responses, espect_responses, strict=True):
                assert res.content == expect.content
