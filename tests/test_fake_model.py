from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from dive_mcp.host.agents.chat_agent import AgentState
from dive_mcp.host.helpers import today_datetime
from dive_mcp.models import load_model
from dive_mcp.models.fake import FakeMessageToolModel


@pytest.mark.asyncio
async def test_fake_model() -> None:
    """Test the fake model."""

    @tool
    def fake_tool(arg: str) -> str:
        """A fake tool."""
        return arg

    responses = [
        AIMessage(
            content="I am a fake model.",
            tool_calls=[ToolCall(name="fake_tool", args={"arg": "arg"}, id="id")],
        ),
        AIMessage(
            content="final AI message",
        ),
    ]
    model = FakeMessageToolModel(responses=responses)
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant."), ("placeholder", "{messages}")],
    )
    agent_executor = create_react_agent(
        model,
        tools=[fake_tool],
        state_schema=AgentState,
        debug=False,
        prompt=prompt,
    )

    input_messages = AgentState(
        messages=[HumanMessage(content="Hello, world!")],
        is_last_step=False,
        today_datetime=today_datetime(),
        remaining_steps=3,
    )

    def check_results(results: list[dict[str, Any]], msg: str) -> None:
        assert len(results) == 3, msg  # noqa: PLR2004
        assert results[0]["agent"]["messages"][0].content == responses[0].content, msg
        assert (
            results[1]["tools"]["messages"][0].content
            == responses[0].tool_calls[0]["args"]["arg"]
        ), msg
        assert results[2]["agent"]["messages"][0].content == responses[1].content, msg

    check_results(
        [a async for a in agent_executor.astream(input_messages)],
        "astream",
    )
    check_results(list(agent_executor.stream(input_messages)), "stream")


def test_load_model() -> None:
    """Test the load model."""
    responses = [
        AIMessage(content="hello"),
    ]
    model = load_model("dive", "fake", responses=responses)
    assert isinstance(model, FakeMessageToolModel)
    assert model.responses == responses
