"""This module contains the ChatAgentFactory for the host.

It uses langgraph.prebuilt.create_react_agent to create the agent.
"""

from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver, V
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt import create_react_agent  # type: ignore[arg-type]
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore

from dive_mcp.host.agents.agent_factory import AgentFactory, initial_messages
from dive_mcp.host.helpers import today_datetime
from dive_mcp.host.prompt import PromptType


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_last_step: IsLastStep
    today_datetime: str
    remaining_steps: int


class ChatAgentFactory(AgentFactory[AgentState]):
    """A factory for ChatAgents."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: Sequence[BaseTool] | ToolNode,
    ) -> None:
        """Initialize the chat agent factory."""
        self._model = model
        self._tools = tools

    def create_agent(
        self,
        *,
        prompt: PromptType | ChatPromptTemplate,
        checkpointer: BaseCheckpointSaver[V] | None = None,
        store: BaseStore | None = None,
        debug: bool = False,
    ) -> CompiledGraph:
        """Create a react agent."""
        return create_react_agent(
            self._model,
            self._tools,
            prompt=prompt,
            checkpointer=checkpointer,
            store=store,
            debug=debug,
        )

    def create_initial_state(
        self,
        *,
        query: str | HumanMessage,
    ) -> AgentState:
        """Create an initial state for the query."""
        return AgentState(
            messages=initial_messages(query),
            is_last_step=False,
            today_datetime=today_datetime(),
            remaining_steps=1,
        )

    def state_type(
        self,
    ) -> type[AgentState]:
        """Get the state type."""
        return AgentState


def get_chat_agent_factory(
    model: BaseChatModel,
    tools: Sequence[BaseTool] | ToolNode,
) -> ChatAgentFactory:
    """Get an agent factory."""
    return ChatAgentFactory(model, tools)
