from collections.abc import Sequence
from typing import Protocol, Self

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver, V
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore

from dive_mcp.host.prompt import PromptType


# XXX is there any better way to do this?
class AgentFactory[T](Protocol):
    """A factory for creating agents.

    Implementing this protocol to create your own custom agent.
    Pass the factory to the host to create an agent for the conversation.
    """

    @classmethod
    def create_factory(
        cls,
        model: BaseChatModel,
        tools: Sequence[BaseTool] | ToolNode,
    ) -> Self:
        """Create an agent factory."""
        ...

    def create_agent(
        self,
        *,
        prompt: PromptType | ChatPromptTemplate,
        checkpointer: BaseCheckpointSaver[V] | None = None,
        store: BaseStore | None = None,
        debug: bool = False,
    ) -> CompiledGraph:
        """Create an agent."""
        ...

    def create_config(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> RunnableConfig | None:
        """Create a config for the agent.

        Override this to customize the config for the agent.
        The default implementation returns this config:
        {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
            },
            "recursion_limit": 100,
        }
        """
        return {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
            },
            "recursion_limit": 100,
        }

    def create_initial_state(
        self,
        *,
        query: str | HumanMessage,
    ) -> T:
        """Create an initial state for the query."""
        ...

    def state_type(
        self,
    ) -> type[T]:
        """Create an agent factory."""
        ...

    def create_prompt(
        self,
        *,
        system_prompt: str,
    ) -> ChatPromptTemplate:
        """Create a prompt for the agent.

        Override this to customize the prompt for the agent.
        The default implementation returns a prompt with a placeholder for the messages.
        """
        return ChatPromptTemplate.from_messages(  # type: ignore[arg-type]
            [
                ("system", system_prompt),
                ("placeholder", self.placeholder()),
            ],
        )

    def placeholder(
        self,
    ) -> str:
        """Return a placeholder of prompt for the agent.

        The default implementation returns "{messages}".
        """
        return "{messages}"


def initial_messages(
    query: str | HumanMessage,
) -> list[HumanMessage]:
    """Create an initial message for your state.

    Args:
        query: The query to create the initial message from.

    Returns:
        A list of HumanMessage objects.
    """
    return [HumanMessage(content=query)] if isinstance(query, str) else [query]
