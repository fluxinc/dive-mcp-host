"""This module contains the agents for the host.

Refer to the design of ChatAgentFactory to implement other agents.
"""

from dive_mcp_host.host.agents.agent_factory import AgentFactory, V
from dive_mcp_host.host.agents.chat_agent import (
    ChatAgentFactory,
    get_chat_agent_factory,
)

__all__ = [
    "AgentFactory",
    "ChatAgentFactory",
    "V",
    "get_chat_agent_factory",
]
