from typing import Annotated, Any, Literal

from pydantic import AnyUrl, BaseModel, Field, UrlConstraints

SpecialProvider = Literal["dive", "__load__"]
"""
special providers:
- dive: use the model in dive_mcp.models
- __load__: load the model from the configuration
"""


class LLMConfig(BaseModel):
    """Configuration for the LLM model."""

    model: str = "gpt-4o"
    provider: str | SpecialProvider = "openai"
    embed: str | None = None
    embed_dims: int = 0
    api_key: str | None = None
    temperature: float = 0
    vector_store: str | None = None

    def model_post_init(self, _: Any) -> None:
        """Set the default embed dimensions for known models."""
        if self.embed and self.embed_dims == 0:
            if self.embed == "text-embedding-3-small":
                self.embed_dims = 1536
            elif self.embed == "text-embedding-3-large":
                self.embed_dims = 3072
            else:
                raise ValueError("invalid dims")


class CheckpointerConfig(BaseModel):
    """Configuration for the checkpointer."""

    # more parameters in the future. like pool size, etc.
    uri: Annotated[
        AnyUrl,
        UrlConstraints(allowed_schemes=["sqlite", "postgresql"]),
    ]


class ServerConfig(BaseModel):
    """Configuration for an MCP server."""

    name: str
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    exclude_tools: list[str] = Field(default_factory=list)
    url: str | None = None

class HostConfig(BaseModel):
    """Configuration for the MCP host."""

    llm: LLMConfig
    checkpointer: CheckpointerConfig | None = None
    mcp_servers: dict[str, ServerConfig]


class AgentConfig(BaseModel):
    """Configuration for an MCP agent."""

    model: str
