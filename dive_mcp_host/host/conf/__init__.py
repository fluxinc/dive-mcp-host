from pathlib import Path
from typing import Annotated, Literal

from pydantic import AnyUrl, BaseModel, Field, SecretStr, UrlConstraints

from dive_mcp_host.host.conf.llm import LLMConfigTypes


class CheckpointerConfig(BaseModel):
    """Configuration for the checkpointer."""

    # more parameters in the future. like pool size, etc.
    uri: Annotated[
        AnyUrl,
        UrlConstraints(allowed_schemes=["sqlite", "postgres", "postgresql"]),
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
    keep_alive: float | None = None
    transport: Literal["stdio", "sse", "websocket"]
    headers: dict[str, SecretStr] = Field(default_factory=dict)


class LogConfig(BaseModel):
    """Config for mcp server logs.

    Attributes:
        log_dir: base directory for log files.
        rotation_files: max log rotation files per mcp server.
        buffer_length: the amount of log entries in log buffer.
    """

    log_dir: Path = Field(default_factory=lambda: Path.cwd() / "logs")
    rotation_files: int = 5
    buffer_length: int = 1000


class HostConfig(BaseModel):
    """Configuration for the MCP host."""

    llm: LLMConfigTypes
    checkpointer: CheckpointerConfig | None = None
    mcp_servers: dict[str, ServerConfig]
    log_config: LogConfig = Field(default_factory=LogConfig)


class AgentConfig(BaseModel):
    """Configuration for an MCP agent."""

    model: str
