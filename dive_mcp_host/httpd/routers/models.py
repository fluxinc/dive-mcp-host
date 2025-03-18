from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from typing import Literal, TypeVar, TypedDict

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ResultResponse(BaseModel):
    """Generic response model with success status and message."""

    success: bool
    message: str | None


Transport = Literal["command", "sse", "websocket"]


class McpServerConfig(BaseModel):
    """MCP server configuration with transport and connection settings."""

    transport: Transport
    enabled: bool | None
    command: str | None
    args: list[str] | None
    env: dict[str, str] | None
    url: str | None


class McpServers(BaseModel):
    """Collection of MCP server configurations."""

    mcp_servers: dict[str, McpServerConfig] = Field(alias="mcpServers")


class McpServerError(BaseModel):
    """Represents an error from an MCP server."""

    server_name: str = Field(alias="serverName")
    error: object  # any


class ModelConfiguration(BaseModel):
    """Basic configuration for a model, including base URL."""

    base_url: str = Field(alias="baseURL")


class ModelType(StrEnum):
    """Model type."""

    OLLAMA = "ollama"
    MISTRAL = "mistralai"
    BEDROCK = "bedrock"
    DEEPSEEK = "deepseek"
    OTHER = "other"


class ModelSettings(BaseModel):
    """Model settings including provider, API key and parameters."""

    model: str
    model_provider: str = Field(alias="modelProvider")
    api_key: str | None = Field(alias="apiKey")
    configuration: ModelConfiguration | None
    temperature: float | None
    top_p: float | None = Field(alias="topP")
    max_tokens: int | None = Field(alias="maxTokens")

    model_config = ConfigDict(extra="allow")

    @property
    def model_type(self) -> ModelType:
        """Model type."""
        if self.model_provider == "ollama":
            return ModelType.OLLAMA

        if self.model_provider == "mistralai":
            return ModelType.MISTRAL

        if self.model_provider == "bedrock":
            return ModelType.BEDROCK

        if "deepseek" in self.model.lower() or (
            "deepseek" in self.configuration.base_url.lower()
            if self.configuration
            else False
        ):
            return ModelType.DEEPSEEK

        return ModelType.OTHER


class ModelConfig(BaseModel):
    """Overall model configuration including active provider and tool settings."""

    active_provider: str = Field(alias="activeProvider")
    enable_tools: bool = Field(alias="enableTools")
    configs: dict[str, ModelSettings]


class ModelSettingsProperty(BaseModel):
    """Defines a property for model settings with type information and metadata."""

    type: Literal["string", "number"]
    description: str
    required: bool
    default: object | None
    placeholder: object | None


class ModelSettingsDefinition(ModelSettingsProperty):
    """Model settings definition with nested properties."""

    type: Literal["string", "number", "object"]
    properties: dict[str, ModelSettingsProperty] | None


class ModelInterfaceDefinition(BaseModel):
    """Defines the interface for model settings."""

    model_settings: dict[str, ModelSettingsDefinition]


class McpTool(BaseModel):
    """Represents an MCP tool with its properties and metadata."""

    name: str
    tools: list
    description: str
    enabled: bool
    icon: str


class ToolCallsContent(BaseModel):
    """Tool call content."""

    name: str
    arguments: object


class ToolResultContent(BaseModel):
    """Tool result content."""

    name: str
    result: object


class ChatInfoContent(BaseModel):
    """Chat info."""

    id: str
    title: str


class MessageInfoContent(BaseModel):
    """Message info."""

    user_message_id: str = Field(alias="userMessageId")
    assistant_message_id: str = Field(alias="assistantMessageId")


class StreamMessage(BaseModel):
    """Stream message."""

    type: Literal[
        "text", "tool_calls", "tool_result", "error", "chat_info", "message_info"
    ]
    content: (
        str
        | list[ToolCallsContent]
        | ToolResultContent
        | ChatInfoContent
        | MessageInfoContent
    )


class TokenUsage(BaseModel):
    """Token usage."""

    total_input_tokens: int = Field(alias="totalInputTokens")
    total_output_tokens: int = Field(alias="totalOutputTokens")
    total_tokens: int = Field(alias="totalTokens")


class UserInputError(Exception):
    """User input error.

    Args:
        Exception (Exception): The exception.
    """

    def __init__(self, message: str) -> None:
        """Initialize the error.

        Args:
            message (str): The error message.
        """
        self.message = message


class FunctionDefinition(TypedDict):
    """Function definition."""

    name: str


class ToolDefinition(TypedDict):
    """Tool definition."""

    type: Literal["function"]
    function: FunctionDefinition


class McpServerManager(ABC):
    """Abstract base class for MCP server managers."""

    @abstractmethod
    async def get_available_tools(self) -> list[ToolDefinition]:
        """Get available tools."""

    @abstractmethod
    async def get_tool_to_server_map(self) -> Mapping[str, object]:
        """Get tool to server map."""

    @abstractmethod
    async def get_tool_infos(self) -> list[McpTool]:
        """Get tool infos."""


class ModelManager(ABC):
    """Abstract base class for model managers."""

    current_model_settings: ModelSettings | None = None
    enable_tools: bool = True

    @abstractmethod
    async def get_model_config(self) -> ModelConfig:
        """Get model."""

    @abstractmethod
    async def init_model(self) -> BaseChatModel | None:
        """Initialize model."""

    @abstractmethod
    async def save_model_config(
        self, provider: str, model_settings: ModelSettings, enable_tools: bool
    ) -> None:
        """Save model config."""

    @abstractmethod
    async def replace_all_model_config(self, model_config: ModelConfig) -> None:
        """Replace all model config."""

    @abstractmethod
    async def generate_title(self, content: str) -> str:
        """Generate title."""

    @abstractmethod
    def get_model(self) -> BaseChatModel | None:
        """Get model."""

    @abstractmethod
    async def reload_model(self) -> None:
        """Reload model."""
