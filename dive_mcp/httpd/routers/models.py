from datetime import datetime
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ResultResponse(BaseModel):
    """Generic response model with success status and message."""

    success: bool
    message: str | None


class Chat(BaseModel):
    id: str
    title: str
    created_at: datetime = Field(alias="createdAt")


class Message(BaseModel):
    id: int
    create_at: datetime = Field(alias="createdAt")
    content: str
    role: str
    chat_id: str = Field(alias="chatId")
    message_id: str = Field(alias="messageId")
    files: object  # TODO: define files


class ChatMessage(BaseModel):
    chat: Chat
    messages: list[Message]


Transport = Literal["command", "sse", "websocket"]


class McpServerConfig(BaseModel):
    transport: Transport
    enabled: bool | None
    command: str | None
    args: list[str] | None
    env: dict[str, str] | None
    url: str | None


class McpServers(BaseModel):
    mcp_servers: dict[str, McpServerConfig] = Field(alias="mcpServers")


class McpServerError(BaseModel):
    server_name: str = Field(alias="serverName")
    error: object  # any


class ModelConfiguration(BaseModel):
    base_url: str = Field(alias="baseURL")


class ModelSettings(BaseModel):
    model: str
    model_provider: str = Field(alias="modelProvider")
    api_key: str | None = Field(alias="apiKey")
    configuration: ModelConfiguration | None
    temperature: float | None
    top_p: float | None = Field(alias="topP")
    max_tokens: int | None = Field(alias="maxTokens")

    model_config = ConfigDict(extra="allow")


class ModelConfig(BaseModel):
    active_provider: str = Field(alias="activeProvider")
    enable_tools: bool = Field(alias="enableTools")
    configs: dict[str, ModelSettings]


class ModelSettingsProperty(BaseModel):
    type: Literal["string", "number"]
    description: str
    required: bool
    default: object | None
    placeholder: object | None


class ModelSettingsDefinition(ModelSettingsProperty):
    type: Literal["string", "number", "object"]
    properties: dict[str, ModelSettingsProperty] | None


class ModelInterfaceDefinition(BaseModel):
    model_settings: dict[str, ModelSettingsDefinition]


class McpTool(BaseModel):
    name: str
    tools: list
    description: str
    enabled: bool
    icon: str
