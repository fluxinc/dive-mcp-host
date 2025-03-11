from typing import TypeVar

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .models import (
    McpServerError,
    McpServers,
    ModelConfig,
    ModelInterfaceDefinition,
    ModelSettings,
    ResultResponse,
)

config = APIRouter(prefix="/config", tags=["config"])

T = TypeVar("T")


class ConfigResult[T](ResultResponse):
    """Generic configuration result that extends ResultResponse with a config field."""

    config: T


class SaveConfigResult(ResultResponse):
    """Result of saving configuration, including any errors that occurred."""

    errors: list[McpServerError]


class InterfaceResult(ResultResponse):
    """Result containing model interface definition."""

    interface: ModelInterfaceDefinition


class RulesResult(ResultResponse):
    """Result containing custom rules as a string."""

    rules: str


class SaveModelSettingsRequest(BaseModel):
    """Request model for saving model settings."""

    provider: str
    model_settings: ModelSettings = Field(alias="modelSettings")
    enable_tools: bool = Field(alias="enableTools")


@config.get("/mcpserver")
async def get_mcp_server() -> ConfigResult[McpServers]:
    """Get MCP server configurations.

    Returns:
        ConfigResult[McpServers]: Configuration for MCP servers.
    """
    raise NotImplementedError


@config.post("/mcpserver")
async def post_mcp_server(servers: McpServers) -> SaveConfigResult:
    """Save MCP server configurations.

    Args:
        servers (McpServers): The server configurations to save.

    Returns:
        SaveConfigResult: Result of the save operation with any errors.
    """
    raise NotImplementedError


@config.get("/model")
async def get_model() -> ConfigResult[ModelConfig]:
    """Get current model configuration.

    Returns:
        ConfigResult[ModelConfig]: Current model configuration.
    """
    raise NotImplementedError


@config.post("/model")
async def post_model(model_settings: SaveModelSettingsRequest) -> ResultResponse:
    """Save model settings for a specific provider.

    Args:
        model_settings (SaveModelSettingsRequest): The model settings to save.

    Returns:
        ResultResponse: Result of the save operation.
    """
    raise NotImplementedError


@config.post("/model/replaceAll")
async def post_model_replace_all(model_config: ModelConfig) -> ResultResponse:
    """Replace all model configurations.

    Args:
        model_config (ModelConfig): The complete model configuration to use.

    Returns:
        ResultResponse: Result of the replace operation.
    """
    raise NotImplementedError


@config.get("/model/interface")
async def get_model_interface() -> InterfaceResult:
    """Get model interface definition.

    Returns:
        InterfaceResult: Model interface definition.
    """
    raise NotImplementedError


@config.get("/customrules")
async def get_custom_rules() -> RulesResult:
    """Get custom rules configuration.

    Returns:
        RulesResult: Custom rules as a string.
    """
    raise NotImplementedError


@config.post("/customrules")
async def post_custom_rules() -> ResultResponse:
    """Save custom rules configuration.

    Returns:
        ResultResponse: Result of the save operation.
    """
    raise NotImplementedError
