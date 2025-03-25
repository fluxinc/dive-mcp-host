import asyncio
from typing import TypeVar

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from dive_mcp_host.host.conf import LLMConfig
from dive_mcp_host.httpd.conf.mcpserver.manager import Config
from dive_mcp_host.httpd.dependencies import get_app
from dive_mcp_host.httpd.server import DiveHostAPI

from .models import (
    McpServerError,
    McpServers,
    ModelConfig,
    ModelInterfaceDefinition,
    ModelSettingsDefinition,
    ResultResponse,
)

config = APIRouter(tags=["config"])

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
    model_settings: LLMConfig = Field(alias="modelSettings")
    enable_tools: bool = Field(alias="enableTools")


@config.get("/mcpserver")
async def get_mcp_server(
    app: DiveHostAPI = Depends(get_app),
) -> ConfigResult[McpServers]:
    """Get MCP server configurations.

    Returns:
        ConfigResult[McpServers]: Configuration for MCP servers.
    """
    if app.mcp_server_config_manager.current_config is None:
        raise ValueError("MCP server configuration not found")

    config = McpServers.model_validate(
        app.mcp_server_config_manager.current_config, from_attributes=True
    )
    return ConfigResult(
        success=True,
        config=config,
    )


@config.post("/mcpserver")
async def post_mcp_server(
    servers: McpServers,
    app: DiveHostAPI = Depends(get_app),
) -> SaveConfigResult:
    """Save MCP server configurations.

    Args:
        servers (McpServers): The server configurations to save.
        app (DiveHostAPI): The DiveHostAPI instance.


    Returns:
        SaveConfigResult: Result of the save operation with any errors.
    """
    # Update conifg
    new_config = Config.model_validate(servers, from_attributes=True)
    if not app.mcp_server_config_manager.update_all_configs(new_config):
        raise ValueError("Failed to update MCP server configurations")

    # Reload host
    # TODO: mcp server reloader
    await app.dive_host["default"].reload(
        new_config=app.load_host_config(),
        reloader=lambda: asyncio.sleep(0),
    )

    # Get failed MCP servers
    failed_servers: list[McpServerError] = []
    for server_name, server_info in app.dive_host["default"].mcp_server_info.items():
        if server_info.error is not None:
            failed_servers.append(
                McpServerError(
                    serverName=server_name,
                    error=str(server_info.error),
                )
            )

    return SaveConfigResult(
        success=True,
        errors=failed_servers,
    )


@config.get("/model")
async def get_model(
    app: DiveHostAPI = Depends(get_app),
) -> ConfigResult["ModelConfig"]:
    """Get current model configuration.

    Returns:
        ConfigResult[ModelConfig]: Current model configuration.
    """
    if app.model_config_manager.full_config is None:
        raise ValueError("Model configuration not found")

    return ConfigResult(success=True, config=app.model_config_manager.full_config)


@config.post("/model")
async def post_model(
    model_settings: SaveModelSettingsRequest,
    app: DiveHostAPI = Depends(get_app),
) -> ResultResponse:
    """Save model settings for a specific provider.

    Args:
        model_settings (SaveModelSettingsRequest): The model settings to save.
        app (DiveHostAPI): The DiveHostAPI instance.

    Returns:
        ResultResponse: Result of the save operation.
    """
    app.model_config_manager.save_single_settings(
        provider=model_settings.provider,
        upload_model_settings=model_settings.model_settings,
        enable_tools_=model_settings.enable_tools,
    )

    # Reload model config
    if not app.model_config_manager.initialize():
        raise ValueError("Failed to reload model configuration")

    # Reload host
    # TODO: model reloader
    await app.dive_host["default"].reload(
        new_config=app.load_host_config(),
        reloader=lambda: asyncio.sleep(0),
    )

    return ResultResponse(success=True)


@config.post("/model/replaceAll")
async def post_model_replace_all(
    model_config: "ModelConfig",
    app: DiveHostAPI = Depends(get_app),
) -> ResultResponse:
    """Replace all model configurations.

    Args:
        model_config (ModelConfig): The complete model configuration to use.
        app (DiveHostAPI): The DiveHostAPI instance.

    Returns:
        ResultResponse: Result of the replace operation.
    """
    app.model_config_manager.replace_all_settings(model_config)
    return ResultResponse(success=True)


@config.get("/model/interface")
async def get_model_interface() -> InterfaceResult:
    """Get model interface definition.

    Returns:
        InterfaceResult: Model interface definition.
    """
    return InterfaceResult(
        success=True,
        interface=ModelInterfaceDefinition(
            model_settings={
                "modelProvider": ModelSettingsDefinition(
                    type="string",
                    description="The provider sdk of the model",
                    required=True,
                    default="",
                    placeholder="openai",
                ),
                "model": ModelSettingsDefinition(
                    type="string",
                    description="The model's name to use",
                    required=True,
                    default="gpt-4o-mini",
                ),
                "apiKey": ModelSettingsDefinition(
                    type="string",
                    description="The Model Provider API key",
                    required=False,
                    default="",
                    placeholder="YOUR_API_KEY",
                ),
                "baseURL": ModelSettingsDefinition(
                    type="string",
                    description="The model's base URL",
                    required=False,
                    default="",
                    placeholder="",
                ),
            },
        ),
    )


@config.get("/customrules")
async def get_custom_rules(app: DiveHostAPI = Depends(get_app)) -> RulesResult:
    """Get custom rules configuration.

    Returns:
        RulesResult: Custom rules as a string.
    """
    custom_rules = app.prompt_config_manager.load_custom_rules()
    return RulesResult(success=True, rules=custom_rules)


@config.post("/customrules")
async def post_custom_rules(
    request: Request,
    app: DiveHostAPI = Depends(get_app),
) -> ResultResponse:
    """Save custom rules configuration.

    Returns:
        ResultResponse: Result of the save operation.
    """
    raw_rules = await request.body()
    rules = raw_rules.decode("utf-8")
    app.prompt_config_manager.write_custom_rules(rules)
    app.prompt_config_manager.update_system_prompt()
    return ResultResponse(success=True)
