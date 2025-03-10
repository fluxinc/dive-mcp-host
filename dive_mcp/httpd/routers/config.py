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
    config: T


class SaveConfigResult(ResultResponse):
    errors: list[McpServerError]


class InterfaceResult(ResultResponse):
    interface: ModelInterfaceDefinition


class RulesResult(ResultResponse):
    rules: str


class SaveModelSettingsRequest(BaseModel):
    provider: str
    model_settings: ModelSettings = Field(alias="modelSettings")
    enable_tools: bool = Field(alias="enableTools")


@config.get("/mcpserver")
async def get_mcp_server() -> ConfigResult[McpServers]:
    raise NotImplementedError


@config.post("/mcpserver")
async def post_mcp_server(servers: McpServers) -> SaveConfigResult:
    raise NotImplementedError


@config.get("/model")
async def get_model() -> ConfigResult[ModelConfig]:
    raise NotImplementedError


@config.post("/model")
async def post_model(model_settings: SaveModelSettingsRequest) -> ResultResponse:
    raise NotImplementedError


@config.post("/model/replaceAll")
async def post_model_replace_all(model_config: ModelConfig) -> ResultResponse:
    raise NotImplementedError


@config.get("/model/interface")
async def get_model_interface() -> InterfaceResult:
    raise NotImplementedError


@config.get("/customrules")
async def get_custom_rules() -> RulesResult:
    raise NotImplementedError


@config.post("/customrules")
async def post_custom_rules() -> ResultResponse:
    raise NotImplementedError
