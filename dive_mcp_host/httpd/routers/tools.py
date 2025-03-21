from logging import getLogger

from fastapi import APIRouter, Depends

from dive_mcp_host.httpd.dependencies import get_app
from dive_mcp_host.httpd.routers.models import (
    McpTool,
    ResultResponse,
    SimpleToolInfo,
    ToolsCache,
)
from dive_mcp_host.httpd.server import DiveHostAPI
from dive_mcp_host.httpd.store.cache import CacheKeys

logger = getLogger(__name__)
tools = APIRouter(prefix="/tools", tags=["tools"])


class ToolsResult(ResultResponse):
    """Response model for listing available MCP tools."""

    tools: list[McpTool]


@tools.get("/")
async def list_tools(
    app: DiveHostAPI = Depends(get_app),
) -> ToolsResult:
    """Lists all available MCP tools.

    Returns:
        ToolsResult: A list of available tools.
    """
    result: dict[str, McpTool] = {}

    # get full list of servers from config
    if app.mcp_server_config_manager.current_config is not None:
        all_servers = set(
            app.mcp_server_config_manager.current_config.mcp_servers.keys()
        )
    else:
        all_servers = set()

    # get tools from dive host
    for server_name, server_info in app.dive_host["default"].mcp_server_info.items():
        result[server_name] = McpTool(
            name=server_name,
            tools=[
                SimpleToolInfo(name=tool.name, description=tool.description or "")
                for tool in server_info.tools
            ],
            description="",
            enabled=True,
            icon="",
            error=str(server_info.error) if server_info.error is not None else None,
        )

    # find missing servers
    missing_servers = all_servers - set(result.keys())

    # get missing from local cache
    raw_cached_tools = app.local_file_cache.get(CacheKeys.LIST_TOOLS)
    if raw_cached_tools is not None and missing_servers:
        cached_tools = ToolsCache.model_validate_json(raw_cached_tools)
        for server_name in missing_servers:
            if server_info := cached_tools.root.get(server_name, None):
                result[server_name] = server_info
            else:
                result[server_name] = McpTool(
                    name=server_name,
                    tools=[],
                    description="",
                    enabled=False,
                    icon="",
                    error=None,
                )

    # update local cache
    app.local_file_cache.set(
        CacheKeys.LIST_TOOLS,
        ToolsCache(root=result).model_dump_json(exclude_none=True),
    )
    return ToolsResult(success=True, message=None, tools=list(result.values()))
