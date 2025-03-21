from fastapi import APIRouter, Request

from dive_mcp_host.httpd.routers.models import McpTool, ResultResponse

tools = APIRouter(prefix="/tools", tags=["tools"])


class ToolsResult(ResultResponse):
    """Response model for listing available MCP tools."""

    tools: list[McpTool]


@tools.get("/")
async def list_tools(request: Request) -> ToolsResult:
    """Lists all available MCP tools.

    Returns:
        ToolsResult: A list of available tools.
    """
    mcp = request.app.state.mcp
    tools = await mcp.get_tool_infos()
    return ToolsResult(success=True, message=None, tools=tools)
