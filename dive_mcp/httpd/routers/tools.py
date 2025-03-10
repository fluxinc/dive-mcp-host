from fastapi import APIRouter

from .models import McpTool, ResultResponse

tools = APIRouter(prefix="/tools", tags=["tools"])


class ToolsResult(ResultResponse):
    """Response model for listing available MCP tools."""

    tools: list[McpTool]


@tools.get("/")
async def list_tools() -> ToolsResult:
    """Lists all available MCP tools.

    Returns:
        ToolsResult: A list of available tools.
    """
    raise NotImplementedError
