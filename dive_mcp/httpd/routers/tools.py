from fastapi import APIRouter

from .models import McpTool, ResultResponse

tools = APIRouter(prefix="/tools", tags=["tools"])


class ToolsResult(ResultResponse):
    tools: list[McpTool]


@tools.get("/")
async def list_tools() -> ToolsResult:
    raise NotImplementedError
