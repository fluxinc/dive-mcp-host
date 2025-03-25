from typing import Any

from fastapi import APIRouter, Depends
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from dive_mcp_host.host.conf import LLMConfig
from dive_mcp_host.httpd.dependencies import get_app
from dive_mcp_host.httpd.server import DiveHostAPI

model_verify = APIRouter(tags=["model_verify"])


class TestTool(BaseTool):
    """Test tool."""

    name: str = "test_tool"
    description: str = "a simple test tool check tool functionality call it any name with any arguments, returns nothing"  # noqa: E501
    called: bool = False

    def _run(self, *_args: Any, **_kwargs: Any) -> Any:
        self.called = True
        return {"result": "success"}


class ModelVerifyResult(BaseModel):
    """Model verify result."""

    success: bool = False
    connecting_success: bool = Field(default=False, alias="connectingSuccess")
    # connecting_result: str = Field(default="", alias="connectingResult")
    support_tools: bool = Field(default=False, alias="supportTools")
    # support_tools_result: str = Field(default="", alias="supportToolsResult")


@model_verify.post("")
async def do_verify_model(
    app: DiveHostAPI = Depends(get_app), _settings: LLMConfig | None = None
) -> ModelVerifyResult:
    """Verify if a model supports streaming capabilities.

    Returns:
        ModelVerifyResult
    """
    dive_host = app.dive_host["default"]
    test_tool = TestTool()
    conversation = dive_host.conversation(
        tools=[test_tool],
        volatile=True,
    )

    async with conversation:
        _responses = [
            response
            async for response in conversation.query(
                "run test_tool", stream_mode=["updates"]
            )
        ]

    return ModelVerifyResult(
        success=True,
        connectingSuccess=True,
        supportTools=test_tool.called,
    )


@model_verify.post("/streaming")
async def verify_model() -> None:
    """Verify if a model supports streaming capabilities.

    Returns:
        Not implemented yet.
    """
    raise NotImplementedError
