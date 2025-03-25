import asyncio
import json
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from dive_mcp_host.host.conf import LLMConfig
from dive_mcp_host.httpd.dependencies import get_app
from dive_mcp_host.httpd.routers.utils import EventStreamContextManager
from dive_mcp_host.httpd.server import DiveHostAPI

if TYPE_CHECKING:
    from dive_mcp_host.httpd.middlewares.general import DiveUser

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
    app: DiveHostAPI = Depends(get_app),
    settings: LLMConfig | None = None,  # noqa: ARG001
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
async def verify_model(
    request: Request,
    app: DiveHostAPI = Depends(get_app),
    settings: LLMConfig | None = None,  # noqa: ARG001
) -> StreamingResponse:
    """Verify if a model supports streaming capabilities.

    Returns:
        CompletionEventStreamContextManager
    """
    dive_host = app.dive_host["default"]
    chat_id = str(uuid.uuid4())
    model_name = dive_host._config.llm.model  # noqa: SLF001
    stream = EventStreamContextManager()
    response = stream.get_response()
    dive_user: DiveUser = request.state.dive_user

    test_tool = TestTool()
    conversation = dive_host.conversation(
        user_id=dive_user["user_id"] or "default",
        tools=[test_tool],
        volatile=True,
    )

    async def abort_handler() -> None:
        while not await request.is_disconnected():
            await asyncio.sleep(1)
        conversation.abort()

    async def process() -> None:
        async with stream:
            task = asyncio.create_task(abort_handler())
            conversation.query("run test_tool", stream_mode=["updates"])
            async with conversation:
                responses = [
                    response[1]  # type: ignore  # noqa: PGH003
                    async for response in conversation.query(
                        "run test_tool", stream_mode=["updates"]
                    )
                ]

            tool_response: ToolMessage | None = None
            ai_response: AIMessage | None = None
            for response in responses[::-1]:
                if isinstance(response, ToolMessage):
                    tool_response = response
                elif isinstance(response, AIMessage):
                    ai_response = response

            await stream.write(
                json.dumps(
                    {
                        "type": "progress",
                        "step": 1,
                        "modelName": model_name,
                        "testType": "tools",
                        "status": "success" if test_tool.called else "error",
                        "error": None,
                    }
                )
            )
            await stream.write(
                json.dumps(
                    {
                        "type": "final",
                        "results": [
                            {
                                "modelName": model_name,
                                "connection": {
                                    "status": "success",
                                    "result": ai_response.content
                                    if ai_response
                                    else None,
                                },
                                "tools": {
                                    "status": "success"
                                    if test_tool.called
                                    else "error",
                                    "result": tool_response.content
                                    if tool_response
                                    else None,
                                },
                            }
                        ],
                    }
                )
            )

            task.cancel()

    stream.add_task(process)
    return response
