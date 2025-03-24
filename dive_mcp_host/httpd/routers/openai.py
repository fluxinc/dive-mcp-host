import time
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, Request
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from dive_mcp_host.httpd.dependencies import get_app
from dive_mcp_host.httpd.routers.models import ResultResponse, StreamMessage
from dive_mcp_host.httpd.routers.utils import ChatProcessor, EventStreamContextManager
from dive_mcp_host.httpd.server import DiveHostAPI

openai = APIRouter(tags=["openai"])


class OpenaiModel(BaseModel):
    """Represents an OpenAI model with its basic properties."""

    id: str
    type: str
    owned_by: str


class ModelsResult(ResultResponse):
    """Response model for listing available OpenAI models."""

    models: list[OpenaiModel]


class CompletionsMessage(BaseModel):
    """A message in the OpenAI completions API."""

    role: str
    content: str


class CompletionsMessageResp(CompletionsMessage):
    """A message in the OpenAI completions API."""

    refusal: None = None
    role: str | None = None


class CompletionsArgs(BaseModel):
    """Arguments for the OpenAI completions API."""

    messages: list[CompletionsMessage]
    stream: bool
    tool_choice: Literal["none", "auto"]


class CompletionsUsage(BaseModel):
    """Usage information for the OpenAI completions API."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionsChoice(BaseModel):
    """A choice in the OpenAI completions API."""

    index: int
    message: CompletionsMessageResp | None = None
    logprobs: None = None
    delta: CompletionsMessageResp | dict | None = None
    finish_reason: str | None = None


class CompletionsResult(BaseModel):
    """Result of the OpenAI completions API."""

    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list  # of what
    usage: CompletionsUsage | None = None
    system_fingerprint: str = "fp_dive"


class CompletionEventStreamContextManager(EventStreamContextManager):
    """Context manager for the OpenAI completions API."""

    chat_id: str
    model: str

    def __init__(self, chat_id: str, model: str) -> None:
        """Initialize the completion event stream context manager."""
        self.chat_id = chat_id
        self.model = model

    async def write(self, data: str | StreamMessage | CompletionsResult) -> None:
        """Write data to the stream."""
        if isinstance(data, StreamMessage) and data.type == "text":
            data = CompletionsResult(
                id=f"chatcmpl-{self.chat_id}",
                model=self.model,
                object="chat.completion.chunk",
                choices=[
                    CompletionsChoice(
                        index=0,
                        delta=CompletionsMessageResp(content=str(data.content))
                        if data.content
                        else CompletionsMessageResp(role="assistant", content=""),
                    )
                ],
            ).model_dump_json()
            await super().write(data)
        elif isinstance(data, CompletionsResult):
            await super().write(data.model_dump_json())


@openai.get("/")
async def get_openai() -> ResultResponse:
    """Returns a welcome message for the Dive Compatible API.

    Returns:
        ResultResponse: A success response with welcome message.
    """
    return ResultResponse(success=True, message="Welcome to Dive Compatible API! 🚀")


@openai.get("/models")
async def list_models(app: DiveHostAPI = Depends(get_app)) -> ModelsResult:
    """Lists all available OpenAI compatible models.

    Returns:
        ModelsResult: A list of available models.
    """
    return ModelsResult(
        success=True,
        models=[
            OpenaiModel(
                id=m._config.llm.model,  # noqa: SLF001
                type="model",
                owned_by=m._config.llm.modelProvider,  # noqa: SLF001
            )
            for m in app.dive_host.values()
        ],
    )


@openai.post("/chat/completions")
async def create_chat_completion(
    request: Request,
    params: CompletionsArgs,
    app: DiveHostAPI = Depends(get_app),
) -> object:  # idk what this actual do...
    """Creates a chat completion using the OpenAI compatible API.

    Returns:
        Not implemented yet.
    """
    has_system_message = False
    messages = []
    for message in params.messages:
        if message.role == "system":
            has_system_message = True
            messages.append(SystemMessage(content=message.content))
        elif message.role == "assistant":
            messages.append(AIMessage(content=message.content))
        else:
            messages.append(HumanMessage(content=message.content))

    if not has_system_message and (
        system_prompt := app.prompt_config_manager.get_prompt("system")
    ):
        messages.insert(
            0,
            SystemMessage(content=system_prompt),
        )

    dive_host = app.dive_host["default"]

    chat_id = str(uuid.uuid4())
    model_name = dive_host._config.llm.model  # noqa: SLF001
    stream = CompletionEventStreamContextManager(chat_id, model_name)

    async def process() -> tuple[CompletionsMessageResp, CompletionsUsage]:
        async with stream:
            processor = ChatProcessor(app, request.state, stream)
            result, usage = await processor.handle_chat_with_history(
                chat_id,
                None,
                messages,
                [] if params.tool_choice != "auto" else None,
            )

            await stream.write(
                CompletionsResult(
                    id=f"chatcmpl-{chat_id}",
                    model=model_name,
                    object="chat.completion.chunk",
                    choices=[
                        CompletionsChoice(
                            index=0,
                            delta={},
                            finish_reason="stop",
                        )
                    ],
                )
            )

            return (
                CompletionsMessageResp(role="assistant", content=result),
                CompletionsUsage(
                    prompt_tokens=usage.total_input_tokens,
                    completion_tokens=usage.total_output_tokens,
                    total_tokens=usage.total_tokens,
                ),
            )

    if params.stream:
        response = stream.get_response()
        stream.add_task(process)
        return response

    result, usage = await process()
    return CompletionsResult(
        id=f"chatcmpl-{chat_id}",
        model=model_name,
        choices=[
            CompletionsChoice(
                index=0,
                message=result,
                finish_reason="stop",
            )
        ],
        usage=usage,
    )
