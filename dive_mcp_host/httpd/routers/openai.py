from fastapi import APIRouter
from pydantic import BaseModel

from .chat import event_stream
from .models import ResultResponse

openai = APIRouter(prefix="/v1/openai", tags=["openai"])


class OpenaiModel(BaseModel):
    """Represents an OpenAI model with its basic properties."""

    id: str
    type: str
    owned_by: str


class ModelsResult(ResultResponse):
    """Response model for listing available OpenAI models."""

    models: list[OpenaiModel]


@openai.get("/")
async def get_openai() -> ResultResponse:
    """Returns a welcome message for the Dive Compatible API.

    Returns:
        ResultResponse: A success response with welcome message.
    """
    return ResultResponse(success=True, message="Welcome to Dive Compatible API! 🚀")


@openai.get("/models")
async def list_models() -> ModelsResult:
    """Lists all available OpenAI compatible models.

    Returns:
        ModelsResult: A list of available models.
    """
    raise NotImplementedError


@openai.post("/chat/completions")
async def create_chat_completion() -> object:  # idk what this actual do...
    """Creates a chat completion using the OpenAI compatible API.

    Returns:
        Not implemented yet.
    """
    raise NotImplementedError
