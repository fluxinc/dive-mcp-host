from fastapi import APIRouter
from pydantic import BaseModel

from .models import ResultResponse

openai = APIRouter(prefix="/v1/openai", tags=["openai"])


class OpenaiModel(BaseModel):
    id: str
    type: str
    owned_by: str


class ModelsResult(ResultResponse):
    models: list[OpenaiModel]


@openai.get("/")
async def get_openai() -> ResultResponse:
    return ResultResponse(success=True, message="Welcome to Dive Compatible API! 🚀")


@openai.get("/models")
async def list_models() -> ModelsResult:
    raise NotImplementedError


@openai.post("/chat/completions")
async def create_chat_completion():  # idk what this actual do...
    raise NotImplementedError
