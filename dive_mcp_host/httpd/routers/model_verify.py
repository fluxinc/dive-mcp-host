from fastapi import APIRouter

from dive_mcp_host.host.conf import LLMConfig

model_verify = APIRouter(prefix="/model_verify", tags=["model_verify"])


@model_verify.post("")
async def do_verify_model(settings: LLMConfig) -> None:
    """Verify if a model supports streaming capabilities.

    Returns:
        Not implemented yet.
    """
    raise NotImplementedError


@model_verify.post("/streaming")
async def verify_model() -> None:
    """Verify if a model supports streaming capabilities.

    Returns:
        Not implemented yet.
    """
    raise NotImplementedError
