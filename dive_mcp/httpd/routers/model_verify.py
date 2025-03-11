from fastapi import APIRouter

model_verify = APIRouter(prefix="/model_verify", tags=["model_verify"])


@model_verify.post("/streaming")
async def verify_model() -> None:
    """Verify if a model supports streaming capabilities.

    Returns:
        Not implemented yet.
    """
    raise NotImplementedError
