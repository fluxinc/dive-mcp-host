from fastapi import APIRouter

model_verify = APIRouter(prefix="/model_verify", tags=["model_verify"])


@model_verify.post("/streaming")
async def verify_model():
    raise NotImplementedError
