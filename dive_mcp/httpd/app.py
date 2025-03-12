from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .routers import chat, config, model_verify, openai, tools


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan for the FastAPI app."""
    app.state.db = {"is": "this"}
    yield
    # shutdown


async def get_app() -> FastAPI:
    """Get the mcp host root app."""
    app = FastAPI(lifespan=lifespan)
    app.include_router(openai)
    app.include_router(chat)
    app.include_router(tools)
    app.include_router(config)
    app.include_router(model_verify)

    return app


# memo: fastapi dev app.py
