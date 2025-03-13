from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .database import SqliteDatabase
from .middlewares import KwargsMiddleware
from .routers import chat, config, model_verify, openai, tools
from .store.local import LocalStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan for the FastAPI app."""
    if not hasattr(app.state, "db"):
        app.state.db = SqliteDatabase("db.sqlite")
    if not hasattr(app.state, "store"):
        app.state.store = LocalStore()
    yield
    # shutdown


app = FastAPI(lifespan=lifespan)

kwargs_func = {}

app.add_middleware(KwargsMiddleware, kwargs_func=kwargs_func)
app.include_router(openai)
app.include_router(chat, prefix="/api")
app.include_router(tools, prefix="/api")
app.include_router(config, prefix="/api")
app.include_router(model_verify, prefix="/api")

# memo: fastapi dev app.py
