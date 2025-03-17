from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

from dive_mcp_host.httpd.database import SqliteDatabase
from dive_mcp_host.httpd.middlewares import (
    KwargsMiddleware,
    default_state,
    error_handler,
)
from dive_mcp_host.httpd.routers import chat, config, model_verify, openai, tools
from dive_mcp_host.httpd.store.local import LocalStore


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

app.add_middleware(BaseHTTPMiddleware, dispatch=default_state)
app.add_middleware(BaseHTTPMiddleware, dispatch=error_handler)
app.add_middleware(KwargsMiddleware, kwargs_func=kwargs_func)
app.include_router(openai)
app.include_router(chat, prefix="/api")
app.include_router(tools, prefix="/api")
app.include_router(config, prefix="/api")
app.include_router(model_verify, prefix="/api")

# memo: fastapi dev app.py
