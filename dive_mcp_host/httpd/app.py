from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from starlette.middleware.base import BaseHTTPMiddleware

from dive_mcp_host.httpd.server import DiveHostAPI

from .middlewares import default_state, error_handler
from .routers import chat, config, model_verify, openai, tools


@asynccontextmanager
async def lifespan(app: DiveHostAPI) -> AsyncGenerator[None, None]:
    """Lifespan for the FastAPI app."""
    async with app.prepare():
        yield
    await app.cleanup()


app = DiveHostAPI(lifespan=lifespan)

app.add_middleware(BaseHTTPMiddleware, dispatch=default_state)
app.add_middleware(BaseHTTPMiddleware, dispatch=error_handler)
app.include_router(openai)
app.include_router(chat, prefix="/api")
app.include_router(tools, prefix="/api")
app.include_router(config, prefix="/api")
app.include_router(model_verify, prefix="/api")

# memo: fastapi dev app.py
