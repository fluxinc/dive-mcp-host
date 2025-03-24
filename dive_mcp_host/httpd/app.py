from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from starlette.middleware.base import BaseHTTPMiddleware

from dive_mcp_host.httpd.middlewares import default_state, error_handler
from dive_mcp_host.httpd.routers.chat import chat
from dive_mcp_host.httpd.routers.config import config
from dive_mcp_host.httpd.routers.model_verify import model_verify
from dive_mcp_host.httpd.routers.openai import openai
from dive_mcp_host.httpd.routers.tools import tools
from dive_mcp_host.httpd.server import DiveHostAPI


@asynccontextmanager
async def lifespan(app: DiveHostAPI) -> AsyncGenerator[None, None]:
    """Lifespan for the FastAPI app."""
    async with app.prepare():
        yield
    await app.cleanup()


def create_app(config_path: str) -> DiveHostAPI:
    """Create the FastAPI app."""
    app = DiveHostAPI(lifespan=lifespan, config_path=config_path)

    app.add_middleware(BaseHTTPMiddleware, dispatch=default_state)
    app.add_middleware(BaseHTTPMiddleware, dispatch=error_handler)
    app.include_router(openai, prefix="/v1/openai")
    app.include_router(chat, prefix="/chat")
    app.include_router(tools, prefix="/tools")
    app.include_router(config, prefix="/config")
    app.include_router(model_verify, prefix="/model_verify")

    # remote endpoints
    app.include_router(chat, prefix="/api/v1/mcp")

    return app
