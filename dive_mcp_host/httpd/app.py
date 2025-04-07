from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from logging import getLogger

from starlette.middleware.base import BaseHTTPMiddleware

from dive_mcp_host.httpd.conf.service.manager import ServiceManager
from dive_mcp_host.httpd.middlewares import default_state, error_handler
from dive_mcp_host.httpd.middlewares.general import allow_cors
from dive_mcp_host.httpd.routers.chat import chat
from dive_mcp_host.httpd.routers.config import config
from dive_mcp_host.httpd.routers.model_verify import model_verify
from dive_mcp_host.httpd.routers.openai import openai
from dive_mcp_host.httpd.routers.tools import tools
from dive_mcp_host.httpd.server import DiveHostAPI

logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(app: DiveHostAPI) -> AsyncGenerator[None, None]:
    """Lifespan for the FastAPI app."""
    try:
        async with app.prepare():
            app.report_status()
            yield
    except Exception as e:
        logger.exception("Error in lifespan")
        app.report_status(error=str(e))
    finally:
        await app.cleanup()


def create_app(
    service_config_manager: ServiceManager,
) -> DiveHostAPI:
    """Create the FastAPI app."""
    app = DiveHostAPI(
        lifespan=lifespan,
        service_config_manager=service_config_manager,
    )

    service_setting = service_config_manager.current_setting
    if service_setting and service_setting.cors_origin:
        app.add_middleware(
            BaseHTTPMiddleware,
            dispatch=allow_cors(service_setting.cors_origin),
        )
    app.add_middleware(BaseHTTPMiddleware, dispatch=default_state)
    app.add_middleware(BaseHTTPMiddleware, dispatch=error_handler)
    app.include_router(openai, prefix="/v1/openai")
    app.include_router(chat, prefix="/api/chat")
    app.include_router(tools, prefix="/api/tools")
    app.include_router(config, prefix="/api/config")
    app.include_router(model_verify, prefix="/model_verify")

    # remote endpoints
    app.include_router(chat, prefix="/api/v1/mcp")

    return app
