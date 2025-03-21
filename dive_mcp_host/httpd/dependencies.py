"""Dependencies for the MCP host."""

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from dive_mcp_host.httpd.middlewares.general import DiveUser
    from dive_mcp_host.httpd.server import DiveHostAPI


def get_app(request: Request) -> "DiveHostAPI":
    """Get the DiveHostAPI instance."""
    return request.app


def get_dive_user(
    request: Request,
) -> "DiveUser":
    """Get the DiveUser instance."""
    return request.state.dive_user
