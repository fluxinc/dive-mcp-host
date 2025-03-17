"""Dependencies for the MCP host."""

from fastapi import Request

from dive_mcp_host.httpd.server import DiveHostAPI


def get_app(request: Request) -> DiveHostAPI:
    """Get the DiveHostAPI instance."""
    return request.app
