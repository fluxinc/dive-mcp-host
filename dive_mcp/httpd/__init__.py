"""Dive MCP Host HTTPD.

Support Restful API and websocket.
"""

from .app import app


def main() -> None:
    """dive_httpd entrypoint."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
