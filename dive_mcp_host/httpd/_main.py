"""Dive MCP Host HTTPD.

Support Restful API and websocket.
"""

import uvicorn

from dive_mcp_host.httpd.app import app


def main() -> None:
    """Dive MCP Host HTTPD entrypoint."""
    # NOTE: Should be removed when we have a proper server config
    temp_logger_config = {
        "disable_existing_loggers": False,
        "version": 1,
        "handlers": {
            "default": {"class": "logging.StreamHandler", "formatter": "default"}
        },
        "formatters": {
            "default": {
                "format": "%(levelname)s %(name)s:%(funcName)s:%(lineno)d :: %(message)s"  # noqa: E501
            }
        },
        "root": {"level": "INFO", "handlers": ["default"]},
        "loggers": {"dive_mcp_host": {"level": "DEBUG"}},
    }

    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=temp_logger_config)
