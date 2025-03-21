"""Dive MCP Host HTTPD.

Support Restful API and websocket.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import uvicorn

from dive_mcp_host.httpd.app import create_app
from dive_mcp_host.httpd.conf.service.manager import ServiceManager


@dataclass
class Args:
    config: str


def main() -> None:
    """Dive MCP Host HTTPD entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=str(Path.cwd() / "serviceConfig.json")
    )
    args = parser.parse_args()

    service_config_manager = ServiceManager(args.config)
    service_config_manager.initialize()
    if service_config_manager.current_setting is None:
        raise ValueError("Service config manager is not initialized")

    app = create_app(args.config)

    uvicorn.run(
        app,
        host="0.0.0.0",  # noqa: S104
        port=8000,
        log_config=service_config_manager.current_setting.logging_config,
    )
