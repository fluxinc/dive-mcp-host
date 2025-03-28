"""Dive MCP Host HTTPD.

Support Restful API and websocket.
"""

import socket

import uvicorn

from dive_mcp_host.httpd.app import create_app
from dive_mcp_host.httpd.conf.arguments import Arguments
from dive_mcp_host.httpd.conf.service.manager import ServiceManager


def main() -> None:
    """Dive MCP Host HTTPD entrypoint."""
    args = Arguments.parse_args()

    service_config_manager = ServiceManager(str(args.httpd_config))
    service_config_manager.initialize()
    if service_config_manager.current_setting is None:
        raise ValueError("Service config manager is not initialized")

    app = create_app(str(args.httpd_config))

    if args.port:
        uvicorn.run(
            app,
            host=args.listen,
            port=args.port,
            log_config=service_config_manager.current_setting.logging_config,
        )
    else:
        serversocket = socket.socket(
            socket.AF_INET6 if ":" in args.listen else socket.AF_INET,
            socket.SOCK_STREAM,
        )
        start = 61990
        serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = 0
        for i in range(1000):
            port = start + i
            try:
                serversocket.bind((args.listen, port))
                break
            except OSError:
                pass
        else:
            raise RuntimeError(f"No available port found in range {start}-{port}")
        uvicorn.run(
            app,
            fd=serversocket.fileno(),
            log_config=service_config_manager.current_setting.logging_config,
        )
