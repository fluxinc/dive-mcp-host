"""Dive MCP Host HTTPD.

Support Restful API and websocket.
"""

import socket
from pathlib import Path

import uvicorn

from dive_mcp_host.httpd.app import create_app
from dive_mcp_host.httpd.conf.arguments import Arguments
from dive_mcp_host.httpd.conf.envs import RESOURCE_DIR
from dive_mcp_host.httpd.conf.service.manager import ConfigLocation, ServiceManager


def main() -> None:
    """Dive MCP Host HTTPD entrypoint."""
    args = Arguments.parse_args()

    service_config_manager = ServiceManager(str(args.httpd_config))
    service_config_manager.initialize()
    if service_config_manager.current_setting is None:
        raise ValueError("Service config manager is not initialized")

    # Overwrite defaults from command line arguments
    resource_dir = Path(args.working_dir) if args.working_dir else RESOURCE_DIR
    service_config_manager.overwrite_paths(
        ConfigLocation(
            mcp_server_config_path=str(args.mcp_config),
            model_config_path=str(args.llm_config),
            prompt_config_path=str(args.custom_rules),
            command_alias_config_path=str(args.command_alias_config),
        ),
        resource_dir=resource_dir,
    )

    app = create_app(service_config_manager)

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
