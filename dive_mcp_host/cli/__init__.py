"""Dive MCP Host CLI."""

import asyncio

from dive_mcp_host.cli.cli import run


def main() -> None:
    """dive_mcp_host CLI entrypoint."""
    asyncio.run(run())
