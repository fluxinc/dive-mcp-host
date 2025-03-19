"""Dive MCP Host CLI."""

import argparse
from pathlib import Path

from langchain_core.messages import HumanMessage

from dive_mcp_host.cli.cli_types import CLIArgs
from dive_mcp_host.host.conf import HostConfig
from dive_mcp_host.host.host import DiveMcpHost


def parse_query(args: type[CLIArgs]) -> HumanMessage:
    """Parse the query from the command line arguments."""
    query = " ".join(args.query)
    return HumanMessage(content=query)


def setup_argument_parser() -> type[CLIArgs]:
    """Setup the argument parser."""
    parser = argparse.ArgumentParser(description="Dive MCP Host CLI")
    parser.add_argument(
        "query",
        nargs="*",
        default=[],
        help="The input query.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="The path to the configuration file.",
        dest="config_path",
    )
    parser.add_argument(
        "-c",
        type=str,
        default=None,
        help="Continue from given THREAD_ID.",
        dest="thread_id",
    )
    return parser.parse_args(namespace=CLIArgs)


def load_config(config_path: str) -> HostConfig:
    """Load the configuration."""
    with Path(config_path).open("r") as f:
        return HostConfig.model_validate_json(f.read())


async def run() -> None:
    """dive_mcp_host CLI entrypoint."""
    args = setup_argument_parser()
    query = parse_query(args)
    config = load_config(args.config_path)

    current_thread_id: str | None = args.thread_id

    async with DiveMcpHost(config) as mcp_host:
        conversation = mcp_host.conversation(thread_id=current_thread_id)
        current_thread_id = conversation.thread_id
        async with conversation:
            async for response in conversation.query(query, stream_mode="messages"):
                print(response[0].content, end="")

    print()
    print(f"Thread ID: {current_thread_id}")
