"""A simple echo mcp server for testing."""

from argparse import ArgumentParser
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP(name="echo")


@mcp.tool(name="echo", description="Echo a message.")
async def echo(
    message: Annotated[str, Field(description="The message to echo.")],
) -> str:
    """Echo a message.i lalala."""
    return message


@mcp.tool(name="ignore", description="Do nothing.")
async def ignore(
    message: Annotated[str, Field(description="The message I should ignore.")],  # noqa: ARG001
) -> None:
    """Do nothing."""
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--transport", type=str, default="stdio")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8800)

    args = parser.parse_args()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        raise ValueError(f"Invalid transport: {args.transport}")
