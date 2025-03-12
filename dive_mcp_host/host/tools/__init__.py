"""Model for the MCP servers."""

from langchain_core.tools import BaseTool


class ToolManager:
    """Manager for the MCP Servers."""

    def tools(self) -> list[BaseTool]:
        """Get the tools for the MCP server."""
        raise NotImplementedError
