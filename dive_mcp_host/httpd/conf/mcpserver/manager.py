import asyncio
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field


# Define necessary types
class FunctionDefinition(BaseModel):
    """Function definition for tool."""

    name: str
    description: str
    parameters: dict[str, Any] = {}


class ToolDefinition(BaseModel):
    """Tool definition."""

    type: str = "function"
    function: FunctionDefinition


class McpTool(BaseModel):
    """Represents an MCP tool with its properties and metadata."""

    name: str
    tools: list[dict[str, str]]
    description: str
    enabled: bool
    icon: str


# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ServerError(BaseModel):
    """Represents an error from an MCP server."""

    server_name: str = Field(alias="serverName")
    error: object


class ServerConfig(BaseModel):
    """Server configuration model."""

    transport: str  # "command", "sse", "websocket"
    enabled: bool = True
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    url: str | None = None


class Config(BaseModel):
    """Configuration model."""

    mcp_servers: dict[str, ServerConfig] = Field(alias="mcpServers")


class MockClient:
    """Mock client for MCP server."""

    def __init__(self, server_name: str) -> None:
        """Initialize the mock client.

        Args:
            server_name: Name of the server.
        """
        self.server_name = server_name
        self.transport = None

    async def list_tools(self) -> dict[str, Any]:
        """List available tools."""
        return {
            "tools": [
                {
                    "name": f"tool_{self.server_name}",
                    "description": f"Tool for {self.server_name}",
                },
            ],
        }

    async def get_server_capabilities(self) -> dict[str, Any]:
        """Get server capabilities."""
        return {
            "description": f"Mock server for {self.server_name}",
            "icon": "mock-icon",
        }

    async def close(self) -> None:
        """Close the client connection."""


class MockTransport:
    """Mock transport for MCP server."""

    def __init__(self, server_params: ServerConfig) -> None:
        """Initialize the mock transport.

        Args:
            server_params: Server configuration parameters.
        """
        self._server_params = server_params

    async def close(self) -> None:
        """Close the transport."""


class MCPServerManager:
    """MCP Server Manager."""

    _instance: ClassVar[Any] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "MCPServerManager":  # noqa: ARG004
        """Create a new instance or return existing instance of MCPServerManager.

        Returns:
            MCPServerManager instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize directly
            cls._instance._initialized = False  # noqa: SLF001
        return cls._instance

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize the MCPServerManager.

        Args:
            config_path: Optional path to the configuration file.
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.servers: dict[str, Any] = {}
        self.transports: dict[str, Any] = {}
        self.tool_to_server_map: dict[str, Any] = {}
        self.available_tools: list[ToolDefinition] = []
        self.tool_infos: list[McpTool] = []
        self.temp_clients: dict[str, Any] = {}
        self.config_path = config_path or str(Path.cwd() / "config.json")
        self._initialized = True
        # Store any active tasks
        self._task: asyncio.Task | None = None

    @classmethod
    def get_instance(cls, config_path: str | None = None) -> "MCPServerManager":
        """Get the singleton instance of MCPServerManager.

        Args:
            config_path: Optional path to the configuration file.

        Returns:
            MCPServerManager instance.
        """
        instance = cls()  # Do not pass parameters to the __new__ method
        if config_path and config_path != instance.config_path:
            instance.config_path = config_path
            instance._task = asyncio.create_task(instance.initialize())
        return instance

    async def initialize(self) -> None:
        """Initialize the MCPServerManager."""
        # Clear all states
        self.servers.clear()
        self.transports.clear()
        self.tool_to_server_map.clear()
        self.available_tools = []
        self.tool_infos = []
        self.temp_clients.clear()

        # Load and connect all servers
        await self.connect_all_servers()

    async def get_config(self) -> tuple[Config, dict[str, Any]]:
        """Get configuration and servers.

        Returns:
            A tuple of (config, servers).
        """
        try:
            with Path(self.config_path).open(encoding="utf-8") as f:
                config_content = f.read()
                config_dict = json.loads(config_content)
                config = Config(**config_dict)
                servers = {}  # Mock empty servers for now
                return config, servers
        except (OSError, json.JSONDecodeError) as error:
            logger.error("Error loading configuration: %s", error)
            return Config(mcpServers={}), {}

    async def connect_all_servers(self) -> list[ServerError]:
        """Connect to all enabled servers.

        Returns:
            A list of server errors.
        """
        error_array: list[ServerError] = []
        config, servers = await self.get_config()

        # Only connect enabled servers
        enabled_servers = [
            server_name
            for server_name, server_config in config.mcp_servers.items()
            if server_config.enabled
        ]
        logger.info("Connect to %d enabled servers...", len(enabled_servers))

        # Combine all environment variables
        all_enabled_specific_env = {}
        for server_name in enabled_servers:
            server_config = config.mcp_servers[server_name]
            if server_config.env:
                all_enabled_specific_env.update(server_config.env)

        # Connect to each server
        for server_name in enabled_servers:
            server_config = config.mcp_servers[server_name]
            result = await self.connect_single_server(
                server_name,
                server_config,
                all_enabled_specific_env,
            )
            if not result["success"]:
                error_array.append(
                    ServerError(
                        serverName=result["server_name"],
                        error=result["error"],
                    ),
                )

        logger.info("Connect all MCP servers completed")
        logger.info("All available tools:")
        for server_name, _client in self.servers.items():
            tool_info = next(
                (info for info in self.tool_infos if info.name == server_name),
                None,
            )
            if tool_info and tool_info.enabled:
                logger.info("%s:", server_name)
                for tool in tool_info.tools:
                    logger.info("  - %s", tool["name"])

        return error_array

    async def connect_single_server(
        self,
        server_name: str,
        config: ServerConfig,
        _all_specific_env: dict[str, str],
    ) -> dict[str, Any]:
        """Connect to a single server.

        Args:
            server_name: The name of the server.
            config: The server configuration.
            _all_specific_env: All environment variables from enabled servers.

        Returns:
            A dictionary with connection result.
        """
        try:
            updated_config = config.model_copy()
            if not updated_config.transport:
                updated_config.transport = "command"
                logger.debug(
                    "No transport specified for server %s, "
                    "defaulting to 'command' transport",
                    server_name,
                )

            # Create mock client and transport
            client = MockClient(server_name)
            transport = MockTransport(updated_config)
            temp_client = None

            # Store client and transport
            self.servers[server_name] = client
            self.transports[server_name] = transport
            if temp_client:
                self.temp_clients[server_name] = temp_client

            # Load server tools and capabilities
            response = await client.list_tools()
            capabilities = await client.get_server_capabilities()

            # Create tool information
            tools_ = []
            for tool in response["tools"]:
                tools_.append(
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                    },
                )

            # Update tool_infos
            self.tool_infos.append(
                McpTool(
                    name=server_name,
                    description=capabilities.get("description", ""),
                    tools=tools_,
                    enabled=config.enabled,
                    icon=capabilities.get("icon", ""),
                ),
            )

            # Load tools if server is enabled
            if config.enabled:
                langchain_tools = self._convert_to_openai_tools(response["tools"])
                self.available_tools.extend(langchain_tools)

                # Record tool to server mapping
                for tool in response["tools"]:
                    self.tool_to_server_map[tool["name"]] = client

            return {
                "success": True,
                "server_name": server_name,
            }
        except (ValueError, TypeError, KeyError) as error:
            # Use more specific exceptions instead of bare Exception
            logger.error("Error connecting to server %s: %s", server_name, error)
            return {
                "success": False,
                "server_name": server_name,
                "error": str(error),
            }

    def _convert_to_openai_tools(
        self,
        tools: list[dict[str, Any]],
    ) -> list[ToolDefinition]:
        """Convert MCP tools to OpenAI tools format.

        Args:
            tools: List of MCP tools.

        Returns:
            List of tools in OpenAI format.
        """
        result = []
        for tool in tools:
            function_def = FunctionDefinition(
                name=tool["name"],
                description=tool["description"],
                parameters={},
            )
            result.append(
                ToolDefinition(
                    type="function",
                    function=function_def,
                ),
            )
        return result

    async def sync_servers_with_config(self) -> list[ServerError]:
        """Sync servers with configuration.

        Returns:
            A list of server errors.
        """
        logger.info("Syncing servers with configuration...")
        try:
            await self.disconnect_all_servers()
            error_array = await self.connect_all_servers()
            logger.info("Server configuration sync completed")
            return error_array
        except (OSError, ValueError, RuntimeError) as error:
            logger.error("Error during server configuration sync: %s", error)
            raise

    async def disconnect_single_server(self, server_name: str) -> None:
        """Disconnect a single server.

        Args:
            server_name: The name of the server to disconnect.
        """
        try:
            client = self.servers.get(server_name)
            if client:
                # Get tools list before disconnecting
                try:
                    response = await client.list_tools()
                    tools_to_remove = {tool["name"] for tool in response["tools"]}

                    # Remove tools from available_tools
                    self.available_tools = [
                        tool
                        for tool in self.available_tools
                        if tool.function.name not in tools_to_remove
                    ]

                    # Clean up tool to server mapping
                    for tool_name in tools_to_remove:
                        if tool_name in self.tool_to_server_map:
                            self.tool_to_server_map.pop(tool_name)
                except (KeyError, AttributeError, ValueError) as error:
                    # Use more specific exceptions
                    logger.error(
                        "Error getting tools list for server %s: %s",
                        server_name,
                        error,
                    )

                # Close transport and clean up server
                transport = self.transports.get(server_name)
                if transport:
                    await transport.close()

                temp_client = self.temp_clients.get(server_name)
                if temp_client:
                    await temp_client.close()

                self.transports.pop(server_name, None)
                self.servers.pop(server_name, None)

                # Remove from tool_infos
                self.tool_infos = [
                    info for info in self.tool_infos if info.name != server_name
                ]

                logger.info("Server %s disconnected", server_name)
        except (OSError, RuntimeError) as error:
            logger.error("Error disconnecting server %s: %s", server_name, error)

    async def update_server_enabled_state(
        self,
        server_name: str,
        enabled: bool,
    ) -> None:
        """Update server enabled state.

        Args:
            server_name: The name of the server.
            enabled: Whether the server is enabled.
        """
        # Update enabled status in tool info
        tool_info = next(
            (info for info in self.tool_infos if info.name == server_name),
            None,
        )
        if not tool_info:
            logger.warning(
                "Cannot update state for server %s: tool info not found",
                server_name,
            )
            return

        tool_info.enabled = enabled

        # Get all tool names for this server
        server_tools = {tool["name"] for tool in tool_info.tools}

        # Update available_tools
        if enabled:
            # If enabling, add server's tools to available_tools
            client = self.servers.get(server_name)
            if client:
                response = await client.list_tools()
                langchain_tools = self._convert_to_openai_tools(response["tools"])
                self.available_tools.extend(langchain_tools)
        else:
            # If disabling, remove server's tools from available_tools
            self.available_tools = [
                tool
                for tool in self.available_tools
                if tool.function.name not in server_tools
            ]

    def check_properties_changed(self, server_name: str, config: ServerConfig) -> bool:
        """Check if server properties have changed.

        Args:
            server_name: The name of the server.
            config: The new server configuration.

        Returns:
            True if properties have changed, False otherwise.
        """
        client = self.servers.get(server_name)
        if not client:
            return True

        current_params = getattr(client.transport, "_server_params", None)
        if not current_params:
            return True

        # Check transport type changed
        if current_params.transport != config.transport:
            return True

        # If command transport, check command, args and env
        if config.transport == "command" and current_params.transport == "command":
            current_command = current_params.command or ""
            new_command = config.command or ""
            current_args = current_params.args or []
            new_args = config.args or []

            return (
                current_command != new_command
                or ",".join(current_args) != ",".join(new_args)
                or json.dumps(current_params.env or {}) != json.dumps(config.env or {})
            )

        # If sse or websocket transport, check url
        if config.transport in ("sse", "websocket") and current_params.transport in (
            "sse",
            "websocket",
        ):
            return current_params.url != config.url

        return True

    async def get_available_tools(self) -> list[ToolDefinition]:
        """Get available tools.

        Returns:
            A list of available tools.
        """
        return self.available_tools

    async def get_tool_infos(self) -> list[McpTool]:
        """Get tool information.

        Returns:
            A list of tool information.
        """
        return self.tool_infos

    async def get_tool_to_server_map(self) -> Mapping[str, Any]:
        """Get tool to server map.

        Returns:
            A mapping of tool names to server clients.
        """
        return self.tool_to_server_map

    async def disconnect_all_servers(self) -> None:
        """Disconnect all servers."""
        logger.info("Disconnect all MCP servers...")
        for server_name in list(self.servers.keys()):
            transport = self.transports.get(server_name)
            if transport:
                await transport.close()

        for server_name in list(self.temp_clients.keys()):
            logger.debug("[%s] Disconnecting temp client", server_name)
            client = self.temp_clients.get(server_name)
            if client:
                await client.close()

        self.servers.clear()
        self.transports.clear()
        self.tool_to_server_map.clear()
        self.available_tools = []
        self.tool_infos = []
        logger.info("Disconnect all MCP servers completed")


if __name__ == "__main__":
    asyncio.run(MCPServerManager.get_instance().initialize())
