import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from dive_mcp_host.httpd.conf.mcpserver_config.manager import (
    Config,
    FunctionDefinition,
    MCPServerManager,
    McpTool,
    ServerConfig,
    ToolDefinition,
)

# Register custom mark
integration = pytest.mark.integration


# Unit tests
class TestMCPServerManager:
    """Unit tests for MCPServerManager class's basic functionality."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton instance before each test."""
        # Reset the singleton instance
        MCPServerManager._instance = None  # noqa: SLF001
        yield
        # Clean up after test
        MCPServerManager._instance = None  # noqa: SLF001

    @pytest.fixture
    def mock_config_file(self):
        """Create a mock configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "mcpServers": {
                        "test_server": {
                            "transport": "command",
                            "enabled": True,
                            "command": "test_cmd",
                            "args": ["--test"],
                            "env": {"TEST_ENV": "test_value"},
                        },
                    },
                },
                f,
            )
            config_path = f.name
        yield config_path
        # Clean up after test
        Path(config_path).unlink()

    def test_singleton_pattern(self):
        """Test if the singleton pattern works properly."""
        instance1 = MCPServerManager.get_instance()
        instance2 = MCPServerManager.get_instance()
        assert instance1 is instance2

    @pytest.mark.asyncio
    async def test_config_path_setting(self):
        """Test if the configuration path is set correctly."""
        test_path = "/test/path.json"
        # Mock initialize to avoid actual initialization
        with patch.object(MCPServerManager, "initialize", return_value=None):
            manager = MCPServerManager.get_instance(test_path)
            assert manager.config_path == test_path

    @pytest.mark.asyncio
    async def test_get_config(self, mock_config_file):
        """Test retrieving server configuration."""
        # Mock initialize to avoid actual initialization
        with patch.object(MCPServerManager, "initialize", return_value=None):
            manager = MCPServerManager.get_instance(mock_config_file)
            config, servers = await manager.get_config()
            assert config is not None
            assert isinstance(config, Config)
            assert "test_server" in config.mcp_servers
            assert config.mcp_servers["test_server"].transport == "command"

    @pytest.mark.asyncio
    @patch("dive_mcp_host.httpd.conf.mcpserver_config.manager.MockClient")
    @patch("dive_mcp_host.httpd.conf.mcpserver_config.manager.MockTransport")
    async def test_connect_single_server(
        self,
        mock_transport_class,
        mock_client_class,
        mock_config_file,
    ):
        """Test connecting to a single server."""
        # Setup mocks - using AsyncMock for async methods
        mock_client = AsyncMock()
        mock_client.list_tools.return_value = {
            "tools": [{"name": "test_tool", "description": "Test tool"}],
        }
        mock_client.get_server_capabilities.return_value = {
            "description": "Test server",
            "icon": "test-icon",
        }
        mock_client_class.return_value = mock_client

        mock_transport = AsyncMock()
        mock_transport_class.return_value = mock_transport

        # Setup manager with initialize mocked
        with patch.object(MCPServerManager, "initialize", return_value=None):
            manager = MCPServerManager.get_instance(mock_config_file)
            config, _ = await manager.get_config()
            server_config = config.mcp_servers["test_server"]

            # Reset manager state to ensure clean test
            manager.servers.clear()
            manager.transports.clear()
            manager.tool_to_server_map.clear()
            manager.available_tools = []
            manager.tool_infos = []

            # Test connect_single_server
            result = await manager.connect_single_server(
                "test_server",
                server_config,
                {},
            )

            assert result["success"] is True
            assert result["server_name"] == "test_server"
            assert "test_server" in manager.servers
            assert "test_server" in manager.transports
            assert manager.tool_infos[0].name == "test_server"
            assert manager.tool_infos[0].enabled is True
            assert len(manager.available_tools) > 0
            assert "test_tool" in manager.tool_to_server_map

    @pytest.mark.asyncio
    @patch("dive_mcp_host.httpd.conf.mcpserver_config.manager.MockClient")
    @patch("dive_mcp_host.httpd.conf.mcpserver_config.manager.MockTransport")
    async def test_disconnect_single_server(
        self,
        mock_transport_class,
        mock_client_class,
        mock_config_file,
    ):
        """Test disconnecting from a single server."""
        # Setup mocks - using AsyncMock for async methods
        mock_client = AsyncMock()
        mock_client.list_tools.return_value = {
            "tools": [{"name": "test_tool", "description": "Test tool"}],
        }
        mock_client.get_server_capabilities.return_value = {
            "description": "Test server",
            "icon": "test-icon",
        }
        mock_client_class.return_value = mock_client

        mock_transport = AsyncMock()
        mock_transport_class.return_value = mock_transport

        # Setup manager with initialize mocked
        with patch.object(MCPServerManager, "initialize", return_value=None):
            manager = MCPServerManager.get_instance(mock_config_file)
            config, _ = await manager.get_config()
            server_config = config.mcp_servers["test_server"]

            # Reset manager state to ensure clean test
            manager.servers.clear()
            manager.transports.clear()
            manager.tool_to_server_map.clear()
            manager.available_tools = []
            manager.tool_infos = []

            # Connect first
            await manager.connect_single_server("test_server", server_config, {})

            # Test disconnect
            await manager.disconnect_single_server("test_server")

            assert "test_server" not in manager.servers
            assert "test_server" not in manager.transports
            assert not any(info.name == "test_server" for info in manager.tool_infos)
            assert all(
                tool.function.name != "test_tool" for tool in manager.available_tools
            )
            assert "test_tool" not in manager.tool_to_server_map

    @pytest.mark.asyncio
    async def test_get_available_tools(self, mock_config_file):
        """Test getting available tools."""
        # Mock initialize to avoid actual initialization
        with patch.object(MCPServerManager, "initialize", return_value=None):
            manager = MCPServerManager.get_instance(mock_config_file)
            # Reset state to ensure clean test
            manager.available_tools = []
            tools = await manager.get_available_tools()
            assert isinstance(tools, list)
            # Initially empty before connecting to servers
            assert len(tools) == 0

    @pytest.mark.asyncio
    async def test_get_tool_infos(self, mock_config_file):
        """Test getting tool information."""
        # Mock initialize to avoid actual initialization
        with patch.object(MCPServerManager, "initialize", return_value=None):
            manager = MCPServerManager.get_instance(mock_config_file)
            # Reset state to ensure clean test
            manager.tool_infos = []
            tool_infos = await manager.get_tool_infos()
            assert isinstance(tool_infos, list)
            # Initially empty before connecting to servers
            assert len(tool_infos) == 0

    @pytest.mark.asyncio
    async def test_get_tool_to_server_map(self, mock_config_file):
        """Test getting tool to server mapping."""
        # Mock initialize to avoid actual initialization
        with patch.object(MCPServerManager, "initialize", return_value=None):
            manager = MCPServerManager.get_instance(mock_config_file)
            # Reset state to ensure clean test
            manager.tool_to_server_map = {}
            tool_map = await manager.get_tool_to_server_map()
            assert isinstance(tool_map, dict)
            # Initially empty before connecting to servers
            assert len(tool_map) == 0

    @pytest.mark.asyncio
    @patch(
        "dive_mcp_host.httpd.conf.mcpserver_config.manager.MCPServerManager.connect_all_servers",
    )
    async def test_initialize(self, mock_connect_all, mock_config_file):
        """Test manager initialization."""
        mock_connect_all.return_value = []

        # Create a new instance for this test to avoid conflicts with fixture
        MCPServerManager._instance = None  # noqa: SLF001

        # Setup the manager without mocking initialize
        manager = MCPServerManager(mock_config_file)
        # This should call the real connect_all_servers which is mocked
        await manager.initialize()

        # Verify that connect_all_servers was called
        mock_connect_all.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "dive_mcp_host.httpd.conf.mcpserver_config.manager.MCPServerManager.disconnect_all_servers",
    )
    @patch(
        "dive_mcp_host.httpd.conf.mcpserver_config.manager.MCPServerManager.connect_all_servers",
    )
    async def test_sync_servers_with_config(
        self,
        mock_connect_all,
        mock_disconnect_all,
        mock_config_file,
    ):
        """Test syncing servers with configuration."""
        mock_connect_all.return_value = []

        # Mock initialize to avoid actual initialization
        with patch.object(MCPServerManager, "initialize", return_value=None):
            manager = MCPServerManager.get_instance(mock_config_file)
            result = await manager.sync_servers_with_config()

            mock_disconnect_all.assert_called_once()
            mock_connect_all.assert_called_once()
            assert isinstance(result, list)
            assert len(result) == 0


# Add custom mark to pytest.ini to avoid warnings
def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line("markers", "integration: mark test as integration test")


# Integration tests
@pytest.mark.integration
class TestMCPServerManagerIntegration:
    """Integration tests for the complete functionality of MCPServerManager class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton instance before each test."""
        # Reset the singleton instance
        MCPServerManager._instance = None  # noqa: SLF001
        yield
        # Clean up after test
        MCPServerManager._instance = None  # noqa: SLF001

    @pytest_asyncio.fixture
    async def test_config_path(self):
        """Set up the test configuration file path."""
        # Use a temporary directory to create the config file
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            test_config = {
                "mcpServers": {
                    "test_server": {
                        "transport": "command",
                        "enabled": True,
                        "command": "echo",
                        "args": ["hello"],
                    },
                },
            }
            with config_path.open("w") as f:
                json.dump(test_config, f)
            yield str(config_path)

    @pytest.mark.asyncio
    @patch(
        "dive_mcp_host.httpd.conf.mcpserver_config.manager.MockClient",
        new=AsyncMock,
    )
    @patch(
        "dive_mcp_host.httpd.conf.mcpserver_config.manager.MockTransport",
        new=AsyncMock,
    )
    async def test_full_workflow(self, test_config_path):
        """Test the complete server connection and tool management workflow."""
        # Create a new instance to avoid conflicts
        MCPServerManager._instance = None  # noqa: SLF001

        # Initialize the manager with the test config
        manager = MCPServerManager(test_config_path)

        # Set up mocks for methods used in initialize
        with (
            patch.object(manager, "get_config") as mock_get_config,
            patch.object(manager, "connect_single_server") as mock_connect,
        ):
            # Mock the config
            mock_get_config.return_value = (
                Config(
                    mcpServers={
                        "test_server": ServerConfig(transport="command", enabled=True),
                    },
                ),
                {},
            )

            # Mock the connect result
            mock_connect.return_value = {
                "success": True,
                "server_name": "test_server",
            }

            # Mock the client and capabilities
            mock_tool = {"name": "test_tool", "description": "Test tool"}
            manager.tool_infos = [
                McpTool(
                    name="test_server",
                    tools=[mock_tool],
                    description="Test server",
                    enabled=True,
                    icon="test-icon",
                ),
            ]
            manager.servers = {"test_server": AsyncMock()}
            manager.transports = {"test_server": AsyncMock()}
            manager.available_tools = [
                ToolDefinition(
                    type="function",
                    function=FunctionDefinition(
                        name="test_tool",
                        description="Test tool",
                        parameters={},
                    ),
                ),
            ]
            manager.tool_to_server_map = {"test_tool": AsyncMock()}

            # Test the update_server_enabled_state method
            with patch.object(
                manager.servers["test_server"],
                "list_tools",
            ) as mock_list_tools:
                mock_list_tools.return_value = {"tools": [mock_tool]}
                await manager.update_server_enabled_state("test_server", False)

            # Verify that the server is disabled
            assert manager.tool_infos[0].enabled is False

            # Test the disconnect_all_servers method - directly check state change
            # Store initial state
            initial_servers = manager.servers.copy()
            initial_transports = manager.transports.copy()

            # Call disconnect_all_servers
            await manager.disconnect_all_servers()

            # Verify that all servers are disconnected by checking the state
            assert len(manager.servers) == 0
            assert len(manager.transports) == 0
            assert len(manager.available_tools) == 0
            assert len(manager.tool_infos) == 0

            # Make sure we had servers before to make the test meaningful
            assert len(initial_servers) > 0
            assert len(initial_transports) > 0
