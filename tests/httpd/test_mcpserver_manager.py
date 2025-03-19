import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio

from dive_mcp_host.httpd.conf.mcpserver.manager import (
    Config,
    MCPServerManager,
    ServerConfig,
)

# Register custom mark
integration = pytest.mark.integration


# Unit tests
class TestMCPServerManager:
    """Unit tests for MCPServerManager class's basic functionality."""

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
                            "command": "test_command",
                            "args": ["--test"],
                            "env": {"TEST_ENV": "test_value"},
                        },
                        "disabled_server": {
                            "transport": "sse",
                            "enabled": False,
                            "url": "http://test.url",
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

    def test_config_path_setting(self):
        """Test if the configuration path is set correctly."""
        test_path = "/test/path.json"
        manager = MCPServerManager.get_instance(test_path)
        assert manager.config_path == test_path

    @pytest.mark.asyncio
    async def test_get_config(self, mock_config_file):
        """Test retrieving MCP server configuration."""
        manager = MCPServerManager.get_instance(mock_config_file)
        config = await manager.get_config()
        assert config is not None
        assert "test_server" in config.mcp_servers
        assert config.mcp_servers["test_server"].transport == "command"

    @pytest.mark.asyncio
    async def test_initialize(self, mock_config_file):
        """Test initializing the manager."""
        manager = MCPServerManager.get_instance(mock_config_file)
        result = await manager.initialize()

        assert result is True
        assert manager.current_config is not None
        assert "test_server" in manager.current_config.mcp_servers
        assert manager.current_config.mcp_servers["test_server"].enabled is True
        assert manager.current_config.mcp_servers["disabled_server"].enabled is False

    @pytest.mark.asyncio
    async def test_get_enabled_servers(self, mock_config_file):
        """Test getting enabled servers."""
        manager = MCPServerManager.get_instance(mock_config_file)
        await manager.initialize()
        enabled_servers = await manager.get_enabled_servers()

        assert enabled_servers is not None
        assert len(enabled_servers) == 1
        assert "test_server" in enabled_servers
        assert "disabled_server" not in enabled_servers

    @pytest.mark.asyncio
    @patch("dive_mcp_host.httpd.conf.mcpserver.manager.json.dump")
    async def test_update_all_configs(self, mock_json_dump, mock_config_file):
        """Test updating all configurations."""
        # Mock json.dump to avoid writing to file
        mock_json_dump.return_value = None

        manager = MCPServerManager.get_instance(mock_config_file)
        await manager.initialize()

        # Create new configuration
        new_config = Config(
            mcpServers={
                "new_server": ServerConfig(
                    transport="websocket",
                    enabled=True,
                    url="ws://new.url",
                ),
            },
        )

        # Update all configurations
        result = await manager.update_all_configs(new_config)
        assert result is True

        # Manager's current_config should be updated
        assert manager.current_config is not None
        assert len(manager.current_config.mcp_servers) == 1
        assert "new_server" in manager.current_config.mcp_servers
        assert manager.current_config.mcp_servers["new_server"].transport == "websocket"
        assert manager.current_config.mcp_servers["new_server"].url == "ws://new.url"


# Integration tests
@pytest.mark.integration
class TestMCPServerManagerIntegration:
    """Integration tests for the complete functionality of MCPServerManager class."""

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
                        "args": ["Hello, MCP!"],
                        "env": {},
                    },
                },
            }
            with config_path.open("w") as f:
                json.dump(test_config, f)
            yield str(config_path)

    @pytest.mark.asyncio
    @patch("dive_mcp_host.httpd.conf.mcpserver.manager.json.dump")
    async def test_full_config_workflow(self, mock_json_dump, test_config_path):
        """Test the complete server configuration workflow."""
        # Mock json.dump to avoid serialization issues
        mock_json_dump.return_value = None

        # Set up MCPServerManager instance
        manager = MCPServerManager.get_instance(test_config_path)

        # Initialize the manager
        result = await manager.initialize()
        assert result is True
        assert manager.current_config is not None
        assert "test_server" in manager.current_config.mcp_servers

        # Get enabled servers
        enabled_servers = await manager.get_enabled_servers()
        assert "test_server" in enabled_servers

        # Create a new configuration with additional server
        current_servers = manager.current_config.mcp_servers.copy()
        current_servers["new_server"] = ServerConfig(
            transport="sse",
            enabled=True,
            url="http://new.server",
        )

        # Update config with new server
        new_config = Config(mcpServers=current_servers)
        result = await manager.update_all_configs(new_config)
        assert result is True

        # Verify new server was added in current_config
        assert manager.current_config is not None
        assert "new_server" in manager.current_config.mcp_servers
        assert manager.current_config.mcp_servers["new_server"].transport == "sse"
        assert (
            manager.current_config.mcp_servers["new_server"].url == "http://new.server"
        )

        # Test environment variable config
        with patch.dict(
            os.environ, {"DIVE_MCP_CONFIG_CONTENT": json.dumps({"mcpServers": {}})}
        ):
            env_manager = MCPServerManager()
            env_config = await env_manager.get_config()
            assert env_config is not None
            assert len(env_config.mcp_servers) == 0
