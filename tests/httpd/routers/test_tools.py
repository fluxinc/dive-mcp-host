from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from dive_mcp_host.httpd.app import DiveHostAPI
from dive_mcp_host.httpd.dependencies import get_app
from dive_mcp_host.httpd.routers.models import SimpleToolInfo
from dive_mcp_host.httpd.routers.tools import McpTool, ToolsResult, tools


class MockTool:
    """Mock Tool class."""

    def __init__(self, name, description=None):
        self.name = name
        self.description = description


class MockServerInfo:
    """Mock server info class."""

    def __init__(self, tools=None, error=None):
        self.tools = tools or []
        self.error = error


class MockDiveHostAPI:
    """Mock DiveHostAPI."""

    def __init__(self, mcp_server_config=None, server_info=None, cached_tools=None):
        self.mcp_server_config_manager = MagicMock()
        self.mcp_server_config_manager.current_config = mcp_server_config

        self.dive_host = {"default": MagicMock(mcp_server_info=server_info or {})}

        self.local_file_cache = MagicMock()
        self.local_file_cache.get.return_value = cached_tools
        self.local_file_cache.set = MagicMock()


@pytest.fixture
def client():
    """Create a test client."""
    app = DiveHostAPI()
    app.include_router(tools)

    mock_app = MockDiveHostAPI(
        mcp_server_config=MagicMock(),
        server_info={
            "test_server": MockServerInfo(
                tools=[MockTool(name="test_tool", description="Test tool description")]
            )
        },
    )

    def get_mock_app():
        return mock_app

    app.dependency_overrides[get_app] = get_mock_app

    with TestClient(app) as client:
        yield client


def test_list_tools(client):
    """Test the GET endpoint."""
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK

    response_data = response.json()

    assert "success" in response_data
    assert isinstance(response_data["success"], bool)
    assert response_data["success"] is True
    assert "tools" in response_data
    assert isinstance(response_data["tools"], list)

    # If there are tools, check structure of first tool
    if response_data["tools"]:
        tool = response_data["tools"][0]
        assert isinstance(tool["name"], str)
        assert tool["name"] == "test_server"

        assert isinstance(tool["description"], str)

        assert isinstance(tool["enabled"], bool)
        assert tool["enabled"] is True

        assert "icon" in tool

        assert isinstance(tool["tools"], list)

        if tool["tools"]:
            subtool = tool["tools"][0]
            assert isinstance(subtool["name"], str)
            assert subtool["name"] == "test_tool"

            assert isinstance(subtool["description"], str)
            assert subtool["description"] == "Test tool description"


def test_tools_result_serialization():
    """Test that ToolsResult can be properly serialized."""
    # Create a sample response
    response = ToolsResult(
        success=True,
        message=None,
        tools=[
            McpTool(
                name="test_tool",
                tools=[SimpleToolInfo(name="test", description="Test function")],
                description="Test tool description",
                enabled=True,
                icon="test",
                error=None,
            ),
        ],
    )

    response_dict = response.model_dump(by_alias=True)

    assert isinstance(response_dict["success"], bool)
    assert response_dict["success"] is True
    assert isinstance(response_dict["tools"], list)

    if response_dict["tools"]:
        tool = response_dict["tools"][0]
        assert isinstance(tool["name"], str)
        assert tool["name"] == "test_tool"

        assert isinstance(tool["tools"], list)

        assert isinstance(tool["description"], str)
        assert tool["description"] == "Test tool description"

        assert isinstance(tool["enabled"], bool)
        assert tool["enabled"] is True

        assert isinstance(tool["icon"], str)
        assert tool["icon"] == "test"

        assert tool["error"] is None

        if tool["tools"]:
            subtool = tool["tools"][0]
            assert isinstance(subtool["name"], str)
            assert subtool["name"] == "test"

            assert isinstance(subtool["description"], str)
            assert subtool["description"] == "Test function"


@pytest.mark.asyncio
@patch("dive_mcp_host.httpd.routers.tools.list_tools")
async def test_list_tools_with_servers(mock_list_tools):
    """Test list_tools function with servers in configuration."""
    # Create Mock configuration
    mock_config = MagicMock()
    mock_config.mcp_servers = {"server1": {}, "server2": {}}

    # Create server information
    server_info = {
        "server1": MockServerInfo(
            tools=[
                MockTool(name="tool1", description="Tool 1 description"),
                MockTool(name="tool2", description="Tool 2 description"),
            ]
        )
    }

    # Create API instance
    app = MockDiveHostAPI(
        mcp_server_config=mock_config,
        server_info=server_info,
    )

    # Mock return value
    mock_list_tools.return_value = ToolsResult(
        success=True,
        message=None,
        tools=[
            McpTool(
                name="server1",
                tools=[
                    SimpleToolInfo(name="tool1", description="Tool 1 description"),
                    SimpleToolInfo(name="tool2", description="Tool 2 description"),
                ],
                description="",
                enabled=True,
                icon="",
                error=None,
            )
        ],
    )

    response = await mock_list_tools(app)

    assert response.success is True
    assert len(response.tools) >= 1

    server1 = next((tool for tool in response.tools if tool.name == "server1"), None)
    assert server1 is not None
    assert server1.enabled is True
    assert server1.description == ""
    assert server1.icon == ""
    assert server1.error is None

    assert len(server1.tools) == 2

    tool1 = next((tool for tool in server1.tools if tool.name == "tool1"), None)
    assert tool1 is not None
    assert tool1.description == "Tool 1 description"

    tool2 = next((tool for tool in server1.tools if tool.name == "tool2"), None)
    assert tool2 is not None
    assert tool2.description == "Tool 2 description"

    mock_list_tools.assert_called_once_with(app)


@pytest.mark.asyncio
@patch("dive_mcp_host.httpd.routers.tools.list_tools")
async def test_list_tools_with_missing_servers(mock_list_tools):
    """Test list_tools function with missing servers in configuration."""
    # Create Mock configuration
    mock_config = MagicMock()
    mock_config.mcp_servers = {"server1": {}, "server2": {}}

    # Create server information (only server1)
    server_info = {
        "server1": MockServerInfo(
            tools=[MockTool(name="tool1", description="Tool 1 description")]
        )
    }

    # Create cache tools (including server2)
    cached_tools_json = (
        '{"root": {"server2": {"name": "server2", "tools": [{"name": "cached_tool", '
        '"description": "Cached tool description"}], "description": "Cached server", '
        '"enabled": true, "icon": "cache", "error": null}}}'
    )

    # Create API instance
    app = MockDiveHostAPI(
        mcp_server_config=mock_config,
        server_info=server_info,
        cached_tools=cached_tools_json,
    )

    # Mock return value
    mock_list_tools.return_value = ToolsResult(
        success=True,
        message=None,
        tools=[
            McpTool(
                name="server1",
                tools=[SimpleToolInfo(name="tool1", description="Tool 1 description")],
                description="",
                enabled=True,
                icon="",
                error=None,
            ),
            McpTool(
                name="server2",
                tools=[
                    SimpleToolInfo(
                        name="cached_tool", description="Cached tool description"
                    )
                ],
                description="Cached server",
                enabled=True,
                icon="cache",
                error=None,
            ),
        ],
    )

    response = await mock_list_tools(app)

    assert response.success is True
    assert len(response.tools) >= 2
    assert any(tool.name == "server2" for tool in response.tools)

    mock_list_tools.assert_called_once_with(app)


@pytest.mark.asyncio
@patch("dive_mcp_host.httpd.routers.tools.list_tools")
async def test_list_tools_with_error(mock_list_tools):
    """Test list_tools function with server error."""
    # Create server information (with error)
    server_info = {
        "error_server": MockServerInfo(tools=[], error=Exception("Test error"))
    }

    # Create API instance
    app = MockDiveHostAPI(
        server_info=server_info,
    )

    # Mock return value
    mock_list_tools.return_value = ToolsResult(
        success=True,
        message=None,
        tools=[
            McpTool(
                name="error_server",
                tools=[],
                description="",
                enabled=True,
                icon="",
                error="Test error",
            ),
        ],
    )

    response = await mock_list_tools(app)

    assert response.success is True
    assert len(response.tools) == 1

    error_server = next((t for t in response.tools if t.name == "error_server"), None)
    assert error_server is not None
    assert error_server.name == "error_server"
    assert error_server.enabled is True
    assert error_server.description == ""
    assert error_server.icon == ""
    assert error_server.error == "Test error"
    assert error_server.tools == []

    mock_list_tools.assert_called_once_with(app)


@pytest.mark.asyncio
@patch("dive_mcp_host.httpd.routers.tools.list_tools")
async def test_list_tools_with_no_config(mock_list_tools):
    """Test list_tools function with no configuration."""
    # Create API instance with no configuration
    app = MockDiveHostAPI(
        mcp_server_config=None,
        server_info={},
    )

    # Mock return value
    mock_list_tools.return_value = ToolsResult(success=True, message=None, tools=[])

    # Call API endpoint test
    response = await mock_list_tools(app)

    # Verify results
    assert response.success is True
    assert isinstance(response.tools, list)

    # Verify mock was called
    mock_list_tools.assert_called_once_with(app)


@pytest.mark.asyncio
@patch("dive_mcp_host.httpd.routers.tools.list_tools")
async def test_list_tools_with_missing_server_not_in_cache(mock_list_tools):
    """Test list_tools function with missing server not in cache."""
    # Create Mock configuration
    mock_config = MagicMock()
    mock_config.mcp_servers = {"server1": {}, "missing_server": {}}

    # Create server information (only server1)
    server_info = {
        "server1": MockServerInfo(
            tools=[MockTool(name="tool1", description="Tool 1 description")]
        )
    }

    # Create cache tools (not including missing_server)
    cached_tools_json = (
        '{"root": {"server2": {"name": "server2", "tools": [], '
        '"description": "", "enabled": true, "icon": "", "error": null}}}'
    )

    # Create API instance
    app = MockDiveHostAPI(
        mcp_server_config=mock_config,
        server_info=server_info,
        cached_tools=cached_tools_json,
    )

    # Mock return value
    mock_list_tools.return_value = ToolsResult(
        success=True,
        message=None,
        tools=[
            McpTool(
                name="server1",
                tools=[SimpleToolInfo(name="tool1", description="Tool 1 description")],
                description="",
                enabled=True,
                icon="",
                error=None,
            ),
            McpTool(
                name="missing_server",
                tools=[],
                description="",
                enabled=False,
                icon="",
                error=None,
            ),
        ],
    )

    response = await mock_list_tools(app)

    assert response.success is True
    assert len(response.tools) == 2

    server1 = next((t for t in response.tools if t.name == "server1"), None)
    assert server1 is not None
    assert server1.enabled is True
    assert server1.description == ""
    assert server1.icon == ""
    assert server1.error is None

    assert len(server1.tools) == 1
    tool1 = server1.tools[0]
    assert tool1.name == "tool1"
    assert tool1.description == "Tool 1 description"

    missing_server = next(
        (t for t in response.tools if t.name == "missing_server"), None
    )
    assert missing_server is not None
    assert missing_server.enabled is False
    assert missing_server.tools == []
    assert missing_server.description == ""
    assert missing_server.icon == ""
    assert missing_server.error is None

    mock_list_tools.assert_called_once_with(app)


def test_empty_tools_result():
    """Test empty ToolsResult serialization."""
    response = ToolsResult(success=True, message=None, tools=[])
    response_dict = response.model_dump(by_alias=True)

    assert "success" in response_dict
    assert response_dict["success"] is True
    assert "tools" in response_dict
    assert isinstance(response_dict["tools"], list)
    assert len(response_dict["tools"]) == 0

    assert "message" in response_dict
    assert response_dict["message"] is None
