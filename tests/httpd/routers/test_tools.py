import httpx
import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from dive_mcp_host.httpd.routers.tools import McpTool, ToolsResult, tools

client_type = "fastapi"


class MockMcpServerManager:
    """Mock MCP server manager."""

    async def get_tool_infos(self):
        """Return mock tool infos."""
        return [
            McpTool(
                name="test_tool",
                tools=[
                    {
                        "name": "test",
                        "description": "Test function",
                    },
                ],
                description="Test tool description",
                enabled=True,
                icon="test",
            )
        ]


@pytest.fixture
def client(request):
    """Create a test client.

    Args:
        request: The pytest request object.

    Returns:
        A TestClient for FastAPI testing or httpx.Client for direct Node.js testing.
    """
    client_type = getattr(request.module, "client_type", "fastapi")

    if client_type == "nodejs":
        return httpx.Client(base_url="http://localhost:4321/api")
    app = FastAPI()
    app.include_router(tools)

    # Add mock MCP server manager
    app.state.mcp = MockMcpServerManager()

    return TestClient(app)


def test_list_tools(client):
    """Test the /tools GET endpoint."""
    response = client.get("/tools")

    # Verify response status code
    assert response.status_code == status.HTTP_200_OK

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert "success" in response_data
    assert isinstance(response_data["success"], bool)
    assert response_data["success"] is True
    assert "tools" in response_data
    assert isinstance(response_data["tools"], list)

    # If there are tools, check structure of first tool
    if response_data["tools"]:
        tool = response_data["tools"][0]
        assert "name" in tool
        assert isinstance(tool["name"], str)
        assert "description" in tool
        assert isinstance(tool["description"], str)
        assert "enabled" in tool
        assert isinstance(tool["enabled"], bool)
        assert "icon" in tool

        assert "tools" in tool
        assert isinstance(tool["tools"], list)

        if tool["tools"]:
            subtool = tool["tools"][0]
            assert "name" in subtool
            assert isinstance(subtool["name"], str)
            assert "description" in subtool
            assert isinstance(subtool["description"], str)


def test_tools_result_serialization():
    """Test that ToolsResult can be properly serialized."""
    # Create a sample response
    response = ToolsResult(
        success=True,
        message=None,
        tools=[
            McpTool(
                name="test_tool",
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "test",
                            "description": "Test function",
                            "parameters": {},
                        },
                    },
                ],
                description="Test tool description",
                enabled=True,
                icon="test",
            ),
        ],
    )

    # Convert to dict
    response_dict = response.model_dump(by_alias=True)

    # Validate structure
    assert "success" in response_dict
    assert isinstance(response_dict["success"], bool)
    assert "tools" in response_dict
    assert isinstance(response_dict["tools"], list)

    # Validate tool structure
    if response_dict["tools"]:
        tool = response_dict["tools"][0]
        assert "name" in tool
        assert isinstance(tool["name"], str)
        assert "tools" in tool
        assert isinstance(tool["tools"], list)
        assert "description" in tool
        assert isinstance(tool["description"], str)
        assert "enabled" in tool
        assert isinstance(tool["enabled"], bool)
        assert "icon" in tool
        assert isinstance(tool["icon"], str)
