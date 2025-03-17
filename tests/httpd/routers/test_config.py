import httpx
import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from dive_mcp_host.httpd.routers.config import (
    ModelSettings,
    SaveModelSettingsRequest,
    config,
)
from dive_mcp_host.httpd.routers.models import McpServerConfig, ModelConfig

# Mock data
MOCK_MCP_CONFIG = {
    "default": McpServerConfig(
        transport="command",
        enabled=True,
        command="node",
        args=["./mcp-server.js"],
        env={"NODE_ENV": "production"},
        url=None,
    ),
}

MOCK_MODEL_CONFIG = ModelConfig(
    activeProvider="openai",
    enableTools=True,
    configs={
        "openai": ModelSettings(
            model="gpt-4",
            modelProvider="openai",
            apiKey="sk-mock-key",
            temperature=0.7,
            topP=0.9,
            maxTokens=4000,
            configuration=None,
        ),
        "anthropic": ModelSettings(
            model="claude-3-opus-20240229",
            modelProvider="anthropic",
            apiKey="sk-ant-mock-key",
            temperature=0.5,
            topP=0.8,
            maxTokens=4000,
            configuration=None,
        ),
    },
)

# Constants
SUCCESS_CODE = status.HTTP_200_OK
BAD_REQUEST_CODE = status.HTTP_400_BAD_REQUEST
TEST_PROVIDER = "openai"


client_type = "fastapi"


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
    app.include_router(config)
    return TestClient(app)


def test_get_mcp_server(client):
    """Test the /config/mcpserver GET endpoint."""
    # Send request
    response = client.get("/config/mcpserver")

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert "success" in response_data
    assert response_data["success"] is True

    if "config" in response_data:
        config_data = response_data["config"]

        if "mcpServers" in config_data:
            assert isinstance(config_data["mcpServers"], dict)

    # Validate server configuration structure
    config_data = response_data["config"]
    assert "mcpServers" in config_data
    assert isinstance(config_data["mcpServers"], dict)

    # Validate server data
    server_data = config_data["mcpServers"]
    server_keys = list(server_data.keys())
    assert len(server_keys) > 0
    default_server = server_data[server_keys[0]]
    assert "transport" in default_server
    assert "enabled" in default_server
    assert "command" in default_server
    assert "args" in default_server
    assert "env" in default_server


def test_post_mcp_server(client):
    """Test the /config/mcpserver POST endpoint."""
    # Prepare request data - convert McpServerConfig objects to dict
    mock_server_dict = {}
    for key, value in MOCK_MCP_CONFIG.items():
        mock_server_dict[key] = value.model_dump()

    server_data = {"mcpServers": mock_server_dict}

    # Send request
    response = client.post(
        "/config/mcpserver",
        json=server_data,
    )

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert "success" in response_data
    assert response_data["success"] is True
    assert "errors" in response_data
    assert isinstance(response_data["errors"], list)


def test_get_model(client):
    """Test the /config/model GET endpoint."""
    # Send request
    response = client.get("/config/model")

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert "success" in response_data
    assert response_data["success"] is True
    assert "config" in response_data

    # Validate model configuration structure
    config_data = response_data["config"]
    assert "activeProvider" in config_data
    assert "enableTools" in config_data
    assert "configs" in config_data
    assert isinstance(config_data["configs"], dict)

    # Validate provider configurations
    provider_configs = config_data["configs"]
    for provider in provider_configs:
        provider_config = provider_configs[provider]
        assert "model" in provider_config
        assert "modelProvider" in provider_config
        assert "temperature" in provider_config


def test_post_model(client):
    """Test the /config/model POST endpoint."""
    # Prepare request data
    model_settings = SaveModelSettingsRequest(
        provider=TEST_PROVIDER,
        modelSettings=ModelSettings(
            model="gpt-4o-mini",
            modelProvider=TEST_PROVIDER,
            apiKey="openai-api-key",
            temperature=0.8,
            topP=0.9,
            maxTokens=8000,
            configuration=None,
        ),
        enableTools=True,
    )

    # Send request
    response = client.post(
        "/config/model",
        json=model_settings.model_dump(by_alias=True),
    )

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert "success" in response_data
    assert response_data["success"] is True


def test_post_model_replace_all(client):
    """Test the /config/model/replaceAll POST endpoint."""
    # Prepare request data - use the mock model config for simplicity
    model_config_data = MOCK_MODEL_CONFIG.model_dump(by_alias=True)

    # Send request
    response = client.post(
        "/config/model/replaceAll",
        json=model_config_data,
    )

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert "success" in response_data
    assert response_data["success"] is True

def test_get_custom_rules(client):
    """Test the /config/customrules GET endpoint."""
    # Send request
    response = client.get("/config/customrules")

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert "success" in response_data
    assert response_data["success"] is True
    assert "rules" in response_data
    assert isinstance(response_data["rules"], str)



def test_post_custom_rules(client):
    """Test the /config/customrules POST endpoint."""
    # Prepare custom rules data
    custom_rules = "# New Custom Rules\n1. Be concise\n2. Use simple language"

    # Send request - This endpoint expects raw text, not JSON
    response = client.post(
        "/config/customrules",
        content=custom_rules.encode("utf-8"),
        headers={"Content-Type": "text/plain"},
    )

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert "success" in response_data
    assert response_data["success"] is True
