from fastapi import status

from dive_mcp_host.host.conf import LLMConfig
from dive_mcp_host.httpd.routers.config import SaveModelSettingsRequest
from dive_mcp_host.httpd.routers.models import McpServerConfig, ModelConfig
from tests import helper

# Mock data
MOCK_MCP_CONFIG = {
    "default": McpServerConfig(
        transport="command",  # type: ignore  # Test backward compatibility
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
        "openai": LLMConfig(
            model="gpt-4o",
            modelProvider="openai",
            apiKey="sk-mock-key",
            temperature=0.7,
            topP=0.9,
            maxTokens=4000,
            configuration=None,
        )
    },
)

# Constants
SUCCESS_CODE = status.HTTP_200_OK
BAD_REQUEST_CODE = status.HTTP_400_BAD_REQUEST
TEST_PROVIDER = "openai"


def test_get_mcp_server(test_client):
    """Test the /api/config/mcpserver GET endpoint."""
    client, _ = test_client
    # Send request
    response = client.get("/api/config/mcpserver")

    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "message": None,
            "config": {
                "mcpServers": {
                    "echo": {
                        "transport": "stdio",
                        "enabled": True,
                        "command": "python3",
                        "args": [
                            "-m",
                            "dive_mcp_host.host.tools.echo",
                            "--transport=stdio",
                        ],
                        "env": {"NODE_ENV": "production"},
                        "url": None,
                    },
                },
            },
        },
    )


def test_post_mcp_server(test_client):
    """Test the /api/config/mcpserver POST endpoint."""
    client, _ = test_client
    # Prepare request data - convert McpServerConfig objects to dict
    mock_server_dict = {}
    for key, value in MOCK_MCP_CONFIG.items():
        mock_server_dict[key] = value.model_dump()

    server_data = {"mcpServers": mock_server_dict}

    response = client.post(
        "/api/config/mcpserver",
        json=server_data,
    )
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
        },
    )

    response = client.get("/api/config/mcpserver")
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "message": None,
            "config": {
                "mcpServers": {
                    "default": {
                        "transport": "stdio",
                        "enabled": True,
                        "command": "node",
                        "args": [
                            "./mcp-server.js",
                        ],
                        "env": {"NODE_ENV": "production"},
                        "url": None,
                    },
                },
            },
        },
    )


def test_get_model(test_client):
    """Test the /api/config/model GET endpoint."""
    client, _ = test_client
    # Send request
    response = client.get("/api/config/model")

    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "message": None,
            "config": {
                "activeProvider": "dive",
                "enableTools": True,
                "configs": {
                    "dive": {
                        "model": "fake",
                        "modelProvider": "dive",
                        "temperature": 0.0,
                        "topP": None,
                        "maxTokens": None,
                        "configuration": {},
                    },
                },
            },
        },
    )


def test_post_model(test_client):
    """Test the /api/config/model POST endpoint."""
    client, _ = test_client
    # Prepare request data
    model_settings = SaveModelSettingsRequest(
        provider=TEST_PROVIDER,
        modelSettings=LLMConfig(
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
        "/api/config/model",
        json=model_settings.model_dump(by_alias=True),
    )

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "message": None,
        },
    )
    response = client.get("/api/config/model")
    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "message": None,
            "config": {
                "activeProvider": TEST_PROVIDER,
                "enableTools": True,
                "configs": {
                    TEST_PROVIDER: {
                        "model": "gpt-4o-mini",
                        "modelProvider": TEST_PROVIDER,
                        "apiKey": "openai-api-key",
                        "temperature": 0.8,
                        "topP": 0.9,
                        "maxTokens": 8000,
                        "configuration": None,
                    },
                    "dive": {
                        "model": "fake",
                        "modelProvider": "dive",
                        "temperature": 0.0,
                        "topP": None,
                        "maxTokens": None,
                        "configuration": {},
                    },
                },
            },
        },
    )


def test_post_model_replace_all(test_client):
    """Test the /api/config/model/replaceAll POST endpoint."""
    client, _ = test_client
    model_config_data = MOCK_MODEL_CONFIG.model_dump(by_alias=True)

    response = client.post(
        "/api/config/model/replaceAll",
        json=model_config_data,
    )

    assert response.status_code == SUCCESS_CODE

    response_data = response.json()

    helper.dict_subset(
        response_data,
        {
            "success": True,
            "message": None,
        },
    )
    response = client.get("/api/config/model")
    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "message": None,
            "config": {
                "activeProvider": "openai",
                "enableTools": True,
                "configs": {
                    "openai": {
                        "model": "gpt-4o",
                        "modelProvider": "openai",
                        "apiKey": "sk-mock-key",
                        "temperature": 0.7,
                        "topP": 0.9,
                        "maxTokens": 4000,
                        "configuration": None,
                    },
                },
            },
        },
    )


def test_get_custom_rules(test_client):
    """Test the /api/config/customrules GET endpoint."""
    client, _ = test_client
    # Send request
    response = client.get("/api/config/customrules")

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Parse JSON response
    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "message": None,
            "rules": "testCustomrules",
        },
    )


def test_post_custom_rules(test_client):
    """Test the /api/config/customrules POST endpoint."""
    client, _ = test_client
    custom_rules = "# New Custom Rules\n1. Be concise\n2. Use simple language"

    # Send request - This endpoint expects raw text, not JSON
    response = client.post(
        "/api/config/customrules",
        content=custom_rules.encode("utf-8"),
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "message": None,
        },
    )

    response = client.get("/api/config/customrules")
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "message": None,
            "rules": custom_rules,
        },
    )
