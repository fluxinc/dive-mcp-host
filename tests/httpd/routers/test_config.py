import json
import os
from typing import TYPE_CHECKING

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from dive_mcp_host.httpd.routers.config import SaveModelSettingsRequest
from dive_mcp_host.httpd.routers.models import McpServerConfig, ModelFullConfigs

if TYPE_CHECKING:
    from dive_mcp_host.httpd.server import DiveHostAPI

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

MOCK_MODEL_CONFIG = ModelFullConfigs.model_validate(
    {
        "activeProvider": "openai",
        "enableTools": True,
        "configs": {
            "openai": {
                "active": True,
                "checked": False,
                "model": "gpt-4o",
                "modelProvider": "openai",
                "apiKey": "sk-mock-key",
                "maxTokens": 4000,
                "configuration": {
                    "temperature": 0.7,
                    "topP": 0.9,
                },
            }
        },
    }
)

MOCK_MODEL_CONFIG_WITH_NONE_PROVIDER = ModelFullConfigs.model_validate(
    {
        "activeProvider": "none",
        "enableTools": True,
        "configs": {
            "openai": {
                "active": True,
                "checked": False,
                "model": "gpt-4o",
                "modelProvider": "openai",
                "apiKey": "sk-mock-key",
                "maxTokens": 4000,
                "configuration": {
                    "temperature": 0.7,
                    "topP": 0.9,
                },
            }
        },
    }
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


def test_post_mcp_server_omit_env_url(test_client):
    """Test the /api/config/mcpserver POST endpoint.

    The url and env can be omitted for stdio transport.
    """
    client, _ = test_client
    # Prepare request data - convert McpServerConfig objects to dict
    server_data = {
        "mcpServers": {
            "default": {
                "transport": "stdio",  # type: ignore  # Test backward compatibility
                "enabled": True,
                "command": "uvx",
                "args": ["dive_mcp_host.host.tools.echo", "--transport=stdio"],
            },
        }
    }

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
                        "command": "uvx",
                        "args": [
                            "dive_mcp_host.host.tools.echo",
                            "--transport=stdio",
                        ],
                        "env": {},
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


def test_post_mcp_server_with_force(test_client: tuple[TestClient, "DiveHostAPI"]):
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
        params={"force": 1},
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
                        "configuration": {
                            "temperature": 0.0,
                            "topP": None,
                        },
                        "model": "fake",
                        "modelProvider": "dive",
                        "maxTokens": None,
                    },
                },
            },
        },
    )


def test_post_model(test_client: tuple[TestClient, "DiveHostAPI"]):
    """Test the /api/config/model POST endpoint."""
    app: DiveHostAPI
    client, app = test_client
    # Prepare request data
    model_settings = SaveModelSettingsRequest.model_validate(
        {
            "provider": TEST_PROVIDER,
            "modelSettings": {
                "active": True,
                "checked": False,
                "model": "gpt-4o-mini",
                "modelProvider": TEST_PROVIDER,
                "apiKey": "openai-api-key",
                "maxTokens": 8000,
                "configuration": {
                    "temperature": 0.8,
                    "topP": 0.9,
                },
            },
            "enableTools": True,
        }
    )

    # Send request
    response = client.post(
        "/api/config/model",
        content=model_settings.model_dump_json(by_alias=True).encode("utf-8"),
    )
    assert app.dive_host["default"].model._llm_type == "openai-chat"

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
                        "maxTokens": 8000,
                        "configuration": {
                            "temperature": 0.8,
                            "topP": 0.9,
                        },
                    },
                    "dive": {
                        "model": "fake",
                        "modelProvider": "dive",
                        "maxTokens": None,
                        "configuration": {
                            "temperature": 0.0,
                            "topP": None,
                        },
                    },
                },
            },
        },
    )


def test_post_model_replace_all(test_client):
    """Test the /api/config/model/replaceAll POST endpoint."""
    app: DiveHostAPI
    client, app = test_client
    model_config_data = MOCK_MODEL_CONFIG.model_dump_json(by_alias=True).encode("utf-8")

    response = client.post(
        "/api/config/model/replaceAll",
        content=model_config_data,
    )

    assert app.dive_host["default"].model._llm_type == "openai-chat"

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
                        "active": True,
                        "checked": False,
                        "model": "gpt-4o",
                        "modelProvider": "openai",
                        "apiKey": "sk-mock-key",
                        "maxTokens": 4000,
                        "configuration": {
                            "temperature": 0.7,
                            "topP": 0.9,
                        },
                    },
                },
            },
        },
    )


def test_post_model_replace_all_with_none_provider(test_client):
    """Test the /api/config/model/replaceAll POST endpoint."""
    app: DiveHostAPI
    client, app = test_client
    model_config_data = MOCK_MODEL_CONFIG_WITH_NONE_PROVIDER.model_dump_json(
        by_alias=True
    ).encode("utf-8")

    response = client.post(
        "/api/config/model/replaceAll",
        content=model_config_data,
    )

    assert app.dive_host["default"].model._llm_type == "fake-model"

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
                "activeProvider": "none",
                "enableTools": True,
                "configs": {
                    "openai": {
                        "active": True,
                        "checked": False,
                        "model": "gpt-4o",
                        "modelProvider": "openai",
                        "apiKey": "sk-mock-key",
                        "maxTokens": 4000,
                        "configuration": {
                            "temperature": 0.7,
                            "topP": 0.9,
                        },
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


@pytest.fixture
def setup_command_alias():
    """Setup the command alias."""
    os.environ["DIVE_COMMAND_ALIAS_CONTENT"] = json.dumps(
        {"python3": "alternate-python3", "node": "alternate-node"}
    )
    yield
    del os.environ["DIVE_COMMAND_ALIAS_CONTENT"]


def test_get_mcp_server_with_alias(setup_command_alias, test_client):
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


def test_post_mcp_server_with_alias(setup_command_alias, test_client):
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


def test_tools_and_mcpserver_enable_status(test_client):
    """Check if tool enable status is currect in both apis."""
    client, _ = test_client

    # check tools api default status
    response = client.get("/api/tools")
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "message": None,
            "tools": [
                {
                    "name": "echo",
                    "tools": [
                        {
                            "name": "echo",
                            "description": "A simple echo tool to verify if the MCP server is working properly.\nIt returns a characteristic response containing the input message.",  # noqa: E501
                        },
                        {"name": "ignore", "description": "Do nothing."},
                    ],
                    "description": "",
                    "enabled": True,
                    "icon": "",
                    "error": None,
                }
            ],
        },
    )

    # Disable tool
    payload = {
        "mcpServers": {
            "echo": McpServerConfig(
                transport="stdio",
                command="python3",
                enabled=False,
                args=[
                    "-m",
                    "dive_mcp_host.host.tools.echo",
                    "--transport=stdio",
                ],
                env={"NODE_ENV": "production"},
                url=None,
            ).model_dump(),
        }
    }

    response = client.post(
        "/api/config/mcpserver",
        json=payload,
    )
    assert response.status_code == SUCCESS_CODE

    # check mcpserver api
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
                        "enabled": False,
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

    # check tools api
    response = client.get("/api/tools")
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "message": None,
            "tools": [
                {
                    "name": "echo",
                    "tools": [
                        {
                            "name": "echo",
                            "description": "A simple echo tool to verify if the MCP server is working properly.\nIt returns a characteristic response containing the input message.",  # noqa: E501
                        },
                        {"name": "ignore", "description": "Do nothing."},
                    ],
                    "description": "",
                    "enabled": False,
                    "icon": "",
                    "error": None,
                }
            ],
        },
    )
