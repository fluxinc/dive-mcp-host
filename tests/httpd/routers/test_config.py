import tempfile
from dataclasses import dataclass

import pytest
import pytest_asyncio
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import AnyUrl

from dive_mcp_host.host.conf import CheckpointerConfig, LLMConfig
from dive_mcp_host.httpd.app import DiveHostAPI
from dive_mcp_host.httpd.conf.mcpserver.manager import Config, ServerConfig
from dive_mcp_host.httpd.conf.service.manager import (
    ConfigLocation,
    DBConfig,
    ServiceConfig,
)
from dive_mcp_host.httpd.routers.config import (
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
        "openai": LLMConfig(
            model="gpt-4",
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


@dataclass(slots=True)
class _ConfigFileNames:
    """Config file names."""

    service_config_file: str
    mcp_server_config_file: str
    model_config_file: str
    prompt_config_file: str


@pytest.fixture
def config_files():
    """Create config files."""
    with (
        tempfile.NamedTemporaryFile(
            prefix="testServiceConfig_", suffix=".json"
        ) as service_config_file,
        tempfile.NamedTemporaryFile(
            prefix="testMcpServerConfig_", suffix=".json"
        ) as mcp_server_config_file,
        tempfile.NamedTemporaryFile(
            prefix="testModelConfig_", suffix=".json"
        ) as model_config_file,
        tempfile.NamedTemporaryFile(suffix=".testCustomrules") as prompt_config_file,
    ):
        service_config_file.write(
            ServiceConfig(
                db=DBConfig(
                    uri="sqlite:///test_db.sqlite",
                    async_uri="sqlite+aiosqlite:///test_db.sqlite",
                ),
                checkpointer=CheckpointerConfig(
                    uri=AnyUrl("sqlite:///test_db.sqlite"),
                ),
                config_location=ConfigLocation(
                    mcp_server_config_path=mcp_server_config_file.name,
                    model_config_path=model_config_file.name,
                    prompt_config_path=prompt_config_file.name,
                ),
            )
            .model_dump_json(by_alias=True)
            .encode("utf-8")
        )
        service_config_file.flush()

        mcp_server_config_file.write(
            Config(
                mcpServers={
                    "echo": ServerConfig(
                        transport="command",
                        command="python3",
                        args=[
                            "-m",
                            "dive_mcp_host.host.tools.echo",
                            "--transport=stdio",
                        ],
                    ),
                },
            )
            .model_dump_json(by_alias=True)
            .encode("utf-8")
        )
        mcp_server_config_file.flush()

        model_config_file.write(
            ModelConfig(
                activeProvider="dive",
                enableTools=True,
                configs={
                    "dive": LLMConfig(
                        modelProvider="dive",
                        model="fake",
                        configuration={},
                    ),
                },
            )
            .model_dump_json(by_alias=True)
            .encode("utf-8")
        )
        model_config_file.flush()

        prompt_config_file.write(b"testCustomrules")
        prompt_config_file.flush()

        yield _ConfigFileNames(
            service_config_file=service_config_file.name,
            mcp_server_config_file=mcp_server_config_file.name,
            model_config_file=model_config_file.name,
            prompt_config_file=prompt_config_file.name,
        )


@pytest_asyncio.fixture
async def client(config_files: _ConfigFileNames):
    """Create a test client."""
    app = DiveHostAPI(
        config_path=config_files.service_config_file,
    )
    app.include_router(config, prefix="/api/config")
    async with app.prepare():
        with TestClient(app) as client:
            yield client


def test_get_mcp_server(client):
    """Test the /api/config/mcpserver GET endpoint."""
    # Send request
    response = client.get("/api/config/mcpserver")

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
    assert default_server["transport"] == "command"
    assert default_server["enabled"] is True
    assert default_server["command"] == "./mcp-server.js"
    assert default_server["args"] == ["./mcp-server.js"]
    assert default_server["env"] == {"NODE_ENV": "production"}
    assert default_server["url"] is None


def test_post_mcp_server(client):
    """Test the /api/config/mcpserver POST endpoint."""
    # Prepare request data - convert McpServerConfig objects to dict
    mock_server_dict = {}
    for key, value in MOCK_MCP_CONFIG.items():
        mock_server_dict[key] = value.model_dump()

    server_data = {"mcpServers": mock_server_dict}

    # Send request
    response = client.post(
        "/api/config/mcpserver",
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
    """Test the /api/config/model GET endpoint."""
    # Send request
    response = client.get("/api/config/model")

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
    assert config_data["activeProvider"] == "openai"
    assert config_data["enableTools"] is True
    assert "configs" in config_data
    assert isinstance(config_data["configs"], dict)

    provider_configs = config_data["configs"]
    assert "openai" in provider_configs
    openai_config = provider_configs["openai"]
    assert openai_config["model"] == "gpt-4"
    assert openai_config["modelProvider"] == "openai"
    assert openai_config["temperature"] == 0.7
    assert openai_config["topP"] == 0.9
    assert openai_config["maxTokens"] == 4000
    assert openai_config["configuration"] is None

def test_post_model(client):
    """Test the /api/config/model POST endpoint."""
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
    assert "success" in response_data
    assert response_data["success"] is True


def test_post_model_replace_all(client):
    """Test the /api/config/model/replaceAll POST endpoint."""
    # Prepare request data - use the mock model config for simplicity
    model_config_data = MOCK_MODEL_CONFIG.model_dump(by_alias=True)

    # Send request
    response = client.post(
        "/api/config/model/replaceAll",
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
    """Test the /api/config/customrules GET endpoint."""
    # Send request
    response = client.get("/api/config/customrules")

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
    """Test the /api/config/customrules POST endpoint."""
    # Prepare custom rules data
    custom_rules = "# New Custom Rules\n1. Be concise\n2. Use simple language"

    # Send request - This endpoint expects raw text, not JSON
    response = client.post(
        "/api/config/customrules",
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
