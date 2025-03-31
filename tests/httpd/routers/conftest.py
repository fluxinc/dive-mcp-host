import tempfile
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from pydantic import AnyUrl

from dive_mcp_host.host.conf import CheckpointerConfig, LLMConfig
from dive_mcp_host.httpd.app import DiveHostAPI, create_app
from dive_mcp_host.httpd.conf.mcpserver.manager import Config, ServerConfig
from dive_mcp_host.httpd.conf.service.manager import (
    ConfigLocation,
    DBConfig,
    ServiceConfig,
    ServiceManager,
)
from dive_mcp_host.httpd.routers.models import ModelConfig


@dataclass(slots=True)
class ConfigFileNames:
    """Config file names."""

    service_config_file: str
    mcp_server_config_file: str
    model_config_file: str
    prompt_config_file: str


@pytest.fixture
def config_files() -> Generator[ConfigFileNames, None, None]:
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
                        transport="stdio",
                        command="python3",
                        args=[
                            "-m",
                            "dive_mcp_host.host.tools.echo",
                            "--transport=stdio",
                        ],
                        env={"NODE_ENV": "production"},
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

        yield ConfigFileNames(
            service_config_file=service_config_file.name,
            mcp_server_config_file=mcp_server_config_file.name,
            model_config_file=model_config_file.name,
            prompt_config_file=prompt_config_file.name,
        )


@pytest_asyncio.fixture
async def test_client(
    config_files: ConfigFileNames,
) -> AsyncGenerator[tuple[TestClient, DiveHostAPI], None]:
    """Create a test client with fake model.

    This fixture creates a test client with a DiveHostAPI instance. The DiveHostAPI
    instance can be used to mock methods and test router endpoints.
    The fixture yields both the test client and app instance to allow access to both
    during testing.

    Returns:
        A tuple of the test client and the app.
    """
    service_manager = ServiceManager(config_files.service_config_file)
    service_manager.initialize()
    app = create_app(service_manager)
    with TestClient(app) as client:
        yield client, app
