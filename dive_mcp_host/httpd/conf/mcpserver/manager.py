import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field


# Define necessary types for configuration
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


# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MCPServerManager:
    """MCP Server Manager for configuration handling."""

    _instance: ClassVar[Any] = None

    def __new__(cls) -> "MCPServerManager":
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

        self.config_path: str = config_path or str(Path.cwd() / "config.json")
        self.current_config: Config | None = None
        self._initialized = True

    @classmethod
    def get_instance(cls, config_path: str | None = None) -> "MCPServerManager":
        """Get the singleton instance of MCPServerManager.

        Args:
            config_path: Optional path to the configuration file.

        Returns:
            MCPServerManager instance.
        """
        instance = cls()
        if config_path:
            instance.config_path = config_path
        return instance

    async def initialize(self) -> bool:
        """Initialize the MCPServerManager.

        Returns:
            True if successful, False otherwise.
        """
        config_result = await self.get_config()

        if not config_result:
            logger.error("Server configuration not found")
            return False

        self.current_config = config_result
        return True

    async def get_config(self) -> Config | None:
        """Get configuration.

        Returns:
            A tuple of (config, servers) or None if configuration is not found.
        """
        try:
            env_config = os.environ.get("DIVE_MCP_CONFIG_CONTENT")

            if env_config:
                config_content = env_config
            else:
                with Path(self.config_path).open(encoding="utf-8") as f:
                    config_content = f.read()

            config_dict = json.loads(config_content)
            return Config(**config_dict)
        except (OSError, json.JSONDecodeError) as error:
            logger.error("Error loading configuration: %s", error)
            return None

    async def get_enabled_servers(self) -> list[str]:
        """Get list of enabled server names.

        Returns:
            List of enabled server names.
        """
        if not self.current_config:
            return []

        return [
            server_name
            for server_name, config in self.current_config.mcp_servers.items()
            if config.enabled
        ]

    async def update_all_configs(self, new_config: Config) -> bool:
        """Replace all configurations.

        Args:
            new_config: New configuration.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with Path(self.config_path).open("w", encoding="utf-8") as f:
                json_dict = new_config.model_dump(by_alias=True)
                json.dump(json_dict, f, indent=2)

            self.current_config = new_config
            return True
        except (OSError, json.JSONDecodeError) as error:
            logger.error("Error replacing all configurations: %s", error)
            return False


if __name__ == "__main__":
    asyncio.run(MCPServerManager.get_instance().initialize())
    configs = asyncio.run(MCPServerManager.get_instance().get_config())
    if configs:
        logger.info(
            "Available server configurations: %s", list(configs.mcp_servers.keys())
        )

    enabled_servers = asyncio.run(MCPServerManager.get_instance().get_enabled_servers())
    logger.info("Enabled servers: %s", enabled_servers)
