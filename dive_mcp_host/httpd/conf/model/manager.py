import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from dive_mcp_host.httpd.routers.models import (
    ModelConfig,
    ModelConfiguration,
    ModelSettings,
)

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelManager:
    """Model Manager."""

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize the ModelManager.

        Args:
            config_path: Optional path to the model configuration file.
        """
        self.config_path: str = config_path or str(Path.cwd() / "modelConfig.json")
        self.current_settings: ModelSettings | None = None
        self.enable_tools: bool = True

    async def initialize(self) -> bool:
        """Initialize the ModelManager."""
        config_dict = await self.get_config()

        if not config_dict:
            logger.error("Model configuration not found")
            return False

        active_provider = config_dict.get(
            "activeProvider",
            config_dict.get("active_provider", ""),
        )
        configs = config_dict.get("configs", {})
        self.enable_tools = config_dict.get(
            "enableTools",
            config_dict.get("enable_tools", True),
        )
        model_config = configs.get(active_provider, {}) if active_provider else {}

        if not model_config:
            logger.error(
                "Model settings not found for active provider: %s",
                active_provider,
            )
            return False

        # Create model settings with models.py version
        model_settings = await self.parse_settings(model_config)
        self.current_settings = model_settings
        return True

    async def parse_settings(
        self, model_config: dict[str, Any]
    ) -> ModelSettings | None:
        """Parse the model settings.

        Args:
            model_config: Single model configuration.

        Returns:
            Model settings or None if configuration or active provider is not found.
        """
        # Create configuration object if base_url exists
        try:
            base_url = (
                model_config.get("configuration", {}).get(
                    "baseURL",
                    model_config.get("configuration", {}).get("base_url"),
                )
                or model_config.get("baseURL", model_config.get("base_url"))
                or ""
            )
            configuration = None
            if base_url:
                configuration = ModelConfiguration(baseURL=base_url)

            # Create model settings with models.py version
            return ModelSettings(
                model=model_config.get("model", ""),
                modelProvider=model_config.get(
                    "modelProvider",
                    model_config.get("model_provider", ""),
                ),
                configuration=configuration,
                apiKey=model_config.get("apiKey", model_config.get("api_key")),
                temperature=model_config.get("temperature"),
                topP=model_config.get("topP", model_config.get("top_p")),
                maxTokens=model_config.get("maxTokens", model_config.get("max_tokens")),
                **{
                    k: v
                    for k, v in model_config.items()
                    if k
                    not in [
                        "model",
                        "modelProvider",
                        "model_provider",
                        "base_url",
                        "apiKey",
                        "api_key",
                        "temperature",
                        "topP",
                        "top_p",
                        "maxTokens",
                        "max_tokens",
                        "configuration",
                    ]
                },
            )
        except (ValueError, TypeError, KeyError) as e:
            logger.error("Error parsing model settings: %s", e)
            return None

    async def get_config(self) -> dict[str, Any] | None:
        """Get model configuration.

        Returns:
            Model configuration dictionary or None if not found.
        """
        try:
            env_config = os.environ.get("DIVE_MODEL_CONFIG_CONTENT")
            logger.debug("[ModelManager] %s", "use env" if env_config else "use config")

            config_path = self.config_path or str(Path.cwd() / "modelConfig.json")

            if env_config:
                config_content = env_config
            else:
                with Path(config_path).open(encoding="utf-8") as f:
                    config_content = f.read()

            return json.loads(config_content)
        except (OSError, json.JSONDecodeError) as error:
            logger.error("Error loading model configuration: %s", error)
            return None

    async def get_active_settings(self) -> ModelSettings | None:
        """Get the active model settings.

        Returns:
            Model settings or None if configuration or active provider is not found.
        """
        return self.current_settings

    async def get_settings_by_provider(self, provider: str) -> ModelSettings | None:
        """Get the model settings by provider.

        Args:
            provider: Model provider name.
        """
        config_dict = await self.get_config()
        if not config_dict:
            return None
        model_config = config_dict.get("configs", {}).get(provider, None)
        if not model_config:
            return None
        return await self.parse_settings(model_config)

    async def get_available_providers(self) -> list[str]:
        """Get the available model providers.

        Returns:
            List of model providers.
        """
        config_dict = await self.get_config()
        if not config_dict:
            return []
        return list(config_dict.get("configs", {}).keys())

    async def save_single_settings(
        self,
        provider: str,
        upload_model_settings: ModelSettings,
        enable_tools_: bool | None = None,
    ) -> None:
        """Save single model configuration.

        Args:
            provider: Model provider name.
            upload_model_settings: Model settings to upload.
            enable_tools_: Whether to enable tools.
        """
        config_dict = await self.get_config()
        enable_tools = True if enable_tools_ is None else enable_tools_

        if not config_dict:
            config_dict = {
                "activeProvider": provider,
                "enableTools": enable_tools,
                "configs": {
                    provider: upload_model_settings,
                },
            }
        else:
            config_dict["activeProvider"] = provider
            if "configs" not in config_dict:
                config_dict["configs"] = {}
            config_dict["configs"][provider] = upload_model_settings
            config_dict["enableTools"] = enable_tools

        with Path(self.config_path).open("w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

    async def replace_all_settings(
        self,
        upload_model_settings: ModelConfig,
    ) -> bool:
        """Replace all model configurations.

        Args:
            upload_model_settings: Model settings to upload.

        Returns:
            True if successful.
        """
        with Path(self.config_path).open("w", encoding="utf-8") as f:
            json.dump(upload_model_settings, f, indent=2)
        return True


if __name__ == "__main__":
    model_manager = ModelManager()
    asyncio.run(model_manager.initialize())
    current_settings = model_manager.current_settings
    if current_settings is not None:
        logger.info("current_settings: %s", current_settings.model_dump(by_alias=False))
    else:
        logger.error("current_settings is None")

    available_providers = asyncio.run(model_manager.get_available_providers())
    logger.info("available_providers: %s", available_providers)

    settings_by_provider = asyncio.run(model_manager.get_settings_by_provider("openai"))
    logger.info("settings_by_provider: %s", settings_by_provider)
