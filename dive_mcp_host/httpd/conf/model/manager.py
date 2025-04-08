import json
import logging
import os
from pathlib import Path
from typing import Any

from dive_mcp_host.host.conf.llm import LLMConfigTypes, get_llm_config_type
from dive_mcp_host.httpd.conf.envs import DIVE_CONFIG_DIR
from dive_mcp_host.httpd.routers.models import ModelFullConfigs

# Logger setup
logger = logging.getLogger(__name__)


class ModelManager:
    """Model Manager."""

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize the ModelManager.

        Args:
            config_path: Optional path to the model configuration file.
                If not provided, it will be set to "modelConfig.json" in current
                working directory.
        """
        self._config_path: str = config_path or str(
            DIVE_CONFIG_DIR / "model_config.json"
        )
        self._current_setting: LLMConfigTypes | None = None
        self._config_dict: dict[str, Any] | None = None

    def initialize(self) -> bool:
        """Initialize the ModelManager."""
        logger.info("Initializing ModelManager from %s", self._config_path)
        if env_config := os.environ.get("DIVE_MODEL_CONFIG_CONTENT"):
            config_content = env_config
        else:
            with Path(self._config_path).open(encoding="utf-8") as f:
                config_content = f.read()
        self._config_dict = json.loads(config_content)
        if not self._config_dict:
            logger.error("Model configuration not found")
            return False

        active_provider = self._config_dict.get(
            "activeProvider",
            self._config_dict.get("active_provider", ""),
        )
        configs = self._config_dict.get("configs", {})
        self._enable_tools = self._config_dict.get(
            "enableTools",
            self._config_dict.get("enable_tools", True),
        )
        if model_config := (
            configs.get(active_provider, {}) if active_provider else {}
        ):
            # Create model settings with models.py version
            self._current_setting = self._parse_settings(model_config)
            return True
        logger.error(
            "Model settings not found for active provider: %s",
            active_provider,
        )
        return False

    def _parse_settings(self, model_config: dict[str, Any]) -> LLMConfigTypes | None:
        """Parse the model settings.

        Args:
            model_config: Single model configuration.

        Returns:
            Model settings or None if configuration or active provider is not found.
        """
        try:
            return get_llm_config_type(model_config["modelProvider"]).model_validate(
                model_config
            )
        except (ValueError, TypeError, KeyError) as e:
            logger.error("Error parsing model settings: %s", e)
            return None

    @property
    def current_setting(self) -> LLMConfigTypes | None:
        """Get the active model settings.

        Returns:
            Model settings or None if configuration or active provider is not found.
        """
        return self._current_setting

    @property
    def full_config(self) -> ModelFullConfigs | None:
        """Get the full model configuration.

        Returns:
            Model configuration or None if configuration is not found.
        """
        if not self._config_dict:
            return None

        return ModelFullConfigs(
            activeProvider=self._config_dict.get(
                "activeProvider", self._config_dict.get("active_provider", "")
            ),
            enableTools=self._config_dict.get(
                "enableTools", self._config_dict.get("enable_tools", True)
            ),
            configs=self._config_dict.get("configs", {}),
        )

    @property
    def config_path(self) -> str:
        """Get the configuration path."""
        return self._config_path

    def get_settings_by_provider(self, provider: str) -> LLMConfigTypes | None:
        """Get the model settings by provider.

        Args:
            provider: Model provider name.
        """
        if not self._config_dict:
            return None
        model_config = self._config_dict.get("configs", {}).get(provider, None)
        if not model_config:
            return None
        return self._parse_settings(model_config)

    def get_available_providers(self) -> list[str]:
        """Get the available model providers.

        Returns:
            List of model providers.
        """
        if not self._config_dict:
            return []
        return list(self._config_dict.get("configs", {}).keys())

    def save_single_settings(
        self,
        provider: str,
        upload_model_settings: LLMConfigTypes,
        enable_tools: bool = True,
    ) -> None:
        """Save single model configuration.

        Args:
            provider: Model provider name.
            upload_model_settings: Model settings to upload.
            enable_tools: Whether to enable tools.
        """
        if not self._config_dict:
            self._config_dict = {
                "activeProvider": provider,
                "enableTools": enable_tools,
                "configs": {
                    provider: upload_model_settings.model_dump(
                        by_alias=True, exclude_none=True
                    ),
                },
            }
        else:
            self._config_dict["activeProvider"] = provider
            if "configs" not in self._config_dict:
                self._config_dict["configs"] = {}
            self._config_dict["configs"][provider] = upload_model_settings.model_dump(
                by_alias=True, exclude_none=True
            )
            self._config_dict["enableTools"] = enable_tools

        tmp = Path(f"{self._config_path}.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._config_dict, f, indent=2)
        tmp.rename(self._config_path)

    def replace_all_settings(
        self,
        upload_model_settings: ModelFullConfigs,
    ) -> None:
        """Replace all model configurations.

        Args:
            upload_model_settings: Model settings to upload.

        Returns:
            True if successful.
        """
        tmp = Path(f"{self._config_path}.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(
                upload_model_settings.model_dump(by_alias=True, exclude_none=True),
                f,
                indent=2,
            )
        tmp.rename(self._config_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_manager = ModelManager()
    model_manager.initialize()
    current_settings = model_manager.current_setting
    if current_settings is not None:
        logger.info("current_settings: %s", current_settings.model_dump(by_alias=False))
    else:
        logger.error("current_settings is None")

    available_providers = model_manager.get_available_providers()
    logger.info("available_providers: %s", available_providers)

    settings_by_provider = model_manager.get_settings_by_provider("openai")
    logger.info("settings_by_provider: %s", settings_by_provider)
