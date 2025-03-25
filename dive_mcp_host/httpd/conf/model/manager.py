import json
import logging
import os
from pathlib import Path
from typing import Any

from dive_mcp_host.host.conf import LLMConfig
from dive_mcp_host.httpd.routers.models import ModelConfig

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
        self._config_path: str = config_path or str(Path.cwd() / "modelConfig.json")
        self._current_setting: LLMConfig | None = None
        self._enable_tools: bool = True
        self._config_dict: dict[str, Any] | None = None

    def initialize(self) -> bool:
        """Initialize the ModelManager."""
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

    def _parse_settings(self, model_config: dict[str, Any]) -> LLMConfig | None:
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
                configuration = {"base_url": base_url}

            # Create model settings with models.py version
            return LLMConfig(
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

    @property
    def current_setting(self) -> LLMConfig | None:
        """Get the active model settings.

        Returns:
            Model settings or None if configuration or active provider is not found.
        """
        return self._current_setting

    @property
    def full_config(self) -> ModelConfig | None:
        """Get the full model configuration.

        Returns:
            Model configuration or None if configuration is not found.
        """
        if not self._config_dict:
            return None

        return ModelConfig(
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

    def get_settings_by_provider(self, provider: str) -> LLMConfig | None:
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
        upload_model_settings: LLMConfig,
        enable_tools_: bool | None = None,
    ) -> None:
        """Save single model configuration.

        Args:
            provider: Model provider name.
            upload_model_settings: Model settings to upload.
            enable_tools_: Whether to enable tools.
        """
        enable_tools = True if enable_tools_ is None else enable_tools_

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

        with Path(self._config_path).open("w", encoding="utf-8") as f:
            json.dump(self._config_dict, f, indent=2)

    def replace_all_settings(
        self,
        upload_model_settings: ModelConfig,
    ) -> None:
        """Replace all model configurations.

        Args:
            upload_model_settings: Model settings to upload.

        Returns:
            True if successful.
        """
        with Path(self._config_path).open("w", encoding="utf-8") as f:
            json.dump(
                upload_model_settings.model_dump(by_alias=True, exclude_none=True),
                f,
                indent=2,
            )


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
