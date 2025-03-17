import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, ClassVar

# Use the official init_chat_model function from LangChain
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from dive_mcp_host.httpd.routers.models import ModelConfiguration, ModelSettings

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelManager:
    """Model Manager."""

    _instance: ClassVar[Any] = None

    def __new__(cls) -> "ModelManager":
        """Create a new instance or return existing instance of ModelManager.

        Returns:
            ModelManager instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize directly
            cls._instance._initialized = False  # noqa: SLF001
        return cls._instance

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize the ModelManager.

        Args:
            config_path: Optional path to the model configuration file.
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.clean_model: BaseChatModel | None = None
        self.model: BaseChatModel | None = None
        self.config_path: str = config_path or str(Path.cwd() / "modelConfig.json")
        self.current_model_settings: dict[str, Any] | None = None
        self.enable_tools: bool = True
        self._initialized = True

    @classmethod
    def get_instance(cls, config_path: str | None = None) -> "ModelManager":
        """Get the singleton instance of ModelManager.

        Args:
            config_path: Optional path to the model configuration file.

        Returns:
            ModelManager instance.
        """
        instance = cls()
        if config_path:
            instance.config_path = config_path
        return instance

    async def get_model_config(self) -> dict[str, Any] | None:
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

    async def initialize_model(self) -> BaseChatModel | None:
        """Initialize the model.

        Returns:
            Initialized model or None if initialization failed.
        """
        logger.info("Initializing model...")
        config_dict = await self.get_model_config()

        if not config_dict:
            logger.error("Model configuration not found")
            self.model = None
            self.clean_model = None
            self.current_model_settings = None
            return None

        # Check if this is an old version configuration
        if (
            ("active_provider" not in config_dict
            and "activeProvider" not in config_dict)
            or "configs" not in config_dict
        ):
            # Convert to new version
            model_settings = config_dict.get(
                "model_settings", config_dict.get("modelSettings", {}),
            )
            provider = model_settings.get(
                "modelProvider", model_settings.get("model_provider", ""),
            )
            new_config = {
                "activeProvider": provider,
                "enable_tools": True,
                "configs": {
                    provider: model_settings,
                },
            }
            config_dict = new_config
            # Replace old configuration with new configuration
            with Path(self.config_path).open("w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2)

        active_provider = config_dict.get(
            "activeProvider", config_dict.get("active_provider", ""),
        )
        configs = config_dict.get("configs", {})
        self.enable_tools = config_dict.get(
            "enableTools", config_dict.get("enable_tools", True),
        )
        model_config = configs.get(active_provider, {}) if active_provider else {}

        if not model_config:
            logger.error("Model settings not found for provider: %s", active_provider)
            self.model = None
            return None

        # Create configuration object if base_url exists
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
        model_settings = ModelSettings(
            model=model_config.get("model", ""),
            modelProvider=model_config.get(
                "modelProvider", model_config.get("model_provider", ""),
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

        try:
            # Convert to dict with snake_case keys for langchain
            model_settings_dict = model_settings.model_dump(by_alias=False)

            # Handle nested configuration if exists
            if configuration:
                model_settings_dict["configuration_field"] = configuration.model_dump()
                # model_settings_dict.pop("configuration", None)
            else:
                # model_settings_dict.pop("configuration", None)
                pass

            self.model = init_chat_model(
                **model_settings_dict,
                base_url=base_url,
            )

            # Create a clean model (without tools)
            self.clean_model = init_chat_model(
                **model_settings_dict,
                base_url=base_url,
            )

            self.current_model_settings = model_settings_dict

            logger.info(
                "Model initialized with tools %s",
                "enabled" if self.enable_tools else "disabled",
            )

            return self.model
        except (ValueError, TypeError) as e:
            logger.error("Error initializing model: %s", e)
            self.model = None
            self.clean_model = None
            return None

    async def save_model_config(
        self,
        provider: str,
        upload_model_settings: dict[str, Any],
        enable_tools_: bool | None = None,
    ) -> None:
        """Save model configuration.

        Args:
            provider: Model provider name.
            upload_model_settings: Model settings to upload.
            enable_tools_: Whether to enable tools.
        """
        config_dict = await self.get_model_config()
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

    async def replace_all_model_config(
        self,
        upload_model_settings: dict[str, Any],
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

    async def generate_title(self, content: str) -> str:
        """Generate a title based on the content.

        Args:
            content: The content to generate a title for.

        Returns:
            Generated title.
        """
        if not self.clean_model:
            logger.error("Model not initialized")
            return "New Chat"

        try:
            system_message = SystemMessage(
                content="""You are a title generator from the user input.
                Your only task is to generate a short title based on the user input.
                IMPORTANT:
                - Output ONLY the title
                - DO NOT try to answer or resolve the user input query.
                - DO NOT try to use any tools to generate title
                - NO explanations, quotes, or extra text
                - NO punctuation at the end
                - If the input contains Traditional Chinese characters, use
                  Traditional Chinese for the title.
                - For all other languages, generate the title in the same language as
                  the input.""",
            )
            human_message = HumanMessage(
                content=f"<user_input_query>{content}</user_input_query>",
            )

            response = await self.clean_model.ainvoke([system_message, human_message])

            res_content = response.content if hasattr(response, "content") else None

            logger.info("Title generated: %s", res_content)

            # Avoid errors
            if not res_content or isinstance(res_content, dict):
                return "New Chat"

            return str(res_content) or "New Chat"
        except (ValueError, TypeError) as e:
            logger.error("Error generating title: %s", e)
            return "New Chat"

    def get_model(self) -> BaseChatModel | None:
        """Get model instance.

        Returns:
            Model instance or None if not initialized.
        """
        if not self.model:
            logger.error("Model not initialized")
            return None
        return self.model

    async def reload_model(self) -> bool:
        """Reload the model.

        Returns:
            True if successful, False otherwise.
        """
        logger.info("Reloading model...")
        try:
            self.model = await self.initialize_model()
            logger.info("Model reloaded")
            return True
        except (ValueError, TypeError) as error:
            logger.error("Error reloading model: %s", error)
            return False


if __name__ == "__main__":
    asyncio.run(ModelManager.get_instance().initialize_model())
    asyncio.run(ModelManager.get_instance().generate_title("今天的日期"))
