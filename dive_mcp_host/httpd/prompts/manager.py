import logging
import os
from pathlib import Path
from typing import Any, ClassVar

from dive_mcp_host.httpd.prompts.system import system_prompt

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PromptManager:
    """Prompt Manager for handling system prompts and custom rules."""

    _instance: ClassVar[Any] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "PromptManager":  # noqa: ARG004
        """Create a new instance or return existing instance of PromptManager.

        Returns:
            PromptManager instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize directly
            cls._instance._initialized = False  # noqa: SLF001
        return cls._instance

    def __init__(self, custom_rules_path: str | None = None) -> None:
        """Initialize the PromptManager.

        Args:
            custom_rules_path: Optional path to the custom rules file.
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.prompts: dict[str, str] = {}
        self.custom_rules_path = custom_rules_path or str(Path.cwd() / ".customrules")
        self._initialized = True

        # Read .customrules file
        try:
            custom_rules = os.environ.get("DIVE_CUSTOM_RULES_CONTENT") or Path(
                self.custom_rules_path
            ).read_text(encoding="utf-8")
            # Combine system prompt and custom rules
            self.prompts["system"] = system_prompt(custom_rules)
        except OSError as error:
            logger.warning("Cannot read %s: %s", self.custom_rules_path, error)
            self.prompts["system"] = system_prompt("")

    @classmethod
    def get_instance(cls, custom_rules_path: str | None = None) -> "PromptManager":
        """Get the singleton instance of PromptManager.

        Args:
            custom_rules_path: Optional path to the custom rules file.

        Returns:
            PromptManager instance.
        """
        instance = cls()
        if custom_rules_path and instance.custom_rules_path != custom_rules_path:
            instance.custom_rules_path = custom_rules_path
            instance.update_system_prompt()
        return instance

    def set_prompt(self, key: str, prompt: str) -> None:
        """Set a prompt by key.

        Args:
            key: The key to store the prompt under.
            prompt: The prompt text.
        """
        self.prompts[key] = prompt

    def get_prompt(self, key: str) -> str | None:
        """Get a prompt by key.

        Args:
            key: The key of the prompt to retrieve.

        Returns:
            The prompt text or None if not found.
        """
        return self.prompts.get(key)

    def load_custom_rules(self) -> str:
        """Load custom rules from file or environment variable.

        Returns:
            The custom rules text.
        """
        try:
            return os.environ.get("DIVE_CUSTOM_RULES_CONTENT") or Path(
                self.custom_rules_path
            ).read_text(encoding="utf-8")
        except OSError as error:
            logger.warning("Cannot read %s: %s", self.custom_rules_path, error)
            return ""

    def update_system_prompt(self) -> None:
        """Update the system prompt with current custom rules."""
        custom_rules = self.load_custom_rules()
        self.prompts["system"] = system_prompt(custom_rules)


if __name__ == "__main__":
    prompt_manager = PromptManager.get_instance()
    system_prompt_text = prompt_manager.get_prompt("system")
    logger.info("System prompt length: %d", len(system_prompt_text or ""))
