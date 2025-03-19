import logging
import os
from pathlib import Path
from typing import Any

from dive_mcp_host.httpd.prompts.system import system_prompt

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PromptManager:
    """Prompt Manager for handling system prompts and custom rules."""

    def __init__(self, custom_rules_path: str | None = None) -> None:
        """Initialize the PromptManager.

        Args:
            custom_rules_path: Optional path to the custom rules file.
        """
        self.prompts: dict[str, str] = {}
        self.custom_rules_path = custom_rules_path or str(Path.cwd() / ".customrules")

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
    prompt_manager = PromptManager()
    system_prompt_text = prompt_manager.get_prompt("system")
    logger.info("System prompt length: %d", len(system_prompt_text or ""))
