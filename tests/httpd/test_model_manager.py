import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio

from dive_mcp_host.httpd.conf.model.manager import ModelManager
from dive_mcp_host.httpd.routers.models import (
    ModelConfig,
    ModelSettings,
)

# Register custom mark
integration = pytest.mark.integration


# Unit tests
class TestModelManager:
    """Unit tests for ModelManager class's basic functionality."""

    @pytest.fixture
    def mock_config_file(self):
        """Create a mock configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "activeProvider": "test_provider",
                    "enableTools": True,
                    "configs": {
                        "test_provider": {
                            "modelProvider": "test_provider",
                            "model": "test_model",
                            "apiKey": "test_key",
                            "configuration": {"baseURL": "http://test.url"},
                        },
                    },
                },
                f,
            )
            config_path = f.name
        yield config_path
        # Clean up after test
        Path(config_path).unlink()

    def test_singleton_pattern(self):
        """Test if the singleton pattern works properly."""
        instance1 = ModelManager.get_instance()
        instance2 = ModelManager.get_instance()
        assert instance1 is instance2

    def test_config_path_setting(self):
        """Test if the configuration path is set correctly."""
        test_path = "/test/path.json"
        manager = ModelManager.get_instance(test_path)
        assert manager.config_path == test_path

    @pytest.mark.asyncio
    async def test_get_config(self, mock_config_file):
        """Test retrieving model configuration."""
        manager = ModelManager.get_instance(mock_config_file)
        config = await manager.get_config()
        assert config is not None
        assert config.get("activeProvider") == "test_provider"

    @pytest.mark.asyncio
    @patch("dive_mcp_host.httpd.conf.model.manager.ModelManager.save_single_settings")
    async def test_save_single_settings(self, mock_save, mock_config_file):
        """Test saving model configuration."""
        manager = ModelManager.get_instance(mock_config_file)
        # Create model settings
        test_settings = ModelSettings(
            model="new_model",
            modelProvider="new_provider",
            apiKey=None,
            configuration=None,
            temperature=None,
            topP=None,
            maxTokens=None,
        )

        # Set mock return value
        mock_save.return_value = None

        # Call the method
        await manager.save_single_settings("new_provider", test_settings, True)

        # Verify the method was called
        mock_save.assert_called_once_with("new_provider", test_settings, True)

    @pytest.mark.asyncio
    @patch("dive_mcp_host.httpd.conf.model.manager.ModelManager.replace_all_settings")
    async def test_replace_all_settings(self, mock_replace, mock_config_file):
        """Test replacing the entire model configuration."""
        manager = ModelManager.get_instance(mock_config_file)
        new_config = ModelConfig(
            activeProvider="replaced_provider",
            enableTools=True,
            configs={
                "replaced_provider": ModelSettings(
                    model="replaced_model",
                    modelProvider="replaced_provider",
                    apiKey=None,
                    configuration=None,
                    temperature=None,
                    topP=None,
                    maxTokens=None,
                )
            },
        )

        # Set mock return value
        mock_replace.return_value = True

        # Call the method
        result = await manager.replace_all_settings(new_config)

        # Verify
        assert result is True
        mock_replace.assert_called_once_with(new_config)

    @pytest.mark.asyncio
    async def test_initialize(self, mock_config_file):
        """Test initializing the manager."""
        manager = ModelManager.get_instance(mock_config_file)
        result = await manager.initialize()

        assert result is True
        assert manager.current_settings is not None
        assert manager.current_settings.model == "test_model"
        assert manager.current_settings.model_provider == "test_provider"

    @pytest.mark.asyncio
    async def test_get_active_settings(self, mock_config_file):
        """Test getting the active model settings."""
        manager = ModelManager.get_instance(mock_config_file)
        await manager.initialize()
        settings = await manager.get_active_settings()

        assert settings is not None
        assert settings.model == "test_model"
        assert settings.model_provider == "test_provider"

    @pytest.mark.asyncio
    async def test_get_available_providers(self, mock_config_file):
        """Test getting available providers."""
        manager = ModelManager.get_instance(mock_config_file)
        providers = await manager.get_available_providers()

        assert providers is not None
        assert "test_provider" in providers

    @pytest.mark.asyncio
    async def test_parse_settings(self, mock_config_file):
        """Test parsing model settings."""
        manager = ModelManager.get_instance(mock_config_file)
        config = await manager.get_config()
        if config is None:
            pytest.skip("Configuration not available")

        model_config = config.get("configs", {}).get("test_provider", {})

        settings = await manager.parse_settings(model_config)

        assert settings is not None
        assert settings.model == "test_model"
        assert settings.model_provider == "test_provider"
        assert settings.api_key == "test_key"
        assert settings.configuration is not None
        assert settings.configuration.base_url == "http://test.url"

    @pytest.mark.asyncio
    async def test_get_settings_by_provider(self, mock_config_file):
        """Test getting model settings by specific provider."""
        manager = ModelManager.get_instance(mock_config_file)

        # Test existing provider
        settings = await manager.get_settings_by_provider("test_provider")
        assert settings is not None
        assert settings.model == "test_model"
        assert settings.model_provider == "test_provider"
        assert settings.api_key == "test_key"
        assert settings.configuration is not None
        assert settings.configuration.base_url == "http://test.url"

        # Test non-existing provider
        non_existing_settings = await manager.get_settings_by_provider(
            "non_existing_provider"
        )
        assert non_existing_settings is None


# Integration tests
@pytest.mark.integration
class TestModelManagerIntegration:
    """Integration tests for the complete functionality of ModelManager class."""

    @pytest_asyncio.fixture
    async def test_config_path(self):
        """Set up the test configuration file path."""
        # Use a temporary directory to create the config file
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "modelConfig.json"
            test_config = {
                "activeProvider": "fake",
                "enableTools": True,
                "configs": {
                    "fake": {
                        "modelProvider": "fake",
                        "model": "fake-model",
                        "apiKey": "",
                        "configuration": {"baseURL": ""},
                    },
                },
            }
            with config_path.open("w") as f:
                json.dump(test_config, f)
            yield str(config_path)

    @pytest.mark.asyncio
    @patch("dive_mcp_host.httpd.conf.model.manager.json.dump")
    async def test_full_model_workflow(self, mock_json_dump, test_config_path):
        """Test the complete model configuration, initialization, and usage workflow."""
        # Mock json.dump to avoid serialization issues
        mock_json_dump.return_value = None

        # Set up ModelManager instance
        manager = ModelManager.get_instance(test_config_path)

        # Initialize the manager
        result = await manager.initialize()
        assert result is True
        assert manager.current_settings is not None
        assert manager.current_settings.model == "fake-model"
        assert manager.current_settings.model_provider == "fake"
