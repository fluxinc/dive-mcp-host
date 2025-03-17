import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from dive_mcp_host.httpd.conf.model_config.manager import ModelManager

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
    async def test_get_model_config(self, mock_config_file):
        """Test retrieving model configuration."""
        manager = ModelManager.get_instance(mock_config_file)
        config = await manager.get_model_config()
        assert config is not None
        assert config.get("activeProvider") == "test_provider"

    @pytest.mark.asyncio
    async def test_save_model_config(self, mock_config_file):
        """Test saving model configuration."""
        manager = ModelManager.get_instance(mock_config_file)
        test_settings = {
            "modelProvider": "new_provider",
            "model": "new_model",
        }
        await manager.save_model_config("new_provider", test_settings, True)

        # Verify the configuration has been saved
        config = await manager.get_model_config()
        assert config is not None
        assert config.get("activeProvider") == "new_provider"
        assert config.get("configs", {}).get("new_provider") == test_settings

    @pytest.mark.asyncio
    async def test_replace_all_model_config(self, mock_config_file):
        """Test replacing the entire model configuration."""
        manager = ModelManager.get_instance(mock_config_file)
        new_config = {
            "activeProvider": "replaced_provider",
            "configs": {
                "replaced_provider": {
                    "model": "replaced_model",
                },
            },
        }
        result = await manager.replace_all_model_config(new_config)
        assert result is True

        # Verify the configuration has been completely replaced
        config = await manager.get_model_config()
        assert config is not None
        assert config.get("activeProvider") == "replaced_provider"

    @pytest.mark.asyncio
    @patch("dive_mcp_host.httpd.conf.model_config.manager.init_chat_model")
    async def test_initialize_model(self, mock_init_chat_model, mock_config_file):
        """Test initializing the model."""
        # Create a mock model instance
        mock_model = MagicMock(spec=BaseChatModel)
        mock_init_chat_model.return_value = mock_model

        manager = ModelManager.get_instance(mock_config_file)
        model = await manager.initialize_model()

        assert model is mock_model
        assert manager.model is mock_model
        assert manager.clean_model is mock_model
        mock_init_chat_model.assert_called()

    @pytest.mark.asyncio
    @patch("dive_mcp_host.httpd.conf.model_config.manager.init_chat_model")
    async def test_get_model(self, mock_init_chat_model, mock_config_file):
        """Test getting the model instance."""
        # Create a mock model instance
        mock_model = MagicMock(spec=BaseChatModel)
        mock_init_chat_model.return_value = mock_model

        manager = ModelManager.get_instance(mock_config_file)
        await manager.initialize_model()
        model = manager.get_model()

        assert model is mock_model

    @pytest.mark.asyncio
    @patch("dive_mcp_host.httpd.conf.model_config.manager.init_chat_model")
    async def test_reload_model(self, mock_init_chat_model, mock_config_file):
        """Test reloading the model."""
        # Create a mock model instance
        mock_model = MagicMock(spec=BaseChatModel)
        mock_init_chat_model.return_value = mock_model

        manager = ModelManager.get_instance(mock_config_file)
        result = await manager.reload_model()

        assert result is True
        assert manager.model is mock_model

    @pytest.mark.asyncio
    @patch("dive_mcp_host.httpd.conf.model_config.manager.init_chat_model")
    async def test_generate_title(self, mock_init_chat_model, mock_config_file):
        """Test generating a title."""
        # Create a mock model instance and response
        mock_model = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content="Test Title")
        mock_model.ainvoke.return_value = mock_response
        mock_init_chat_model.return_value = mock_model

        manager = ModelManager.get_instance(mock_config_file)
        await manager.initialize_model()
        title = await manager.generate_title("This is a test content")

        assert title == "Test Title"
        mock_model.ainvoke.assert_called_once()


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
    @patch("dive_mcp_host.httpd.conf.model_config.manager.init_chat_model")
    async def test_full_model_workflow(self, mock_init_chat_model, test_config_path):
        """Test the complete model configuration, initialization, and usage workflow."""
        # Create a mock model instance
        mock_model = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content="Generated Title")
        mock_model.ainvoke.return_value = mock_response
        mock_init_chat_model.return_value = mock_model

        # Set up the ModelManager instance
        manager = ModelManager.get_instance(test_config_path)

        # Initialize the model
        await manager.initialize_model()
        assert manager.model is mock_model
        assert manager.clean_model is mock_model

        # Update the configuration
        new_settings = {
            "model_provider": "new_fake",
            "model": "new_fake_model",
        }
        await manager.save_model_config("new_fake", new_settings)

        # Check that the configuration has been updated
        config = await manager.get_model_config()
        assert config is not None
        assert config.get("activeProvider") == "new_fake"
        assert config.get("configs", {}).get("new_fake") == new_settings

        # Reload the model
        await manager.reload_model()

        # Generate a title
        title = await manager.generate_title("Test content")
        assert title == "Generated Title"

        # Check that the model was called
        mock_model.ainvoke.assert_called()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
