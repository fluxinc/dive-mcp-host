import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from dive_mcp_host.httpd.app import DiveHostAPI
from dive_mcp_host.httpd.dependencies import get_app
from dive_mcp_host.httpd.routers.openai import (
    CompletionEventStreamContextManager,
    OpenaiModel,
    StreamMessage,
    openai,
)


class MagicDict(dict):
    """Custom dictionary class that simulates the values method of dive_host."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.values_mock = MagicMock()

    def values(self):
        """Simulate the values method of a dictionary."""
        return self.values_mock()


class MockDiveHostAPI:
    """Mock DiveHostAPI FastAPI application for testing."""

    def __init__(self):
        super().__init__()

        # Create mock dive_host
        mock_config = MagicMock()
        mock_config.llm.model = "test-model"
        mock_config.llm.modelProvider = "test-provider"

        mock_dive_host = MagicMock()
        mock_dive_host._config = mock_config

        # Use custom MagicDict
        magic_dict = MagicDict(default=mock_dive_host)
        magic_dict.values_mock.return_value = [mock_dive_host]

        # Add dive_host attribute
        self.dive_host = magic_dict

        # Add prompt_config_manager attribute
        self.prompt_config_manager = MagicMock()
        self.prompt_config_manager.get_prompt.return_value = "This is a system prompt"


@pytest.fixture
def client():
    """Create a test client with the mock app."""
    app = DiveHostAPI()
    app.include_router(openai)

    mock_app = MockDiveHostAPI()

    def get_mock_app():
        return mock_app

    app.dependency_overrides[get_app] = get_mock_app

    with TestClient(app) as client:
        yield client


@pytest.fixture(autouse=True)
def mock_event_stream():
    """Mock EventStreamContextManager to prevent tests from hanging."""
    mock_instance = MagicMock()
    mock_instance.queue = asyncio.Queue()
    mock_instance.get_response.return_value = StreamingResponse(
        content=iter(["data: [DONE]\n\n"]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
    mock_instance.__aenter__.return_value = mock_instance

    # Add mock for async methods
    async def mock_write(*args, **kwargs):
        return None

    mock_instance.write = mock_write

    with (
        patch(
            "dive_mcp_host.httpd.routers.utils.EventStreamContextManager",
            return_value=mock_instance,
        ),
        patch(
            "dive_mcp_host.httpd.routers.openai.CompletionEventStreamContextManager",
            return_value=mock_instance,
        ),
    ):
        yield mock_instance


def test_get_openai(client):
    """Test the / GET endpoint."""
    # Send request
    response = client.get("/")

    # Verify response status code
    assert response.status_code == status.HTTP_200_OK

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert "success" in response_data
    assert isinstance(response_data["success"], bool)
    assert "message" in response_data
    assert isinstance(response_data["message"], str)


def test_list_models(client):
    """Test the /models GET endpoint with mocked dive host."""
    # Send request
    response = client.get("/models")

    # Verify response status code
    assert response.status_code == status.HTTP_200_OK

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert response_data["success"] is True
    assert "models" in response_data
    assert isinstance(response_data["models"], list)
    assert len(response_data["models"]) > 0

    # Validate model structure
    model = response_data["models"][0]
    assert model["id"] == "test-model"
    assert model["type"] == "model"
    assert model["owned_by"] == "test-provider"


@patch("dive_mcp_host.httpd.routers.openai.ChatProcessor")
def test_chat_completions_with_system_message(mock_chat_processor, client):
    """Test chat completions with a system message."""
    # Setup mock
    processor_instance = AsyncMock()
    processor_instance.handle_chat_with_history.return_value = (
        "This is a test response",
        MagicMock(total_input_tokens=10, total_output_tokens=20, total_tokens=30),
    )
    mock_chat_processor.return_value = processor_instance

    test_data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        "stream": False,
        "tool_choice": "auto",
    }

    response = client.post("/chat/completions", json=test_data)

    assert response.status_code == status.HTTP_200_OK

    response_data = response.json()
    mock_chat_processor.assert_called_once()

    assert "id" in response_data
    assert response_data["id"].startswith("chatcmpl-")
    assert response_data["object"] == "chat.completion"
    assert response_data["model"] == "test-model"
    assert "choices" in response_data
    assert len(response_data["choices"]) == 1
    assert response_data["choices"][0]["index"] == 0
    assert response_data["choices"][0]["message"]["role"] == "assistant"
    assert (
        response_data["choices"][0]["message"]["content"] == "This is a test response"
    )
    assert response_data["choices"][0]["finish_reason"] == "stop"
    assert "usage" in response_data
    assert response_data["usage"]["prompt_tokens"] == 10
    assert response_data["usage"]["completion_tokens"] == 20
    assert response_data["usage"]["total_tokens"] == 30
    assert response_data["system_fingerprint"] == "fp_dive"


@patch("dive_mcp_host.httpd.routers.openai.ChatProcessor")
def test_chat_completions_without_system_message(mock_chat_processor, client):
    """Test chat completions without a system message."""
    # Setup mock
    processor_instance = AsyncMock()
    processor_instance.handle_chat_with_history.return_value = (
        "This is a test response",
        MagicMock(total_input_tokens=10, total_output_tokens=20, total_tokens=30),
    )
    mock_chat_processor.return_value = processor_instance

    # Prepare test data - no system message
    test_data = {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
        ],
        "stream": False,
        "tool_choice": "auto",
    }

    response = client.post("/chat/completions", json=test_data)

    assert response.status_code == status.HTTP_200_OK

    mock_chat_processor.assert_called_once()
    response_data = response.json()

    assert "id" in response_data
    assert response_data["id"].startswith("chatcmpl-")
    assert response_data["object"] == "chat.completion"
    assert response_data["model"] == "test-model"
    assert "choices" in response_data
    assert len(response_data["choices"]) == 1
    assert response_data["choices"][0]["index"] == 0
    assert response_data["choices"][0]["message"]["role"] == "assistant"
    assert (
        response_data["choices"][0]["message"]["content"] == "This is a test response"
    )
    assert response_data["choices"][0]["finish_reason"] == "stop"
    assert "usage" in response_data
    assert response_data["usage"]["prompt_tokens"] == 10
    assert response_data["usage"]["completion_tokens"] == 20
    assert response_data["usage"]["total_tokens"] == 30
    assert response_data["system_fingerprint"] == "fp_dive"


@patch("dive_mcp_host.httpd.routers.openai.ChatProcessor")
def test_chat_completions_with_assistant_message(mock_chat_processor, client):
    """Test chat completions with assistant messages included."""
    # Setup mock
    processor_instance = AsyncMock()
    processor_instance.handle_chat_with_history.return_value = (
        "This is a test response",
        MagicMock(total_input_tokens=15, total_output_tokens=25, total_tokens=40),
    )
    mock_chat_processor.return_value = processor_instance

    # Prepare test data - including assistant message
    test_data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help you?"},
            {"role": "user", "content": "Can you tell me about the weather today?"},
        ],
        "stream": False,
        "tool_choice": "auto",
    }

    response = client.post("/chat/completions", json=test_data)

    assert response.status_code == status.HTTP_200_OK

    mock_chat_processor.assert_called_once()
    response_data = response.json()

    assert "id" in response_data
    assert response_data["id"].startswith("chatcmpl-")
    assert response_data["object"] == "chat.completion"
    assert response_data["model"] == "test-model"
    assert "choices" in response_data
    assert len(response_data["choices"]) == 1
    assert response_data["choices"][0]["index"] == 0
    assert response_data["choices"][0]["message"]["role"] == "assistant"
    assert (
        response_data["choices"][0]["message"]["content"] == "This is a test response"
    )
    assert response_data["choices"][0]["finish_reason"] == "stop"
    assert "usage" in response_data
    assert response_data["usage"]["prompt_tokens"] == 15
    assert response_data["usage"]["completion_tokens"] == 25
    assert response_data["usage"]["total_tokens"] == 40
    assert response_data["system_fingerprint"] == "fp_dive"


@patch("dive_mcp_host.httpd.routers.openai.ChatProcessor")
def test_chat_completions_with_tool_choice_none(mock_chat_processor, client):
    """Test chat completions with tool_choice=none."""
    # Setup mock
    processor_instance = AsyncMock()
    processor_instance.handle_chat_with_history.return_value = (
        "This is a test response",
        MagicMock(total_input_tokens=10, total_output_tokens=20, total_tokens=30),
    )
    mock_chat_processor.return_value = processor_instance

    # Prepare test data
    test_data = {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
        ],
        "stream": False,
        "tool_choice": "none",
    }

    response = client.post("/chat/completions", json=test_data)

    assert response.status_code == status.HTTP_200_OK

    mock_chat_processor.assert_called_once()
    response_data = response.json()

    assert "id" in response_data
    assert response_data["id"].startswith("chatcmpl-")
    assert response_data["object"] == "chat.completion"
    assert response_data["model"] == "test-model"
    assert "choices" in response_data
    assert len(response_data["choices"]) == 1
    assert response_data["choices"][0]["index"] == 0
    assert response_data["choices"][0]["message"]["role"] == "assistant"
    assert (
        response_data["choices"][0]["message"]["content"] == "This is a test response"
    )
    assert response_data["choices"][0]["finish_reason"] == "stop"
    assert "usage" in response_data
    assert response_data["usage"]["prompt_tokens"] == 10
    assert response_data["usage"]["completion_tokens"] == 20
    assert response_data["usage"]["total_tokens"] == 30
    assert response_data["system_fingerprint"] == "fp_dive"


@patch("dive_mcp_host.httpd.routers.openai.ChatProcessor")
def test_chat_completions_streaming(mock_chat_processor, client, mock_event_stream):
    """Test streaming chat completions."""
    # Setup mock
    processor_instance = AsyncMock()
    processor_instance.handle_chat_with_history.return_value = (
        "This is a test response",
        MagicMock(total_input_tokens=10, total_output_tokens=20, total_tokens=30),
    )
    mock_chat_processor.return_value = processor_instance

    # Mock get_response return
    mock_event_stream.get_response.return_value = StreamingResponse(
        content=iter(
            [
                (
                    'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
                    '"model":"test-model","choices":[{"index":0,"delta":'
                    '{"content":"Test response"},"finish_reason":null}]}\n\n'
                ),
                (
                    'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
                    '"model":"test-model","choices":[{"index":0,"delta":{},'
                    '"finish_reason":"stop"}]}\n\n'
                ),
                "data: [DONE]\n\n",
            ]
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

    test_data = {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
        ],
        "stream": True,
        "tool_choice": "auto",
    }

    response = client.post("/chat/completions", json=test_data)

    assert response.status_code == status.HTTP_200_OK
    assert "text/event-stream" in response.headers["Content-Type"]

    mock_event_stream.add_task.assert_called_once()
    content = response.text

    assert "data: " in content
    assert "chat.completion.chunk" in content
    assert "test-model" in content
    assert "Test response" in content
    assert "finish_reason" in content
    assert "data: [DONE]\n\n" in content


@pytest.mark.asyncio
async def test_completion_event_stream_context_manager():
    """Test the CompletionEventStreamContextManager functionality."""
    # Create context manager
    chat_id = "test-chat-id"
    model = "test-model"
    stream = CompletionEventStreamContextManager(chat_id, model)

    # Mock response writing
    with patch.object(stream, "write") as mock_write:
        # Test writing text content
        text_message = StreamMessage(type="text", content="Hello")
        await stream.write(text_message)
        mock_write.assert_called_once()

        # Get arguments passed to write
        called_with = mock_write.call_args[0][0]
        assert isinstance(called_with, StreamMessage)
        assert called_with.type == "text"
        assert called_with.content == "Hello"


def test_chat_completions_invalid_request(client):
    """Test the /chat/completions POST endpoint with invalid request."""
    # Test with missing messages
    response = client.post("/chat/completions", json={"stream": False})
    # Use 422 status code,
    # as FastAPI returns 422 for missing required fields rather than 400
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Test with invalid message format
    response = client.post(
        "/chat/completions",
        json={"messages": [{"invalid": "format"}], "stream": False},
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_openai_model_serialization():
    """Test that OpenaiModel can be properly serialized."""
    model = OpenaiModel(id="test-model", type="model", owned_by="test-provider")
    model_dict = model.model_dump()

    assert "id" in model_dict
    assert isinstance(model_dict["id"], str)
    assert "type" in model_dict
    assert isinstance(model_dict["type"], str)
    assert "owned_by" in model_dict
    assert isinstance(model_dict["owned_by"], str)
