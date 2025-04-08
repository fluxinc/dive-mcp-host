import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.responses import StreamingResponse

from dive_mcp_host.httpd.routers.openai import (
    CompletionEventStreamContextManager,
    OpenaiModel,
    StreamMessage,
)
from tests import helper


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


def test_get_openai(test_client):
    """Test the / GET endpoint."""
    # Send request
    client, _ = test_client
    response = client.get("/v1/openai/")

    # Verify response status code
    assert response.status_code == status.HTTP_200_OK

    # Parse JSON response
    response_data = cast(dict, response.json())
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "message": "Welcome to Dive Compatible API! 🚀",
        },
    )


def test_list_models(test_client):
    """Test the /models GET endpoint with mocked dive host."""
    # Send request
    client, _ = test_client
    response = client.get("/v1/openai/models")

    # Verify response status code
    assert response.status_code == status.HTTP_200_OK

    # Parse JSON response
    response_data = cast(dict[str, Any], response.json())

    # Validate response structure
    assert response_data["success"] is True
    assert "models" in response_data
    assert isinstance(response_data["models"], list)
    assert len(response_data["models"]) > 0

    # Validate model structure
    model = cast(dict[str, str], response_data["models"][0])
    assert model["id"] == "fake"
    assert model["type"] == "model"
    assert model["owned_by"] == "dive"


@patch("dive_mcp_host.httpd.routers.openai.ChatProcessor")
def test_chat_completions_with_system_message(mock_chat_processor, test_client):
    """Test chat completions with a system message."""
    client, _ = test_client
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

    response = client.post("/v1/openai/chat/completions", json=test_data)

    assert response.status_code == status.HTTP_200_OK

    response_data = cast(dict[str, Any], response.json())
    mock_chat_processor.assert_called_once()

    assert response_data["id"].startswith("chatcmpl-")
    assert response_data["object"] == "chat.completion"
    assert response_data["model"] == "fake"
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
def test_chat_completions_without_system_message(mock_chat_processor, test_client):
    """Test chat completions without a system message."""
    client, _ = test_client
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

    response = client.post("/v1/openai/chat/completions", json=test_data)

    assert response.status_code == status.HTTP_200_OK

    mock_chat_processor.assert_called_once()
    response_data = cast(dict[str, Any], response.json())

    assert response_data["id"].startswith("chatcmpl-")
    assert response_data["object"] == "chat.completion"
    assert response_data["model"] == "fake"
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
def test_chat_completions_with_assistant_message(mock_chat_processor, test_client):
    """Test chat completions with assistant messages included."""
    client, _ = test_client
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

    response = client.post("/v1/openai/chat/completions", json=test_data)

    assert response.status_code == status.HTTP_200_OK

    mock_chat_processor.assert_called_once()
    response_data = cast(dict[str, Any], response.json())

    assert response_data["id"].startswith("chatcmpl-")
    assert response_data["object"] == "chat.completion"
    assert response_data["model"] == "fake"
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
def test_chat_completions_with_tool_choice_none(mock_chat_processor, test_client):
    """Test chat completions with tool_choice=none."""
    client, _ = test_client
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

    response = client.post("/v1/openai/chat/completions", json=test_data)

    assert response.status_code == status.HTTP_200_OK

    mock_chat_processor.assert_called_once()
    response_data = cast(dict[str, Any], response.json())

    assert response_data["id"].startswith("chatcmpl-")
    assert response_data["object"] == "chat.completion"
    assert response_data["model"] == "fake"
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
def test_chat_completions_streaming(
    mock_chat_processor,
    test_client,
    mock_event_stream,
):
    """Test streaming chat completions."""
    client, _ = test_client
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
                    '"model":"fake","choices":[{"index":0,"delta":'
                    '{"content":"Test response"},"finish_reason":null}]}\n\n'
                ),
                (
                    'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
                    '"model":"fake","choices":[{"index":0,"delta":{},'
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

    response = client.post("/v1/openai/chat/completions", json=test_data)

    assert response.status_code == status.HTTP_200_OK
    assert "text/event-stream" in response.headers.get("Content-Type")

    mock_event_stream.add_task.assert_called_once()
    content = response.text

    assert "data: " in content
    assert "chat.completion.chunk" in content
    assert "fake" in content
    assert "Test response" in content
    assert "finish_reason" in content
    assert "data: [DONE]\n\n" in content


@pytest.mark.asyncio
async def test_completion_event_stream_context_manager():
    """Test the CompletionEventStreamContextManager functionality."""
    # Create context manager
    chat_id = "test-chat-id"
    model = "fake"
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


def test_chat_completions_invalid_request(test_client):
    """Test the /chat/completions POST endpoint with invalid request."""
    client, _ = test_client
    # Test with missing messages
    response = client.post("/v1/openai/chat/completions", json={"stream": False})
    # Use 422 status code,
    # as FastAPI returns 422 for missing required fields rather than 400
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Test with invalid message format
    response = client.post(
        "/v1/openai/chat/completions",
        json={"messages": [{"invalid": "format"}], "stream": False},
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_openai_model_serialization():
    """Test that OpenaiModel can be properly serialized."""
    model = OpenaiModel(id="fake", type="model", owned_by="test-provider")
    model_dict = model.model_dump()

    assert "id" in model_dict
    assert isinstance(model_dict["id"], str)
    assert "type" in model_dict
    assert isinstance(model_dict["type"], str)
    assert "owned_by" in model_dict
    assert isinstance(model_dict["owned_by"], str)
