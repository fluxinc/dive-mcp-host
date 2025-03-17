import json

import httpx
import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from dive_mcp_host.httpd.routers.openai import OpenaiModel, openai

client_type = "fastapi"


@pytest.fixture
def client(request):
    """Create a test client.

    Args:
        request: The pytest request object.

    Returns:
        A TestClient for FastAPI testing or httpx.Client for direct Node.js testing.
    """
    client_type = getattr(request.module, "client_type", "fastapi")

    if client_type == "nodejs":
        return httpx.Client(base_url="http://localhost:4321/api")
    app = FastAPI()
    app.include_router(openai)
    return TestClient(app)


def test_get_openai(client):
    """Test the /v1/openai GET endpoint."""
    # Send request
    response = client.get("/v1/openai")

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
    """Test the /v1/openai/models GET endpoint."""
    # Send request
    response = client.get("/v1/openai/models")

    # Verify response status code
    assert response.status_code == status.HTTP_200_OK

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert "success" in response_data
    assert isinstance(response_data["success"], bool)
    assert "data" in response_data
    assert isinstance(response_data["data"], list)

    # If there are models, check the structure of the first model
    if response_data["data"]:
        model = response_data["data"][0]
        assert "id" in model
        assert isinstance(model["id"], str)
        assert "type" in model
        assert isinstance(model["type"], str)
        assert "owned_by" in model
        assert isinstance(model["owned_by"], str)


def test_chat_completions(client):
    """Test the /v1/openai/chat/completions POST endpoint."""
    # Prepare test data
    test_data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        "stream": False,
        "tool_choice": "auto",
    }

    # Send request
    response = client.post("/v1/openai/chat/completions", json=test_data)

    # Verify response status code
    assert response.status_code == status.HTTP_200_OK

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert "id" in response_data
    assert isinstance(response_data["id"], str)
    assert "object" in response_data
    assert isinstance(response_data["object"], str)
    assert "created" in response_data
    assert isinstance(response_data["created"], int)
    assert "model" in response_data
    assert isinstance(response_data["model"], str)
    assert "choices" in response_data
    assert isinstance(response_data["choices"], list)

    # Validate choices structure
    if response_data["choices"]:
        choice = response_data["choices"][0]
        assert "index" in choice
        assert isinstance(choice["index"], int)
        assert "message" in choice
        assert "role" in choice["message"]
        assert isinstance(choice["message"]["role"], str)
        assert "content" in choice["message"]
        assert isinstance(choice["message"]["content"], str)
        assert "finish_reason" in choice

    # Validate usage structure
    assert "usage" in response_data
    usage = response_data["usage"]
    assert "prompt_tokens" in usage
    assert isinstance(usage["prompt_tokens"], int)
    assert "completion_tokens" in usage
    assert isinstance(usage["completion_tokens"], int)
    assert "total_tokens" in usage
    assert isinstance(usage["total_tokens"], int)


def test_chat_completions_streaming(client):
    """Test the /v1/openai/chat/completions POST endpoint with streaming."""
    # Prepare test data
    test_data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        "stream": True,
        "tool_choice": "auto",
    }

    # Send request
    response = client.post("/v1/openai/chat/completions", json=test_data)

    # Verify response status code
    assert response.status_code == status.HTTP_200_OK

    # Verify response headers for SSE
    assert "text/event-stream" in response.headers["Content-Type"]
    assert "Cache-Control" in response.headers
    assert "Connection" in response.headers

    # Verify content contains expected SSE format
    content = response.content.decode()
    assert "data: " in content

    # Parse the first chunk to validate structure
    lines = content.split("\n\n")
    for line in lines:
        if line.startswith("data: ") and not line.startswith("data: [DONE]"):
            # Remove 'data: ' prefix and parse JSON
            chunk_str = line[6:]  # Skip 'data: '
            if chunk_str:
                try:
                    chunk = json.loads(chunk_str)

                    # Validate chunk structure
                    assert "id" in chunk
                    assert isinstance(chunk["id"], str)
                    assert "object" in chunk
                    assert "created" in chunk
                    assert isinstance(chunk["created"], int)
                    assert "model" in chunk
                    assert "choices" in chunk
                    assert isinstance(chunk["choices"], list)

                    # Only validate the first valid chunk
                    if chunk["choices"]:
                        choice = chunk["choices"][0]
                        assert "index" in choice
                        assert isinstance(choice["index"], int)
                        assert "delta" in choice
                        # Either role or content or both should be in delta
                        if "role" in choice["delta"]:
                            assert isinstance(choice["delta"]["role"], str)
                        if "content" in choice["delta"]:
                            assert isinstance(choice["delta"]["content"], str)

                    # Only need to check one valid chunk
                    break
                except json.JSONDecodeError:
                    continue

    # Check for completion signal
    assert '"finish_reason":"stop"' in content


def test_chat_completions_invalid_request(client):
    """Test the /v1/openai/chat/completions POST endpoint with invalid request."""
    # Test with missing messages
    response = client.post("/v1/openai/chat/completions", json={"stream": False})
    assert response.status_code == status.HTTP_400_BAD_REQUEST

    # Test with invalid message format
    response = client.post(
        "/v1/openai/chat/completions",
        json={"messages": [{"invalid": "format"}], "stream": False},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_openai_model_serialization():
    """Test that OpenaiModel can be properly serialized."""
    # Create a sample model
    model = OpenaiModel(id="test-model", type="model", owned_by="test-provider")

    # Convert to dict
    model_dict = model.model_dump()

    # Validate structure
    assert "id" in model_dict
    assert isinstance(model_dict["id"], str)
    assert "type" in model_dict
    assert isinstance(model_dict["type"], str)
    assert "owned_by" in model_dict
    assert isinstance(model_dict["owned_by"], str)
