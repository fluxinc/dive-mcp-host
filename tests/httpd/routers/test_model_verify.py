import json
import re

from fastapi import status

from tests import helper

MOCK_MODEL_SETTING = {
    "model": "gpt-4o-mini",
    "modelProvider": "openai",
    "apiKey": "openai-api-key",
    "temperature": 0.7,
    "topP": 0.9,
    "maxTokens": 4000,
    "configuration": {"baseURL": "https://api.openai.com/v1"},
}


def test_do_verify_model(test_client):
    """Test the /api/model_verify POST endpoint."""
    client, _ = test_client

    # Prepare test data
    test_settings = {
        "modelSettings": MOCK_MODEL_SETTING,
    }

    # Send request
    response = client.post(
        "/model_verify",
        json=test_settings,
    )

    # Verify response status code
    assert response.status_code == status.HTTP_200_OK

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert response_data["success"] is True

    # Validate result structure
    assert isinstance(response_data["connectingSuccess"], bool)
    assert isinstance(response_data["supportTools"], bool)


def test_verify_model_streaming(test_client):
    """Test the /api/model_verify/streaming POST endpoint."""
    # Get client and app from test_client fixture
    client, _ = test_client

    # Prepare test data
    test_model_settings = {
        "modelSettings": [
            MOCK_MODEL_SETTING,
        ],
    }

    # Send request
    response = client.post(
        "/model_verify/streaming",
        json=test_model_settings,
    )

    # Verify response status code for SSE stream
    assert response.status_code == status.HTTP_200_OK

    # Verify response headers for SSE
    assert "text/event-stream" in response.headers["Content-Type"]
    assert "Cache-Control" in response.headers
    assert "Connection" in response.headers

    # Verify content contains expected SSE format
    content = response.content.decode()
    # assert the basic format
    assert "data: " in content
    assert "data: [DONE]\n\n" in content

    # extract and parse the JSON data
    data_messages = re.findall(r"data: (.*?)\n\n", content)
    for data in data_messages:
        if data != "[DONE]":
            json_obj = json.loads(data)
            assert "type" in json_obj

            if json_obj["type"] == "progress":
                helper.dict_subset(
                    json_obj,
                    {
                        "step": 1,
                        "modelName": "fake",
                        "testType": "tools",
                        "status": "error",
                        "error": None,
                    },
                )

            elif json_obj["type"] == "final":
                helper.dict_subset(
                    json_obj,
                    {
                        "type": "final",
                        "results": [
                            {
                                "modelName": "fake",
                                "connection": {
                                    "status": "success",
                                },
                                "tools": {
                                    "status": "error",
                                },
                            },
                        ],
                    },
                )
