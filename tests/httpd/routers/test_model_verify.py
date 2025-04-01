import json
import os
import re
from contextlib import suppress
from typing import cast

import pytest
from fastapi import status
from openai import AuthenticationError

from tests import helper

MOCK_MODEL_SETTING = {
    "model": "gpt-4o-mini",
    "modelProvider": "openai",
    "apiKey": "openai-api-key",
    "temperature": 0.7,
    "topP": 0.9,
    "maxTokens": 4000,
    "configuration": {"base_url": "https://api.openai.com/v1"},
}


def test_do_verify_model_with_env_api_key(test_client):
    """Test the /api/model_verify POST endpoint."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")
    client, _ = test_client

    # Prepare test data
    test_settings = {
        "modelSettings": {
            **MOCK_MODEL_SETTING,
            "apiKey": os.environ.get("OPENAI_API_KEY"),
        },
    }

    # Send request
    response = client.post(
        "/model_verify",
        json=test_settings,
    )

    # Verify response status code
    assert response.status_code == status.HTTP_200_OK

    # Parse JSON response
    response_data = cast("dict", response.json())

    # Validate response structure
    assert response_data["success"] is True

    # Validate result structure
    assert isinstance(response_data["connectingSuccess"], bool)
    assert isinstance(response_data["supportTools"], bool)


def test_verify_model_streaming_with_env_api_key(test_client):
    """Test the /api/model_verify/streaming POST endpoint."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")
    # Get client and app from test_client fixture
    client, _ = test_client

    # Prepare test data
    test_model_settings = {
        "modelSettings": [
            {
                **MOCK_MODEL_SETTING,
                "apiKey": os.environ.get("OPENAI_API_KEY"),
            },
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
                step = json_obj.get("step")
                test_type = json_obj.get("testType")

                if step == 1 and test_type == "connection":
                    helper.dict_subset(
                        json_obj,
                        {
                            "step": 1,
                            "modelName": "gpt-4o-mini",
                            "testType": "connection",
                            "status": "success",
                            "error": None,
                        },
                    )
                elif step == 2 and test_type == "tools":
                    helper.dict_subset(
                        json_obj,
                        {
                            "step": 2,
                            "modelName": "gpt-4o-mini",
                            "testType": "tools",
                            "status": "success",
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
                                "modelName": "gpt-4o-mini",
                                "connection": {
                                    "status": "success",
                                },
                                "tools": {
                                    "status": "success",
                                },
                            },
                        ],
                    },
                )


def test_do_verify_model_with_mock_key_should_fail(test_client):
    """Test the /api/model_verify POST endpoint with mock API key."""
    client, _ = test_client

    # Prepare test data
    test_settings = {
        "modelSettings": MOCK_MODEL_SETTING,
    }

    with suppress(AuthenticationError):
        client.post(
            "/model_verify",
            json=test_settings,
        )


def test_verify_model_streaming_with_mock_key_should_fail(test_client):
    """Test the /api/model_verify/streaming POST endpoint with mock API key."""
    # Get client and app from test_client fixture
    client, _ = test_client

    # Prepare test data
    test_model_settings = {
        "modelSettings": [
            MOCK_MODEL_SETTING,
        ],
    }

    with suppress(AuthenticationError):
        client.post(
            "/model_verify/streaming",
            json=test_model_settings,
        )
