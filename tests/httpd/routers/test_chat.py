import io
import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest
from fastapi import status
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from dive_mcp_host.host.host import DiveMcpHost
from dive_mcp_host.httpd.database.models import Chat, Message
from dive_mcp_host.httpd.routers.models import UserInputError
from tests import helper

from .conftest import TEST_CHAT_ID

# Constants
SUCCESS_CODE = status.HTTP_200_OK
BAD_REQUEST_CODE = status.HTTP_400_BAD_REQUEST


@dataclass
class ChatWithMessages:
    """Mock chat with messages response."""

    chat: Chat
    messages: list[Message]


def test_list_chat(test_client):
    """Test the /api/chat/list endpoint."""
    client, app = test_client

    # Call the API
    response = client.get("/api/chat/list")
    # Verify response status code
    assert response.status_code == SUCCESS_CODE
    # Parse JSON response
    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "data": [
                {
                    "id": TEST_CHAT_ID,
                    "title": "I am a fake model.",
                    "user_id": None,
                }
            ],
        },
    )


def test_get_chat(test_client):
    """Test the /api/chat/{chat_id} endpoint."""
    client, app = test_client

    # Send request
    response = client.get(f"/api/chat/{TEST_CHAT_ID}")

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Parse JSON response
    response_data = response.json()

    helper.dict_subset(
        response_data,
        {
            "success": True,
            "data": {
                "chat": {
                    "id": TEST_CHAT_ID,
                    "title": "I am a fake model.",
                    "user_id": None,
                },
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, world!",
                        "chatId": TEST_CHAT_ID,
                        "id": 1,
                        "files": "[]",
                    },
                    {
                        "role": "assistant",
                        "content": "I am a fake model.",
                        "chatId": TEST_CHAT_ID,
                        "id": 2,
                    },
                ],
            },
        },
    )


def test_delete_chat(test_client):
    """Test the /api/chat/{chat_id} DELETE endpoint."""
    client, app = test_client

    chat_id = uuid.uuid4()
    # create a chat
    client.post("/api/chat", data={"message": "Hello, world!", "chatId": chat_id})

    # Send request
    response = client.delete(f"/api/chat/{chat_id}")

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert "success" in response_data
    assert response_data["success"] is True


def test_abort_chat(test_client):
    """Test the /api/chat/{chat_id}/abort endpoint."""
    client, app = test_client
    # fake model sleep few seconds
    app.dive_host["default"]._model.sleep = 3  # type: ignore

    # abort a non-existent chat
    with pytest.raises(UserInputError) as excinfo:
        client.post("/api/chat/00000000-0000-0000-0000-000000000000/abort")
    assert "Chat not found" in str(excinfo.value)

    fake_id = uuid.uuid4()

    def create_chat():
        response = client.post(
            "/api/chat",
            data={"message": "long long time", "chatId": fake_id},
        )
        line = next(response.iter_lines())
        message = json.loads(line[5:])["message"]  # type: ignore
        assert message["content"]["id"] == fake_id

    with ThreadPoolExecutor(1) as executor:
        executor.submit(create_chat)
        time.sleep(2)

        abort_response = client.post(f"/api/chat/{fake_id}/abort")
        assert abort_response.status_code == SUCCESS_CODE
        abort_message = abort_response.json()
        assert abort_message["success"]  # type: ignore


def test_create_chat(test_client):
    """Test the /api/chat POST endpoint."""
    client, app = test_client

    chat_id = str(uuid.uuid4())

    test_file = io.BytesIO(b"test file content")
    response = client.post(
        "/api/chat",
        data={
            "chatId": chat_id,
            "message": "test message",
            "filepaths": ["test_path.txt"],
        },
        files={"files": ("test.txt", test_file, "text/plain")},
    )

    assert response.status_code == SUCCESS_CODE
    assert "text/event-stream" in response.headers["Content-Type"]

    content = response.text

    # assert the basic format
    assert "data: " in content
    assert "data: [DONE]\n\n" in content

    # extract and parse the JSON data
    data_messages = re.findall(r"data: (.*?)\n\n", content)
    for data in data_messages:
        if data != "[DONE]":
            # parse the outer JSON
            json_obj = json.loads(data)
            assert "message" in json_obj

            # parse the inner JSON string
            if json_obj["message"]:
                inner_json = json.loads(json_obj["message"])
                assert "type" in inner_json
                assert "content" in inner_json

                # assert the specific type of message
                if inner_json["type"] == "chat_info":
                    helper.dict_subset(
                        inner_json["content"],
                        {
                            "id": "test-chat-id",
                            "title": "New Chat",
                        },
                    )
                elif inner_json["type"] == "message_info":
                    helper.dict_subset(
                        inner_json["content"],
                        {
                            "userMessageId": "test-user-msg",
                            "assistantMessageId": "test-ai-msg",
                        },
                    )


def test_edit_chat(test_client):
    """Test the /api/chat/edit endpoint."""
    client, app = test_client
    test_file = io.BytesIO(b"test file content")
    test_chat_id = "test_edit_chat"
    response = client.post(
        "/api/chat",
        data={
            "chatId": test_chat_id,
            "message": "test message",
            "filepaths": ["test_path.txt"],
        },
        files={"files": ("test.txt", test_file, "text/plain")},
    )
    assert response.status_code == SUCCESS_CODE
    assert response.headers.get("Content-Type").startswith("text/event-stream")

    user_message_id = ""
    ai_message_id = ""
    fist_ai_reply = ""
    for json_obj in helper.extract_stream(response.text):
        content = json_obj["message"]["content"]
        match json_obj["message"]["type"]:
            case "chat_info":
                assert content["id"] == test_chat_id  # type: ignore
            case "message_info":
                user_message_id = content["userMessageId"]  # type: ignore
                ai_message_id = content["assistantMessageId"]  # type: ignore
                assert user_message_id
                assert ai_message_id
            case "text":
                fist_ai_reply = content

    ai_messages = [
        AIMessage(content="message 1"),
        AIMessage(content="message 2"),
    ]
    host = cast("dict[str, DiveMcpHost]", app.dive_host)["default"]
    host.model.responses = ai_messages  # type: ignore
    response = client.post(
        "/api/chat/edit",
        data={
            "chatId": test_chat_id,
            "messageId": user_message_id,
            "content": "edited message",
            "filepaths": ["test_edit_path.txt"],
        },
        files={"files": ("test_edit.txt", test_file, "text/plain")},
    )

    assert response.status_code == SUCCESS_CODE
    assert response.headers.get("Content-Type").startswith("text/event-stream")

    new_user_message_id = ""
    new_ai_message_id = ""
    for json_obj in helper.extract_stream(response.text):
        content = json_obj["message"]["content"]
        match json_obj["message"]["type"]:
            case "chat_info":
                assert content["id"] == test_chat_id  # type: ignore
            case "message_info":
                new_user_message_id = content["userMessageId"]  # type: ignore
                new_ai_message_id = content["assistantMessageId"]  # type: ignore
                assert new_user_message_id
                assert new_ai_message_id
                assert new_ai_message_id != ai_message_id
                assert new_user_message_id == user_message_id
            case "text":
                assert fist_ai_reply != content
    response = client.get(f"/api/chat/{test_chat_id}")
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "data": {
                "messages": [
                    {"messageId": new_user_message_id},
                    {"messageId": new_ai_message_id},
                ]
            },
        },
    )


def test_edit_chat_missing_params(test_client):
    """Test the /api/chat/edit endpoint with missing required parameters."""
    client, app = test_client
    with pytest.raises(UserInputError) as excinfo:
        client.post(
            "/api/chat/edit",
            data={
                "content": "edited message",
            },
        )
    assert "Chat ID and Message ID are required" in str(excinfo.value)


def test_retry_chat(test_client):
    """Test the /api/chat/retry endpoint."""
    client, app = test_client

    # get message id
    response = client.get(f"/api/chat/{TEST_CHAT_ID}")
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    message_id = response_data["data"]["messages"][0]["id"]

    response = client.post(
        "/api/chat/retry",
        data={
            "chatId": TEST_CHAT_ID,
            "messageId": message_id,
        },
    )

    assert response.status_code == SUCCESS_CODE
    assert "text/event-stream" in response.headers["Content-Type"]

    content = response.text

    # assert the basic format
    assert "data: " in content
    assert "data: [DONE]\n\n" in content

    has_chat_info = False
    has_message_info = False

    # extract and parse the JSON data
    data_messages = re.findall(r"data: (.*?)\n\n", content)
    for data in data_messages:
        if data != "[DONE]":
            # parse the outer JSON
            json_obj = json.loads(data)
            assert "message" in json_obj

            # parse the inner JSON string
            if json_obj["message"]:
                inner_json = json.loads(json_obj["message"])
                assert "type" in inner_json
                assert "content" in inner_json

                # assert the specific type of message
                if inner_json["type"] == "chat_info":
                    has_chat_info = True
                    helper.dict_subset(
                        inner_json["content"],
                        {
                            "id": TEST_CHAT_ID,
                            "title": "New Chat",
                        },
                    )
                if inner_json["type"] == "message_info":
                    has_message_info = True
                    helper.dict_subset(
                        inner_json["content"],
                        {
                            "userMessageId": "test-user-msg",
                            "assistantMessageId": "test-ai-msg",
                        },
                    )

    assert has_chat_info
    assert has_message_info


def test_retry_chat_missing_params(test_client):
    """Test the /api/chat/retry endpoint with missing required parameters."""
    client, app = test_client
    with pytest.raises(UserInputError) as excinfo:
        client.post("/api/chat/retry", data={})

    # Verify the exception message
    assert "Chat ID and Message ID are required" in str(excinfo.value)
