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
                        "files": [],
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
    assert isinstance(response_data, dict)
    assert response_data["success"] is True


def test_abort_chat(test_client):
    """Test the /api/chat/{chat_id}/abort endpoint."""
    client, app = test_client
    # fake model sleep few seconds
    app.dive_host["default"]._model.sleep = 3  # type: ignore

    # abort a non-existent chat
    response = client.post("/api/chat/00000000-0000-0000-0000-000000000000/abort")
    assert response.status_code == BAD_REQUEST_CODE
    body = response.json()
    assert "Chat not found" in body["message"]

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
    assert "text/event-stream" in response.headers.get("Content-Type")

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
                            "id": chat_id,
                        },
                    )
                    assert "title" in inner_json["content"]
                if inner_json["type"] == "message_info":
                    has_message_info = True
                    assert "userMessageId" in inner_json["content"]
                    assert "assistantMessageId" in inner_json["content"]

    assert has_chat_info
    assert has_message_info


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
        message = json.loads(json_obj["message"])  # type: ignore
        content = message["content"]
        match message["type"]:
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
        message = json.loads(json_obj["message"])  # type: ignore
        content = message["content"]
        match message["type"]:
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
    assert new_ai_message_id
    assert new_user_message_id
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
    response = client.post(
        "/api/chat/edit",
        data={
            "content": "edited message",
        },
    )
    assert response.status_code == BAD_REQUEST_CODE
    body = response.json()
    assert "Chat ID and Message ID are required" in body["message"]


def test_retry_chat(test_client):
    """Test the /api/chat/retry endpoint."""
    client, app = test_client

    # get message id
    response = client.get(f"/api/chat/{TEST_CHAT_ID}")
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    message_id = response_data["data"]["messages"][0]["messageId"]  # type: ignore

    response = client.post(
        "/api/chat/retry",
        json={
            "chatId": TEST_CHAT_ID,
            "messageId": message_id,
        },
    )

    assert response.status_code == SUCCESS_CODE
    assert "text/event-stream" in response.headers.get("Content-Type")

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
                        },
                    )
                    assert "title" in inner_json["content"]
                if inner_json["type"] == "message_info":
                    has_message_info = True
                    assert "userMessageId" in inner_json["content"]
                    assert "assistantMessageId" in inner_json["content"]

    assert has_chat_info
    assert has_message_info


def test_retry_chat_missing_params(test_client):
    """Test the /api/chat/retry endpoint with missing required parameters."""
    client, app = test_client
    response = client.post("/api/chat/retry", data={})
    assert response.status_code == BAD_REQUEST_CODE

    body = response.json()
    # Verify the exception message
    assert "Chat ID and Message ID are required" in body["message"]


def test_chat_with_tool_calls(test_client, monkeypatch):  # noqa: C901, PLR0915
    """Test the chat endpoint with tool calls."""
    client, app = test_client

    # Import necessary message types
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    # 模擬 query 方法來產生工具呼叫和工具結果
    def mock_query(*args, **kwargs):
        async def title_generator():
            yield {"agent": {"messages": [AIMessage(content="Calculate 2+2")]}}

        if kwargs.get("stream_mode") == "updates":
            return title_generator()

        # 創建模擬的響應生成器
        async def response_generator():
            # 模擬工具呼叫
            yield (
                "messages",
                (
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "calculator",
                                "args": {"expression": "2+2"},
                                "id": "tool-call-id",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    None,
                ),
            )

            # 模擬工具結果
            yield (
                "messages",
                (
                    ToolMessage(
                        content=json.dumps(4),
                        name="calculator",
                        tool_call_id="tool-call-id",
                    ),
                    None,
                ),
            )

            # 模擬最終回應
            yield (
                "messages",
                (
                    AIMessage(
                        content="The result of 2+2 is 4.",
                    ),
                    None,
                ),
            )

            # 模擬 values 響應
            user_message = HumanMessage(content="Calculate 2+2", id="user-msg-id")
            ai_message = AIMessage(
                content="The result of 2+2 is 4.",
                id="assistant-msg-id",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 15,
                    "total_tokens": 25,
                },
            )

            current_messages = [
                user_message,
                AIMessage(
                    content="",
                    id="tool-call-msg-id",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"expression": "2+2"},
                            "id": "tool-call-id",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content=json.dumps(4),
                    name="calculator",
                    id="tool-result-msg-id",
                    tool_call_id="tool-call-id",
                ),
                ai_message,
            ]

            yield "values", {"messages": current_messages}

        return response_generator()

    # 應用 monkeypatch
    monkeypatch.setattr("dive_mcp_host.host.chat.Chat.query", mock_query)

    chat_id = str(uuid.uuid4())

    # Make the request
    response = client.post(
        "/api/chat", data={"chatId": chat_id, "message": "Calculate 2+2"}
    )

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Get the content
    content = response.content.decode("utf-8")

    # Assert the basic format
    assert "data: " in content
    assert "data: [DONE]\n\n" in content

    # Extract and parse the JSON data
    data_messages = re.findall(r"data: (.*?)\n\n", content)

    has_tool_calls = False
    has_tool_result = False
    has_text_response = False
    has_chat_info = False
    has_message_info = False

    for data in data_messages:
        if data != "[DONE]":
            # Parse the outer JSON
            json_obj = json.loads(data)
            assert "message" in json_obj

            # Parse the inner JSON string
            if json_obj["message"]:
                inner_json = json.loads(json_obj["message"])
                assert "type" in inner_json
                assert "content" in inner_json

                # Check for tool calls
                if inner_json["type"] == "tool_calls":
                    has_tool_calls = True
                    tool_call = inner_json["content"][0]
                    assert tool_call["name"] == "calculator"
                    assert "arguments" in tool_call

                # Check for tool result
                if inner_json["type"] == "tool_result":
                    has_tool_result = True
                    tool_result = inner_json["content"]
                    assert tool_result["name"] == "calculator"
                    assert "result" in tool_result

                # Check for text response
                if inner_json["type"] == "text":
                    has_text_response = True
                    assert "The result of 2+2 is 4." in inner_json["content"]

                # Check for chat info
                if inner_json["type"] == "chat_info":
                    has_chat_info = True
                    assert inner_json["content"]["id"] == chat_id
                    assert "title" in inner_json["content"]

                # Check for message info
                if inner_json["type"] == "message_info":
                    has_message_info = True
                    assert "userMessageId" in inner_json["content"]
                    assert "assistantMessageId" in inner_json["content"]

    # Verify all message types were received
    assert has_tool_calls, "Tool calls message not found"
    assert has_tool_result, "Tool result message not found"
    assert has_text_response, "Text response message not found"
    assert has_chat_info, "Chat info message not found"
    assert has_message_info, "Message info message not found"

    # Check that the messages were stored in the database
    with ThreadPoolExecutor() as executor:
        future = executor.submit(lambda: client.get(f"/api/chat/{chat_id}"))
        response = future.result()

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Parse the response
    response_data = response.json()
    assert response_data["success"] is True
    assert response_data["data"] is not None

    # Check for tool call and tool result messages in history
    has_tool_call_msg = False
    has_tool_result_msg = False
    has_assistant_msg = False

    for msg in response_data["data"]["messages"]:
        if msg["role"] == "tool_call":
            has_tool_call_msg = True
            tool_call_content = json.loads(msg["content"])
            assert tool_call_content[0]["name"] == "calculator"
            assert "args" in tool_call_content[0]

        if msg["role"] == "tool_result":
            has_tool_result_msg = True
            tool_result_content = json.loads(msg["content"])
            assert tool_result_content == "4"

        if msg["role"] == "assistant" and "The result of 2+2 is 4." in msg["content"]:
            has_assistant_msg = True

    assert has_tool_call_msg, "Tool call message not found in database"
    assert has_tool_result_msg, "Tool result message not found in database"
    assert has_assistant_msg, "Assistant message not found in database"
