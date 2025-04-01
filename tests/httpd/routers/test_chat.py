import asyncio
import io
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from unittest import mock

import pytest
from fastapi import status
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from dive_mcp_host.host.host import DiveMcpHost
from dive_mcp_host.httpd.app import DiveHostAPI
from dive_mcp_host.httpd.conf.service.manager import ServiceManager
from dive_mcp_host.httpd.database.models import Chat
from dive_mcp_host.httpd.dependencies import get_app, get_dive_user
from dive_mcp_host.httpd.routers.chat import chat
from dive_mcp_host.httpd.routers.models import UserInputError
from dive_mcp_host.httpd.routers.utils import EventStreamContextManager
from tests import helper

# Constants
SUCCESS_CODE = status.HTTP_200_OK
BAD_REQUEST_CODE = status.HTTP_400_BAD_REQUEST
TEST_CHAT_ID = "test_chat_123"
TEST_MESSAGE_ID = "test_message_456"
TEST_USER_ID = "test_user_123"


@dataclass
class Message:
    """Mock message for testing."""

    role: str
    content: str
    chatId: str  # noqa: N815
    messageId: str  # noqa: N815
    id: int
    createdAt: datetime  # noqa: N815
    files: str = "[]"  # Modified to string type to match the actual database model


@dataclass
class ChatWithMessages:
    """Mock chat with messages response."""

    chat: Chat
    messages: list[Message]


class MockDatabase:
    """Mock database object."""

    async def get_all_chats(self, _user_id=TEST_USER_ID, **_kwargs):
        """Get all available chats."""
        return [
            Chat(
                id="mock_chat_1",
                title="Mock Chat 1",
                createdAt=datetime.now(UTC),
                user_id=TEST_USER_ID,
            ),
            Chat(
                id="mock_chat_2",
                title="Mock Chat 2",
                createdAt=datetime.now(UTC),
                user_id=TEST_USER_ID,
            ),
        ]

    async def get_chat_with_messages(self, chat_id, _user_id=TEST_USER_ID, **_kwargs):
        """Get a specific chat with its messages."""
        return ChatWithMessages(
            chat=Chat(
                id=chat_id,
                title="Mock Chat Title",
                createdAt=datetime.now(UTC),
                user_id=TEST_USER_ID,
            ),
            messages=[
                Message(
                    role="user",
                    content="Mock user message",
                    chatId=chat_id,
                    messageId="msg_user",
                    id=1,
                    createdAt=datetime.now(UTC),
                ),
                Message(
                    role="assistant",
                    content="Mock assistant message",
                    chatId=chat_id,
                    messageId="msg_assistant",
                    id=2,
                    createdAt=datetime.now(UTC),
                ),
            ],
        )

    async def delete_chat(self, _chat_id, _user_id=TEST_USER_ID, **_kwargs):
        """Delete a chat."""
        return True

    async def update_message_content(
        self, _message_id, _query_input, _user_id=TEST_USER_ID, **_kwargs
    ):
        """Update message content."""
        return True

    async def get_next_ai_message(self, chat_id, _message_id, **_kwargs):
        """Get the next AI message."""
        return Message(
            role="assistant",
            content="Mock next AI message",
            chatId=chat_id,
            messageId="next_ai_msg",
            id=3,
            createdAt=datetime.now(UTC),
        )

    async def check_chat_exists(self, _chat_id, **_kwargs):
        """Check if a chat exists."""
        return True

    async def create_chat_with_messages(self, _chat_id, _title, _messages, **_kwargs):
        """Create a chat with messages."""
        return True

    async def create_message(self, _message, **_kwargs):
        """Create a new message."""
        return True

    async def create_chat(self, _chat_id, _title, **_kwargs):
        """Create a new chat."""
        return True

    async def delete_messages_after(self, _chat_id, _message_id, **_kwargs):
        """Delete messages after a specific message."""
        return True


class MockStore:
    """Mock storage object."""

    async def upload_files(self, _files, _filepaths):
        """Mock file upload."""
        return [], []


class MockMcpServerManager:
    """Mock MCP server manager."""

    async def process_chat_message(self, *args, **kwargs):  # noqa: ARG002
        """Mock processing chat message."""
        yield "mock message chunk"  # noqa: RUF100

    async def process_chat_edit(self, *args, **kwargs):  # noqa: ARG002
        """Mock processing chat edit."""
        yield "mock edit chunk"  # noqa: RUF100

    async def process_chat_retry(self, *args, **kwargs):  # noqa: ARG002
        """Mock processing chat retry."""
        yield "mock retry chunk"  # noqa: RUF100

    async def get_tool_to_server_map(self):
        """Mock getting tool to server map."""
        return {}

    async def get_available_tools(self):
        """Mock getting available tools."""
        return []


class MockDiveHostAPI:
    """Mock DiveHostAPI for testing."""

    def __init__(self):
        self.db = MockDatabase()
        self.store = MockStore()
        self.mcp = MockMcpServerManager()

    def db_sessionmaker(self):
        """Return a mock session context manager."""

        class MockSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                pass

            async def commit(self):
                pass

        return MockSession()

    def msg_store(self, _session):
        """Return the database mock."""
        return self.db


@pytest.fixture
def client():
    """Create a test client."""
    service_manager = ServiceManager()
    service_manager.initialize()
    app = DiveHostAPI(service_config_manager=service_manager)
    app.include_router(chat, prefix="/api/chat")

    mock_app = MockDiveHostAPI()

    def get_mock_app():
        return mock_app

    def get_mock_user():
        return {"user_id": TEST_USER_ID}

    app.dependency_overrides[get_app] = get_mock_app
    app.dependency_overrides[get_dive_user] = get_mock_user

    with TestClient(app) as client:
        yield client


@pytest.fixture(autouse=True)
def mock_event_stream():
    """Mock EventStreamContextManager for testing."""
    mock_instance = mock.MagicMock()
    mock_instance.queue = asyncio.Queue()
    mock_instance.get_response.return_value = StreamingResponse(
        content=iter(["data: [DONE]\n\n"]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
    mock_instance.__aenter__.return_value = mock_instance

    with mock.patch(
        "dive_mcp_host.httpd.routers.utils.EventStreamContextManager",
        return_value=mock_instance,
    ):
        yield


def test_list_chat(client):
    """Test the /api/chat/list endpoint."""
    # Mock the get_all_chats method
    with mock.patch.object(MockDatabase, "get_all_chats") as mock_get_all_chats:
        mock_get_all_chats.return_value = [
            Chat(
                id="test_chat_1",
                title="Test Chat 1",
                createdAt=datetime.now(UTC),
                user_id=TEST_USER_ID,
            )
        ]

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
                        "id": "test_chat_1",
                        "title": "Test Chat 1",
                        "user_id": TEST_USER_ID,
                    }
                ],
            },
        )


def test_get_chat(client):
    """Test the /api/chat/{chat_id} endpoint."""
    # Mock the get_chat_with_messages method
    with mock.patch.object(MockDatabase, "get_chat_with_messages") as mock_get_chat:
        now = datetime.now(UTC)
        mock_get_chat.return_value = ChatWithMessages(
            chat=Chat(
                id=TEST_CHAT_ID,
                title="Test Chat Title",
                createdAt=now,
                user_id=TEST_USER_ID,
            ),
            messages=[
                Message(
                    role="user",
                    content="Test user message",
                    chatId=TEST_CHAT_ID,
                    messageId="test_msg_user",
                    id=1,
                    createdAt=now,
                    files="[]",
                )
            ],
        )

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
                        "title": "Test Chat Title",
                        "user_id": TEST_USER_ID,
                    },
                    "messages": [
                        {
                            "role": "user",
                            "content": "Test user message",
                            "chatId": TEST_CHAT_ID,
                            "messageId": "test_msg_user",
                            "id": 1,
                            "files": "[]",
                        },
                    ],
                },
            },
        )


def test_delete_chat(client):
    """Test the /api/chat/{chat_id} DELETE endpoint."""
    # Mock the delete_chat method
    with mock.patch.object(MockDatabase, "delete_chat") as mock_delete_chat:
        mock_delete_chat.return_value = True

        # Send request
        response = client.delete(f"/api/chat/{TEST_CHAT_ID}")

        # Verify response status code
        assert response.status_code == SUCCESS_CODE

        # Parse JSON response
        response_data = response.json()

        # Validate response structure
        assert "success" in response_data
        assert response_data["success"] is True


# def test_abort_chat(client):
#     """Test the /api/chat/{chat_id}/abort endpoint."""
#     # Send request
#     response = client.post(f"/api/chat/{TEST_CHAT_ID}/abort")

#     print(response.json())

#     # Verify response status code
#     assert response.status_code == SUCCESS_CODE

#     # Parse JSON response
#     response_data = response.json()

#     # Validate response structure
#     assert "success" in response_data
#     assert response_data["success"] is True
#     assert "message" in response_data
#     assert "Chat abort signal sent successfully" in response_data["message"]


def test_create_chat(client, monkeypatch):
    """Test the /api/chat POST endpoint."""

    # Mock EventStreamContextManager.get_response to return a custom StreamingResponse
    def mock_response(*args, **kwargs):
        # Return a streaming response with specific test content
        return StreamingResponse(
            content=iter(
                [
                    (
                        'data: {"message":"{\\"type\\":\\"chat_info\\",'
                        '\\"content\\":{\\"id\\":\\"test-chat-id\\",'
                        '\\"title\\":\\"New Chat\\"}}"}\n\n'
                    ),
                    (
                        'data: {"message":"{\\"type\\":\\"text\\",'
                        '\\"content\\":\\"\\"}"}\n\n'
                    ),
                    (
                        'data: {"message":"{\\"type\\":\\"text\\",'
                        '\\"content\\":\\"Test response\\"}"}\n\n'
                    ),
                    (
                        'data: {"message":"{\\"type\\":\\"message_info\\",'
                        '\\"content\\":{\\"userMessageId\\":\\"test-user-msg\\",'
                        '\\"assistantMessageId\\":\\"test-ai-msg\\"}}"}\n\n'
                    ),
                    (
                        'data: {"message":"{\\"type\\":\\"chat_info\\",'
                        '\\"content\\":{\\"id\\":\\"test-chat-id\\",'
                        '\\"title\\":\\"New Chat\\"}}"}\n\n'
                    ),
                    "data: [DONE]\n\n",
                ]
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    monkeypatch.setattr(EventStreamContextManager, "get_response", mock_response)
    monkeypatch.setattr(
        EventStreamContextManager,
        "add_task",
        lambda *_args, **_kwargs: None,
    )

    test_file = io.BytesIO(b"test file content")
    response = client.post(
        "/api/chat",
        data={
            "chatId": TEST_CHAT_ID,
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
    host = cast(dict[str, DiveMcpHost], app.dive_host)["default"]
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


def test_edit_chat_missing_params(client):
    """Test the /api/chat/edit endpoint with missing required parameters."""
    with pytest.raises(UserInputError) as excinfo:
        client.post(
            "/api/chat/edit",
            data={
                "content": "edited message",
            },
        )
    assert "Chat ID and Message ID are required" in str(excinfo.value)


def test_retry_chat(client, monkeypatch):
    """Test the /api/chat/retry endpoint."""

    # Mock EventStreamContextManager.get_response to return a custom StreamingResponse
    def mock_response(*args, **kwargs):
        # Return a streaming response with specific test content
        return StreamingResponse(
            content=iter(
                [
                    (
                        'data: {"message":"{\\"type\\":\\"chat_info\\",'
                        '\\"content\\":{\\"id\\":\\"test-chat-id\\",'
                        '\\"title\\":\\"Test Chat\\"}}"}\n\n'
                    ),
                    (
                        'data: {"message":"{\\"type\\":\\"text\\",'
                        '\\"content\\":\\"\\"}"}\n\n'
                    ),
                    (
                        'data: {"message":"{\\"type\\":\\"text\\",'
                        '\\"content\\":\\"Retry response\\"}"}\n\n'
                    ),
                    (
                        'data: {"message":"{\\"type\\":\\"message_info\\",'
                        '\\"content\\":{\\"userMessageId\\":\\"test-user-msg\\",'
                        '\\"assistantMessageId\\":\\"test-ai-msg\\"}}"}\n\n'
                    ),
                    (
                        'data: {"message":"{\\"type\\":\\"chat_info\\",'
                        '\\"content\\":{\\"id\\":\\"test-chat-id\\",'
                        '\\"title\\":\\"Test Chat\\"}}"}\n\n'
                    ),
                    "data: [DONE]\n\n",
                ]
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    monkeypatch.setattr(EventStreamContextManager, "get_response", mock_response)
    monkeypatch.setattr(
        EventStreamContextManager,
        "add_task",
        lambda *_args, **_kwargs: None,
    )

    response = client.post(
        "/api/chat/retry",
        data={
            "chatId": TEST_CHAT_ID,
            "messageId": TEST_MESSAGE_ID,
        },
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
                            "title": "Test Chat",
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


def test_retry_chat_missing_params(client):
    """Test the /api/chat/retry endpoint with missing required parameters."""
    with pytest.raises(UserInputError) as excinfo:
        client.post("/api/chat/retry", data={})

    # Verify the exception message
    assert "Chat ID and Message ID are required" in str(excinfo.value)
