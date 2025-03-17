import io
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest import mock

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from dive_mcp_host.httpd.routers.chat import chat
from dive_mcp_host.httpd.routers.models import (
    Chat,
    Message,
    UserInputError,
)
from dive_mcp_host.httpd.routers.utils import ChatProcessor

# Constants
SUCCESS_CODE = status.HTTP_200_OK
BAD_REQUEST_CODE = status.HTTP_400_BAD_REQUEST
TEST_CHAT_ID = "test_chat_123"
TEST_MESSAGE_ID = "test_message_456"


@dataclass
class ChatWithMessages:
    """Mock chat with messages response."""

    chat: Chat
    messages: list[Message]


class MockDatabase:
    """Mock database object."""

    async def get_all_chats(self, **_kwargs):
        """Get all available chats."""
        return [
            Chat(
                id="mock_chat_1",
                title="Mock Chat 1",
                createdAt=datetime.now(UTC),
            ),
            Chat(
                id="mock_chat_2",
                title="Mock Chat 2",
                createdAt=datetime.now(UTC),
            ),
        ]

    async def get_chat_with_messages(self, chat_id, **_kwargs):
        """Get a specific chat with its messages."""
        return ChatWithMessages(
            chat=Chat(
                id=chat_id,
                title="Mock Chat Title",
                createdAt=datetime.now(UTC),
            ),
            messages=[
                Message(
                    role="user",
                    content="Mock user message",
                    chatId=chat_id,
                    messageId="msg_user",
                    id=1,
                    createdAt=datetime.now(UTC),
                    files=[],
                ),
                Message(
                    role="assistant",
                    content="Mock assistant message",
                    chatId=chat_id,
                    messageId="msg_assistant",
                    id=2,
                    createdAt=datetime.now(UTC),
                    files=[],
                ),
            ],
        )

    async def delete_chat(self, _chat_id, **_kwargs):
        """Delete a chat."""
        return True

    async def update_message_content(self, _message_id, _query_input, **_kwargs):
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
            files=[],
        )

    async def check_chat_exists(self, chat_id, **_kwargs):
        """Check if a chat exists."""
        return True

    async def create_chat_with_messages(self, chat_id, title, messages, **_kwargs):
        """Create a chat with messages."""
        return True

    async def create_message(self, message, **_kwargs):
        """Create a new message."""
        return True

    async def create_chat(self, chat_id, title, **_kwargs):
        """Create a new chat."""
        return True

    async def delete_messages_after(self, chat_id, message_id, **_kwargs):
        """Delete messages after a specific message."""
        return True


class MockStore:
    """Mock storage object."""

    async def upload_files(self, _files, _filepaths):
        """Mock file upload."""
        return [], []


class MockMcpServerManager:
    """Mock MCP server manager."""

    async def process_chat_message(self, *args, **kwargs):
        """Mock processing chat message."""
        yield "mock message chunk"

    async def process_chat_edit(self, *args, **kwargs):
        """Mock processing chat edit."""
        yield "mock edit chunk"

    async def process_chat_retry(self, *args, **kwargs):
        """Mock processing chat retry."""
        yield "mock retry chunk"

    async def get_tool_to_server_map(self):
        """Mock getting tool to server map."""
        return {}

    async def get_available_tools(self):
        """Mock getting available tools."""
        return []


@pytest.fixture
def client():
    """Create a test client."""
    app = FastAPI()
    app.include_router(chat)

    # Mock request.app.state
    app.state.db = MockDatabase()
    app.state.store = MockStore()
    app.state.mcp = MockMcpServerManager()

    # Mock request state
    app.middleware("http")(mock_state_middleware)

    return TestClient(app)


async def mock_state_middleware(request, call_next):
    """Mock middleware to add state attribute."""
    request.state.get_kwargs = lambda _name: {}
    return await call_next(request)


def test_list_chat(client):
    """Test the /chat/list endpoint."""
    # Send request
    response = client.get("/chat/list")

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert "success" in response_data
    assert response_data["success"] is True
    assert "data" in response_data
    assert isinstance(response_data["data"], list)

    # Validate chat data structure
    if response_data["data"]:
        chat = response_data["data"][0]
        assert "id" in chat
        assert "title" in chat
        assert "createdAt" in chat


def test_get_chat(client):
    """Test the /chat/{chat_id} endpoint."""
    # Send request
    response = client.get(f"/chat/{TEST_CHAT_ID}")

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert "success" in response_data
    assert response_data["success"] is True
    assert "data" in response_data

    # Validate chat data structure
    data = response_data["data"]
    assert "chat" in data
    assert "id" in data["chat"]
    assert "title" in data["chat"]
    assert "createdAt" in data["chat"]

    # Validate message list structure
    assert "messages" in data
    assert isinstance(data["messages"], list)

    # Validate message structure
    if data["messages"]:
        message = data["messages"][0]
        assert "role" in message
        assert "content" in message
        assert "chatId" in message
        assert "messageId" in message


def test_delete_chat(client):
    """Test the /chat/{chat_id} DELETE endpoint."""
    # Send request
    response = client.delete(f"/chat/{TEST_CHAT_ID}")

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert "success" in response_data
    assert response_data["success"] is True


def test_abort_chat(client):
    """Test the /chat/{chat_id}/abort endpoint."""
    # Send request
    response = client.post(f"/chat/{TEST_CHAT_ID}/abort")

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert "success" in response_data
    assert response_data["success"] is True


def test_create_chat(client):
    """Test the /chat POST endpoint."""
    # Mock file upload
    test_file = io.BytesIO(b"test file content")

    # Patch process_chat to avoid NotImplementedError
    with mock.patch.object(ChatProcessor, "process_chat") as mock_process_chat:
        # Mock the return value of process_chat
        mock_process_chat.return_value = (
            "test_message_id",
            {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        # Send request
        response = client.post(
            "/chat",
            data={
                "chatId": TEST_CHAT_ID,
                "message": "test message",
                "filepaths": ["test_path.txt"],
            },
            files={"files": ("test.txt", test_file, "text/plain")},
        )

        # Verify response status code
        assert response.status_code == SUCCESS_CODE

        # Verify response Content-Type
        assert response.headers["Content-Type"].startswith("text/event-stream")

        # Check response content contains 'data: ' prefix
        content = response.content.decode()
        assert "data: " in content


def test_edit_chat(client):
    """Test the /chat/edit endpoint."""
    # Mock file upload
    test_file = io.BytesIO(b"test file content")

    # Send request
    response = client.post(
        "/chat/edit",
        data={
            "chatId": TEST_CHAT_ID,
            "messageId": TEST_MESSAGE_ID,
            "content": "edited message",
            "filepaths": ["test_edit_path.txt"],
        },
        files={"files": ("test_edit.txt", test_file, "text/plain")},
    )

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Verify response Content-Type
    assert response.headers["Content-Type"].startswith("text/event-stream")

    # Check response content contains 'data: ' prefix
    content = response.content.decode()
    assert "data: " in content


def test_edit_chat_missing_params(client):
    """Test the /chat/edit endpoint with missing required parameters."""
    # Use pytest.raises to check for the expected exception
    with pytest.raises(UserInputError) as excinfo:
        client.post(
            "/chat/edit",
            data={
                "content": "edited message",
            },
        )

    # Verify the exception message
    assert "Chat ID and Message ID are required" in str(excinfo.value)


def test_retry_chat(client):
    """Test the /chat/retry endpoint."""
    # Send request
    response = client.post(
        "/chat/retry",
        data={
            "chatId": TEST_CHAT_ID,
            "messageId": TEST_MESSAGE_ID,
        },
    )

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Verify response Content-Type
    assert response.headers["Content-Type"].startswith("text/event-stream")

    # Check response content contains 'data: ' prefix
    content = response.content.decode()
    assert "data: " in content


def test_retry_chat_missing_params(client):
    """Test the /chat/retry endpoint with missing required parameters."""
    # Use pytest.raises to check for the expected exception
    with pytest.raises(UserInputError) as excinfo:
        client.post("/chat/retry", data={})

    # Verify the exception message
    assert "Chat ID and Message ID are required" in str(excinfo.value)
