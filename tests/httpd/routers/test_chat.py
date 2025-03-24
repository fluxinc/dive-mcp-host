import io
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest import mock

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from dive_mcp_host.httpd.database.models import Chat
from dive_mcp_host.httpd.dependencies import get_app, get_dive_user
from dive_mcp_host.httpd.routers.chat import chat
from dive_mcp_host.httpd.routers.models import (
    UserInputError,
)
from dive_mcp_host.httpd.routers.utils import EventStreamContextManager

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
    chatId: str
    messageId: str
    id: int
    createdAt: datetime
    files: str = "[]"  # Modified to string type to match the actual database model


@dataclass
class ChatWithMessages:
    """Mock chat with messages response."""

    chat: Chat
    messages: list[Message]


class MockDatabase:
    """Mock database object."""

    async def get_all_chats(self, user_id=TEST_USER_ID, **_kwargs):
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

    async def get_chat_with_messages(self, chat_id, user_id=TEST_USER_ID, **_kwargs):
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

    async def delete_chat(self, chat_id, user_id=TEST_USER_ID, **_kwargs):
        """Delete a chat."""
        return True

    async def update_message_content(
        self, message_id, query_input, user_id=TEST_USER_ID, **_kwargs
    ):
        """Update message content."""
        return True

    async def get_next_ai_message(self, chat_id, message_id, **_kwargs):
        """Get the next AI message."""
        return Message(
            role="assistant",
            content="Mock next AI message",
            chatId=chat_id,
            messageId="next_ai_msg",
            id=3,
            createdAt=datetime.now(UTC),
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

    async def upload_files(self, files, filepaths):
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

            async def __aexit__(self, *args):
                pass

            async def commit(self):
                pass

        return MockSession()

    def msg_store(self, session):
        """Return the database mock."""
        return self.db


@pytest.fixture
def app():
    """Create a mock DiveHostAPI for testing."""
    return MockDiveHostAPI()


@pytest.fixture
def client():
    """Create a test client."""
    app = FastAPI()
    app.include_router(chat)

    # Mock dependencies
    mock_app = MockDiveHostAPI()

    def get_mock_app():
        return mock_app

    def get_mock_user():
        return {"user_id": TEST_USER_ID}

    # Properly configure dependency overrides
    app.dependency_overrides[get_app] = get_mock_app
    app.dependency_overrides[get_dive_user] = get_mock_user

    # Mock request state
    app.middleware("http")(mock_state_middleware)

    return TestClient(app)


async def mock_state_middleware(request, call_next):
    """Mock middleware to add state attribute."""
    request.state.get_kwargs = lambda _name: {}
    request.state.dive_user = {"user_id": TEST_USER_ID}
    return await call_next(request)


# Patch EventStreamContextManager to avoid actual streaming in tests
@pytest.fixture(autouse=True)
def mock_event_stream():
    """Mock EventStreamContextManager for testing."""
    with mock.patch.object(EventStreamContextManager, "__init__", return_value=None):
        with mock.patch.object(
            EventStreamContextManager, "get_response"
        ) as mock_response:
            mock_response.return_value = None
            with mock.patch.object(
                EventStreamContextManager, "add_task"
            ) as mock_add_task:
                with mock.patch.object(
                    EventStreamContextManager, "__aenter__", return_value=None
                ):
                    with mock.patch.object(
                        EventStreamContextManager, "__aexit__", return_value=None
                    ):
                        yield


def test_list_chat(client, app):
    """Test the /chat/list endpoint."""
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


def test_get_chat(client, app):
    """Test the /chat/{chat_id} endpoint."""
    # Mock the get_chat_with_messages method
    with mock.patch.object(MockDatabase, "get_chat_with_messages") as mock_get_chat:
        mock_get_chat.return_value = ChatWithMessages(
            chat=Chat(
                id=TEST_CHAT_ID,
                title="Test Chat Title",
                createdAt=datetime.now(UTC),
                user_id=TEST_USER_ID,
            ),
            messages=[
                Message(
                    role="user",
                    content="Test user message",
                    chatId=TEST_CHAT_ID,
                    messageId="test_msg_user",
                    id=1,
                    createdAt=datetime.now(UTC),
                )
            ],
        )

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


def test_delete_chat(client, app):
    """Test the /chat/{chat_id} DELETE endpoint."""
    # Mock the delete_chat method
    with mock.patch.object(MockDatabase, "delete_chat") as mock_delete_chat:
        mock_delete_chat.return_value = True

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
    assert "message" in response_data
    assert "Chat abort signal sent successfully" in response_data["message"]


def test_create_chat(client, app, monkeypatch):
    """Test the /chat POST endpoint."""

    # Mock EventStreamContextManager.get_response to return a valid StreamingResponse
    def mock_response(*args, **kwargs):
        from fastapi.responses import JSONResponse

        return JSONResponse(content={"success": True})

    monkeypatch.setattr(EventStreamContextManager, "get_response", mock_response)
    monkeypatch.setattr(
        EventStreamContextManager, "add_task", lambda *args, **kwargs: None
    )

    # Mock file upload
    test_file = io.BytesIO(b"test file content")

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

    # Only check HTTP status code
    assert response.status_code == SUCCESS_CODE


def test_edit_chat(client, app, monkeypatch):
    """Test the /chat/edit endpoint."""

    # Mock EventStreamContextManager.get_response to return a valid StreamingResponse
    def mock_response(*args, **kwargs):
        from fastapi.responses import JSONResponse

        return JSONResponse(content={"success": True})

    monkeypatch.setattr(EventStreamContextManager, "get_response", mock_response)
    monkeypatch.setattr(
        EventStreamContextManager, "add_task", lambda *args, **kwargs: None
    )

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

    # Only check HTTP status code
    assert response.status_code == SUCCESS_CODE


def test_edit_chat_missing_params(client):
    """Test the /chat/edit endpoint with missing required parameters."""
    with pytest.raises(UserInputError) as excinfo:
        client.post(
            "/chat/edit",
            data={
                "content": "edited message",
            },
        )

    # Verify the exception message
    assert "Chat ID and Message ID are required" in str(excinfo.value)


def test_retry_chat(client, app, monkeypatch):
    """Test the /chat/retry endpoint."""

    # Mock EventStreamContextManager.get_response to return a valid StreamingResponse
    def mock_response(*args, **kwargs):
        from fastapi.responses import JSONResponse

        return JSONResponse(content={"success": True})

    monkeypatch.setattr(EventStreamContextManager, "get_response", mock_response)
    monkeypatch.setattr(
        EventStreamContextManager, "add_task", lambda *args, **kwargs: None
    )

    # Send request
    response = client.post(
        "/chat/retry",
        data={
            "chatId": TEST_CHAT_ID,
            "messageId": TEST_MESSAGE_ID,
        },
    )

    # Only check HTTP status code
    assert response.status_code == SUCCESS_CODE


def test_retry_chat_missing_params(client):
    """Test the /chat/retry endpoint with missing required parameters."""
    with pytest.raises(UserInputError) as excinfo:
        client.post("/chat/retry", data={})

    # Verify the exception message
    assert "Chat ID and Message ID are required" in str(excinfo.value)
