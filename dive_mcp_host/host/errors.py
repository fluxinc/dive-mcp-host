"""Errors for the MCP host."""


class MCPHostError(Exception):
    """Base exception for MCP host errors."""


class ThreadNotFoundError(MCPHostError):
    """Exception raised when a thread is not found."""

    def __init__(self, thread_id: str) -> None:
        """Initialize the error.

        Args:
            thread_id: The thread ID that was not found.
        """
        self.thread_id = thread_id
        super().__init__(f"Thread {thread_id} not found")


class GraphNotCompiledError(MCPHostError):
    """Exception raised when the graph is not compiled."""

    def __init__(self, thread_id: str | None = None) -> None:
        """Initialize the error.

        Args:
            thread_id: The thread ID that was not found.
        """
        self.thread_id = thread_id
        super().__init__(f"Graph not compiled for thread {thread_id}")
