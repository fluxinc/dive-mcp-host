from langchain_core.tools import tool


class TestTool:
    """Test tool."""

    def __init__(self) -> None:
        """Initialize test state."""
        self._called: bool = False

    @property
    def called(self) -> bool:
        """Whether the tool has been called."""
        return self._called

    @tool
    def weather_tool(self, city: str) -> str:
        """Get current weather information."""
        self._called = True
        return f"The weather in {city} is sunny."
