from collections.abc import AsyncGenerator
from typing import Self

import pytest

from dive_mcp_host.host.helpers.context import ContextProtocol


class FakeContextImplementation(ContextProtocol):
    """A concrete implementation of ContextProtocol for testing."""

    def __init__(self):
        self.setup_called = False
        self.cleanup_called = False
        self.exception_handled = False
        self.custom_exception = None

    async def _run_in_context(self) -> AsyncGenerator[Self, None]:
        self.setup_called = True
        try:
            yield self
        except Exception as e:
            self.exception_handled = True
            self.custom_exception = e
            raise
        finally:
            self.cleanup_called = True


class FakeContextWithError(ContextProtocol):
    """A context implementation that raises an error during setup."""

    def __init__(self):
        self.cleanup_called = False

    async def _run_in_context(self) -> AsyncGenerator[Self, None]:
        try:
            raise ValueError("Setup error")
            yield self  # This will never be reached
        finally:
            self.cleanup_called = True


@pytest.mark.asyncio
async def test_context_normal_flow() -> None:
    """Test the normal flow of a context manager."""
    context = FakeContextImplementation()

    async with context as ctx:
        assert ctx is context
        assert context.setup_called is True
        assert context.cleanup_called is False

    assert context.cleanup_called is True
    assert context.exception_handled is False
    assert context.custom_exception is None


@pytest.mark.asyncio
async def test_context_with_exception():
    """Test that exceptions are properly propagated through the context manager."""
    context = FakeContextImplementation()

    # Workaround for PT012 (pytest-asyncio)
    # - need to wrap in function to handle async context
    async def _f():
        async with context as ctx:
            assert context is ctx
            assert context.setup_called is True
            raise ValueError("Test exception")

    with pytest.raises(ValueError, match="Test exception"):
        await _f()

    assert context.cleanup_called is True
    assert context.exception_handled is True
    assert isinstance(context.custom_exception, ValueError)
    assert str(context.custom_exception) == "Test exception"


@pytest.mark.asyncio
async def test_context_setup_error():
    """Test that errors during context setup are properly handled."""
    context = FakeContextWithError()

    with pytest.raises(ValueError, match="Setup error"):
        async with context:
            pytest.fail("This code should not be executed")

    assert context.cleanup_called is True


@pytest.mark.asyncio
async def test_no_reentrant_usage():
    """Test that reentrant usage of the context manager is prevented."""
    context = FakeContextImplementation()

    async with context:
        with pytest.raises(RuntimeError, match="No Reentrant"):
            async with context:
                pytest.fail("This code should not be executed")

    # After exiting the first context, we should be able to use it again
    async with context:
        pass  # This should work fine


@pytest.mark.asyncio
async def test_generator_didnt_stop():
    """Test the error when a generator doesn't stop properly."""

    class BadGenerator(ContextProtocol):
        async def _run_in_context(self) -> AsyncGenerator[Self, None]:
            yield self
            yield self  # This second yield will cause an error

    context = BadGenerator()

    with pytest.raises(RuntimeError, match="generator didn't stop"):
        async with context:
            pass
