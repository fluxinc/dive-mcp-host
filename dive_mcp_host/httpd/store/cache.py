import os
import platform
from enum import StrEnum
from logging import getLogger
from pathlib import Path

logger = getLogger(__name__)


class CacheKeys(StrEnum):
    """Cache keys for the MCP host."""

    LIST_TOOLS = "list_tools"


class LocalFileCache:
    """Local file cache for the MCP host."""

    def __init__(self, cache_file_prefix: str = "dive_mcp_host") -> None:
        """Initialize the local file cache.

        Args:
            cache_file_prefix: The prefix of the cache file.

        """
        self._cache_file_prefix = cache_file_prefix

        self._system = platform.system()
        if self._system == "Windows":
            self._cache_dir = Path(os.path.expandvars(r"%LOCALAPPDATA%\Dive\Cache"))
        elif self._system == "Linux":
            self._cache_dir = Path.expanduser(Path("~/.cache/dive"))
        elif self._system == "Darwin":
            self._cache_dir = Path.expanduser(Path("~/Library/Caches/Dive"))
        else:
            raise NotImplementedError(f"Unsupported system: {self._system}")

        logger.info("LocalCache directory: %s", self._cache_dir)
        logger.info("LocalCache file prefix: %s", self._cache_file_prefix)
        self._cache_dir.mkdir(mode=0o744, parents=True, exist_ok=True)
        logger.info("LocalCache directory prepared")

    def get_cache_file_path(self, key: CacheKeys, extension: str = "json") -> Path:
        """Get the cache file name.

        Args:
            key: The key of the cache.
            extension: The extension of the cache file.

        Returns:
            The cache file name.
        """
        return self._cache_dir / f"{self._cache_file_prefix}_{key.value}.{extension}"

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory."""
        return self._cache_dir

    def get(self, key: CacheKeys, extension: str = "json") -> str | None:
        """Get the value of the key.

        Args:
            key: The key of the cache.
            extension: The extension of the cache file.

        Returns:
            The value of the key.
        """
        cache_file_path = self.get_cache_file_path(key, extension)
        if not cache_file_path.exists():
            return None
        with cache_file_path.open("r") as f:
            return f.read()

    def set(self, key: CacheKeys, value: str, extension: str = "json") -> None:
        """Set the value of the key.

        Args:
            key: The key of the cache.
            value: The value of the key.
            extension: The extension of the cache file.

        Returns:
            None
        """
        cache_file_path = self.get_cache_file_path(key, extension)
        with cache_file_path.open("w") as f:
            f.write(value)

    def delete(self, key: CacheKeys, extension: str = "json") -> None:
        """Delete cache file.

        Args:
            key: The key of the cache.
            extension: The extension of the cache file.

        Returns:
            None
        """
        cache_file_path = self.get_cache_file_path(key, extension)
        if cache_file_path.exists():
            cache_file_path.unlink()
