from abc import ABC, abstractmethod

from fastapi import UploadFile


class Store(ABC):
    """Abstract base class for store operations."""

    @abstractmethod
    async def upload_files(
        self,
        files: list[UploadFile],
        file_paths: list[str],
    ) -> tuple[list[str], list[str]]:
        """Upload files to the store."""
