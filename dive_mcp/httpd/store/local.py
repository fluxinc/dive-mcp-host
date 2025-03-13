from hashlib import md5

from fastapi import UploadFile

from .store import Store


class LocalStore(Store):
    """Local store implementation."""

    async def upload_files(
        self,
        files: list[UploadFile],
        file_paths: list[str],
    ) -> tuple[list[str], list[str]]:
        """Upload files to the local store."""
        return [], []
