import os
from typing import List, Dict, Any, Optional

import httpx

import logging

logger = logging.getLogger(__name__)

# Constants from original script
BASE_URL = os.getenv("DATABRIDGE_SERVER_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = 10


class MorphikDocumentManager:
    """
    Handles document management in the Morphik system.
    """

    def __init__(self, base_url: str = BASE_URL, auth_token: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token or os.getenv("MORPHIK_AUTH_TOKEN")

        headers = {'Content-Type': 'application/json'}
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'
        logger.info(f"\n\nUsing BASE_URL: {self.base_url}\n\n")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=DEFAULT_TIMEOUT
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        await self._client.aclose()

    async def list_documents(self, skip: int = 0, limit: int = 100, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List documents from the Morphik API."""
        try:
            params = {"skip": skip, "limit": limit} 
            if status:
                params["status"] = status
            response = await self._client.post(
                "/documents",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            # logger.error(f"❌ Failed to list documents: {e}")
            raise
        except httpx.HTTPStatusError as e:
            # logger.error(f"❌ Failed to list documents: {e.response.status_code} - {e.response.text}")
            raise

    async def delete_document(self, external_id: str):
        """Delete a document from the Morphik API."""
        try:
            response = await self._client.delete(f"/documents/{external_id}")
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            # logger.error(f"❌ Failed to delete document: {e}")
            raise