from fastapi import APIRouter, Depends
from typing import List, Dict, Any, Optional

from dive_mcp_host.host.documents import MorphikDocumentManager

router = APIRouter()
PAGINATION_LIMIT = 10
def get_document_manager() -> MorphikDocumentManager:
    return MorphikDocumentManager()

@router.get("", response_model=List[Dict[str, Any]])
async def list_documents(
    page: int = 0,
    status: Optional[str] = None,
    manager: MorphikDocumentManager = Depends(get_document_manager)
):
    skip = page * PAGINATION_LIMIT
    async with manager:
        return await manager.list_documents(skip=skip, limit=PAGINATION_LIMIT, status=status)

@router.delete("/{external_id}")
async def delete_document(
    external_id: str,
    manager: MorphikDocumentManager = Depends(get_document_manager)
):
    async with manager:
        return await manager.delete_document(external_id=external_id)