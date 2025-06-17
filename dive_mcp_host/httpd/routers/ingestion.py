# Standard Library Imports
import logging
import os
import shutil
import tempfile
import time
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any

# Third-Party Imports
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import JSONResponse

# Local Application Imports
from dive_mcp_host.ingestion.document_ingestion import MorphikDocumentIngester, get_ingester, INDIVIDUAL_DELAY

logger = logging.getLogger(__name__)

ingestion = APIRouter()


def get_request_id() -> str:
    """Generates a unique request ID for logging and tracing."""
    return os.urandom(8).hex()

# --- Helper Functions for Ingestion ---
def _save_uploaded_files(files: List[UploadFile], temp_dir: Path, ingester: MorphikDocumentIngester, request_id: str) -> List[Dict[str, Any]]:
    """Saves uploaded files to a temporary directory and gets their info."""
    file_infos = []
    for file in files:
        file_location = temp_dir / file.filename
        try:
            with open(file_location, "wb+") as file_object:
                shutil.copyfileobj(file.file, file_object)
            file_infos.append(ingester.get_file_info(file_location))
        except Exception as e:
            logger.error(f"Failed to save uploaded file {file.filename}: {e}", extra={'request_id': request_id})
    return file_infos

async def _perform_initial_ingestion(file_infos: List[Dict[str, Any]], folder_name: Optional[str], ingester: MorphikDocumentIngester, request_id: str) -> (List, List):
    """Performs the initial batch or individual ingestion."""
    successful_ingestions = []
    failed_ingestions = []

    if len(file_infos) == 1:
        file_info = file_infos[0]
        logger.info(f"Attempting individual ingestion for {file_info['name']}", extra={'request_id': request_id})
        if await ingester.individual_ingest_file(file_info, folder_name):
            successful_ingestions.append(file_info)
        else:
            failed_ingestions.append(file_info)
    else:
        logger.info(f"Attempting batch ingestion for {len(file_infos)} files.", extra={'request_id': request_id})
        successful, failed = await ingester.batch_ingest_files(file_infos, folder_name)
        successful_ingestions.extend(successful)
        failed_ingestions.extend(failed)

    return successful_ingestions, failed_ingestions

async def _retry_failed_ingestions(failed_files: List[Dict[str, Any]], folder_name: Optional[str], ingester: MorphikDocumentIngester, request_id: str) -> List[Dict[str, Any]]:
    """Retries failed ingestions one by one."""
    if not failed_files:
        return []

    logger.info(f"{len(failed_files)} files failed. Retrying individually.", extra={'request_id': request_id})
    final_failures = []
    for file_info in failed_files:
        await asyncio.sleep(INDIVIDUAL_DELAY)
        if not await ingester.individual_ingest_file(file_info, folder_name):
            final_failures.append(file_info)

    return final_failures

# --- API Endpoints ---
@ingestion.post("/ingest_files", summary="Ingest one or more files")
async def ingest_files(
    files: List[UploadFile] = File(...),
    folder_name: Optional[str] = Form(None),
    ingester: MorphikDocumentIngester = Depends(get_ingester),
    request_id: str = Depends(get_request_id)
):
    """
    Accepts one or more files and ingests them into the Morphik database.

    - **Batching**: If multiple files are provided, they are ingested as a batch.
    - **Retries**: Includes retry logic for ingestions that initially fail.
    - **Temporary Storage**: Files are temporarily stored on disk before ingestion.
    """
    logger.info(f"Received request to ingest {len(files)} files. Folder: {folder_name or 'N/A'}", extra={'request_id': request_id})

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    if not await ingester.check_health():
        logger.error("Morphik server is not healthy. Aborting.", extra={'request_id': request_id})
        raise HTTPException(status_code=503, detail="Morphik server is unavailable.")

    with tempfile.TemporaryDirectory() as temp_dir:
        file_infos = _save_uploaded_files(files, Path(temp_dir), ingester, request_id)

        if not file_infos:
            logger.error("No files could be processed from the upload.", extra={'request_id': request_id})
            return JSONResponse(
                status_code=500,
                content={"message": "Failed to process any of the uploaded files.", "failed_documents": [f.filename for f in files]}
            )

        successful_ingestions, failed_ingestions = await _perform_initial_ingestion(file_infos, folder_name, ingester, request_id)

        # Retry logic for any files that failed the first time
        final_failures = await _retry_failed_ingestions(failed_ingestions, folder_name, ingester, request_id)

        # Update success list with those that succeeded on retry
        succeeded_on_retry = [f for f in failed_ingestions if f not in final_failures]
        successful_ingestions.extend(succeeded_on_retry)

    failed_doc_names = [f['name'] for f in final_failures]
    success_count = len(successful_ingestions)
    failed_count = len(failed_doc_names)

    logger.info(f"Ingestion complete. Success: {success_count}, Failed: {failed_count}.", extra={'request_id': request_id})
    if failed_doc_names:
        logger.warning(f"Failed documents: {', '.join(failed_doc_names)}", extra={'request_id': request_id})

    if not successful_ingestions:
        logger.error("No documents were successfully ingested.", extra={'request_id': request_id})
        return JSONResponse(
            status_code=500,
            content={"message": "No documents could be ingested.", "failed_documents": failed_doc_names}
        )

    return JSONResponse(
        status_code=200,
        content={
            "message": f"Ingestion complete. {success_count} succeeded, {failed_count} failed.",
            "failed_documents": failed_doc_names
        }
    )

@ingestion.post("/ingest_urls", summary="Ingest content from a list of URLs")
async def ingest_urls(
    urls: List[str] = Form(...),
    folder_name: Optional[str] = Form(None),
    ingester: MorphikDocumentIngester = Depends(get_ingester),
    request_id: str = Depends(get_request_id)
):
    """
    Accepts one or more URLs, crawls them, and ingests the content into the Morphik database.

    - **Language Detection**: It checks the page's HTML for the 'lang' attribute and only ingests English pages.
    - **Content Format**: It captures the page as a PDF to include images and text.
    - **File Naming**: The filename for ingestion is derived from the page title.
    """
    logger.info(f"Received request to ingest from {len(urls)} URLs. Folder: {folder_name or 'N/A'}", extra={'request_id': request_id})

    if not urls:
        raise HTTPException(status_code=400, detail="No URLs provided.")

    if not await ingester.check_health():
        logger.error("Morphik server is not healthy. Aborting.", extra={'request_id': request_id})
        raise HTTPException(status_code=503, detail="Morphik server is unavailable.")

    try:
        results = await ingester.ingest_urls(urls, folder_name)
        
        success_count = len(results["successful"])
        failed_count = len(results["failed"])
        skipped_count = len(results["skipped"])

        logger.info(f"URL ingestion complete. Success: {success_count}, Failed: {failed_count}, Skipped: {skipped_count}", extra={'request_id': request_id})

        return JSONResponse(
            status_code=200,
            content={
                "message": f"Ingestion process complete. {success_count} URLs succeeded, {failed_count} failed, {skipped_count} skipped.",
                "successful_urls": results["successful"],
                "failed_urls": results["failed"],
                "skipped_urls": results["skipped"]
            }
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during URL ingestion: {e}", exc_info=True, extra={'request_id': request_id})
        raise HTTPException(status_code=500, detail="An unexpected error occurred during URL ingestion.")

@ingestion.post("/crawl_site", summary="Crawl a website from a URL")
async def crawl_site_for_urls(
    site: str = Form(...),
    ingester: MorphikDocumentIngester = Depends(get_ingester),
    request_id: str = Depends(get_request_id)
):
    """
    Accepts a URL (including sitemaps) and crawls it.
    If the URL points to a sitemap or a sitemap index, it parses URLs directly.
    Otherwise, it uses crawl4ai for general-purpose web crawling.
    """
    clean_site = site.strip().lstrip('@')
    logger.info(f"Received request to crawl site: {clean_site}", extra={'request_id': request_id})

    try:
        all_page_urls = await ingester.crawl_site_for_urls(clean_site)

        if not all_page_urls:
            msg = "Crawl complete, but no URLs were found in the sitemaps or via crawling."
            logger.warning(f"{msg} from {clean_site}", extra={'request_id': request_id})
            return JSONResponse(status_code=404, content={"message": msg})

        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully parsed/crawled {len(all_page_urls)} URLs.",
                "urls": all_page_urls,
            }
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred while crawling {clean_site}: {e}", exc_info=True, extra={'request_id': request_id})
        return JSONResponse(
            status_code=500,
            content={"message": f"An unexpected error occurred while crawling {clean_site}."}
        )

