from pathlib import Path
import sys
import logging
import json
import os
import shutil
from typing import List, Tuple, Dict, Any, Optional
import time
import mimetypes
from urllib3 import encode_multipart_formdata

# New imports for crawling
import asyncio
import re
import tempfile
import xml.etree.ElementTree as ET
import httpx
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from urllib.parse import urlparse

# Constants from original script
BASE_URL = os.getenv("DATABRIDGE_SERVER_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = 10
BATCH_TIMEOUT = 300
INDIVIDUAL_TIMEOUT = 120
MAX_BATCH_SIZE_MB = 500
MAX_FILES_PER_BATCH = 500
RETRY_DELAY = 1.0
INDIVIDUAL_DELAY = 0.5
MB = 1024 * 1024

logger = logging.getLogger(__name__)

# --- Helper Functions from original script ---

def _extract_urls_from_sitemap_content(sitemap_content: str, namespace: Dict[str, str]) -> List[str]:
    """Extracts URLs from a sitemap XML content string."""
    try:
        sitemap_root = ET.fromstring(sitemap_content)
        if not sitemap_root.tag.endswith('urlset'):
            logger.warning("Content is not a valid sitemap urlset.")
            return []
        return [elem.text for elem in sitemap_root.findall('sm:url/sm:loc', namespace)]
    except ET.ParseError as e:
        logger.error(f"Failed to parse sitemap content: {e}")
        return []

def _slugify(value: str, to_lower: bool = False, separator: str = '_') -> str:
    """Converts a string into a slug for use in filenames."""
    if not value:
        return ""
    value = re.sub(r'[^\w\s-]', '', value).strip()
    value = re.sub(r'[-\s]+', separator, value)
    if to_lower:
        value = value.lower()
    return value

def _extract_lang_from_html(html_content: str) -> Optional[str]:
    """Extracts the language from the lang attribute of the <html> tag."""
    if not html_content:
        return None
    match = re.search(r'<html\s[^>]*lang="([^"]*)"[^>]*>', html_content, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None

class MorphikDocumentIngester:
    """
    Handles document ingestion into the Morphik system.
    Adapted to be asynchronous for FastAPI integration.
    """
    def __init__(self, base_url: str = BASE_URL, auth_token: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token or os.getenv("MORPHIK_AUTH_TOKEN")
        
        headers = {'Content-Type': 'application/json'}
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'
        
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

    async def check_health(self) -> bool:
        """Check if the Morphik server is healthy (async)"""
        try:
            response = await self._client.get("/ping", timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            logger.info("✅ Morphik server is healthy")
            return True
        except httpx.RequestError as e:
            logger.error(f"❌ Failed to connect to Morphik server: {e}")
            return False

    @staticmethod
    def get_file_info(file_path: Path) -> Dict[str, Any]:
        """Get file information including size and mime type"""
        try:
            stat = file_path.stat()
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return {
                'path': file_path,
                'size': stat.st_size,
                'mime_type': mime_type or 'application/octet-stream',
                'name': file_path.name
            }
        except OSError as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            raise

    def _prepare_batch_payload(self, files: List[Dict[str, Any]], folder_name: Optional[str] = None) -> Tuple[bytes, str]:
        """Prepare multipart/form-data payload using urllib3."""
        fields = []
        for file_info in files:
            try:
                # This part remains synchronous as it reads from disk
                with open(file_info['path'], 'rb') as f:
                    file_content = f.read()
                    fields.append(('files', (file_info['name'], file_content, file_info['mime_type'])))
            except IOError as e:
                logger.error(f"Failed to read file {file_info['path']}: {e}")
                raise
        
        if folder_name:
            fields.append(('folder_name', folder_name))
            
        body, content_type = encode_multipart_formdata(fields)
        return body, content_type

    async def batch_ingest_files(self, files: List[Dict[str, Any]], folder_name: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Batch ingest files using Morphik API (async)"""
        if not files:
            return [], []
        
        try:
            body, content_type = self._prepare_batch_payload(files, folder_name)
            
            headers = self._client.headers.copy()
            headers['Content-Type'] = content_type
            
            response = await self._client.post(
                "/ingest/files",
                content=body,
                headers=headers,
                timeout=BATCH_TIMEOUT
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Successfully batch ingested {len(files)} files")
                return files.copy(), []
            else:
                logger.error(f"❌ Batch ingestion failed: {response.status_code} - {response.text}")
                return [], files.copy()
                
        except httpx.RequestError as e:
            logger.error(f"❌ Batch ingestion request failed: {e}")
            return [], files.copy()
        except Exception as e:
            logger.error(f"❌ Unexpected error during batch ingestion: {e}")
            return [], files.copy()
    
    async def individual_ingest_file(self, file_info: Dict[str, Any], folder_name: Optional[str] = None) -> bool:
        """Ingest a single file using Morphik API (async)"""
        try:
            with open(file_info['path'], 'rb') as f:
                file_content = f.read()

                fields = [
                    ('file', (file_info['name'], file_content, file_info['mime_type'])),
                    ('use_colpali', 'true')
                ]
                if folder_name:
                    fields.append(('folder_name', folder_name))
                
                body, content_type = encode_multipart_formdata(fields)

                headers = self._client.headers.copy()
                headers['Content-Type'] = content_type

                response = await self._client.post(
                    "/ingest/file",
                    content=body,
                    headers=headers,
                    timeout=INDIVIDUAL_TIMEOUT
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ Successfully ingested: {file_info['name']}")
                    return True
                else:
                    logger.error(f"❌ Failed to ingest {file_info['name']}: {response.status_code}")
                    return False
                    
        except httpx.RequestError as e:
            logger.error(f"❌ Failed to ingest {file_info['name']}: {e}")
            return False
        except IOError as e:
            logger.error(f"❌ File read error for {file_info['name']}: {e}")
            return False

    async def crawl_site_for_urls(self, site_url: str) -> List[str]:
        """
        Accepts a URL (including sitemaps) and crawls it to return a list of URLs.
        If the URL points to a sitemap or a sitemap index, it parses URLs directly.
        Otherwise, it uses crawl4ai for general-purpose web crawling.
        """
        clean_site = site_url.strip().lstrip('@')
        logger.info(f"Received request to crawl site: {clean_site}")

        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0, headers=headers) as client:
                try:
                    response = await client.get(clean_site)
                    response.raise_for_status()
                    content = response.text
                    
                    root = ET.fromstring(content)
                    namespace = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

                    if root.tag.endswith('sitemapindex'):
                        logger.info(f"Detected sitemap index at {clean_site}. Parsing...")
                        sitemap_urls = [elem.text for elem in root.findall('sm:sitemap/sm:loc', namespace)]
                        all_page_urls = []

                        for sitemap_url in sitemap_urls:
                            try:
                                sitemap_response = await client.get(sitemap_url)
                                sitemap_response.raise_for_status()
                                page_urls = _extract_urls_from_sitemap_content(sitemap_response.text, namespace)
                                all_page_urls.extend(page_urls)
                                logger.info(f"Found {len(page_urls)} URLs in sitemap {sitemap_url}")
                            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                                logger.warning(f"Failed to process sitemap {sitemap_url}: {e}")
                        # Filter out sitemaps from response
                        sitemap_urls_set = set(sitemap_urls)
                        all_page_urls = [url for url in all_page_urls if url not in sitemap_urls_set]
                        return all_page_urls

                    elif root.tag.endswith('urlset'):
                        logger.info(f"Detected a standard sitemap at {clean_site}. Parsing...")
                        return _extract_urls_from_sitemap_content(content, namespace)
                    else:
                        logger.info(f"URL content for {clean_site} is XML but not a sitemap. Falling back to crawler.")

                except (httpx.RequestError, httpx.HTTPStatusError, ET.ParseError):
                    logger.info(f"URL content for {clean_site} is not XML or failed to fetch. Falling back to crawler.")
            
            logger.info(f"Using crawl4ai for general crawling of {clean_site}")
            crawled_urls = []
            async with AsyncWebCrawler() as crawler:
                # Configure deep crawling using BFSDeepCrawlStrategy within CrawlerRunConfig.
                # The 'arun' method with a deep_crawl_strategy will perform a deep crawl.
                config = CrawlerRunConfig(
                    deep_crawl_strategy=BFSDeepCrawlStrategy(
                        max_depth=2,
                        max_pages=1000,
                        include_external=False  # This corresponds to the old scope='domain'
                    ),
                    check_robots_txt=True,  # This is the correct parameter to respect robots.txt
                    stream=True # To get results as an async generator
                )
                
                # The arun method returns an async generator when stream=True
                crawl_generator = await crawler.arun(
                    url=clean_site,
                    config=config
                )

                async for result in crawl_generator:
                    if result.success:
                        lang = _extract_lang_from_html(result.html)
                        if lang and lang.startswith("en"):
                            crawled_urls.append(result.url)
                        else:
                            logger.info(f"Skipping URL {result.url} due to non-English language detected ('{lang}').")

            return crawled_urls
        except Exception as e:
            logger.error(f"An unexpected error occurred while crawling {clean_site}: {e}", exc_info=True)
            return []

    async def ingest_urls(self, urls: List[str], folder_name: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Accepts one or more URLs, crawls them, and ingests the content as PDF.
        """
        logger.info(f"Received request to ingest from {len(urls)} URLs. Folder: {folder_name or 'N/A'}")

        if not urls:
            return {"successful": [], "failed": [], "skipped": []}

        successful_ingestions, failed_ingestions, skipped_ingestions = [], [], []

        run_config = CrawlerRunConfig(pdf=True)
        async with AsyncWebCrawler() as crawler:
            for url in urls:
                try:
                    result = await crawler.arun(url=url, config=run_config)
                    if not result.success:
                        failed_ingestions.append({"url": url, "error": result.error_message})
                        continue

                    lang = _extract_lang_from_html(result.html)
                    if not lang or not lang.startswith("en"):
                        skipped_ingestions.append({"url": url, "error": f"Language '{lang}' is not English."})
                        continue
                    
                    if not result.pdf:
                        failed_ingestions.append({"url": url, "error": "Failed to generate PDF."})
                        continue
                    
                    title = result.metadata.get('title') if result.metadata else None
                    
                    if title:
                        file_name = f"{_slugify(title, to_lower=False, separator='_')}.pdf"
                    else:
                        # Fallback to a filename derived from the URL if no title is available
                        parsed_url = urlparse(url)
                        # Use the last part of the path as the filename
                        url_path_last_segment = parsed_url.path.strip('/').split('/')[-1]
                        
                        if url_path_last_segment:
                            file_name_base = url_path_last_segment
                        else:
                            # If path is empty (e.g., homepage), use the hostname
                            file_name_base = parsed_url.hostname or f"crawled_{os.urandom(4).hex()}"
                        
                        file_name = f"{_slugify(file_name_base, to_lower=True)}.pdf"

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(result.pdf)
                        temp_file_path = Path(temp_file.name)

                    file_info = self.get_file_info(temp_file_path)
                    file_info['name'] = file_name

                    if await self.individual_ingest_file(file_info, folder_name):
                        successful_ingestions.append({"url": url, "filename": file_name})
                    else:
                        failed_ingestions.append({"url": url, "error": "Morphik API ingestion failed."})

                    os.unlink(temp_file_path)

                except Exception as e:
                    logger.error(f"An unexpected error occurred while processing {url}: {e}", exc_info=True)
                    failed_ingestions.append({"url": url, "error": str(e)})

        return {
            "successful": successful_ingestions,
            "failed": failed_ingestions,
            "skipped": skipped_ingestions
        }

def get_ingester() -> MorphikDocumentIngester:
    """Dependency provider for the ingester."""
    return MorphikDocumentIngester(base_url=BASE_URL)
