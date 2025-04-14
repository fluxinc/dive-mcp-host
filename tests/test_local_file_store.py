import base64
import tempfile

import pytest

from dive_mcp_host.httpd.store.cache import CacheKeys, LocalFileCache
from dive_mcp_host.httpd.store.local import LocalStore


@pytest.fixture
def cleanup_cache():
    """Cleanup the cache."""
    cache = LocalFileCache()
    cache.delete(CacheKeys.LIST_TOOLS)
    try:
        yield
    finally:
        cache.delete(CacheKeys.LIST_TOOLS)


def test_local_file_cache(cleanup_cache):
    """Test the local file cache."""
    cache = LocalFileCache()

    assert not cache.get_cache_file_path(CacheKeys.LIST_TOOLS).exists()
    assert cache.get(CacheKeys.LIST_TOOLS) is None

    cache.set(CacheKeys.LIST_TOOLS, "test")
    assert cache.get(CacheKeys.LIST_TOOLS) == "test"
    assert cache.get_cache_file_path(CacheKeys.LIST_TOOLS).exists()


def test_local_store_get_image():
    """Test the local store get image."""
    store = LocalStore()
    png_data = "iVBORw0KGgoAAAANSUhEUgAAAgAAAAABCAYAAACouxZ2AAAAFklEQVR4AWMYBaNgFIyCUTAKRsHIAwAIAQABmducEQAAAABJRU5ErkJggg=="  # noqa: E501
    tmpfile = tempfile.TemporaryFile()  # noqa: SIM115
    tmpfile.write(base64.b64decode(png_data))
    tmpfile.flush()
    image = store.get_image(tmpfile.name)
    assert image
    tmpfile.close()
