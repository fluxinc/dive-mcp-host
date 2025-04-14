from os import getenv
from pathlib import Path

RESOURCE_DIR = Path(getenv("RESOURCE_DIR", Path.cwd()))
DIVE_CONFIG_DIR = Path(getenv("DIVE_CONFIG_DIR", Path.cwd()))


def write_then_replace(path: Path, content: str) -> None:
    """Write the content to a temporary file and then replace the target file."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(content)
    tmp_path.replace(path)
