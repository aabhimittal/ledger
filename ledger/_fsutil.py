"""Small filesystem helpers with the durability guarantees LEDGER relies on."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def fsync_dir(path: Path) -> None:
    """Persist a directory entry so a rename survives a power loss."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:  # some filesystems refuse fsync on directories
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write then rename, so readers never observe a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def atomic_write_json(path: Path, obj: Any, *, indent: int | None = 2) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=indent, sort_keys=True).encode())


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def clear_dir(root: Path, *, keep: set[str] = frozenset()) -> None:
    """Empty ``root`` in place, leaving top-level names in ``keep`` alone."""
    root.mkdir(parents=True, exist_ok=True)
    for child in root.iterdir():
        if child.name in keep:
            continue
        # Only real directories get rmtree. Everything else -- files, symlinks,
        # fifos, and the character devices overlayfs uses as whiteouts -- is
        # unlinked: opening a whiteout device to walk it fails with ENXIO.
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
