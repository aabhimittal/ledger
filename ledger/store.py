"""On-disk layout for runs, snapshots and blobs.

::

    <root>/runs/<run_id>/log.jsonl     the write-ahead log
    <root>/runs/<run_id>/meta.json     run status and lineage
    <root>/snapshots/<snap_id>/        manifest.json + ref.json (commit marker)
    <root>/cas/<ab>/<sha256>           deduplicated file content

The store deliberately lives *outside* the agent workspace. Putting it inside
would make snapshots capture the log that describes them.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ._fsutil import atomic_write_json, read_json
from .snapshots import CAS, SnapshotStore


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{secrets.token_hex(4)}"


@dataclass
class RunMeta:
    run_id: str
    agent: str = "agent"
    created_at: float = field(default_factory=time.time)
    status: str = "running"
    """``running`` | ``completed`` | ``failed`` | ``max_steps`` | ``abandoned``"""
    mode: str = "live"
    parent_run_id: str | None = None
    forked_at_step: int | None = None
    forked_at_seq: int | None = None
    workspace: str | None = None
    steps: int = 0
    replayed_steps: int = 0
    result: Any = None
    error: str | None = None
    finished_at: float | None = None
    tags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunMeta":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


class RunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "runs").mkdir(parents=True, exist_ok=True)
        self.cas = CAS(self.root / "cas")
        self.snapshots = SnapshotStore(self.root / "snapshots", self.cas)

    # -- paths ---------------------------------------------------------
    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / run_id

    def log_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "log.jsonl"

    def meta_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "meta.json"

    # -- metadata ------------------------------------------------------
    def create_run(self, meta: RunMeta) -> RunMeta:
        d = self.run_dir(meta.run_id)
        if d.exists():
            raise FileExistsError(f"run {meta.run_id} already exists")
        d.mkdir(parents=True)
        self.save_meta(meta)
        return meta

    def save_meta(self, meta: RunMeta) -> None:
        atomic_write_json(self.meta_path(meta.run_id), meta.to_dict())

    def load_meta(self, run_id: str) -> RunMeta:
        p = self.meta_path(run_id)
        if not p.exists():
            raise KeyError(f"no such run: {run_id}")
        return RunMeta.from_dict(read_json(p))

    def exists(self, run_id: str) -> bool:
        return self.meta_path(run_id).exists()

    def list_runs(self) -> list[RunMeta]:
        out = []
        for d in (self.root / "runs").iterdir():
            if (d / "meta.json").is_file():
                out.append(self.load_meta(d.name))
        return sorted(out, key=lambda m: m.created_at)

    # -- lineage -------------------------------------------------------
    def lineage(self, run_id: str) -> list[RunMeta]:
        """Root-first chain of ancestors, ending with ``run_id`` itself."""
        chain: list[RunMeta] = []
        seen: set[str] = set()
        cur: str | None = run_id
        while cur and cur not in seen:
            seen.add(cur)
            meta = self.load_meta(cur)
            chain.append(meta)
            cur = meta.parent_run_id
        return list(reversed(chain))

    def lineage_ids(self, run_id: str) -> list[str]:
        return [m.run_id for m in self.lineage(run_id)]

    def children(self, run_id: str) -> list[RunMeta]:
        return [m for m in self.list_runs() if m.parent_run_id == run_id]

    def tree(self, run_id: str | None = None) -> dict[str, list[str]]:
        kids: dict[str, list[str]] = {}
        for m in self.list_runs():
            kids.setdefault(m.parent_run_id or "", []).append(m.run_id)
        return kids
