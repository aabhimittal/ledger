"""Environment snapshots: the half of recovery the log cannot provide.

The log replays an agent's *decisions*. It cannot replay the world the agent
mutated -- a half-written build tree, a database file, an installed package.
That is what a snapshot is for, and it is the part of LEDGER that durable
execution engines (Temporal, Step Functions, LangGraph checkpointers) leave
to you.

Three backends, one interface:

``DirSnapshotter``
    Userspace content-addressed copy of a directory tree. Portable, no
    privileges, dedupes unchanged files across snapshots -- which is the
    whole game for a day-long run where hour 9 looks much like hour 8.
``OverlayFSSnapshotter``
    Captures only the overlay upperdir (the delta), including whiteouts.
    Needs mount privileges.
``FirecrackerSnapshotter``
    Full microVM memory + device state via the Firecracker API socket.
    The only backend that restores *process* state, not just files.

Commit protocol for all three: write blobs, write ``manifest.json``, then
atomically write ``ref.json``. A snapshot without ``ref.json`` never existed.
Callers append the ``snapshot`` log record only *after* the commit, so a crash
in between leaves an unreferenced-but-valid snapshot (harmless) rather than a
log pointing at a snapshot that is not there (unrecoverable).
"""

from __future__ import annotations

import fnmatch
import hashlib
import http.client
import json
import os
import secrets
import shutil
import socket
import stat
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

from ._fsutil import atomic_write_json, clear_dir, fsync_dir, read_json

CHUNK = 1 << 20


class SnapshotError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# content-addressed blob store
# --------------------------------------------------------------------------
class CAS:
    """Content-addressed store. Identical bytes are stored once, ever."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bytes_written = 0
        self.bytes_deduped = 0

    def path(self, digest: str) -> Path:
        return self.root / digest[:2] / digest

    def has(self, digest: str) -> bool:
        return self.path(digest).exists()

    @staticmethod
    def digest_file(src: Path) -> tuple[str, int]:
        h, size = hashlib.sha256(), 0
        with open(src, "rb") as f:
            while chunk := f.read(CHUNK):
                h.update(chunk)
                size += len(chunk)
        return h.hexdigest(), size

    def put_file(self, src: Path) -> tuple[str, int]:
        digest, size = self.digest_file(src)
        dst = self.path(digest)
        if dst.exists():
            self.bytes_deduped += size
            return digest, size
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{digest}.tmp{os.getpid()}{secrets.token_hex(3)}")
        shutil.copyfile(src, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, dst)
        self.bytes_written += size
        return digest, size

    def put_bytes(self, data: bytes) -> tuple[str, int]:
        digest = hashlib.sha256(data).hexdigest()
        dst = self.path(digest)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{digest}.tmp{os.getpid()}")
            tmp.write_bytes(data)
            os.replace(tmp, dst)
            self.bytes_written += len(data)
        else:
            self.bytes_deduped += len(data)
        return digest, len(data)

    def materialize(self, digest: str, dst: Path, mode: int = 0o644) -> None:
        src = self.path(digest)
        if not src.exists():
            raise SnapshotError(f"missing blob {digest[:12]} for {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        os.chmod(dst, mode)

    def digests(self) -> set[str]:
        return {p.name for p in self.root.glob("*/*") if not p.name.startswith(".")}


# --------------------------------------------------------------------------
# snapshotter interface
# --------------------------------------------------------------------------
@runtime_checkable
class Snapshotter(Protocol):
    backend: str

    def available(self) -> bool: ...
    def capture(self, cas: CAS) -> dict: ...
    def restore(self, manifest: dict, cas: CAS) -> None: ...


DEFAULT_EXCLUDE = ("__pycache__", "*.pyc", ".git/objects/pack/tmp_*")


class DirSnapshotter:
    """Snapshot a directory tree into a CAS; restore by rehydrating it.

    Capture cost is one hash per file (plus a copy only for *changed* files).
    Restore cost is a full copy of the tree. That asymmetry is deliberate:
    snapshots happen every k steps, restores happen once per crash.
    """

    backend = "dir-cas"

    def __init__(self, root: str | Path, *, exclude: Iterable[str] = DEFAULT_EXCLUDE):
        self.root = Path(root)
        self.exclude = tuple(exclude)

    def available(self) -> bool:
        return True

    def _excluded(self, rel: str) -> bool:
        return any(
            fnmatch.fnmatch(rel, pat) or any(fnmatch.fnmatch(part, pat) for part in Path(rel).parts)
            for pat in self.exclude
        )

    def capture(self, cas: CAS) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        dirs: list[dict] = []
        files: list[dict] = []
        links: list[dict] = []
        total = 0
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            here = Path(dirpath)
            rel_dir = here.relative_to(self.root).as_posix()
            dirnames[:] = sorted(
                d for d in dirnames if not self._excluded((Path(rel_dir) / d).as_posix().lstrip("./"))
            )
            if rel_dir != ".":
                dirs.append({"path": rel_dir, "mode": stat.S_IMODE(here.stat().st_mode)})
            for name in sorted(filenames):
                p = here / name
                rel = (p.relative_to(self.root)).as_posix()
                if self._excluded(rel):
                    continue
                if p.is_symlink():
                    links.append({"path": rel, "target": os.readlink(p)})
                    continue
                if not p.is_file():  # sockets, fifos: not reproducible, skip loudly
                    files.append({"path": rel, "skipped": "not-a-regular-file"})
                    continue
                digest, size = cas.put_file(p)
                files.append({"path": rel, "digest": digest, "size": size,
                              "mode": stat.S_IMODE(p.stat().st_mode)})
                total += size
        return {
            "backend": self.backend,
            "root": str(self.root),
            "dirs": dirs,
            "files": files,
            "symlinks": links,
            "bytes": total,
        }

    def restore(self, manifest: dict, cas: CAS) -> None:
        keep = {p for p in self.exclude if "*" not in p and "/" not in p}
        clear_dir(self.root, keep=keep)
        for d in manifest.get("dirs", []):
            target = self.root / d["path"]
            target.mkdir(parents=True, exist_ok=True)
            os.chmod(target, d.get("mode", 0o755))
        for f in manifest.get("files", []):
            if "digest" not in f:
                continue
            cas.materialize(f["digest"], self.root / f["path"], f.get("mode", 0o644))
        for link in manifest.get("symlinks", []):
            p = self.root / link["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.is_symlink() or p.exists():
                p.unlink()
            os.symlink(link["target"], p)


class OverlayFSSnapshotter:
    """Snapshot the overlay *delta* (upperdir) instead of the whole tree.

    For a day-long run over a large base image this is the difference between
    snapshotting megabytes and gigabytes: the lowerdir never changes, so only
    the upperdir needs capturing. Whiteouts (deletions) are char devices 0:0
    and are recorded as such -- recreating them on restore needs ``mknod``,
    hence root.
    """

    backend = "overlayfs"

    def __init__(self, lower: str | Path, upper: str | Path, work: str | Path, merged: str | Path):
        self.lower, self.upper = Path(lower), Path(upper)
        self.work, self.merged = Path(work), Path(merged)

    @property
    def root(self) -> Path:
        """Where the agent works."""
        return self.merged

    def available(self) -> bool:
        if os.geteuid() != 0:
            return False
        try:
            return "overlay" in Path("/proc/filesystems").read_text()
        except OSError:
            return False

    def mount(self) -> None:
        for d in (self.lower, self.upper, self.work, self.merged):
            d.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["mount", "-t", "overlay", "overlay", "-o",
             f"lowerdir={self.lower},upperdir={self.upper},workdir={self.work}", str(self.merged)],
            check=True, capture_output=True,
        )

    def unmount(self) -> None:
        subprocess.run(["umount", str(self.merged)], check=False, capture_output=True)

    def capture(self, cas: CAS) -> dict:
        # Syncing the merged mount pushes dirty pages into upperdir first.
        subprocess.run(["sync", "-f", str(self.merged)], check=False, capture_output=True)
        delta: list[dict] = []
        for dirpath, _dirnames, filenames in os.walk(self.upper):
            here = Path(dirpath)
            rel_dir = here.relative_to(self.upper).as_posix()
            if rel_dir != ".":
                delta.append({"kind": "dir", "path": rel_dir,
                              "mode": stat.S_IMODE(here.stat().st_mode)})
            for name in sorted(filenames):
                p = here / name
                rel = p.relative_to(self.upper).as_posix()
                st = p.lstat()
                if stat.S_ISCHR(st.st_mode) and st.st_rdev == 0:
                    delta.append({"kind": "whiteout", "path": rel})
                elif stat.S_ISLNK(st.st_mode):
                    delta.append({"kind": "symlink", "path": rel, "target": os.readlink(p)})
                elif stat.S_ISREG(st.st_mode):
                    digest, size = cas.put_file(p)
                    delta.append({"kind": "file", "path": rel, "digest": digest,
                                  "size": size, "mode": stat.S_IMODE(st.st_mode)})
        return {"backend": self.backend, "lower": str(self.lower), "upper": str(self.upper),
                "merged": str(self.merged), "delta": delta}

    def restore(self, manifest: dict, cas: CAS) -> None:
        self.unmount()
        clear_dir(self.upper)
        shutil.rmtree(self.work, ignore_errors=True)
        for item in manifest.get("delta", []):
            target = self.upper / item["path"]
            kind = item["kind"]
            if kind == "dir":
                target.mkdir(parents=True, exist_ok=True)
                os.chmod(target, item.get("mode", 0o755))
            elif kind == "file":
                cas.materialize(item["digest"], target, item.get("mode", 0o644))
            elif kind == "symlink":
                target.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(item["target"], target)
            elif kind == "whiteout":
                target.parent.mkdir(parents=True, exist_ok=True)
                os.mknod(target, 0o600 | stat.S_IFCHR, os.makedev(0, 0))
        self.mount()


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, timeout: float = 30.0):
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._socket_path)
        self.sock = sock


class FirecrackerSnapshotter:
    """Full microVM snapshots over the Firecracker API socket.

    This is the only backend that survives a crash mid-`pip install`: it
    captures guest memory and device state, so restore resumes *processes*,
    not just files. Cost is the memory file -- snapshot every k steps means
    every k steps you write out the guest's RAM, so pair it with a large k or
    diff snapshots.
    """

    backend = "firecracker"

    def __init__(self, api_socket: str | Path, scratch: str | Path, *, snapshot_type: str = "Full"):
        self.api_socket = str(api_socket)
        self.scratch = Path(scratch)
        self.snapshot_type = snapshot_type

    def available(self) -> bool:
        return Path(self.api_socket).exists()

    def _api(self, method: str, route: str, body: dict | None = None) -> dict:
        conn = _UnixHTTPConnection(self.api_socket)
        try:
            payload = json.dumps(body).encode() if body is not None else None
            conn.request(method, route, body=payload,
                         headers={"Accept": "application/json", "Content-Type": "application/json"})
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status >= 400:
                raise SnapshotError(f"firecracker {method} {route} -> {resp.status}: {raw!r}")
            return json.loads(raw) if raw else {}
        finally:
            conn.close()

    def capture(self, cas: CAS) -> dict:
        self.scratch.mkdir(parents=True, exist_ok=True)
        tag = secrets.token_hex(6)
        state = self.scratch / f"vmstate-{tag}"
        mem = self.scratch / f"vmmem-{tag}"
        self._api("PATCH", "/vm", {"state": "Paused"})
        try:
            self._api("PUT", "/snapshot/create", {
                "snapshot_type": self.snapshot_type,
                "snapshot_path": str(state),
                "mem_file_path": str(mem),
            })
            state_digest, state_size = cas.put_file(state)
            mem_digest, mem_size = cas.put_file(mem)
        finally:
            self._api("PATCH", "/vm", {"state": "Resumed"})
            state.unlink(missing_ok=True)
            mem.unlink(missing_ok=True)
        return {"backend": self.backend, "api_socket": self.api_socket,
                "state": {"digest": state_digest, "size": state_size},
                "mem": {"digest": mem_digest, "size": mem_size},
                "bytes": state_size + mem_size}

    def restore(self, manifest: dict, cas: CAS) -> None:
        self.scratch.mkdir(parents=True, exist_ok=True)
        tag = secrets.token_hex(6)
        state = self.scratch / f"restore-state-{tag}"
        mem = self.scratch / f"restore-mem-{tag}"
        cas.materialize(manifest["state"]["digest"], state)
        cas.materialize(manifest["mem"]["digest"], mem)
        self._api("PUT", "/snapshot/load", {
            "snapshot_path": str(state),
            "mem_backend": {"backend_type": "File", "backend_path": str(mem)},
            "enable_diff_snapshots": False,
            "resume_vm": True,
        })


# --------------------------------------------------------------------------
# snapshot store
# --------------------------------------------------------------------------
@dataclass
class SnapshotRef:
    snapshot_id: str
    run_id: str
    step: int
    at_seq: int
    """Log sequence number this snapshot is consistent with.

    The invariant the whole recovery path rests on: the captured environment
    reflects exactly the effects of records ``1..at_seq``. Replay therefore
    resumes at ``at_seq + 1``.
    """
    backend: str
    created_at: float
    bytes: int = 0
    manifest_digest: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class SnapshotStore:
    def __init__(self, root: str | Path, cas: CAS):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cas = cas

    def _dir(self, snapshot_id: str) -> Path:
        return self.root / snapshot_id

    def save(self, run_id: str, step: int, at_seq: int, manifest: dict) -> SnapshotRef:
        snapshot_id = f"snap-{step:06d}-{secrets.token_hex(5)}"
        d = self._dir(snapshot_id)
        d.mkdir(parents=True, exist_ok=True)
        blob = json.dumps(manifest, sort_keys=True).encode()
        atomic_write_json(d / "manifest.json", manifest)
        ref = SnapshotRef(
            snapshot_id=snapshot_id, run_id=run_id, step=step, at_seq=at_seq,
            backend=manifest.get("backend", "unknown"), created_at=time.time(),
            bytes=int(manifest.get("bytes", 0)),
            manifest_digest=hashlib.sha256(blob).hexdigest(),
        )
        atomic_write_json(d / "ref.json", ref.to_dict())  # commit point
        fsync_dir(self.root)
        return ref

    def load(self, snapshot_id: str) -> tuple[SnapshotRef, dict]:
        d = self._dir(snapshot_id)
        if not (d / "ref.json").exists():
            raise SnapshotError(f"snapshot {snapshot_id} is not committed")
        return SnapshotRef(**read_json(d / "ref.json")), read_json(d / "manifest.json")

    def list(self, run_ids: Iterable[str] | None = None) -> list[SnapshotRef]:
        wanted = set(run_ids) if run_ids is not None else None
        out = []
        for d in sorted(self.root.iterdir()) if self.root.exists() else []:
            ref_path = d / "ref.json"
            if not ref_path.is_file():
                continue  # uncommitted: a crash mid-snapshot, correctly invisible
            ref = SnapshotRef(**read_json(ref_path))
            if wanted is None or ref.run_id in wanted:
                out.append(ref)
        return sorted(out, key=lambda r: (r.at_seq, r.created_at))

    def latest_at_or_before(self, run_ids: Iterable[str], seq: int) -> SnapshotRef | None:
        cands = [r for r in self.list(run_ids) if r.at_seq <= seq]
        return cands[-1] if cands else None

    def gc(self, *, dry_run: bool = False) -> dict:
        """Delete CAS blobs no committed snapshot references."""
        live: set[str] = set()
        for ref in self.list():
            manifest = read_json(self._dir(ref.snapshot_id) / "manifest.json")
            for f in manifest.get("files", []):
                if "digest" in f:
                    live.add(f["digest"])
            for item in manifest.get("delta", []):
                if "digest" in item:
                    live.add(item["digest"])
            for key in ("state", "mem"):
                if key in manifest:
                    live.add(manifest[key]["digest"])
        freed = 0
        removed = 0
        for digest in self.cas.digests() - live:
            p = self.cas.path(digest)
            freed += p.stat().st_size
            removed += 1
            if not dry_run:
                p.unlink()
        return {"removed": removed, "freed_bytes": freed, "live_blobs": len(live)}
