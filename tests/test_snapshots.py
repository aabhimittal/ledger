import os

import pytest

from ledger.snapshots import (
    CAS,
    DirSnapshotter,
    FirecrackerSnapshotter,
    OverlayFSSnapshotter,
    SnapshotError,
    SnapshotStore,
)


def seed(root):
    (root / "sub").mkdir(parents=True, exist_ok=True)
    (root / "a.txt").write_text("alpha\n")
    (root / "sub" / "b.txt").write_text("beta\n")
    script = root / "run.sh"
    script.write_text("#!/bin/sh\n")
    os.chmod(script, 0o755)
    os.symlink("a.txt", root / "link")


def test_capture_restore_round_trip(tmp_path):
    ws, cas = tmp_path / "ws", CAS(tmp_path / "cas")
    ws.mkdir()
    seed(ws)
    snap = DirSnapshotter(ws)
    manifest = snap.capture(cas)

    # Mutate the world the way an agent would between snapshots.
    (ws / "a.txt").write_text("corrupted\n")
    (ws / "sub" / "b.txt").unlink()
    (ws / "new.txt").write_text("junk\n")

    snap.restore(manifest, cas)
    assert (ws / "a.txt").read_text() == "alpha\n"
    assert (ws / "sub" / "b.txt").read_text() == "beta\n"
    assert not (ws / "new.txt").exists()
    assert os.readlink(ws / "link") == "a.txt"
    assert os.stat(ws / "run.sh").st_mode & 0o111


def test_unchanged_files_are_deduplicated(tmp_path):
    ws, cas = tmp_path / "ws", CAS(tmp_path / "cas")
    ws.mkdir()
    seed(ws)
    snap = DirSnapshotter(ws)
    snap.capture(cas)
    first = cas.bytes_written
    cas.bytes_written = 0

    (ws / "a.txt").write_text("changed\n")
    snap.capture(cas)
    assert first > 0
    assert cas.bytes_written == len("changed\n")  # only the changed file is stored


def test_excluded_paths_are_skipped(tmp_path):
    ws, cas = tmp_path / "ws", CAS(tmp_path / "cas")
    (ws / "__pycache__").mkdir(parents=True)
    (ws / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    (ws / "keep.txt").write_text("k")
    manifest = DirSnapshotter(ws).capture(cas)
    assert [f["path"] for f in manifest["files"]] == ["keep.txt"]


def test_uncommitted_snapshot_is_invisible(tmp_path):
    cas = CAS(tmp_path / "cas")
    store = SnapshotStore(tmp_path / "snaps", cas)
    ref = store.save("run-1", 3, 12, {"backend": "dir-cas", "files": []})
    assert [r.snapshot_id for r in store.list()] == [ref.snapshot_id]

    (store.root / ref.snapshot_id / "ref.json").unlink()  # crash before commit
    assert store.list() == []
    with pytest.raises(SnapshotError):
        store.load(ref.snapshot_id)


def test_latest_at_or_before(tmp_path):
    cas = CAS(tmp_path / "cas")
    store = SnapshotStore(tmp_path / "snaps", cas)
    a = store.save("parent", 2, 10, {"backend": "dir-cas"})
    b = store.save("parent", 4, 20, {"backend": "dir-cas"})
    store.save("other", 6, 30, {"backend": "dir-cas"})

    assert store.latest_at_or_before(["parent"], 25).snapshot_id == b.snapshot_id
    assert store.latest_at_or_before(["parent"], 15).snapshot_id == a.snapshot_id
    assert store.latest_at_or_before(["parent"], 5) is None
    # a fork sees its ancestor's snapshots, not a sibling's
    assert store.latest_at_or_before(["parent", "child"], 100).snapshot_id == b.snapshot_id


def test_gc_removes_unreferenced_blobs(tmp_path):
    ws, cas = tmp_path / "ws", CAS(tmp_path / "cas")
    ws.mkdir()
    seed(ws)
    snap = DirSnapshotter(ws)
    store = SnapshotStore(tmp_path / "snaps", cas)
    store.save("r", 1, 5, snap.capture(cas))

    (ws / "a.txt").write_text("orphan-me\n")
    snap.capture(cas)  # blobs written, never committed to a snapshot

    report = store.gc(dry_run=True)
    assert report["removed"] == 1
    assert store.gc()["removed"] == 1
    assert store.gc()["removed"] == 0


def test_missing_blob_is_reported(tmp_path):
    ws, cas = tmp_path / "ws", CAS(tmp_path / "cas")
    ws.mkdir()
    seed(ws)
    manifest = DirSnapshotter(ws).capture(cas)
    digest = next(f["digest"] for f in manifest["files"] if f["path"] == "a.txt")
    cas.path(digest).unlink()
    with pytest.raises(SnapshotError, match="missing blob"):
        DirSnapshotter(ws).restore(manifest, cas)


def test_privileged_backends_degrade_instead_of_crashing(tmp_path):
    overlay = OverlayFSSnapshotter(tmp_path / "l", tmp_path / "u", tmp_path / "w", tmp_path / "m")
    assert overlay.available() is (os.geteuid() == 0 and overlay.available())
    fc = FirecrackerSnapshotter(tmp_path / "no-such.sock", tmp_path / "scratch")
    assert fc.available() is False
