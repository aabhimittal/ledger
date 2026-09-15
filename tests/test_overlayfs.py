"""Real overlayfs snapshots: mount, mutate, capture, restore.

These need root and a kernel that will actually mount overlay, so the fixture
probes by mounting for real and skips if it cannot. That probe is deliberately
not ``available()``: inside an unprivileged container ``available()`` can say
yes (euid 0, overlay listed in /proc/filesystems) and the mount still fail on
CAP_SYS_ADMIN, and a test that reports "skipped" there is honest where one that
errors is just noise.
"""

import os
import stat
import subprocess
from pathlib import Path

import pytest

from ledger.snapshots import CAS, OverlayFSSnapshotter, SnapshotStore


def _can_mount(tmp: Path) -> bool:
    if os.geteuid() != 0:
        return False
    dirs = {name: tmp / f"probe-{name}" for name in ("l", "u", "w", "m")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["mount", "-t", "overlay", "overlay", "-o",
         f"lowerdir={dirs['l']},upperdir={dirs['u']},workdir={dirs['w']}", str(dirs["m"])],
        capture_output=True,
    )
    if proc.returncode == 0:
        subprocess.run(["umount", str(dirs["m"])], check=False, capture_output=True)
        return True
    return False


@pytest.fixture
def overlay(tmp_path):
    if not _can_mount(tmp_path):
        pytest.skip("overlayfs mount unavailable (needs root + CAP_SYS_ADMIN)")
    lower = tmp_path / "lower"
    lower.mkdir()
    (lower / "base.txt").write_text("from the image\n")
    (lower / "keep.txt").write_text("untouched\n")
    (lower / "pkg").mkdir()
    (lower / "pkg" / "a.txt").write_text("a\n")

    snap = OverlayFSSnapshotter(lower, tmp_path / "upper", tmp_path / "work",
                                tmp_path / "merged")
    assert snap.available()
    snap.mount()
    try:
        yield snap, CAS(tmp_path / "cas")
    finally:
        snap.unmount()


def test_capture_records_only_the_delta(overlay):
    snap, cas = overlay
    (snap.merged / "new.txt").write_text("written by the agent\n")

    manifest = snap.capture(cas)

    paths = {item["path"] for item in manifest["delta"]}
    assert "new.txt" in paths
    assert "keep.txt" not in paths          # unchanged lower files cost nothing
    assert manifest["backend"] == "overlayfs"


def test_restore_rolls_back_writes_modifications_and_deletions(overlay):
    snap, cas = overlay
    merged = snap.merged
    (merged / "new.txt").write_text("v1\n")
    (merged / "base.txt").write_text("modified\n")      # copy-up
    (merged / "keep.txt").unlink()                      # whiteout

    manifest = snap.capture(cas)
    kinds = {item["path"]: item["kind"] for item in manifest["delta"]}
    assert kinds["keep.txt"] == "whiteout"
    assert kinds["base.txt"] == "file"

    # Now diverge further, the way a crashed run would have.
    (merged / "new.txt").write_text("v2-garbage\n")
    (merged / "later.txt").write_text("should vanish\n")
    (merged / "base.txt").write_text("more garbage\n")
    (merged / "pkg" / "a.txt").unlink()

    snap.restore(manifest, cas)

    assert (merged / "new.txt").read_text() == "v1\n"
    assert (merged / "base.txt").read_text() == "modified\n"
    assert not (merged / "later.txt").exists()
    assert not (merged / "keep.txt").exists()           # deletion stayed deleted
    assert (merged / "pkg" / "a.txt").read_text() == "a\n"   # rollback resurrects it


def test_whiteouts_are_real_char_devices_in_the_upper_dir(overlay):
    snap, cas = overlay
    (snap.merged / "keep.txt").unlink()
    snap.capture(cas)

    st = (snap.upper / "keep.txt").lstat()
    assert stat.S_ISCHR(st.st_mode)
    assert st.st_rdev == 0


def test_opaque_directories_survive_a_round_trip(overlay):
    """A deleted-and-recreated directory must not re-merge the lower one.

    Overlayfs marks this with ``trusted.overlay.opaque``. Lose that xattr and a
    restore silently resurrects files the agent deleted -- the manifest looks
    fine and the environment is wrong.
    """
    snap, cas = overlay
    merged = snap.merged
    import shutil

    shutil.rmtree(merged / "pkg")
    (merged / "pkg").mkdir()
    (merged / "pkg" / "b.txt").write_text("b\n")
    assert not (merged / "pkg" / "a.txt").exists()

    manifest = snap.capture(cas)
    pkg = next(i for i in manifest["delta"] if i["path"] == "pkg")
    assert "trusted.overlay.opaque" in pkg["xattrs"]

    (merged / "pkg" / "c.txt").write_text("c\n")
    snap.restore(manifest, cas)

    assert (merged / "pkg" / "b.txt").read_text() == "b\n"
    assert not (merged / "pkg" / "c.txt").exists()
    assert not (merged / "pkg" / "a.txt").exists()   # the opaque marker held


def test_symlinks_in_the_delta(overlay):
    snap, cas = overlay
    os.symlink("base.txt", snap.merged / "link")
    manifest = snap.capture(cas)

    (snap.merged / "link").unlink()
    snap.restore(manifest, cas)
    assert os.readlink(snap.merged / "link") == "base.txt"


def test_snapshot_store_round_trip_with_the_overlay_backend(overlay, tmp_path):
    """The store does not care which backend produced the manifest."""
    snap, cas = overlay
    store = SnapshotStore(tmp_path / "snaps", cas)
    (snap.merged / "work.txt").write_text("step 5\n")

    ref = store.save("run-1", 5, 42, snap.capture(cas))
    (snap.merged / "work.txt").write_text("step 9, then a crash\n")

    _, manifest = store.load(ref.snapshot_id)
    snap.restore(manifest, cas)
    assert (snap.merged / "work.txt").read_text() == "step 5\n"
    assert ref.backend == "overlayfs"
    assert ref.at_seq == 42
