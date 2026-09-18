"""Firecracker snapshots against the *real* binary.

``test_firecracker.py`` drives a fake socket, which pins the protocol LEDGER
*believes* is right. It cannot catch the one failure that matters most: the real
API having a different shape than assumed. These tests close that gap without
needing a bootable guest, because Firecracker starts its API server before it
touches KVM -- so the real request parser can judge the real payloads.

The trick is that Firecracker's three rejection modes are distinguishable:

=====================================  ==========================================
``Invalid request method and/or path``  the route does not exist
``deserializing the json body``        the route exists, the body is wrong
``not supported before starting``      route and body both accepted; only the
                                       VM's lifecycle state is wrong
=====================================  ==========================================

So "the payload reached a *semantic* error" is a positive result, and there is a
negative control below to prove the assertion is not vacuous.

Set ``LEDGER_FIRECRACKER_BIN`` (or have ``firecracker`` on PATH) to run these.
The full boot/snapshot/restore test additionally needs ``/dev/kvm`` plus guest
images in ``LEDGER_FC_KERNEL`` and ``LEDGER_FC_ROOTFS``; it skips otherwise, and
a skip is the honest outcome on a host that cannot virtualise -- including inside
a microVM, where nested KVM is usually unavailable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from ledger.snapshots import CAS, FirecrackerSnapshotter, SnapshotError

PATH_ERROR = "Invalid request method and/or path"
BODY_ERROR = "deserializing the json body"
LIFECYCLE_ERROR = "not supported before starting the microVM"


def _binary() -> str | None:
    return os.environ.get("LEDGER_FIRECRACKER_BIN") or shutil.which("firecracker")


requires_firecracker = pytest.mark.skipif(
    _binary() is None,
    reason="no firecracker binary (set LEDGER_FIRECRACKER_BIN or put it on PATH)",
)


class _Vmm:
    """A real Firecracker process serving its API on a unix socket."""

    def __init__(self, tmp_path: Path, name: str = "fc"):
        self.socket = tmp_path / f"{name}.sock"
        self.log = tmp_path / f"{name}.log"
        self.proc = subprocess.Popen(
            [str(_binary()), "--api-sock", str(self.socket)],
            stdout=open(self.log, "wb"), stderr=subprocess.STDOUT,
        )
        for _ in range(100):  # the socket appears a beat after exec
            if self.socket.exists():
                return
            if self.proc.poll() is not None:
                raise RuntimeError(f"firecracker exited: {self.log.read_text()[:400]}")
            time.sleep(0.02)
        raise RuntimeError("firecracker never created its API socket")

    def snapshotter(self, tmp_path: Path) -> FirecrackerSnapshotter:
        return FirecrackerSnapshotter(self.socket, tmp_path / "scratch")

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


@pytest.fixture
def vmm(tmp_path):
    """A fresh VMM per test: a failed snapshot/load is fatal to the process."""
    vm = _Vmm(tmp_path)
    try:
        yield vm
    finally:
        vm.close()


@requires_firecracker
def test_the_binary_is_the_version_we_think_it_is():
    out = subprocess.run([str(_binary()), "--version"], capture_output=True, text=True)
    assert out.returncode == 0
    assert "Firecracker v" in out.stdout


@requires_firecracker
def test_capture_payloads_are_accepted_by_the_real_api(vmm, tmp_path):
    """Pause + snapshot/create reach a lifecycle error, not a parse error."""
    snap = vmm.snapshotter(tmp_path)
    with pytest.raises(SnapshotError) as excinfo:
        snap.capture(CAS(tmp_path / "cas"))

    message = str(excinfo.value)
    assert LIFECYCLE_ERROR in message, message
    assert PATH_ERROR not in message
    assert BODY_ERROR not in message


@requires_firecracker
def test_restore_payload_is_parsed_including_the_nested_mem_backend(vmm, tmp_path):
    """snapshot/load gets as far as opening the file we named.

    Reaching file-open means the whole body deserialized -- ``snapshot_path``,
    ``resume_vm``, ``enable_diff_snapshots`` and the nested ``mem_backend``
    object. That object is the part most likely to drift between versions.
    """
    cas = CAS(tmp_path / "cas")
    state_digest, _ = cas.put_bytes(b"not-a-real-snapshot")
    mem_digest, _ = cas.put_bytes(b"not-real-memory")
    manifest = {"backend": "firecracker",
                "state": {"digest": state_digest, "size": 19},
                "mem": {"digest": mem_digest, "size": 15}}

    snap = vmm.snapshotter(tmp_path)
    with pytest.raises(SnapshotError) as excinfo:
        snap.restore(manifest, cas)

    message = str(excinfo.value)
    assert PATH_ERROR not in message, message
    assert BODY_ERROR not in message, message
    # It failed on the *contents* of the snapshot file, which is as far as a
    # fabricated snapshot can get.
    assert "napshot" in message


@requires_firecracker
def test_negative_control_a_wrong_body_really_is_rejected(vmm, tmp_path):
    """Without this, the assertions above could pass vacuously."""
    snap = vmm.snapshotter(tmp_path)

    with pytest.raises(SnapshotError) as unknown_field:
        snap._api("PUT", "/snapshot/create", {
            "snapshot_type": "Full", "snapshot_path": "/tmp/s",
            "mem_file_path": "/tmp/m", "not_a_real_field": 1})
    assert BODY_ERROR in str(unknown_field.value)

    with pytest.raises(SnapshotError) as unknown_route:
        snap._api("PUT", "/snapshot/craete", {"snapshot_type": "Full"})
    assert PATH_ERROR in str(unknown_route.value)

    with pytest.raises(SnapshotError) as bad_enum:
        snap._api("PATCH", "/vm", {"state": "Sleeping"})
    assert BODY_ERROR in str(bad_enum.value)


@requires_firecracker
def test_snapshot_type_enum_matches_what_firecracker_accepts(vmm, tmp_path):
    """``Full`` must remain a valid variant; ``Diff`` is the other one."""
    snap = vmm.snapshotter(tmp_path)
    for variant, ok in (("Full", True), ("Diff", True), ("Complete", False)):
        with pytest.raises(SnapshotError) as excinfo:
            snap._api("PUT", "/snapshot/create", {
                "snapshot_type": variant, "snapshot_path": "/tmp/s",
                "mem_file_path": "/tmp/m"})
        rejected_for_body = BODY_ERROR in str(excinfo.value)
        assert rejected_for_body is not ok, f"{variant}: {excinfo.value}"


# --------------------------------------------------------------------------
# the full thing, where the host can virtualise
# --------------------------------------------------------------------------
KERNEL = os.environ.get("LEDGER_FC_KERNEL")
ROOTFS = os.environ.get("LEDGER_FC_ROOTFS")

requires_kvm = pytest.mark.skipif(
    not (Path("/dev/kvm").exists() and KERNEL and ROOTFS),
    reason="needs /dev/kvm plus LEDGER_FC_KERNEL and LEDGER_FC_ROOTFS guest images",
)


def _boot(snap: FirecrackerSnapshotter, rootfs: Path) -> None:
    snap._api("PUT", "/boot-source", {
        "kernel_image_path": KERNEL,
        "boot_args": "console=ttyS0 reboot=k panic=1 pci=off",
    })
    snap._api("PUT", "/drives/rootfs", {
        "drive_id": "rootfs", "path_on_host": str(rootfs),
        "is_root_device": True, "is_read_only": False,
    })
    snap._api("PUT", "/machine-config", {"vcpu_count": 1, "mem_size_mib": 128})
    snap._api("PUT", "/actions", {"action_type": "InstanceStart"})
    time.sleep(2)  # let the guest get past early boot before we freeze it


@requires_firecracker
@requires_kvm
def test_real_microvm_snapshot_and_restore(tmp_path):
    """Capture a booted guest, then restore it into a *different* process.

    This is the scenario the backend exists for: the original VMM is gone, and
    recovery has to rebuild guest memory and device state from the CAS alone.
    """
    rootfs = tmp_path / "rootfs.ext4"
    shutil.copyfile(ROOTFS, rootfs)  # boot writes to it; keep the original clean
    cas = CAS(tmp_path / "cas")

    first = _Vmm(tmp_path, "boot")
    try:
        snap = first.snapshotter(tmp_path)
        _boot(snap, rootfs)
        manifest = snap.capture(cas)
    finally:
        first.close()

    assert manifest["mem"]["size"] >= 128 * 1024 * 1024 * 0.5  # guest RAM, really captured
    assert cas.has(manifest["state"]["digest"])

    second = _Vmm(tmp_path, "restored")
    try:
        restored = second.snapshotter(tmp_path)
        restored.restore(manifest, cas)          # raises if the hypervisor refuses
        # The restored guest is live, not merely loaded: pausing it now succeeds,
        # where the same call on an unstarted VMM raises the lifecycle error.
        restored._api("PATCH", "/vm", {"state": "Paused"})
        restored._api("PATCH", "/vm", {"state": "Resumed"})
    finally:
        second.close()
