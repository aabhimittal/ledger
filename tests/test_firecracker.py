"""Firecracker snapshots, exercised against a fake API socket.

No hypervisor here, and that is the point: the part most likely to be wrong is
the *protocol* -- which routes, in which order, with which JSON bodies, and
whether a failure mid-sequence still resumes the VM. A fake unix-socket server
tests all of that honestly. What it cannot test is whether Firecracker accepts
the bodies, so treat these as protocol tests, not proof against a real microVM.
"""

import json
import socket
import socketserver
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

from ledger.snapshots import CAS, FirecrackerSnapshotter, SnapshotError, SnapshotStore

STATE_BYTES = b"vm-state-blob"
MEM_BYTES = b"guest-memory-blob" * 64


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def address_string(self) -> str:  # client_address is meaningless on AF_UNIX
        return "local"

    def log_message(self, *args) -> None:
        pass

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length)) if length else {}

    def _handle(self, method: str) -> None:
        body = self._body()
        self.server.calls.append((method, self.path, body))

        if self.path in self.server.fail_routes:
            self.send_response(400)
            payload = json.dumps({"fault_message": "refused by fake"}).encode()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/snapshot/create":
            Path(body["snapshot_path"]).write_bytes(STATE_BYTES)
            Path(body["mem_file_path"]).write_bytes(MEM_BYTES)
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_PUT(self) -> None:
        self._handle("PUT")

    def do_PATCH(self) -> None:
        self._handle("PATCH")


class _UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def get_request(self):
        conn, _ = super().get_request()
        return conn, ("local", 0)


@pytest.fixture
def fake_vm(tmp_path):
    sock = tmp_path / "fc.sock"
    server = _UnixHTTPServer(str(sock), _Handler)
    server.calls: list[tuple[str, str, dict]] = []
    server.fail_routes: set[str] = set()
    # A short poll interval keeps shutdown() from waiting out the default 0.5 s,
    # which the suite would otherwise pay once per test in teardown.
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02},
                              daemon=True)
    thread.start()
    try:
        yield server, FirecrackerSnapshotter(sock, tmp_path / "scratch"), CAS(tmp_path / "cas")
    finally:
        server.shutdown()
        server.server_close()


def test_availability_follows_the_socket(tmp_path, fake_vm):
    server, snap, _ = fake_vm
    assert snap.available() is True
    assert FirecrackerSnapshotter(tmp_path / "nope.sock", tmp_path / "s").available() is False


def test_capture_pauses_snapshots_then_resumes(fake_vm):
    server, snap, cas = fake_vm

    manifest = snap.capture(cas)

    routes = [(method, path) for method, path, _ in server.calls]
    assert routes == [("PATCH", "/vm"), ("PUT", "/snapshot/create"), ("PATCH", "/vm")]
    assert server.calls[0][2] == {"state": "Paused"}
    assert server.calls[2][2] == {"state": "Resumed"}   # never left paused

    create = server.calls[1][2]
    assert create["snapshot_type"] == "Full"
    assert set(create) == {"snapshot_type", "snapshot_path", "mem_file_path"}

    assert manifest["backend"] == "firecracker"
    assert cas.path(manifest["state"]["digest"]).read_bytes() == STATE_BYTES
    assert cas.path(manifest["mem"]["digest"]).read_bytes() == MEM_BYTES
    assert manifest["bytes"] == len(STATE_BYTES) + len(MEM_BYTES)


def test_capture_cleans_up_its_scratch_files(fake_vm):
    server, snap, cas = fake_vm
    snap.capture(cas)
    # The blobs live in the CAS; leaving a copy of guest RAM behind per snapshot
    # would fill the disk within an hour of a day-long run.
    assert list(snap.scratch.iterdir()) == []


def test_a_failed_create_still_resumes_the_vm(fake_vm):
    server, snap, cas = fake_vm
    server.fail_routes = {"/snapshot/create"}

    with pytest.raises(SnapshotError, match="snapshot/create"):
        snap.capture(cas)

    routes = [(method, path) for method, path, _ in server.calls]
    assert routes[-1] == ("PATCH", "/vm")
    assert server.calls[-1][2] == {"state": "Resumed"}


def test_restore_loads_the_snapshot_and_resumes(fake_vm):
    server, snap, cas = fake_vm
    manifest = snap.capture(cas)
    server.calls.clear()

    snap.restore(manifest, cas)

    assert len(server.calls) == 1
    method, path, body = server.calls[0]
    assert (method, path) == ("PUT", "/snapshot/load")
    assert body["mem_backend"] == {"backend_type": "File",
                                   "backend_path": body["mem_backend"]["backend_path"]}
    assert body["resume_vm"] is True
    assert body["enable_diff_snapshots"] is False
    # The files handed to the hypervisor are the captured bytes, rehydrated.
    assert Path(body["snapshot_path"]).read_bytes() == STATE_BYTES
    assert Path(body["mem_backend"]["backend_path"]).read_bytes() == MEM_BYTES


def test_restore_reports_a_hypervisor_refusal(fake_vm):
    server, snap, cas = fake_vm
    manifest = snap.capture(cas)
    server.fail_routes = {"/snapshot/load"}

    with pytest.raises(SnapshotError, match="snapshot/load"):
        snap.restore(manifest, cas)


def test_a_missing_blob_is_caught_before_the_api_call(fake_vm):
    server, snap, cas = fake_vm
    manifest = snap.capture(cas)
    cas.path(manifest["mem"]["digest"]).unlink()
    server.calls.clear()

    with pytest.raises(SnapshotError, match="missing blob"):
        snap.restore(manifest, cas)
    assert server.calls == []   # the VM was never touched


def test_store_round_trip_dedupes_identical_memory(fake_vm, tmp_path):
    server, snap, cas = fake_vm
    store = SnapshotStore(tmp_path / "snaps", cas)

    first = store.save("run-1", 10, 99, snap.capture(cas))
    written_after_first = cas.bytes_written
    second = store.save("run-1", 20, 180, snap.capture(cas))

    # The fake returns identical bytes, so the second capture stores nothing new.
    # On a real VM this is what makes diff-style cadences affordable.
    assert cas.bytes_written == written_after_first
    assert cas.bytes_deduped > 0
    assert store.latest_at_or_before(["run-1"], 150).snapshot_id == first.snapshot_id
    assert store.latest_at_or_before(["run-1"], 500).snapshot_id == second.snapshot_id
