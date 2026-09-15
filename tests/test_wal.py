import json

import pytest

from ledger.wal import GENESIS, LogCorruption, RecordKind, WriteAheadLog, load, verify


def test_append_and_chain(tmp_path):
    path = tmp_path / "log.jsonl"
    with WriteAheadLog(path, sync="never") as log:
        log.append(RecordKind.RUN_START, {"run_id": "r1"})
        log.append(RecordKind.STEP_START, {}, step=1)
        log.append(RecordKind.STEP_END, {"done": True}, step=1)
        assert log.tail_seq == 3
        assert log.tail_step == 1

    records = load(path).records
    assert [r.seq for r in records] == [1, 2, 3]
    assert records[0].prev == GENESIS
    assert records[1].prev == records[0].h
    assert verify(path).ok


def test_reopen_continues_the_chain(tmp_path):
    path = tmp_path / "log.jsonl"
    WriteAheadLog(path, sync="never").append(RecordKind.NOTE, {"i": 1})
    log = WriteAheadLog(path, sync="never")
    assert log.tail_seq == 1
    log.append(RecordKind.NOTE, {"i": 2})
    assert verify(path).ok


def test_torn_tail_is_repaired(tmp_path):
    path = tmp_path / "log.jsonl"
    with WriteAheadLog(path, sync="never") as log:
        for i in range(4):
            log.append(RecordKind.NOTE, {"i": i})
    data = path.read_bytes()
    path.write_bytes(data[: -len(data.split(b"\n")[-2]) // 2])  # half a final line

    res = load(path)
    assert res.torn_offset is not None
    log = WriteAheadLog(path, sync="never")
    assert log.repaired_bytes > 0
    assert log.tail_seq == 3
    log.append(RecordKind.NOTE, {"i": "after-repair"})
    assert verify(path).ok


def test_complete_record_missing_newline(tmp_path):
    path = tmp_path / "log.jsonl"
    with WriteAheadLog(path, sync="never") as log:
        log.append(RecordKind.NOTE, {"i": 0})
        log.append(RecordKind.NOTE, {"i": 1})
    path.write_bytes(path.read_bytes().rstrip(b"\n"))

    log = WriteAheadLog(path, sync="never")
    assert log.repaired_bytes == 0  # nothing was lost, only the newline
    assert log.tail_seq == 2
    log.append(RecordKind.NOTE, {"i": 2})
    assert verify(path).ok


def test_midfile_corruption_is_not_silently_repaired(tmp_path):
    path = tmp_path / "log.jsonl"
    with WriteAheadLog(path, sync="never") as log:
        for i in range(4):
            log.append(RecordKind.NOTE, {"i": i})
    lines = path.read_text().splitlines()
    tampered = json.loads(lines[1])
    tampered["payload"]["i"] = 999
    lines[1] = json.dumps(tampered)
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(LogCorruption):
        load(path)
    with pytest.raises(LogCorruption):
        WriteAheadLog(path, sync="never")
    assert not verify(path).ok


def test_copy_prefix_preserves_hashes(tmp_path):
    src, dst = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    with WriteAheadLog(src, sync="never") as log:
        for i in range(5):
            log.append(RecordKind.NOTE, {"i": i}, step=i)

    assert WriteAheadLog.copy_prefix(src, dst, 3) == 3
    a, b = load(src).records, load(dst).records
    assert [r.h for r in a[:3]] == [r.h for r in b]
    child = WriteAheadLog(dst, sync="never")
    assert child.last_hash == a[2].h
    child.append(RecordKind.FORK, {"at_seq": 3})
    assert verify(dst).ok


def test_bad_sync_policy_rejected(tmp_path):
    with pytest.raises(ValueError):
        WriteAheadLog(tmp_path / "log.jsonl", sync="sometimes")
