"""Append-only, hash-chained write-ahead log.

Every observable event of a run lands here as one JSON object per line. Each
record commits to its predecessor's digest, so the log is a Merkle chain: a
forked run that byte-copies a prefix is cryptographically provably derived
from its parent, and silent middle-of-file edits cannot pass verification.

Crash tolerance: a process killed mid-``write`` can leave a torn final line.
That is expected and repairable (drop it -- the step it belonged to never
committed). A bad record *followed* by good records is real corruption and
raises, because the chain cannot be honestly recovered.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

GENESIS = "0" * 64


class RecordKind:
    """The closed vocabulary of log record kinds."""

    RUN_START = "run_start"
    FORK = "fork"
    STEP_START = "step_start"
    PROMPT = "prompt"
    SAMPLE = "sample"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ENTROPY = "entropy"
    SNAPSHOT = "snapshot"
    STEP_END = "step_end"
    NOTE = "note"
    RUN_END = "run_end"
    ABANDON = "abandon"
    """Declares a range of records dead: the attempt they belong to was cut
    short by a crash and superseded by what follows.

    Without this, a log that has been resumed once is no longer replayable. The
    interrupted attempt leaves an orphan ``prompt`` with no ``sample``, or a
    ``tool_call`` with no ``tool_result``, sitting in the middle of the history;
    a later replay reads it, looks for the matching record, and finds the *next*
    attempt's records instead. Marking the range explicitly -- rather than
    guessing by look-ahead -- keeps recovery composable, so a run that crashed
    three times replays as cleanly as one that never crashed."""


#: Records that carry bookkeeping rather than agent decisions. Replay skips
#: them transparently so the tape stays aligned with what the agent does.
TRANSPARENT = frozenset(
    {
        RecordKind.RUN_START,
        RecordKind.FORK,
        RecordKind.SNAPSHOT,
        RecordKind.NOTE,
        RecordKind.RUN_END,
        RecordKind.ABANDON,
    }
)


def abandoned_ranges(records: Iterable[Record]) -> list[tuple[int, int]]:
    """Half-open ``(from_seq, to_seq]`` ranges superseded by a later attempt."""
    return [(int(r.payload["from_seq"]), int(r.payload["to_seq"]))
            for r in records if r.kind == RecordKind.ABANDON]


class LogCorruption(RuntimeError):
    """A record that is not the final one fails chain verification."""


def canon(obj: Any) -> str:
    """Canonical JSON: stable key order, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def digest(seq: int, step: int | None, kind: str, payload: Any, prev: str) -> str:
    return hash_text(canon([seq, step, kind, payload, prev]))


@dataclass(frozen=True)
class Record:
    seq: int
    step: int | None
    kind: str
    payload: dict
    prev: str
    h: str

    @classmethod
    def build(cls, seq: int, step: int | None, kind: str, payload: dict, prev: str) -> "Record":
        return cls(seq, step, kind, payload, prev, digest(seq, step, kind, payload, prev))

    @classmethod
    def from_obj(cls, o: dict) -> "Record":
        return cls(int(o["seq"]), o["step"], str(o["kind"]), o["payload"], str(o["prev"]), str(o["h"]))

    @property
    def expected_hash(self) -> str:
        return digest(self.seq, self.step, self.kind, self.payload, self.prev)

    def to_line(self) -> str:
        return canon(
            {
                "seq": self.seq,
                "step": self.step,
                "kind": self.kind,
                "payload": self.payload,
                "prev": self.prev,
                "h": self.h,
            }
        ) + "\n"


@dataclass
class LoadResult:
    records: list[Record]
    torn_offset: int | None
    """Byte offset of an unusable trailing record, or ``None``."""
    ends_with_newline: bool


def load(path: str | Path) -> LoadResult:
    """Parse and verify a log, tolerating (but reporting) a torn final record."""
    p = Path(path)
    if not p.exists():
        return LoadResult([], None, True)
    data = p.read_bytes()
    records: list[Record] = []
    prev, expect, offset, torn = GENESIS, 1, 0, None
    lines = data.split(b"\n")
    for i, line in enumerate(lines):
        is_tail = i == len(lines) - 1
        if not line:
            if is_tail:
                break
            raise LogCorruption(f"{p}: empty record at line {i + 1}")
        try:
            rec = Record.from_obj(json.loads(line))
            ok = rec.seq == expect and rec.prev == prev and rec.h == rec.expected_hash
        except Exception:
            ok = False
        if not ok:
            if is_tail:
                torn = offset
                break
            raise LogCorruption(
                f"{p}: record at line {i + 1} (byte {offset}) fails verification "
                "and is followed by more records -- the chain is broken, not torn"
            )
        records.append(rec)
        prev, expect = rec.h, expect + 1
        offset += len(line) + 1
    return LoadResult(records, torn, (not data) or data.endswith(b"\n"))


class WriteAheadLog:
    """Open a log for appending, repairing a torn tail on the way in.

    ``sync="always"`` fsyncs every record. That is the only setting under
    which "the log survived the crash" is a real claim; ``"batch"`` and
    ``"never"`` trade that for throughput and are for tests and benchmarks.
    """

    def __init__(self, path: str | Path, *, sync: str = "always", repair: bool = True):
        if sync not in ("always", "batch", "never"):
            raise ValueError(f"bad sync policy: {sync!r}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sync = sync
        self.repaired_bytes = 0

        res = load(self.path)
        if res.torn_offset is not None:
            if not repair:
                raise LogCorruption(f"{self.path}: torn trailing record at byte {res.torn_offset}")
            self.repaired_bytes = self.path.stat().st_size - res.torn_offset
            with open(self.path, "r+b") as f:
                f.truncate(res.torn_offset)
                f.flush()
                os.fsync(f.fileno())
        elif not res.ends_with_newline:
            # A complete record whose trailing newline never reached disk.
            with open(self.path, "ab") as f:
                f.write(b"\n")
                f.flush()
                os.fsync(f.fileno())

        self._tail = res.records[-1] if res.records else None
        self._step = next((r.step for r in reversed(res.records) if r.step is not None), 0)
        self._f = open(self.path, "ab")

    # -- state ---------------------------------------------------------
    @property
    def tail_seq(self) -> int:
        return self._tail.seq if self._tail else 0

    @property
    def last_hash(self) -> str:
        return self._tail.h if self._tail else GENESIS

    @property
    def tail_step(self) -> int:
        return self._step

    @property
    def next_seq(self) -> int:
        return self.tail_seq + 1

    # -- writing -------------------------------------------------------
    def append(self, kind: str, payload: dict | None = None, step: int | None = None) -> Record:
        rec = Record.build(self.next_seq, step, kind, payload or {}, self.last_hash)
        self._f.write(rec.to_line().encode())
        self._f.flush()
        if self.sync == "always":
            os.fsync(self._f.fileno())
        self._tail = rec
        if step is not None:
            self._step = step
        return rec

    def sync_now(self) -> None:
        self._f.flush()
        if self.sync != "never":
            os.fsync(self._f.fileno())

    def close(self) -> None:
        if not self._f.closed:
            self.sync_now()
            self._f.close()

    def __enter__(self) -> "WriteAheadLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- reading -------------------------------------------------------
    def records(self, start_seq: int = 1) -> list[Record]:
        return [r for r in load(self.path).records if r.seq >= start_seq]

    def read(self, start_seq: int = 1) -> Iterator[Record]:
        yield from self.records(start_seq)

    # -- helpers -------------------------------------------------------
    @staticmethod
    def copy_prefix(src: str | Path, dst: str | Path, boundary_seq: int) -> int:
        """Byte-copy records with ``seq <= boundary_seq`` into a new log.

        Copying bytes rather than re-chaining keeps every digest identical, so
        a fork's ancestry is verifiable from the child log alone.
        """
        recs = load(src).records
        dstp = Path(dst)
        dstp.parent.mkdir(parents=True, exist_ok=True)
        copied = 0
        with open(dstp, "wb") as f:
            for r in recs:
                if r.seq > boundary_seq:
                    break
                f.write(r.to_line().encode())
                copied = r.seq
            f.flush()
            os.fsync(f.fileno())
        return copied


@dataclass
class VerifyReport:
    path: str
    records: int
    torn: bool
    ok: bool
    error: str | None = None

    def __str__(self) -> str:
        state = "ok" if self.ok else f"BROKEN: {self.error}"
        torn = " (torn tail)" if self.torn else ""
        return f"{self.path}: {self.records} records{torn} -- {state}"


def verify(path: str | Path) -> VerifyReport:
    try:
        res = load(path)
    except LogCorruption as e:
        return VerifyReport(str(path), 0, False, False, str(e))
    return VerifyReport(str(path), len(res.records), res.torn_offset is not None, True)
