"""Reading a log back: step summaries, rendering, and run-vs-run diffs.

``diff_runs`` is the piece failure attribution actually consumes: given a parent
run and a fork of it, report the first place the two histories part company and
what changed there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .wal import TRANSPARENT, Record, RecordKind, abandoned_ranges, load


@dataclass
class StepView:
    step: int
    first_seq: int
    last_seq: int
    samples: int = 0
    tool_calls: list[str] = field(default_factory=list)
    entropy_draws: int = 0
    notes: list[dict] = field(default_factory=list)
    closed: bool = False
    """False for the step a crash interrupted -- it has no ``step_end``."""

    @property
    def label(self) -> str:
        tools = ",".join(self.tool_calls) or "-"
        state = "" if self.closed else "  <-- unclosed (crash landed here)"
        return (f"step {self.step:>4}  seq {self.first_seq}-{self.last_seq}  "
                f"samples={self.samples} tools={tools} entropy={self.entropy_draws}{state}")


def summarize(records: Iterable[Record]) -> list[StepView]:
    steps: dict[int, StepView] = {}
    for rec in records:
        if rec.step is None:
            continue
        view = steps.get(rec.step)
        if view is None:
            view = steps[rec.step] = StepView(rec.step, rec.seq, rec.seq)
        view.last_seq = max(view.last_seq, rec.seq)
        view.first_seq = min(view.first_seq, rec.seq)
        if rec.kind == RecordKind.SAMPLE:
            view.samples += 1
        elif rec.kind == RecordKind.TOOL_CALL:
            view.tool_calls.append(str(rec.payload.get("tool")))
        elif rec.kind == RecordKind.ENTROPY:
            view.entropy_draws += 1
        elif rec.kind == RecordKind.NOTE:
            view.notes.append(rec.payload)
        elif rec.kind == RecordKind.STEP_END:
            view.closed = True
    return [steps[k] for k in sorted(steps)]


def live_records(records: Iterable[Record]) -> list[Record]:
    """Drop bookkeeping and records retired by an ``abandon`` marker."""
    records = list(records)
    dead = abandoned_ranges(records)
    return [r for r in records
            if r.kind not in TRANSPARENT and not any(lo < r.seq <= hi for lo, hi in dead)]


def render(records: Iterable[Record], *, kinds: Iterable[str] | None = None,
           width: int = 90) -> str:
    """Raw record stream. Records retired by a recovery are marked ``x``."""
    records = list(records)
    wanted = set(kinds) if kinds else None
    dead = abandoned_ranges(records)
    lines = []
    for rec in records:
        if wanted and rec.kind not in wanted:
            continue
        mark = "x" if any(lo < rec.seq <= hi for lo, hi in dead) else " "
        step = "-" if rec.step is None else str(rec.step)
        lines.append(f"{mark}{rec.seq:>6}  s{step:<4} {rec.kind:<12} {_detail(rec)[:width]}")
    return "\n".join(lines)


def _detail(rec: Record) -> str:
    p = rec.payload
    k = rec.kind
    if k == RecordKind.PROMPT:
        return f"{p.get('chars')} chars  h={str(p.get('prompt_hash'))[:10]}"
    if k == RecordKind.SAMPLE:
        return repr(str(p.get("text", ""))[:70])
    if k == RecordKind.TOOL_CALL:
        return f"{p.get('tool')}({_short(p.get('args'))}) [{p.get('kind')}/{p.get('scope')}]"
    if k == RecordKind.TOOL_RESULT:
        state = "ok" if p.get("ok") else f"ERR {p.get('error')}"
        return f"{p.get('call_id')} {state} {p.get('ms')}ms {_short(p.get('result'))}"
    if k == RecordKind.ENTROPY:
        return f"{p.get('dkind')}={_short(p.get('value'))}"
    if k == RecordKind.SNAPSHOT:
        return f"{p.get('snapshot_id')} at_seq={p.get('at_seq')} {p.get('bytes')}B"
    return _short(p)


def _short(obj: Any, n: int = 60) -> str:
    s = repr(obj)
    return s if len(s) <= n else s[: n - 3] + "..."


# --------------------------------------------------------------------------
# run-vs-run comparison
# --------------------------------------------------------------------------
@dataclass
class RunDiff:
    shared_prefix_seq: int
    """Last sequence number where both runs are byte-identical."""
    diverged_at_step: int | None
    kind: str | None
    left: str | None = None
    right: str | None = None

    def __str__(self) -> str:
        if self.diverged_at_step is None:
            return f"identical through seq {self.shared_prefix_seq}"
        return (f"shared prefix through seq {self.shared_prefix_seq}; diverged at step "
                f"{self.diverged_at_step} on {self.kind}\n  left : {self.left}\n"
                f"  right: {self.right}")


def diff_runs(left: Iterable[Record], right: Iterable[Record]) -> RunDiff:
    """Find the first place two runs actually behave differently.

    Two passes in one: while the record hashes match, the runs are provably
    byte-identical (that is what a forked prefix guarantees). Past that point
    hashes necessarily differ -- the chain has moved -- so comparison falls back
    to content, which skips over the bookkeeping a fork inserts and lands on the
    first genuinely different decision.
    """
    lhs = live_records(left)
    rhs = live_records(right)
    shared = 0
    for a, b in zip(lhs, rhs):
        if a.h == b.h:
            shared = a.seq
            continue
        if a.kind == b.kind and a.payload == b.payload:
            continue  # same decision, different position in the chain
        step = a.step if a.step is not None else b.step
        return RunDiff(shared, step, f"{a.kind}/{b.kind}", _detail(a), _detail(b))
    return RunDiff(shared, None, None)


def load_run(path: str | Path) -> list[Record]:
    return load(path).records
