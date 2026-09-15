"""Shared fixtures: a small agent with one INTERNAL and one EXTERNAL tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ledger import (
    AgentRunner,
    CallableModel,
    EffectScope,
    RunStore,
    StepContext,
    StepResult,
    ToolKind,
    ToolRegistry,
)
from ledger.wal import Record, RecordKind, WriteAheadLog, load


@dataclass
class Sink:
    """Stands in for the outside world."""

    sent: list[dict] = field(default_factory=list)
    charged: list[dict] = field(default_factory=list)
    published: list[dict] = field(default_factory=list)
    internal_calls: int = 0
    charge_calls: int = 0


class NoteAgent:
    """Samples a line per step, appends it to a file, mails a report at step 3.

    The prompt embeds the agent's own accumulated state, so a replay whose
    in-memory state was rebuilt incorrectly fails the prompt-hash check rather
    than quietly producing a different run.
    """

    def __init__(self, total_steps: int = 6, report_step: int = 3, use_entropy: bool = False):
        self.total_steps = total_steps
        self.report_step = report_step
        self.use_entropy = use_entropy
        self.lines: list[str] = []
        self.stamps: list[float] = []

    def step(self, ctx: StepContext) -> StepResult | None:
        prompt = f"step {ctx.step}; so far: {self.lines}"
        text = ctx.sample(prompt).text
        self.lines.append(text)
        if self.use_entropy:
            self.stamps.append(ctx.entropy.time())
        ctx.call("append_line", path="notes.txt", text=text)
        if ctx.step == self.report_step:
            ctx.call("send_report", to="ops@example.com", body=text)
        if ctx.step >= self.total_steps:
            return ctx.done({"lines": self.lines, "stamps": self.stamps})
        return None


def make_tools(sink: Sink) -> ToolRegistry:
    tools = ToolRegistry()

    def append_line(path: str, text: str, workspace: Path) -> int:
        """Deterministic given the restored workspace: appends and returns the line count."""
        sink.internal_calls += 1
        target = workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a") as f:
            f.write(text + "\n")
        return sum(1 for _ in open(target))

    def send_report(to: str, body: str) -> str:
        sink.sent.append({"to": to, "body": body})
        return f"msg-{len(sink.sent)}"

    def charge(amount: int, idempotency_key: str | None = None) -> str:
        sink.charge_calls += 1
        if any(c["key"] == idempotency_key for c in sink.charged):
            return "duplicate-suppressed"
        sink.charged.append({"amount": amount, "key": idempotency_key})
        return f"charge-{len(sink.charged)}"

    def publish(doc: str, idempotency_key: str | None = None) -> str:
        """Irreversible, but with a read-back path -- the only honest recovery story."""
        sink.published.append({"doc": doc, "key": idempotency_key})
        return f"pub-{len(sink.published)}"

    tools.register("append_line", append_line, kind=ToolKind.PURE, scope=EffectScope.INTERNAL)
    tools.register("send_report", send_report, kind=ToolKind.IRREVERSIBLE,
                   scope=EffectScope.EXTERNAL)
    tools.register("charge", charge, kind=ToolKind.IDEMPOTENT, scope=EffectScope.EXTERNAL,
                   idempotency_key=lambda args: f"chg-{args['amount']}",
                   reconcile=lambda key: (
                       any(c["key"] == key for c in sink.charged),
                       "reconciled-charge",
                   ))
    tools.register("publish", publish, kind=ToolKind.IRREVERSIBLE, scope=EffectScope.EXTERNAL,
                   idempotency_key=lambda args: f"doc-{args['doc']}",
                   reconcile=lambda key: (
                       any(p["key"] == key for p in sink.published),
                       "reconciled-publish",
                   ))
    return tools


class EffectAgent:
    """Calls one declared-effect tool at a chosen step, then finishes."""

    def __init__(self, tool: str = "publish", args: dict | None = None, at_step: int = 3,
                 total_steps: int = 4):
        self.tool = tool
        self.args = args if args is not None else {"doc": "report"}
        self.at_step = at_step
        self.total_steps = total_steps
        self.effect: Any = None

    def step(self, ctx: StepContext) -> StepResult | None:
        text = ctx.sample(f"step {ctx.step}").text
        ctx.call("append_line", path="notes.txt", text=text)
        if ctx.step == self.at_step:
            self.effect = ctx.call(self.tool, **self.args)
        if ctx.step >= self.total_steps:
            return ctx.done(self.effect)
        return None


def step_number(prompt: str) -> str:
    return prompt.split()[1].rstrip(";")


def deterministic_model() -> CallableModel:
    """Reply depends only on the prompt, so live and replayed segments agree."""
    return CallableModel(lambda p: f"note-{step_number(p)}")


@dataclass
class Harness:
    root: Path
    store: RunStore
    ws: Path
    sink: Sink
    tools: ToolRegistry
    model: Any
    runner: AgentRunner


def build(root: Path, *, model: Any = None, agent: Callable[[], Any] | None = None,
          sink: Sink | None = None, **runner_kw: Any) -> Harness:
    root = Path(root)
    ws = root / "work"
    ws.mkdir(parents=True, exist_ok=True)
    store = RunStore(root / ".ledger")
    sink = sink if sink is not None else Sink()
    tools = make_tools(sink)
    model = model or deterministic_model()
    kw = {"snapshot_every": 2, "sync": "never"} | runner_kw
    runner = AgentRunner(agent_factory=agent or NoteAgent, store=store, workspace=ws,
                         model=model, tools=tools, **kw)
    return Harness(root, store, ws, sink, tools, model, runner)


# -- crash simulation ------------------------------------------------------
def records_of(store: RunStore, run_id: str) -> list[Record]:
    return load(store.log_path(run_id)).records


def seq_of(records: list[Record], kind: str, *, step: int | None = None, nth: int = 0) -> int:
    hits = [r for r in records if r.kind == kind and (step is None or r.step == step)]
    return hits[nth].seq


def crash_at(store: RunStore, run_id: str, keep_through_seq: int, *, tear: int = 0) -> None:
    """Truncate the log as an unclean process death would, and reopen the run.

    ``tear`` additionally removes bytes from the middle of the next record, so
    the log ends in a half-written line -- the failure mode an fsync cannot
    prevent, only detect.
    """
    path = store.log_path(run_id)
    records = load(path).records
    kept = [r for r in records if r.seq <= keep_through_seq]
    data = "".join(r.to_line() for r in kept).encode()
    if tear:
        nxt = next((r for r in records if r.seq == keep_through_seq + 1), None)
        if nxt is not None:
            partial = nxt.to_line().encode()
            data += partial[: max(1, len(partial) - tear)]
    path.write_bytes(data)
    meta = store.load_meta(run_id)
    meta.status = "running"
    meta.result = None
    store.save_meta(meta)


def notes(ws: Path) -> list[str]:
    p = ws / "notes.txt"
    return p.read_text().splitlines() if p.exists() else []
