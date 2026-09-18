#!/usr/bin/env python3
"""Two agents, one store: messaging, a contended resource, and recovery.

    python examples/multi_agent.py

Shows the four properties that make multi-agent runs recoverable:

1. A message crosses between agents and lands in the shared coordination log.
2. A resumed agent replays *what it saw*, not what the log holds now -- a peer's
   later message does not appear at a step that historically never saw it.
3. A contended claim has exactly one winner, and the loser replays as a loser
   even after the resource has been freed.
4. The joint timeline orders a send before its receive, across two logs whose
   sequence numbers are unrelated.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledger import (  # noqa: E402
    AgentRunner,
    CallableModel,
    EffectScope,
    RunStore,
    ToolKind,
    ToolRegistry,
)
from ledger.multi import (  # noqa: E402
    Coordinator,
    CoordKind,
    LamportClock,
    consistent_cut,
    merge_timeline,
    receive,
    send,
    register_coordination,
)
from ledger.wal import RecordKind, load, verify  # noqa: E402


class Planner:
    """Hands out three tasks, then reports what came back."""

    def __init__(self, steps: int = 5):
        self.steps = steps
        self.replies: list[str] = []
        self.after_seq = 0

    def step(self, ctx):
        ctx.sample(f"planner step {ctx.step}")
        if ctx.step <= 3:
            send(ctx, "builder", f"task-{ctx.step}")
        for message in receive(ctx, after_seq=self.after_seq):
            self.replies.append(str(message["body"]))
            self.after_seq = max(self.after_seq, message["seq"])
        if ctx.step >= self.steps:
            return ctx.done({"replies": self.replies})
        return None


class Builder:
    """Takes tasks, writes them into its own workspace, answers the planner."""

    def __init__(self, steps: int = 5):
        self.steps = steps
        self.done: list[str] = []
        self.after_seq = 0

    def step(self, ctx):
        ctx.sample(f"builder step {ctx.step}")
        for message in receive(ctx, after_seq=self.after_seq):
            task = str(message["body"])
            self.done.append(task)
            self.after_seq = max(self.after_seq, message["seq"])
            ctx.call("write_output", name=task)
            send(ctx, "planner", f"built-{task}")
        if ctx.step >= self.steps:
            return ctx.done({"built": self.done})
        return None


class Deployer:
    """Wants the one deploy slot."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    def step(self, ctx):
        ctx.sample(f"{self.agent_id} wants the slot")
        outcome = ctx.call("claim_resource", resource="deploy-slot")
        return ctx.done({"won": outcome["won"], "holder": outcome["holder"]})


def make_runner(root: Path, store: RunStore, coord: Coordinator, agent: str, factory):
    workspace = root / "workspaces" / agent
    workspace.mkdir(parents=True, exist_ok=True)
    tools = ToolRegistry()
    tools.register(
        "write_output",
        lambda name, workspace: (workspace / f"{name}.out").write_text(name),
        kind=ToolKind.PURE, scope=EffectScope.INTERNAL,
    )
    register_coordination(tools, coord, agent, LamportClock())
    return AgentRunner(agent_factory=factory, store=store, workspace=workspace,
                       model=CallableModel(lambda p: "ok"), tools=tools,
                       snapshot_every=2, agent_name=agent)


def crash_after_step(store: RunStore, run_id: str, step: int) -> None:
    """Truncate a log the way an unclean death would."""
    path = store.log_path(run_id)
    records = load(path).records
    cut = [r.seq for r in records if r.kind == RecordKind.STEP_END and r.step == step][-1]
    path.write_bytes(b"".join(r.to_line().encode() for r in records if r.seq <= cut))
    meta = store.load_meta(run_id)
    meta.status = "running"
    store.save_meta(meta)


def main() -> int:
    root = Path("./multi-demo")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    store = RunStore(root / "store")
    coord = Coordinator(root / "store")

    print("1. Planner hands out three tasks; builder does them and replies.")
    planner = make_runner(root, store, coord, "planner", Planner)
    builder = make_runner(root, store, coord, "builder", Builder)
    p1 = planner.start()
    b1 = builder.start()
    print(f"   builder built: {b1.result['built']}")
    # The builder answered after the planner had already finished, which is the
    # ordinary asynchrony of independent agents: the replies wait in its inbox.
    print(f"   replies waiting for the planner: "
          f"{[m['body'] for m in coord.inbox('planner')]}")
    sends = [r for r in coord.records() if r.kind == CoordKind.SEND]
    print(f"   coordination log: {len(sends)} messages, chain {verify(coord.path).ok}")

    print("\n2. Crash the builder back to step 1, then inject a message it never saw.")
    crash_after_step(store, b1.run_id, 1)
    coord.send("ops", "builder", "urgent-hotfix", "msg-ops", lamport=99)
    resumed = builder.resume(b1.run_id)
    print(f"   resumed: {resumed.status}, {resumed.replayed_steps} steps replayed")
    print(f"   built after recovery: {resumed.result['built']}")
    print("   the injected message appears only at a live step, never inside the replay")
    delivered = len([r for r in coord.records() if r.kind == CoordKind.SEND])
    print(f"   messages in the log: {delivered} -- replayed sends were not re-delivered")

    print("\n3. Two agents race for one deploy slot.")
    first = make_runner(root, store, coord, "deploy-a", lambda: Deployer("deploy-a"))
    second = make_runner(root, store, coord, "deploy-b", lambda: Deployer("deploy-b"))
    a = first.start()
    b = second.start()
    print(f"   deploy-a won={a.result['won']}   deploy-b won={b.result['won']}"
          f"   holder={coord.holder('deploy-slot')}")
    coord.release("deploy-a", "deploy-slot", lamport=200)
    print(f"   slot released; holder is now {coord.holder('deploy-slot')}")
    print("   (a resumed loser still loses: the outcome was recorded, see tests)")

    print("\n4. One causal timeline across two unrelated logs "
          "(coordination events only).")
    joint = [e for e in merge_timeline(store, [p1.run_id, b1.run_id])
             if e.detail in ("send_message", "receive_messages")
             and e.kind == RecordKind.TOOL_RESULT]
    # The planner's sends carry lower stamps than the builder's receives of them:
    # the builder observed those stamps and ticked past them.
    for event in joint[:4] + joint[-4:]:
        print(f"   {event}")

    print("\n5. A fork point that would invent history gets retracted.")
    cut = consistent_cut(coord, {"planner": 1, "builder": 9})
    print(f"   wanted builder@9, planner@1 -> {cut.points}, consistent={cut.consistent}")
    print(f"\nStore: {root}. Try: python -m ledger --root {root / 'store'} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
