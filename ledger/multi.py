"""Several agents, one environment, one store.

``docs/DESIGN.md`` called this "a project, not an extension", and the estimate
was half wrong. The replay machinery needs **no** changes, because of an
observation that does all the work:

    another agent is part of the outside world.

An agent that reads what a peer wrote, or takes a message from it, has observed
something outside its own snapshot -- which is exactly what ``EXTERNAL`` scope
already means. Record what was observed, never re-execute the observation, and
each agent stays independently replayable with the rules that already exist. A
crash in agent A recovers A without touching B.

What genuinely was missing, and is here:

**Causal order across logs.** Per-agent sequence numbers cannot say whether A's
step 7 preceded B's step 3. Lamport clocks stamped on every cross-agent event
can, and ``merge_timeline`` uses them to build one joint history.

**Coordination that replays.** ``send_message``, ``receive_messages`` and
``claim_resource`` are ordinary tool registrations, declared so that replay
serves their recorded results. A *race* therefore replays with the same winner --
determinism through a contended lock, which is the part people assume is
impossible.

**Consistent cuts.** Forking one agent of an ensemble leaves its peers on the
unforked timeline, so a naive per-agent fork point can be causally impossible: an
agent that has received a message its peer never sent. ``consistent_cut``
retracts cut points until that cannot happen (the Chandy-Lamport condition),
which is what makes forking the whole ensemble meaningful.

One constraint this imposes, and it is not optional: **each agent needs its own
snapshotted workspace.** Restoring a snapshot rolls the whole directory back, so
two agents sharing one would find that recovering either undoes the other's work.
Shared mutable state must therefore live outside every agent's snapshot and be
reached through ``EXTERNAL`` tools -- a claimed resource, a message, a database.
The constraint falls out of what a snapshot is, so no amount of coordination
machinery removes it.

Deliberately not solved: one machine only. Cross-machine agents need agreement on
the coordination log, and that is consensus -- a different project, and this time
the estimate stands. The coordination log is also for coordination events, not
bulk data: every append re-validates the chain under a lock, which is fine for
the hundreds of events a run produces and wrong for millions.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from .effects import EffectScope, ToolKind, ToolRegistry
from .runner import StepContext
from .store import RunStore
from .wal import Record, RecordKind, WriteAheadLog, load

try:  # pragma: no cover - platform dependent
    import fcntl

    HAVE_FLOCK = True
except ImportError:  # pragma: no cover
    HAVE_FLOCK = False


# --------------------------------------------------------------------------
# logical time
# --------------------------------------------------------------------------
@dataclass
class LamportClock:
    """Per-agent logical clock.

    Never persisted, and it does not need to be: its value is a pure function of
    the coordination events the agent has seen, and replay serves those from the
    agent's own log. So a resumed agent rebuilds the same clock for free -- the
    same reason its message history needs no checkpoint schema.
    """

    value: int = 0

    def tick(self) -> int:
        self.value += 1
        return self.value

    def observe(self, other: int) -> int:
        self.value = max(self.value, int(other))
        return self.value

    def receive(self, stamps: Iterable[int]) -> int:
        for stamp in stamps:
            self.observe(stamp)
        return self.tick()


class CoordKind:
    SEND = "send"
    CLAIM = "claim"
    RELEASE = "release"


# --------------------------------------------------------------------------
# the shared coordination log
# --------------------------------------------------------------------------
class Coordinator:
    """An append-only log of cross-agent events, shared by every agent in a store.

    Appends take an exclusive file lock and reopen the log, so separate processes
    can share it without corrupting the hash chain. That costs a full chain
    verification per append; coordination events are rare enough for it to be the
    right trade, and the docstring above says when it is not.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.dir = self.root / "coord"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "log.jsonl"
        self.lock_path = self.dir / "log.lock"
        self.path.touch(exist_ok=True)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        if not HAVE_FLOCK:  # pragma: no cover - single-process fallback
            yield
            return
        with open(self.lock_path, "w") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _append(self, kind: str, payload: dict) -> Record:
        with self._locked():
            log = WriteAheadLog(self.path, sync="always")
            try:
                return log.append(kind, payload)
            finally:
                log.close()

    def records(self) -> list[Record]:
        return load(self.path).records

    # -- messaging -----------------------------------------------------
    def send(self, sender: str, to: str, body: Any, msg_id: str, lamport: int) -> dict:
        rec = self._append(CoordKind.SEND, {
            "msg_id": msg_id, "from": sender, "to": to, "body": body,
            "lamport": lamport,
        })
        return {"msg_id": msg_id, "seq": rec.seq, "lamport": lamport}

    def inbox(self, agent: str, after_seq: int = 0) -> list[dict]:
        return [
            {"msg_id": r.payload["msg_id"], "from": r.payload["from"],
             "body": r.payload["body"], "lamport": r.payload["lamport"], "seq": r.seq}
            for r in self.records()
            if r.kind == CoordKind.SEND and r.payload.get("to") == agent
            and r.seq > after_seq
        ]

    def already_sent(self, msg_id: str) -> bool:
        """Read-back for the undecidable case: did this send actually land?"""
        return any(r.kind == CoordKind.SEND and r.payload.get("msg_id") == msg_id
                   for r in self.records())

    # -- mutual exclusion ---------------------------------------------
    def holder(self, resource: str) -> str | None:
        owner: str | None = None
        for rec in self.records():
            if rec.payload.get("resource") != resource:
                continue
            if rec.kind == CoordKind.CLAIM and owner is None:
                owner = rec.payload.get("agent")
            elif rec.kind == CoordKind.RELEASE and owner == rec.payload.get("agent"):
                owner = None
        return owner

    def claim(self, agent: str, resource: str, lamport: int) -> dict:
        """Take ``resource`` if free. The whole check-and-set is under the lock."""
        with self._locked():
            owner = self.holder(resource)
            if owner is not None and owner != agent:
                return {"won": False, "holder": owner, "lamport": lamport}
            log = WriteAheadLog(self.path, sync="always")
            try:
                rec = log.append(CoordKind.CLAIM, {
                    "agent": agent, "resource": resource, "lamport": lamport})
            finally:
                log.close()
        return {"won": True, "holder": agent, "seq": rec.seq, "lamport": lamport}

    def release(self, agent: str, resource: str, lamport: int) -> dict:
        rec = self._append(CoordKind.RELEASE, {
            "agent": agent, "resource": resource, "lamport": lamport})
        return {"released": True, "seq": rec.seq, "lamport": lamport}


# --------------------------------------------------------------------------
# agent-facing tools
# --------------------------------------------------------------------------
def register_coordination(tools: ToolRegistry, coord: Coordinator, agent: str,
                          clock: LamportClock) -> ToolRegistry:
    """Register send/receive/claim/release for one agent.

    The declarations are where the design lives, so they are worth reading:

    ``send_message``  IRREVERSIBLE + EXTERNAL. Delivering a message twice is a
        real-world effect, and replay must never redo it. ``msg_id`` makes it
        idempotent on retry and gives ``reconcile`` something to look up, which
        is what turns "the crash landed inside the send" from undecidable into a
        read-back against the coordination log.
    ``receive_messages``  PURE + EXTERNAL. Nothing is mutated, so it is PURE --
        but the *answer* depends on what peers have done, so it must be served
        from the log rather than re-polled. This is the case that shows why
        reversibility and scope have to be separate axes.
    ``claim_resource``  IDEMPOTENT + EXTERNAL. Re-claiming with the same key is
        harmless; the recorded outcome means a contended race replays with the
        same winner.
    """
    def send_message(to: str, body: Any, msg_id: str) -> dict:
        return coord.send(agent, to, body, msg_id, clock.tick())

    def receive_messages(after_seq: int = 0) -> list[dict]:
        messages = coord.inbox(agent, after_seq)
        clock.receive(m["lamport"] for m in messages)
        return messages

    def claim_resource(resource: str) -> dict:
        return coord.claim(agent, resource, clock.tick())

    def release_resource(resource: str) -> dict:
        return coord.release(agent, resource, clock.tick())

    tools.register("send_message", send_message, kind=ToolKind.IRREVERSIBLE,
                   scope=EffectScope.EXTERNAL,
                   idempotency_key=lambda args: f"msg-{args['msg_id']}",
                   reconcile=lambda key: (
                       coord.already_sent(str(key).removeprefix("msg-")),
                       {"msg_id": str(key).removeprefix("msg-"), "reconciled": True},
                   ))
    tools.register("receive_messages", receive_messages, kind=ToolKind.PURE,
                   scope=EffectScope.EXTERNAL)
    tools.register("claim_resource", claim_resource, kind=ToolKind.IDEMPOTENT,
                   scope=EffectScope.EXTERNAL,
                   idempotency_key=lambda args: f"{agent}:{args['resource']}",
                   reconcile=lambda key: (False, None))
    tools.register("release_resource", release_resource, kind=ToolKind.IDEMPOTENT,
                   scope=EffectScope.EXTERNAL,
                   idempotency_key=lambda args: f"rel-{agent}:{args['resource']}",
                   reconcile=lambda key: (False, None))
    return tools


def send(ctx: StepContext, to: str, body: Any) -> dict:
    """Send a message, minting the id from recorded entropy.

    Using ``ctx.entropy`` rather than ``uuid4()`` directly is the point: replay
    reproduces the same id, so the idempotency key and the reconcile lookup still
    refer to the same message after a crash.
    """
    return ctx.call("send_message", to=to, body=body, msg_id=ctx.entropy.uuid4())


def receive(ctx: StepContext, after_seq: int = 0) -> list[dict]:
    return ctx.call("receive_messages", after_seq=after_seq)


# --------------------------------------------------------------------------
# the joint view
# --------------------------------------------------------------------------
@dataclass
class JointEvent:
    lamport: int
    agent: str
    """The run id -- unambiguous, since two agents can share a class name."""
    seq: int
    kind: str
    detail: str = ""
    label: str = ""
    """The agent's human name, for display only."""

    def __str__(self) -> str:
        who = self.label or self.agent
        return f"L{self.lamport:<4} {who:<10} seq {self.seq:<5} {self.kind:<14} {self.detail}"


def _lamport_of(payload: dict) -> int | None:
    for key in ("lamport",):
        if key in payload:
            return int(payload[key])
    result = payload.get("result")
    if isinstance(result, dict) and "lamport" in result:
        return int(result["lamport"])
    if isinstance(result, list) and result and isinstance(result[0], dict):
        stamps = [int(m["lamport"]) for m in result if isinstance(m, dict) and "lamport" in m]
        if stamps:
            return max(stamps)
    return None


def merge_timeline(store: RunStore, run_ids: Iterable[str]) -> list[JointEvent]:
    """One causally ordered history across several agents' logs.

    Records that carry no stamp of their own inherit the agent's clock as it
    stood, which is the best any logical clock can offer: events between two
    coordination points are genuinely unordered relative to other agents, and
    pretending otherwise would invent precision that is not there.
    """
    events: list[JointEvent] = []
    for run_id in run_ids:
        try:
            label = store.load_meta(run_id).agent
        except KeyError:  # pragma: no cover - a log without metadata
            label = run_id
        records = load(store.log_path(run_id)).records

        # A tool_result names only its call_id, and a coordination event's stamp
        # arrives with the result rather than the call. Resolve both up front so
        # every event carries the tool that produced it and the clock it belongs
        # to -- otherwise the joint view labels half its rows "c37".
        tool_of: dict[str, str] = {}
        stamp_of: dict[str, int] = {}
        for rec in records:
            call_id = str(rec.payload.get("call_id") or "")
            if rec.kind == RecordKind.TOOL_CALL and call_id:
                tool_of[call_id] = str(rec.payload.get("tool") or "")
            elif rec.kind == RecordKind.TOOL_RESULT and call_id:
                stamp = _lamport_of(rec.payload)
                if stamp is not None:
                    stamp_of[call_id] = stamp

        clock = 0
        for rec in records:
            call_id = str(rec.payload.get("call_id") or "")
            stamp = stamp_of.get(call_id, _lamport_of(rec.payload))
            if stamp is not None:
                clock = max(clock, stamp)
            if rec.kind in (RecordKind.TOOL_CALL, RecordKind.TOOL_RESULT,
                            RecordKind.SAMPLE, RecordKind.STEP_END):
                detail = str(rec.payload.get("tool") or tool_of.get(call_id) or "")
                events.append(JointEvent(clock, run_id, rec.seq, rec.kind, detail, label))
    return sorted(events, key=lambda e: (e.lamport, e.agent, e.seq))


@dataclass
class Cut:
    """A per-agent set of cut points, and whether it is causally possible."""

    points: dict[str, int] = field(default_factory=dict)
    retracted: dict[str, int] = field(default_factory=dict)

    @property
    def consistent(self) -> bool:
        return not self.retracted


def consistent_cut(coord: Coordinator, wanted: dict[str, int]) -> Cut:
    """Retract cut points until no message is received before it was sent.

    ``wanted`` maps agent to the Lamport time it should be cut at. A cut that
    keeps a receive while dropping its send describes a history that never
    happened, and a fork taken there would replay a message out of thin air. The
    fix is to move the receiving agent's cut back before that receive.
    """
    cut = Cut(points=dict(wanted))
    sends = [r for r in coord.records() if r.kind == CoordKind.SEND]
    changed = True
    while changed:
        changed = False
        for rec in sends:
            sender, receiver = rec.payload["from"], rec.payload["to"]
            stamp = int(rec.payload["lamport"])
            if sender not in cut.points or receiver not in cut.points:
                continue
            # The receive can only be inside the cut if the send is too.
            if cut.points[receiver] >= stamp > cut.points[sender]:
                cut.retracted[receiver] = cut.points[receiver]
                cut.points[receiver] = stamp - 1
                changed = True
    return cut
