"""Several agents in one store: ordering, recovery, and deterministic races."""

import threading

import pytest

from helpers import crash_at, records_of, seq_of
from ledger import AgentRunner, CallableModel, EffectScope, RunStore, ToolKind, ToolRegistry
from ledger.multi import (
    Coordinator,
    CoordKind,
    LamportClock,
    consistent_cut,
    merge_timeline,
    receive,
    register_coordination,
    send,
)
from ledger.wal import RecordKind, verify


class Chatter:
    """Sends to a peer on some steps, drains its inbox on others."""

    def __init__(self, agent_id: str, peer: str, steps: int = 4,
                 send_on: tuple[int, ...] = (1,), recv_on: tuple[int, ...] = (2, 3, 4)):
        self.agent_id = agent_id
        self.peer = peer
        self.steps = steps
        self.send_on = send_on
        self.recv_on = recv_on
        self.seen: list[str] = []
        self.sent: list[dict] = []
        self.after_seq = 0

    def step(self, ctx):
        ctx.sample(f"{self.agent_id} step {ctx.step}")
        if ctx.step in self.send_on:
            self.sent.append(send(ctx, self.peer, f"from-{self.agent_id}-s{ctx.step}"))
        if ctx.step in self.recv_on:
            for message in receive(ctx, after_seq=self.after_seq):
                self.seen.append(message["body"])
                self.after_seq = max(self.after_seq, message["seq"])
        if ctx.step >= self.steps:
            return ctx.done({"seen": self.seen, "sent": len(self.sent)})
        return None


class Claimer:
    """Races for one shared resource, then reports whether it won."""

    def __init__(self, agent_id: str, resource: str = "deploy-slot"):
        self.agent_id = agent_id
        self.resource = resource
        self.won: bool | None = None

    def step(self, ctx):
        ctx.sample(f"{self.agent_id} attempts {self.resource}")
        outcome = ctx.call("claim_resource", resource=self.resource)
        self.won = bool(outcome["won"])
        return ctx.done({"won": self.won, "holder": outcome["holder"]})


def build(root, store, coord, agent_id, factory, **kw):
    """One runner per agent: separate workspace, shared store and coordinator."""
    workspace = root / agent_id
    workspace.mkdir(parents=True, exist_ok=True)
    tools = ToolRegistry()
    tools.register("note_local", lambda text, workspace: len(text),
                   kind=ToolKind.PURE, scope=EffectScope.INTERNAL)
    register_coordination(tools, coord, agent_id, LamportClock())
    return AgentRunner(
        agent_factory=factory, store=store, workspace=workspace,
        model=CallableModel(lambda p: "ok"), tools=tools,
        snapshot_every=kw.pop("snapshot_every", 2), sync="never", **kw)


@pytest.fixture
def pair(tmp_path):
    store = RunStore(tmp_path / ".ledger")
    coord = Coordinator(tmp_path / ".ledger")
    return tmp_path, store, coord


# -- messaging -------------------------------------------------------------
def test_messages_cross_between_agents(pair):
    root, store, coord = pair
    alice = build(root, store, coord, "alice", lambda: Chatter("alice", "bob"))
    bob = build(root, store, coord, "bob", lambda: Chatter("bob", "alice"))

    a = alice.start()
    b = bob.start()

    assert a.ok and b.ok
    # Alice sent before Bob ran, so Bob saw it; Bob's reply lands after Alice
    # finished, which is the ordinary asynchrony of independent agents.
    assert b.result["seen"] == ["from-alice-s1"]
    assert len([r for r in coord.records() if r.kind == CoordKind.SEND]) == 2
    assert verify(coord.path).ok


def test_a_resumed_agent_replays_what_it_saw_not_what_is_there_now(pair):
    """The crux of multi-agent replay: a peer's later message must not appear.

    Agent B's inbox read is EXTERNAL, so replay serves the recorded answer. If it
    re-polled, a message the peer sent *after* the crash would appear at a step
    that historically never saw it -- and B's history would diverge from the one
    its own log describes.
    """
    root, store, coord = pair
    alice = build(root, store, coord, "alice", lambda: Chatter("alice", "bob"))
    bob = build(root, store, coord, "bob", lambda: Chatter("bob", "alice", steps=4))
    alice.start()
    original = bob.start()
    assert original.result["seen"] == ["from-alice-s1"]

    # Crash Bob back to step 2, then have a third party inject a new message.
    crash_at(store, original.run_id,
             seq_of(records_of(store, original.run_id), RecordKind.STEP_END, step=2))
    coord.send("carol", "bob", "sent-after-the-crash", "msg-late", lamport=99)

    resumed = bob.resume(original.run_id)

    assert resumed.ok
    # Steps 2-3 replay their recorded inboxes; only the live step 4 can see the
    # late message, and it does -- replay is faithful, not frozen.
    assert resumed.result["seen"] == ["from-alice-s1", "sent-after-the-crash"]
    assert verify(store.log_path(original.run_id)).ok


def test_resuming_one_agent_does_not_resend_its_messages(pair):
    root, store, coord = pair
    alice = build(root, store, coord, "alice",
                  lambda: Chatter("alice", "bob", steps=4, send_on=(1, 2)))
    original = alice.start()
    assert len([r for r in coord.records() if r.kind == CoordKind.SEND]) == 2

    crash_at(store, original.run_id,
             seq_of(records_of(store, original.run_id), RecordKind.STEP_END, step=3))
    resumed = alice.resume(original.run_id)

    assert resumed.ok
    # send_message is IRREVERSIBLE + EXTERNAL, so the replayed steps served their
    # recorded receipts instead of delivering again.
    assert len([r for r in coord.records() if r.kind == CoordKind.SEND]) == 2


def test_message_ids_come_from_recorded_entropy(pair):
    """Same id after a crash, so the idempotency key still identifies the message."""
    root, store, coord = pair
    alice = build(root, store, coord, "alice", lambda: Chatter("alice", "bob"))
    original = alice.start()
    ids_before = [r.payload["msg_id"] for r in coord.records()
                  if r.kind == CoordKind.SEND]

    crash_at(store, original.run_id,
             seq_of(records_of(store, original.run_id), RecordKind.STEP_END, step=3))
    alice.resume(original.run_id)

    assert [r.payload["msg_id"] for r in coord.records()
            if r.kind == CoordKind.SEND] == ids_before


# -- contention ------------------------------------------------------------
def test_exactly_one_agent_wins_a_concurrent_claim(pair):
    """The claim is a real check-and-set under a file lock, run from threads."""
    root, store, coord = pair
    outcomes: dict[str, bool] = {}

    def race(agent_id):
        runner = build(root, store, coord, agent_id, lambda: Claimer(agent_id))
        outcomes[agent_id] = runner.start().result["won"]

    threads = [threading.Thread(target=race, args=(name,))
               for name in ("alice", "bob", "carol")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(outcomes.values()) == 1, outcomes
    assert coord.holder("deploy-slot") in outcomes
    assert verify(coord.path).ok


def test_a_lost_race_replays_as_lost(pair):
    """Determinism through contention: the loser loses again on replay.

    The outcome was recorded, and the claim is EXTERNAL, so recovery does not
    re-run the race. Without that, a resumed loser could win the second time and
    two agents would believe they hold the same resource.
    """
    root, store, coord = pair
    winner = build(root, store, coord, "alice", lambda: Claimer("alice"))
    loser = build(root, store, coord, "bob", lambda: Claimer("bob", "deploy-slot"))
    assert winner.start().result["won"] is True
    original = loser.start()
    assert original.result["won"] is False

    # Free the resource, so a re-run of the race *would* now succeed.
    coord.release("alice", "deploy-slot", lamport=50)
    assert coord.holder("deploy-slot") is None

    crash_at(store, original.run_id, seq_of(
        records_of(store, original.run_id), RecordKind.TOOL_RESULT, step=1))
    # Truncating at the tool_result leaves the step unclosed, so the replay
    # reaches the recorded claim and must reuse its answer.
    resumed = loser.resume(original.run_id)
    assert resumed.result["won"] is False
    assert coord.holder("deploy-slot") is None


# -- the joint view --------------------------------------------------------
def test_merged_timeline_orders_a_send_before_its_receive(pair):
    root, store, coord = pair
    alice = build(root, store, coord, "alice", lambda: Chatter("alice", "bob"))
    bob = build(root, store, coord, "bob", lambda: Chatter("bob", "alice"))
    a = alice.start()
    b = bob.start()

    timeline = merge_timeline(store, [a.run_id, b.run_id])

    sends = [e for e in timeline if e.detail == "send_message"]
    receives = [e for e in timeline if e.detail == "receive_messages"]
    assert sends and receives
    # Alice's send carries a lower Lamport stamp than Bob's receive of it.
    alice_send = next(e for e in sends if e.agent == a.run_id)
    bob_receive = next(e for e in receives if e.agent == b.run_id and e.lamport > 0)
    assert alice_send.lamport <= bob_receive.lamport
    assert timeline == sorted(timeline, key=lambda e: (e.lamport, e.agent, e.seq))


def test_consistent_cut_retracts_a_receive_whose_send_is_excluded(pair):
    root, store, coord = pair
    coord.send("alice", "bob", "hello", "m1", lamport=7)

    # Cutting Bob after the message but Alice before it describes a history where
    # Bob holds a message nobody sent.
    cut = consistent_cut(coord, {"alice": 3, "bob": 9})

    assert not cut.consistent
    assert cut.points["bob"] == 6          # moved back before the send
    assert cut.points["alice"] == 3
    assert cut.retracted == {"bob": 9}


def test_a_cut_that_includes_both_ends_is_left_alone(pair):
    root, store, coord = pair
    coord.send("alice", "bob", "hello", "m1", lamport=7)

    cut = consistent_cut(coord, {"alice": 10, "bob": 10})

    assert cut.consistent
    assert cut.points == {"alice": 10, "bob": 10}


def test_cut_retraction_cascades(pair):
    """Retracting one agent can orphan a message it had itself sent."""
    root, store, coord = pair
    coord.send("alice", "bob", "first", "m1", lamport=5)
    coord.send("bob", "carol", "forwarded", "m2", lamport=6)

    cut = consistent_cut(coord, {"alice": 1, "bob": 9, "carol": 9})

    assert cut.points["bob"] == 4        # before alice's send
    assert cut.points["carol"] == 5      # and carol before bob's forward
