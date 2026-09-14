"""Forking: same machinery as recovery, truncated deliberately instead of by a crash."""

import pytest

from helpers import CallableModel, NoteAgent, build, notes, records_of, step_number
from ledger import EffectPolicy, QuarantineError, boundary_seq, diff_runs
from ledger.wal import RecordKind, verify


def alt_model():
    return CallableModel(lambda p: f"alt-{step_number(p)}")


def test_fork_shares_a_hash_identical_prefix(tmp_path):
    h = build(tmp_path)
    parent = h.runner.start()
    boundary = boundary_seq(records_of(h.store, parent.run_id), 3)

    child = h.runner.fork(parent.run_id, after_step=3, workspace=tmp_path / "fork-ws",
                          model=alt_model())

    diff = diff_runs(records_of(h.store, parent.run_id), records_of(h.store, child.run_id))
    assert diff.shared_prefix_seq == boundary
    assert diff.diverged_at_step == 4
    assert verify(h.store.log_path(child.run_id)).ok


def test_fork_samples_fresh_after_the_boundary(tmp_path):
    h = build(tmp_path)
    parent = h.runner.start()

    child = h.runner.fork(parent.run_id, after_step=3, workspace=tmp_path / "fork-ws",
                          model=alt_model())

    assert child.ok
    assert child.replayed_steps == 3          # steps 1-3 replayed, 4-6 sampled fresh
    assert child.result["lines"] == ["note-1", "note-2", "note-3", "alt-4", "alt-5", "alt-6"]
    # The branch has its own environment; the parent's is untouched.
    assert notes(tmp_path / "fork-ws") == child.result["lines"]
    assert notes(h.ws) == [f"note-{i}" for i in range(1, 7)]
    assert h.store.lineage_ids(child.run_id) == [parent.run_id, child.run_id]
    assert child.forked_at_step == 3


def test_forks_quarantine_irreversible_effects_by_default(tmp_path):
    h = build(tmp_path, agent=lambda: NoteAgent(report_step=5))
    parent = h.runner.start()
    assert len(h.sink.sent) == 1

    with pytest.raises(QuarantineError, match="quarantined"):
        h.runner.fork(parent.run_id, after_step=3, workspace=tmp_path / "fork-ws")
    assert len(h.sink.sent) == 1              # nothing left the building


def test_quarantine_can_be_lifted_explicitly(tmp_path):
    h = build(tmp_path, agent=lambda: NoteAgent(report_step=5))
    parent = h.runner.start()

    child = h.runner.fork(parent.run_id, after_step=3, workspace=tmp_path / "fork-ws",
                          effects=EffectPolicy.ALLOW)
    assert child.ok
    assert len(h.sink.sent) == 2

    dry = h.runner.fork(parent.run_id, after_step=3, workspace=tmp_path / "dry-ws",
                        effects=EffectPolicy.DRY_RUN)
    assert dry.ok
    assert len(h.sink.sent) == 2              # logged, not performed


def test_fork_from_step_zero_replays_nothing(tmp_path):
    h = build(tmp_path)
    parent = h.runner.start()

    child = h.runner.fork(parent.run_id, after_step=0, workspace=tmp_path / "fork-ws",
                          model=alt_model(), effects=EffectPolicy.ALLOW)
    assert child.replayed_steps == 0
    assert child.result["lines"] == [f"alt-{i}" for i in range(1, 7)]


def test_fork_of_a_fork_keeps_the_whole_lineage(tmp_path):
    h = build(tmp_path)
    parent = h.runner.start()
    child = h.runner.fork(parent.run_id, after_step=4, workspace=tmp_path / "c1")
    grandchild = h.runner.fork(child.run_id, after_step=5, workspace=tmp_path / "c2",
                               model=alt_model())

    assert h.store.lineage_ids(grandchild.run_id) == [parent.run_id, child.run_id,
                                                      grandchild.run_id]
    assert grandchild.result["lines"][-1] == "alt-6"
    # The grandchild's prefix is still verifiable back to the root's genesis.
    assert verify(h.store.log_path(grandchild.run_id)).ok


def test_fork_boundary_must_exist(tmp_path):
    h = build(tmp_path)
    parent = h.runner.start()
    with pytest.raises(ValueError, match="no completed step 9"):
        h.runner.fork(parent.run_id, after_step=9)


def test_fork_records_provenance_in_the_child_log(tmp_path):
    h = build(tmp_path)
    parent = h.runner.start()
    child = h.runner.fork(parent.run_id, after_step=3, workspace=tmp_path / "fork-ws",
                          model=alt_model())

    fork_rec = next(r for r in records_of(h.store, child.run_id)
                    if r.kind == RecordKind.FORK)
    parent_records = records_of(h.store, parent.run_id)
    boundary = boundary_seq(parent_records, 3)
    assert fork_rec.payload["parent_run_id"] == parent.run_id
    assert fork_rec.payload["at_seq"] == boundary
    assert fork_rec.payload["parent_hash"] == parent_records[boundary - 1].h
