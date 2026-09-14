"""A crash *inside* a call: the case no amount of logging can settle.

The log records that the call was made and nothing about whether it landed.
Every automatic answer is a guess, so LEDGER refuses to guess by default.
"""

import pytest

from helpers import EffectAgent, build, crash_at, records_of, seq_of
from ledger import Undecidable, UndecidableEffect
from ledger.wal import RecordKind


def publish_run(tmp_path, **kw):
    h = build(tmp_path, agent=lambda: EffectAgent(tool="publish", at_step=3, total_steps=4), **kw)
    out = h.runner.start()
    assert len(h.sink.published) == 1
    # Amputate the log immediately after the publish call was recorded: the
    # effect reached the world, the result never reached the log.
    records = records_of(h.store, out.run_id)
    call_seq = next(r.seq for r in records
                    if r.kind == RecordKind.TOOL_CALL and r.payload["tool"] == "publish")
    crash_at(h.store, out.run_id, call_seq)
    return h, out


def test_default_policy_refuses_to_guess(tmp_path):
    h, out = publish_run(tmp_path)
    with pytest.raises(UndecidableEffect, match="not knowable"):
        h.runner.resume(out.run_id)
    assert len(h.sink.published) == 1     # and it did not publish again while failing


def test_reconcile_probe_finds_the_effect_already_applied(tmp_path):
    h, out = publish_run(tmp_path, undecidable=Undecidable.RECONCILE)

    resumed = h.runner.resume(out.run_id)

    assert resumed.ok
    assert resumed.result == "reconciled-publish"   # the probe's answer, not a re-run
    assert len(h.sink.published) == 1               # published exactly once, overall


def test_reconcile_probe_finds_it_never_landed_and_retries(tmp_path):
    h, out = publish_run(tmp_path, undecidable=Undecidable.RECONCILE)
    h.sink.published.clear()                        # the call never reached the service

    resumed = h.runner.resume(out.run_id)

    assert resumed.ok
    assert len(h.sink.published) == 1
    assert h.sink.published[0]["key"] == "doc-report"   # the *recorded* key, reused


def test_idempotent_tools_retry_with_the_recorded_key(tmp_path):
    h = build(tmp_path,
              agent=lambda: EffectAgent(tool="charge", args={"amount": 500}, at_step=3,
                                        total_steps=4))
    out = h.runner.start()
    records = records_of(h.store, out.run_id)
    call_seq = next(r.seq for r in records
                    if r.kind == RecordKind.TOOL_CALL and r.payload["tool"] == "charge")
    assert records[call_seq - 1].payload["idempotency_key"] == "chg-500"
    crash_at(h.store, out.run_id, call_seq)

    resumed = h.runner.resume(out.run_id)

    assert resumed.ok
    assert h.sink.charge_calls == 2                 # the call really was retried
    assert len(h.sink.charged) == 1                 # and the key let the peer dedupe it
    assert resumed.result == "duplicate-suppressed"


def test_pure_tools_just_rerun(tmp_path):
    h = build(tmp_path, agent=lambda: EffectAgent(tool="append_line",
                                                  args={"path": "extra.txt", "text": "x"},
                                                  at_step=3, total_steps=4))
    out = h.runner.start()
    records = records_of(h.store, out.run_id)
    call_seq = [r.seq for r in records
                if r.kind == RecordKind.TOOL_CALL and r.payload["args"].get("path") == "extra.txt"][0]
    crash_at(h.store, out.run_id, call_seq)

    assert h.runner.resume(out.run_id).ok


def test_assume_applied_needs_a_declared_result(tmp_path):
    h, out = publish_run(tmp_path, undecidable=Undecidable.ASSUME_APPLIED)
    with pytest.raises(UndecidableEffect, match="assumed_result"):
        h.runner.resume(out.run_id)


def test_the_decision_is_written_to_the_log(tmp_path):
    h, out = publish_run(tmp_path, undecidable=Undecidable.RECONCILE)
    h.runner.resume(out.run_id)

    notes = [r.payload for r in records_of(h.store, out.run_id)
             if r.kind == RecordKind.NOTE and r.payload.get("event") == "incomplete_effect"]
    assert notes and notes[0]["decision"] == "reconciled"
    assert notes[0]["applied"] is True
    assert notes[0]["tool"] == "publish"
