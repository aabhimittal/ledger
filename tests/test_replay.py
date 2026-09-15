"""Crash recovery: restore the snapshot, replay the log, never re-sample."""

import pytest

from helpers import NoteAgent, build, crash_at, notes, records_of, seq_of
from ledger import Divergence, DivergenceError, StepResult
from ledger.wal import RecordKind, verify


def test_live_run_is_logged_and_verifiable(tmp_path):
    h = build(tmp_path)
    out = h.runner.start()

    assert out.ok and out.steps == 6
    assert out.result["lines"] == [f"note-{i}" for i in range(1, 7)]
    assert verify(h.store.log_path(out.run_id)).ok
    assert notes(h.ws) == [f"note-{i}" for i in range(1, 7)]
    assert len(h.sink.sent) == 1
    # baseline at step 0, cadence at 2/4/6
    assert {r.step for r in h.store.snapshots.list([out.run_id])} == {0, 2, 4, 6}


def test_resume_replays_without_resampling_or_refiring_effects(tmp_path):
    h = build(tmp_path)
    out = h.runner.start()
    records = records_of(h.store, out.run_id)

    # Die just after step 3 committed -- the step that mailed the report.
    crash_at(h.store, out.run_id, seq_of(records, RecordKind.STEP_END, step=3), tear=20)
    samples_before = h.model.n_calls
    internal_before = h.sink.internal_calls

    resumed = h.runner.resume(out.run_id)

    assert resumed.ok
    assert resumed.result == out.result
    assert resumed.repaired_bytes > 0          # the torn tail was detected and dropped
    # Steps 1-3 are replayed to rebuild the agent's own state; only 4-6 sample.
    assert resumed.replayed_steps == 3
    assert resumed.replayed_samples == 3
    assert h.model.n_calls - samples_before == 3
    # The INTERNAL tool re-ran only for step 3 -- steps 1-2 are already inside
    # the restored snapshot, so re-running them would have doubled the file.
    assert h.sink.internal_calls - internal_before == 4
    assert len(h.sink.sent) == 1               # the EXTERNAL effect did not re-fire
    assert notes(h.ws) == [f"note-{i}" for i in range(1, 7)]
    assert verify(h.store.log_path(out.run_id)).ok


def test_resume_mid_step(tmp_path):
    """A crash between a tool call and the end of its step still recovers."""
    h = build(tmp_path)
    out = h.runner.start()
    records = records_of(h.store, out.run_id)
    crash_at(h.store, out.run_id, seq_of(records, RecordKind.SAMPLE, step=5))

    resumed = h.runner.resume(out.run_id)
    assert resumed.ok
    assert resumed.result == out.result
    assert notes(h.ws) == [f"note-{i}" for i in range(1, 7)]


def test_crash_inside_the_model_call_resamples_that_step(tmp_path):
    """A prompt with no sample: the step never committed, so re-sampling is correct."""
    h = build(tmp_path)
    out = h.runner.start()
    records = records_of(h.store, out.run_id)
    crash_at(h.store, out.run_id, seq_of(records, RecordKind.PROMPT, step=5))

    samples_before = h.model.n_calls
    resumed = h.runner.resume(out.run_id)

    assert resumed.ok
    assert resumed.replayed_samples == 4               # steps 1-4 came from the log
    assert h.model.n_calls - samples_before == 2       # step 5 re-sampled, 6 fresh
    assert notes(h.ws) == [f"note-{i}" for i in range(1, 7)]
    reasons = [r.payload.get("reason") for r in records_of(h.store, out.run_id)
               if r.payload.get("event") == "replay_end"]
    assert reasons == ["incomplete_sample"]


def test_internal_effects_are_rebuilt_without_an_intervening_snapshot(tmp_path):
    """The crux: with only a step-0 baseline, replay must rebuild the workspace.

    The log holds no filesystem state. If replay skipped INTERNAL tools, the
    restored workspace would be four steps stale and the run would append lines
    5 and 6 onto an empty file.
    """
    h = build(tmp_path, snapshot_every=0)
    out = h.runner.start()
    # Only the step-0 baseline and the terminal snapshot; nothing in between.
    assert {r.step for r in h.store.snapshots.list([out.run_id])} == {0, 6}

    records = records_of(h.store, out.run_id)
    crash_at(h.store, out.run_id, seq_of(records, RecordKind.STEP_END, step=4))
    resumed = h.runner.resume(out.run_id)

    assert resumed.replayed_steps == 4
    assert resumed.replayed_samples == 4
    assert notes(h.ws) == [f"note-{i}" for i in range(1, 7)]   # no gaps, no duplicates


def test_entropy_is_replayed_not_redrawn(tmp_path):
    h = build(tmp_path, agent=lambda: NoteAgent(use_entropy=True))
    out = h.runner.start()
    records = records_of(h.store, out.run_id)
    crash_at(h.store, out.run_id, seq_of(records, RecordKind.STEP_END, step=4))

    resumed = h.runner.resume(out.run_id)
    # The replayed prefix gets the recorded clock readings back, not new ones.
    assert resumed.result["stamps"][:4] == out.result["stamps"][:4]
    assert resumed.result["stamps"][4:] != out.result["stamps"][4:]


class ChangedPromptAgent(NoteAgent):
    def step(self, ctx):
        prompt = f"STEP {ctx.step} (reworded); so far: {self.lines}"
        text = ctx.sample(prompt).text
        self.lines.append(text)
        ctx.call("append_line", path="notes.txt", text=text)
        if ctx.step == self.report_step:
            ctx.call("send_report", to="ops@example.com", body=text)
        if ctx.step >= self.total_steps:
            return ctx.done({"lines": self.lines, "stamps": []})
        return None


def test_changed_agent_code_is_caught_not_papered_over(tmp_path):
    h = build(tmp_path)
    out = h.runner.start()
    crash_at(h.store, out.run_id,
             seq_of(records_of(h.store, out.run_id), RecordKind.STEP_END, step=4))

    h.runner.agent_factory = ChangedPromptAgent
    with pytest.raises(DivergenceError, match="prompt"):
        h.runner.resume(out.run_id)


def test_resample_policy_turns_divergence_into_a_live_branch(tmp_path):
    h = build(tmp_path, divergence=Divergence.RESAMPLE)
    out = h.runner.start()
    crash_at(h.store, out.run_id,
             seq_of(records_of(h.store, out.run_id), RecordKind.STEP_END, step=4))

    h.runner.agent_factory = ChangedPromptAgent
    resumed = h.runner.resume(out.run_id)

    assert resumed.ok
    assert resumed.divergences == 1
    # Honest consequence: going live at step 1 replays nothing, so the report
    # is mailed a second time. Changing agent code between crash and resume is
    # not free.
    assert len(h.sink.sent) == 2


def test_resume_refuses_a_completed_run(tmp_path):
    h = build(tmp_path)
    out = h.runner.start()
    with pytest.raises(ValueError, match="fork it instead"):
        h.runner.resume(out.run_id)


def test_max_steps_is_enforced(tmp_path):
    h = build(tmp_path, agent=lambda: NoteAgent(total_steps=100), max_steps=3)
    out = h.runner.start()
    assert out.status == "max_steps" and out.steps == 3


def test_failure_snapshots_the_wreckage(tmp_path):
    class Exploding(NoteAgent):
        def step(self, ctx):
            if ctx.step == 3:
                raise RuntimeError("boom")
            return super().step(ctx)

    h = build(tmp_path, agent=Exploding, on_error="return")
    out = h.runner.start()
    assert out.status == "failed" and "boom" in out.error
    # Step 3's state is captured for later attribution work.
    assert 3 in {r.step for r in h.store.snapshots.list([out.run_id])}
