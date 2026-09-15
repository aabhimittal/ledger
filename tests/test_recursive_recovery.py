"""A log that contains its own recovery must still replay.

Long runs crash more than once, and each crash leaves half a step behind. If
those orphans stayed live, the second resume -- or any later fork -- would read
an interrupted attempt's ``prompt`` and then find the *next* attempt's records
where the matching ``sample`` should be. The ``abandon`` record retires them.
"""

from helpers import build, crash_at, notes, records_of, seq_of
from ledger import boundary_seq, diff_runs
from ledger.wal import RecordKind, abandoned_ranges, verify


def test_crash_twice_resume_twice(tmp_path):
    h = build(tmp_path, agent=lambda: __import__("helpers").NoteAgent(total_steps=9))
    out = h.runner.start()

    # First death mid-sample at step 4, second mid-sample at step 7.
    crash_at(h.store, out.run_id,
             seq_of(records_of(h.store, out.run_id), RecordKind.PROMPT, step=4))
    h.runner.resume(out.run_id)

    records = records_of(h.store, out.run_id)
    assert abandoned_ranges(records)
    crash_at(h.store, out.run_id, seq_of(records, RecordKind.PROMPT, step=7, nth=-1))
    resumed = h.runner.resume(out.run_id)

    assert resumed.ok
    assert resumed.result["lines"] == [f"note-{i}" for i in range(1, 10)]
    assert notes(h.ws) == [f"note-{i}" for i in range(1, 10)]
    assert len(h.sink.sent) == 1                      # still exactly one report
    assert verify(h.store.log_path(out.run_id)).ok
    assert len(abandoned_ranges(records_of(h.store, out.run_id))) == 2


def test_forking_a_recovered_run_skips_the_dead_records(tmp_path):
    """The case that motivated this: fork across the seam left by a resume."""
    h = build(tmp_path)
    out = h.runner.start()
    crash_at(h.store, out.run_id,
             seq_of(records_of(h.store, out.run_id), RecordKind.PROMPT, step=5))
    h.runner.resume(out.run_id)

    from helpers import CallableModel, step_number
    branch = h.runner.fork(out.run_id, after_step=5, workspace=tmp_path / "fork-ws",
                           model=CallableModel(lambda p: f"alt-{step_number(p)}"))

    assert branch.ok
    assert branch.replayed_steps == 5
    assert branch.result["lines"] == ["note-1", "note-2", "note-3", "note-4", "note-5", "alt-6"]
    assert notes(tmp_path / "fork-ws") == branch.result["lines"]


def test_the_abandoned_step_is_reopened_not_left_headless(tmp_path):
    h = build(tmp_path)
    out = h.runner.start()
    crash_at(h.store, out.run_id,
             seq_of(records_of(h.store, out.run_id), RecordKind.PROMPT, step=5))
    h.runner.resume(out.run_id)

    records = records_of(h.store, out.run_id)
    (lo, hi) = abandoned_ranges(records)[0]
    live_step_5 = [r for r in records
                   if r.step == 5 and not (lo < r.seq <= hi) and r.kind not in
                   (RecordKind.ABANDON, RecordKind.NOTE, RecordKind.SNAPSHOT)]
    # Exactly one well-formed step: start, prompt, sample, call, result, end.
    assert [r.kind for r in live_step_5] == [
        RecordKind.STEP_START, RecordKind.PROMPT, RecordKind.SAMPLE,
        RecordKind.TOOL_CALL, RecordKind.TOOL_RESULT, RecordKind.STEP_END,
    ]
    assert boundary_seq(records, 5) == live_step_5[-1].seq


def test_a_recovered_run_and_its_fork_diff_cleanly(tmp_path):
    h = build(tmp_path)
    out = h.runner.start()
    crash_at(h.store, out.run_id,
             seq_of(records_of(h.store, out.run_id), RecordKind.STEP_END, step=3))
    h.runner.resume(out.run_id)

    from helpers import CallableModel, step_number
    branch = h.runner.fork(out.run_id, after_step=4, workspace=tmp_path / "fork-ws",
                           model=CallableModel(lambda p: f"alt-{step_number(p)}"))

    diff = diff_runs(records_of(h.store, out.run_id), records_of(h.store, branch.run_id))
    assert diff.diverged_at_step == 5
    assert "alt-5" in diff.right
