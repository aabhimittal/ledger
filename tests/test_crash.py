"""A real crash: a child process killed with ``os._exit`` mid-model-call.

Everything else in the suite truncates a log to *simulate* a crash. This test
lets the kernel do it, which is the only way to exercise the parts no in-process
test can reach: an unflushed buffer, a missing ``finally``, a possibly torn
final line, and a store that has to be picked up cold by a different process.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import demo_agent  # noqa: E402
from ledger import RunStore, verify  # noqa: E402

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo_agent.py"


@pytest.fixture
def crashed(tmp_path, monkeypatch):
    """Run the demo agent in a child process that dies at step 5."""
    monkeypatch.setenv(demo_agent.WORLD_ENV, str(tmp_path / "world.json"))
    proc = subprocess.run(
        [sys.executable, str(DEMO), "run", "--root", str(tmp_path), "--crash-at", "5"],
        env=dict(os.environ, **{demo_agent.WORLD_ENV: str(tmp_path / "world.json")}),
        capture_output=True, text=True,
    )
    assert proc.returncode == 9, proc.stderr
    store = RunStore(tmp_path / "store")
    runs = store.list_runs()
    assert len(runs) == 1
    return tmp_path, store, runs[0].run_id


def test_the_log_survives_an_unclean_death(crashed):
    _, store, run_id = crashed
    report = verify(store.log_path(run_id))
    assert report.ok
    assert report.records > 0
    assert store.load_meta(run_id).status == "running"   # no RUN_END was written
    # The snapshot committed before the crash is still usable.
    assert store.snapshots.list([run_id])


def test_a_fresh_process_can_resume_and_finish(crashed):
    tmp_path, store, run_id = crashed

    runner = demo_agent.make_runner(tmp_path)
    outcome = runner.resume(run_id)

    assert outcome.ok
    assert outcome.steps == 8
    assert outcome.replayed_steps >= 4
    assert outcome.model_calls < 8            # the replayed prefix cost no samples
    notes = (runner.workspace / "NOTES.md").read_text().splitlines()
    assert notes == [f"- fix item {i}" for i in range(1, 9)]   # no gaps, no duplicates
    assert verify(store.log_path(run_id)).ok


def test_the_outbound_effect_happens_exactly_once(crashed):
    tmp_path, store, run_id = crashed
    world = json.loads((tmp_path / "world.json").read_text()) if (
        tmp_path / "world.json").exists() else {"published": []}
    assert world["published"] == []            # step 8 was never reached

    demo_agent.make_runner(tmp_path).resume(run_id)

    published = json.loads((tmp_path / "world.json").read_text())["published"]
    assert len(published) == 1
    assert published[0]["key"] == "changelog-v0.8"


def test_resuming_twice_is_a_no_op_not_a_second_publication(crashed):
    tmp_path, store, run_id = crashed
    demo_agent.make_runner(tmp_path).resume(run_id)

    with pytest.raises(ValueError, match="fork it instead"):
        demo_agent.make_runner(tmp_path).resume(run_id)
    assert len(json.loads((tmp_path / "world.json").read_text())["published"]) == 1


def test_cli_reads_a_crashed_store(crashed):
    tmp_path, store, run_id = crashed
    out = subprocess.run(
        [sys.executable, "-m", "ledger", "--root", str(tmp_path / "store"), "show", run_id],
        capture_output=True, text=True, cwd=str(DEMO.parents[1]),
    )
    assert out.returncode == 0, out.stderr
    assert "unclosed (crash landed here)" in out.stdout
