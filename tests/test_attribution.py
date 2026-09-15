"""Failure attribution by resampling a suspect step -- what forking is for."""

from helpers import build
from ledger import CallableModel, StepContext, StepResult, resample_probe, scan_steps


class PoisonSensitiveAgent:
    """Fails at the end if any step ever sampled the word ``bad``."""

    def __init__(self, total_steps: int = 5):
        self.total_steps = total_steps
        self.lines: list[str] = []

    def step(self, ctx: StepContext) -> StepResult | None:
        self.lines.append(ctx.sample(f"step {ctx.step}").text)
        ctx.call("append_line", path="notes.txt", text=self.lines[-1])
        if ctx.step >= self.total_steps:
            if any("bad" in line for line in self.lines):
                raise RuntimeError(f"poisoned by {self.lines}")
            return ctx.done("ok")
        return None


def step_of(prompt: str) -> int:
    return int(prompt.split()[1])


def poison_at(step: int) -> CallableModel:
    return CallableModel(lambda p: "bad" if step_of(p) == step else "good")


def always_good() -> CallableModel:
    return CallableModel(lambda p: "good")


def failed(outcome) -> bool:
    return outcome.status == "failed"


def test_resampling_the_guilty_step_clears_the_failure(tmp_path):
    h = build(tmp_path, agent=PoisonSensitiveAgent, model=poison_at(3), on_error="return")
    parent = h.runner.start()
    assert parent.status == "failed"

    probe = resample_probe(h.runner, parent.run_id, suspect_step=3, judge=failed,
                           trials=3, model=always_good(), workspace_root=tmp_path / "probes")

    assert probe.n == 3
    assert probe.reproduced == 0
    assert probe.attribution == 1.0
    assert "attribution 1.00" in str(probe)


def test_resampling_an_innocent_step_leaves_the_failure_in_place(tmp_path):
    """Step 4 is downstream of the real cause, so re-rolling it changes nothing."""
    h = build(tmp_path, agent=PoisonSensitiveAgent, model=poison_at(3), on_error="return")
    parent = h.runner.start()

    probe = resample_probe(h.runner, parent.run_id, suspect_step=4, judge=failed,
                           trials=2, model=always_good(), workspace_root=tmp_path / "probes")

    assert probe.reproduced == 2         # the poison is already in the replayed prefix
    assert probe.attribution == 0.0


def test_scan_ranks_candidate_steps(tmp_path):
    h = build(tmp_path, agent=PoisonSensitiveAgent, model=poison_at(2), on_error="return")
    parent = h.runner.start()

    ranked = scan_steps(h.runner, parent.run_id, [1, 2, 3], judge=failed, trials=1,
                        model=always_good(), workspace_root=tmp_path / "probes")

    # Attribution is monotone, not step-local: forking at step 1 resamples step 2
    # as well, so it "fixes" the run too. The culprit is the *latest* step whose
    # resampling still clears the failure, which is why ties break towards later.
    assert {r.suspect_step: r.attribution for r in ranked} == {1: 1.0, 2: 1.0, 3: 0.0}
    assert ranked[0].suspect_step == 2


def test_probes_do_not_disturb_the_parent_run(tmp_path):
    h = build(tmp_path, agent=PoisonSensitiveAgent, model=poison_at(3), on_error="return")
    parent = h.runner.start()
    before = h.store.load_meta(parent.run_id).to_dict()

    resample_probe(h.runner, parent.run_id, suspect_step=3, judge=failed, trials=2,
                   model=always_good(), workspace_root=tmp_path / "probes")

    assert h.store.load_meta(parent.run_id).to_dict() == before
    assert h.runner.on_error == "return"   # the temporary override was restored
    assert len(h.store.children(parent.run_id)) == 2
