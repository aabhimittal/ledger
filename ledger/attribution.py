"""Failure attribution by resampling -- the reason forking is worth building.

A day-long run fails. Somewhere in 400 steps one sample went wrong. Reading the
log tells you *what* happened; it does not tell you which step *caused* it,
because everything after a bad step is downstream of it.

Forking answers that counterfactually. Restore the environment as it stood at
step k-1, let the agent sample step k again, and see whether the failure still
appears. Do it n times to get past sampling noise:

- failure reproduces in most trials  ->  step k was not the cause; the run was
  already doomed, look earlier.
- failure mostly disappears  ->  step k's sample is where it went wrong.

This is the standard causal-intervention move (do-calculus by brute force), and
it only works if you can re-run from an exact mid-run state -- which is what
snapshot-plus-log buys you and what a log alone does not.

Honest limits: each trial costs a real re-run of the tail, so a 400-step run is
expensive to bisect; irreversible effects are quarantined in trials, so a
failure whose cause *is* an outbound effect will not reproduce faithfully; and
a noisy judge turns this into a coin-flip counter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .effects import EffectPolicy
from .model import ModelClient
from .runner import AgentRunner, RunOutcome


@dataclass
class Trial:
    run_id: str
    status: str
    reproduced: bool
    result: object = None
    error: str | None = None


@dataclass
class ProbeResult:
    parent_run_id: str
    suspect_step: int
    trials: list[Trial] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.trials)

    @property
    def reproduced(self) -> int:
        return sum(1 for t in self.trials if t.reproduced)

    @property
    def reproduction_rate(self) -> float:
        return self.reproduced / self.n if self.n else 0.0

    @property
    def attribution(self) -> float:
        """How much of the failure this step's sample accounts for, in [0, 1].

        ``1.0`` means resampling the step always fixed the run; ``0.0`` means
        the failure was already inevitable before it.
        """
        return 1.0 - self.reproduction_rate

    def __str__(self) -> str:
        return (f"step {self.suspect_step}: failure reproduced in {self.reproduced}/{self.n} "
                f"resamples -> attribution {self.attribution:.2f}")


def resample_probe(
    runner: AgentRunner,
    run_id: str,
    suspect_step: int,
    *,
    judge: Callable[[RunOutcome], bool],
    trials: int = 5,
    workspace_root: str | Path | None = None,
    model: ModelClient | None = None,
    effects: EffectPolicy = EffectPolicy.BLOCK,
) -> ProbeResult:
    """Fork ``trials`` times from just before ``suspect_step`` and judge each branch.

    ``judge(outcome) -> True`` means "the failure reproduced". Each trial gets
    its own workspace so branches cannot contaminate each other or the original
    run's final state.
    """
    if suspect_step < 1:
        raise ValueError("suspect_step is 1-based")
    root = Path(workspace_root) if workspace_root else Path(runner.workspace).parent / "probes"
    probe = ProbeResult(parent_run_id=run_id, suspect_step=suspect_step)
    saved_on_error = runner.on_error
    runner.on_error = "return"
    try:
        for i in range(trials):
            ws = root / f"{run_id}-s{suspect_step}-t{i}"
            ws.mkdir(parents=True, exist_ok=True)
            outcome = runner.fork(run_id, suspect_step - 1, workspace=ws, model=model,
                                  effects=effects, tags={"probe": suspect_step, "trial": i})
            probe.trials.append(Trial(
                run_id=outcome.run_id, status=outcome.status, reproduced=bool(judge(outcome)),
                result=outcome.result, error=outcome.error,
            ))
    finally:
        runner.on_error = saved_on_error
    return probe


def scan_steps(
    runner: AgentRunner,
    run_id: str,
    steps: list[int],
    *,
    judge: Callable[[RunOutcome], bool],
    trials: int = 3,
    **kw,
) -> list[ProbeResult]:
    """Probe several candidate steps and rank them.

    Read the ranking carefully, because attribution here is not step-local.
    Forking at ``k`` resamples step ``k`` *and everything after it*, so any
    ``k`` at or before the real culprit will appear to fix the run. Attribution
    is therefore monotone: high for every early step, collapsing to zero once
    ``k`` passes the guilty step.

    The culprit is the **last** step whose resampling still clears the failure,
    so ties on attribution are broken towards the later step -- and a binary
    search over the step range finds it in log(n) forks rather than n.
    """
    results = [resample_probe(runner, run_id, s, judge=judge, trials=trials, **kw)
               for s in steps]
    return sorted(results, key=lambda r: (r.attribution, r.suspect_step), reverse=True)
