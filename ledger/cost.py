"""What a day-long run actually costs, and how to pick ``k``.

Two questions were unanswered when the machinery was built, and both are
arithmetic once something measures the constants.

**How big does the log get?** ``LogProfile`` measures bytes per record kind on a
real log and projects forward. This is what decides whether storing full prompt
text is affordable: prompts dominate an agent log, so the ``store_prompts``
default is the single biggest lever on log size.

**How often should the environment be snapshotted?** Recovery has two costs, and
only one of them depends on ``k`` -- which is easy to get wrong, and the
benchmark caught it:

*In-memory replay* re-runs the agent's code for every step from 1 to the crash,
because that is the only thing that rebuilds its history. Its cost is
``crash_step * C_m`` and **``k`` does not change it at all**. Snapshotting every
single step would not save a millisecond of it. That is the price LEDGER pays
for needing no checkpoint schema, and it is a floor, not a knob.

*Effect re-execution* re-runs the internal tools recorded above the snapshot's
``at_seq``, since those effects are the delta the snapshot is missing. Only this
part is proportional to how stale the snapshot is:

    overhead(k) = (n / k) * C_s  +  f * (k / 2) * C_e

with ``n`` steps, ``C_s`` seconds per snapshot, ``C_e`` seconds to re-execute one
step's internal effects, and ``f`` expected crashes. The ``k/2`` is the expected
distance back to the last snapshot when a crash lands uniformly in the window.
Differentiating gives

    k* = sqrt(2 * n * C_s / (f * C_e))

Using the *whole* per-step replay time in place of ``C_e`` is the tempting
mistake, and it errs in **both** directions depending on which cost dominates the
resume. When internal effects are cheap (a file append) the total is mostly the
in-memory floor, so the naive figure overstates ``C_e`` and buys snapshots that
save nothing. When internal effects are expensive (a rebuild) the total is spread
over every replayed step, so it *understates* ``C_e`` and snapshots too rarely.
Measured on one machine, ``C_e`` spans 0.02 ms to 27 ms across those two
workloads -- three orders of magnitude -- which is why it has to be measured
rather than reasoned about.

``k*`` grows as sqrt(n): a run ten times longer wants a window only about three
times wider. And ``C_s`` and ``C_e`` are properties of a workload and a machine,
not constants anyone can ship. ``python -m ledger.bench`` measures them for
yours; the numbers in ``docs/DESIGN.md`` are one calibration, not a default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .wal import Record, RecordKind, load


@dataclass
class KindStat:
    count: int = 0
    bytes: int = 0

    @property
    def mean_bytes(self) -> float:
        return self.bytes / self.count if self.count else 0.0


@dataclass
class LogProfile:
    """Measured size of a log, broken down by record kind."""

    steps: int
    records: int
    bytes: int
    by_kind: dict[str, KindStat] = field(default_factory=dict)

    @property
    def bytes_per_step(self) -> float:
        return self.bytes / self.steps if self.steps else 0.0

    def project(self, steps: int) -> int:
        """Extrapolate to a longer run of the same shape."""
        return int(self.bytes_per_step * steps)

    @classmethod
    def from_records(cls, records: Iterable[Record]) -> "LogProfile":
        by_kind: dict[str, KindStat] = {}
        total = 0
        steps = 0
        for rec in records:
            size = len(rec.to_line().encode())
            total += size
            stat = by_kind.setdefault(rec.kind, KindStat())
            stat.count += 1
            stat.bytes += size
            if rec.kind == RecordKind.STEP_END:
                steps += 1
        return cls(steps=steps, records=sum(s.count for s in by_kind.values()),
                   bytes=total, by_kind=by_kind)

    @classmethod
    def from_log(cls, path: str | Path) -> "LogProfile":
        return cls.from_records(load(path).records)

    def dominant(self, n: int = 3) -> list[tuple[str, KindStat]]:
        return sorted(self.by_kind.items(), key=lambda kv: kv[1].bytes, reverse=True)[:n]

    def render(self, projections: Iterable[int] = (1_000, 5_000, 20_000)) -> str:
        lines = [f"{self.records} records over {self.steps} steps, "
                 f"{_human(self.bytes)} ({self.bytes_per_step:.0f} B/step)"]
        for kind, stat in sorted(self.by_kind.items(), key=lambda kv: -kv[1].bytes):
            share = 100 * stat.bytes / self.bytes if self.bytes else 0
            lines.append(f"  {kind:<12} {stat.count:>6} recs  {_human(stat.bytes):>9}  "
                         f"{share:5.1f}%  mean {stat.mean_bytes:.0f} B")
        for n in projections:
            lines.append(f"  projected at {n:>6} steps: {_human(self.project(n))}")
        return "\n".join(lines)


@dataclass
class Cadence:
    """Predicted overhead of one snapshot interval."""

    k: int
    steps: int
    snapshots: int
    snapshot_seconds: float
    expected_effect_replay_seconds: float

    @property
    def total_seconds(self) -> float:
        """Only the ``k``-dependent costs. The in-memory replay floor is excluded
        precisely because no choice of ``k`` moves it."""
        return self.snapshot_seconds + self.expected_effect_replay_seconds


@dataclass
class CadenceModel:
    steps: int
    snapshot_seconds: float
    """``C_s``: wall time for one environment snapshot."""
    effect_replay_seconds_per_step: float
    """``C_e``: wall time to re-execute one step's internal effects on replay.

    Not the whole per-step replay cost -- see the module docstring. Passing the
    latter here is the mistake that makes ``k`` look far smaller than it is.
    """
    crashes: float = 1.0
    """``f``: expected crashes per run. Below 1 means most runs never crash."""
    fixed_replay_seconds: float = 0.0
    """Informational: the in-memory replay floor after a crash, which ``k``
    cannot reduce. Reported so the optimum is not mistaken for total recovery
    time."""

    def at(self, k: int) -> Cadence:
        if k < 1:
            raise ValueError("k must be >= 1")
        snapshots = math.ceil(self.steps / k)
        return Cadence(
            k=k, steps=self.steps, snapshots=snapshots,
            snapshot_seconds=snapshots * self.snapshot_seconds,
            expected_effect_replay_seconds=(
                self.crashes * (k / 2) * self.effect_replay_seconds_per_step),
        )

    @property
    def continuous_optimal_k(self) -> float:
        """The closed form, ``sqrt(2 n C_s / (f C_e))``, before discretisation."""
        if self.effect_replay_seconds_per_step <= 0 or self.crashes <= 0:
            return float(self.steps)
        return math.sqrt(2 * self.steps * self.snapshot_seconds
                         / (self.crashes * self.effect_replay_seconds_per_step))

    @property
    def optimal_k(self) -> int:
        """The exact discrete ``k*``.

        The closed form treats the snapshot count as ``n/k``; the real count is
        ``ceil(n/k)``, a staircase. Between two treads a larger ``k`` takes the
        same number of snapshots while leaving a staler window, so the true
        optimum is the smallest ``k`` on the best tread -- generally a little
        below the closed form, never above it by much. Candidates are therefore
        one ``k`` per achievable snapshot count, walked upward until the
        snapshot term alone exceeds the best total found.
        """
        if self.effect_replay_seconds_per_step <= 0 or self.crashes <= 0:
            return self.steps  # nothing to re-execute: snapshot once at the end
        best_k, best_cost = self.steps, self.at(self.steps).total_seconds
        for snapshots in range(1, self.steps + 1):
            if snapshots * self.snapshot_seconds >= best_cost:
                break  # every larger count is worse on the snapshot term alone
            k = max(1, min(self.steps, math.ceil(self.steps / snapshots)))
            cost = self.at(k).total_seconds
            if cost < best_cost:
                best_k, best_cost = k, cost
        return best_k

    def table(self, ks: Iterable[int]) -> list[Cadence]:
        return [self.at(k) for k in ks]

    def render(self, ks: Iterable[int] | None = None) -> str:
        best = self.optimal_k
        ks = sorted({*(ks or ()), best})
        out = [f"C_s={self.snapshot_seconds * 1000:.1f} ms/snapshot  "
               f"C_e={self.effect_replay_seconds_per_step * 1000:.3f} ms/step of stale effects  "
               f"n={self.steps}  f={self.crashes:g}",
               f"k* = {best}",
               f"  {'k':>5} {'snaps':>6} {'snapshot s':>11} {'exp. effects s':>15} "
               f"{'k-dependent s':>14}"]
        for c in self.table(ks):
            mark = "  <- k*" if c.k == best else ""
            out.append(f"  {c.k:>5} {c.snapshots:>6} {c.snapshot_seconds:>11.2f} "
                       f"{c.expected_effect_replay_seconds:>15.2f} "
                       f"{c.total_seconds:>14.2f}{mark}")
        if self.fixed_replay_seconds:
            out.append(f"  plus a {self.fixed_replay_seconds:.2f} s in-memory replay floor "
                       "at every k -- recovery is never free")
        return "\n".join(out)


def recommend_k(steps: int, snapshot_seconds: float, effect_replay_seconds_per_step: float,
                crashes: float = 1.0) -> int:
    """``k*`` for a run of ``steps`` with these measured constants."""
    return CadenceModel(steps, snapshot_seconds, effect_replay_seconds_per_step,
                        crashes).optimal_k


# --------------------------------------------------------------------------
# estimating f, the crash rate
# --------------------------------------------------------------------------
@dataclass
class CrashRate:
    """``f`` estimated from a store's own history rather than assumed.

    Crashes per run is a count over exposure, so the natural model is Poisson.
    The interval is Byar's approximation to the exact Poisson interval -- closed
    form, accurate down to small counts, and it needs no SciPy.
    """

    runs: int
    crashes: int
    lo: float
    hi: float
    """Approximate 95% interval on the rate."""

    @property
    def rate(self) -> float:
        return self.crashes / self.runs if self.runs else 0.0

    @property
    def observed(self) -> bool:
        return self.runs > 0

    def __str__(self) -> str:
        if not self.observed:
            return "no runs recorded: f cannot be estimated, assume a value"
        return (f"f = {self.rate:.3f} crashes/run "
                f"(95% CI {self.lo:.3f}-{self.hi:.3f}, from {self.crashes} crashes "
                f"over {self.runs} runs)")


def _byar(count: int, exposure: float, z: float = 1.959964) -> tuple[float, float]:
    if exposure <= 0:
        return 0.0, 0.0
    lo = 0.0
    if count > 0:
        lo = count * (1 - 1 / (9 * count) - z / (3 * math.sqrt(count))) ** 3
    hi = (count + 1) * (1 - 1 / (9 * (count + 1)) + z / (3 * math.sqrt(count + 1))) ** 3
    return max(0.0, lo / exposure), hi / exposure


def estimate_crash_rate(store: Any, *, include_forks: bool = False) -> CrashRate:
    """Count crashes per run from the logs a store already holds.

    Every recovery leaves evidence in the log: a ``resume`` note, and an
    ``abandon`` record when the crash landed mid-step. Counting resume notes
    counts crashes, because a resume only happens after one.

    Forks are excluded by default: a fork is a deliberate branch, not a run that
    someone had to rescue, and counting them would inflate ``f``.
    """
    from .wal import RecordKind, load  # local import: cost.py stays dependency-free

    runs = crashes = 0
    for meta in store.list_runs():
        if not include_forks and meta.parent_run_id:
            continue
        runs += 1
        records = load(store.log_path(meta.run_id)).records
        crashes += sum(1 for r in records if r.kind == RecordKind.NOTE
                       and r.payload.get("event") == "resume")
    lo, hi = _byar(crashes, runs)
    return CrashRate(runs=runs, crashes=crashes, lo=lo, hi=hi)


def k_range_for_rate(steps: int, snapshot_seconds: float,
                     effect_replay_seconds_per_step: float,
                     rate: CrashRate) -> tuple[int, int, int]:
    """``k*`` at the low, point and high ends of an estimated crash rate.

    Worth computing before agonising over ``f``: since ``k* ∝ f^(-1/2)``, being
    wrong about the crash rate by 10x moves ``k*`` by only ~3x, and the total
    overhead near the optimum is flatter still. An uncertain ``f`` is a weak
    excuse for not using the model -- but note the mapping is inverted, so the
    *high* crash rate gives the *small* k.
    """
    def k_at(f: float) -> int:
        return CadenceModel(steps, snapshot_seconds, effect_replay_seconds_per_step,
                            max(f, 1e-9)).optimal_k

    return k_at(rate.hi), k_at(rate.rate if rate.rate > 0 else rate.hi), k_at(max(rate.lo, 1e-9))


def _human(nbytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(nbytes) < 1024 or unit == "GiB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{int(nbytes)} B"
        nbytes /= 1024
    return f"{nbytes:.1f} GiB"
