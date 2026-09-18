"""``python -m ledger.bench`` -- measure this machine, then pick ``k``.

Runs a synthetic agent of a given length against a workspace of a given size and
reports the two things the design notes could previously only gesture at:

1. **Log volume.** Bytes per step with prompt text stored and not stored, and a
   projection to day-long lengths.
2. **Snapshot cadence.** ``C_s`` from timing real captures, ``C_e`` from a
   two-point calibration, then the cadence model's ``k*``.

Separating the two recovery costs is the interesting part. Recovery splits into a
floor that ``k`` cannot touch (re-running the agent's code from step 1 to rebuild
in-memory state) and a term proportional to snapshot staleness (re-executing the
internal effects the snapshot is missing). Only the second belongs in the cadence
model, and the runner times it directly -- ``RunOutcome.effect_replay_seconds``
over ``replayed_effects``.

An earlier version of this benchmark inferred ``C_e`` by differencing two resumes
with different snapshot coverage. That measurement was contaminated: the
``k=1`` arm kept snapshotting *after* recovery finished, and those captures
dominated the difference. The two-point run is still reported below as a
cross-check, but the model uses the direct measurement.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import shutil
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .cost import CadenceModel, LogProfile, _human
from .effects import EffectScope, ToolKind, ToolRegistry
from .model import Completion
from .runner import AgentRunner, StepContext, StepResult
from .snapshots import DirSnapshotter
from .store import RunStore
from .wal import RecordKind, load


# --------------------------------------------------------------------------
# synthetic workload
# --------------------------------------------------------------------------
class PaddedModel:
    """Returns a fixed-size reply instantly, so timings measure LEDGER, not a model."""

    def __init__(self, reply_chars: int = 400):
        self.reply_chars = reply_chars
        self.n_calls = 0

    def sample(self, prompt: str, **params: Any) -> Completion:
        self.n_calls += 1
        step = prompt.split("\n", 1)[0].rsplit(" ", 1)[-1]
        return Completion(f"step-{step} " + "x" * max(0, self.reply_chars - 12))


class ChurnAgent:
    """Touches one of ``files`` workspace files per step and carries history.

    The history in the prompt is what forces replay to re-run every step; the
    rotating file writes are what make snapshots non-trivial while staying
    dedupe-friendly -- the shape of a real long run.
    """

    def __init__(self, steps: int, files: int, prompt_chars: int,
                 effect: str = "append_chunk"):
        self.steps = steps
        self.files = files
        self.prompt_chars = prompt_chars
        self.effect = effect
        self.history: list[str] = []

    def step(self, ctx: StepContext) -> StepResult | None:
        context = " ".join(self.history[-6:])
        pad = "c" * max(0, self.prompt_chars - len(context) - 40)
        text = ctx.sample(f"draft step {ctx.step}\ncontext: {context}\n{pad}").text
        self.history.append(text[:24])
        ctx.call(self.effect, slot=ctx.step % self.files, text=text[:80])
        if ctx.step >= self.steps:
            return ctx.done(len(self.history))
        return None


def _tools(workload: str = "append", effect_kb: int = 256) -> ToolRegistry:
    """Two INTERNAL tools spanning the range of C_e that actually matters.

    ``append`` is the cheap end: an 80-byte write, microseconds. ``rebuild`` is
    the expensive end -- write a chunk, then re-digest everything written so far,
    the way a build or a test suite redoes work proportional to accumulated
    state. Both return a value that is deterministic given the restored
    workspace, which is what lets replay re-execute them and verify the result;
    ``rebuild``'s digest covers the whole workspace, so it doubles as a check
    that the snapshot boundary was honoured exactly.
    """
    tools = ToolRegistry()

    def append_chunk(slot: int, text: str, workspace: Path) -> int:
        target = workspace / f"part-{slot:04d}.txt"
        with open(target, "a") as f:
            f.write(text + "\n")
        return target.stat().st_size

    def rebuild(slot: int, text: str, workspace: Path) -> str:
        target = workspace / f"build-{slot:04d}.bin"
        target.write_bytes((text * 64).encode()[:effect_kb * 1024].ljust(effect_kb * 1024, b"."))
        digest = hashlib.sha256()
        for part in sorted(workspace.glob("build-*.bin")):
            digest.update(part.read_bytes())
        return digest.hexdigest()

    tools.register("append_chunk", append_chunk, kind=ToolKind.PURE,
                   scope=EffectScope.INTERNAL)
    tools.register("rebuild", rebuild, kind=ToolKind.PURE, scope=EffectScope.INTERNAL)
    return tools


def _seed(workspace: Path, megabytes: float, files: int = 40) -> None:
    """Pre-populate the workspace so ``C_s`` reflects a real checkout, not an empty dir."""
    workspace.mkdir(parents=True, exist_ok=True)
    if megabytes <= 0:
        return
    per_file = int(megabytes * 1024 * 1024 / max(1, files))
    blob = b"seed" * (per_file // 4 + 1)
    for i in range(files):
        (workspace / f"seed-{i:04d}.bin").write_bytes(blob[:per_file])


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------
@dataclass
class RunSample:
    k: int
    steps: int
    live_seconds: float
    log_bytes: int
    bytes_per_step: float
    snapshots: int
    cas_bytes: int
    crash_step: int
    stale_steps: int
    """Steps whose internal effects had to be re-executed (crash step minus the
    restored snapshot's step)."""
    resume_seconds: float
    replayed_steps: int
    replayed_effects: int = 0
    effect_replay_seconds: float = 0.0

    @property
    def effect_seconds_per_step(self) -> float:
        return (self.effect_replay_seconds / self.replayed_effects
                if self.replayed_effects else 0.0)


@dataclass
class BenchReport:
    steps: int
    files: int
    prompt_chars: int
    seed_mb: float
    workload: str
    snapshot_seconds: float
    effect_replay_seconds_per_step: float
    differential_effect_seconds_per_step: float
    fixed_replay_seconds: float
    optimal_k: dict[str, int]
    naive_optimal_k: dict[str, int]
    log_bytes_per_step: dict[str, float]
    projections: dict[str, dict[str, int]]
    calibration: list[RunSample] = field(default_factory=list)
    samples: list[RunSample] = field(default_factory=list)
    profile_text: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("samples", "calibration"):
            d[key] = [asdict(s) for s in getattr(self, key)]
        return d


def _run_once(root: Path, *, k: int, steps: int, files: int, prompt_chars: int,
              store_prompts: bool, crash_fraction: float, seed_mb: float,
              workload: str = "append", effect_kb: int = 256) -> RunSample:
    """One live run, then a crash at ``crash_fraction`` and a timed resume."""
    if root.exists():
        shutil.rmtree(root)
    workspace = root / "workspace"
    _seed(workspace, seed_mb)
    store = RunStore(root / "store")
    runner = AgentRunner(
        agent_factory=lambda: ChurnAgent(
            steps, files, prompt_chars,
            effect="rebuild" if workload == "rebuild" else "append_chunk"),
        store=store, workspace=workspace, model=PaddedModel(),
        tools=_tools(workload, effect_kb),
        snapshot_every=k, max_steps=steps + 5, store_prompts=store_prompts,
        sync="batch",
    )

    t0 = time.perf_counter()
    out = runner.start()
    live_seconds = time.perf_counter() - t0
    assert out.ok, out.error

    log_path = store.log_path(out.run_id)
    log_bytes = log_path.stat().st_size
    snapshots = store.snapshots.list([out.run_id])
    cas_bytes = sum(p.stat().st_size for p in (store.root / "cas").glob("*/*"))

    # Crash at a step boundary partway in, then time the real recovery.
    records = load(log_path).records
    crash_step = max(1, int(steps * crash_fraction))
    cut = [r.seq for r in records
           if r.kind == RecordKind.STEP_END and r.step == crash_step][-1]
    restored = max((r.step for r in snapshots if r.at_seq <= cut), default=0)
    log_path.write_bytes(b"".join(r.to_line().encode() for r in records if r.seq <= cut))
    meta = store.load_meta(out.run_id)
    meta.status = "running"
    store.save_meta(meta)

    t0 = time.perf_counter()
    resumed = runner.resume(out.run_id)
    resume_seconds = time.perf_counter() - t0
    assert resumed.ok, resumed.error

    return RunSample(
        k=k, steps=steps, live_seconds=live_seconds, log_bytes=log_bytes,
        bytes_per_step=log_bytes / steps, snapshots=len(snapshots), cas_bytes=cas_bytes,
        crash_step=crash_step, stale_steps=crash_step - restored,
        resume_seconds=resume_seconds, replayed_steps=resumed.replayed_steps,
        replayed_effects=resumed.replayed_effects,
        effect_replay_seconds=resumed.effect_replay_seconds,
    )


def _time_capture(root: Path, repeats: int = 3) -> float:
    """Mean seconds for one snapshot of the workspace left by a finished run."""
    store = RunStore(root / "store")
    snapper = DirSnapshotter(root / "workspace")
    return statistics.mean(
        _elapsed(lambda: snapper.capture(store.cas)) for _ in range(repeats))


def _elapsed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def measure(root: Path, *, steps: int = 120, files: int = 30, prompt_chars: int = 2_000,
            ks: list[int] | None = None, crashes: float = 1.0,
            crash_fraction: float = 0.85, seed_mb: float = 0.0,
            workload: str = "append", effect_kb: int = 256) -> BenchReport:
    ks = ks or [5, 10, 25]
    common = dict(steps=steps, files=files, prompt_chars=prompt_chars,
                  crash_fraction=crash_fraction, seed_mb=seed_mb,
                  workload=workload, effect_kb=effect_kb)

    samples = [_run_once(root / f"k{k}", k=k, store_prompts=False, **common) for k in ks]

    # Cross-check pair: a snapshot at every step vs only the step-0 baseline.
    dense = _run_once(root / "cal-dense", k=1, store_prompts=False, **common)
    sparse = _run_once(root / "cal-sparse", k=steps + 1, store_prompts=False, **common)

    # C_e comes from the runner's own timing of re-executed effects, taken from
    # the arm that actually had stale effects to re-execute.
    with_stale = [s for s in samples + [sparse] if s.replayed_effects]
    c_e = statistics.median(s.effect_seconds_per_step for s in with_stale) if with_stale else 0.0
    # The floor: a resume whose snapshot was perfectly fresh still replays
    # every step's in-memory state.
    fixed = max(0.0, dense.resume_seconds - dense.effect_replay_seconds)

    # Same workload with full prompt text kept, to price the store_prompts knob.
    with_prompts = _run_once(root / "prompts", k=ks[0], store_prompts=True, **common)

    c_s = _time_capture(root / f"k{ks[0]}")
    hashed = statistics.median(s.bytes_per_step for s in samples)
    naive_c_r = statistics.median(
        s.resume_seconds / max(1, s.replayed_steps) for s in samples)
    span = sparse.stale_steps - dense.stale_steps
    differential_c_e = ((sparse.resume_seconds - dense.resume_seconds) / span
                        if span else 0.0)

    store = RunStore(root / f"k{ks[0]}" / "store")
    profile = LogProfile.from_log(store.log_path(store.list_runs()[0].run_id))

    lengths = (steps, 1_000, 5_000, 20_000)
    return BenchReport(
        steps=steps, files=files, prompt_chars=prompt_chars, seed_mb=seed_mb,
        workload=workload, snapshot_seconds=c_s, effect_replay_seconds_per_step=c_e,
        differential_effect_seconds_per_step=differential_c_e,
        fixed_replay_seconds=fixed,
        optimal_k={str(n): CadenceModel(n, c_s, c_e, crashes).optimal_k for n in lengths},
        naive_optimal_k={str(n): CadenceModel(n, c_s, naive_c_r, crashes).optimal_k
                         for n in lengths},
        log_bytes_per_step={"hashed": hashed,
                            "store_prompts": with_prompts.bytes_per_step},
        projections={
            mode: {str(n): int(bps * n) for n in (1_000, 5_000, 20_000)}
            for mode, bps in (("hashed", hashed),
                              ("store_prompts", with_prompts.bytes_per_step))
        },
        calibration=[dense, sparse], samples=samples, profile_text=profile.render(),
    )


def render(report: BenchReport, ks: list[int], crashes: float) -> str:
    out = [f"workload: {report.steps} steps, {report.files} churned files, "
           f"~{report.prompt_chars} char prompts, {report.seed_mb:g} MiB seeded "
           f"workspace, internal effect = {report.workload!r}",
           "", "LOG VOLUME", report.profile_text]
    for mode, bps in report.log_bytes_per_step.items():
        proj = report.projections[mode]
        label = "prompt hash + tail (default)" if mode == "hashed" else "store_prompts=True"
        out.append(f"  {label:<30} {bps:>8.0f} B/step   "
                   + "  ".join(f"{n} steps: {_human(v)}" for n, v in proj.items()))
    ratio = (report.log_bytes_per_step["store_prompts"]
             / max(1e-9, report.log_bytes_per_step["hashed"]))
    out.append(f"  storing full prompts costs {ratio:.1f}x the log at this prompt length")

    out += ["", "MEASURED RUNS"]
    out.append(f"  {'k':>5} {'snaps':>6} {'cas':>10} {'live s':>8} {'resume s':>9} "
               f"{'replayed':>9} {'stale':>6} {'effects s':>10}")
    for s in report.samples + report.calibration:
        out.append(f"  {s.k:>5} {s.snapshots:>6} {_human(s.cas_bytes):>10} "
                   f"{s.live_seconds:>8.2f} {s.resume_seconds:>9.2f} "
                   f"{s.replayed_steps:>9} {s.stale_steps:>6} "
                   f"{s.effect_replay_seconds:>10.4f}")
    out.append("  note: 'replayed' is identical at every k -- in-memory replay always")
    out.append("  starts at step 1. Only the 'effects s' column responds to k.")

    out += ["", "SNAPSHOT CADENCE"]
    out.append(CadenceModel(report.steps, report.snapshot_seconds,
                            report.effect_replay_seconds_per_step, crashes,
                            report.fixed_replay_seconds).render(ks))
    out.append(f"  cross-check: differencing the k=1 and k>n resumes gives "
               f"C_e={report.differential_effect_seconds_per_step * 1000:.3f} ms/step "
               f"(vs {report.effect_replay_seconds_per_step * 1000:.3f} measured directly);")
    out.append("  the differential is contaminated by post-recovery snapshots and is "
               "shown only to make that visible.")
    out += ["", f"  {'n':>8} {'k*':>7} {'naive k*':>9}   (naive uses total replay time "
                f"per step -- the mistake)"]
    for n, k in report.optimal_k.items():
        out.append(f"  {n:>8} {k:>7} {report.naive_optimal_k[n]:>9}")
    out += ["",
            "  C_s scales with workspace size and C_e with what the agent's internal",
            "  tools do per step, so re-measure for your workload rather than copying",
            "  these. Both are measured here; f (crash rate) is yours to estimate."]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ledger.bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--files", type=int, default=30)
    ap.add_argument("--prompt-chars", type=int, default=2_000)
    ap.add_argument("--workload", choices=["append", "rebuild"], default="append",
                    help="internal effect per step: a cheap append, or an expensive "
                         "rebuild that re-digests accumulated state")
    ap.add_argument("--effect-kb", type=int, default=256,
                    help="bytes written per rebuild step")
    ap.add_argument("--seed-mb", type=float, default=0.0,
                    help="pre-populate the workspace with this many MiB")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 25])
    ap.add_argument("--crashes", type=float, default=1.0,
                    help="expected crashes per run (f); below 1 if most runs finish")
    ap.add_argument("--root", default=None, help="scratch dir (default: a temp dir)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    import tempfile

    root = Path(args.root) if args.root else Path(tempfile.mkdtemp(prefix="ledger-bench-"))
    try:
        report = measure(root, steps=args.steps, files=args.files,
                         prompt_chars=args.prompt_chars, ks=args.k,
                         crashes=args.crashes, seed_mb=args.seed_mb,
                         workload=args.workload, effect_kb=args.effect_kb)
        print(json.dumps(report.to_dict(), indent=2) if args.json
              else render(report, args.k, args.crashes))
    finally:
        if args.root is None:
            shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
