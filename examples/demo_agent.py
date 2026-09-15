#!/usr/bin/env python3
"""A runnable end-to-end demonstration: crash, resume, fork, attribute.

The crash is real -- ``--crash-at N`` calls ``os._exit`` from inside the model
call, so the process dies with no unwinding, no ``finally``, and a possibly
half-written final log line. That is the failure mode fsync can only make
*detectable*, never impossible.

::

    python examples/demo_agent.py demo             # the whole story, narrated
    python examples/demo_agent.py run --crash-at 5 --root /tmp/led
    python examples/demo_agent.py resume <run_id>  --root /tmp/led
    python examples/demo_agent.py fork <run_id> --after-step 4 --root /tmp/led
    python examples/demo_agent.py autopsy <run_id> --suspect 4 --root /tmp/led

The "outside world" lives in ``world.json`` next to the store, outside the
snapshotted workspace, so effects that escaped survive the crash and a restore
cannot roll them back -- which is the reason irreversible tools get special
treatment in the first place.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledger import (  # noqa: E402
    AgentRunner,
    Completion,
    EffectPolicy,
    EffectScope,
    RunStore,
    StepContext,
    StepResult,
    ToolKind,
    ToolRegistry,
    Undecidable,
    diff_runs,
    load_run,
    resample_probe,
    summarize,
    verify,
)

WORLD_ENV = "LEDGER_DEMO_WORLD"
CRASH_ENV = "LEDGER_DEMO_CRASH_AT"


# --------------------------------------------------------------------------
# the outside world (persisted, because it must survive the crash)
# --------------------------------------------------------------------------
def world_path() -> Path:
    return Path(os.environ.get(WORLD_ENV, "./demo-world.json"))


def world_read() -> dict:
    p = world_path()
    return json.loads(p.read_text()) if p.exists() else {"published": []}


def world_write(state: dict) -> None:
    p = world_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------
tools = ToolRegistry()


@tools.tool(kind=ToolKind.PURE, scope=EffectScope.INTERNAL)
def write_note(text: str, workspace: Path) -> int:
    """Append a bullet to NOTES.md. Deterministic given the restored workspace."""
    target = workspace / "NOTES.md"
    with open(target, "a") as f:
        f.write(f"- {text}\n")
    return sum(1 for _ in open(target))


@tools.tool(kind=ToolKind.PURE, scope=EffectScope.INTERNAL)
def lint_notes(workspace: Path) -> dict:
    """Inspect the workspace. Its result is derived from state, so replay verifies it."""
    target = workspace / "NOTES.md"
    lines = target.read_text().splitlines() if target.exists() else []
    return {"bullets": len(lines), "blank": sum(1 for line in lines if line.strip() == "-")}


@tools.tool(
    kind=ToolKind.IRREVERSIBLE,
    scope=EffectScope.EXTERNAL,
    idempotency_key=lambda args: f"changelog-{args['version']}",
    reconcile=lambda key: (
        any(p["key"] == key for p in world_read()["published"]),
        "already-published",
    ),
)
def publish_changelog(version: str, body: str, idempotency_key: str | None = None) -> str:
    """Escapes the sandbox. Never re-executed on replay; probed on an ambiguous crash."""
    state = world_read()
    state["published"].append({"key": idempotency_key, "version": version, "body": body})
    world_write(state)
    return f"published-{len(state['published'])}"


# --------------------------------------------------------------------------
# agent
# --------------------------------------------------------------------------
class ReleaseNotesAgent:
    """Writes release notes bullet by bullet, then publishes a changelog."""

    def __init__(self, total_steps: int = 8):
        self.total_steps = total_steps
        self.bullets: list[str] = []

    def step(self, ctx: StepContext) -> StepResult | None:
        prompt = (f"Draft bullet {ctx.step} of {self.total_steps} for the release notes.\n"
                  f"Existing bullets: {self.bullets}")
        bullet = ctx.sample(prompt).text
        self.bullets.append(bullet)
        ctx.call("write_note", text=bullet)

        if ctx.step % 3 == 0:
            report = ctx.call("lint_notes")
            ctx.note(lint=report)

        if ctx.step >= self.total_steps:
            version = f"v0.{len(self.bullets)}"
            receipt = ctx.call("publish_changelog", version=version,
                               body="\n".join(self.bullets))
            return ctx.done({"version": version, "receipt": receipt,
                             "bullets": self.bullets})
        return None


class DemoModel:
    """Deterministic given the prompt, so a resume can be checked against the original.

    A real client goes here unchanged; LEDGER only needs ``sample``.
    """

    def __init__(self, flavour: str = "fix", crash_at: int | None = None):
        self.flavour = flavour
        self.crash_at = crash_at
        self.n_calls = 0

    def sample(self, prompt: str, **params) -> Completion:
        self.n_calls += 1
        step = int(prompt.split()[2])
        if self.crash_at is not None and step == self.crash_at:
            print(f"  !! process killed inside the model call at step {step}", flush=True)
            sys.stdout.flush()
            os._exit(9)          # no unwinding, no cleanup: a real crash
        return Completion(f"{self.flavour} item {step}")


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------
def make_runner(root: str | Path, *, flavour: str = "fix", crash_at: int | None = None,
                total_steps: int = 8, **kw) -> AgentRunner:
    root = Path(root)
    workspace = Path(kw.pop("workspace", root / "workspace"))
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(WORLD_ENV, str(root / "world.json"))
    return AgentRunner(
        agent_factory=lambda: ReleaseNotesAgent(total_steps),
        store=RunStore(root / "store"),
        workspace=workspace,
        model=DemoModel(flavour, crash_at),
        tools=tools,
        snapshot_every=kw.pop("snapshot_every", 3),
        undecidable=kw.pop("undecidable", Undecidable.RECONCILE),
        **kw,
    )


# --------------------------------------------------------------------------
# narrated walkthrough
# --------------------------------------------------------------------------
def walkthrough(root: Path) -> int:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    env = dict(os.environ, **{WORLD_ENV: str(root / "world.json")})

    print("1. Run the agent in a child process that dies at step 5.")
    proc = subprocess.run(
        [sys.executable, __file__, "run", "--root", str(root), "--crash-at", "5"],
        env=env, capture_output=True, text=True,
    )
    print(f"   child exited {proc.returncode}: {proc.stdout.strip().splitlines()[-1]}")

    store = RunStore(root / "store")
    run_id = store.list_runs()[-1].run_id
    report = verify(store.log_path(run_id))
    print(f"   {report}")
    print(f"   steps that made it to disk: "
          f"{[v.step for v in summarize(load_run(store.log_path(run_id)))]}")

    print("\n2. Resume. The snapshot restores the workspace, the log replays the "
          "decisions,\n   and only the un-recorded tail is sampled again.")
    runner = make_runner(root)
    resumed = runner.resume(run_id)
    print(f"   {resumed.status}: {resumed.steps} steps, {resumed.replayed_steps} replayed, "
          f"{resumed.model_calls} fresh samples, {resumed.replayed_samples} from the log")
    print(f"   NOTES.md: {(runner.workspace / 'NOTES.md').read_text().splitlines()}")
    print(f"   published exactly once: {len(world_read()['published'])}")

    print("\n3. Fork after step 4 with a different sampling flavour: a counterfactual "
          "branch.")
    forked = make_runner(root, flavour="feature", workspace=root / "fork-workspace")
    try:
        forked.fork(run_id, after_step=4, workspace=root / "blocked-workspace")
    except Exception as e:
        print(f"   default policy stops the branch at the outbound call:\n     "
              f"{type(e).__name__}: {str(e).splitlines()[0]}")
    branch = forked.fork(run_id, after_step=4, effects=EffectPolicy.DRY_RUN)
    print(f"   with DRY_RUN -- {branch.status}: forked at step {branch.forked_at_step}, "
          f"{branch.replayed_steps} steps replayed")
    print(f"   branch NOTES.md: "
          f"{(forked.workspace / 'NOTES.md').read_text().splitlines()[-3:]}")
    print(f"   parent still intact: "
          f"{(runner.workspace / 'NOTES.md').read_text().splitlines()[-1]}")
    print(f"   {diff_runs(load_run(store.log_path(run_id)), load_run(store.log_path(branch.run_id)))}")
    print(f"   world unchanged by the branch: {len(world_read()['published'])} publication(s)")

    print("\n4. Attribution: resample step 6 three times and see whether the run's "
          "shape\n   depends on it. (Here nothing fails, so the judge just asks whether "
          "the\n   bullet text changed -- the same mechanism a real autopsy uses.)")
    probe = resample_probe(
        forked, run_id, 6,
        judge=lambda o: o.result and "fix item 6" in o.result["bullets"],
        trials=3, model=DemoModel("feature"), workspace_root=root / "probes",
        effects=EffectPolicy.BLOCK,
    )
    print(f"   {probe}")
    print(f"\nStore: {root}. Try:  python -m ledger --root {root / 'store'} runs")
    return 0


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["demo", "run", "resume", "fork", "autopsy"])
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--root", default="./demo-ledger")
    ap.add_argument("--crash-at", type=int, default=None)
    ap.add_argument("--after-step", type=int, default=4)
    ap.add_argument("--suspect", type=int, default=4)
    ap.add_argument("--flavour", default="fix")
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args(argv)
    root = Path(args.root)

    if args.cmd == "demo":
        return walkthrough(root)

    crash_at = args.crash_at if args.crash_at is not None else (
        int(os.environ[CRASH_ENV]) if CRASH_ENV in os.environ else None)
    runner = make_runner(root, flavour=args.flavour, crash_at=crash_at,
                         total_steps=args.steps)

    if args.cmd == "run":
        outcome = runner.start()
    elif args.cmd == "resume":
        outcome = runner.resume(args.run_id)
    elif args.cmd == "fork":
        outcome = runner.fork(args.run_id, args.after_step)
    else:
        probe = resample_probe(runner, args.run_id, args.suspect,
                               judge=lambda o: o.status == "failed", trials=3,
                               workspace_root=root / "probes")
        print(probe)
        return 0

    print(json.dumps(outcome.__dict__, indent=2, default=repr))
    return 0 if outcome.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
