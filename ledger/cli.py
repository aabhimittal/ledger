"""``python -m ledger`` -- inspect, verify, resume and fork runs from a shell.

Read-only commands need nothing but the store. ``resume`` and ``fork`` need the
agent code itself, since replay works by re-running it: point ``--agent`` at a
``module:factory`` that returns a fresh agent, and ``--model`` at a client.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from .cost import CadenceModel, LogProfile
from .effects import EffectPolicy, ToolRegistry
from .runner import AgentRunner
from .store import RunStore
from .timeline import diff_runs, load_run, render, summarize
from .wal import verify


def _resolve(spec: str) -> Any:
    """Import ``module:attribute``."""
    if ":" not in spec:
        raise SystemExit(f"expected module:attribute, got {spec!r}")
    mod, _, attr = spec.partition(":")
    sys.path.insert(0, str(Path.cwd()))
    return getattr(importlib.import_module(mod), attr)


def _build_runner(args: argparse.Namespace, store: RunStore) -> AgentRunner:
    if not args.agent:
        raise SystemExit("--agent module:factory is required for this command")
    factory = _resolve(args.agent)
    model = _resolve(args.model) if args.model else None
    if model is None:
        raise SystemExit("--model module:attribute is required for this command")
    tools = _resolve(args.tools) if args.tools else ToolRegistry()
    return AgentRunner(
        agent_factory=factory, store=store, workspace=args.workspace or "./work",
        model=model() if callable(model) and not hasattr(model, "sample") else model,
        tools=tools() if callable(tools) and not isinstance(tools, ToolRegistry) else tools,
        snapshot_every=args.snapshot_every,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ledger", description=__doc__)
    ap.add_argument("--root", default=".ledger", help="store root (default: .ledger)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("runs", help="list runs")

    for name, helptext in [
        ("show", "run metadata and per-step summary"),
        ("verify", "check the log's hash chain"),
        ("snapshots", "list snapshots usable by a run"),
        ("lineage", "ancestry of a forked run"),
    ]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("run_id")

    p = sub.add_parser("timeline", help="render the raw record stream")
    p.add_argument("run_id")
    p.add_argument("--kinds", help="comma-separated record kinds to keep")
    p.add_argument("--from-seq", type=int, default=1)

    p = sub.add_parser("profile", help="log size breakdown and projections")
    p.add_argument("run_id")
    p.add_argument("--project", type=int, nargs="+", default=[1_000, 5_000, 20_000],
                   help="run lengths to extrapolate to")

    p = sub.add_parser("cadence", help="k* for measured constants (see python -m ledger.bench)")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--snapshot-ms", type=float, required=True, help="C_s")
    p.add_argument("--effect-ms", type=float, required=True,
                   help="C_e: per-step cost of RE-EXECUTING internal effects, not of replay")
    p.add_argument("--crashes", type=float, default=1.0)
    p.add_argument("--k", type=int, nargs="*", default=[])

    p = sub.add_parser("diff", help="find where two runs diverge")
    p.add_argument("left")
    p.add_argument("right")

    p = sub.add_parser("gc", help="delete CAS blobs no snapshot references")
    p.add_argument("--dry-run", action="store_true")

    for name, helptext in [("resume", "restore, replay, continue"),
                           ("fork", "branch a counterfactual run")]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("run_id")
        if name == "fork":
            p.add_argument("--after-step", type=int, required=True)
            p.add_argument("--effects", choices=[e.value for e in EffectPolicy], default="block")
        p.add_argument("--agent", help="module:factory returning a fresh agent")
        p.add_argument("--model", help="module:attribute model client")
        p.add_argument("--tools", help="module:attribute ToolRegistry")
        p.add_argument("--workspace")
        p.add_argument("--snapshot-every", type=int, default=5)

    args = ap.parse_args(argv)
    store = RunStore(args.root)

    if args.cmd == "runs":
        for m in store.list_runs():
            forked = f"  <- {m.parent_run_id}@{m.forked_at_step}" if m.parent_run_id else ""
            print(f"{m.run_id}  {m.status:<10} {m.mode:<7} steps={m.steps:<5} "
                  f"agent={m.agent}{forked}")
        return 0

    if args.cmd == "show":
        meta = store.load_meta(args.run_id)
        print(json.dumps(meta.to_dict(), indent=2, sort_keys=True))
        for view in summarize(load_run(store.log_path(args.run_id))):
            print(view.label)
        return 0

    if args.cmd == "verify":
        report = verify(store.log_path(args.run_id))
        print(report)
        return 0 if report.ok else 1

    if args.cmd == "timeline":
        kinds = args.kinds.split(",") if args.kinds else None
        records = [r for r in load_run(store.log_path(args.run_id)) if r.seq >= args.from_seq]
        print(render(records, kinds=kinds))
        return 0

    if args.cmd == "snapshots":
        for ref in store.snapshots.list(store.lineage_ids(args.run_id)):
            print(f"{ref.snapshot_id}  run={ref.run_id} step={ref.step} at_seq={ref.at_seq} "
                  f"backend={ref.backend} bytes={ref.bytes}")
        return 0

    if args.cmd == "lineage":
        for depth, m in enumerate(store.lineage(args.run_id)):
            at = f" (forked after step {m.forked_at_step})" if m.parent_run_id else ""
            print(f"{'  ' * depth}{m.run_id}  {m.status}{at}")
        return 0

    if args.cmd == "profile":
        print(LogProfile.from_log(store.log_path(args.run_id)).render(args.project))
        return 0

    if args.cmd == "cadence":
        print(CadenceModel(args.steps, args.snapshot_ms / 1000, args.effect_ms / 1000,
                           args.crashes).render(args.k))
        return 0

    if args.cmd == "diff":
        print(diff_runs(load_run(store.log_path(args.left)),
                        load_run(store.log_path(args.right))))
        return 0

    if args.cmd == "gc":
        print(json.dumps(store.snapshots.gc(dry_run=args.dry_run), indent=2))
        return 0

    runner = _build_runner(args, store)
    if args.cmd == "resume":
        outcome = runner.resume(args.run_id)
    else:
        outcome = runner.fork(args.run_id, args.after_step,
                              effects=EffectPolicy(args.effects))
    print(json.dumps(outcome.__dict__, indent=2, default=repr))
    return 0 if outcome.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
