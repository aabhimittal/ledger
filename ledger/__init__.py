"""LEDGER -- crash recovery and forking for day-long agent runs.

When one agent run lasts a working day, a crash at hour 9 stops being an agent
problem and becomes an infrastructure problem. LEDGER makes a run *restartable
from the middle* and *branchable from the middle*:

1. **Log** every step to a hash-chained write-ahead log: prompt hash, sampled
   output, tool call, tool result, and every nondeterministic draw.
2. **Snapshot** the environment every k steps (directory CAS, overlayfs delta,
   or Firecracker microVM).
3. **Resume** by restoring the newest snapshot and replaying the recorded
   outputs. Never re-sample -- re-sampling would produce a different run.
4. **Fork** by restoring at step k and letting the agent sample fresh, which
   produces a counterfactual branch.

A redo log plus save points. After a crash you load the last save and replay
the log; to fork, you load the save from before the boss fight and try the
other door.

Quick start::

    from ledger import AgentRunner, RunStore, ToolRegistry, EffectScope, ToolKind

    tools = ToolRegistry()

    @tools.tool(scope=EffectScope.INTERNAL)          # re-executed on replay
    def write_file(path: str, text: str) -> int: ...

    @tools.tool(kind=ToolKind.IRREVERSIBLE, scope=EffectScope.EXTERNAL)
    def send_email(to: str, body: str) -> str: ...   # never re-executed

    runner = AgentRunner(agent_factory=MyAgent, store=RunStore("./.ledger"),
                         workspace="./work", model=my_client, tools=tools,
                         snapshot_every=10)
    out = runner.start()          # crash at hour 9...
    out = runner.resume(out.run_id)         # ...costs minutes
    alt = runner.fork(out.run_id, after_step=120)   # counterfactual branch
"""

from .attribution import ProbeResult, Trial, resample_probe, scan_steps
from .effects import (
    Divergence,
    EffectPolicy,
    EffectScope,
    QuarantineError,
    ToolError,
    ToolKind,
    ToolRegistry,
    ToolRegistryError,
    ToolSpec,
    Undecidable,
    UndecidableEffect,
)
from .entropy import Entropy
from .model import CallableModel, Completion, ModelClient, ScriptedModel, StochasticModel
from .runner import (
    Agent,
    AgentRunner,
    DivergenceError,
    Mode,
    Recorder,
    RunOutcome,
    StepContext,
    StepResult,
    Tape,
    boundary_seq,
)
from .snapshots import (
    CAS,
    DirSnapshotter,
    FirecrackerSnapshotter,
    OverlayFSSnapshotter,
    SnapshotError,
    SnapshotRef,
    SnapshotStore,
    Snapshotter,
)
from .store import RunMeta, RunStore, new_run_id
from .timeline import (
    RunDiff,
    StepView,
    diff_runs,
    live_records,
    load_run,
    render,
    summarize,
)
from .wal import (
    GENESIS,
    LogCorruption,
    Record,
    RecordKind,
    VerifyReport,
    WriteAheadLog,
    abandoned_ranges,
    canon,
    hash_text,
    load,
    verify,
)

__version__ = "0.1.0"

__all__ = [
    "Agent", "AgentRunner", "CAS", "CallableModel", "Completion", "DirSnapshotter",
    "Divergence", "DivergenceError", "EffectPolicy", "EffectScope", "Entropy",
    "FirecrackerSnapshotter", "GENESIS", "LogCorruption", "Mode", "ModelClient",
    "OverlayFSSnapshotter", "ProbeResult", "Record", "RecordKind", "Recorder", "RunDiff",
    "RunMeta", "RunOutcome", "RunStore", "ScriptedModel", "SnapshotError", "SnapshotRef",
    "SnapshotStore", "Snapshotter", "StepContext", "StepResult", "StepView",
    "StochasticModel", "Tape", "ToolError", "ToolKind", "ToolRegistry", "ToolRegistryError",
    "ToolSpec", "Trial", "Undecidable", "UndecidableEffect", "VerifyReport",
    "WriteAheadLog", "__version__", "abandoned_ranges", "boundary_seq", "canon",
    "diff_runs", "hash_text", "live_records", "load", "load_run", "new_run_id",
    "render", "resample_probe", "scan_steps", "summarize", "verify",
]
