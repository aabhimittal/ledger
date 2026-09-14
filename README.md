# LEDGER

**Crash recovery and forking for day-long agent runs.**

METR's public frontier for autonomous task length sat around 12 hours at a 50%
success rate in early 2026, with internal models plausibly past 16 — the point
where METR itself cautions that measurements stop being reliable. Once a single
agent run lasts a working day, a crash at hour 9 stops being an agent problem
and becomes an infrastructure problem. Nine hours of sampling, tool calls and
accumulated environment state evaporate because a process died.

LEDGER makes a run **restartable from the middle** and **branchable from the
middle**:

| | |
|---|---|
| **Log** | A hash-chained write-ahead log of every step: prompt hash, sampled output, tool call, tool result, and every nondeterministic draw. |
| **Snapshot** | The environment, every *k* steps — content-addressed directory, overlayfs delta, or Firecracker microVM. |
| **Resume** | Restore the newest snapshot, replay the recorded outputs, continue live where the log ends. Never re-sample: re-sampling produces a different run, not the same one. |
| **Fork** | Restore at step *k*, let the agent sample fresh, and get a counterfactual branch. |

A database redo log plus video-game save points. After a crash you load the last
save and replay the log. To fork, you load the save from before the boss fight
and try the other door.

```bash
python examples/demo_agent.py demo      # kills a child process at step 5, then recovers it
```

```
1. Run the agent in a child process that dies at step 5.
   child exited 9: !! process killed inside the model call at step 5
   .../log.jsonl: 32 records -- ok
   steps that made it to disk: [0, 1, 2, 3, 4, 5]

2. Resume.
   completed: 8 steps, 5 replayed, 4 fresh samples, 4 from the log
   published exactly once: 1

3. Fork after step 4 with a different sampling flavour.
   default policy stops the branch at the outbound call:
     QuarantineError: tool 'publish_changelog' is irreversible and outbound
     effects are quarantined in fork mode.
   shared prefix through seq 30; diverged at step 5 on sample/sample
     left : 'fix item 5'
     right: 'feature item 5'
```

Pure standard library, Python 3.10+.

## Usage

```python
from ledger import AgentRunner, RunStore, ToolRegistry, ToolKind, EffectScope

tools = ToolRegistry()

@tools.tool(scope=EffectScope.INTERNAL)             # re-executed on replay
def write_file(path: str, text: str, workspace) -> int:
    (workspace / path).write_text(text)
    return len(text)

@tools.tool(kind=ToolKind.IRREVERSIBLE, scope=EffectScope.EXTERNAL,
            idempotency_key=lambda a: f"mail-{a['to']}",
            reconcile=lambda key: (mail_api.was_sent(key), "already-sent"))
def send_email(to: str, body: str) -> str:          # never re-executed
    return mail_api.send(to, body)

class MyAgent:
    def __init__(self):
        self.history = []                            # plain Python state, no schema

    def step(self, ctx):
        reply = ctx.sample(f"history: {self.history}\nnext action?")
        self.history.append(reply.text)
        ctx.call("write_file", path="out.md", text=reply.text)
        if "DONE" in reply.text:
            return ctx.done(self.history)

runner = AgentRunner(agent_factory=MyAgent, store=RunStore("./.ledger"),
                     workspace="./work", model=my_client, tools=tools,
                     snapshot_every=10)

out = runner.start()                    # ... crashes at hour 9
out = runner.resume(out.run_id)         # ... costs minutes
alt = runner.fork(out.run_id, after_step=120)   # counterfactual branch
```

`model` is anything with `sample(prompt, **params) -> Completion | str`, so real
clients drop in unchanged. The agent's in-memory state needs no serializer: it is
re-derived by re-running the agent's code against the recorded samples.

```bash
python -m ledger --root .ledger runs                # every run and its lineage
python -m ledger --root .ledger show   <run_id>     # metadata + per-step summary
python -m ledger --root .ledger timeline <run_id>   # the raw record stream
python -m ledger --root .ledger verify <run_id>     # hash-chain integrity
python -m ledger --root .ledger diff <run_a> <run_b>  # where two runs part ways
python -m ledger --root .ledger gc                  # drop unreferenced blobs
```

## The four decisions that make this more than plumbing

Durable execution is a solved commodity — Temporal, Step Functions and LangGraph
checkpointers all replay a workflow from a log. The parts worth building are the
ones they leave to you, and all four fall out of one question: *what exactly is
being rebuilt?*

**1. Two kinds of state, two mechanisms, one dividing line.**
In-memory state (message history, counters) exists only as a consequence of
running the agent's code, so replay must re-run *every* step from the beginning.
Environment state (files, installed packages, a half-finished build) comes from
the snapshot. That means the snapshot's sequence number is a hard boundary:
internal effects recorded at or below it are already present and must **not** run
again; those above it must. Get it wrong in one direction and the agent resumes
against a workspace *k* steps stale; get it wrong in the other and every file
write is applied twice.

**2. Reversibility and scope are different axes.** The famous failure — "a
replayed *send email* sends the email twice" — is usually framed as needing
idempotency keys. It actually needs a second declaration: *where* does the effect
live? A tool inside the snapshot (`INTERNAL`) must be re-executed on replay,
because that is the only thing that rebuilds the environment delta. A tool
outside it (`EXTERNAL`) must never be. Quarantine is then not a special replay
mode anyone has to remember to switch on — it is what the declaration means.

**3. Nondeterminism that is not the model.** An agent that calls `time.time()`
for a filename or `random.choice` for a retry target has pulled a value out of
thin air, and a replay that redraws it diverges silently — a wrong file path, not
an error. Every draw goes through `ctx.entropy` and lands in the log. This is
also what makes re-executing `INTERNAL` tools legitimate: given the restored
filesystem plus the recorded entropy stream, they are pure functions.

**4. Recovery has to be composable.** A day-long run crashes more than once, and
each crash leaves half a step behind: a `prompt` with no `sample`, a `tool_call`
with no `tool_result`. Left live, those orphans make a once-resumed log
*unreplayable* — the next replay reads the interrupted attempt and finds the next
attempt's records where the match should be. LEDGER writes an explicit `abandon`
record retiring the range, the same way a real WAL writes abort records. A run
that crashed three times then replays as cleanly as one that never crashed.

## What it does not solve

- **Undecidable effects.** If the crash landed *between* a `tool_call` and its
  `tool_result`, whether the effect reached the outside world is not knowable
  from the log. LEDGER refuses to guess: the default raises `UndecidableEffect`.
  The only correct answer is a read-back — give the tool a `reconcile(key)` probe
  that asks the downstream system — and that requires work from the tool author
  that no framework can do for them.
- **Determinism you did not route through the log.** Nondeterminism inside a
  tool's own implementation (thread scheduling, a clock read in a subprocess) is
  not captured. Snapshot more often to shrink the exposure.
- **Replay cost.** Re-deriving in-memory state means re-running the agent's code
  for every step. That is cheap (no sampling, no tool execution below the
  snapshot line) but not free, and it is the price of not writing checkpoint
  schemas.
- **The log is not the differentiator.** If you only need durable execution, use
  Temporal. What is here that those do not give you is environment snapshots,
  mid-run forking, and the effect-scope model that makes replay safe.
- **Snapshot cost is real.** `DirSnapshotter` dedupes unchanged files by content,
  so hour 9 costs little more than hour 8 — but a Firecracker full snapshot
  writes out guest RAM every time. Pick `k` accordingly.

## Forking is the primitive failure attribution needs

A 400-step run fails. The log tells you *what* happened; it does not tell you
which step *caused* it, because everything after a bad step is downstream of it.
Forking answers that counterfactually — restore the state as of step *k-1*, let
the agent sample step *k* again, and see whether the failure survives:

```python
from ledger import resample_probe
probe = resample_probe(runner, run_id, suspect_step=137,
                       judge=lambda out: out.status == "failed", trials=5)
print(probe)   # step 137: failure reproduced in 0/5 resamples -> attribution 1.00
```

With one caveat that matters: forking at *k* resamples step *k* **and everything
after it**, so attribution is monotone rather than step-local. Any *k* at or
before the real culprit will appear to fix the run. The culprit is the **last**
step whose resampling still clears the failure — which makes this a binary search
over the step range, not a linear scan.

## Layout

```
ledger/wal.py          hash-chained log, torn-tail repair, abandon records
ledger/snapshots.py    CAS + dir / overlayfs / Firecracker backends
ledger/effects.py      tool kinds, effect scopes, policies
ledger/entropy.py      recorded clock, randomness, identity
ledger/runner.py       the one loop that serves live, resume and fork
ledger/attribution.py  resampling probes
ledger/timeline.py     step summaries, rendering, run-vs-run diffs
ledger/cli.py          python -m ledger
examples/demo_agent.py runnable crash / resume / fork / attribute walkthrough
docs/DESIGN.md         invariants, crash matrix, ordering rules, open problems
```

```bash
pip install -e ".[dev]" && pytest      # 62 tests, including a real os._exit crash
```
