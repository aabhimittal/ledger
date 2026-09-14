# LEDGER design notes

The mechanism in one paragraph: a run appends every observable event to a
hash-chained write-ahead log and captures its environment every *k* steps. To
recover, restore the newest snapshot and re-run the agent's code with the model
and outbound tools replaced by the recorded log; the moment the log runs out, the
run continues live. To fork, do the same thing but cut the log short on purpose.

Everything below is the part that is not obvious.

## Invariants

1. **Chain integrity.** Record *n* commits to record *n-1*'s digest. A middle
   record cannot be edited without detection, and a forked run that byte-copies
   a prefix is *provably* derived from its parent (`RunDiff.shared_prefix_seq`
   is established by hash equality, not by trust).
2. **`SnapshotRef.at_seq` is exact.** The captured environment reflects the
   effects of records `1..at_seq` and nothing more. Every recovery decision
   keys off this one number.
3. **A torn final record is expected; a torn middle record is corruption.**
   A process killed during `write` leaves a half-line. That step never
   committed, so dropping it is correct and `WriteAheadLog` does it on open.
   A bad record *followed* by good ones cannot be honestly recovered, so it
   raises `LogCorruption` instead.
4. **Effects fire once.** Across any number of crashes, resumes and forks, an
   `EXTERNAL` tool call reaches the outside world exactly once — except in the
   undecidable case below, where the log cannot establish whether it fired at
   all.
5. **A replayed step re-derives, never re-samples.** Divergence between the
   agent's behaviour and the log is an error to report, not a difference to
   absorb (configurable: `STRICT` / `WARN` / `RESAMPLE`).
6. **A resumed log is still replayable.** Recovery composes; see *abandonment*.

## Two kinds of state

This is the design's load-bearing distinction.

| | in-memory state | environment state |
|---|---|---|
| examples | message history, counters, parsed plans | files, packages, a half-built tree |
| rebuilt by | re-running the agent's code | restoring a snapshot |
| replay must | re-run **every** step from step 1 | not re-apply what the snapshot holds |

Hence: replay always starts at step 1 (in-memory state exists only as a
consequence of execution), but internal effects **at or below** the snapshot's
`at_seq` are served from the log while those **above** it are re-executed. That
single threshold is `Recorder.effects_through_seq`.

The two failure modes if you get it wrong are instructive, because both are
silent:

- Replay from the snapshot's *step* instead of step 1 → the agent's history is
  empty, its next prompt differs, and the prompt-hash check fires. (Loud, at
  least, which is why the check exists.)
- Re-execute every internal tool regardless of the threshold → every file write
  below the snapshot line is applied twice. Nothing errors. The workspace is
  just wrong.

The alternative design — serialize the agent's state into the checkpoint, as a
LangGraph checkpointer does — trades a determinism requirement for a schema
requirement. Replay-derived state handles arbitrary Python objects with no
serializers and reconstructs *why* the agent was in a state, not just what it
was; it costs replay time and requires the agent's own code to be deterministic
given its inputs. For agents, whose state is mostly an append-only transcript,
that trade favours replay.

## Effect declarations

Two independent axes. Collapsing them is the bug.

**`ToolKind`** — can this be run again? `PURE` / `IDEMPOTENT` (safe given the
same key) / `IRREVERSIBLE`.

**`EffectScope`** — where does the effect live? `INTERNAL` (inside the snapshot)
/ `EXTERNAL` (outside it) / `MIXED`.

Scope decides replay:

| scope | on replay |
|---|---|
| `INTERNAL`, above the snapshot line | re-execute, then verify the result hash against the log |
| `INTERNAL`, at or below the line | serve the recorded result; re-running would double-apply |
| `EXTERNAL` | serve the recorded result, never execute — this *is* the outbound quarantine |
| `MIXED` | serve the recorded result, so the internal half is lost; `snapshot_after=True` is therefore required at registration |

The registry rejects declarations whose replay behaviour would be undefined:
`MIXED` without `snapshot_after`, `IDEMPOTENT` without an idempotency key (an
unenforceable claim), `IRREVERSIBLE` with `INTERNAL` scope (a contradiction —
replay would re-run it).

`MIXED` is a smell, not a feature. Splitting the tool in two is almost always
right; the flag exists so the framework does not silently do the wrong thing to
code that has not been split yet.

## The crash matrix

Where the process died determines what recovery can honestly do.

| crash point | recovery |
|---|---|
| between steps | replay to the boundary, continue live. The easy case. |
| inside the model call (`prompt`, no `sample`) | **re-sample**. The likeliest crash point, since sampling is the slow part, and safe: a sample has no side effects. The step never committed. |
| inside a `PURE` tool | re-execute. |
| inside an `IDEMPOTENT` tool | re-execute with the **recorded** key, so the peer deduplicates. |
| inside an `IRREVERSIBLE` tool | **undecidable.** See below. |
| during the snapshot commit | the snapshot is invisible (no `ref.json`); fall back to the previous one. |
| between the snapshot commit and its log record | the snapshot is valid but unreferenced by the log. Still usable, because restore consults the snapshot store, not the log. |

That last pair is why the ordering is *capture → commit → log*, never the
reverse: a crash in the gap leaves a harmless orphan snapshot rather than a log
pointing at something that does not exist.

## The undecidable case

A `tool_call` with no `tool_result` for an `IRREVERSIBLE` tool. The log records
that the call was made and nothing about whether it landed. No amount of extra
logging fixes this — the gap is between the process and the outside world, and
any record written before the call is a record of *intent*, not of effect. This
is the same problem as a payment gateway timeout, and it has the same answers:

- `FAIL` (default) — raise `UndecidableEffect`. Refusing to guess is a feature.
- `RECONCILE` — call `spec.reconcile(key)`, which asks the downstream system
  whether that idempotency key was already applied. The only actually correct
  answer, and it requires the tool author to provide a read-back path.
- `RETRY` — assume it did not land. Safe only if the peer deduplicates.
- `ASSUME_APPLIED` — assume it did; requires a declared `assumed_result`.

Whatever is chosen is written to the log as a `note`, so an audit later shows
what was decided and why.

## Abandonment: why a resumed log stays replayable

An interrupted attempt leaves orphans mid-log: a `prompt` with no `sample`, or a
`tool_call` with no `tool_result`, followed by the retry's records. A later
replay reads the orphan and then finds the *next* attempt's records where the
match should be — a spurious divergence, or worse, a spurious "undecidable
effect" for a call the log clearly shows was retried successfully.

Look-ahead heuristics ("if the next record is not a `sample`, assume the prompt
was abandoned") almost work and then silently swallow real divergences. So
abandonment is explicit instead: on flipping from replay to live, the recorder
appends

```
abandon {from_seq: <last completed step_end>, to_seq: <end of the old tail>}
```

and `Tape` filters those ranges out at construction. If the flip happened
mid-step, the step's `step_start` was abandoned with the rest, so a fresh one is
appended — the live continuation is a well-formed step, not a headless one.

Consequence: recovery composes. Crash → resume → crash → resume → fork across
the seam all work, and `python -m ledger timeline` marks retired records with
`x` so the history remains auditable rather than merely correct.

## Snapshot backends

| backend | captures | needs | cost |
|---|---|---|---|
| `DirSnapshotter` | whole tree, content-addressed | nothing | one hash per file; copies only changed files. Restore is a full copy. |
| `OverlayFSSnapshotter` | upperdir delta incl. whiteouts | mount privileges | proportional to the delta, not the image |
| `FirecrackerSnapshotter` | guest RAM + device state | a running microVM | a memory-file write per snapshot; the only backend that restores *processes* |

Capture/restore asymmetry in `DirSnapshotter` is deliberate: snapshots happen
every *k* steps, restores happen once per crash. Optimise the frequent one.

Firecracker is the backend that makes LEDGER strictly more capable than a
file-level checkpointer: it survives a crash in the middle of a `pip install`,
because it restores the process tree and not just the filesystem. It is also the
most expensive, which is the honest trade.

## Attribution is monotone, not step-local

`resample_probe(run, k)` forks at `k-1` and lets the agent re-sample from `k`
onward. So it measures "does resampling *from* `k` fix the run", not "is `k` at
fault". Any `k` at or before the true culprit appears to fix it. The culprit is
the **last** step whose resampling still clears the failure, so a binary search
over the step range finds it in log(n) forks. `scan_steps` breaks ties on
attribution towards the later step for this reason.

Two further honesty notes: trials quarantine irreversible effects by default, so
a failure whose *cause* is an outbound effect will not reproduce faithfully; and
a noisy judge turns the whole thing into a coin-flip counter.

## Deliberate omissions

- **No distributed coordination.** One run, one process, one log. Multi-agent
  runs would need per-agent logs plus a causal order across them — a real
  project, not an extension.
- **No log compaction.** A day-long run's log is dominated by prompts, which is
  why full prompt text is off by default (`store_prompts=False` keeps a hash
  plus a tail). Compaction would mean rewriting the chain, which breaks
  provenance; the honest version is a separate archival format.
- **No automatic snapshot cadence tuning.** `k` is a cost/recovery-time knob
  the operator sets. Auto-tuning it would need a cost model nobody has
  calibrated yet.
- **No sandbox.** LEDGER records and replays what the agent does; it does not
  constrain it. Effect quarantine is a replay-safety mechanism, not a security
  boundary — a tool declared `PURE` that mails your customers will mail them.
