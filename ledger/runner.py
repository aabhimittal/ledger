"""The run loop: record live, replay on resume, sample fresh on fork.

One loop serves all three modes. The only state that distinguishes them is a
*tape* -- the prefix of the log that this run must reproduce rather than
generate:

live
    empty tape. Everything is generated and logged.
resume
    tape = the whole surviving log. Restore the newest snapshot, replay the log
    from the beginning, and the moment the tape runs out, keep going live. The
    flip is silent and automatic, which is the property that makes a crash at
    hour 9 cost minutes instead of nine hours.

    Replay starts at step 1 even though the snapshot covers step S, because two
    different kinds of state are being rebuilt and they need different
    treatment. In-memory state exists only as a consequence of running the
    agent's code, so every step has to be re-run. Environment state comes from
    the snapshot, so internal effects recorded at or below the snapshot's
    ``at_seq`` must *not* be re-executed, while those above it must be. That
    single sequence number is the whole dividing line (``effects_through_seq``);
    getting it wrong means either an environment four steps stale or every file
    write applied twice.
fork
    tape = the log truncated at the end of step k. Same replay, but the tape
    is *deliberately* short, so the agent starts sampling freshly at step k+1
    against the environment as it stood at step k.

Recovery and forking are therefore the same mechanism with a different
truncation point -- which is why building forking costs almost nothing once
recovery works, and why it is the primitive failure attribution needs.

The agent's own Python state (message history, scratch variables) is never
serialized. It is *re-derived* by re-running the agent code against recorded
samples. That buys arbitrary in-memory state with no checkpoint schema, and
costs replay time plus a hard determinism requirement on the agent's own code.
That trade is the main difference from a state-serializing checkpointer.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from .entropy import Entropy
from .effects import (
    _MISSING,
    Divergence,
    EffectPolicy,
    EffectScope,
    QuarantineError,
    ToolError,
    ToolKind,
    ToolRegistry,
    ToolSpec,
    Undecidable,
    UndecidableEffect,
)
from .model import Completion, ModelClient, normalize, params_hash
from .snapshots import DirSnapshotter, SnapshotRef, Snapshotter
from .store import RunMeta, RunStore, new_run_id
from .wal import (
    TRANSPARENT,
    Record,
    RecordKind,
    WriteAheadLog,
    abandoned_ranges,
    canon,
    hash_text,
    load,
)


class Mode(str, Enum):
    LIVE = "live"
    RESUME = "resume"
    FORK = "fork"


class TapeExhausted(Exception):
    """The replayed prefix ended; from here the run is live."""


class DivergenceError(RuntimeError):
    """Replay produced something the log does not contain."""


def _loggable(value: Any) -> tuple[Any, bool]:
    try:
        canon(value)
        return value, True
    except (TypeError, ValueError):
        return repr(value), False


def _result_hash(ok: bool, value: Any, err_type: str | None = None) -> str:
    """Hash a tool outcome for replay verification.

    Failures hash on the exception *type*, not its message: messages routinely
    carry paths, timings and addresses that differ between runs without
    meaning the replay diverged.
    """
    subject = value if ok else err_type
    try:
        return hash_text(canon([ok, subject]))
    except (TypeError, ValueError):
        return hash_text(canon([ok, repr(subject)]))


# --------------------------------------------------------------------------
# the tape
# --------------------------------------------------------------------------
class Tape:
    """A cursor over the log prefix a replay must reproduce."""

    def __init__(self, records: Iterable[Record], *, until_seq: int | None = None):
        records = list(records)
        self.until_seq = until_seq
        dead = abandoned_ranges(records)
        self._q: deque[Record] = deque(
            r for r in records
            if (until_seq is None or r.seq <= until_seq)
            and not any(lo < r.seq <= hi for lo, hi in dead)
        )
        self.size = len(self._q)
        self.last_seq = self._q[-1].seq if self._q else 0

    def _skip_transparent(self) -> None:
        while self._q and self._q[0].kind in TRANSPARENT:
            self._q.popleft()

    def peek(self) -> Record | None:
        self._skip_transparent()
        return self._q[0] if self._q else None

    @property
    def empty(self) -> bool:
        return self.peek() is None

    @property
    def remaining(self) -> int:
        self._skip_transparent()
        return len(self._q)

    def take(self, *kinds: str) -> Record:
        rec = self.peek()
        if rec is None:
            raise TapeExhausted()
        if kinds and rec.kind not in kinds:
            raise DivergenceError(
                f"replay expected {'/'.join(kinds)} but log seq {rec.seq} is "
                f"{rec.kind!r} (step {rec.step}); the agent is making different "
                "calls than it did originally"
            )
        self._q.popleft()
        return rec


# --------------------------------------------------------------------------
# recorder: the single place that knows live-vs-replay
# --------------------------------------------------------------------------
class Recorder:
    def __init__(self, *, log: WriteAheadLog, tape: Tape, model: ModelClient,
                 tools: ToolRegistry, mode: Mode, divergence: Divergence,
                 effect_policy: EffectPolicy, undecidable: Undecidable,
                 workspace: Path | None = None, effects_through_seq: int = 0,
                 store_prompts: bool = False, prompt_preview: int = 240):
        self.log = log
        self.workspace = Path(workspace) if workspace else Path.cwd()
        # Records up to here already have their environment effects baked into
        # the restored snapshot, so replaying them must NOT re-execute internal
        # tools. Past this line the snapshot is stale and re-execution is the
        # only thing that rebuilds the delta. See _replay_call.
        self.effects_through_seq = effects_through_seq
        self.tape = tape
        self.model = model
        self.tools = tools
        self.mode = mode
        self.divergence = divergence
        self.effect_policy = effect_policy
        self.undecidable = undecidable
        self.store_prompts = store_prompts
        self.prompt_preview = prompt_preview

        self.live = tape.empty
        self.step = 0
        self.replayed_steps = 0
        self.replayed_samples = 0
        self.model_calls = 0
        self.divergences = 0
        self.flip_reason: str | None = None
        self._snapshot_requested = False
        self._replaying_step = False
        self._last_step_end_seq = 0

    # -- mode transitions ---------------------------------------------
    def _flip(self, reason: str, **kv: Any) -> None:
        """Stop replaying and start generating, retiring the interrupted attempt."""
        if self.live:
            return
        self.live = True
        self.flip_reason = reason
        # Everything the tape still holds past the last completed step belongs to
        # an attempt that is now superseded. Say so in the log, so replaying
        # *this* log later skips it instead of tripping over half a step.
        if self.tape.last_seq > self._last_step_end_seq:
            self.log.append(
                RecordKind.ABANDON,
                {"from_seq": self._last_step_end_seq, "to_seq": self.tape.last_seq,
                 "reason": reason},
                step=self.step,
            )
        self.log.append(
            RecordKind.NOTE,
            {"event": "replay_end", "reason": reason, "tape_remaining": self.tape.remaining, **kv},
            step=self.step,
        )
        if self._replaying_step:
            # This step's step_start was just abandoned along with the rest;
            # re-open it so the live continuation is a well-formed step.
            self.log.append(RecordKind.STEP_START, {}, step=self.step)
            self._replaying_step = False

    def _on_divergence(self, what: str, rec: Record, detail: dict) -> None:
        self.divergences += 1
        action = self.divergence.value
        self.log.append(
            RecordKind.NOTE,
            {"event": "divergence", "what": what, "at_seq": rec.seq, "action": action,
             "detail": _loggable(detail)[0]},
            step=self.step,
        )
        msg = (f"replay divergence at log seq {rec.seq} ({what}): {detail}. "
               "The agent code, prompt template or tool arguments differ from the "
               "recorded run.")
        if self.divergence is Divergence.STRICT:
            raise DivergenceError(msg)
        if self.divergence is Divergence.RESAMPLE:
            self._flip("divergence", what=what)

    # -- step boundaries ----------------------------------------------
    def begin_step(self, step: int) -> bool:
        """Returns True if this step runs live, False if it is being replayed."""
        self.step = step
        if not self.live:
            try:
                rec = self.tape.take(RecordKind.STEP_START)
            except TapeExhausted:
                self._flip("tape_exhausted")
            else:
                self._replaying_step = True
                if rec.step != step:
                    self._on_divergence("step_start", rec,
                                        {"expected_step": step, "logged_step": rec.step})
                if not self.live:
                    self.replayed_steps += 1
                    return False
        self._replaying_step = False
        self.log.append(RecordKind.STEP_START, {}, step=step)
        return True

    def end_step(self, step: int, done: bool) -> None:
        if not self.live:
            try:
                rec = self.tape.take(RecordKind.STEP_END)
            except TapeExhausted:
                self._flip("tape_exhausted")
            else:
                self._last_step_end_seq = rec.seq
                self._replaying_step = False
                return
        self.log.append(RecordKind.STEP_END, {"done": done}, step=step)
        self._replaying_step = False

    def consume_snapshot_request(self) -> bool:
        requested, self._snapshot_requested = self._snapshot_requested, False
        return requested

    def request_snapshot(self) -> None:
        if self.live:
            self._snapshot_requested = True

    # -- model --------------------------------------------------------
    def sample(self, prompt: str, **params: Any) -> Completion:
        ph = hash_text(prompt)
        if not self.live:
            try:
                rec = self.tape.take(RecordKind.PROMPT)
            except TapeExhausted:
                self._flip("tape_exhausted")
            else:
                if rec.payload.get("prompt_hash") != ph:
                    self._on_divergence("prompt", rec, {
                        "logged_prompt_hash": str(rec.payload.get("prompt_hash"))[:12],
                        "actual_prompt_hash": ph[:12],
                        "logged_chars": rec.payload.get("chars"),
                        "actual_chars": len(prompt),
                    })
                if not self.live:
                    try:
                        srec = self.tape.take(RecordKind.SAMPLE)
                    except TapeExhausted:
                        # The crash landed inside the model call -- the likeliest
                        # place for it, since sampling is the slow part. There is
                        # no recorded output to replay, and re-sampling is safe
                        # because a sample has no side effects of its own.
                        self._flip("incomplete_sample")
                    else:
                        self.replayed_samples += 1
                        return Completion(
                            srec.payload["text"],
                            srec.payload.get("finish_reason", "stop"),
                            srec.payload.get("usage"),
                        )
        payload: dict[str, Any] = {
            "prompt_hash": ph, "params_hash": params_hash(params), "chars": len(prompt),
        }
        if self.store_prompts:
            payload["prompt"] = prompt
        elif self.prompt_preview:
            payload["tail"] = prompt[-self.prompt_preview:]
        self.log.append(RecordKind.PROMPT, payload, step=self.step)
        completion = normalize(self.model.sample(prompt, **params))
        self.model_calls += 1
        self.log.append(
            RecordKind.SAMPLE,
            {"text": completion.text, "finish_reason": completion.finish_reason,
             "usage": completion.usage, "text_hash": hash_text(completion.text)},
            step=self.step,
        )
        return completion

    # -- entropy ------------------------------------------------------
    def draw(self, dkind: str, thunk: Callable[[], Any], extra: dict | None = None) -> Any:
        if not self.live:
            try:
                rec = self.tape.take(RecordKind.ENTROPY)
            except TapeExhausted:
                self._flip("tape_exhausted")
            else:
                if rec.payload.get("dkind") != dkind:
                    self._on_divergence("entropy", rec, {
                        "logged": rec.payload.get("dkind"), "actual": dkind})
                if not self.live:
                    return rec.payload["value"]
        value = thunk()
        payload = {"dkind": dkind, "value": value}
        if extra:
            payload.update(extra)
        self.log.append(RecordKind.ENTROPY, payload, step=self.step)
        return value

    # -- tools --------------------------------------------------------
    def call(self, name: str, args: dict) -> Any:
        spec = self.tools.get(name)
        try:
            args_hash = hash_text(canon(args))
        except (TypeError, ValueError) as e:
            raise ToolError(
                f"arguments to {name!r} must be JSON-serializable to be logged: {e}"
            ) from None
        if not self.live:
            try:
                crec = self.tape.take(RecordKind.TOOL_CALL)
            except TapeExhausted:
                self._flip("tape_exhausted")
            else:
                if crec.payload.get("tool") != name or crec.payload.get("args_hash") != args_hash:
                    self._on_divergence("tool_call", crec, {
                        "logged_tool": crec.payload.get("tool"), "actual_tool": name,
                        "args_match": crec.payload.get("args_hash") == args_hash})
                if not self.live:
                    return self._replay_call(spec, crec, args)
        return self._live_call(spec, args)

    def _replay_call(self, spec: ToolSpec, crec: Record, args: dict) -> Any:
        nxt = self.tape.peek()
        if (nxt is None or nxt.kind != RecordKind.TOOL_RESULT
                or nxt.payload.get("call_id") != crec.payload.get("call_id")):
            # The crash landed between the call and its result.
            return self._recover_incomplete(spec, crec, args)
        rrec = self.tape.take(RecordKind.TOOL_RESULT)
        covered_by_snapshot = crec.seq <= self.effects_through_seq

        if spec.replay_executes and not covered_by_snapshot:
            # INTERNAL scope past the snapshot: re-run it. This is what rebuilds
            # the environment delta accumulated between the restored snapshot
            # and the crash. Below the snapshot line the effect is already
            # present, and re-running would double-apply it.
            failure: Exception | None = None
            value: Any = None
            try:
                value = self._invoke(spec, args, crec.payload.get("idempotency_key"))
            except Exception as e:  # noqa: BLE001 - re-raised below
                failure = e
            observed = _result_hash(failure is None, value,
                                    None if failure is None else type(failure).__name__)
            if observed != rrec.payload.get("result_hash"):
                self._on_divergence("tool_result", rrec, {
                    "tool": spec.name, "logged_ok": rrec.payload.get("ok"),
                    "actual_ok": failure is None,
                    "hint": "an INTERNAL tool is not deterministic given the restored "
                            "environment and recorded entropy",
                })
            if failure is not None:
                raise failure
            return value

        # Either EXTERNAL scope (quarantined by construction) or an internal
        # effect the snapshot already contains: serve the recorded result.
        if not rrec.payload.get("ok", True):
            raise ToolError(f"{spec.name} (replayed failure): {rrec.payload.get('error')}")
        if "result" not in rrec.payload:
            raise ToolError(
                f"cannot replay {spec.name!r}: its effect is already inside the restored "
                "snapshot so it must not run again, but its recorded result was not "
                "JSON-serializable and there is nothing to hand back. INTERNAL tools must "
                "return serializable results to survive a resume across a snapshot boundary."
            )
        return rrec.payload["result"]

    def _recover_incomplete(self, spec: ToolSpec, crec: Record, args: dict) -> Any:
        key = crec.payload.get("idempotency_key")
        base = {"event": "incomplete_effect", "tool": spec.name,
                "call_id": crec.payload.get("call_id"), "kind": spec.kind.value,
                "scope": spec.scope.value, "idempotency_key": key}

        if spec.kind is ToolKind.PURE:
            decision = "reexecute"
        elif spec.kind is ToolKind.IDEMPOTENT:
            decision = "reexecute_with_key"
        elif self.undecidable is Undecidable.RECONCILE:
            if spec.reconcile is None:
                self.log.append(RecordKind.NOTE,
                                {**base, "decision": "fail", "why": "no reconcile probe"},
                                step=self.step)
                raise UndecidableEffect(
                    f"{spec.name!r} needs a reconcile(key) probe for Undecidable.RECONCILE")
            applied, value = spec.reconcile(key)
            self.log.append(RecordKind.NOTE,
                            {**base, "decision": "reconciled", "applied": bool(applied)},
                            step=self.step)
            self._flip("incomplete_effect")
            if applied:
                return value
            decision = "reexecute_with_key"
        elif self.undecidable is Undecidable.RETRY:
            decision = "reexecute_with_key"
        elif self.undecidable is Undecidable.ASSUME_APPLIED:
            if spec.assumed_result is _MISSING:
                raise UndecidableEffect(
                    f"Undecidable.ASSUME_APPLIED needs {spec.name!r} to declare assumed_result")
            self.log.append(RecordKind.NOTE, {**base, "decision": "assume_applied"}, step=self.step)
            self._flip("incomplete_effect")
            return spec.assumed_result
        else:
            self.log.append(RecordKind.NOTE, {**base, "decision": "fail"}, step=self.step)
            raise UndecidableEffect(
                f"the crash landed inside irreversible tool {spec.name!r} "
                f"(call {crec.payload.get('call_id')}): the log records the call but no "
                "result, so whether the effect reached the outside world is not knowable "
                "from the log alone. Give the tool a reconcile(key) probe and use "
                "Undecidable.RECONCILE, or choose RETRY / ASSUME_APPLIED deliberately."
            )

        self.log.append(RecordKind.NOTE, {**base, "decision": decision}, step=self.step)
        self._flip("incomplete_effect")
        return self._live_call(spec, args, idempotency_key=key)

    def _live_call(self, spec: ToolSpec, args: dict, idempotency_key: str | None = None) -> Any:
        args_hash = hash_text(canon(args))
        if spec.kind is ToolKind.IRREVERSIBLE and self.effect_policy is EffectPolicy.BLOCK:
            self.log.append(
                RecordKind.NOTE,
                {"event": "effect_blocked", "tool": spec.name, "args_hash": args_hash,
                 "policy": self.effect_policy.value, "mode": self.mode.value},
                step=self.step,
            )
            raise QuarantineError(
                f"tool {spec.name!r} is irreversible and outbound effects are quarantined "
                f"in {self.mode.value} mode. A counterfactual branch re-running this would "
                "hit the outside world twice. Pass effects=EffectPolicy.ALLOW (or DRY_RUN) "
                "if that is what you want."
            )

        key = idempotency_key
        if key is None and spec.idempotency_key is not None:
            key = spec.idempotency_key(dict(args))
        # The call_id is the seq the tool_call record is about to take, which
        # makes it unique across resumes and forks without extra bookkeeping.
        call_id = f"c{self.log.next_seq}"
        self.log.append(
            RecordKind.TOOL_CALL,
            {"call_id": call_id, "tool": spec.name, "args": args, "args_hash": args_hash,
             "kind": spec.kind.value, "scope": spec.scope.value, "idempotency_key": key},
            step=self.step,
        )

        if spec.kind is ToolKind.IRREVERSIBLE and self.effect_policy is EffectPolicy.DRY_RUN:
            self.log.append(
                RecordKind.TOOL_RESULT,
                {"call_id": call_id, "ok": True, "result": spec.dry_run_result, "dry_run": True,
                 "result_hash": _result_hash(True, spec.dry_run_result), "ms": 0.0},
                step=self.step,
            )
            return spec.dry_run_result

        t0 = time.perf_counter()
        try:
            value = self._invoke(spec, args, key)
        except Exception as e:
            self.log.append(
                RecordKind.TOOL_RESULT,
                {"call_id": call_id, "ok": False, "error": f"{type(e).__name__}: {e}",
                 "result_hash": _result_hash(False, None, type(e).__name__),
                 "ms": round((time.perf_counter() - t0) * 1000, 3)},
                step=self.step,
            )
            if spec.snapshot_after:
                self._snapshot_requested = True
            raise

        ms = round((time.perf_counter() - t0) * 1000, 3)
        payload: dict[str, Any] = {"call_id": call_id, "ok": True, "ms": ms,
                                   "result_hash": _result_hash(True, value)}
        logged, serializable = _loggable(value)
        unreplayable = not serializable and spec.scope is not EffectScope.INTERNAL
        payload["result" if serializable else "result_repr"] = logged
        self.log.append(RecordKind.TOOL_RESULT, payload, step=self.step)
        if spec.snapshot_after:
            self._snapshot_requested = True
        if unreplayable:
            raise ToolError(
                f"tool {spec.name!r} has {spec.scope.value} scope but returned a "
                f"non-serializable {type(value).__name__}; replay would have nothing to "
                "hand back. Return JSON-serializable data, or declare INTERNAL scope."
            )
        return value

    def _invoke(self, spec: ToolSpec, args: dict, key: str | None) -> Any:
        kwargs = dict(args)
        if spec.wants_key:
            kwargs["idempotency_key"] = key
        if spec.wants_workspace:
            kwargs["workspace"] = self.workspace
        return spec.fn(**kwargs)


# --------------------------------------------------------------------------
# agent-facing surface
# --------------------------------------------------------------------------
@dataclass
class StepResult:
    done: bool = False
    result: Any = None


@dataclass
class StepContext:
    run_id: str
    step: int
    workspace: Path
    entropy: Entropy
    mode: Mode
    rec: Recorder

    @property
    def replaying(self) -> bool:
        """True while this step's outputs are coming from the log."""
        return not self.rec.live

    def sample(self, prompt: str, **params: Any) -> Completion:
        return self.rec.sample(prompt, **params)

    def call(self, tool: str, **args: Any) -> Any:
        return self.rec.call(tool, args)

    def note(self, **kv: Any) -> None:
        """Annotate the log. Suppressed during replay, where it would be noise."""
        if self.rec.live:
            self.rec.log.append(RecordKind.NOTE, {"event": "agent", **kv}, step=self.step)

    def request_snapshot(self) -> None:
        self.rec.request_snapshot()

    def path(self, *parts: str) -> Path:
        return self.workspace.joinpath(*parts)

    def done(self, result: Any = None) -> StepResult:
        return StepResult(True, result)


class Agent(Protocol):
    def step(self, ctx: StepContext) -> StepResult | None: ...


@dataclass
class RunOutcome:
    run_id: str
    status: str
    result: Any = None
    steps: int = 0
    mode: str = Mode.LIVE.value
    replayed_steps: int = 0
    replayed_samples: int = 0
    model_calls: int = 0
    divergences: int = 0
    snapshots: int = 0
    parent_run_id: str | None = None
    forked_at_step: int | None = None
    error: str | None = None
    workspace: str | None = None
    repaired_bytes: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "completed"


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
class AgentRunner:
    """Drives an agent while keeping the log and snapshots consistent.

    ``agent_factory`` must build a *fresh* agent. Resume and fork rebuild the
    agent's in-memory state by replay, so handing over a warm object would
    silently mix pre-crash state with replayed state.
    """

    def __init__(self, *, agent_factory: Callable[[], Agent], store: RunStore,
                 workspace: str | Path, model: ModelClient,
                 tools: ToolRegistry | None = None,
                 snapshotter_factory: Callable[[Path], Snapshotter] | None = None,
                 snapshot_every: int = 5, max_steps: int = 500,
                 max_seconds: float | None = None,
                 divergence: Divergence = Divergence.STRICT,
                 effects: EffectPolicy = EffectPolicy.ALLOW,
                 fork_effects: EffectPolicy = EffectPolicy.BLOCK,
                 undecidable: Undecidable = Undecidable.FAIL,
                 sync: str = "always", store_prompts: bool = False,
                 prompt_preview: int = 240, seed: int | None = None,
                 on_error: str = "raise", agent_name: str | None = None):
        self.agent_factory = agent_factory
        self.store = store
        self.workspace = Path(workspace)
        self.model = model
        self.tools = tools if tools is not None else ToolRegistry()
        self.snapshotter_factory = snapshotter_factory or self._default_snapshotter
        self.snapshot_every = snapshot_every
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.divergence = Divergence(divergence)
        self.effects = EffectPolicy(effects)
        self.fork_effects = EffectPolicy(fork_effects)
        self.undecidable = Undecidable(undecidable)
        self.sync = sync
        self.store_prompts = store_prompts
        self.prompt_preview = prompt_preview
        self.seed = seed
        self.on_error = on_error
        self.agent_name = agent_name or getattr(agent_factory, "__name__", "agent")

    def _default_snapshotter(self, root: Path) -> Snapshotter:
        exclude = list(DirSnapshotter(root).exclude)
        try:  # never snapshot the store that describes the snapshot
            exclude.append(self.store.root.resolve().relative_to(root.resolve()).as_posix())
        except ValueError:
            pass
        return DirSnapshotter(root, exclude=exclude)

    # -- public entry points ------------------------------------------
    def start(self, *, tags: dict | None = None, run_id: str | None = None) -> RunOutcome:
        run_id = run_id or new_run_id()
        ws = self.workspace
        ws.mkdir(parents=True, exist_ok=True)
        meta = RunMeta(run_id=run_id, agent=self.agent_name, mode=Mode.LIVE.value,
                       workspace=str(ws), tags=tags or {})
        self.store.create_run(meta)
        log = WriteAheadLog(self.store.log_path(run_id), sync=self.sync)
        log.append(RecordKind.RUN_START, {
            "run_id": run_id, "agent": self.agent_name, "workspace": str(ws),
            "snapshot_every": self.snapshot_every, "tools": self.tools.names,
        })
        snapper = self.snapshotter_factory(ws)
        # A baseline snapshot at step 0 means every later step has *some*
        # restore point, so recovery never has to replay against a workspace
        # whose starting state was never captured.
        self._snapshot(log, run_id, 0, snapper)
        return self._loop(run_id=run_id, meta=meta, log=log, tape=Tape([]), start_step=0,
                          mode=Mode.LIVE, workspace=ws, snapshotter=snapper,
                          effect_policy=self.effects, model=self.model)

    def resume(self, run_id: str, *, model: ModelClient | None = None,
               workspace: str | Path | None = None) -> RunOutcome:
        meta = self.store.load_meta(run_id)
        if meta.status == "completed":
            raise ValueError(f"run {run_id} already completed; fork it instead of resuming")
        log = WriteAheadLog(self.store.log_path(run_id), sync=self.sync)
        records = log.records()
        ws = Path(workspace or meta.workspace or self.workspace)
        snapper = self.snapshotter_factory(ws)
        ref = self._restore(self.store.lineage_ids(run_id), log.tail_seq, snapper)
        log.append(RecordKind.NOTE, {
            "event": "resume", "from_snapshot": ref.snapshot_id if ref else None,
            "at_seq": ref.at_seq if ref else 0, "at_step": ref.step if ref else 0,
            "tail_seq": log.tail_seq, "repaired_bytes": log.repaired_bytes,
        })
        meta.status = "running"
        meta.mode = Mode.RESUME.value
        self.store.save_meta(meta)
        # Replay starts at step 1, not at the snapshot's step: the agent's
        # in-memory state (message history, counters) exists only as a
        # consequence of running its code, so the covered prefix has to be
        # re-run too -- with its environment effects served from the log rather
        # than re-executed, since the snapshot already holds them.
        outcome = self._loop(
            run_id=run_id, meta=meta, log=log, tape=Tape(records), start_step=0,
            mode=Mode.RESUME, workspace=ws, snapshotter=snapper,
            effect_policy=self.effects, model=model or self.model,
            effects_through_seq=ref.at_seq if ref else 0,
        )
        outcome.repaired_bytes = log.repaired_bytes
        return outcome

    def fork(self, run_id: str, after_step: int, *, workspace: str | Path | None = None,
             model: ModelClient | None = None, effects: EffectPolicy | None = None,
             tags: dict | None = None) -> RunOutcome:
        """Branch a new run that replays through ``after_step`` then samples fresh.

        ``after_step=0`` forks from the baseline environment with an empty tape.
        """
        parent = self.store.load_meta(run_id)
        parent_records = load(self.store.log_path(run_id)).records
        boundary = boundary_seq(parent_records, after_step)

        child_id = new_run_id("fork")
        ws = Path(workspace or self.workspace)
        ws.mkdir(parents=True, exist_ok=True)
        child_meta = RunMeta(
            run_id=child_id, agent=parent.agent, mode=Mode.FORK.value,
            parent_run_id=run_id, forked_at_step=after_step, forked_at_seq=boundary,
            workspace=str(ws), tags=tags or {},
        )
        self.store.create_run(child_meta)

        WriteAheadLog.copy_prefix(self.store.log_path(run_id),
                                  self.store.log_path(child_id), boundary)
        log = WriteAheadLog(self.store.log_path(child_id), sync=self.sync)
        log.append(RecordKind.FORK, {
            "parent_run_id": run_id, "child_run_id": child_id, "at_step": after_step,
            "at_seq": boundary, "parent_hash": log.last_hash,
        })

        snapper = self.snapshotter_factory(ws)
        ref = self._restore(self.store.lineage_ids(child_id), boundary, snapper)
        return self._loop(
            run_id=child_id, meta=child_meta, log=log,
            tape=Tape(parent_records, until_seq=boundary), start_step=0, mode=Mode.FORK,
            workspace=ws, snapshotter=snapper,
            effect_policy=EffectPolicy(effects) if effects else self.fork_effects,
            model=model or self.model, effects_through_seq=ref.at_seq if ref else 0,
        )

    # -- internals ----------------------------------------------------
    def _restore(self, lineage: list[str], upto_seq: int,
                 snapshotter: Snapshotter) -> SnapshotRef | None:
        ref = self.store.snapshots.latest_at_or_before(lineage, upto_seq)
        if ref is None:
            return None
        _, manifest = self.store.snapshots.load(ref.snapshot_id)
        snapshotter.restore(manifest, self.store.cas)
        return ref

    def _snapshot(self, log: WriteAheadLog, run_id: str, step: int,
                  snapshotter: Snapshotter) -> SnapshotRef | None:
        if snapshotter is None or not snapshotter.available():
            return None
        at_seq = log.tail_seq
        manifest = snapshotter.capture(self.store.cas)
        ref = self.store.snapshots.save(run_id, step, at_seq, manifest)
        # Log *after* committing: a crash in between leaves a valid orphan
        # snapshot, which is recoverable. The reverse order would leave the log
        # pointing at something that does not exist.
        log.append(RecordKind.SNAPSHOT, {
            "snapshot_id": ref.snapshot_id, "at_seq": at_seq, "backend": ref.backend,
            "bytes": ref.bytes,
        }, step=step)
        return ref

    def _loop(self, *, run_id: str, meta: RunMeta, log: WriteAheadLog, tape: Tape,
              start_step: int, mode: Mode, workspace: Path, snapshotter: Snapshotter,
              effect_policy: EffectPolicy, model: ModelClient,
              effects_through_seq: int = 0) -> RunOutcome:
        agent = self.agent_factory()
        if meta.agent in ("<lambda>", "agent", ""):  # a factory lambda tells us nothing
            meta.agent = type(agent).__name__
        rec = Recorder(log=log, tape=tape, model=model, tools=self.tools, mode=mode,
                       divergence=self.divergence, effect_policy=effect_policy,
                       undecidable=self.undecidable, workspace=workspace,
                       effects_through_seq=effects_through_seq,
                       store_prompts=self.store_prompts, prompt_preview=self.prompt_preview)
        entropy = Entropy(rec.draw, seed=self.seed)
        step = start_step
        status, result, error = "completed", None, None
        snapshots = 0
        last_snapshot_step = start_step
        t0 = time.monotonic()
        try:
            while True:
                if step >= self.max_steps:
                    status = "max_steps"
                    break
                if self.max_seconds and rec.live and time.monotonic() - t0 > self.max_seconds:
                    status = "budget_exhausted"
                    break
                step += 1
                rec.begin_step(step)
                ctx = StepContext(run_id=run_id, step=step, workspace=workspace,
                                  entropy=entropy, mode=mode, rec=rec)
                outcome = agent.step(ctx) or StepResult()
                rec.end_step(step, outcome.done)
                if outcome.done:
                    result = outcome.result
                    break
                if rec.live and self._due(step, rec):
                    if self._snapshot(log, run_id, step, snapshotter):
                        snapshots += 1
                        last_snapshot_step = step
        except BaseException as e:
            status = "failed"
            error = f"{type(e).__name__}: {e}"
            # Snapshot the wreckage: failure attribution wants the state the
            # run died in, and it is unrecoverable once the process exits.
            try:
                if rec.live:
                    self._snapshot(log, run_id, step, snapshotter)
            except Exception:  # noqa: BLE001 - never mask the original failure
                pass
            self._finish(run_id, meta, log, rec, status, None, step, error)
            if self.on_error == "raise":
                raise
            return self._outcome(run_id, meta, rec, status, None, step, mode, snapshots, error,
                                 workspace)
        else:
            # Snapshot the terminal state so a finished run can still be forked
            # from its last step and its environment inspected later.
            if rec.live and last_snapshot_step != step:
                if self._snapshot(log, run_id, step, snapshotter):
                    snapshots += 1
            self._finish(run_id, meta, log, rec, status, result, step, None)
            return self._outcome(run_id, meta, rec, status, result, step, mode, snapshots, None,
                                 workspace)

    def _due(self, step: int, rec: Recorder) -> bool:
        requested = rec.consume_snapshot_request()  # always consume, never short-circuit
        return requested or bool(self.snapshot_every) and step % self.snapshot_every == 0

    def _finish(self, run_id: str, meta: RunMeta, log: WriteAheadLog, rec: Recorder,
                status: str, result: Any, step: int, error: str | None) -> None:
        log.append(RecordKind.RUN_END, {
            "status": status, "result": _loggable(result)[0], "steps": step, "error": error,
            "replayed_steps": rec.replayed_steps, "model_calls": rec.model_calls,
        }, step=step)
        log.close()
        meta.status = status
        meta.steps = step
        meta.replayed_steps = rec.replayed_steps
        meta.result = _loggable(result)[0]
        meta.error = error
        meta.finished_at = time.time()
        self.store.save_meta(meta)

    def _outcome(self, run_id: str, meta: RunMeta, rec: Recorder, status: str, result: Any,
                 step: int, mode: Mode, snapshots: int, error: str | None,
                 workspace: Path) -> RunOutcome:
        return RunOutcome(
            run_id=run_id, status=status, result=result, steps=step, mode=mode.value,
            replayed_steps=rec.replayed_steps, replayed_samples=rec.replayed_samples,
            model_calls=rec.model_calls, divergences=rec.divergences, snapshots=snapshots,
            parent_run_id=meta.parent_run_id, forked_at_step=meta.forked_at_step,
            error=error, workspace=str(workspace),
        )


def boundary_seq(records: list[Record], after_step: int) -> int:
    """The log sequence number at which a fork of ``after_step`` truncates."""
    if after_step <= 0:
        boundary = 0
        for rec in records:
            if rec.kind == RecordKind.STEP_START:
                break
            boundary = rec.seq
        return boundary
    # Last match, not first: a resumed run whose replay diverged can carry two
    # step_end records for the same step, and the later one is the real end.
    seqs = [r.seq for r in records
            if r.kind == RecordKind.STEP_END and r.step == after_step]
    if seqs:
        return seqs[-1]
    completed = sorted({r.step for r in records if r.kind == RecordKind.STEP_END})
    raise ValueError(f"no completed step {after_step} in this run; completed steps: {completed}")
