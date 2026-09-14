"""Tool declarations and effect policy -- where replay stops being a log problem.

Replaying a tool call is only safe if you know what the call *did*. LEDGER
makes authors declare that along two independent axes, because collapsing them
is the bug that makes naive replay send the email twice.

**Reversibility** (``ToolKind``) -- can this be run again at all?
  ``PURE`` no effect worth worrying about; ``IDEMPOTENT`` safe to repeat given
  the same key; ``IRREVERSIBLE`` running it twice is a real-world incident.

**Scope** (``EffectScope``) -- *where* does the effect live?
  ``INTERNAL`` inside the snapshotted environment (writes a file, runs a
  build); ``EXTERNAL`` outside it (sends mail, charges a card, posts to an
  API); ``MIXED`` both.

Scope, not reversibility, decides replay behaviour:

===========  ==========================================================
scope        what replay does
===========  ==========================================================
INTERNAL     re-executes the call, then verifies the result against the
             log. This is mandatory, not an optimisation: the snapshot
             is up to k steps stale, and re-executing internal tools is
             the only thing that rebuilds the environment delta between
             the snapshot and the crash.
EXTERNAL     serves the logged result, never executes. This *is* the
             outbound quarantine -- it needs no special replay flag,
             because it falls out of the declaration.
MIXED        serves the logged result (the external half must not fire)
             and therefore loses the internal half, so LEDGER requires
             ``snapshot_after=True`` and refuses to register the tool
             otherwise. Prefer splitting the tool in two.
===========  ==========================================================
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator


class ToolKind(str, Enum):
    PURE = "pure"
    IDEMPOTENT = "idempotent"
    IRREVERSIBLE = "irreversible"


class EffectScope(str, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"
    MIXED = "mixed"


class Divergence(str, Enum):
    """What to do when replay does not match the log."""

    STRICT = "strict"
    """Raise. The default: a mismatch means the replay is a different run."""
    WARN = "warn"
    """Note it and keep replaying the recorded values. For forensics."""
    RESAMPLE = "resample"
    """Abandon the tape and go live from here -- a fork on divergence.
    The honest way to resume a run whose agent code has since changed."""


class EffectPolicy(str, Enum):
    """Whether irreversible tools may fire in a *live* segment."""

    ALLOW = "allow"
    BLOCK = "block"
    """Raise ``QuarantineError``. The default for forks: a counterfactual
    branch must not mail your customers a second time."""
    DRY_RUN = "dry_run"
    """Skip the call, log it, return ``spec.dry_run_result``."""


class Undecidable(str, Enum):
    """A ``tool_call`` with no ``tool_result``: the crash landed inside the call.

    Whether the effect took place is not knowable from the log. Any automatic
    choice is a guess, so the choice is explicit.
    """

    FAIL = "fail"
    RECONCILE = "reconcile"
    """Ask the downstream system via ``spec.reconcile(key)``. The only answer
    that is actually correct, and it requires the tool author to provide a
    read-back path."""
    RETRY = "retry"
    ASSUME_APPLIED = "assume_applied"


class ToolError(RuntimeError):
    """A tool raised, live or on replay."""


class QuarantineError(RuntimeError):
    """An irreversible effect was blocked by policy."""


class UndecidableEffect(RuntimeError):
    """Recovery cannot determine whether an irreversible effect landed."""


class ToolRegistryError(ValueError):
    pass


_MISSING = object()


@dataclass
class ToolSpec:
    name: str
    fn: Callable[..., Any]
    kind: ToolKind = ToolKind.PURE
    scope: EffectScope = EffectScope.INTERNAL
    idempotency_key: Callable[[dict], str] | None = None
    reconcile: Callable[[str | None], tuple[bool, Any]] | None = None
    """``key -> (already_applied, result)``. Queries the downstream system."""
    snapshot_after: bool = False
    dry_run_result: Any = None
    assumed_result: Any = _MISSING
    doc: str = ""
    wants_key: bool = field(init=False, default=False)
    wants_workspace: bool = field(init=False, default=False)
    """Set when the tool declares a ``workspace`` parameter.

    A fork runs against its own copy of the environment, so a tool that hard-codes
    a directory captured at registration time would quietly write into the parent's
    workspace. Declaring the parameter gets the *active* one injected per call.
    """

    def __post_init__(self) -> None:
        self.kind = ToolKind(self.kind)
        self.scope = EffectScope(self.scope)
        if self.scope is EffectScope.MIXED and not self.snapshot_after:
            raise ToolRegistryError(
                f"tool {self.name!r} is MIXED scope: replay cannot re-run it (external half) "
                "nor skip it (internal half). Pass snapshot_after=True so the environment "
                "delta is captured, or split it into an INTERNAL and an EXTERNAL tool."
            )
        if self.kind is ToolKind.IDEMPOTENT and self.idempotency_key is None:
            raise ToolRegistryError(
                f"tool {self.name!r} is IDEMPOTENT but supplies no idempotency_key(args); "
                "without a key the claim is unenforceable on retry"
            )
        if self.kind is ToolKind.IRREVERSIBLE and self.scope is EffectScope.INTERNAL:
            raise ToolRegistryError(
                f"tool {self.name!r} is IRREVERSIBLE with INTERNAL scope, which replay would "
                "re-execute. If the effect really is inside the snapshot it is not "
                "irreversible; if it is outside, declare EXTERNAL."
            )
        try:
            params = inspect.signature(self.fn).parameters
        except (TypeError, ValueError):
            params = {}
        self.wants_key = "idempotency_key" in params
        self.wants_workspace = "workspace" in params

    @property
    def replay_executes(self) -> bool:
        return self.scope is EffectScope.INTERNAL


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, name: str, fn: Callable[..., Any], **kw: Any) -> ToolSpec:
        if name in self._tools:
            raise ToolRegistryError(f"tool {name!r} already registered")
        spec = ToolSpec(name=name, fn=fn, doc=(fn.__doc__ or "").strip(), **kw)
        self._tools[name] = spec
        return spec

    def tool(self, name: str | None = None, **kw: Any) -> Callable:
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(name or fn.__name__, fn, **kw)
            return fn

        return deco

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolRegistryError(
                f"unknown tool {name!r}; registered: {sorted(self._tools)}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)
