"""Recorded nondeterminism.

The log captures model samples and tool results, which is what everyone
remembers to capture. It is not sufficient. An agent that calls ``time.time()``
to stamp a filename, or ``random.choice`` to pick a retry target, has pulled a
value out of thin air that replay cannot reproduce -- and the divergence shows
up later as a mysteriously wrong file path, not as an error.

So every draw goes through this facade and lands in the log as an ``entropy``
record. On replay the recorded value is handed back. This is the same trick a
deterministic simulator or a record/replay debugger uses, and it is the reason
``INTERNAL`` tools can be safely re-executed: given the restored filesystem and
the recorded entropy stream, they are pure functions.

Uncovered, and honestly so: nondeterminism inside a tool's own implementation
(thread scheduling, dict iteration over pointer addresses, a clock read in a
subprocess). Route what you can through here; snapshot more often for the rest.
"""

from __future__ import annotations

import random
import time
import uuid
from typing import Any, Callable, Sequence

DrawFn = Callable[[str, Callable[[], Any], dict | None], Any]


class Entropy:
    """Nondeterministic primitives, recorded on the way out and replayed back in.

    ``seed`` seeds only the *live* RNG. It is not what makes replay
    deterministic -- the log is. A seed would not help anyway, because the
    number of draws before a crash is itself nondeterministic.
    """

    def __init__(self, draw: DrawFn, *, seed: int | None = None):
        self._draw = draw
        self._rng = random.Random(seed)

    # -- clock ---------------------------------------------------------
    def time(self) -> float:
        return self._draw("time", time.time, None)

    def monotonic(self) -> float:
        return self._draw("monotonic", time.monotonic, None)

    def now_iso(self) -> str:
        return self._draw(
            "now_iso", lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), None
        )

    # -- randomness ----------------------------------------------------
    def random(self) -> float:
        return self._draw("random", self._rng.random, None)

    def randint(self, a: int, b: int) -> int:
        return self._draw("randint", lambda: self._rng.randint(a, b), {"a": a, "b": b})

    def choice(self, seq: Sequence[Any]) -> Any:
        """Records the *index*, not the item -- items need not be serializable."""
        if not seq:
            raise IndexError("choice from empty sequence")
        i = self._draw("choice_index", lambda: self._rng.randrange(len(seq)), {"n": len(seq)})
        return seq[i]

    def shuffled(self, seq: Sequence[Any]) -> list[Any]:
        order = self._draw(
            "shuffle_order",
            lambda: self._rng.sample(range(len(seq)), len(seq)),
            {"n": len(seq)},
        )
        return [seq[i] for i in order]

    # -- identity ------------------------------------------------------
    def uuid4(self) -> str:
        return self._draw("uuid4", lambda: str(uuid.uuid4()), None)

    def token(self, nbytes: int = 8) -> str:
        return self._draw(
            "token", lambda: "%0*x" % (nbytes * 2, self._rng.getrandbits(nbytes * 8)),
            {"nbytes": nbytes},
        )

    # -- escape hatch --------------------------------------------------
    def external(self, dkind: str, thunk: Callable[[], Any], **extra: Any) -> Any:
        """Record an arbitrary nondeterministic read (env var, API clock, ...).

        The value must be JSON-serializable, since replay hands back exactly
        what was logged.
        """
        return self._draw(dkind, thunk, extra or None)
