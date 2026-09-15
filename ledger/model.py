"""Model client boundary.

LEDGER never re-samples a replayed step. The only thing it needs from a model
client is a ``sample(prompt, **params) -> Completion | str`` method, so real
clients (Anthropic, OpenAI, a local server) drop in unchanged, and replay
substitutes the recorded text without the client being involved at all.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from .wal import canon, hash_text


@dataclass
class Completion:
    text: str
    finish_reason: str = "stop"
    usage: dict | None = None


@runtime_checkable
class ModelClient(Protocol):
    def sample(self, prompt: str, **params: Any) -> "Completion | str": ...


def normalize(out: "Completion | str") -> Completion:
    return out if isinstance(out, Completion) else Completion(str(out))


def params_hash(params: dict) -> str:
    """Hash sampling params, ignoring ones that cannot affect the output."""
    filtered = {k: v for k, v in sorted(params.items()) if k not in ("timeout", "metadata")}
    return hash_text(canon(filtered))


class ScriptedModel:
    """Deterministic test double: hands back a fixed sequence of replies."""

    def __init__(self, replies: Sequence[str], *, on_exhaust: str = "raise"):
        self.replies = list(replies)
        self.on_exhaust = on_exhaust
        self.calls: list[str] = []

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    def sample(self, prompt: str, **params: Any) -> Completion:
        i = len(self.calls)
        self.calls.append(prompt)
        if i < len(self.replies):
            return Completion(self.replies[i], usage={"input_chars": len(prompt)})
        if self.on_exhaust == "loop" and self.replies:
            return Completion(self.replies[i % len(self.replies)])
        raise RuntimeError(f"ScriptedModel exhausted after {len(self.replies)} replies")


class StochasticModel:
    """Test double that genuinely samples, so forks can diverge.

    Used to demonstrate the point of forking: restore at step k, sample again,
    get a different branch.
    """

    def __init__(self, options: Sequence[str], *, seed: int | None = None):
        self.options = list(options)
        self.rng = random.Random(seed)
        self.calls: list[str] = []

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    def sample(self, prompt: str, **params: Any) -> Completion:
        self.calls.append(prompt)
        return Completion(self.rng.choice(self.options))


class CallableModel:
    """Adapt a plain ``prompt -> str`` function into a model client."""

    def __init__(self, fn: Callable[[str], str]):
        self.fn = fn
        self.calls: list[str] = []

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    def sample(self, prompt: str, **params: Any) -> Completion:
        self.calls.append(prompt)
        return normalize(self.fn(prompt))
