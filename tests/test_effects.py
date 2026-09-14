"""The registry refuses declarations whose replay behaviour is undefined."""

import pytest

from ledger import EffectScope, ToolKind, ToolRegistry, ToolRegistryError


def noop(**kw):
    return None


def test_mixed_scope_requires_a_snapshot():
    tools = ToolRegistry()
    with pytest.raises(ToolRegistryError, match="snapshot_after"):
        tools.register("both", noop, scope=EffectScope.MIXED)
    tools.register("both", noop, scope=EffectScope.MIXED, snapshot_after=True)


def test_idempotent_without_a_key_is_an_empty_claim():
    with pytest.raises(ToolRegistryError, match="idempotency_key"):
        ToolRegistry().register("retryable", noop, kind=ToolKind.IDEMPOTENT)


def test_irreversible_internal_is_contradictory():
    with pytest.raises(ToolRegistryError, match="IRREVERSIBLE with INTERNAL"):
        ToolRegistry().register("wat", noop, kind=ToolKind.IRREVERSIBLE,
                                scope=EffectScope.INTERNAL)


def test_duplicate_registration_rejected():
    tools = ToolRegistry()
    tools.register("a", noop)
    with pytest.raises(ToolRegistryError, match="already registered"):
        tools.register("a", noop)


def test_unknown_tool_names_the_alternatives():
    tools = ToolRegistry()
    tools.register("read_file", noop)
    with pytest.raises(ToolRegistryError, match=r"read_file"):
        tools.get("raed_file")


def test_replay_executes_follows_scope():
    tools = ToolRegistry()
    internal = tools.register("w", noop, scope=EffectScope.INTERNAL)
    external = tools.register("m", noop, kind=ToolKind.IRREVERSIBLE, scope=EffectScope.EXTERNAL)
    assert internal.replay_executes is True
    assert external.replay_executes is False


def test_idempotency_key_injection_is_detected():
    tools = ToolRegistry()

    def with_key(x: int, idempotency_key: str | None = None):
        return x

    spec = tools.register("k", with_key, kind=ToolKind.IDEMPOTENT,
                          idempotency_key=lambda a: str(a["x"]))
    assert spec.wants_key is True
    assert tools.register("nk", noop).wants_key is False


def test_decorator_registration():
    tools = ToolRegistry()

    @tools.tool(scope=EffectScope.INTERNAL)
    def write_thing(path: str) -> str:
        """docstring is kept for introspection"""
        return path

    assert tools.names == ["write_thing"]
    assert "docstring" in tools.get("write_thing").doc
    assert len(tools) == 1
