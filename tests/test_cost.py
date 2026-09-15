"""The cadence model, and the measurement it depends on."""

import math

import pytest

from helpers import build, crash_at, records_of, seq_of
from ledger.cost import CadenceModel, LogProfile, recommend_k
from ledger.wal import RecordKind


# -- log volume ------------------------------------------------------------
def test_profile_breaks_a_real_log_down_by_kind(tmp_path):
    h = build(tmp_path)
    out = h.runner.start()

    profile = LogProfile.from_log(h.store.log_path(out.run_id))

    assert profile.steps == 6
    assert profile.bytes_per_step > 0
    assert profile.bytes == sum(s.bytes for s in profile.by_kind.values())
    assert {RecordKind.PROMPT, RecordKind.SAMPLE} <= set(profile.by_kind)
    # A linear projection is exactly that -- the same shape, ten times longer.
    assert profile.project(60) == pytest.approx(profile.bytes * 10, rel=0.02)


def test_storing_prompts_is_the_big_lever_on_log_size(tmp_path):
    lean = build(tmp_path / "lean")
    fat = build(tmp_path / "fat", store_prompts=True)
    a = lean.runner.start()
    b = fat.runner.start()

    hashed = LogProfile.from_log(lean.store.log_path(a.run_id))
    stored = LogProfile.from_log(fat.store.log_path(b.run_id))

    assert stored.bytes_per_step > hashed.bytes_per_step
    assert stored.by_kind[RecordKind.PROMPT].bytes > hashed.by_kind[RecordKind.PROMPT].bytes


# -- the model -------------------------------------------------------------
def test_cadence_arithmetic():
    model = CadenceModel(steps=100, snapshot_seconds=0.5,
                         effect_replay_seconds_per_step=0.1, crashes=1.0)
    c = model.at(10)
    assert c.snapshots == 10
    assert c.snapshot_seconds == pytest.approx(5.0)
    assert c.expected_effect_replay_seconds == pytest.approx(0.5)   # (10/2)*0.1
    assert c.total_seconds == pytest.approx(5.5)

    with pytest.raises(ValueError):
        model.at(0)


def test_optimum_matches_the_closed_form_and_beats_its_neighbours():
    model = CadenceModel(steps=10_000, snapshot_seconds=0.4,
                         effect_replay_seconds_per_step=0.02, crashes=1.0)
    # The discrete optimum sits just below the closed form, never far from it:
    # the snapshot count is ceil(n/k), so there are treads to sit at the start of.
    assert model.continuous_optimal_k == pytest.approx(math.sqrt(2 * 10_000 * 0.4 / 0.02))
    assert 0.9 * model.continuous_optimal_k <= model.optimal_k <= model.continuous_optimal_k

    best = model.at(model.optimal_k).total_seconds
    assert all(model.at(k).total_seconds >= best - 1e-9
               for k in range(2, 3 * model.optimal_k))


def test_k_star_responds_to_each_constant_in_the_right_direction():
    base = dict(steps=5_000, snapshot_seconds=0.5,
                effect_replay_seconds_per_step=0.05, crashes=1.0)
    k0 = CadenceModel(**base).optimal_k

    assert CadenceModel(**{**base, "snapshot_seconds": 2.0}).optimal_k > k0   # dearer snaps
    assert CadenceModel(**{**base, "effect_replay_seconds_per_step": 0.5}).optimal_k < k0
    assert CadenceModel(**{**base, "crashes": 10.0}).optimal_k < k0           # crash often
    # sqrt(n): ten times the run wants roughly three times the window.
    longer = CadenceModel(**{**base, "steps": 50_000}).optimal_k
    assert longer / k0 == pytest.approx(math.sqrt(10), rel=0.05)


def test_free_effects_mean_no_periodic_snapshots():
    """If nothing has to be re-executed, a mid-run snapshot buys nothing."""
    assert recommend_k(500, snapshot_seconds=0.1,
                       effect_replay_seconds_per_step=0.0) == 500
    assert recommend_k(500, snapshot_seconds=0.1,
                       effect_replay_seconds_per_step=0.01, crashes=0.0) == 500


def test_using_total_replay_time_understates_k():
    """The mistake the benchmark caught, pinned so it cannot creep back.

    In-memory replay is k-independent; feeding it to the model as if k could
    reduce it recommends far more snapshots than are worth taking.
    """
    honest = recommend_k(5_000, snapshot_seconds=0.02,
                         effect_replay_seconds_per_step=0.00002)
    naive = recommend_k(5_000, snapshot_seconds=0.02,
                        effect_replay_seconds_per_step=0.0006)
    assert naive < honest / 4


# -- the measurement the model consumes ------------------------------------
def test_only_steps_above_the_snapshot_line_count_as_replayed_effects(tmp_path):
    h = build(tmp_path)          # snapshot_every=2 -> snapshots at 0,2,4,6
    out = h.runner.start()
    crash_at(h.store, out.run_id,
             seq_of(records_of(h.store, out.run_id), RecordKind.STEP_END, step=3))

    resumed = h.runner.resume(out.run_id)

    # Restored at step 2, replayed steps 1-3: only step 3's effect re-executes.
    assert resumed.replayed_steps == 3
    assert resumed.replayed_effects == 1
    assert resumed.effect_replay_seconds >= 0.0


def test_a_fresh_snapshot_leaves_no_effects_to_replay(tmp_path):
    h = build(tmp_path)
    out = h.runner.start()
    crash_at(h.store, out.run_id,
             seq_of(records_of(h.store, out.run_id), RecordKind.STEP_END, step=4))

    resumed = h.runner.resume(out.run_id)

    assert resumed.replayed_steps == 4       # the floor: still replays every step
    assert resumed.replayed_effects == 0     # but re-executes nothing
