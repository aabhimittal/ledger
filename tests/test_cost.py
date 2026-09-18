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


def test_using_total_replay_time_errs_in_both_directions():
    """The mistake the benchmark caught, pinned so it cannot creep back.

    In-memory replay is k-independent, so feeding total per-step replay time to
    the model is wrong -- and which way it is wrong depends on which cost
    dominates the resume. Both regimes are measured in docs/DESIGN.md.
    """
    # Cheap effects: the floor dominates the resume, so the naive figure
    # overstates C_e and buys snapshots that save nothing.
    honest_cheap = recommend_k(5_000, 0.02, 0.00002)      # C_e = 0.02 ms
    naive_cheap = recommend_k(5_000, 0.02, 0.0006)        # total/step = 0.6 ms
    assert naive_cheap < honest_cheap / 4

    # Expensive effects: effect replay is most of the resume but is amortised
    # over every replayed step, so the naive figure understates C_e instead.
    honest_dear = recommend_k(5_000, 0.029, 0.027)        # C_e = 27 ms
    naive_dear = recommend_k(5_000, 0.029, 0.009)         # total/step = 9 ms
    assert naive_dear > honest_dear * 1.5


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


# -- estimating f ----------------------------------------------------------
def test_crash_rate_counts_resumes_not_forks(tmp_path):
    """Every recovery leaves a resume note; a fork is a choice, not a rescue."""
    from ledger.cost import estimate_crash_rate

    h = build(tmp_path)
    out = h.runner.start()
    assert estimate_crash_rate(h.store).crashes == 0     # finished cleanly

    crash_at(h.store, out.run_id,
             seq_of(records_of(h.store, out.run_id), RecordKind.STEP_END, step=3))
    h.runner.resume(out.run_id)
    # after_step=4 keeps the report inside the replayed prefix, so the branch
    # does not trip the outbound quarantine.
    h.runner.fork(out.run_id, after_step=4, workspace=tmp_path / "branch")

    rate = estimate_crash_rate(h.store)
    assert (rate.runs, rate.crashes) == (1, 1)           # the fork is excluded
    assert rate.rate == 1.0
    assert estimate_crash_rate(h.store, include_forks=True).runs == 2
    assert "crashes/run" in str(rate)


def test_crash_rate_interval_brackets_the_estimate(tmp_path):
    from ledger.cost import estimate_crash_rate

    h = build(tmp_path)
    for _ in range(3):
        h.runner.start()

    rate = estimate_crash_rate(h.store)
    assert rate.runs == 3 and rate.crashes == 0
    assert rate.rate == 0.0
    # Zero observed crashes does not mean a zero rate: the interval stays open
    # upwards, which is the honest reading of three quiet runs.
    assert rate.lo == 0.0 and rate.hi > 0.0


def test_no_history_says_so_rather_than_guessing(tmp_path):
    from ledger.cost import estimate_crash_rate
    from ledger import RunStore

    rate = estimate_crash_rate(RunStore(tmp_path / "empty"))
    assert not rate.observed
    assert "cannot be estimated" in str(rate)


def test_k_star_is_insensitive_to_a_wide_interval_on_f(tmp_path):
    """The practical answer to "we don't know f": it barely matters.

    k* scales as f^(-1/2), so a 100x uncertainty in the crash rate is only a 10x
    spread in k* -- and the total overhead near the optimum is flatter still.
    """
    from ledger.cost import CrashRate, k_range_for_rate

    wide = CrashRate(runs=1, crashes=1, lo=0.01, hi=5.5)
    small_k, mid_k, large_k = k_range_for_rate(5_000, 0.02, 0.00002, wide)

    assert small_k < mid_k < large_k          # high f -> small k, and vice versa
    assert (wide.hi / wide.lo) > 100
    assert (large_k / small_k) < 25           # 550x in f becomes <25x in k
