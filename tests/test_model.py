"""Invariants that must hold for the results to mean anything.

These are the checks that would have caught the defects worth catching: a
cancellation curve indexed backwards, a transition kernel that disagrees with the
simulator it is supposed to describe, or a "baseline" that beats the proven
optimum. Run with ``pytest tests/`` after notebooks 01-02 have produced
``results/``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import HORIZON, MULTIPLIERS, N_OCC_BINS, N_REGIMES  # noqa: E402
from environment import HotelPricingEnv  # noqa: E402
import mdp  # noqa: E402
import policies as P  # noqa: E402
from metrics import compare, paired_tests  # noqa: E402


@pytest.fixture(scope="module")
def env():
    return HotelPricingEnv("City Hotel", "May", seed=0)


@pytest.fixture(scope="module")
def solution(env):
    v, policy, residuals = mdp.value_iteration(env)
    return v, policy, residuals


def test_cancellation_risk_falls_toward_arrival(env):
    """Week 0 is 11 weeks out and week 11 is the arrival week.

    Indexing the lead time the wrong way round inverts this curve and silently
    mis-weights the most valuable week of the season.
    """
    curve = env.weekly_pcancel
    assert curve[0] > curve[-1]
    assert curve[-1] < 0.2, "arrival-week bookings should rarely cancel"
    assert curve[0] > 0.3, "bookings made months ahead should often cancel"


def test_booking_curve_is_back_loaded(env):
    """Demand concentrates near arrival, so late inventory is scarce."""
    assert env.weekly_base[-1] > env.weekly_base[0]
    assert env.weekly_base.sum() > env.capacity, "capacity must bind at list price"


def test_regime_chain_is_stochastic(env):
    rows = env.regime_transition.sum(axis=1)
    assert np.allclose(rows, 1.0)
    assert (env.regime_transition >= 0).all()
    assert np.diag(env.regime_transition).mean() > 1 / N_REGIMES, "regimes should persist"


def test_state_is_always_in_bounds(env):
    """Every state the simulator emits must index the tabular agents safely."""
    env.set_common_randoms(None, None)
    for action in range(env.n_actions):
        state = env.reset()
        done = False
        while not done:
            w, o, g = state
            assert 0 <= w <= HORIZON
            assert 0 <= o < N_OCC_BINS
            assert 0 <= g < N_REGIMES
            state, _, done, _ = env.step(action)
        assert env.sold <= env.capacity, "cannot oversell inventory"


def test_value_iteration_converges_in_horizon_sweeps(solution):
    """A finite-horizon DAG MDP is exact after H sweeps -- no more, no fewer."""
    _, _, residuals = solution
    assert len(residuals) == HORIZON + 1
    assert residuals[-1] < 1e-6
    assert residuals[-2] > 1.0, "the residual should still be large at sweep H"


def test_kernel_matches_simulator(env, solution):
    """The analytic model and the sampled simulator must describe one MDP."""
    v, policy, _ = solution
    analytic = mdp.optimal_value(env, v)
    summary, _ = compare(env, {"dp": mdp.dp_policy(policy)}, n_episodes=1500, seed=777)
    simulated = summary.loc["dp", "net_revenue"]
    assert abs(simulated - analytic) < 3 * summary.loc["dp", "ci95"]


def test_no_policy_beats_the_optimum(env, solution):
    """V* is an upper bound; anything above it means the solver is wrong."""
    v, policy, _ = solution
    optimum = mdp.optimal_value(env, v)
    strategies = {
        "naive": P.naive_policy(),
        "random": P.random_policy(0),
        **{f"fixed{a}": P.fixed_policy(a) for a in range(len(MULTIPLIERS))},
    }
    summary, _ = compare(env, strategies, n_episodes=400, seed=99)
    assert (summary["net_revenue"] <= optimum + 3 * summary["ci95"]).all()


def test_common_random_numbers_are_reproducible(env):
    """The same randomness bundle must give the same episode, twice."""
    first, _ = compare(env, {"naive": P.naive_policy()}, n_episodes=200, seed=4242)
    second, _ = compare(env, {"naive": P.naive_policy()}, n_episodes=200, seed=4242)
    assert first.loc["naive", "net_revenue"] == pytest.approx(second.loc["naive", "net_revenue"])


def test_holm_correction_is_monotone_and_conservative(env):
    """Adjusted p-values must never decrease, and never fall below the raw one."""
    _, raw = compare(env, {
        "naive": P.naive_policy(),
        "random": P.random_policy(0),
        "high": P.fixed_policy(4),
        "low": P.fixed_policy(0),
    }, n_episodes=300, seed=11)
    tests = paired_tests(raw, "naive")
    assert (tests["p_holm"].to_numpy() >= tests["p_raw"].to_numpy() - 1e-12).all()
    assert (np.diff(tests["p_holm"].to_numpy()) >= -1e-12).all()
