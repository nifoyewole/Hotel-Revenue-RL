"""Pricing policies: the baselines the learned strategies are measured against.

Every policy is a callable ``fn(state, env) -> action_index``. Passing the
environment as well as the abstracted state lets model-based policies (the DP
optimum) read the exact inventory level, while the tabular agents and the
heuristics use only ``state``.

The baseline set spans the comparisons the brief asks for: no pricing decision
at all (``random``), fixed pricing (``naive``, ``best_fixed``), and a rule-based
industry heuristic (``pace``).
"""

from __future__ import annotations

import numpy as np

from config import HORIZON, MULTIPLIERS, N_OCC_BINS, NAIVE_ACTION


def fixed_policy(action: int):
    """Always post the same multiplier."""
    action = int(action)
    return lambda state, env: action


def naive_policy():
    """List price every week -- the "no dynamic adjustment" baseline."""
    return fixed_policy(NAIVE_ACTION)


def random_policy(seed: int = 0):
    """Uniformly random rate each week; a floor for any sensible strategy."""
    rng = np.random.default_rng(seed)
    return lambda state, env: int(rng.integers(len(MULTIPLIERS)))


def greedy_q_policy(q_table: np.ndarray):
    """Greedy with respect to a learned action-value table."""
    return lambda state, env: int(np.argmax(q_table[state[0], state[1], state[2]]))


def lp_policy(schedule: np.ndarray):
    """Replay the MILP's open-loop schedule; ignores how the season is going."""
    schedule = np.asarray(schedule, dtype=int)

    def act(state, env):
        week = min(max(HORIZON - state[0], 0), len(schedule) - 1)
        return int(schedule[week])

    return act


def naive_pace_curve(env, n: int = 2000, seed: int = 7) -> np.ndarray:
    """Mean occupancy by week under list pricing -- the heuristic's target pace.

    This is the "booking pace" a revenue manager compares against: where the
    hotel would normally be at each point in the booking window.
    """
    env.rng = np.random.default_rng(seed)
    naive = naive_policy()
    occupancy = np.zeros(HORIZON)
    counts = np.zeros(HORIZON)
    for _ in range(n):
        state = env.reset()
        done = False
        while not done:
            week = HORIZON - state[0]
            if week < HORIZON:
                occupancy[week] += env.sold / env.capacity
                counts[week] += 1
            state, _, done, _ = env.step(naive(state, env))
    return occupancy / np.maximum(counts, 1)


def pace_heuristic(target_curve: np.ndarray, band: float = 0.06,
                   low: int = 0, mid: int = NAIVE_ACTION,
                   high: int = len(MULTIPLIERS) - 1):
    """Rule-based pace management, the standard industry practice.

    Ahead of the target booking pace, push the rate up; behind it, discount;
    otherwise hold list price. Occupancy is reconstructed from the bin *midpoint*
    rather than its lower edge, which would otherwise understate occupancy by up
    to one full bin width and bias the rule toward discounting.
    """
    target_curve = np.asarray(target_curve, dtype=float)

    def act(state, env):
        weeks_remaining, occ_bin, _ = state
        week = min(HORIZON - weeks_remaining, len(target_curve) - 1)
        occupancy = (occ_bin + 0.5) / N_OCC_BINS
        if occupancy > target_curve[week] + band:
            return int(high)
        if occupancy < target_curve[week] - band:
            return int(low)
        return int(mid)

    return act
