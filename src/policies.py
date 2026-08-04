from __future__ import annotations

import numpy as np

from config import HORIZON, MULTIPLIERS, N_OCC_BINS, NAIVE_ACTION


def fixed_policy(action: int):
    action = int(action)
    return lambda state, env: action


def naive_policy():
    return fixed_policy(NAIVE_ACTION)


def random_policy(seed: int = 0):
    rng = np.random.default_rng(seed)
    return lambda state, env: int(rng.integers(len(MULTIPLIERS)))


def greedy_q_policy(q_table: np.ndarray):
    return lambda state, env: int(np.argmax(q_table[state[0], state[1], state[2]]))


def lp_policy(schedule: np.ndarray):
    schedule = np.asarray(schedule, dtype=int)

    def act(state, env):
        week = min(max(HORIZON - state[0], 0), len(schedule) - 1)
        return int(schedule[week])

    return act


def naive_pace_curve(env, n: int = 2000, seed: int = 7) -> np.ndarray:
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


def pace_heuristic(
    target_curve: np.ndarray,
    band: float = 0.06,
    low: int = 0,
    mid: int = NAIVE_ACTION,
    high: int = len(MULTIPLIERS) - 1,
):
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
