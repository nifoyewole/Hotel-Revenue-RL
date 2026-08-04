from __future__ import annotations

import numpy as np
import pulp

from config import AVG_NIGHTS, HORIZON, MULTIPLIERS, N_REGIMES, OVERFLOW_PENALTY
from environment import HotelPricingEnv


def expected_demand_matrix(env) -> np.ndarray:
    from demand import stationary_distribution

    pi = (
        stationary_distribution(env.regime_transition)
        if env.use_regimes
        else np.eye(N_REGIMES)[1]
    )
    return np.array(
        [
            [
                sum(pi[g] * env.expected_demand(w, g, k) for g in range(N_REGIMES))
                for k in range(env.n_actions)
            ]
            for w in range(HORIZON)
        ]
    )


def solve_lp(hotel: str = "City Hotel", month: str = "May", env=None, **env_kwargs):
    if env is None:
        env = HotelPricingEnv(hotel, month, seed=0, **env_kwargs)

    n_weeks, n_levels = HORIZON, env.n_actions
    d = expected_demand_matrix(env)
    rate = env.ref_price * MULTIPLIERS
    r = np.array(
        [
            [
                rate[k] * AVG_NIGHTS * (1.0 - env.weekly_pcancel[w])
                for k in range(n_levels)
            ]
            for w in range(n_weeks)
        ]
    )

    prob = pulp.LpProblem("hotel_rate_schedule", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", (range(n_weeks), range(n_levels)), cat="Binary")
    y = pulp.LpVariable.dicts("y", (range(n_weeks), range(n_levels)), lowBound=0)

    prob += pulp.lpSum(
        (r[w][k] + OVERFLOW_PENALTY * rate[k]) * y[w][k]
        - OVERFLOW_PENALTY * rate[k] * d[w][k] * x[w][k]
        for w in range(n_weeks)
        for k in range(n_levels)
    )
    for w in range(n_weeks):
        prob += pulp.lpSum(x[w][k] for k in range(n_levels)) == 1, f"one_rate_week_{w}"
        for k in range(n_levels):
            prob += y[w][k] <= d[w][k] * x[w][k], f"sales_cap_{w}_{k}"
    prob += (
        pulp.lpSum(y[w][k] for w in range(n_weeks) for k in range(n_levels))
        <= env.capacity,
        "inventory",
    )

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(
            f"MILP did not solve to optimality: {pulp.LpStatus[prob.status]}"
        )

    schedule = np.array(
        [
            int(np.argmax([pulp.value(x[w][k]) for k in range(n_levels)]))
            for w in range(n_weeks)
        ]
    )
    return schedule, float(pulp.value(prob.objective)), env


def capacity_sensitivity(env, factors=(0.8, 0.9, 1.0, 1.1, 1.2)) -> list[dict]:
    import copy

    base_capacity = env.capacity
    rows = []
    for f in factors:
        scaled = copy.copy(env)
        scaled.capacity = max(1, int(base_capacity * f))
        _, obj, _ = solve_lp(env=scaled)
        rows.append({"factor": f, "capacity": scaled.capacity, "objective": obj})
    for i, row in enumerate(rows):
        if i == 0:
            row["marginal_revenue_per_room"] = float("nan")
        else:
            d_obj = row["objective"] - rows[i - 1]["objective"]
            d_cap = row["capacity"] - rows[i - 1]["capacity"]
            row["marginal_revenue_per_room"] = d_obj / d_cap if d_cap else float("nan")
    return rows
