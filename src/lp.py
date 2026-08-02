"""Mixed-integer linear program for the seasonal rate schedule.

This is the second decision-making technique required by the brief (Integer
Programming). It answers a different question from the MDP: *given expected
demand, what is the best open-loop schedule of weekly rates?*

Formulation
-----------
Binary ``x[w][k]`` selects price level ``k`` in week ``w``; continuous
``y[w][k] >= 0`` is the number of rooms sold at that level.

    maximise    sum_wk (r[w][k] + c * rate[k]) * y[w][k]
                - sum_wk c * rate[k] * d[w][k] * x[w][k]

    subject to  sum_k x[w][k] = 1                for every week w
                y[w][k] <= d[w][k] * x[w][k]     sales cannot exceed demand,
                                                 and are zero at unchosen prices
                sum_wk y[w][k] <= capacity       inventory
                x in {0,1}, y >= 0

``r[w][k] = rate[k] * L * (1 - p_cancel[w])`` is cancellation-discounted revenue
per room and ``d[w][k]`` is expected demand averaged over the stationary regime
distribution. The objective is the linear rearrangement of the MDP's reward:
unsold demand ``d * x - y`` is turned away and charged at the same denied-service
rate ``c``, so the MILP optimises the same economics as the RL agent.

Relationship to the MDP
-----------------------
The MILP replaces the Poisson arrivals with their means and commits to the whole
schedule up front. It therefore has *more* information than any policy can have
(it knows demand exactly) and *less* flexibility (it cannot react to how the
season unfolds). Its objective is consequently an optimistic estimate rather
than an attainable target -- ``mdp.value_iteration`` supplies the true optimum.
"""

from __future__ import annotations

import numpy as np
import pulp

from config import AVG_NIGHTS, HORIZON, MULTIPLIERS, N_REGIMES, OVERFLOW_PENALTY
from environment import HotelPricingEnv


def expected_demand_matrix(env) -> np.ndarray:
    """``d[w][k]``: expected arrivals, averaged over the stationary regime law."""
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
    """Solve the rate-schedule MILP with CBC.

    Returns ``(schedule, objective, env)`` where ``schedule[w]`` is the chosen
    action index for week ``w``.
    """
    if env is None:
        env = HotelPricingEnv(hotel, month, seed=0, **env_kwargs)

    n_weeks, n_levels = HORIZON, env.n_actions
    d = expected_demand_matrix(env)
    rate = env.ref_price * MULTIPLIERS
    r = np.array(
        [
            [rate[k] * AVG_NIGHTS * (1.0 - env.weekly_pcancel[w]) for k in range(n_levels)]
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
        pulp.lpSum(y[w][k] for w in range(n_weeks) for k in range(n_levels)) <= env.capacity,
        "inventory",
    )

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"MILP did not solve to optimality: {pulp.LpStatus[prob.status]}")

    schedule = np.array(
        [int(np.argmax([pulp.value(x[w][k]) for k in range(n_levels)])) for w in range(n_weeks)]
    )
    return schedule, float(pulp.value(prob.objective)), env


def capacity_sensitivity(env, factors=(0.8, 0.9, 1.0, 1.1, 1.2)) -> list[dict]:
    """Re-solve at scaled capacities to trace the shadow price of inventory.

    Binary variables make the LP duals unavailable directly, so the marginal
    value of a room is obtained numerically as the finite difference of the
    optimal objective with respect to capacity.
    """
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
