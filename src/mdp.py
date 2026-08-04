from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import poisson

from config import AVG_NIGHTS, HORIZON, MULTIPLIERS, N_REGIMES, OVERFLOW_PENALTY


def _poisson_kernel(lam: float, capacity: int) -> np.ndarray:
    k = np.arange(capacity + 1)
    pmf = poisson.pmf(k, lam)
    pmf[-1] = max(0.0, 1.0 - pmf[:-1].sum())
    return pmf


def build_model(env) -> dict:
    capacity = env.capacity
    n_actions = env.n_actions
    sold = np.arange(capacity + 1)
    rooms_left = capacity - sold

    reward = np.zeros((HORIZON, capacity + 1, N_REGIMES, n_actions))
    kernels: list[list[list[np.ndarray]]] = []

    for w in range(HORIZON):
        kept = 1.0 - env.weekly_pcancel[w]
        per_regime = []
        for g in range(N_REGIMES):
            per_action = []
            for a in range(n_actions):
                lam = env.expected_demand(w, g, a)
                rate = env.ref_price * MULTIPLIERS[a]

                tail = poisson.sf(np.arange(capacity), lam)
                e_min = np.concatenate([[0.0], np.cumsum(tail)])
                e_sold = e_min[rooms_left]
                e_turned = np.maximum(lam - e_sold, 0.0)

                reward[w, :, g, a] = (
                    e_sold * rate * AVG_NIGHTS * kept
                    - OVERFLOW_PENALTY * e_turned * rate
                )
                per_action.append(_poisson_kernel(lam, capacity))
            per_regime.append(per_action)
        kernels.append(per_regime)

    reward[:, capacity, :, :] = 0.0
    return {
        "reward": reward,
        "kernels": kernels,
        "capacity": capacity,
        "n_actions": n_actions,
    }


def _continuation(u: np.ndarray, pmf: np.ndarray) -> np.ndarray:
    full = fftconvolve(u, pmf[::-1])
    return full[len(pmf) - 1 : len(pmf) - 1 + len(u)]


def _week_q(
    model: dict, transition: np.ndarray, v_next: np.ndarray, w: int
) -> np.ndarray:
    """Action values ``Q(w, n, g, a)`` given next week's value function."""
    capacity, n_actions = model["capacity"], model["n_actions"]
    reward, kernels = model["reward"], model["kernels"]

    u = v_next @ transition.T
    u[capacity, :] = 0.0

    q = np.empty((capacity + 1, N_REGIMES, n_actions))
    for g in range(N_REGIMES):
        for a in range(n_actions):
            q[:, g, a] = reward[w, :, g, a] + _continuation(u[:, g], kernels[w][g][a])
    q[capacity, :, :] = 0.0
    return q


def bellman_sweep(
    model: dict, transition: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    v_new = np.zeros_like(v)
    policy = np.zeros((HORIZON, model["capacity"] + 1, N_REGIMES), dtype=np.int8)

    for w in range(HORIZON):
        q = _week_q(model, transition, v[w + 1], w)
        v_new[w] = q.max(axis=2)
        policy[w] = q.argmax(axis=2)

    v_new[HORIZON] = 0.0
    return v_new, policy


def value_iteration(
    env, tol: float = 1e-6, max_sweeps: int = 100, verbose: bool = False
):
    model = build_model(env)
    transition = env.regime_transition
    v = np.zeros((HORIZON + 1, model["capacity"] + 1, N_REGIMES))

    residuals = []
    policy = None
    for sweep in range(max_sweeps):
        v_new, policy = bellman_sweep(model, transition, v)
        residual = float(np.abs(v_new - v).max())
        residuals.append(residual)
        v = v_new
        if verbose:
            print(f"sweep {sweep + 1:3d}  ||dV||_inf = {residual:,.4f}")
        if residual < tol:
            break
    return v, policy, residuals


def policy_evaluation(
    env, fine_policy: np.ndarray, model: dict | None = None
) -> np.ndarray:
    model = model or build_model(env)
    transition = env.regime_transition
    v = np.zeros((HORIZON + 1, model["capacity"] + 1, N_REGIMES))

    for w in range(HORIZON - 1, -1, -1):
        q = _week_q(model, transition, v[w + 1], w)
        chosen = fine_policy[w].astype(int)[:, :, None]
        v[w] = np.take_along_axis(q, chosen, axis=2)[:, :, 0]
    return v


def optimal_value(env, v: np.ndarray) -> float:
    from demand import stationary_distribution

    pi = (
        stationary_distribution(env.regime_transition)
        if env.use_regimes
        else np.eye(N_REGIMES)[1]
    )
    return float(v[0, 0] @ pi)


def dp_policy(policy: np.ndarray):
    def act(state, env):
        week = min(HORIZON - state[0], HORIZON - 1)
        return int(policy[week, min(env.sold, env.capacity), env.regime])

    return act
