"""Exact solution of the pricing MDP by Value Iteration.

The simulator in ``environment`` defines the dynamics by *sampling*; this module
writes the same dynamics down explicitly as a transition kernel and solves the
MDP to optimality. It serves three purposes:

1. it states ``P(s' | s, a)`` in closed form, as the formulation requires;
2. it produces the exact optimal value ``V*`` and policy ``pi*``, the correct
   benchmark against which the model-free learner is measured (the MILP in
   ``lp`` is a *deterministic-demand relaxation* and is a looser, clairvoyant
   bound rather than the true optimum);
3. it demonstrates the convergence property of Value Iteration on a
   finite-horizon problem.

Transition kernel
-----------------
The unabstracted state is ``(w, n, g)`` -- week, rooms sold, market regime. With
``K ~ Poisson(lambda(w, g, a))`` arrivals and capacity ``C``,

    n'  = min(n + K, C)                 and      g' ~ P_regime[g, .]

so, because arrivals and the regime evolve independently,

    P(n', g' | n, g, a) = P_regime[g, g'] * { pmf(n' - n)        n <= n' < C
                                            { P(K >= C - n)      n' = C

States with ``n = C`` are absorbing and earn nothing: the season stops once the
hotel is full. The expected one-step reward is

    R(w, n, g, a) = E[min(K, C - n)] * rate * L * (1 - p_cancel[w])
                    - c * (lambda - E[min(K, C - n)]) * rate

using ``E[min(K, L)] = sum_{j<L} P(K > j)``.

Convergence
-----------
The state graph is acyclic in ``w``, so with ``gamma = 1`` the Bellman operator
is a contraction in the horizon rather than in a discount factor: a synchronous
sweep makes one further week exact, and the residual ``||V_{k+1} - V_k||_inf``
reaches zero after exactly ``H`` sweeps. ``value_iteration`` records that
residual so the property can be shown empirically rather than merely asserted.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import poisson

from config import AVG_NIGHTS, HORIZON, MULTIPLIERS, N_REGIMES, OVERFLOW_PENALTY


def _poisson_kernel(lam: float, capacity: int) -> np.ndarray:
    """Arrival pmf truncated at ``capacity``.

    Arrivals beyond the remaining inventory are indistinguishable from exactly
    filling it, so the tail is folded into the last entry. This keeps every
    kernel at most ``capacity + 1`` long regardless of how large ``lambda`` is.
    """
    k = np.arange(capacity + 1)
    pmf = poisson.pmf(k, lam)
    pmf[-1] = max(0.0, 1.0 - pmf[:-1].sum())
    return pmf


def build_model(env) -> dict:
    """Tabulate rewards and arrival kernels for every ``(week, regime, action)``.

    Returns arrays used by the solvers:

    ``reward``  -- ``(H, C+1, R, A)`` expected one-step reward
    ``kernels`` -- list indexed ``[w][g][a]`` of truncated arrival pmfs
    """
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

                # E[min(K, L)] = sum_{j < L} P(K > j), evaluated for every L at once.
                tail = poisson.sf(np.arange(capacity), lam)
                e_min = np.concatenate([[0.0], np.cumsum(tail)])
                e_sold = e_min[rooms_left]
                e_turned = np.maximum(lam - e_sold, 0.0)

                reward[w, :, g, a] = (
                    e_sold * rate * AVG_NIGHTS * kept - OVERFLOW_PENALTY * e_turned * rate
                )
                per_action.append(_poisson_kernel(lam, capacity))
            per_regime.append(per_action)
        kernels.append(per_regime)

    # A full hotel is absorbing and earns nothing.
    reward[:, capacity, :, :] = 0.0
    return {"reward": reward, "kernels": kernels, "capacity": capacity, "n_actions": n_actions}


def _continuation(u: np.ndarray, pmf: np.ndarray) -> np.ndarray:
    """``C[n] = sum_k pmf[k] * U[min(n + k, C)]`` for every ``n``.

    ``U[C] = 0`` by the absorbing-terminal convention, so indices past the end of
    ``u`` contribute nothing and a plain correlation is exact. FFT convolution
    keeps this ``O(C log C)`` instead of ``O(C^2)``.
    """
    full = fftconvolve(u, pmf[::-1])
    return full[len(pmf) - 1 : len(pmf) - 1 + len(u)]


def _week_q(model: dict, transition: np.ndarray, v_next: np.ndarray, w: int) -> np.ndarray:
    """Action values ``Q(w, n, g, a)`` given next week's value function."""
    capacity, n_actions = model["capacity"], model["n_actions"]
    reward, kernels = model["reward"], model["kernels"]

    # u[n', g] = sum_g' P[g, g'] V(w+1, n', g'): the regime chain is independent
    # of the action, so it can be applied once before the arrival convolution.
    u = v_next @ transition.T
    u[capacity, :] = 0.0

    q = np.empty((capacity + 1, N_REGIMES, n_actions))
    for g in range(N_REGIMES):
        for a in range(n_actions):
            q[:, g, a] = reward[w, :, g, a] + _continuation(u[:, g], kernels[w][g][a])
    q[capacity, :, :] = 0.0
    return q


def bellman_sweep(model: dict, transition: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One synchronous sweep of the Bellman optimality operator.

    ``v`` has shape ``(H + 1, C + 1, R)``; ``v[H]`` is the terminal value and is
    held at zero. Returns the updated value array and the greedy action array.
    """
    v_new = np.zeros_like(v)
    policy = np.zeros((HORIZON, model["capacity"] + 1, N_REGIMES), dtype=np.int8)

    for w in range(HORIZON):
        q = _week_q(model, transition, v[w + 1], w)
        v_new[w] = q.max(axis=2)
        policy[w] = q.argmax(axis=2)

    v_new[HORIZON] = 0.0
    return v_new, policy


def value_iteration(env, tol: float = 1e-6, max_sweeps: int = 100, verbose: bool = False):
    """Solve the MDP exactly by synchronous Value Iteration.

    Returns ``(v, policy, residuals)`` where ``v`` has shape ``(H + 1, C + 1, R)``,
    ``policy`` has shape ``(H, C + 1, R)`` holding action indices, and
    ``residuals`` is the list of ``||V_{k+1} - V_k||_inf`` per sweep.
    """
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


def policy_evaluation(env, fine_policy: np.ndarray, model: dict | None = None) -> np.ndarray:
    """Exact value of a deterministic policy over the unabstracted state space.

    ``fine_policy`` has shape ``(H, C + 1, N_REGIMES)`` and gives the action taken
    in every reachable state. Because the horizon is finite the value is obtained
    in a single backward pass -- no iteration and no Monte-Carlo noise, so the
    cost of a learned policy's state abstraction can be read off exactly rather
    than estimated.
    """
    model = model or build_model(env)
    transition = env.regime_transition
    v = np.zeros((HORIZON + 1, model["capacity"] + 1, N_REGIMES))

    for w in range(HORIZON - 1, -1, -1):
        q = _week_q(model, transition, v[w + 1], w)
        chosen = fine_policy[w].astype(int)[:, :, None]
        v[w] = np.take_along_axis(q, chosen, axis=2)[:, :, 0]
    return v


def optimal_value(env, v: np.ndarray) -> float:
    """Expected season revenue of the optimal policy from an empty hotel.

    Averages ``V*(week 0, sold 0, g)`` over the stationary regime distribution,
    matching how ``HotelPricingEnv.reset`` draws the opening regime.
    """
    from demand import stationary_distribution

    pi = stationary_distribution(env.regime_transition) if env.use_regimes else np.eye(N_REGIMES)[1]
    return float(v[0, 0] @ pi)


def dp_policy(policy: np.ndarray):
    """Wrap a DP policy table as a callable ``fn(state, env) -> action``.

    The table is indexed by the *unabstracted* state, so it reads ``env.sold``
    directly rather than the agent-visible occupancy bin.
    """

    def act(state, env):
        week = min(HORIZON - state[0], HORIZON - 1)
        return int(policy[week, min(env.sold, env.capacity), env.regime])

    return act
