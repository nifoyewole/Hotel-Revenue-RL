from __future__ import annotations

import numpy as np

from config import HORIZON, N_OCC_BINS, N_REGIMES


def train_double_q(
    env,
    episodes: int = 100_000,
    n_step: int = HORIZON,
    alpha0: float = 0.30,
    alpha_min: float = 0.01,
    gamma: float = 1.0,
    eps_start: float = 1.0,
    eps_min: float = 0.02,
    eps_decay: float = 0.9999,
    seed: int = 0,
    eval_every: int | None = None,
    eval_episodes: int = 500,
):
    rng = np.random.default_rng(seed)
    env.rng = np.random.default_rng(seed + 1)
    env.set_common_randoms(None, None)

    n_actions = env.n_actions
    shape = (HORIZON + 1, N_OCC_BINS, N_REGIMES, n_actions)
    qa = np.zeros(shape)
    qb = np.zeros(shape)

    history = np.empty(episodes)
    checkpoints: list[tuple[int, float]] = []
    eps = eps_start

    for episode in range(episodes):
        alpha = max(alpha_min, alpha0 * (1.0 - episode / episodes))
        state = env.reset()
        done = False
        trajectory = []

        while not done:
            w, o, g = state
            if rng.random() < eps:
                action = int(rng.integers(n_actions))
            else:
                action = int(np.argmax(qa[w, o, g] + qb[w, o, g]))
            nxt, reward, done, _ = env.step(action)
            trajectory.append((state, action, reward, nxt, done))
            state = nxt

        history[episode] = sum(step[2] for step in trajectory)
        _apply_updates(trajectory, qa, qb, rng, alpha, gamma, n_step)

        eps = max(eps_min, eps * eps_decay)

        if eval_every and (episode + 1) % eval_every == 0:
            checkpoints.append(
                (
                    episode + 1,
                    greedy_value(env, (qa + qb) / 2.0, eval_episodes, seed=99),
                )
            )

    return (qa + qb) / 2.0, history, checkpoints


def _apply_updates(trajectory, qa, qb, rng, alpha, gamma, n_step) -> None:
    length = len(trajectory)
    for t in range(length - 1, -1, -1):
        (w, o, g), action, _, _, _ = trajectory[t]
        end = min(t + n_step, length)

        target = 0.0
        discount = 1.0
        for i in range(t, end):
            target += discount * trajectory[i][2]
            discount *= gamma

        terminal = trajectory[end - 1][4]
        update_a = rng.random() < 0.5
        if not terminal:
            w2, o2, g2 = trajectory[end - 1][3]
            if update_a:
                best = int(np.argmax(qa[w2, o2, g2]))
                target += discount * qb[w2, o2, g2, best]
            else:
                best = int(np.argmax(qb[w2, o2, g2]))
                target += discount * qa[w2, o2, g2, best]

        if update_a:
            qa[w, o, g, action] += alpha * (target - qa[w, o, g, action])
        else:
            qb[w, o, g, action] += alpha * (target - qb[w, o, g, action])


def greedy_value(
    env, q_table: np.ndarray, n_episodes: int = 1000, seed: int = 99
) -> float:
    saved_rng = env.rng
    env.rng = np.random.default_rng(seed)
    env.set_common_randoms(None, None)
    total = 0.0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = int(np.argmax(q_table[state[0], state[1], state[2]]))
            state, reward, done, _ = env.step(action)
            total += reward
    env.rng = saved_rng
    return total / n_episodes


def policy_table(q_table: np.ndarray) -> np.ndarray:
    return q_table.argmax(axis=3)


def fine_policy(q_table: np.ndarray, env) -> np.ndarray:
    greedy = policy_table(q_table)
    sold = np.arange(env.capacity + 1)
    bins = np.minimum((sold / env.capacity * N_OCC_BINS).astype(int), N_OCC_BINS - 1)
    out = np.empty((HORIZON, env.capacity + 1, N_REGIMES), dtype=np.int8)
    for w in range(HORIZON):
        out[w] = greedy[HORIZON - w][bins]
    return out
