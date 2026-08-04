from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from config import AVG_NIGHTS, HORIZON, N_REGIMES


def draw_common_randoms(env, n_episodes: int, seed: int = 12345) -> list[dict]:
    from demand import stationary_distribution

    rng = np.random.default_rng(seed)
    pi = (
        stationary_distribution(env.regime_transition)
        if env.use_regimes
        else np.eye(N_REGIMES)[1]
    )
    bundles = []
    for _ in range(n_episodes):
        regimes = np.empty(HORIZON + 1, dtype=int)
        regimes[0] = rng.choice(N_REGIMES, p=pi)
        for w in range(HORIZON):
            regimes[w + 1] = (
                rng.choice(N_REGIMES, p=env.regime_transition[regimes[w]])
                if env.use_regimes
                else regimes[w]
            )
        bundles.append({"uniforms": rng.random(HORIZON), "regime_path": regimes})
    return bundles


def run_episode(env, policy, common=None) -> dict:
    env.set_common_randoms(**(common or {"uniforms": None, "regime_path": None}))
    state = env.reset()
    done = False
    reward_total = 0.0
    gross_revenue = 0.0
    turned_away = 0
    multipliers = []
    while not done:
        state, reward, done, info = env.step(policy(state, env))
        reward_total += reward
        gross_revenue += info["revenue"]
        turned_away += info["turned_away"]
        multipliers.append(info["multiplier"])

    room_nights = env.capacity * AVG_NIGHTS
    return {
        "reward": reward_total,
        "gross_revenue": gross_revenue,
        "rooms_sold": env.sold,
        "occupancy": env.sold / env.capacity,
        "revpar": gross_revenue / room_nights,
        "adr": gross_revenue / (env.sold * AVG_NIGHTS) if env.sold else np.nan,
        "turned_away": turned_away,
        "sold_out": float(env.sold >= env.capacity),
        "weeks_open": len(multipliers),
        "mean_multiplier": float(np.mean(multipliers)),
    }


def evaluate(env, policy, commons: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([run_episode(env, policy, c) for c in commons])


def compare(
    env, strategies: dict, n_episodes: int = 3000, seed: int = 12345
) -> tuple[pd.DataFrame, dict]:
    commons = draw_common_randoms(env, n_episodes, seed=seed)
    raw = {name: evaluate(env, policy, commons) for name, policy in strategies.items()}

    rows = []
    for name, frame in raw.items():
        reward = frame["reward"].to_numpy()
        rows.append(
            {
                "strategy": name,
                "net_revenue": reward.mean(),
                "ci95": 1.96 * reward.std(ddof=1) / np.sqrt(len(reward)),
                "gross_revenue": frame["gross_revenue"].mean(),
                "revpar": frame["revpar"].mean(),
                "adr": frame["adr"].mean(),
                "occupancy": frame["occupancy"].mean(),
                "sell_out_rate": frame["sold_out"].mean(),
                "turned_away": frame["turned_away"].mean(),
                "mean_multiplier": frame["mean_multiplier"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("strategy"), raw


def paired_tests(raw: dict, reference: str, column: str = "reward") -> pd.DataFrame:
    base = raw[reference][column].to_numpy()
    rows = []
    for name, frame in raw.items():
        if name == reference:
            continue
        arr = frame[column].to_numpy()
        diff = arr - base
        t_stat, p_value = stats.ttest_rel(arr, base)
        rows.append(
            {
                "strategy": name,
                "mean_diff": diff.mean(),
                "lift_pct": 100.0 * (arr.mean() / base.mean() - 1.0),
                "t": t_stat,
                "p_raw": p_value,
                "cohens_d": (
                    diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else np.nan
                ),
            }
        )

    out = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    m = len(out)
    adjusted, running = [], 0.0
    for i, p in enumerate(out["p_raw"]):
        running = max(running, min((m - i) * p, 1.0))
        adjusted.append(running)
    out["p_holm"] = adjusted
    out["significant_05"] = out["p_holm"] < 0.05
    return out.set_index("strategy")


def occupancy_paths(env, policy, commons: list[dict]) -> np.ndarray:
    paths = np.full((len(commons), HORIZON), np.nan)
    for i, common in enumerate(commons):
        env.set_common_randoms(**common)
        state = env.reset()
        done = False
        while not done:
            week = HORIZON - state[0]
            if week < HORIZON:
                paths[i, week] = env.sold / env.capacity
            state, _, done, _ = env.step(policy(state, env))
    return paths


def price_paths(env, policy, commons: list[dict]) -> np.ndarray:
    """Posted multiplier in each week, ``(n_episodes, HORIZON)``."""
    paths = np.full((len(commons), HORIZON), np.nan)
    for i, common in enumerate(commons):
        env.set_common_randoms(**common)
        state = env.reset()
        done = False
        while not done:
            week = HORIZON - state[0]
            state, _, done, info = env.step(policy(state, env))
            if week < HORIZON:
                paths[i, week] = info["multiplier"]
    return paths
