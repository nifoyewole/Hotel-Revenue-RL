"""Evaluation harness: paired Monte-Carlo comparison of pricing strategies.

Revenue alone is a thin basis for judging a pricing strategy, so every episode
also records utilisation and rate outcomes: a policy that maximises revenue by
selling out early at a low rate is a different proposition from one that holds
inventory for late high-rate demand, and the operator cares about the difference.

Comparisons use **common random numbers**: all strategies face the same pre-drawn
demand percentiles and the same market-regime path, so differences reflect the
policies rather than luck. That makes the samples paired, which both licenses a
paired t-test and cuts the standard error by roughly an order of magnitude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from config import AVG_NIGHTS, HORIZON, N_REGIMES


def draw_common_randoms(env, n_episodes: int, seed: int = 12345) -> list[dict]:
    """Pre-draw one randomness bundle per episode, independent of any policy.

    The regime path is simulated from the estimated Markov chain and the demand
    percentiles are uniform draws; neither depends on the actions a policy will
    later take, which is what makes them reusable across strategies.
    """
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
    """Play one season and return its revenue and operating statistics."""
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
        # `reward` is the MDP objective: cancellation-discounted revenue less the
        # denied-service cost. `gross_revenue` is the cash figure before that
        # imputed goodwill charge; the two are kept distinct because RevPAR and
        # ADR are operating statistics and must be computed from cash.
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
    """Run one policy over a shared set of randomness bundles."""
    return pd.DataFrame([run_episode(env, policy, c) for c in commons])


def compare(env, strategies: dict, n_episodes: int = 3000, seed: int = 12345
            ) -> tuple[pd.DataFrame, dict]:
    """Evaluate every strategy on identical episodes.

    Returns ``(summary, raw)``: a per-strategy summary table and the dictionary
    of per-episode frames for downstream significance testing.

    ``net_revenue`` is the quantity every algorithm optimises -- realised revenue
    less the denied-service cost -- and is what strategies are ranked on.
    ``gross_revenue`` is the same figure before that imputed charge; the two
    differ by under 1% here, but they are reported separately so that no number
    quoted downstream is ambiguous about which it is.
    """
    commons = draw_common_randoms(env, n_episodes, seed=seed)
    raw = {name: evaluate(env, policy, commons) for name, policy in strategies.items()}

    rows = []
    for name, frame in raw.items():
        reward = frame["reward"].to_numpy()
        rows.append({
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
        })
    return pd.DataFrame(rows).set_index("strategy"), raw


def paired_tests(raw: dict, reference: str, column: str = "reward") -> pd.DataFrame:
    """Paired t-tests of every strategy against ``reference``, Holm-corrected.

    Reports Cohen's ``d`` for paired samples (mean difference over the standard
    deviation of the differences) because with thousands of episodes a t
    statistic says almost nothing about whether a gap is economically meaningful.
    Holm's step-down correction controls the family-wise error rate across the
    several comparisons made from one experiment.
    """
    base = raw[reference][column].to_numpy()
    rows = []
    for name, frame in raw.items():
        if name == reference:
            continue
        arr = frame[column].to_numpy()
        diff = arr - base
        t_stat, p_value = stats.ttest_rel(arr, base)
        rows.append({
            "strategy": name,
            "mean_diff": diff.mean(),
            "lift_pct": 100.0 * (arr.mean() / base.mean() - 1.0),
            "t": t_stat,
            "p_raw": p_value,
            "cohens_d": diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else np.nan,
        })

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
    """Occupancy at the start of each week, ``(n_episodes, HORIZON)``.

    Used for the booking-pace plot: it shows *how* a strategy fills the hotel,
    which the headline revenue number hides.
    """
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
