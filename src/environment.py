from __future__ import annotations

import numpy as np

from config import (
    AVG_NIGHTS,
    CANCEL_MODEL,
    CAPACITY_FACTOR,
    DEFAULT_ELASTICITY,
    DEMAND_TABLES,
    HORIZON,
    MULTIPLIERS,
    N_OCC_BINS,
    N_REGIMES,
    OVERFLOW_PENALTY,
)

_CACHE: dict = {}


def load_tables() -> dict:
    import joblib

    if "tables" not in _CACHE:
        missing = [p.name for p in (DEMAND_TABLES, CANCEL_MODEL) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"missing build artefacts in results/: {', '.join(missing)}. "
                "Run notebooks 01_eda.ipynb and 02_demand_model.ipynb first."
            )
        _CACHE["tables"] = joblib.load(DEMAND_TABLES)
        _CACHE["cancel"] = joblib.load(CANCEL_MODEL)
    return _CACHE["tables"]


def load_cancel_model():
    load_tables()
    return _CACHE["cancel"]


class HotelPricingEnv:
    def __init__(
        self,
        hotel: str,
        month: str,
        elasticity: float | None = None,
        capacity_factor: float = CAPACITY_FACTOR,
        use_regimes: bool = True,
        seed: int | None = None,
    ):
        tables = load_tables()
        self.hotel, self.month = hotel, month
        self.use_regimes = use_regimes
        self.rng = np.random.default_rng(seed)

        self.ref_price = float(tables["ref_price"].loc[month, hotel])
        season_demand = float(tables["demand_vol"].loc[month, hotel])
        self.capacity = max(1, int(season_demand * capacity_factor))

        if elasticity is None:
            est = tables.get("elasticity")
            elasticity = (
                float(est.loc[month, hotel]) if est is not None else DEFAULT_ELASTICITY
            )
        self.elasticity = float(elasticity)

        comp = tables.get("competitor")
        cross_beta = tables.get("price_regression", {}).get("cross_beta", 0.0)
        self.competitor_index = (
            float(comp.loc[month, hotel]) if comp is not None else 1.0
        )
        comp_factor = self.competitor_index**cross_beta

        share = tables["timing"].reindex(range(HORIZON)).fillna(0).to_numpy()[::-1]
        share = share / share.sum()
        self.weekly_base = season_demand * share * comp_factor

        regime = tables.get("regime")
        if regime is not None and use_regimes:
            self.regime_factors = np.asarray(regime["factors"], dtype=float)
            self.regime_transition = np.asarray(regime["transition"], dtype=float)
        else:
            self.regime_factors = np.ones(N_REGIMES)
            self.regime_transition = np.eye(N_REGIMES)
        self._regime_cdf = np.cumsum(self.regime_transition, axis=1)

        self.weekly_pcancel = self._cancellation_curve()
        self._uniforms = None
        self._regime_path = None
        self.reset()

    def set_common_randoms(self, uniforms=None, regime_path=None) -> None:
        self._uniforms = None if uniforms is None else np.asarray(uniforms, dtype=float)
        self._regime_path = (
            None if regime_path is None else np.asarray(regime_path, dtype=int)
        )

    def _cancellation_curve(self) -> np.ndarray:
        from demand import cancellation_curve

        return cancellation_curve(load_cancel_model(), self.hotel, self.month, HORIZON)

    def expected_demand(self, week: int, regime: int, action: int) -> float:
        return float(
            self.weekly_base[week]
            * self.regime_factors[regime]
            * MULTIPLIERS[action] ** (-self.elasticity)
        )

    @property
    def n_actions(self) -> int:
        return len(MULTIPLIERS)

    @property
    def n_states(self) -> tuple[int, int, int]:
        return (HORIZON + 1, N_OCC_BINS, N_REGIMES)

    def occupancy_bin(self, sold: int) -> int:
        return min(int((sold / self.capacity) * N_OCC_BINS), N_OCC_BINS - 1)

    def _state(self) -> tuple[int, int, int]:
        return (HORIZON - self.week, self.occupancy_bin(self.sold), self.regime)

    def reset(self, regime: int | None = None) -> tuple[int, int, int]:
        self.week = 0
        self.sold = 0
        if self._regime_path is not None:
            self.regime = int(self._regime_path[0])
        elif regime is not None:
            self.regime = int(regime)
        elif self.use_regimes:
            if not hasattr(self, "_stationary_cdf"):
                from demand import stationary_distribution

                self._stationary_cdf = np.cumsum(
                    stationary_distribution(self.regime_transition)
                )
            self.regime = int(np.searchsorted(self._stationary_cdf, self.rng.random()))
        else:
            self.regime = 1
        return self._state()

    def step(self, action: int):
        action = int(action)
        rate = self.ref_price * MULTIPLIERS[action]
        lam = max(self.expected_demand(self.week, self.regime, action), 0.0)
        if self._uniforms is not None:
            from scipy.stats import poisson

            arrivals = int(poisson.ppf(self._uniforms[self.week], lam))
        else:
            arrivals = int(self.rng.poisson(lam))

        rooms_left = self.capacity - self.sold
        sold_now = int(min(arrivals, rooms_left))
        turned_away = int(arrivals - sold_now)

        kept = 1.0 - self.weekly_pcancel[self.week]
        realised = sold_now * rate * AVG_NIGHTS * kept
        penalty = OVERFLOW_PENALTY * turned_away * rate
        reward = realised - penalty

        self.sold += sold_now
        self.week += 1
        if self._regime_path is not None:
            self.regime = int(
                self._regime_path[min(self.week, len(self._regime_path) - 1)]
            )
        elif self.use_regimes:
            self.regime = int(
                np.searchsorted(self._regime_cdf[self.regime], self.rng.random())
            )

        done = (self.week >= HORIZON) or (self.sold >= self.capacity)
        info = {
            "sold_now": sold_now,
            "turned_away": turned_away,
            "multiplier": MULTIPLIERS[action],
            "rate": rate,
            "revenue": realised,
            "week": self.week - 1,
        }
        return self._state(), float(reward), done, info
