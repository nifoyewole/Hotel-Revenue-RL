"""Markov decision process for weekly hotel room pricing.

Formulation
-----------
A single (hotel, arrival-month) selling season is modelled as a finite-horizon
MDP ``<S, A, P, R, gamma, H>`` with ``H = 12`` weekly decision epochs.

**State.** The agent observes ``s = (weeks_remaining, occupancy_bin, regime)``:

* ``weeks_remaining in {0, ..., 12}`` -- selling time left (time-of-season),
* ``occupancy_bin in {0, ..., 9}``    -- rooms sold so far as a decile of capacity
  (the supply / inventory level),
* ``regime in {0, 1, 2}``            -- latent market condition, soft / normal /
  strong, estimated from historical weekly demand shocks (the external factor).

The underlying (unabstracted) state is ``(week, rooms_sold, regime)``; the agent
sees ``rooms_sold`` only through its decile bin. ``src.mdp`` solves the
unabstracted problem exactly, which lets the cost of this abstraction be measured.

**Action.** ``a in {0, ..., 5}`` selects a price multiplier from
``config.MULTIPLIERS``; the posted rate is ``ref_price * MULTIPLIERS[a]``.

**Transition.** Bookings arriving in week ``w`` are Poisson with mean

    lambda(w, g, a) = base[w] * phi[g] * MULTIPLIERS[a] ** (-eps)

where ``base[w]`` is the estimated booking curve, ``phi[g]`` the regime demand
factor and ``eps`` the estimated own-price elasticity. Sales are capped by the
remaining inventory, and the regime evolves independently of the action under the
estimated transition matrix ``P_regime``:

    rooms_sold' = min(rooms_sold + Poisson(lambda), capacity)
    regime'     ~ P_regime[regime, .]

**Reward.** Cancellation-discounted realised revenue less a denied-service cost:

    R = sold_now * rate * AVG_NIGHTS * (1 - p_cancel[w]) - c * turned_away * rate

**Constraints.** One rate per week; the rate is bounded to
``[0.70, 1.45] x ref_price``; cumulative sales may not exceed capacity.

``gamma = 1`` -- the horizon is finite and short, so revenue is not discounted.
"""

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
    """Load the estimated demand tables and cancellation model (cached).

    Raises a directed error rather than an opaque ``FileNotFoundError`` when the
    artefacts have not been produced yet, because they are build outputs and are
    deliberately not committed to the repository.
    """
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
    """Load the fitted cancellation classifier (cached)."""
    load_tables()
    return _CACHE["cancel"]


class HotelPricingEnv:
    """Simulator for one selling season, exposing a Gym-style ``reset``/``step``.

    Parameters
    ----------
    hotel, month
        Selects the (hotel, month) cell whose estimated primitives are used.
    elasticity
        Own-price elasticity. ``None`` uses the WTP estimate for this cell.
    capacity_factor
        Sellable rooms as a fraction of one season's historical demand.
    use_regimes
        If ``False`` the market regime is pinned to ``normal``, which recovers the
        simpler 2-D state used as an ablation.
    seed
        Seed for the arrival and regime random streams.
    """

    def __init__(self, hotel: str, month: str, elasticity: float | None = None,
                 capacity_factor: float = CAPACITY_FACTOR, use_regimes: bool = True,
                 seed: int | None = None):
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

        # Competitor rates shift our demand base. The cross-price coefficient is
        # taken from the fixed-effects regression; it is weakly identified (its
        # confidence interval spans zero), so its influence is modest by design.
        comp = tables.get("competitor")
        cross_beta = tables.get("price_regression", {}).get("cross_beta", 0.0)
        self.competitor_index = (
            float(comp.loc[month, hotel]) if comp is not None else 1.0
        )
        comp_factor = self.competitor_index ** cross_beta

        # Booking curve. `timing` is indexed by weeks *before* arrival, so it is
        # reversed to run forward in decision time: week 0 is the earliest epoch.
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
        # Row-wise CDF for inverse-transform regime sampling; `rng.choice` with a
        # probability vector dominates the runtime of a training run otherwise.
        self._regime_cdf = np.cumsum(self.regime_transition, axis=1)

        self.weekly_pcancel = self._cancellation_curve()
        self._uniforms = None
        self._regime_path = None
        self.reset()

    # -- common random numbers ---------------------------------------------
    def set_common_randoms(self, uniforms=None, regime_path=None) -> None:
        """Pin the episode's randomness so policies can be compared pairwise.

        ``uniforms[w]`` is inverted through the Poisson quantile function to draw
        week ``w``'s arrivals, so two policies facing the same ``uniforms`` meet
        the same demand *percentile* even though their rates -- and hence their
        arrival means -- differ. ``regime_path`` fixes the market conditions
        outright, since regimes evolve independently of the action. Together
        these implement common random numbers, which removes most of the
        Monte-Carlo noise from a strategy comparison and licenses a paired test.

        Pass ``None`` to return to independent sampling.
        """
        self._uniforms = None if uniforms is None else np.asarray(uniforms, dtype=float)
        self._regime_path = None if regime_path is None else np.asarray(regime_path, dtype=int)

    # -- model primitives ---------------------------------------------------
    def _cancellation_curve(self) -> np.ndarray:
        from demand import cancellation_curve

        return cancellation_curve(load_cancel_model(), self.hotel, self.month, HORIZON)

    def expected_demand(self, week: int, regime: int, action: int) -> float:
        """Poisson mean arrivals in ``week`` under ``regime`` and ``action``."""
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
        """Shape of the tabular (abstracted) state space."""
        return (HORIZON + 1, N_OCC_BINS, N_REGIMES)

    # -- MDP interface ------------------------------------------------------
    def occupancy_bin(self, sold: int) -> int:
        """Decile bin of current occupancy, clipped to the last bin when full."""
        return min(int((sold / self.capacity) * N_OCC_BINS), N_OCC_BINS - 1)

    def _state(self) -> tuple[int, int, int]:
        return (HORIZON - self.week, self.occupancy_bin(self.sold), self.regime)

    def reset(self, regime: int | None = None) -> tuple[int, int, int]:
        """Start a new season. The initial regime is drawn from the stationary law."""
        self.week = 0
        self.sold = 0
        if self._regime_path is not None:
            self.regime = int(self._regime_path[0])
        elif regime is not None:
            self.regime = int(regime)
        elif self.use_regimes:
            if not hasattr(self, "_stationary_cdf"):
                from demand import stationary_distribution

                self._stationary_cdf = np.cumsum(stationary_distribution(self.regime_transition))
            self.regime = int(np.searchsorted(self._stationary_cdf, self.rng.random()))
        else:
            self.regime = 1
        return self._state()

    def step(self, action: int):
        """Post a rate for the current week and advance one epoch.

        Returns ``(next_state, reward, done, info)``.
        """
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
            self.regime = int(self._regime_path[min(self.week, len(self._regime_path) - 1)])
        elif self.use_regimes:
            self.regime = int(np.searchsorted(self._regime_cdf[self.regime], self.rng.random()))

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
