"""Estimation of the environment's demand primitives from historical bookings.

The assignment requires the environment dynamics to be *learned from data* rather
than assumed. This module estimates every quantity the simulator needs:

* ``ref_price``   -- reference rate per (hotel, month)          [median ADR]
* ``demand_vol``  -- season demand volume per (hotel, month)    [bookings/year]
* ``timing``      -- booking curve: share of demand by weeks before arrival
* ``elasticity``  -- own-price elasticity per (hotel, month)    [WTP curve]
* ``regime``      -- market-condition Markov chain              [factors + P]
* ``competitor``  -- competitor price index per (hotel, month)

Identification note
-------------------
Price is not randomly assigned in observational hotel data: rates are raised
precisely when demand is strong. A naive log-log regression of bookings on the
realised rate therefore returns an *upward*-biased slope (in this dataset it is
positive, i.e. an apparent upward-sloping demand curve, which is economically
impossible). We therefore estimate the price response from the empirical
willingness-to-pay (WTP) curve instead -- the standard approach in the revenue
management literature -- and report the naive regression only as a diagnostic.
Both estimators are returned so the notebook can show the contrast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    AVG_NIGHTS,
    CAT_COLS,
    FEATURE_COLS,
    HORIZON,
    MONTH_ORDER,
    MULTIPLIERS,
    N_REGIMES,
)


# --------------------------------------------------------------------------- #
# Reference prices, volumes and the booking curve
# --------------------------------------------------------------------------- #
def reference_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return ``(ref_price, demand_vol, timing)``.

    ``ref_price`` is the median ADR per (month, hotel); the median is used rather
    than the mean because ADR is right-skewed. ``demand_vol`` is bookings per
    (month, hotel) divided by the number of years in the sample, i.e. the size of
    one season's demand. ``timing`` is the share of bookings made ``w`` weeks
    before arrival and defines the booking curve.
    """
    ref_price = (
        df.groupby(["hotel", "arrival_date_month"], observed=True)["adr"]
        .median()
        .unstack(0)
        .reindex(MONTH_ORDER)
    )
    n_years = df["arrival_date_year"].nunique()
    demand_vol = (
        df.groupby(["hotel", "arrival_date_month"], observed=True)
        .size()
        .unstack(0)
        .reindex(MONTH_ORDER)
        / n_years
    )

    weeks_before = (df["lead_time"] // 7).clip(upper=26)
    timing = weeks_before.value_counts().sort_index()
    timing = timing / timing.sum()
    timing.index.name = "weeks_before"
    return ref_price, demand_vol, timing


# --------------------------------------------------------------------------- #
# Own-price elasticity
# --------------------------------------------------------------------------- #
def wtp_elasticity(adr: np.ndarray, ref_price: float,
                   multipliers: np.ndarray = MULTIPLIERS
                   ) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate own-price elasticity from the empirical willingness-to-pay curve.

    The share of historical bookings transacted at or above a rate ``p`` is a
    non-parametric estimate of the survival function ``S(p) = P(WTP >= p)``: it
    is the fraction of the market that would still book at that rate. Fitting a
    constant-elasticity form ``S(m) = c * m ** (-eps)`` across the action grid by
    least squares on logs gives the elasticity implied by the data.

    Returns ``(elasticity, survival, fitted)``, where ``survival[k]`` is ``S``
    evaluated at ``ref_price * multipliers[k]`` and ``fitted`` is the fitted
    curve including its intercept -- the environment only ever uses the slope,
    since its demand base is already calibrated at the list rate, but the
    intercept is needed to plot the fit against the data.
    """
    a = np.asarray(adr, dtype=float)
    survival = np.array([(a >= ref_price * m).mean() for m in multipliers])
    ok = survival > 0
    if ok.sum() < 2:
        raise ValueError("survival curve is degenerate; too few observations")
    slope, intercept = np.polyfit(np.log(multipliers[ok]), np.log(survival[ok]), 1)
    fitted = np.exp(intercept) * multipliers ** slope
    return float(-slope), survival, fitted


def elasticity_table(df: pd.DataFrame, ref_price: pd.DataFrame) -> pd.DataFrame:
    """Per-(month, hotel) WTP elasticity, laid out like ``ref_price``."""
    out = pd.DataFrame(index=ref_price.index, columns=ref_price.columns, dtype=float)
    for (hotel, month), sub in df.groupby(["hotel", "arrival_date_month"], observed=True):
        if month in out.index and hotel in out.columns:
            eps, _, _ = wtp_elasticity(sub["adr"].to_numpy(), float(ref_price.loc[month, hotel]))
            out.loc[month, hotel] = eps
    return out


def naive_price_regression(df: pd.DataFrame, ref_price: pd.DataFrame) -> dict:
    """Diagnostic OLS of log bookings on log relative rate, with (hotel x month) FE.

    Reported purely to *document the endogeneity problem*: the own-price
    coefficient comes out positive, which is why the WTP estimator is preferred.
    The cross-price (competitor) coefficient from the same fit is used as the
    competitor sensitivity, with the caveat that it is weakly identified.
    """
    import statsmodels.formula.api as smf

    cells = (
        df.groupby(
            ["hotel", "arrival_date_year", "arrival_date_month", "arrival_date_week_number"],
            observed=True,
        )
        .agg(n=("adr", "size"), adr=("adr", "median"))
        .reset_index()
    )
    ref_long = ref_price.stack().rename("ref").rename_axis(["arrival_date_month", "hotel"])
    cells = cells.join(ref_long, on=["arrival_date_month", "hotel"])
    cells["rel"] = cells["adr"] / cells["ref"]

    wide = cells.pivot_table(
        index=["arrival_date_year", "arrival_date_week_number"], columns="hotel", values="rel"
    )
    cells = cells.merge(wide.reset_index(), on=["arrival_date_year", "arrival_date_week_number"],
                        how="left", suffixes=("", "_w"))
    hotels = list(wide.columns)
    other = {hotels[0]: hotels[1], hotels[1]: hotels[0]}
    cells["comp_rel"] = [row[other[row["hotel"]]] for _, row in cells.iterrows()]
    cells = cells.dropna(subset=["comp_rel", "rel"])

    cells["ln_n"] = np.log(cells["n"])
    cells["ln_p"] = np.log(cells["rel"])
    cells["ln_c"] = np.log(cells["comp_rel"])
    fit = smf.ols("ln_n ~ ln_p + ln_c + C(hotel):C(arrival_date_month)", data=cells).fit()

    ci = fit.conf_int()
    return {
        "own_beta": float(fit.params["ln_p"]),
        "own_ci": tuple(np.round(ci.loc["ln_p"].to_numpy(), 3)),
        "own_p": float(fit.pvalues["ln_p"]),
        "cross_beta": float(fit.params["ln_c"]),
        "cross_ci": tuple(np.round(ci.loc["ln_c"].to_numpy(), 3)),
        "cross_p": float(fit.pvalues["ln_c"]),
        "n_obs": int(fit.nobs),
        "r2": float(fit.rsquared),
    }


# --------------------------------------------------------------------------- #
# Market-condition regimes (the external-factor state variable)
# --------------------------------------------------------------------------- #
def market_regimes(df: pd.DataFrame, n_regimes: int = N_REGIMES) -> dict:
    """Estimate a Markov chain over latent market conditions.

    Weekly booking volume per (hotel, calendar week) is divided by its across-year
    average for that same week, which strips out seasonality and leaves a demand
    *shock*. Cutting the shock into terciles yields discrete regimes
    (soft / normal / strong); the transition matrix is then counted from
    consecutive calendar weeks within each (hotel, year).

    Returns ``{"factors": (n,), "transition": (n, n), "cuts": (n-1,)}`` where
    ``factors[g]`` multiplies the weekly demand base under regime ``g``.
    """
    cells = (
        df.groupby(["hotel", "arrival_date_year", "arrival_date_week_number"], observed=True)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["hotel", "arrival_date_year", "arrival_date_week_number"])
    )
    seasonal = cells.groupby(["hotel", "arrival_date_week_number"])["n"].transform("mean")
    cells["shock"] = cells["n"] / seasonal

    qs = np.linspace(0, 1, n_regimes + 1)[1:-1]
    cuts = cells["shock"].quantile(qs).to_numpy()
    cells["regime"] = np.digitize(cells["shock"].to_numpy(), cuts)

    factors = cells.groupby("regime")["shock"].mean().reindex(range(n_regimes)).to_numpy()

    counts = np.zeros((n_regimes, n_regimes))
    for _, grp in cells.groupby(["hotel", "arrival_date_year"], observed=True):
        grp = grp.sort_values("arrival_date_week_number")
        weeks = grp["arrival_date_week_number"].to_numpy()
        regs = grp["regime"].to_numpy()
        consecutive = np.diff(weeks) == 1
        for i in np.flatnonzero(consecutive):
            counts[regs[i], regs[i + 1]] += 1
    transition = counts / counts.sum(axis=1, keepdims=True)

    return {"factors": factors, "transition": transition, "cuts": cuts}


def stationary_distribution(transition: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Stationary distribution of a row-stochastic matrix, by power iteration."""
    pi = np.full(transition.shape[0], 1.0 / transition.shape[0])
    for _ in range(10_000):
        nxt = pi @ transition
        if np.abs(nxt - pi).max() < tol:
            break
        pi = nxt
    return pi


# --------------------------------------------------------------------------- #
# Competitor price index
# --------------------------------------------------------------------------- #
def competitor_index(ref_price: pd.DataFrame) -> pd.DataFrame:
    """Competitor rate relative to own rate, per (month, hotel).

    With two properties in the dataset each acts as the other's competitor. A
    value above 1 means the rival is priced above us that month, which -- for
    substitute goods -- should shift demand toward us.
    """
    out = pd.DataFrame(index=ref_price.index, columns=ref_price.columns, dtype=float)
    hotels = list(ref_price.columns)
    for h in hotels:
        rival = [x for x in hotels if x != h]
        out[h] = ref_price[rival].mean(axis=1) / ref_price[h]
    return out


# --------------------------------------------------------------------------- #
# Cancellation curve
# --------------------------------------------------------------------------- #
def cancellation_curve(cancel_model, hotel: str, month: str,
                       horizon: int = HORIZON) -> np.ndarray:
    """P(cancel) for a representative booking, indexed by *decision week*.

    Decision week 0 sits ``horizon - 1`` weeks before arrival and week
    ``horizon - 1`` is the arrival week, so the lead time attached to week ``w``
    is ``(horizon - 1 - w) * 7`` days. Getting this mapping the wrong way round
    silently inverts the cancellation curve, so it is asserted downstream.
    """
    rows = [
        {
            "hotel": hotel,
            "lead_time": (horizon - 1 - w) * 7,
            "arrival_date_month": month,
            "arrival_date_week_number": 26,
            "stays_in_weekend_nights": 1,
            "stays_in_week_nights": 2,
            "total_nights": AVG_NIGHTS,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "meal": "BB",
            "country": "PRT",
            "market_segment": "Online TA",
            "distribution_channel": "TA/TO",
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": "A",
            "deposit_type": "No Deposit",
            "customer_type": "Transient",
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 1,
        }
        for w in range(horizon)
    ]
    X = pd.DataFrame(rows)[FEATURE_COLS]
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    return cancel_model.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_demand_tables(df: pd.DataFrame) -> dict:
    """Estimate every demand primitive and return them in one dictionary."""
    ref_price, demand_vol, timing = reference_tables(df)
    return {
        "ref_price": ref_price,
        "demand_vol": demand_vol,
        "timing": timing,
        "elasticity": elasticity_table(df, ref_price),
        "competitor": competitor_index(ref_price),
        "regime": market_regimes(df),
        "price_regression": naive_price_regression(df, ref_price),
    }
