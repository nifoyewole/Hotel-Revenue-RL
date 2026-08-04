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

def reference_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
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


def wtp_elasticity(
    adr: np.ndarray, ref_price: float, multipliers: np.ndarray = MULTIPLIERS
) -> tuple[float, np.ndarray, np.ndarray]:
    a = np.asarray(adr, dtype=float)
    survival = np.array([(a >= ref_price * m).mean() for m in multipliers])
    ok = survival > 0
    if ok.sum() < 2:
        raise ValueError("survival curve is degenerate; too few observations")
    slope, intercept = np.polyfit(np.log(multipliers[ok]), np.log(survival[ok]), 1)
    fitted = np.exp(intercept) * multipliers**slope
    return float(-slope), survival, fitted


def elasticity_table(df: pd.DataFrame, ref_price: pd.DataFrame) -> pd.DataFrame:
    """Per-(month, hotel) WTP elasticity, laid out like ``ref_price``."""
    out = pd.DataFrame(index=ref_price.index, columns=ref_price.columns, dtype=float)
    for (hotel, month), sub in df.groupby(
        ["hotel", "arrival_date_month"], observed=True
    ):
        if month in out.index and hotel in out.columns:
            eps, _, _ = wtp_elasticity(
                sub["adr"].to_numpy(), float(ref_price.loc[month, hotel])
            )
            out.loc[month, hotel] = eps
    return out


def naive_price_regression(df: pd.DataFrame, ref_price: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf

    cells = (
        df.groupby(
            [
                "hotel",
                "arrival_date_year",
                "arrival_date_month",
                "arrival_date_week_number",
            ],
            observed=True,
        )
        .agg(n=("adr", "size"), adr=("adr", "median"))
        .reset_index()
    )
    ref_long = (
        ref_price.stack().rename("ref").rename_axis(["arrival_date_month", "hotel"])
    )
    cells = cells.join(ref_long, on=["arrival_date_month", "hotel"])
    cells["rel"] = cells["adr"] / cells["ref"]

    wide = cells.pivot_table(
        index=["arrival_date_year", "arrival_date_week_number"],
        columns="hotel",
        values="rel",
    )
    cells = cells.merge(
        wide.reset_index(),
        on=["arrival_date_year", "arrival_date_week_number"],
        how="left",
        suffixes=("", "_w"),
    )
    hotels = list(wide.columns)
    other = {hotels[0]: hotels[1], hotels[1]: hotels[0]}
    cells["comp_rel"] = [row[other[row["hotel"]]] for _, row in cells.iterrows()]
    cells = cells.dropna(subset=["comp_rel", "rel"])

    cells["ln_n"] = np.log(cells["n"])
    cells["ln_p"] = np.log(cells["rel"])
    cells["ln_c"] = np.log(cells["comp_rel"])
    fit = smf.ols(
        "ln_n ~ ln_p + ln_c + C(hotel):C(arrival_date_month)", data=cells
    ).fit()

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


def market_regimes(df: pd.DataFrame, n_regimes: int = N_REGIMES) -> dict:
    cells = (
        df.groupby(
            ["hotel", "arrival_date_year", "arrival_date_week_number"], observed=True
        )
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["hotel", "arrival_date_year", "arrival_date_week_number"])
    )
    seasonal = cells.groupby(["hotel", "arrival_date_week_number"])["n"].transform(
        "mean"
    )
    cells["shock"] = cells["n"] / seasonal

    qs = np.linspace(0, 1, n_regimes + 1)[1:-1]
    cuts = cells["shock"].quantile(qs).to_numpy()
    cells["regime"] = np.digitize(cells["shock"].to_numpy(), cuts)

    factors = (
        cells.groupby("regime")["shock"].mean().reindex(range(n_regimes)).to_numpy()
    )

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


def competitor_index(ref_price: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=ref_price.index, columns=ref_price.columns, dtype=float)
    hotels = list(ref_price.columns)
    for h in hotels:
        rival = [x for x in hotels if x != h]
        out[h] = ref_price[rival].mean(axis=1) / ref_price[h]
    return out


def cancellation_curve(
    cancel_model, hotel: str, month: str, horizon: int = HORIZON
) -> np.ndarray:
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
