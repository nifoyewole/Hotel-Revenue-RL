from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

MULTIPLIERS = np.array([0.70, 0.85, 1.00, 1.15, 1.30, 1.45])
NAIVE_ACTION = 2

HORIZON = 12
N_OCC_BINS = 10
N_REGIMES = 3
REGIME_NAMES = ("soft", "normal", "strong")

AVG_NIGHTS = 3  # assumed average length of stay (median in the cleaned data)
CAPACITY_FACTOR = 0.7  # sellable rooms as a fraction of one season's demand

OVERFLOW_PENALTY = 0.10

# used only if the estimated tables are unavailable.
DEFAULT_ELASTICITY = 2.5

DEMAND_TABLES = RESULTS / "demand_tables.joblib"
CANCEL_MODEL = RESULTS / "cancel_model.joblib"

MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

# feature columns consumed by the cancellation classifier.
FEATURE_COLS = [
    "hotel",
    "lead_time",
    "arrival_date_month",
    "arrival_date_week_number",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "total_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

CAT_COLS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]
