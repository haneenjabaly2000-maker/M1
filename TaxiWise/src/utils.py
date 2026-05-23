"""
TaxiWise - Utility Functions
============================
Logging, path helpers, schema validation, and synthetic data generation.
"""

import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_RAW   = ROOT_DIR / "data" / "raw"
DATA_PROC  = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
ZONE_CSV   = ROOT_DIR / "data" / "taxi_zone_lookup.csv"

for _d in (DATA_RAW, DATA_PROC, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def get_logger(name: str = "taxiwise") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Zone Lookup
# ─────────────────────────────────────────────────────────────────────────────
def load_zone_lookup() -> pd.DataFrame:
    """Load the NYC TLC taxi zone lookup CSV."""
    if ZONE_CSV.exists():
        df = pd.read_csv(ZONE_CSV)
        df.columns = [c.strip() for c in df.columns]
        return df
    raise FileNotFoundError(f"Zone lookup CSV not found: {ZONE_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Data Generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_data(
    n_rows: int = 600_000,
    seed: int = 42,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic NYC Yellow Taxi dataset.

    Schema matches the official TLC PARQUET schema so that downstream
    preprocessing works identically on both real and synthetic data.
    """
    if years is None:
        years = [2023, 2024, 2025]

    logger = get_logger("utils.synth")
    logger.info(f"Generating {n_rows:,} synthetic trip rows (seed={seed}) ...")

    rng = np.random.default_rng(seed)
    random.seed(seed)

    # ── Zone weights: Manhattan + airports much more popular ──────────────────
    zone_ids = list(range(1, 266))
    weights   = np.ones(265)

    # Boost Manhattan (approx 4–265 range with yellow zones)
    manhattan_ids = [4,12,13,24,37,38,39,41,44,46,48,62,68,69,73,81,82,93,
                     104,105,111,116,119,120,129,132,133,134,135,136,140,147,
                     148,149,150,153,157,161,171,179,187,195,199,203,208,209,
                     210,211,212,214,216,217,218,219,223,224,226,229,241,242,
                     243,247,249,250,251,252,253,258,259,260,261,262,263,264,265]
    for mid in manhattan_ids:
        if mid <= 265:
            weights[mid - 1] = 5.0

    # Boost airports
    for aid in [124, 130]:
        weights[aid - 1] = 8.0

    weights /= weights.sum()

    # ── Pickup timestamps ─────────────────────────────────────────────────────
    year_choices = rng.choice(years, size=n_rows)
    month_choices = rng.integers(1, 13, size=n_rows)

    # Days per month (simplified)
    days = rng.integers(1, 29, size=n_rows)
    hours_raw = rng.choice(
        np.arange(24),
        size=n_rows,
        p=_hour_probs(rng),
    )
    minutes = rng.integers(0, 60, size=n_rows)
    seconds = rng.integers(0, 60, size=n_rows)

    pickup_dt = pd.to_datetime(
        {
            "year":   year_choices,
            "month":  month_choices,
            "day":    days,
            "hour":   hours_raw,
            "minute": minutes,
            "second": seconds,
        }
    )

    # ── Trip duration: 5–45 min weighted ─────────────────────────────────────
    duration_min = rng.exponential(scale=12, size=n_rows).clip(1, 120)
    dropoff_dt = pickup_dt + pd.to_timedelta(duration_min, unit="m")

    # ── Locations ─────────────────────────────────────────────────────────────
    pu_loc = rng.choice(zone_ids, size=n_rows, p=weights)
    do_loc = rng.choice(zone_ids, size=n_rows, p=weights)

    # ── Trip distance ─────────────────────────────────────────────────────────
    trip_distance = (duration_min / 60) * rng.uniform(8, 22, size=n_rows)
    trip_distance = trip_distance.clip(0.1, 50)

    # ── Fare ──────────────────────────────────────────────────────────────────
    base_fare   = 3.0 + trip_distance * rng.uniform(1.8, 2.8, size=n_rows)
    tip_amount  = base_fare * rng.uniform(0, 0.3, size=n_rows)
    total_amount = base_fare + tip_amount + rng.uniform(0.5, 3.5, size=n_rows)

    # ── Passenger count ───────────────────────────────────────────────────────
    passenger_count = rng.choice([1, 1, 1, 2, 2, 3, 4, 5, 6], size=n_rows)

    df = pd.DataFrame(
        {
            "tpep_pickup_datetime":  pickup_dt,
            "tpep_dropoff_datetime": dropoff_dt,
            "PULocationID":          pu_loc,
            "DOLocationID":          do_loc,
            "passenger_count":       passenger_count,
            "trip_distance":         trip_distance.round(2),
            "fare_amount":           base_fare.round(2),
            "tip_amount":            tip_amount.round(2),
            "total_amount":          total_amount.round(2),
            "payment_type":          rng.choice([1, 2, 3, 4], size=n_rows),
            "RatecodeID":            rng.choice([1, 1, 1, 2, 3, 4, 5, 6], size=n_rows),
            "store_and_fwd_flag":    rng.choice(["N", "Y"], size=n_rows, p=[0.97, 0.03]),
        }
    )

    logger.info("Synthetic data generated ✓")
    return df


def _hour_probs(rng) -> np.ndarray:
    """Probability distribution over 24 hours reflecting NYC taxi demand."""
    # Low at night, peaks 8-9am and 5-8pm
    raw = np.array([
        0.5, 0.3, 0.2, 0.2, 0.3, 0.6,   # 0-5
        1.2, 2.5, 3.5, 3.0, 2.5, 2.5,   # 6-11
        2.8, 2.5, 2.3, 2.3, 2.5, 3.5,   # 12-17
        4.0, 3.8, 3.0, 2.5, 1.8, 1.0,   # 18-23
    ], dtype=float)
    return raw / raw.sum()
