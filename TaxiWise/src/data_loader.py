"""
TaxiWise — Data Loading & Feature Engineering
"""

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT     = Path(__file__).parent.parent
TAXI_CSV = ROOT / "yellow_taxi_2025.csv"
ZONE_CSV = ROOT / "data" / "taxi_zone_lookup.csv"

PAYMENT_LABELS = {
    1: "Credit Card", 2: "Cash", 3: "No Charge",
    4: "Dispute",     5: "Unknown", 6: "Voided",
}
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@st.cache_data(show_spinner=False)
def load_trips() -> pd.DataFrame:
    df = pd.read_csv(
        TAXI_CSV,
        parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"],
    )
    dt = df["tpep_pickup_datetime"]
    df["hour"]          = dt.dt.hour
    df["dow"]           = dt.dt.dayofweek
    df["month"]         = dt.dt.month
    df["date"]          = dt.dt.date
    df["day_name"]      = df["dow"].map(dict(enumerate(DAY_NAMES)))
    df["payment_label"] = df["payment_type"].map(PAYMENT_LABELS).fillna("Unknown")

    # Drop invalid rows
    df = df[
        (df["fare_amount"]       > 0)   & (df["fare_amount"]       < 500) &
        (df["trip_distance"]     >= 0)  & (df["trip_distance"]     < 200) &
        (df["trip_duration_min"] > 0)   & (df["trip_duration_min"] < 600)
    ].copy()
    return df


@st.cache_data(show_spinner=False)
def load_zones() -> pd.DataFrame:
    return pd.read_csv(ZONE_CSV)


@st.cache_data(show_spinner=False)
def compute_demand() -> pd.DataFrame:
    """Aggregate trip counts + stats per (zone, hour, dow, month) for the ML model."""
    df = load_trips()
    agg = (
        df.groupby(["PULocationID", "hour", "dow", "month"])
        .agg(
            trip_count   =("VendorID",        "count"),
            avg_fare     =("fare_amount",      "mean"),
            avg_distance =("trip_distance",    "mean"),
            avg_duration =("trip_duration_min","mean"),
            avg_tip      =("tip_amount",       "mean"),
        )
        .reset_index()
    )
    zone_totals = df.groupby("PULocationID").size().rename("zone_total_trips")
    agg = agg.merge(zone_totals, on="PULocationID", how="left")
    return agg


@st.cache_data(show_spinner=False)
def get_kpis() -> dict:
    df = load_trips()
    clean = df[(df["total_amount"] > 0) & (df["total_amount"] < 500)]
    top_zone = (
        df["pickup_zone"].value_counts().idxmax()
        if "pickup_zone" in df.columns else "N/A"
    )
    return {
        "total_trips":   len(df),
        "avg_fare":      clean["total_amount"].mean(),
        "avg_distance":  df["trip_distance"].mean(),
        "avg_duration":  df["trip_duration_min"].mean(),
        "top_zone":      top_zone,
        "peak_hour":     int(df["hour"].value_counts().idxmax()),
        "active_zones":  df["PULocationID"].nunique(),
        "total_revenue": clean["total_amount"].sum(),
        "credit_pct":    (df["payment_type"] == 1).mean() * 100,
    }
