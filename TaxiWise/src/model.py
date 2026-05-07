"""
TaxiWise — XGBoost Demand Forecasting Model
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

FEATURE_COLS = [
    "PULocationID", "hour", "dow", "month",
    "zone_total_trips", "avg_fare", "avg_distance", "avg_duration",
]
TARGET = "trip_count"

FEATURE_LABELS = {
    "PULocationID":      "Zone ID",
    "hour":              "Hour of Day",
    "dow":               "Day of Week",
    "month":             "Month",
    "zone_total_trips":  "Zone Total Trips",
    "avg_fare":          "Avg Fare ($)",
    "avg_distance":      "Avg Distance (mi)",
    "avg_duration":      "Avg Duration (min)",
}


@st.cache_resource(show_spinner=False)
def get_model():
    """Train once and cache the XGBoost model for the entire session."""
    from src.data_loader import compute_demand
    demand = compute_demand()
    return _train(demand)


def _train(demand: pd.DataFrame):
    df = demand.dropna(subset=FEATURE_COLS + [TARGET])
    X  = df[FEATURE_COLS].values
    y  = df[TARGET].values.astype(float)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators     = 400,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0,
    )
    model.fit(X_tr, y_tr)

    y_pred = np.maximum(model.predict(X_te), 0)

    metrics = {
        "mae":     mean_absolute_error(y_te, y_pred),
        "rmse":    float(np.sqrt(mean_squared_error(y_te, y_pred))),
        "r2":      r2_score(y_te, y_pred),
        "n_train": len(X_tr),
        "n_test":  len(X_te),
    }

    fi = (
        pd.DataFrame({
            "feature":    FEATURE_COLS,
            "label":      [FEATURE_LABELS[f] for f in FEATURE_COLS],
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return model, metrics, fi, y_te, y_pred


def predict_single(
    model,
    loc_id: int, hour: int, dow: int, month: int,
    zone_total: float, avg_fare: float, avg_dist: float, avg_dur: float,
) -> float:
    X = np.array([[loc_id, hour, dow, month, zone_total, avg_fare, avg_dist, avg_dur]])
    return float(max(model.predict(X)[0], 0))


def get_hot_zones(
    model,
    demand: pd.DataFrame,
    zones: pd.DataFrame,
    hour: int,
    dow: int,
    month: int,
    top_n: int = 10,
) -> pd.DataFrame:
    mask = (
        (demand["hour"] == hour) &
        (demand["dow"]  == dow)  &
        (demand["month"] == month)
    )
    cands = demand[mask].copy()

    # Fallback: same hour, any day/month
    if cands.empty:
        cands = (
            demand[demand["hour"] == hour]
            .groupby("PULocationID", as_index=False)
            .agg(
                avg_fare        =("avg_fare",       "mean"),
                avg_distance    =("avg_distance",   "mean"),
                avg_duration    =("avg_duration",   "mean"),
                zone_total_trips=("zone_total_trips","first"),
            )
        )
        cands["hour"]  = hour
        cands["dow"]   = dow
        cands["month"] = month

    if cands.empty:
        return pd.DataFrame()

    cands = cands.dropna(subset=FEATURE_COLS)
    cands["predicted_demand"] = np.maximum(
        model.predict(cands[FEATURE_COLS]), 0
    )

    result = (
        cands
        .nlargest(top_n, "predicted_demand")
        [["PULocationID", "predicted_demand", "avg_fare",
          "avg_distance", "avg_duration", "zone_total_trips"]]
        .reset_index(drop=True)
    )
    result.index = range(1, len(result) + 1)

    z_lookup = (
        zones[["LocationID", "Zone", "Borough"]]
        .rename(columns={"LocationID": "PULocationID"})
    )
    result = result.merge(z_lookup, on="PULocationID", how="left")
    return result
