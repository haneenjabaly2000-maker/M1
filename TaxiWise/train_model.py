"""
TaxiWise — Standalone Model Training Script
============================================
Trains Linear Regression and Random Forest on 2023-2025 NYC taxi demand data,
validates on 2026, selects the best model by R², and saves it to models/model.pkl.

Usage:
    python train_model.py

The saved payload is loaded by the Streamlit app via @st.cache_resource in
src/model.load_regression_model(). If model.pkl does not exist, the app
trains inline automatically, but running this script first is recommended for
faster startup and reproducibility.

Output file: models/model.pkl
"""

import sys
from pathlib import Path

# Ensure project root is on the Python path
sys.path.insert(0, str(Path(__file__).parent))

import joblib

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "model.pkl"


def main() -> dict:
    print("=" * 60)
    print("  TaxiWise — Regression Model Training")
    print("=" * 60)
    print()
    print("Features  :", [
        "pickup_location_id", "pickup_hour", "pickup_day_of_week",
        "pickup_month", "historical_trip_count", "avg_fare_amount",
        "avg_trip_distance", "avg_trip_duration", "year",
    ])
    print("Target    : trip_count")
    print("Data      : 2023 + 2024 + 2025 + 2026  (all years)")
    print("Split     : 80/20 random split (consistent scale across years)")
    print()

    from src.regression import build_model_payload
    payload = build_model_payload(verbose=True)

    joblib.dump(payload, MODEL_PATH)

    m = payload["metrics"]
    print()
    print("=" * 60)
    print(f"  Saved → {MODEL_PATH}")
    print(f"  Best model : {payload['model_name']}")
    print(f"  MAE        : {m['mae']:.3f} trips")
    print(f"  RMSE       : {m['rmse']:.3f} trips")
    print(f"  R²         : {m['r2']:.4f}")
    print(f"  Train rows : {payload['n_train']:,}")
    print(f"  Test rows  : {payload['n_test']:,}  (80/20 random split)"
          f"{' · includes 2026' if payload['has_2026'] else ''}")
    print()
    print("  All models:")
    for name, met in payload["all_metrics"].items():
        print(f"    {name:<22} MAE={met['mae']:.3f}  "
              f"RMSE={met['rmse']:.3f}  R²={met['r2']:.4f}")
    print("=" * 60)

    return payload


if __name__ == "__main__":
    main()
