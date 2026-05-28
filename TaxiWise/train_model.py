"""
TaxiWise — Standalone Model Training Script
============================================
Trains ALL models and saves them to models/.
Must be run once before deploying; the Streamlit app only loads pre-trained files.

    python train_model.py

Output:
    models/model.pkl      — best LR / Random Forest (main prediction)
    models/xgb_model.pkl  — XGBoost (Zone Recommendations)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib

MODELS_DIR   = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH   = MODELS_DIR / "model.pkl"
XGB_PATH     = MODELS_DIR / "xgb_model.pkl"


def main():
    print("=" * 60)
    print("  TaxiWise — Model Training")
    print("=" * 60)

    # ── 1. Regression model (LR / Random Forest) ──────────────────
    print("\n[1/2] Regression model (LR + Random Forest)")
    print("  Features :", [
        "pickup_location_id", "pickup_hour", "pickup_day_of_week",
        "pickup_month", "historical_trip_count", "avg_fare_amount",
        "avg_trip_distance", "avg_trip_duration", "year",
    ])
    print("  Target   : trip_count")
    print("  Split    : 80/20 random across all years")
    print()

    from src.regression import build_model_payload
    reg_payload = build_model_payload(verbose=True)
    joblib.dump(reg_payload, MODEL_PATH)

    m = reg_payload["metrics"]
    print(f"\n  ✅ Saved → {MODEL_PATH}")
    print(f"     Best model : {reg_payload['model_name']}")
    print(f"     MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R²={m['r2']:.4f}")
    print(f"     Train: {reg_payload['n_train']:,}  Test: {reg_payload['n_test']:,}")

    # ── 2. XGBoost model (Zone Recommendations) ───────────────────
    print("\n[2/2] XGBoost model (Zone Recommendations)")
    from src.data_loader import compute_demand
    from src.model import _train

    print("  Loading demand data …")
    demand = compute_demand()
    print(f"  Demand records: {len(demand):,}")
    print("  Training XGBoost …")

    xgb_model, xgb_metrics, xgb_fi, xgb_yte, xgb_ypred = _train(demand)

    xgb_payload = {
        "model":              xgb_model,
        "metrics":            xgb_metrics,
        "feature_importance": xgb_fi,
        "y_test":             xgb_yte,
        "y_pred":             xgb_ypred,
    }
    joblib.dump(xgb_payload, XGB_PATH)

    xm = xgb_metrics
    print(f"\n  ✅ Saved → {XGB_PATH}")
    print(f"     MAE={xm['mae']:.3f}  RMSE={xm['rmse']:.3f}  R²={xm['r2']:.4f}")
    print(f"     Train: {xm['n_train']:,}  Test: {xm['n_test']:,}")

    print("\n" + "=" * 60)
    print("  Both models saved. Now commit and push:")
    print("    git add models/")
    print("    git commit -m 'Add pre-trained model pkl files'")
    print("    git push")
    print("=" * 60)


if __name__ == "__main__":
    main()
