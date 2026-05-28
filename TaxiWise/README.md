# 🚖 TaxiWise — Real-Time Transportation Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green?style=flat-square)
![Years](https://img.shields.io/badge/Data-2023--2026-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-FF4B4B?style=flat-square)](https://hnb79jc2gumya5rprevmhx.streamlit.app/)

---

## 🚀 Live Demo

**[👉 Open TaxiWise Live App](https://hnb79jc2gumya5rprevmhx.streamlit.app/)**

---

## 🧠 What Is TaxiWise?

TaxiWise is an **AI-powered Real-Time Transportation Intelligence Platform** for NYC Yellow Taxi data (2023–2026).  
It goes beyond a simple dashboard — it provides demand forecasting, relocation simulation, driver profit estimation, and interactive geo-demand mapping at the level of Uber / Waze-style intelligence systems.

| User | Value |
|------|-------|
| 🚖 Taxi Drivers | Know which zone to head to, and when, to maximize earnings |
| 🏢 Fleet Managers | Plan optimal fleet deployment across NYC boroughs |
| 📊 Data Analysts | Explore demand trends, compare years, audit model performance |
| 🎓 Researchers | Full ML pipeline: LR, RF, XGBoost, KMeans, PCA, SHAP-style explainability |

---

## 🏗️ Architecture

```
CSV / PARQUET 2023–2026
        ↓
  data_loader.py  ──  Feature Engineering  ──  Aggregation per (zone, hour, dow, month, year)
        ↓
  regression.py + model.py  ──  LR + Random Forest + XGBoost  ──  models/*.pkl
        ↓
  app.py (Streamlit)  ──  12-page AI Intelligence Platform
```

---

## 📁 Project Structure

```
TaxiWise/
├── data/
│   ├── raw/                      # Parquet files (2023–2026)
│   └── taxi_zone_lookup.csv      # 265 NYC TLC zones
├── models/
│   ├── model.pkl                 # Regression model (LR / RF) — auto-trained if missing
│   └── xgb_model.pkl             # XGBoost model for zone ranking — auto-trained if missing
├── src/
│   ├── data_loader.py            # Load: Parquet → CSV → Synthetic fallback
│   ├── model.py                  # Model loading, XGBoost training, zone recommendations
│   ├── regression.py             # LR + RF training, build_model_payload, charts
│   ├── clustering.py             # KMeans + PCA + Elbow method
│   ├── charts.py                 # Plotly chart library
│   ├── zone_coords.py            # Approximate NYC zone centroid coordinates (map)
│   └── utils.py                  # Synthetic data generator
├── app.py                        # 12-page Streamlit dashboard
├── train_model.py                # Offline model training script
├── prepare_data.py               # Pre-generate synthetic parquet data
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd TaxiWise
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train models (recommended before running)

```bash
python train_model.py
```

Output:
```
[1/2] Training Regression model (LR + RF) …
  Train: 180,000+ demand records  |  Test: 80/20 split
  Random Forest  MAE=1.2  RMSE=2.9  R²=0.942
  ✅ Saved → models/model.pkl

[2/2] Training XGBoost model …
  ✅ Saved → models/xgb_model.pkl
```

> If you skip this step, both models train automatically on first app launch.

### 3. Run the dashboard

```bash
streamlit run app.py
# Open: http://localhost:8501
```

---

## 📊 Dashboard — 12 Pages

| Page | Description |
|------|-------------|
| **📊 Overview** | KPI cards, Top zones, Demand heatmap, Data sample |
| **📈 Historical Analytics** | Demand patterns, Borough breakdown, interactive filters |
| **📅 Year Comparison** | 2023–2026 side-by-side: volume, fare, peak hours |
| **🔮 Demand Prediction** | Interactive form → trip-count forecast + demand level |
| **🤖 Real-Time Prediction** | Generalize to any input (unseen combos) · Confidence range · Alerts · Explainability |
| **🌡️ AI Demand Map** | Interactive Plotly map — zone demand heatmap with hover details |
| **🚗 Driver Tools** | Relocation Simulator + Profit Estimator — AI advice for drivers |
| **🔭 Demand Forecast** | 24-hour, Day-of-week, Monthly demand forecast per zone |
| **🗺️ Zone Recommendations** | Top-N hot zones ranked by XGBoost predicted demand |
| **⚙️ Model Performance** | Metrics, Actual vs Predicted, Feature Importance, Model config |
| **🔵 Clustering** | KMeans + Elbow + PCA — zone behavior clusters |
| **📉 Regression** | LR vs RF comparison with selectable features |

---

## 🌡️ AI Demand Map

Visualizes predicted trip demand across all NYC taxi zones for any selected hour, day, and month.

- **Technology**: Plotly Mapbox (`carto-darkmatter` — no API key needed)
- **Color scale**: 🔵 Blue = low demand → 🟠 Orange = medium → 🔴 Red = high
- **Hover**: Zone name · Borough · Predicted demand · Avg fare · Avg distance · Peak hour
- **Zone coordinates**: Golden-ratio spread within borough bounding boxes

---

## 🚗 Driver Relocation Simulator

Compares two zones at any time window and recommends whether to relocate.

**Output:**
```
Moving from Lower East Side → Midtown Center
  ▲ +42 trips/hr (+35%)     ▲ +$18.40/hr revenue
  ✅ Strongly Recommended
```

Factors considered: demand delta, revenue per trip, driver share (adjustable 50–100%).

---

## 💰 Driver Profit Estimator

For a chosen zone, day, and working-hour window — estimates total trips and revenue.

- Configurable driver share (default 70%)
- Hourly bar chart + revenue overlay
- Revenue = Predicted Trips × Avg Fare × Driver Share

---

## 🔭 Demand Forecast

Three forecast views per zone:

| Tab | What it shows |
|-----|--------------|
| ⏰ 24-Hour | Demand for every hour of the selected day |
| 📅 Day-of-Week | Demand by day (Mon–Sun) at 18:00 |
| 🗓️ Monthly | Demand by month (Jan–Dec) at 18:00 |

Supports future year predictions through **2035** via model extrapolation.

---

## 🤖 Real-Time AI Prediction

Accepts **any input combination** — including combinations never seen in training data — and generalizes using the trained model.

- Inputs: zone, hour, day, month, year (2023–2035), fare, distance, duration, historical count, passenger count
- Output: predicted trip count, demand level badge, confidence range (RF P10–P90)
- **Smart Alerts**:
  - 🔴 Extreme Demand — surge pricing likely
  - 🟠 High Demand Alert — good driver opportunity
  - 🟡 Normal Demand — stable conditions
- **Explainability cards**: Rush hour type, weekday/weekend, zone percentile, distance vs median, seasonal context

---

## 🧠 ML Models

### Regression Model (LR / Random Forest)
**Target**: `trip_count` — aggregated pickups per (zone, hour, day, month, year)

| Feature | Description |
|---------|-------------|
| `pickup_location_id` | Zone ID (1–265) |
| `pickup_hour` | Hour of day (0–23) |
| `pickup_day_of_week` | Day (0=Mon … 6=Sun) |
| `pickup_month` | Month (1–12) |
| `historical_trip_count` | Total zone trips (historical) |
| `avg_fare_amount` | Avg fare in zone ($) |
| `avg_trip_distance` | Avg distance (miles) |
| `avg_trip_duration` | Avg duration (minutes) |
| `year` | Year (2023–2026+) |

**Split**: 80/20 random across all years (avoids R²<0 from synthetic/real scale mismatch)

| Metric | Linear Regression | Random Forest |
|--------|:-----------------:|:-------------:|
| MAE  | ~3–6 trips  | ~1–3 trips  |
| RMSE | ~5–9 trips  | ~2–5 trips  |
| R²   | ~0.55–0.70  | ~0.88–0.96  |

### XGBoost Model (Zone Recommendations)
Separate model trained on demand features. Used for hot-zone ranking only.  
`n_estimators=100, max_depth=6, learning_rate=0.05`

---

## 🔵 Clustering

Unsupervised zone analysis with:
- **KMeans** (K selectable 2–8)
- **Elbow Method** — normalized vs raw comparison
- **PCA** — 2D projection when >2 features selected
- **Cluster Statistics** — feature means per cluster

---

## 🛠️ Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Frontend | Streamlit | 1.35 |
| Visualization | Plotly (go + express) | 5.22 |
| AI Map | Plotly Mapbox (carto-darkmatter) | — |
| Data | Pandas + NumPy | 2.2 / 1.26 |
| ML — Main | scikit-learn (LR + RF) | 1.5 |
| ML — Zones | XGBoost | 2.0 |
| Clustering | KMeans + PCA (sklearn) | 1.5 |
| Model Persistence | Joblib | 1.4 |
| Data Format | PyArrow (Parquet) | 16.1 |

---

## 🔄 Data Loading Pipeline

For each year (2023–2026), the system tries in order:

1. **Parquet** — `data/raw/*{year}*.parquet`
2. **CSV** — `yellow_taxi_{year}.csv` in project root
3. **Merged CSV** — `yellow_taxi_2023_2024_small_merged.csv` (real data for 2023–2024)
4. **Synthetic fallback** — 20k realistic rows (run `prepare_data.py` to avoid this)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

**🚖 TaxiWise — Drive Smarter, Earn More**  
*Built with Python, scikit-learn, XGBoost, and Streamlit · NYC Yellow Taxi 2023–2026*
