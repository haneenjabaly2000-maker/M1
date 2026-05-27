# 🚖 TaxiWise — NYC Yellow Taxi Demand Forecasting

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?style=flat-square)
![Years](https://img.shields.io/badge/Data-2023--2026-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-FF4B4B?style=flat-square)](https://hnb79jc2gumya5rprevmhx.streamlit.app/)

## 🚀 Live Demo

**[👉 פתח את האפליקציה החיה](https://hnb79jc2gumya5rprevmhx.streamlit.app/)**

---

מערכת Full-Stack חכמה לניתוח וחיזוי ביקוש לנסיעות מוניות בניו יורק (NYC Yellow Taxi) לשנים **2023–2026**.

| משתמש | תועלת |
|-------|--------|
| 🚖 נהגי מונית | לדעת באיזה אזור כדאי להמתין בכל שעה |
| 🏢 מנהלי צי | לתכנן פריסה אופטימלית של הצי |
| 📊 אנליסטים | לנתח מגמות ביקוש ולהשוות בין שנים |

---

## 🏗️ ארכיטקטורה

```
CSV/PARQUET 2023–2026
       ↓
  data_loader.py  ──  Feature Engineering  ──  Aggregation per (zone, hour, dow, month, year)
       ↓
  train_model.py  ──  Linear Regression + Random Forest  ──  models/model.pkl
       ↓
  app.py (Streamlit)  ──  8-page Dashboard + Interactive Prediction
```

---

## 📁 מבנה הפרויקט

```
TaxiWise/
├── data/
│   ├── raw/                      # קבצי PARQUET גולמיים
│   ├── processed/                # נתונים מנוקים
│   └── taxi_zone_lookup.csv      # 265 אזורי NYC TLC
├── src/
│   ├── utils.py                  # Logging + מחולל נתונים סינתטיים
│   ├── data_loader.py            # טעינה: PARQUET → CSV → Synthetic (2023–2026)
│   ├── charts.py                 # גרפי Plotly
│   ├── model.py                  # load_regression_model + XGBoost (zone recs)
│   ├── regression.py             # build_model_payload + chart functions
│   └── clustering.py             # KMeans + PCA + Elbow
├── models/
│   └── model.pkl                 # מודל מאומן (נוצר ע"י train_model.py)
├── train_model.py                # סקריפט אימון עצמאי
├── app.py                        # Streamlit Dashboard (8 עמודים)
├── yellow_taxi_2025.csv          # נתוני 2025 (CSV אמיתי)
├── yellow_taxi_2026.csv          # נתוני 2026 (CSV — Test set)
├── findings.md                   # דוח ממצאים
├── requirements.txt
└── README.md
```

---

## 🚀 התחלה מהירה

### 1. התקנת תלויות

```bash
cd TaxiWise
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. אימון המודל (מומלץ לפני הרצת האפליקציה)

```bash
python train_model.py
```

הפלט:
```
============================================================
  TaxiWise — Regression Model Training
============================================================
  2023: 52,416 demand records
  2024: 52,416 demand records
  2025: 61,230 demand records
  2026: 58,104 demand records   ← test set

Training Linear Regression …
  MAE=4.231  RMSE=6.847  R²=0.6312

Training Random Forest …
  MAE=1.204  RMSE=2.891  R²=0.9421

✅  Best: Random Forest  R²=0.9421

Saved → models/model.pkl
============================================================
```

> אם לא מריצים את הסקריפט, האפליקציה תאמן אוטומטית בעת ההפעלה.

### 3. הרצת הדשבורד

```bash
streamlit run app.py
# פתח: http://localhost:8501
```

---

## 🧠 מודל ML — Regression

### Target

```
trip_count  — מספר נסיעות מצרפי לכל (zone, hour, day_of_week, month, year)
```

### Features (X)

| Feature | תיאור |
|---------|-------|
| `pickup_location_id` | מזהה אזור האיסוף (1–265) |
| `pickup_hour` | שעת האיסוף (0–23) |
| `pickup_day_of_week` | יום בשבוע (0=Mon … 6=Sun) |
| `pickup_month` | חודש (1–12) |
| `historical_trip_count` | סך נסיעות היסטורי באזור |
| `avg_fare_amount` | תעריף ממוצע ($) |
| `avg_trip_distance` | מרחק ממוצע (מייל) |
| `avg_trip_duration` | משך ממוצע (דקות) |
| `year` | שנה (2023–2026) |

### Train / Test Split

```
Train set : 2023 + 2024 + 2025  (נתונים היסטוריים)
Test  set : 2026                 (future validation — אם קיים yellow_taxi_2026.csv)
Fallback  : sklearn train_test_split(test_size=0.2, random_state=42)
```

### מודלים ותהליך בחירה

1. **Linear Regression** — `StandardScaler` → `LinearRegression()`
2. **Random Forest**     — `RandomForestRegressor(n_estimators=150, max_depth=12)`

הבחירה מתבצעת לפי R² Score על ה-Test set. המודל הטוב יותר נשמר ב-`models/model.pkl`.

### Metrics טיפוסיים

| מדד | Linear Regression | Random Forest |
|-----|:-----------------:|:-------------:|
| MAE  | ~3–6 trips  | ~1–3 trips  |
| RMSE | ~5–9 trips  | ~2–5 trips  |
| R²   | ~0.55–0.70  | ~0.88–0.96  |

---

## 🔵 Clustering & PCA

עמוד ה-Clustering מאפשר ניתוח unsupervised על נתוני הביקוש:

- **KMeans** — בחירת K (2–8), תצוגת אשכולות צבעונית
- **StandardScaler** — Checkbox להפעלה/כיבוי של Normalization
- **Elbow Plot** — השוואה בין Normalized ו-Raw לבחירת K אופטימלי
- **PCA** — אם נבחרים יותר מ-2 Features, הורדה ל-2 מרכיבים עיקריים עם % variance

> Clustering הוא עמוד ניתוח עצמאי — הוא אינו המודל הראשי.

---

## 📊 דשבורד — 8 עמודים

| עמוד | תוכן |
|------|------|
| **Overview** | KPI Cards, Top Zones, Heatmap |
| **Historical Analytics** | גרפי זמן, Borough breakdown, פילטרים |
| **Year Comparison** | השוואת 2023–2026 |
| **Demand Prediction** | טופס אינטראקטיבי → חיזוי + רמת ביקוש |
| **Zone Recommendations** | Top-N אזורים חמים (XGBoost) |
| **Model Performance** | Metrics, Actual vs Predicted, Feature Importance |
| **Clustering** | KMeans + Elbow + PCA |
| **Regression** | השוואת Linear Regression vs Random Forest |

### פילטר שנה (Sidebar)
בחר שנה אחת עד ארבע (2023–2026) — כל הדפים יתעדכנו בהתאם (למעט Year Comparison שמציג תמיד הכל).

---

## 🔄 Prediction Workflow

```
User Input (zone, hour, day, month, year)
          ↓
  lookup historical stats from demand table
  (historical_trip_count, avg_fare, avg_distance, avg_duration)
          ↓
  load_regression_model()  →  models/model.pkl  (@st.cache_resource)
          ↓
  predict_regression(payload, features)
          ↓
  trip_count prediction  +  demand level (Low / Medium / High / Very High)
```

---

## 🗂️ Pipeline טעינת נתונים

עבור כל שנה (2023–2026) המערכת מבצעת:

1. **חיפוש PARQUET** — `data/raw/*{year}*.parquet`
2. **חיפוש CSV** — `yellow_taxi_{year}.csv` בתיקיית הפרויקט
3. **Fallback סינתטי** — 200,000 שורות ריאליסטיות עם seed לפי השנה

---

## 🛠️ טכנולוגיות

| שכבה | טכנולוגיה | גרסה |
|------|-----------|------|
| Frontend | Streamlit | 1.35 |
| ויזואליזציה | Plotly | 5.22 |
| עיבוד נתונים | Pandas + NumPy | 2.2 / 1.26 |
| ML (ראשי) | scikit-learn (LR + RF) | 1.5 |
| ML (zones) | XGBoost | 2.0 |
| Clustering | KMeans + PCA | scikit-learn |
| שמירת מודל | Joblib | 1.4 |
| נתונים | PyArrow (PARQUET) | 16.1 |

---

## 📄 רישיון

MIT License — חופשי לשימוש, שינוי והפצה.

---

**🚖 TaxiWise — Drive Smarter, Earn More**  
*נבנה עם Python, scikit-learn ו-Streamlit · נתוני NYC Yellow Taxi 2023–2026*
