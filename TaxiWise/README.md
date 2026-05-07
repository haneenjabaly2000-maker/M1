# 🚖 TaxiWise — NYC Yellow Taxi Demand Forecasting

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

מערכת Full-Stack חכמה לניתוח וחיזוי ביקוש לנסיעות מוניות בניו יורק (NYC Yellow Taxi).

---

## 🎯 מטרת המערכת

TaxiWise מנתחת נתוני נסיעות מוניות של NYC, מחזה ביקוש עתידי לפי אזור ושעה, וממליצה לנהגים באילו אזורים כדאי להתמקם.

| משתמש | תועלת |
|-------|--------|
| 🚖 נהגי מונית | לדעת באיזה אזור כדאי להמתין בכל שעה |
| 🏢 מנהלי צי | לתכנן פריסה אופטימלית של הצי |
| 📊 אנליסטים | לנתח מגמות ביקוש לאורך זמן |

---

## 🏗️ ארכיטקטורה

```
PARQUET Data → Preprocessing → Feature Engineering → XGBoost Model → Streamlit Dashboard
```

---

## 📁 מבנה הפרויקט

```
TaxiWise/
├── data/
│   ├── raw/                     # קבצי PARQUET גולמיים
│   ├── processed/               # נתונים מנוקים ומעובדים
│   └── taxi_zone_lookup.csv     # 265 אזורי NYC TLC
├── src/
│   ├── utils.py                 # כלים, logging, מחולל סינתטי
│   ├── preprocess.py            # ניקוי נתונים
│   ├── features.py              # Feature Engineering
│   ├── train.py                 # אימון XGBoost
│   └── predict.py               # חיזוי והמלצות
├── models/                      # מודלים שמורים
├── notebooks/                   # EDA
├── app.py                       # Streamlit Dashboard
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

### 2. הכנת נתונים

**אפשרות א׳ — נתונים סינתטיים (אוטומטי):**
אין צורך בפעולה — המערכת מייצרת 600,000 שורות אוטומטית.

**אפשרות ב׳ — נתונים אמיתיים:**
הורד קבצי PARQUET מ-[NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) ומקם ב-`data/raw/`.

### 3. הרצת Pipeline

```bash
python src/preprocess.py   # ניקוי נתונים
python src/features.py     # Feature Engineering
python src/train.py        # אימון מודל
```

### 4. הרצת הדשבורד

```bash
streamlit run app.py
# פתח: http://localhost:8501
```

---

## 🛠️ טכנולוגיות

| שכבה | טכנולוגיה | גרסה |
|------|-----------|------|
| Frontend | Streamlit | 1.35 |
| ויזואליזציה | Plotly | 5.22 |
| עיבוד נתונים | Pandas + NumPy | 2.2 / 1.26 |
| ML | XGBoost + Scikit-learn | 2.0 / 1.5 |
| נתונים | PyArrow (PARQUET) | 16.1 |
| שמירת מודל | Joblib | 1.4 |

---

## 📦 מודולים

### `src/utils.py`
- Logging מוגדר
- טעינת טבלת אזורים (265 zones)
- מחולל נתונים סינתטיים ריאליסטיים

### `src/preprocess.py`
סינון: נסיעות שגויות, ערכים חסרים, outliers, אזורים לא חוקיים, זמני נסיעה בלתי סבירים.

### `src/features.py`
פיצ'רים: `pickup_hour`, `pickup_day_of_week`, `pickup_month`, `is_weekend`, `is_rush_hour`, `trip_duration`, `historical_trip_count`, `avg_fare_amount`, `avg_trip_distance`, `avg_trip_duration`.

### `src/train.py`
- XGBoost Regressor (n=300, depth=6, lr=0.1)
- Train/Test Split 80/20 (time-based)
- מדדים: MAE, RMSE, R²
- שומר: `models/taxiwise_model.joblib`

### `src/predict.py`
- `predict_demand(zone, hour, day, month)` → ביקוש צפוי
- `get_top_zones(hour, day, month, top_n)` → אזורים מומלצים

---

## 🧠 מודל ML

**Input:**
```
pickup_location_id, pickup_hour, pickup_day_of_week,
pickup_month, is_weekend, is_rush_hour,
historical_trip_count, avg_fare_amount,
avg_trip_distance, avg_trip_duration
```

**Output:** `predicted_trip_demand` (מספר נסיעות צפוי)

**Feature Importance (סדר ירידה):**
1. `historical_trip_count`
2. `pickup_hour`
3. `pickup_location_id`
4. `pickup_day_of_week`
5. `is_rush_hour`

---

## 📊 דשבורד — 5 עמודים

| עמוד | תוכן |
|------|------|
| **Overview** | KPI Cards, Top Zones, Heatmap שעה×יום |
| **Historical Analytics** | גרפי זמן, השוואת 2025 vs 2026, Borough breakdown |
| **Demand Prediction** | בחירת אזור/שעה/יום → חיזוי ביקוש |
| **Zone Recommendations** | Top-N אזורים חמים עם הסברים |
| **Model Performance** | MAE/RMSE/R², Feature Importance, Residuals |

---

## 📊 נתוני NYC TLC

- **מקור:** [nyc.gov/site/tlc](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **פורמט:** PARQUET
- **תקופה:** 2025–2026
- **שדות:** pickup/dropoff datetime, LocationID, distance, fare, tip, payment_type

---

## 📈 ביצועי מודל (טיפוסי)

| מדד | ערך |
|-----|-----|
| MAE | ~2–5 נסיעות |
| RMSE | ~4–8 נסיעות |
| R² | 0.85–0.95 |

---

## 🔧 הגדרות מתקדמות (.env)

```env
OPENWEATHER_API_KEY=your_key   # Weather API (אופציונלי)
SYNTHETIC_ROWS=600000           # גודל נתונים סינתטיים
XGB_N_ESTIMATORS=300            # עצים במודל
```

---

## 📄 רישיון

MIT License — חופשי לשימוש, שינוי והפצה.

---

**🚖 TaxiWise — Drive Smarter, Earn More**
*נבנה עם Python, XGBoost ו-Streamlit*
