# 🚖 TaxiWise — NYC Yellow Taxi Demand Forecasting

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green?style=flat-square)
![Years](https://img.shields.io/badge/Data-2023--2025-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-FF4B4B?style=flat-square)](https://hnb79jc2gumya5rprevmhx.streamlit.app/)

## 🚀 Live Demo

**[👉 פתח את האפליקציה החיה](https://hnb79jc2gumya5rprevmhx.streamlit.app/)**

---

מערכת Full-Stack חכמה לניתוח וחיזוי ביקוש לנסיעות מוניות בניו יורק (NYC Yellow Taxi) לשנים **2023, 2024 ו-2025**.

---

## 🎯 מטרת המערכת

TaxiWise מנתחת נתוני נסיעות מוניות של NYC על פני שלוש שנים, מחזה ביקוש עתידי לפי אזור ושעה, וממליצה לנהגים באילו אזורים כדאי להתמקם.

| משתמש | תועלת |
|-------|--------|
| 🚖 נהגי מונית | לדעת באיזה אזור כדאי להמתין בכל שעה |
| 🏢 מנהלי צי | לתכנן פריסה אופטימלית של הצי |
| 📊 אנליסטים | לנתח מגמות ביקוש לאורך 3 שנים |

---

## 🏗️ ארכיטקטורה

```
PARQUET/CSV 2023+2024+2025 → Preprocessing → Feature Engineering → XGBoost Model → Streamlit Dashboard
```

---

## 📁 מבנה הפרויקט

```
TaxiWise/
├── data/
│   ├── raw/                     # קבצי PARQUET גולמיים (2023/2024/2025)
│   ├── processed/               # נתונים מנוקים ומעובדים
│   └── taxi_zone_lookup.csv     # 265 אזורי NYC TLC
├── src/
│   ├── utils.py                 # כלים, logging, מחולל סינתטי (2023-2025)
│   ├── data_loader.py           # טעינה מרובת שנים: PARQUET → CSV → Synthetic
│   ├── charts.py                # גרפי Plotly + גרפי השוואה בין שנים
│   ├── model.py                 # XGBoost מאומן על נתוני 3 שנים
│   └── predict.py               # חיזוי והמלצות
├── models/                      # מודלים שמורים
├── notebooks/                   # EDA
├── app.py                       # Streamlit Dashboard (6 עמודים)
├── yellow_taxi_2025.csv         # נתוני 2025 (CSV)
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
אין צורך בפעולה — המערכת מייצרת 200,000 שורות לכל שנה (2023/2024/2025) אוטומטית.

**אפשרות ב׳ — נתונים אמיתיים:**
הורד קבצי PARQUET מ-[NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) ומקם ב-`data/raw/`. הקבצים ייטענו אוטומטית לפי שנה.

**אפשרות ג׳ — קבצי CSV:**
מקם קבצים בשם `yellow_taxi_2023.csv`, `yellow_taxi_2024.csv`, `yellow_taxi_2025.csv` בתיקיית הפרויקט.

### 3. הרצת הדשבורד

```bash
streamlit run app.py
# פתח: http://localhost:8501
```

---

## 🗂️ Pipeline טעינת נתונים (Multi-Year)

עבור כל שנה (2023, 2024, 2025) המערכת מבצעת:

1. **חיפוש PARQUET** — `data/raw/*{year}*.parquet`
2. **חיפוש CSV** — `yellow_taxi_{year}.csv` בתיקיית הפרויקט
3. **Fallback סינתטי** — 200,000 שורות ריאליסטיות עם seed לפי השנה

לאחר הטעינה:
- חישוב `trip_duration_min` מ-datetime אם חסר
- הוספת שמות אזורים מ-`taxi_zone_lookup.csv`
- ניקוי נתונים אחיד (fare, distance, duration)
- איחוד לתוך DataFrame אחד עם עמודת `year`

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

## 📊 דשבורד — 6 עמודים

| עמוד | תוכן |
|------|------|
| **Overview** | KPI Cards, Top Zones, Heatmap שעה×יום |
| **Historical Analytics** | גרפי זמן, Borough breakdown, פילטר שנה/חודש/שעה |
| **Year Comparison** | השוואת נסיעות / מגמות / אזורים / תעריפים בין 2023, 2024, 2025 |
| **Demand Prediction** | בחירת אזור/שעה/יום → חיזוי ביקוש |
| **Zone Recommendations** | Top-N אזורים חמים עם הסברים |
| **Model Performance** | MAE/RMSE/R², Feature Importance, Residuals |

### פילטר שנה (Sidebar)
בחרי שנה אחת, שתיים, או את כולן — כל הדפים (מלבד Year Comparison) יתעדכנו בהתאם.

---

## 🧠 מודל ML

**Input:**
```
pickup_location_id, pickup_hour, pickup_day_of_week,
pickup_month, zone_total_trips, avg_fare_amount,
avg_trip_distance, avg_trip_duration
```

**Output:** `predicted_trip_demand` (מספר נסיעות צפוי)

המודל מאומן על נתונים היסטוריים משולבים מ-2023, 2024 ו-2025 לשיפור הדיוק.

**Feature Importance (סדר ירידה):**
1. `zone_total_trips`
2. `pickup_hour`
3. `PULocationID`
4. `pickup_day_of_week`
5. `avg_fare`

---

## 📊 נתוני NYC TLC

- **מקור:** [nyc.gov/site/tlc](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **פורמט:** PARQUET / CSV
- **תקופה:** 2023–2025
- **שדות:** pickup/dropoff datetime, LocationID, distance, fare, tip, payment_type

---

## 📈 ביצועי מודל (טיפוסי)

| מדד | ערך |
|-----|-----|
| MAE | ~2–5 נסיעות |
| RMSE | ~4–8 נסיעות |
| R² | 0.85–0.95 |

---

## 📄 רישיון

MIT License — חופשי לשימוי, שינוי והפצה.

---

**🚖 TaxiWise — Drive Smarter, Earn More**
*נבנה עם Python, XGBoost ו-Streamlit · נתוני NYC Yellow Taxi 2023–2025*
