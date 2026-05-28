"""
TaxiWise — NYC Yellow Taxi Demand Forecasting
Production-ready Streamlit application
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import numpy as np
import pandas as pd

# ── Must be the very first Streamlit call ─────────────────────────────────────
st.set_page_config(
    page_title    = "TaxiWise — NYC Taxi Analytics",
    page_icon     = "🚕",
    layout        = "wide",
    initial_sidebar_state = "expanded",
)

# ── Src imports ───────────────────────────────────────────────────────────────
from src.data_loader import load_trips, load_zones, compute_demand, compute_kpis
from src.model       import (load_xgb_model, get_hot_zones,
                              load_regression_model, predict_regression)
import src.charts     as charts
import src.clustering as clust
import src.regression as reg

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark-mode polish
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── Layout ── */
.main                  { background: #0E1117; }
.block-container       { padding: 1.4rem 2.2rem; max-width: 1400px; }
section[data-testid="stSidebar"] {
    background: #12141E;
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* ── Page titles ── */
.page-title {
    font-size: 2rem; font-weight: 800; line-height: 1.1;
    background: linear-gradient(90deg, #F7C948 0%, #3B82F6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: .25rem;
}
.page-subtitle { color: #9CA3AF; font-size: .88rem; margin-bottom: 1.4rem; }

/* ── Section headers ── */
.section-hdr {
    font-size: 1.05rem; font-weight: 700; color: #FAFAFA;
    border-left: 3px solid #F7C948; padding-left: 10px;
    margin: 1.6rem 0 .9rem;
}

/* ── KPI grid ── */
.kpi-grid { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: .8rem; }
.kpi-card {
    background: #1A1D27;
    border: 1px solid rgba(247,201,72,.10);
    border-radius: 14px; padding: 16px 20px;
    flex: 1; min-width: 145px;
    transition: transform .18s, border-color .18s;
}
.kpi-card:hover { transform: translateY(-2px); border-color: rgba(247,201,72,.32); }
.kpi-icon  { font-size: 1.55rem; margin-bottom: 5px; }
.kpi-value { font-size: 1.55rem; font-weight: 800; color: #F7C948; line-height: 1; }
.kpi-label { font-size: .72rem; color: #9CA3AF; text-transform: uppercase;
             letter-spacing: .06em; margin-top: 5px; }
.kpi-sub   { font-size: .68rem; color: #6B7280; margin-top: 2px; }

/* ── Prediction result card ── */
.pred-card {
    background: linear-gradient(135deg, #1A1D27, #252836);
    border: 1.5px solid rgba(247,201,72,.22);
    border-radius: 16px; padding: 22px 26px; text-align: center;
}
.pred-value { font-size: 3rem; font-weight: 800; color: #F7C948; }
.pred-label { color: #9CA3AF; font-size: .82rem; margin-top: 4px; }

/* ── Metric pills ── */
.pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.pill {
    background: #252836; border: 1px solid rgba(255,255,255,.07);
    border-radius: 8px; padding: 8px 14px; text-align: center; min-width: 105px;
}
.pill-val { font-size: 1.1rem; font-weight: 700; color: #F7C948; }
.pill-lbl { font-size: .68rem; color: #9CA3AF; text-transform: uppercase; margin-top: 2px; }

/* ── Zone recommendation cards ── */
.zone-card {
    background: #1A1D27; border-left: 4px solid #F7C948;
    border-radius: 0 10px 10px 0; padding: 12px 16px; margin-bottom: 9px;
}
.zone-card.r1 { border-left-color: #EF4444; }
.zone-card.r2 { border-left-color: #F97316; }
.zone-card.r3 { border-left-color: #F7C948; }
.zone-name  { font-size: .97rem; font-weight: 700; color: #FAFAFA; }
.zone-badge {
    display: inline-block; background: rgba(247,201,72,.12);
    color: #F7C948; font-size: .65rem; font-weight: 600;
    padding: 1px 7px; border-radius: 20px; margin-left: 6px;
}
.zone-stats  { color: #9CA3AF; font-size: .78rem; margin-top: 4px; }
.zone-reason { color: #6EE7B7; font-size: .75rem; margin-top: 3px; }

/* ── Info banner ── */
.info-banner {
    background: rgba(59,130,246,.10); border: 1px solid rgba(59,130,246,.25);
    border-radius: 10px; padding: 10px 16px; color: #93C5FD;
    font-size: .82rem; margin-bottom: 1rem;
}

/* ── Tabs ── */
[data-testid="stTabs"] button           { font-weight: 600; color: #9CA3AF !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #F7C948 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar       { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #12141E; }
::-webkit-scrollbar-thumb { background: #2D3044; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Data bootstrap ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _bootstrap():
    return load_trips(), load_zones(), compute_demand()


with st.spinner("Loading NYC Taxi data …"):
    try:
        df_all, zones, demand = _bootstrap()
        if df_all.empty:
            st.error("הנתונים לא נטענו. הרץ `python prepare_data.py` מקומית ודחוף את הקבצים ל-GitHub.")
            st.stop()
    except Exception as _e:
        st.error(f"שגיאה בטעינת נתונים: {_e}")
        st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 18px">
      <div style="font-size:1.55rem;font-weight:800;color:#F7C948;">🚕 TaxiWise</div>
      <div style="color:#9CA3AF;font-size:.75rem;margin-top:2px;">
        NYC Taxi Demand Intelligence
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    PAGES = {
        "📊  Overview":              "overview",
        "📈  Historical Analytics":  "analytics",
        "📅  Year Comparison":       "comparison",
        "🔮  Demand Prediction":     "prediction",
        "🤖  Real-Time Prediction":  "realtime",
        "🌡️  AI Demand Map":         "demand_map",
        "🚗  Driver Tools":          "driver",
        "🔭  Demand Forecast":       "forecast",
        "🗺️  Zone Recommendations":  "zones",
        "⚙️  Model Performance":     "performance",
        "🔵  Clustering":            "clustering",
        "📉  Regression":            "regression",
    }
    page_key = PAGES[
        st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    ]

    st.markdown("---")
    st.markdown(
        '<div style="color:#9CA3AF;font-size:.75rem;font-weight:600;'
        'letter-spacing:.05em;margin-bottom:6px;">YEAR FILTER</div>',
        unsafe_allow_html=True,
    )
    sel_years = st.multiselect(
        "Select years",
        options=[2023, 2024, 2025, 2026],
        default=[2023, 2024, 2025, 2026],
        label_visibility="collapsed",
    )


# ── Apply year filter ─────────────────────────────────────────────────────────
active_years = sorted(sel_years) if sel_years else [2023, 2024, 2025, 2026]
df = (
    df_all[df_all["year"].isin(active_years)].reset_index(drop=True)
    if active_years else df_all
)
kpis      = compute_kpis(df)
years_str = " · ".join(str(y) for y in active_years)

with st.sidebar:
    st.markdown("---")
    st.markdown(f"""
    <div style="color:#6B7280;font-size:.71rem;line-height:1.7;">
      <b style="color:#9CA3AF;">Data</b><br>
      NYC TLC Yellow Taxi<br>
      {years_str}<br>
      {kpis['total_trips']:,} trips · 28 features<br><br>
      <b style="color:#9CA3AF;">Model</b><br>
      LR / Random Forest Regression<br>
      Train 2023–2025 · Test 2026
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _kpi_row(items: list[tuple]) -> None:
    html = '<div class="kpi-grid">'
    for icon, value, label, sub in items:
        html += (
            f'<div class="kpi-card">'
            f'  <div class="kpi-icon">{icon}</div>'
            f'  <div class="kpi-value">{value}</div>'
            f'  <div class="kpi-label">{label}</div>'
            + (f'<div class="kpi-sub">{sub}</div>' if sub else "")
            + "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _section(title: str) -> None:
    st.markdown(f'<div class="section-hdr">{title}</div>', unsafe_allow_html=True)


def _pchart(fig, height: int | None = None, **kw):
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False}, **kw)


# ═════════════════════════════════════════════════════════════════════════════
# Page 1 — Overview
# ═════════════════════════════════════════════════════════════════════════════

def page_overview():
    st.markdown('<div class="page-title">TaxiWise Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-subtitle">Real-time insights · NYC Yellow Taxi {years_str}</div>',
        unsafe_allow_html=True,
    )

    _kpi_row([
        ("🚖",  f"{kpis['total_trips']:,}",       "Total Trips",       ""),
        ("💰",  f"${kpis['avg_fare']:.2f}",         "Avg Total Fare",    "per trip"),
        ("📍",  f"{kpis['avg_distance']:.1f} mi",   "Avg Distance",      "per trip"),
        ("⏱️", f"{kpis['avg_duration']:.1f} min",  "Avg Duration",      "per trip"),
        ("⚡",  f"{kpis['peak_hour']}:00",           "Peak Hour",         "most demand"),
        ("🗺️", f"{kpis['active_zones']}",           "Active Zones",      "pickup areas"),
    ])
    _kpi_row([
        ("💳",  f"{kpis['credit_pct']:.1f}%",         "Credit Card",       "of payments"),
        ("🏆",  kpis['top_zone'][:30],                 "Busiest Zone",      ""),
        ("💵",  f"${kpis['total_revenue']:,.0f}",      "Total Revenue",     "gross fares"),
    ])

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1: _pchart(charts.trips_by_hour(df))
    with c2: _pchart(charts.trips_by_dow(df))

    c3, c4 = st.columns(2)
    with c3: _pchart(charts.monthly_trend(df))
    with c4: _pchart(charts.top_zones(df, top_n=10))

    _section("Data Sample")
    cols = [c for c in [
        "tpep_pickup_datetime", "pickup_zone", "dropoff_zone",
        "trip_distance", "fare_amount", "tip_amount",
        "total_amount", "payment_label", "passenger_count",
    ] if c in df.columns]
    rename = {
        "tpep_pickup_datetime": "Pickup Time",
        "pickup_zone":          "Pickup Zone",
        "dropoff_zone":         "Dropoff Zone",
        "trip_distance":        "Distance (mi)",
        "fare_amount":          "Fare ($)",
        "tip_amount":           "Tip ($)",
        "total_amount":         "Total ($)",
        "payment_label":        "Payment",
        "passenger_count":      "Passengers",
    }
    st.dataframe(
        df[cols].head(20).rename(columns=rename),
        use_container_width=True, hide_index=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Page 2 — Historical Analytics
# ═════════════════════════════════════════════════════════════════════════════

def page_analytics():
    st.markdown('<div class="page-title">Historical Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Deep-dive into patterns · filtered by borough, month & hour</div>',
        unsafe_allow_html=True,
    )

    with st.expander("🎛️  Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            boroughs = ["All"] + sorted(df["pickup_borough"].dropna().unique()) \
                       if "pickup_borough" in df.columns else ["All"]
            sel_boro = st.selectbox("Borough", boroughs)
        with f2:
            mon_names = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]
            avail_months = sorted(int(m) for m in df["month"].dropna().unique() if 1 <= int(m) <= 12)
            avail_labels = [mon_names[m - 1] for m in avail_months]
            sel_m_labels = st.multiselect("Month", avail_labels, default=avail_labels)
            sel_months   = [avail_months[avail_labels.index(l)] for l in sel_m_labels] \
                           if sel_m_labels else avail_months
        with f3:
            hr_range = st.slider("Hour range", 0, 23, (0, 23))

    fdf = df.copy()
    if sel_boro != "All" and "pickup_borough" in fdf.columns:
        fdf = fdf[fdf["pickup_borough"] == sel_boro]
    fdf = fdf[fdf["month"].isin(sel_months)]
    fdf = fdf[(fdf["hour"] >= hr_range[0]) & (fdf["hour"] <= hr_range[1])]

    if fdf.empty:
        st.warning("No trips match the current filters.")
        return

    st.markdown(
        f'<div class="info-banner">'
        f'Showing <b>{len(fdf):,}</b> trips · Years: <b>{years_str}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _section("Demand Patterns")
    _pchart(charts.demand_heatmap(fdf))

    c1, c2 = st.columns(2)
    with c1: _pchart(charts.trips_by_hour(fdf))
    with c2: _pchart(charts.trips_by_dow(fdf))

    _section("Geographic Analysis")
    if "pickup_borough" in fdf.columns:
        _pchart(charts.borough_flow(fdf))

    c3, c4 = st.columns(2)
    with c3: _pchart(charts.top_zones(fdf, top_n=15))
    with c4: _pchart(charts.scatter_dist_fare(fdf))

    _section("Fare & Speed Analysis")
    _pchart(charts.fare_dist(fdf))

    c5, c6 = st.columns(2)
    with c5: _pchart(charts.speed_by_hour(fdf))
    with c6: _pchart(charts.tip_by_hour(fdf))


# ═════════════════════════════════════════════════════════════════════════════
# Page 3 — Year Comparison
# ═════════════════════════════════════════════════════════════════════════════

def page_comparison():
    st.markdown('<div class="page-title">Year Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'השוואה מלאה בין 2023 · 2024 · 2025 · 2026 — מגמות ביקוש, אזורים ותעריפים'
        '</div>',
        unsafe_allow_html=True,
    )

    _section("Trip Volume by Year")
    c1, c2 = st.columns(2)
    with c1: _pchart(charts.yearly_trip_comparison(df_all))
    with c2: _pchart(charts.yearly_monthly_trend(df_all))

    _section("Peak Hour Patterns")
    _pchart(charts.yearly_peak_hours(df_all), height=400)

    _section("Geographic & Fare Trends")
    c3, c4 = st.columns(2)
    with c3: _pchart(charts.yearly_top_zones(df_all, top_n=10))
    with c4: _pchart(charts.yearly_fare_trend(df_all))

    _section("Year-over-Year Summary")
    yearly = df_all.groupby("year").agg(
        trips        =("fare_amount",      "count"),
        avg_fare     =("fare_amount",      "mean"),
        avg_distance =("trip_distance",    "mean"),
        avg_duration =("trip_duration_min","mean"),
    ).reset_index()
    yearly.columns = ["Year", "Total Trips", "Avg Fare ($)", "Avg Distance (mi)", "Avg Duration (min)"]
    yearly["Year"]             = yearly["Year"].astype(str)
    yearly["Total Trips"]      = yearly["Total Trips"].apply(lambda x: f"{x:,}")
    yearly["Avg Fare ($)"]     = yearly["Avg Fare ($)"].apply(lambda x: f"${x:.2f}")
    yearly["Avg Distance (mi)"]= yearly["Avg Distance (mi)"].apply(lambda x: f"{x:.1f}")
    yearly["Avg Duration (min)"]= yearly["Avg Duration (min)"].apply(lambda x: f"{x:.1f}")
    st.dataframe(yearly, use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="info-banner" style="margin-top:1rem;">'
        '💡 דף זה מציג תמיד את כל השנים ללא תלות בפילטר השנה בסרגל הצד.'
        '</div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Page 4 — Demand Prediction  (main regression model)
# ═════════════════════════════════════════════════════════════════════════════

def page_prediction():
    st.markdown('<div class="page-title">Demand Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Interactive trip-demand forecast · Regression model trained on 2023–2025 · Validated on 2026'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading regression model …"):
        payload = load_regression_model()

    m          = payload["metrics"]
    model_name = payload["model_name"]

    _kpi_row([
        ("🤖", model_name,                          "Active Model",  "best by R²"),
        ("📉", f"{m['mae']:.2f}",                   "MAE",           "trips error"),
        ("📊", f"{m['rmse']:.2f}",                  "RMSE",          "trips error"),
        ("📈", f"{m['r2']:.3f}",                    "R² Score",      "variance explained"),
        ("🏋️", f"{payload['n_train']:,}",           "Train rows",    "2023–2025"),
        ("🧪", f"{payload['n_test']:,}",            "Test rows",
         "2026" if payload["has_2026"] else "80/20 split"),
    ])

    st.markdown("---")
    col_form, col_res = st.columns([1, 1], gap="large")

    with col_form:
        _section("Input Parameters")

        zone_opts = zones[["LocationID", "Zone", "Borough"]].copy()
        zone_opts["label"] = zone_opts.apply(
            lambda r: f"{r['Zone']} ({r['Borough']}) — ID {r['LocationID']}", axis=1
        )
        zone_label = st.selectbox("Pickup Zone", zone_opts["label"].tolist(), index=0)
        loc_id     = int(zone_label.split("— ID ")[-1])

        hour = st.slider("Hour of Day", 0, 23, 8)

        dow_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_sel   = st.selectbox("Day of Week", dow_names)
        dow_num   = dow_names.index(dow_sel)

        mon_names_full = ["January","February","March","April","May","June",
                          "July","August","September","October","November","December"]
        month_sel = st.selectbox("Month", mon_names_full)
        month_num = mon_names_full.index(month_sel) + 1

        year_sel = st.selectbox("Year", [2023, 2024, 2025, 2026], index=3)

    # ── Historical zone stats (from demand table) ─────────────────────────────
    z_hist = demand[demand["PULocationID"] == loc_id]
    if not z_hist.empty:
        hist_count = float(z_hist["zone_total_trips"].iloc[0])
        avg_fare   = float(z_hist["avg_fare"].mean())
        avg_dist   = float(z_hist["avg_distance"].mean())
        avg_dur    = float(z_hist["avg_duration"].mean())
    else:
        hist_count = float(demand["zone_total_trips"].mean())
        avg_fare   = float(demand["avg_fare"].mean())
        avg_dist   = float(demand["avg_distance"].mean())
        avg_dur    = float(demand["avg_duration"].mean())

    features = {
        "pickup_location_id":    loc_id,
        "pickup_hour":           hour,
        "pickup_day_of_week":    dow_num,
        "pickup_month":          month_num,
        "historical_trip_count": hist_count,
        "avg_fare_amount":       avg_fare,
        "avg_trip_distance":     avg_dist,
        "avg_trip_duration":     avg_dur,
        "year":                  year_sel,
    }

    pred = predict_regression(payload, features)

    # ── Demand level (percentile-based) ───────────────────────────────────────
    y_test = payload["y_test"]
    p25 = float(np.percentile(y_test, 25))
    p75 = float(np.percentile(y_test, 75))
    p90 = float(np.percentile(y_test, 90))

    if pred < p25:
        level, lvl_color = "Low",       "#6B7280"
    elif pred < p75:
        level, lvl_color = "Medium",    "#F7C948"
    elif pred < p90:
        level, lvl_color = "High",      "#F97316"
    else:
        level, lvl_color = "Very High", "#EF4444"

    # ── Historical avg for comparison ─────────────────────────────────────────
    hist_rows = demand[(demand["PULocationID"] == loc_id) & (demand["hour"] == hour)]
    hist_avg  = float(hist_rows["trip_count"].mean()) if len(hist_rows) > 0 else 0.0
    diff_pct  = ((pred - hist_avg) / max(hist_avg, 1)) * 100

    with col_res:
        _section("Prediction Result")
        arrow    = "▲" if diff_pct >= 0 else "▼"
        clr_diff = "#10B981" if diff_pct >= 0 else "#EF4444"
        st.markdown(f"""
        <div class="pred-card">
          <div class="pred-value">{pred:.0f}</div>
          <div class="pred-label">Predicted trips / hour</div>
          <div style="margin-top:10px;">
            <span style="background:{lvl_color};color:#fff;font-weight:700;
                   padding:4px 16px;border-radius:20px;font-size:.85rem;">
              {level} Demand
            </span>
          </div>
          <div style="margin-top:12px;font-size:.8rem;color:#9CA3AF;">
            Historical avg at {hour}:00 →
            <b style="color:#FAFAFA;">{hist_avg:.0f} trips</b>
          </div>
          <div style="margin-top:5px;font-size:.82rem;color:{clr_diff};">
            {arrow} {abs(diff_pct):.1f}% vs. historical average
          </div>
        </div>
        """, unsafe_allow_html=True)

        pills = [
            ("Avg Fare",     f"${avg_fare:.2f}"),
            ("Avg Distance", f"{avg_dist:.1f} mi"),
            ("Avg Duration", f"{avg_dur:.1f} min"),
            ("Zone Hist.",   f"{int(hist_count):,}"),
        ]
        pill_html = '<div class="pill-row">'
        for lbl, val in pills:
            pill_html += (
                f'<div class="pill">'
                f'  <div class="pill-val">{val}</div>'
                f'  <div class="pill-lbl">{lbl}</div>'
                f'</div>'
            )
        pill_html += "</div>"
        st.markdown(pill_html, unsafe_allow_html=True)

    # ── Model comparison charts ───────────────────────────────────────────────
    if "all_metrics" in payload:
        st.markdown("---")
        _section("Model Comparison (training results)")
        mc1, mc2 = st.columns(2)
        with mc1: _pchart(reg.chart_metrics_bar(payload["all_metrics"]))
        with mc2: _pchart(reg.chart_r2_bar(payload["all_metrics"]))

    # ── Feature importance ────────────────────────────────────────────────────
    if payload.get("feature_importance") is not None:
        _section("Feature Importance")
        _pchart(reg.chart_feature_importance(payload["feature_importance"], model_name))


# ═════════════════════════════════════════════════════════════════════════════
# Page 4 — Zone Recommendations
# ═════════════════════════════════════════════════════════════════════════════

def page_recommendations():
    st.markdown('<div class="page-title">Zone Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Find the hottest pickup zones for any time window</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading model …"):
        model, metrics, fi, y_te, y_pred = load_xgb_model()

    _section("Time Parameters")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        rec_hour = st.slider("Hour", 0, 23, 18, key="rh")
    with r2:
        dow_names = ["Monday","Tuesday","Wednesday","Thursday",
                     "Friday","Saturday","Sunday"]
        rec_dow_lbl = st.selectbox("Day", dow_names, index=4, key="rd")
        rec_dow     = dow_names.index(rec_dow_lbl)
    with r3:
        mon_full = ["January","February","March","April","May","June",
                    "July","August","September","October","November","December"]
        rec_mon_lbl = st.selectbox("Month", mon_full, index=0, key="rm")
        rec_month   = mon_full.index(rec_mon_lbl) + 1
    with r4:
        top_n = st.slider("Top N zones", 5, 20, 10, key="tn")

    with st.spinner("Analysing zones …"):
        hot = get_hot_zones(model, demand, zones, rec_hour, rec_dow, rec_month, top_n)

    if hot.empty:
        st.warning("No data for the selected time window.")
        return

    # Chart + table side-by-side
    ch1, ch2 = st.columns([2, 1])
    with ch1:
        _pchart(charts.hot_zones_chart(hot))
    with ch2:
        _section("Rankings")
        disp = hot[["Zone","Borough","predicted_demand","avg_fare","avg_distance"]].copy()
        disp.columns = ["Zone","Borough","Demand","Avg Fare","Avg Dist"]
        disp["Demand"]   = disp["Demand"].round(1)
        disp["Avg Fare"] = disp["Avg Fare"].apply(lambda x: f"${x:.2f}")
        disp["Avg Dist"] = disp["Avg Dist"].apply(lambda x: f"{x:.1f} mi")
        st.dataframe(disp, use_container_width=True)

    # Detail cards
    _section("Top Zone Details")
    border_cls = ["r1", "r2", "r3"]
    rank_emoji = ["🥇 #1", "🥈 #2", "🥉 #3"]

    cards_html = ""
    for rank_i, (_, row) in enumerate(hot.head(min(5, len(hot))).iterrows()):
        cls        = border_cls[rank_i] if rank_i < 3 else ""
        rank_label = rank_emoji[rank_i]  if rank_i < 3 else f"#{rank_i + 1}"
        zone_name  = str(row.get("Zone", f"Zone {row['PULocationID']}"))
        borough    = str(row.get("Borough", ""))
        pred_d     = float(row["predicted_demand"])
        a_fare     = float(row["avg_fare"])
        a_dist     = float(row["avg_distance"])
        a_dur      = float(row["avg_duration"])
        hist_tot   = int(row["zone_total_trips"])

        reasons: list[str] = []
        if hist_tot > demand["zone_total_trips"].quantile(0.75):
            reasons.append("High historical demand")
        if a_fare > demand["avg_fare"].mean():
            reasons.append(f"Above-avg fare (${a_fare:.0f})")
        if a_dist > demand["avg_distance"].mean():
            reasons.append("Longer trips → more revenue")
        if not reasons:
            reasons.append("Strong demand at this time")

        cards_html += f"""
        <div class="zone-card {cls}">
          <div class="zone-name">{rank_label}&nbsp; {zone_name}
            <span class="zone-badge">{borough}</span>
          </div>
          <div class="zone-stats">
            🔮 {pred_d:.0f} trips/hr &nbsp;·&nbsp;
            💰 ${a_fare:.2f} avg &nbsp;·&nbsp;
            📍 {a_dist:.1f} mi &nbsp;·&nbsp;
            ⏱️ {a_dur:.1f} min
          </div>
          <div class="zone-reason">✅ {" · ".join(reasons)}</div>
        </div>"""

    st.markdown(cards_html, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Page 5 — Model Performance  (regression model)
# ═════════════════════════════════════════════════════════════════════════════

def page_performance():
    st.markdown('<div class="page-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Regression model diagnostics · train 2023–2025 · test 2026'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading model …"):
        payload = load_regression_model()

    m          = payload["metrics"]
    model_name = payload["model_name"]

    _section("Performance Metrics")
    _kpi_row([
        ("🤖", model_name,              "Active Model",  "best by R²"),
        ("📉", f"{m['mae']:.2f}",       "MAE",           "mean absolute error"),
        ("📊", f"{m['rmse']:.2f}",      "RMSE",          "root mean sq. error"),
        ("📈", f"{m['r2']:.3f}",        "R² Score",      "explained variance"),
        ("🏋️", f"{payload['n_train']:,}", "Train rows",  "2023–2025"),
        ("🧪", f"{payload['n_test']:,}",  "Test rows",
         "2026" if payload["has_2026"] else "80/20 split"),
    ])

    r2 = m["r2"]
    if r2 > 0.85:
        badge, note = "🟢 **Excellent**", "Model explains >85% of demand variance."
    elif r2 > 0.70:
        badge, note = "🟡 **Good**",      "Solid predictive power for demand forecasting."
    else:
        badge, note = "🔴 **Fair**",      "More historical data would improve accuracy."

    st.info(
        f"{badge} — {note}  "
        f"MAE = **{m['mae']:.2f}** trips means predictions are off "
        f"by ~{m['mae']:.0f} trips on average."
    )

    st.markdown("---")

    # ── All-models comparison ─────────────────────────────────────────────────
    if "all_metrics" in payload:
        _section("All Models Comparison")
        rows = [
            {
                "Model": name,
                "MAE":  round(met["mae"],  3),
                "RMSE": round(met["rmse"], 3),
                "R²":   round(met["r2"],   4),
                "Best": "✅" if name == model_name else "",
            }
            for name, met in payload["all_metrics"].items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        ac1, ac2 = st.columns(2)
        with ac1: _pchart(reg.chart_metrics_bar(payload["all_metrics"]))
        with ac2: _pchart(reg.chart_r2_bar(payload["all_metrics"]))

    # ── Diagnostics ───────────────────────────────────────────────────────────
    _section("Diagnostics — Best Model")
    y_te   = payload["y_test"]
    y_pred = payload["y_pred"]

    dc1, dc2 = st.columns(2)
    with dc1:
        _pchart(reg.chart_actual_vs_pred(y_te, y_pred, model_name))
    with dc2:
        if payload.get("feature_importance") is not None:
            _pchart(reg.chart_feature_importance(payload["feature_importance"], model_name))
        else:
            st.info("Feature importance is not available for Linear Regression.")

    # ── Feature importance table ──────────────────────────────────────────────
    if payload.get("feature_importance") is not None:
        _section("Feature Importance Details")
        fi = payload["feature_importance"].copy()
        fi["importance_pct"] = (fi["importance"] / fi["importance"].sum() * 100).round(1)
        fi_disp = fi[["label", "importance", "importance_pct"]].copy()
        fi_disp.columns = ["Feature", "Importance Score", "% of Total"]
        fi_disp["Importance Score"] = fi_disp["Importance Score"].round(4)
        st.dataframe(fi_disp, use_container_width=True, hide_index=True)

    # ── Model configuration ───────────────────────────────────────────────────
    _section("Model Configuration")
    cfg_cols = st.columns(3)
    if model_name == "Random Forest":
        cfgs = [
            ("Algorithm",     "Random Forest Regressor"),
            ("n_estimators",  "150"),
            ("max_depth",     "12"),
            ("random_state",  "42"),
            ("n_jobs",        "-1  (all cores)"),
            ("Test split",    "2026 data" if payload["has_2026"] else "80/20 (test_size=0.2)"),
        ]
    else:
        cfgs = [
            ("Algorithm",     "Linear Regression"),
            ("Scaling",       "StandardScaler"),
            ("Fit intercept", "True"),
            ("random_state",  "42"),
            ("Test split",    "2026 data" if payload["has_2026"] else "80/20 (test_size=0.2)"),
            ("Train years",   "2023, 2024, 2025"),
        ]
    for i, (k, v) in enumerate(cfgs):
        with cfg_cols[i % 3]:
            st.metric(k, v)

    st.markdown(
        '<div class="info-banner" style="margin-top:1rem;">'
        '💡 Zone Recommendations uses a separate XGBoost model for hot-zone ranking. '
        'Run <code>python train_model.py</code> to pre-train and save <code>models/model.pkl</code>.'
        '</div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Page 7 — Clustering
# ═════════════════════════════════════════════════════════════════════════════

def page_clustering():
    st.markdown('<div class="page-title">Clustering Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'KMeans clustering · Elbow Method · PCA visualization on aggregated demand features'
        '</div>',
        unsafe_allow_html=True,
    )

    feat_map = clust.available_features(demand)
    if not feat_map:
        st.error("Demand data has no numeric features for clustering.")
        return

    c_left, c_right = st.columns([2, 1])
    with c_left:
        _section("Feature Selection")
        sel_feats = st.multiselect(
            "Choose 2–5 numeric features",
            options=list(feat_map.keys()),
            default=["trip_count", "avg_fare", "avg_distance"],
            format_func=lambda x: feat_map[x],
        )
    with c_right:
        _section("Parameters")
        k         = st.slider("Number of clusters K", 2, 8, 3)
        normalize = st.checkbox("Apply StandardScaler", value=True)

    if len(sel_feats) < 2:
        st.warning("Please select at least 2 features.")
        return

    with st.spinner("Running KMeans …"):
        labels, X_proc, inertia = clust.run_kmeans(demand, sel_feats, k, normalize)

    use_pca = len(sel_feats) > 2
    if use_pca:
        X_2d, var_ratio = clust.apply_pca(X_proc)
        x_lbl = f"PC1 ({var_ratio[0]*100:.1f}% var)"
        y_lbl = f"PC2 ({var_ratio[1]*100:.1f}% var)"
        title = f"KMeans K={k} — PCA 2D Projection"
        X_plot = X_2d
    else:
        X_plot = X_proc
        x_lbl  = feat_map[sel_feats[0]]
        y_lbl  = feat_map[sel_feats[1]]
        title  = f"KMeans K={k} — {x_lbl} vs {y_lbl}"

    _pchart(clust.chart_scatter(X_plot, labels, x_lbl, y_lbl, title))

    st.markdown(
        f'<div class="info-banner">'
        f'K={k} · Inertia (WCSS): <b>{inertia:,.1f}</b> · '
        f'{"Normalization ON ✅" if normalize else "Normalization OFF ⚠️"}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Cluster statistics table
    _section("Cluster Statistics (Feature Means)")
    data_slice = demand[sel_feats].dropna().copy()
    data_slice["Cluster"] = labels
    cluster_stats = (
        data_slice.groupby("Cluster")[sel_feats]
        .mean().round(3)
        .rename(index=lambda i: f"Cluster {i}")
        .rename(columns=feat_map)
    )
    st.dataframe(cluster_stats, use_container_width=True)

    # Elbow plot
    st.markdown("---")
    _section("Elbow Method — Choosing Optimal K")
    st.markdown("""
    <div class="info-banner">
    <b>Elbow Method:</b> plot WCSS (inertia) for K = 1 … 10.
    The optimal K is at the "elbow" — where inertia stops dropping sharply.
    The chart compares results <b>with</b> and <b>without</b> StandardScaler to show
    how normalization affects cluster geometry.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Computing elbow curves …"):
        ks_n, in_n = clust.compute_elbow(demand, sel_feats, normalize=True)
        ks_r, in_r = clust.compute_elbow(demand, sel_feats, normalize=False)

    _pchart(clust.chart_elbow(ks_n, in_n, in_r))

    if use_pca:
        _section("PCA — Explained Variance")
        _, vr = clust.apply_pca(X_proc)
        st.markdown(
            f'<div class="info-banner">'
            f'PC1 explains <b>{vr[0]*100:.1f}%</b> of variance · '
            f'PC2 explains <b>{vr[1]*100:.1f}%</b> · '
            f'Total <b>{(vr[0]+vr[1])*100:.1f}%</b> captured in 2D'
            f'</div>',
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Page 8 — Regression
# ═════════════════════════════════════════════════════════════════════════════

def page_regression():
    st.markdown('<div class="page-title">Regression Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Linear Regression vs Random Forest · Train 2023–2025 · Validate on 2026'
        '</div>',
        unsafe_allow_html=True,
    )

    # Feature selection
    _section("Feature Selection")
    c1, c2 = st.columns([3, 1])
    with c1:
        sel_features = st.multiselect(
            "Features used for prediction",
            options=list(reg.REGRESSION_FEATURES.keys()),
            default=list(reg.REGRESSION_FEATURES.keys()),
            format_func=lambda x: reg.REGRESSION_FEATURES[x],
        )
    with c2:
        st.markdown('<div style="margin-top:2rem;color:#9CA3AF;font-size:.8rem;">'
                    '<b>Target:</b> trip_count<br>'
                    '<b>Train:</b> 2023–2025<br>'
                    '<b>Test:</b> 2026</div>', unsafe_allow_html=True)

    if not sel_features:
        st.warning("Select at least one feature.")
        return

    with st.spinner("Training models on 2023–2025 data …"):
        out = reg.get_regression_results(tuple(sorted(sel_features)))

    results  = out["results"]
    y_te     = out["y_te"]
    has_2026 = out["has_2026"]

    if has_2026:
        st.markdown(
            '<div class="info-banner">✅ Testing on <b>2026</b> data — real future validation</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="info-banner">⚠️ 2026 data not available — using 80/20 split of 2023–2025</div>',
            unsafe_allow_html=True,
        )

    # KPI row — best model
    best_name  = max(results, key=lambda m: results[m]["r2"])
    best       = results[best_name]
    _kpi_row([
        ("🏆", best_name,                    "Best Model",   "by R² score"),
        ("📉", f"{best['mae']:.2f}",         "Best MAE",     "trips error"),
        ("📊", f"{best['rmse']:.2f}",        "Best RMSE",    "trips error"),
        ("📈", f"{best['r2']:.3f}",          "Best R²",      "variance explained"),
        ("🏋️", f"{out['n_train']:,}",        "Train rows",   "demand records"),
        ("🧪", f"{out['n_test']:,}",         "Test rows",    "demand records"),
    ])

    # Metrics comparison
    _section("Model Comparison")
    metrics_rows = []
    for m, r in results.items():
        metrics_rows.append({
            "Model": m,
            "MAE":   round(r["mae"],  3),
            "RMSE":  round(r["rmse"], 3),
            "R²":    round(r["r2"],   4),
        })
    st.dataframe(
        pd.DataFrame(metrics_rows),
        use_container_width=True, hide_index=True,
    )

    mc1, mc2 = st.columns(2)
    with mc1: _pchart(reg.chart_metrics_bar(results))
    with mc2: _pchart(reg.chart_r2_bar(results))

    # Actual vs predicted (tabs per model)
    _section("Actual vs Predicted")
    tabs = st.tabs([f"📊 {m}" for m in results])
    for tab, (model_name, res) in zip(tabs, results.items()):
        with tab:
            _pchart(reg.chart_actual_vs_pred(y_te, res["y_pred"], model_name))

    # Feature importance (models that expose it)
    fi_entries = [(m, r) for m, r in results.items() if "feature_importance" in r]
    if fi_entries:
        _section("Feature Importance")
        fi_cols = st.columns(len(fi_entries))
        for col, (model_name, res) in zip(fi_cols, fi_entries):
            with col:
                _pchart(reg.chart_feature_importance(
                    res["feature_importance"], model_name
                ))

    # Optimization tips
    _section("Optimization Notes")
    st.markdown("""
    <div class="info-banner">
    💡 <b>Tips to improve accuracy:</b>
    Remove low-importance features (e.g. <i>dow</i>, <i>month</i> if R² is already high) to
    reduce overfitting · Add <i>avg_tip</i> as a feature if available · Try feature
    interactions (e.g. hour × zone) · For Linear Regression, features are auto-scaled
    (StandardScaler) so raw vs normalised values are comparable.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Page — Real-Time Prediction
# ═════════════════════════════════════════════════════════════════════════════

def page_realtime_prediction():
    st.markdown('<div class="page-title">Real-Time Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'AI demand forecast for any input — generalises to combinations never seen in training data'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading model …"):
        payload = load_regression_model()

    m          = payload["metrics"]
    model_name = payload["model_name"]
    model_obj  = payload["model"]
    scaler     = payload["scaler"]
    feat_cols  = payload["feature_cols"]
    y_test     = payload["y_test"]

    _kpi_row([
        ("🤖", model_name,           "Active Model",   "generalises to new inputs"),
        ("📉", f"{m['mae']:.2f}",    "MAE",            "avg trips error"),
        ("📈", f"{m['r2']:.3f}",     "R² Score",       "variance explained"),
        ("🏋️", f"{payload['n_train']:,}", "Train rows", "all years"),
    ])

    st.markdown("---")

    # ── Default stats (pre-fill from historical median) ───────────────────────
    def _zone_defaults(loc_id):
        z = demand[demand["PULocationID"] == loc_id]
        src = z if not z.empty else demand
        # Cast to Python float and clamp within number_input bounds
        return {
            "hist": float(src["zone_total_trips"].iloc[0]) if not z.empty
                    else float(demand["zone_total_trips"].median()),
            "fare": max(1.0,  min(500.0, float(src["avg_fare"].mean()))),
            "dist": max(0.1,  min(100.0, float(src["avg_distance"].mean()))),
            "dur":  max(1.0,  min(300.0, float(src["avg_duration"].mean()))),
        }

    # ── Layout ────────────────────────────────────────────────────────────────
    col_form, col_res = st.columns([1, 1], gap="large")

    with col_form:
        _section("Input Parameters")
        st.markdown(
            '<div class="info-banner" style="margin-bottom:12px;">'
            '💡 ניתן להזין כל ערך — המודל יבצע הכללה גם עבור קומבינציות חדשות.'
            '</div>', unsafe_allow_html=True,
        )

        # Zone
        zone_opts = zones[["LocationID", "Zone", "Borough"]].copy()
        zone_opts["label"] = zone_opts.apply(
            lambda r: f"{r['Zone']} ({r['Borough']}) — ID {r['LocationID']}", axis=1
        )
        zone_label = st.selectbox("Pickup Zone", zone_opts["label"].tolist(),
                                  index=0, key="rt_zone")
        loc_id = int(zone_label.split("— ID ")[-1])
        defs   = _zone_defaults(loc_id)

        # Time
        c1, c2 = st.columns(2)
        with c1:
            hour = st.slider("Hour of Day", 0, 23, 8, key="rt_hour",
                             help="0 = midnight, 8 = morning rush, 18 = evening rush")
        with c2:
            year_sel = st.selectbox("Year", list(range(2023, 2036)), index=3, key="rt_year")

        dow_names  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        mon_names  = ["January","February","March","April","May","June",
                      "July","August","September","October","November","December"]
        c3, c4 = st.columns(2)
        with c3:
            dow_sel  = st.selectbox("Day of Week", dow_names, key="rt_dow")
            dow_num  = dow_names.index(dow_sel)
        with c4:
            month_sel = st.selectbox("Month", mon_names, key="rt_month")
            month_num = mon_names.index(month_sel) + 1

        st.markdown("**Trip Statistics** *(editable — override with any value)*")
        c5, c6 = st.columns(2)
        with c5:
            avg_fare = st.number_input("Avg Fare ($)", min_value=1.0, max_value=500.0,
                                       value=float(round(defs["fare"], 2)), step=0.5, key="rt_fare")
            avg_dist = st.number_input("Avg Distance (mi)", min_value=0.1, max_value=100.0,
                                       value=float(round(defs["dist"], 2)), step=0.1, key="rt_dist")
        with c6:
            avg_dur  = st.number_input("Avg Duration (min)", min_value=1.0, max_value=300.0,
                                       value=float(round(defs["dur"], 1)), step=1.0, key="rt_dur")
            hist_cnt = st.number_input("Historical Trip Count", min_value=0, max_value=500000,
                                       value=int(defs["hist"]), step=100, key="rt_hist")

        pax = st.slider("Passenger Count *(informational)*", 1, 6, 1, key="rt_pax",
                        help="Not a model feature — used for explainability context only")

        st.markdown("")
        predict_clicked = st.button("🔮  Predict Demand", use_container_width=True,
                                    type="primary", key="rt_btn")

    # ── Validation ────────────────────────────────────────────────────────────
    errors = []
    if not (0 <= hour <= 23):     errors.append("Hour must be 0–23")
    if not (1 <= month_num <= 12): errors.append("Month must be 1–12")
    if avg_fare <= 0:              errors.append("Fare must be > 0")
    if avg_dist <= 0:              errors.append("Distance must be > 0")
    if avg_dur  <= 0:              errors.append("Duration must be > 0")

    with col_res:
        _section("Prediction Result")

        if errors:
            for e in errors:
                st.error(f"⚠️ {e}")
            st.stop()

        # ── Build feature vector ───────────────────────────────────────────────
        features = {
            "pickup_location_id":    float(loc_id),
            "pickup_hour":           float(hour),
            "pickup_day_of_week":    float(dow_num),
            "pickup_month":          float(month_num),
            "historical_trip_count": float(hist_cnt),
            "avg_fare_amount":       float(avg_fare),
            "avg_trip_distance":     float(avg_dist),
            "avg_trip_duration":     float(avg_dur),
            "year":                  float(year_sel),
        }
        pred = predict_regression(payload, features)

        # ── Confidence range (RF: 10th–90th percentile of tree preds) ─────────
        ci_lo = ci_hi = None
        if hasattr(model_obj, "estimators_"):
            X_raw = np.array([[features[f] for f in feat_cols]], dtype=float)
            tree_preds = np.maximum(
                [est.predict(X_raw)[0] for est in model_obj.estimators_], 0
            )
            ci_lo = float(np.percentile(tree_preds, 10))
            ci_hi = float(np.percentile(tree_preds, 90))

        # ── Demand level ──────────────────────────────────────────────────────
        p25 = float(np.percentile(y_test, 25))
        p75 = float(np.percentile(y_test, 75))
        p90 = float(np.percentile(y_test, 90))

        if pred < p25:
            level, lvl_color, lvl_emoji = "Low",       "#6B7280", "🟤"
        elif pred < p75:
            level, lvl_color, lvl_emoji = "Medium",    "#F7C948", "🟡"
        elif pred < p90:
            level, lvl_color, lvl_emoji = "High",      "#F97316", "🟠"
        else:
            level, lvl_color, lvl_emoji = "Very High", "#EF4444", "🔴"

        # ── Historical comparison ──────────────────────────────────────────────
        hist_rows = demand[(demand["PULocationID"] == loc_id) & (demand["hour"] == hour)]
        hist_avg  = float(hist_rows["trip_count"].mean()) if len(hist_rows) > 0 else 0.0
        diff_pct  = ((pred - hist_avg) / max(hist_avg, 1)) * 100
        arrow     = "▲" if diff_pct >= 0 else "▼"
        clr_diff  = "#10B981" if diff_pct >= 0 else "#EF4444"

        # ── Prediction card ────────────────────────────────────────────────────
        ci_html = (
            f'<div style="margin-top:8px;font-size:.78rem;color:#9CA3AF;">'
            f'Confidence range: <b style="color:#FAFAFA;">{ci_lo:.0f} – {ci_hi:.0f} trips</b>'
            f'</div>'
        ) if ci_lo is not None else ""

        st.markdown(f"""
        <div class="pred-card">
          <div class="pred-value">{pred:.0f}</div>
          <div class="pred-label">Predicted trips / hour</div>
          <div style="margin-top:10px;">
            <span style="background:{lvl_color};color:#fff;font-weight:700;
                   padding:4px 18px;border-radius:20px;font-size:.88rem;">
              {lvl_emoji} {level} Demand
            </span>
          </div>
          <div style="margin-top:12px;font-size:.8rem;color:#9CA3AF;">
            Historical avg at {hour:02d}:00 →
            <b style="color:#FAFAFA;">{hist_avg:.0f} trips</b>
          </div>
          <div style="font-size:.82rem;color:{clr_diff};">
            {arrow} {abs(diff_pct):.1f}% vs. historical average
          </div>
          {ci_html}
        </div>
        """, unsafe_allow_html=True)

        # ── Feature pills ──────────────────────────────────────────────────────
        pills = [
            ("Avg Fare",  f"${avg_fare:.2f}"),
            ("Distance",  f"{avg_dist:.1f} mi"),
            ("Duration",  f"{avg_dur:.0f} min"),
            ("Zone Hist.",f"{int(hist_cnt):,}"),
            ("Passengers",f"{pax}"),
        ]
        pill_html = '<div class="pill-row">'
        for lbl, val in pills:
            pill_html += (
                f'<div class="pill">'
                f'<span style="color:#9CA3AF;font-size:.7rem;">{lbl}</span>'
                f'<br><b>{val}</b></div>'
            )
        pill_html += "</div>"
        st.markdown(pill_html, unsafe_allow_html=True)

        # ── Smart demand alerts ────────────────────────────────────────────────
        if level == "Very High":
            st.error(
                "🔴 **Extreme Demand Zone** — Surge pricing is likely active. "
                "Drivers: head to this zone immediately for maximum earnings."
            )
        elif level == "High":
            st.warning(
                "🟠 **High Demand Alert** — Strong pickup opportunities in this zone. "
                "Consider positioning here for the next hour."
            )
        elif level == "Medium":
            st.info(
                "🟡 **Normal Demand** — Average activity expected. "
                "Conditions are stable for this zone and time."
            )

    # ── Explainability ────────────────────────────────────────────────────────
    st.markdown("---")
    _section("Why This Prediction? — Feature Insights")

    def _insight(icon, title, detail, color):
        return (
            f'<div style="background:#1E2130;border-left:3px solid {color};'
            f'padding:10px 14px;border-radius:6px;margin-bottom:8px;">'
            f'<span style="font-size:1.1rem;">{icon}</span> '
            f'<b style="color:#FAFAFA;">{title}</b>'
            f'<div style="color:#9CA3AF;font-size:.78rem;margin-top:3px;">{detail}</div>'
            f'</div>'
        )

    insights_html = ""

    # Hour
    if hour in range(7, 10):
        insights_html += _insight("🌅","Morning Rush Hour",
            f"שעה {hour}:00 — שיא הביקוש לנסיעות בוקר (7–9)", "#F97316")
    elif hour in range(17, 20):
        insights_html += _insight("🌆","Evening Rush Hour",
            f"שעה {hour}:00 — שיא הביקוש לנסיעות ערב (17–19)", "#F97316")
    elif hour in range(22, 24) or hour < 3:
        insights_html += _insight("🌙","Night Hours",
            f"שעה {hour}:00 — ביקוש נמוך בשעות הלילה", "#6B7280")
    else:
        insights_html += _insight("☀️","Standard Hours",
            f"שעה {hour}:00 — עומס ממוצע", "#3B82F6")

    # Day
    if dow_num >= 5:
        insights_html += _insight("📅","Weekend",
            f"{dow_sel} — נסיעות פנאי, פחות נסיעות עבודה", "#8B5CF6")
    else:
        insights_html += _insight("📅","Weekday",
            f"{dow_sel} — דפוסי עומס רגילים", "#3B82F6")

    # Zone demand level
    zone_pct = float(np.mean(demand["zone_total_trips"] <= hist_cnt) * 100)
    zone_info = zones[zones["LocationID"] == loc_id]
    zone_name = zone_info["Zone"].iloc[0] if not zone_info.empty else f"Zone {loc_id}"
    if zone_pct >= 80:
        insights_html += _insight("📍","High-Demand Zone",
            f"{zone_name} — Top {100-int(zone_pct):.0f}% of all zones by historical volume", "#10B981")
    elif zone_pct <= 20:
        insights_html += _insight("📍","Low-Demand Zone",
            f"{zone_name} — Below average zone activity ({int(zone_pct):.0f}th percentile)", "#6B7280")
    else:
        insights_html += _insight("📍","Average Zone",
            f"{zone_name} — {int(zone_pct):.0f}th percentile by historical trip volume", "#3B82F6")

    # Trip distance
    dist_median = float(demand["avg_distance"].median())
    if avg_dist > dist_median * 1.5:
        insights_html += _insight("🛣️","Long Trip",
            f"{avg_dist:.1f} mi — {((avg_dist/dist_median)-1)*100:.0f}% above median distance ({dist_median:.1f} mi)", "#F7C948")
    elif avg_dist < dist_median * 0.5:
        insights_html += _insight("🛣️","Short Trip",
            f"{avg_dist:.1f} mi — below median distance ({dist_median:.1f} mi)", "#6B7280")
    else:
        insights_html += _insight("🛣️","Typical Distance",
            f"{avg_dist:.1f} mi — close to median distance ({dist_median:.1f} mi)", "#3B82F6")

    # Month / Season
    seasons = {
        (12,1,2): ("❄️","Winter","חודשי חורף — מזג אוויר עלול להפחית ביקוש","#6B7280"),
        (3,4,5):  ("🌸","Spring","אביב — ביקוש מאוזן","#3B82F6"),
        (6,7,8):  ("☀️","Summer","קיץ — עלייה בתיירות ונסיעות פנאי","#F97316"),
        (9,10,11):("🍂","Autumn","סתיו — ביקוש ממוצע","#8B5CF6"),
    }
    for months_tuple, (icon, name, detail, color) in seasons.items():
        if month_num in months_tuple:
            insights_html += _insight(icon, f"{name} — {mon_names[month_num-1]}",
                                      detail, color)
            break

    # Passenger count context
    if pax >= 3:
        insights_html += _insight("👥","Multiple Passengers",
            f"{pax} נוסעים — נסיעות קבוצתיות לרוב קצרות יותר", "#8B5CF6")

    st.markdown(
        f'<div style="columns:2;column-gap:16px;">{insights_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Model confidence summary ───────────────────────────────────────────────
    if ci_lo is not None:
        st.markdown("---")
        _section("Confidence Analysis")
        c_a, c_b, c_c = st.columns(3)
        c_a.metric("Prediction", f"{pred:.0f} trips")
        c_b.metric("Lower Bound (P10)", f"{ci_lo:.0f} trips")
        c_c.metric("Upper Bound (P90)", f"{ci_hi:.0f} trips")
        spread = ci_hi - ci_lo
        confidence_pct = max(0, 100 - (spread / max(pred, 1)) * 50)
        st.progress(min(int(confidence_pct), 100),
                    text=f"Confidence Score: {confidence_pct:.0f}%  "
                         f"(spread = {spread:.0f} trips across 150 trees)")


# ═════════════════════════════════════════════════════════════════════════════
# Page — AI Demand Map
# ═════════════════════════════════════════════════════════════════════════════

def page_demand_map():
    import plotly.express as px
    from src.zone_coords import get_zone_coord
    from src.model import FEATURE_COLS

    st.markdown('<div class="page-title">AI Demand Map</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Interactive demand heatmap — predicted trip demand per NYC zone · 🔵 Low · 🟠 Medium · 🔴 High'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading model …"):
        model, _, _, _, _ = load_xgb_model()

    _section("Time Parameters")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        map_hour = st.slider("Hour of Day", 0, 23, 18, key="map_hour")
    with mc2:
        _dow_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        map_dow_lbl = st.selectbox("Day of Week", _dow_names, index=4, key="map_dow")
        map_dow = _dow_names.index(map_dow_lbl)
    with mc3:
        _mon_full = ["January","February","March","April","May","June",
                     "July","August","September","October","November","December"]
        map_mon_lbl = st.selectbox("Month", _mon_full, key="map_month")
        map_month = _mon_full.index(map_mon_lbl) + 1

    # Per-zone feature matrix using demand aggregates
    zone_stats = (
        demand.groupby("PULocationID")
        .agg(
            avg_fare        =("avg_fare",         "mean"),
            avg_distance    =("avg_distance",     "mean"),
            avg_duration    =("avg_duration",     "mean"),
            zone_total_trips=("zone_total_trips",  "first"),
            hist_demand     =("trip_count",        "mean"),
        )
        .reset_index()
    )
    zone_stats["hour"]  = map_hour
    zone_stats["dow"]   = map_dow
    zone_stats["month"] = map_month

    for col in ["avg_fare", "avg_distance", "avg_duration", "zone_total_trips"]:
        med = float(demand[col].median())
        zone_stats[col] = zone_stats[col].fillna(med)

    zone_stats = zone_stats.dropna(subset=FEATURE_COLS)
    if zone_stats.empty:
        st.warning("No demand data available.")
        return

    zone_stats["predicted_demand"] = np.maximum(
        model.predict(zone_stats[FEATURE_COLS].values), 0
    )

    merged = zone_stats.merge(
        zones[["LocationID", "Zone", "Borough"]],
        left_on="PULocationID", right_on="LocationID", how="left",
    )
    merged["Zone"]    = merged["Zone"].fillna(merged["PULocationID"].astype(str))
    merged["Borough"] = merged["Borough"].fillna("Unknown")

    coords = [
        get_zone_coord(int(row["PULocationID"]), str(row["Borough"]))
        for _, row in merged.iterrows()
    ]
    merged["lat"] = [c[0] for c in coords]
    merged["lon"] = [c[1] for c in coords]

    # Peak hour per zone
    peak_h = (
        demand.sort_values("trip_count", ascending=False)
        .groupby("PULocationID")["hour"].first()
        .reset_index().rename(columns={"hour": "peak_hour"})
    )
    merged = merged.merge(peak_h, on="PULocationID", how="left")

    p33 = float(np.percentile(merged["predicted_demand"], 33))
    p66 = float(np.percentile(merged["predicted_demand"], 66))
    merged["Demand Level"] = merged["predicted_demand"].apply(
        lambda x: "Low" if x < p33 else ("Medium" if x < p66 else "High")
    )
    merged["Avg Fare ($)"]       = merged["avg_fare"].round(2)
    merged["Avg Distance (mi)"]  = merged["avg_distance"].round(2)
    merged["Historical Avg"]     = merged["hist_demand"].round(1)
    merged["Predicted Demand"]   = merged["predicted_demand"].round(1)

    fig = px.scatter_mapbox(
        merged,
        lat="lat", lon="lon",
        color="predicted_demand",
        size="predicted_demand",
        size_max=30,
        hover_name="Zone",
        hover_data={
            "Borough": True,
            "Predicted Demand": True,
            "Avg Fare ($)": True,
            "Avg Distance (mi)": True,
            "Historical Avg": True,
            "peak_hour": True,
            "Demand Level": True,
            "lat": False, "lon": False,
            "predicted_demand": False,
        },
        labels={"peak_hour": "Peak Hour"},
        color_continuous_scale=[[0.0,"#3B82F6"],[0.5,"#F97316"],[1.0,"#EF4444"]],
        mapbox_style="carto-darkmatter",
        zoom=10,
        center={"lat": 40.730, "lon": -73.985},
        opacity=0.88,
        title=f"NYC Taxi Demand — {map_dow_lbl} {map_hour:02d}:00 · {map_mon_lbl}",
    )
    fig.update_layout(
        height=600,
        paper_bgcolor="#1A1D27",
        font=dict(color="#FAFAFA"),
        coloraxis_colorbar=dict(title="Trips/hr", tickfont=dict(color="#FAFAFA")),
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

    st.markdown("---")
    top1 = merged.nlargest(1, "predicted_demand").iloc[0]
    _kpi_row([
        ("🔴", f"{merged['predicted_demand'].max():.0f}",  "Peak Zone Demand",  "trips/hr"),
        ("🟡", f"{merged['predicted_demand'].mean():.0f}", "City Average",       "trips/hr"),
        ("🟢", f"{merged['predicted_demand'].min():.0f}",  "Lowest Demand",     "trips/hr"),
        ("📍", str(top1["Zone"])[:25],                      "Hottest Zone",      str(top1.get("Borough",""))),
    ])

    _section("Top 10 Demand Zones")
    top10 = (
        merged.nlargest(10, "predicted_demand")
        [["Zone","Borough","Predicted Demand","Avg Fare ($)","Demand Level","peak_hour"]]
        .rename(columns={"peak_hour": "Peak Hour"})
        .reset_index(drop=True)
    )
    top10.index = range(1, 11)
    st.dataframe(top10, use_container_width=True)

    st.markdown(
        '<div class="info-banner">'
        '💡 Zone positions are distributed within borough bounds for visualization. '
        'Demand is predicted by the XGBoost model for the selected time window.'
        '</div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Page — Driver Intelligence Tools
# ═════════════════════════════════════════════════════════════════════════════

def page_driver_tools():
    import plotly.graph_objects as go

    st.markdown('<div class="page-title">Driver Intelligence Tools</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Relocation Simulator · Profit Estimator — AI-powered decisions for NYC taxi drivers'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading model …"):
        payload = load_regression_model()

    _dow_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    _mon_full  = ["January","February","March","April","May","June",
                  "July","August","September","October","November","December"]

    zone_opts = zones[["LocationID", "Zone", "Borough"]].copy()
    zone_opts["label"] = zone_opts.apply(
        lambda r: f"{r['Zone']} ({r['Borough']}) — ID {r['LocationID']}", axis=1
    )
    labels = zone_opts["label"].tolist()

    tab_reloc, tab_profit = st.tabs(["🚗  Relocation Simulator", "💰  Profit Estimator"])

    # ── Relocation Simulator ─────────────────────────────────────────────────
    with tab_reloc:
        _section("Relocation Parameters")
        rc1, rc2 = st.columns(2)
        with rc1:
            cur_lbl = st.selectbox("Current Zone", labels, index=0, key="rel_cur")
            cur_id  = int(cur_lbl.split("— ID ")[-1])
        with rc2:
            tgt_lbl = st.selectbox("Target Zone", labels, index=min(1, len(labels)-1), key="rel_tgt")
            tgt_id  = int(tgt_lbl.split("— ID ")[-1])

        rr1, rr2, rr3, rr4 = st.columns(4)
        with rr1:
            rel_hour = st.slider("Hour", 0, 23, 18, key="rel_hour")
        with rr2:
            rel_dow_lbl = st.selectbox("Day", _dow_names, key="rel_dow")
            rel_dow = _dow_names.index(rel_dow_lbl)
        with rr3:
            rel_mon_lbl = st.selectbox("Month", _mon_full, key="rel_month")
            rel_month = _mon_full.index(rel_mon_lbl) + 1
        with rr4:
            rel_year = st.selectbox("Year", list(range(2023, 2036)), index=3, key="rel_year")

        def _zone_feats(loc_id, hour, dow, month, year):
            z = demand[demand["PULocationID"] == loc_id]
            src = z if not z.empty else demand
            return {
                "pickup_location_id":    float(loc_id),
                "pickup_hour":           float(hour),
                "pickup_day_of_week":    float(dow),
                "pickup_month":          float(month),
                "historical_trip_count": float(
                    src["zone_total_trips"].iloc[0] if not z.empty
                    else float(demand["zone_total_trips"].median())
                ),
                "avg_fare_amount":   max(1.0,  min(500.0, float(src["avg_fare"].mean()))),
                "avg_trip_distance": max(0.1,  min(100.0, float(src["avg_distance"].mean()))),
                "avg_trip_duration": max(1.0,  min(300.0, float(src["avg_duration"].mean()))),
                "year":              float(year),
            }

        cur_f = _zone_feats(cur_id, rel_hour, rel_dow, rel_month, rel_year)
        tgt_f = _zone_feats(tgt_id, rel_hour, rel_dow, rel_month, rel_year)
        cur_pred = predict_regression(payload, cur_f)
        tgt_pred = predict_regression(payload, tgt_f)

        delta_abs = tgt_pred - cur_pred
        delta_pct = (delta_abs / max(cur_pred, 1)) * 100
        cur_rev   = cur_pred * cur_f["avg_fare_amount"] * 0.70
        tgt_rev   = tgt_pred * tgt_f["avg_fare_amount"] * 0.70
        rev_delta = tgt_rev - cur_rev

        cur_name = zone_opts[zone_opts["LocationID"] == cur_id]["Zone"].iloc[0] \
                   if cur_id in zone_opts["LocationID"].values else f"Zone {cur_id}"
        tgt_name = zone_opts[zone_opts["LocationID"] == tgt_id]["Zone"].iloc[0] \
                   if tgt_id in zone_opts["LocationID"].values else f"Zone {tgt_id}"

        if delta_pct > 20:
            rec_color, rec_icon, rec_text = "#10B981", "✅", "Strongly Recommended"
        elif delta_pct > 5:
            rec_color, rec_icon, rec_text = "#F7C948", "⚡", "Recommended"
        elif delta_pct > -5:
            rec_color, rec_icon, rec_text = "#3B82F6", "ℹ️", "Neutral — similar demand"
        else:
            rec_color, rec_icon, rec_text = "#EF4444", "⚠️", "Not Recommended"

        arrow_d  = "▲" if delta_abs >= 0 else "▼"
        arrow_r  = "▲" if rev_delta >= 0 else "▼"
        clr_rev  = "#10B981" if rev_delta >= 0 else "#EF4444"

        _section("Simulation Result")
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1A1D27,#252836);
                    border:1.5px solid {rec_color};border-radius:16px;
                    padding:22px 28px;margin-bottom:16px;">
          <div style="font-size:1.1rem;font-weight:700;color:#FAFAFA;margin-bottom:14px;">
            {rec_icon}&nbsp; Moving from
            <span style="color:#F7C948;">{cur_name}</span> →
            <span style="color:#3B82F6;">{tgt_name}</span>
          </div>
          <div style="display:flex;gap:32px;flex-wrap:wrap;">
            <div>
              <div style="color:#9CA3AF;font-size:.75rem;margin-bottom:4px;">Demand Change</div>
              <div style="font-size:1.55rem;font-weight:800;color:#F7C948;">
                {arrow_d} {abs(delta_abs):.0f} trips/hr
              </div>
              <div style="font-size:.8rem;color:#9CA3AF;">({delta_pct:+.1f}%)</div>
            </div>
            <div>
              <div style="color:#9CA3AF;font-size:.75rem;margin-bottom:4px;">Est. Revenue/hr Δ</div>
              <div style="font-size:1.55rem;font-weight:800;color:{clr_rev};">
                {arrow_r} ${abs(rev_delta):.2f}
              </div>
              <div style="font-size:.8rem;color:#9CA3AF;">driver share 70%</div>
            </div>
            <div>
              <div style="color:#9CA3AF;font-size:.75rem;margin-bottom:4px;">Recommendation</div>
              <div style="font-size:1.1rem;font-weight:700;color:{rec_color};">{rec_text}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        _kpi_row([
            ("📍", f"{cur_pred:.0f}", f"Demand · {cur_name[:22]}", "trips/hr"),
            ("🎯", f"{tgt_pred:.0f}", f"Demand · {tgt_name[:22]}", "trips/hr"),
            ("💰", f"${cur_rev:.2f}",  "Current est. revenue/hr",  "70% driver share"),
            ("💵", f"${tgt_rev:.2f}",  "Target est. revenue/hr",   "70% driver share"),
        ])

        if delta_pct > 5:
            st.success(
                f"💡 Moving to **{tgt_name}** could increase demand by "
                f"**{abs(delta_abs):.0f} trips/hr** ({delta_pct:.1f}%) "
                f"and revenue by **${rev_delta:.2f}/hr**."
            )
        elif delta_pct < -5:
            st.warning(
                f"⚠️ Moving to **{tgt_name}** would decrease demand by "
                f"**{abs(delta_abs):.0f} trips/hr** ({abs(delta_pct):.1f}%)."
            )

        # Side-by-side bar comparison
        fig_cmp = go.Figure(go.Bar(
            x=[cur_name[:20], tgt_name[:20]],
            y=[cur_pred, tgt_pred],
            marker_color=["#F7C948", "#3B82F6"],
            text=[f"{cur_pred:.0f}", f"{tgt_pred:.0f}"],
            textposition="outside",
        ))
        fig_cmp.update_layout(
            title="Demand Comparison",
            yaxis_title="Predicted Trips/hr",
            template="plotly_dark",
            paper_bgcolor="#1A1D27",
            plot_bgcolor="#1A1D27",
            font=dict(color="#FAFAFA"),
            height=320,
        )
        _pchart(fig_cmp)

    # ── Profit Estimator ─────────────────────────────────────────────────────
    with tab_profit:
        _section("Profit Estimation Parameters")
        pe1, pe2 = st.columns(2)
        with pe1:
            prf_lbl     = st.selectbox("Zone", labels, key="prf_zone")
            prf_id      = int(prf_lbl.split("— ID ")[-1])
            prf_dow_lbl = st.selectbox("Day of Week", _dow_names, key="prf_dow")
            prf_dow     = _dow_names.index(prf_dow_lbl)
            prf_mon_lbl = st.selectbox("Month", _mon_full, key="prf_month")
            prf_month   = _mon_full.index(prf_mon_lbl) + 1
        with pe2:
            prf_year   = st.selectbox("Year", list(range(2023, 2036)), index=3, key="prf_year")
            hr_start, hr_end = st.slider("Working Hours", 0, 23, (8, 20), key="prf_hours")
            drv_share  = st.slider("Driver Share (%)", 50, 100, 70, key="prf_share") / 100.0

        pz = demand[demand["PULocationID"] == prf_id]
        ps = pz if not pz.empty else demand
        prf_fare = max(1.0,  min(500.0, float(ps["avg_fare"].mean())))
        prf_dist = max(0.1,  min(100.0, float(ps["avg_distance"].mean())))
        prf_dur  = max(1.0,  min(300.0, float(ps["avg_duration"].mean())))
        prf_hist = float(ps["zone_total_trips"].iloc[0] if not pz.empty
                         else float(demand["zone_total_trips"].median()))

        hourly_data = []
        for h in range(hr_start, hr_end + 1):
            p = predict_regression(payload, {
                "pickup_location_id":    float(prf_id),
                "pickup_hour":           float(h),
                "pickup_day_of_week":    float(prf_dow),
                "pickup_month":          float(prf_month),
                "historical_trip_count": prf_hist,
                "avg_fare_amount":       prf_fare,
                "avg_trip_distance":     prf_dist,
                "avg_trip_duration":     prf_dur,
                "year":                  float(prf_year),
            })
            hourly_data.append({"Hour": h, "Trips": p, "Revenue": p * prf_fare * drv_share})

        hdf = pd.DataFrame(hourly_data)
        total_trips = float(hdf["Trips"].sum())
        total_rev   = float(hdf["Revenue"].sum())
        n_hrs = hr_end - hr_start + 1

        _kpi_row([
            ("⏰", f"{hr_start:02d}:00–{hr_end:02d}:00", "Working Hours",     ""),
            ("🚖", f"{total_trips:.0f}",                  "Predicted Trips",   "total"),
            ("💵", f"${total_rev:.2f}",                   "Estimated Revenue", f"{int(drv_share*100)}% share"),
            ("📊", f"${total_rev/n_hrs:.2f}",             "Avg Revenue/hr",    ""),
        ])

        fig_prf = go.Figure()
        fig_prf.add_trace(go.Bar(
            x=hdf["Hour"], y=hdf["Trips"],
            name="Trips/hr", marker_color="#F7C948", yaxis="y",
        ))
        fig_prf.add_trace(go.Scatter(
            x=hdf["Hour"], y=hdf["Revenue"],
            name="Revenue ($)", line=dict(color="#10B981", width=2.5),
            mode="lines+markers", yaxis="y2",
        ))
        fig_prf.update_layout(
            title=f"Hourly Breakdown — {prf_lbl.split(' (')[0]}",
            xaxis_title="Hour of Day",
            yaxis =dict(title="Trips/hr",    side="left"),
            yaxis2=dict(title="Revenue ($)", side="right", overlaying="y", showgrid=False),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            template="plotly_dark",
            paper_bgcolor="#1A1D27",
            plot_bgcolor="#1A1D27",
            font=dict(color="#FAFAFA"),
            height=370,
        )
        _pchart(fig_prf)

        st.markdown(
            '<div class="info-banner">'
            '💡 Revenue = Predicted Trips × Avg Fare × Driver Share. '
            'Actual revenue depends on acceptance rate, traffic, and waiting time.'
            '</div>',
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Page — Demand Forecast
# ═════════════════════════════════════════════════════════════════════════════

def page_forecast():
    import plotly.graph_objects as go

    st.markdown('<div class="page-title">Demand Forecast</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Hourly · Day-of-week · Monthly demand forecast per zone — supports future years 2027–2035'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading model …"):
        payload = load_regression_model()

    _dow_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    _mon_full  = ["January","February","March","April","May","June",
                  "July","August","September","October","November","December"]
    _mon_short = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    _section("Forecast Parameters")
    fp1, fp2, fp3, fp4 = st.columns(4)
    with fp1:
        fz_opts = zones[["LocationID","Zone","Borough"]].copy()
        fz_opts["label"] = fz_opts.apply(
            lambda r: f"{r['Zone']} ({r['Borough']}) — ID {r['LocationID']}", axis=1
        )
        fc_lbl = st.selectbox("Zone", fz_opts["label"].tolist(), key="fc_zone")
        fc_id  = int(fc_lbl.split("— ID ")[-1])
    with fp2:
        fc_dow_lbl = st.selectbox("Day of Week", _dow_names, key="fc_dow")
        fc_dow = _dow_names.index(fc_dow_lbl)
    with fp3:
        fc_mon_lbl = st.selectbox("Month", _mon_full, key="fc_month")
        fc_month = _mon_full.index(fc_mon_lbl) + 1
    with fp4:
        fc_year = st.selectbox("Year", list(range(2023, 2036)), index=3, key="fc_year")

    fz = demand[demand["PULocationID"] == fc_id]
    fs = fz if not fz.empty else demand
    fc_hist = float(fs["zone_total_trips"].iloc[0] if not fz.empty
                    else float(demand["zone_total_trips"].median()))
    fc_fare = max(1.0,  min(500.0, float(fs["avg_fare"].mean())))
    fc_dist = max(0.1,  min(100.0, float(fs["avg_distance"].mean())))
    fc_dur  = max(1.0,  min(300.0, float(fs["avg_duration"].mean())))

    def _fc_pred(hour, dow, month, year):
        return predict_regression(payload, {
            "pickup_location_id":    float(fc_id),
            "pickup_hour":           float(hour),
            "pickup_day_of_week":    float(dow),
            "pickup_month":          float(month),
            "historical_trip_count": fc_hist,
            "avg_fare_amount":       fc_fare,
            "avg_trip_distance":     fc_dist,
            "avg_trip_duration":     fc_dur,
            "year":                  float(year),
        })

    _DARK = dict(
        template="plotly_dark",
        paper_bgcolor="#1A1D27",
        plot_bgcolor="#1A1D27",
        font=dict(color="#FAFAFA"),
        height=390,
    )

    tab_h, tab_d, tab_m = st.tabs(
        ["⏰  24-Hour Forecast", "📅  Day-of-Week Pattern", "🗓️  Monthly Pattern"]
    )

    with tab_h:
        hours  = list(range(24))
        preds  = [_fc_pred(h, fc_dow, fc_month, fc_year) for h in hours]
        peak_h = hours[int(np.argmax(preds))]
        colors = ["#EF4444" if h == peak_h else "#F7C948" for h in hours]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hours, y=preds, marker_color=colors, name="Predicted Demand",
            text=[f"{p:.0f}" if h in (peak_h, hours[int(np.argmin(preds))]) else ""
                  for h, p in zip(hours, preds)],
            textposition="outside",
        ))
        fig.add_trace(go.Scatter(
            x=hours, y=preds, mode="lines",
            line=dict(color="rgba(247,201,72,0.4)", width=2, dash="dot"),
            showlegend=False,
        ))
        fig.update_layout(
            title=f"24-Hour Forecast — {fc_lbl.split(' (')[0]} · {fc_dow_lbl} · {fc_mon_lbl} {fc_year}",
            xaxis_title="Hour of Day", yaxis_title="Predicted Trips/hr",
            xaxis=dict(tickmode="linear", tick0=0, dtick=2),
            **_DARK,
        )
        _pchart(fig)
        _kpi_row([
            ("⚡", f"{max(preds):.0f}",        f"Peak Demand ({peak_h}:00)", "trips/hr"),
            ("📉", f"{min(preds):.0f}",        "Lowest Demand",              "trips/hr"),
            ("📊", f"{np.mean(preds):.0f}",    "Daily Average",              "trips/hr"),
            ("💰", f"${max(preds)*fc_fare*0.7:.2f}", "Peak Hour Revenue est.", "70% share"),
        ])

    with tab_d:
        preds_d  = [_fc_pred(18, d, fc_month, fc_year) for d in range(7)]
        peak_day = int(np.argmax(preds_d))
        clrs_d   = ["#EF4444" if i == peak_day else "#3B82F6" for i in range(7)]

        fig_d = go.Figure(go.Bar(
            x=_dow_names, y=preds_d, marker_color=clrs_d,
            text=[f"{p:.0f}" for p in preds_d], textposition="outside",
        ))
        fig_d.update_layout(
            title=f"Day-of-Week Pattern — {fc_lbl.split(' (')[0]} · 18:00 · {fc_mon_lbl} {fc_year}",
            xaxis_title="Day", yaxis_title="Predicted Trips/hr",
            **_DARK,
        )
        _pchart(fig_d)
        _kpi_row([
            ("🏆", _dow_names[peak_day],       "Busiest Day",   "at 18:00"),
            ("⚡", f"{max(preds_d):.0f}",      "Peak Demand",   "trips/hr"),
            ("📊", f"{np.mean(preds_d):.0f}",  "Weekly Avg",    "trips/hr"),
            ("📉", f"{min(preds_d):.0f}",      "Quietest Day",  "trips/hr"),
        ])

    with tab_m:
        preds_m  = [_fc_pred(18, fc_dow, m, fc_year) for m in range(1, 13)]
        peak_mon = int(np.argmax(preds_m))

        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(
            x=_mon_short, y=preds_m, mode="lines+markers",
            line=dict(color="#3B82F6", width=2.5),
            marker=dict(
                color=["#EF4444" if i == peak_mon else "#3B82F6" for i in range(12)],
                size=[12 if i == peak_mon else 7 for i in range(12)],
            ),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
            name="Predicted Demand",
        ))
        fig_m.update_layout(
            title=f"Monthly Pattern — {fc_lbl.split(' (')[0]} · {fc_dow_lbl} · 18:00 · {fc_year}",
            xaxis_title="Month", yaxis_title="Predicted Trips/hr",
            **_DARK,
        )
        _pchart(fig_m)
        _kpi_row([
            ("🏆", _mon_short[peak_mon],       "Busiest Month", f"at 18:00"),
            ("⚡", f"{max(preds_m):.0f}",      "Peak Demand",   "trips/hr"),
            ("📊", f"{np.mean(preds_m):.0f}",  "Annual Avg",    "trips/hr"),
            ("📉", f"{min(preds_m):.0f}",      "Quietest Month","trips/hr"),
        ])

    st.markdown("---")
    st.markdown(
        '<div class="info-banner">'
        '💡 Predictions use the regression model trained on 2023–2026 data. '
        'Future year forecasts (2027+) extrapolate based on learned temporal patterns. '
        'All forecasts are per-hour trip counts for the selected zone.'
        '</div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Router
# ═════════════════════════════════════════════════════════════════════════════

_ROUTES = {
    "overview":    page_overview,
    "analytics":   page_analytics,
    "comparison":  page_comparison,
    "prediction":  page_prediction,
    "realtime":    page_realtime_prediction,
    "demand_map":  page_demand_map,
    "driver":      page_driver_tools,
    "forecast":    page_forecast,
    "zones":       page_recommendations,
    "performance": page_performance,
    "clustering":  page_clustering,
    "regression":  page_regression,
}
_ROUTES[page_key]()
