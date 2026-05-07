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

# ── Must be the very first Streamlit call ─────────────────────────────────────
st.set_page_config(
    page_title    = "TaxiWise — NYC Taxi Analytics",
    page_icon     = "🚕",
    layout        = "wide",
    initial_sidebar_state = "expanded",
)

# ── Src imports ───────────────────────────────────────────────────────────────
from src.data_loader import load_trips, load_zones, compute_demand, get_kpis
from src.model       import get_model, predict_single, get_hot_zones
import src.charts as charts

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
    return load_trips(), load_zones(), compute_demand(), get_kpis()


with st.spinner("Loading NYC Taxi data …"):
    df, zones, demand, kpis = _bootstrap()


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
        "🔮  Demand Prediction":     "prediction",
        "🗺️  Zone Recommendations":  "zones",
        "⚙️  Model Performance":     "performance",
    }
    page_key = PAGES[
        st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    ]

    st.markdown("---")
    st.markdown(f"""
    <div style="color:#6B7280;font-size:.71rem;line-height:1.7;">
      <b style="color:#9CA3AF;">Data</b><br>
      NYC TLC Yellow Taxi 2025<br>
      {kpis['total_trips']:,} trips · 28 features<br><br>
      <b style="color:#9CA3AF;">Model</b><br>
      XGBoost Regressor<br>
      Demand forecasting by zone
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
        '<div class="page-subtitle">Real-time insights · NYC Yellow Taxi 2025</div>',
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
            avail_months = sorted(df["month"].unique())
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
        f'<div class="info-banner">Showing <b>{len(fdf):,}</b> trips</div>',
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
# Page 3 — Demand Prediction
# ═════════════════════════════════════════════════════════════════════════════

def page_prediction():
    st.markdown('<div class="page-title">Demand Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Predict trip demand for any zone & time using XGBoost</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Training model …"):
        model, metrics, fi, y_te, y_pred = get_model()

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
        month_sel  = st.selectbox("Month", mon_names_full)
        month_num  = mon_names_full.index(month_sel) + 1

    # Historical stats for this zone
    z_hist    = demand[demand["PULocationID"] == loc_id]
    if not z_hist.empty:
        zone_total = float(z_hist["zone_total_trips"].iloc[0])
        avg_fare   = float(z_hist["avg_fare"].mean())
        avg_dist   = float(z_hist["avg_distance"].mean())
        avg_dur    = float(z_hist["avg_duration"].mean())
    else:
        zone_total = float(demand["zone_total_trips"].mean())
        avg_fare   = float(demand["avg_fare"].mean())
        avg_dist   = float(demand["avg_distance"].mean())
        avg_dur    = float(demand["avg_duration"].mean())

    pred = predict_single(
        model, loc_id, hour, dow_num, month_num,
        zone_total, avg_fare, avg_dist, avg_dur,
    )

    # Historical average at this zone + hour
    hist_rows = demand[
        (demand["PULocationID"] == loc_id) & (demand["hour"] == hour)
    ]
    hist_avg = float(hist_rows["trip_count"].mean()) if len(hist_rows) > 0 else 0.0
    diff_pct = ((pred - hist_avg) / max(hist_avg, 1)) * 100

    with col_res:
        _section("Prediction Result")
        arrow   = "▲" if diff_pct >= 0 else "▼"
        clr_diff = "#10B981" if diff_pct >= 0 else "#EF4444"
        st.markdown(f"""
        <div class="pred-card">
          <div class="pred-value">{pred:.0f}</div>
          <div class="pred-label">Predicted trips / hour</div>
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
            ("Hist. Trips",  f"{int(zone_total):,}"),
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

    # Explainability
    st.markdown("---")
    _section("Why This Prediction?")

    ex1, ex2 = st.columns([1, 2])
    with ex1:
        peak_hours = list(range(7, 10)) + list(range(17, 20))
        reasons: list[str] = []
        for feat in fi["feature"].tolist()[:4]:
            if feat == "zone_total_trips":
                tier = "high" if zone_total > demand["zone_total_trips"].quantile(0.75) else "moderate"
                reasons.append(f"**Zone historical demand** is {tier} ({int(zone_total):,} total trips).")
            elif feat == "hour":
                period = "peak" if hour in peak_hours else "off-peak"
                reasons.append(f"**{hour}:00** is a {period} hour.")
            elif feat == "avg_fare":
                tier = "premium" if avg_fare > demand["avg_fare"].mean() else "standard"
                reasons.append(f"**Avg fare** ${avg_fare:.2f} — {tier} zone.")
            elif feat == "avg_distance":
                tier = "long" if avg_dist > demand["avg_distance"].mean() else "short"
                reasons.append(f"**Trip distance** avg {avg_dist:.1f} mi — {tier} rides.")
            elif feat == "dow":
                tier = "weekend" if dow_num >= 5 else "weekday"
                reasons.append(f"**{dow_sel}** is a {tier} — typical demand pattern.")
            elif feat == "PULocationID":
                reasons.append(f"**Zone {loc_id}** has a unique demand fingerprint.")
        for r in reasons:
            st.markdown(f"• {r}")

    with ex2:
        _pchart(charts.feature_importance(fi))


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
        model, metrics, fi, y_te, y_pred = get_model()

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
# Page 5 — Model Performance
# ═════════════════════════════════════════════════════════════════════════════

def page_performance():
    st.markdown('<div class="page-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">XGBoost demand-forecasting evaluation & explainability</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Evaluating model …"):
        model, metrics, fi, y_te, y_pred = get_model()

    _section("Performance Metrics")
    _kpi_row([
        ("📉", f"{metrics['mae']:.2f}",     "MAE",           "mean absolute error"),
        ("📊", f"{metrics['rmse']:.2f}",    "RMSE",          "root mean sq. error"),
        ("📈", f"{metrics['r2']:.3f}",      "R² Score",      "explained variance"),
        ("🏋️",f"{metrics['n_train']:,}",   "Train Samples", "aggregated records"),
        ("🧪", f"{metrics['n_test']:,}",    "Test Samples",  "held-out set"),
    ])

    r2 = metrics["r2"]
    if r2 > 0.85:
        badge = "🟢 **Excellent**"
        note  = "Model explains >85 % of demand variance."
    elif r2 > 0.70:
        badge = "🟡 **Good**"
        note  = "Solid predictive power for demand forecasting."
    else:
        badge = "🔴 **Fair**"
        note  = "More historical data would improve accuracy."

    st.info(
        f"{badge} — {note}  "
        f"MAE = **{metrics['mae']:.2f}** trips means predictions "
        f"are off by ~{metrics['mae']:.0f} trips on average."
    )

    st.markdown("---")
    _section("Diagnostics")
    dc1, dc2 = st.columns(2)
    with dc1: _pchart(charts.feature_importance(fi))
    with dc2: _pchart(charts.actual_vs_predicted(y_te, y_pred))

    _section("Feature Importance Details")
    fi_disp = fi[["label","importance"]].copy()
    fi_disp["importance_pct"] = (
        fi_disp["importance"] / fi_disp["importance"].sum() * 100
    ).round(1)
    fi_disp.columns = ["Feature", "Importance Score", "% of Total"]
    fi_disp["Importance Score"] = fi_disp["Importance Score"].round(4)
    st.dataframe(fi_disp, use_container_width=True, hide_index=True)

    _section("Model Configuration")
    cfg_cols = st.columns(3)
    cfgs = [
        ("Algorithm",        "XGBoost Regressor"),
        ("n_estimators",     "400"),
        ("max_depth",        "6"),
        ("learning_rate",    "0.05"),
        ("subsample",        "0.8"),
        ("colsample_bytree", "0.8"),
    ]
    for i, (k, v) in enumerate(cfgs):
        with cfg_cols[i % 3]:
            st.metric(k, v)


# ═════════════════════════════════════════════════════════════════════════════
# Router
# ═════════════════════════════════════════════════════════════════════════════

_ROUTES = {
    "overview":    page_overview,
    "analytics":   page_analytics,
    "prediction":  page_prediction,
    "zones":       page_recommendations,
    "performance": page_performance,
}
_ROUTES[page_key]()
