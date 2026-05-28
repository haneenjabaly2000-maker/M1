"""
TaxiWise — AI Transportation Intelligence Platform
4-page redesign: Dashboard · AI Prediction · AI Insights · Forecasting
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime as _dt

st.set_page_config(
    page_title="TaxiWise — AI Transportation",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.data_loader import load_trips, load_zones, compute_demand, compute_kpis
from src.model       import (load_xgb_model, get_hot_zones,
                              load_regression_model, predict_regression)
import src.charts     as charts
import src.clustering as clust
import src.regression as reg

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark mode + glassmorphism
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important}

.main{background:#0E1117}
.block-container{padding:1.2rem 2rem;max-width:1440px}
section[data-testid="stSidebar"]{background:#0B0D14;border-right:1px solid rgba(255,255,255,.04)}

/* ── gradient page title ── */
.page-title{font-size:2rem;font-weight:900;line-height:1.1;
  background:linear-gradient(90deg,#F7C948 0%,#F97316 50%,#3B82F6 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  margin-bottom:.2rem}
.page-sub{color:#6B7280;font-size:.85rem;margin-bottom:1.2rem}

/* ── section header ── */
.sec{font-size:1rem;font-weight:700;color:#FAFAFA;
  border-left:3px solid #F7C948;padding-left:10px;margin:1.4rem 0 .8rem}

/* ── glass card ── */
.glass{background:rgba(26,29,39,.75);backdrop-filter:blur(14px);
  -webkit-backdrop-filter:blur(14px);border:1px solid rgba(255,255,255,.06);
  border-radius:18px;padding:20px 24px;margin-bottom:1rem}

/* ── KPI grid ── */
.kpi-grid{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:.8rem}
.kpi-card{background:#1A1D27;border:1px solid rgba(247,201,72,.10);border-radius:14px;
  padding:14px 18px;flex:1;min-width:130px;
  transition:transform .18s,border-color .18s}
.kpi-card:hover{transform:translateY(-2px);border-color:rgba(247,201,72,.30)}
.kpi-card.top{animation:pulse-gold 3s infinite}
.kpi-icon{font-size:1.45rem;margin-bottom:4px}
.kpi-value{font-size:1.5rem;font-weight:800;color:#F7C948;line-height:1}
.kpi-label{font-size:.7rem;color:#9CA3AF;text-transform:uppercase;letter-spacing:.06em;margin-top:4px}
.kpi-sub{font-size:.66rem;color:#6B7280;margin-top:2px}
@keyframes pulse-gold{0%{box-shadow:0 0 0 rgba(247,201,72,0)}
  50%{box-shadow:0 0 18px rgba(247,201,72,.18)}100%{box-shadow:0 0 0 rgba(247,201,72,0)}}

/* ── prediction card ── */
.pred-main{background:linear-gradient(135deg,#1A1D27,#252836);
  border:2px solid rgba(247,201,72,.28);border-radius:20px;
  padding:22px 26px;text-align:center;margin-bottom:10px}
.pred-number{font-size:3.4rem;font-weight:900;color:#F7C948;line-height:1}
.pred-unit{color:#9CA3AF;font-size:.82rem;margin-top:4px}

/* ── alert cards ── */
.alert-extreme{background:rgba(239,68,68,.08);border-left:4px solid #EF4444;
  border-radius:8px;padding:11px 15px;margin:6px 0;color:#FCA5A5;font-size:.84rem;font-weight:600}
.alert-high{background:rgba(249,115,22,.08);border-left:4px solid #F97316;
  border-radius:8px;padding:11px 15px;margin:6px 0;color:#FDBA74;font-size:.84rem;font-weight:600}
.alert-ok{background:rgba(16,185,129,.07);border-left:4px solid #10B981;
  border-radius:8px;padding:11px 15px;margin:6px 0;color:#6EE7B7;font-size:.84rem;font-weight:600}

/* ── AI assistant card ── */
.ai-assist{background:linear-gradient(135deg,rgba(59,130,246,.10),rgba(247,201,72,.07));
  border:1px solid rgba(247,201,72,.22);border-radius:16px;padding:16px 20px;margin-top:14px}
.ai-hdr{font-size:.88rem;font-weight:700;color:#F7C948;margin-bottom:10px}
.ai-zone{font-size:1.15rem;font-weight:800;color:#FAFAFA;margin-bottom:6px}
.ai-detail{color:#9CA3AF;font-size:.77rem;line-height:1.7}

/* ── insight cards ── */
.insight{background:#1E2130;border-left:3px solid #F7C948;border-radius:6px;
  padding:9px 13px;margin-bottom:7px}
.insight b{color:#FAFAFA}
.insight-detail{color:#9CA3AF;font-size:.76rem;margin-top:2px}

/* ── revenue card ── */
.rev-card{background:#1A1D27;border:1px solid rgba(16,185,129,.20);border-radius:14px;
  padding:14px 18px;margin-bottom:10px}

/* ── pill row ── */
.pill-row{display:flex;flex-wrap:wrap;gap:7px;margin-top:8px}
.pill{background:#252836;border:1px solid rgba(255,255,255,.06);border-radius:8px;
  padding:7px 13px;text-align:center;min-width:90px}
.pill-val{font-size:1.05rem;font-weight:700;color:#F7C948}
.pill-lbl{font-size:.66rem;color:#9CA3AF;text-transform:uppercase;margin-top:2px}

/* ── zone rec cards ── */
.zone-card{background:#1A1D27;border-left:4px solid #F7C948;
  border-radius:0 10px 10px 0;padding:11px 15px;margin-bottom:8px}
.zone-card.r1{border-left-color:#EF4444}.zone-card.r2{border-left-color:#F97316}
.zone-name{font-size:.96rem;font-weight:700;color:#FAFAFA}
.zone-badge{display:inline-block;background:rgba(247,201,72,.12);color:#F7C948;
  font-size:.63rem;font-weight:600;padding:1px 7px;border-radius:20px;margin-left:5px}
.zone-stats{color:#9CA3AF;font-size:.76rem;margin-top:3px}
.zone-why{color:#6EE7B7;font-size:.73rem;margin-top:2px}

/* ── tabs ── */
[data-testid="stTabs"] button{font-weight:600;color:#9CA3AF!important}
[data-testid="stTabs"] button[aria-selected="true"]{color:#F7C948!important}

/* ── sidebar nav ── */
[data-testid="stRadio"] label{font-size:.88rem;padding:6px 0}

/* ── info banner ── */
.banner{background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.22);
  border-radius:10px;padding:9px 15px;color:#93C5FD;font-size:.81rem;margin:.6rem 0}

::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#0B0D14}
::-webkit-scrollbar-thumb{background:#2D3044;border-radius:3px}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data bootstrap
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _bootstrap():
    return load_trips(), load_zones(), compute_demand()


with st.spinner("Loading NYC Taxi data …"):
    try:
        df_all, zones, demand = _bootstrap()
        if df_all.empty:
            st.error("Data not loaded. Run `python prepare_data.py` locally and push.")
            st.stop()
    except Exception as _e:
        st.error(f"Data error: {_e}")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 14px">
      <div style="font-size:1.5rem;font-weight:900;color:#F7C948">🚕 TaxiWise</div>
      <div style="color:#6B7280;font-size:.72rem;margin-top:1px">AI Transportation Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:rgba(255,255,255,.05);margin-bottom:12px"></div>',
                unsafe_allow_html=True)

    PAGES = {
        "🚕  Dashboard":      "dashboard",
        "🤖  AI Prediction":  "ai_pred",
        "📈  AI Insights":    "ai_insights",
        "🔮  Forecasting":    "forecasting",
    }
    page_key = PAGES[
        st.radio("nav", list(PAGES.keys()), label_visibility="collapsed")
    ]

    st.markdown('<div style="height:1px;background:rgba(255,255,255,.05);margin:12px 0"></div>',
                unsafe_allow_html=True)
    st.markdown('<div style="color:#6B7280;font-size:.7rem;font-weight:600;letter-spacing:.05em;margin-bottom:5px">YEAR FILTER</div>',
                unsafe_allow_html=True)
    sel_years = st.multiselect("yf", [2023,2024,2025,2026],
                               default=[2023,2024,2025,2026],
                               label_visibility="collapsed")

# Apply year filter
active_years = sorted(sel_years) if sel_years else [2023,2024,2025,2026]
df       = df_all[df_all["year"].isin(active_years)].reset_index(drop=True)
kpis     = compute_kpis(df)
yrs_str  = " · ".join(str(y) for y in active_years)

with st.sidebar:
    st.markdown('<div style="height:1px;background:rgba(255,255,255,.05);margin:12px 0"></div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style="color:#4B5563;font-size:.69rem;line-height:1.75">
      <b style="color:#6B7280">Data</b><br>
      NYC TLC Yellow Taxi<br>
      {yrs_str}<br>
      {kpis['total_trips']:,} trips<br><br>
      <b style="color:#6B7280">Models</b><br>
      LR · RF · XGBoost
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
_DOW   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
_MON   = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]
_MONS  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# Today's defaults — set once per session so all selectors start at the current date/time
_today      = _dt.now()
_now_year   = _today.year
_now_mon    = _today.month      # 1–12
_now_dow    = _today.weekday()  # 0=Mon … 6=Sun
_now_hour   = _today.hour
_YEAR_LIST  = list(range(2023, 2036))
_year_idx   = _YEAR_LIST.index(_now_year) if _now_year in _YEAR_LIST else 3


def _kpi_row(items, top_idx=0):
    html = '<div class="kpi-grid">'
    for i,(icon,val,label,sub) in enumerate(items):
        cls = "kpi-card top" if i == top_idx else "kpi-card"
        html += (f'<div class="{cls}"><div class="kpi-icon">{icon}</div>'
                 f'<div class="kpi-value">{val}</div>'
                 f'<div class="kpi-label">{label}</div>'
                 + (f'<div class="kpi-sub">{sub}</div>' if sub else "")
                 + "</div>")
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _section(t):
    st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)


def _pchart(fig, h=None, **kw):
    if h:
        fig.update_layout(height=h)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, **kw)


def _zone_defaults(loc_id):
    z = demand[demand["PULocationID"] == loc_id]
    s = z if not z.empty else demand
    return {
        "hist": float(s["zone_total_trips"].iloc[0]) if not z.empty
                else float(demand["zone_total_trips"].median()),
        "fare": max(1.0,  min(500.0, float(s["avg_fare"].mean()))),
        "dist": max(0.1,  min(100.0, float(s["avg_distance"].mean()))),
        "dur":  max(1.0,  min(300.0, float(s["avg_duration"].mean()))),
    }


def _zone_label_list():
    z = zones[["LocationID","Zone","Borough"]].copy()
    z["label"] = z.apply(lambda r: f"{r['Zone']} ({r['Borough']}) — ID {r['LocationID']}", axis=1)
    return z["label"].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Demand map builder (cached by time params)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _zone_preds(hour: int, dow: int, month: int) -> pd.DataFrame:
    from src.model import FEATURE_COLS
    from src.zone_coords import get_zone_coord
    model, *_ = load_xgb_model()

    zs = (demand.groupby("PULocationID")
          .agg(avg_fare        =("avg_fare",         "mean"),
               avg_distance    =("avg_distance",     "mean"),
               avg_duration    =("avg_duration",     "mean"),
               zone_total_trips=("zone_total_trips",  "first"),
               hist_demand     =("trip_count",        "mean"))
          .reset_index())
    zs["hour"]  = hour
    zs["dow"]   = dow
    zs["month"] = month
    for col in ["avg_fare","avg_distance","avg_duration","zone_total_trips"]:
        zs[col] = zs[col].fillna(float(demand[col].median()))
    zs = zs.dropna(subset=FEATURE_COLS)
    if zs.empty:
        return pd.DataFrame()
    zs["predicted_demand"] = np.maximum(model.predict(zs[FEATURE_COLS].values), 0)

    merged = zs.merge(zones[["LocationID","Zone","Borough"]],
                      left_on="PULocationID", right_on="LocationID", how="left")
    merged["Zone"]    = merged["Zone"].fillna("Zone " + merged["PULocationID"].astype(str))
    merged["Borough"] = merged["Borough"].fillna("Unknown")
    coords = [get_zone_coord(int(r["PULocationID"]), str(r["Borough"])) for _,r in merged.iterrows()]
    merged["lat"] = [c[0] for c in coords]
    merged["lon"] = [c[1] for c in coords]

    p33 = float(np.percentile(merged["predicted_demand"], 33))
    p66 = float(np.percentile(merged["predicted_demand"], 66))
    merged["Demand Level"]         = merged["predicted_demand"].apply(
        lambda x: "Low" if x<p33 else ("Medium" if x<p66 else "High"))
    merged["Predicted Trips/hr"]   = merged["predicted_demand"].round(1)
    merged["Avg Fare ($)"]         = merged["avg_fare"].round(2)
    merged["Revenue est ($/hr)"]   = (merged["predicted_demand"]*merged["avg_fare"]*0.7).round(2)
    return merged


def _build_map(merged: pd.DataFrame, sel_id: int | None = None, height: int = 360):
    import plotly.express as px
    import plotly.graph_objects as go

    if merged.empty:
        return go.Figure()

    fig = px.scatter_mapbox(
        merged, lat="lat", lon="lon",
        color="predicted_demand",
        size="predicted_demand", size_max=30,
        hover_name="Zone",
        hover_data={"Borough":True,"Predicted Trips/hr":True,"Avg Fare ($)":True,
                    "Revenue est ($/hr)":True,"Demand Level":True,
                    "lat":False,"lon":False,"predicted_demand":False},
        color_continuous_scale=[[0,"#3B82F6"],[0.5,"#F97316"],[1.0,"#EF4444"]],
        mapbox_style="carto-darkmatter",
        zoom=10, center={"lat":40.730,"lon":-73.985}, opacity=0.88,
    )
    if sel_id is not None:
        row = merged[merged["PULocationID"] == sel_id]
        if not row.empty:
            fig.add_trace(go.Scattermapbox(
                lat=row["lat"], lon=row["lon"], mode="markers",
                marker=dict(size=22, color="#FFFFFF", opacity=0.9),
                hoverinfo="skip", showlegend=False,
            ))
    fig.update_layout(
        height=height, paper_bgcolor="#1A1D27",
        font=dict(color="#FAFAFA"),
        coloraxis_colorbar=dict(title="Trips/hr", tickfont=dict(color="#9CA3AF")),
        margin={"r":0,"t":0,"l":0,"b":0},
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Auto-insights
# ─────────────────────────────────────────────────────────────────────────────
def _auto_insights(df_in: pd.DataFrame, kpis_in: dict) -> list[tuple]:
    out = []
    ph = kpis_in["peak_hour"]
    if 7 <= ph <= 9:
        out.append(("🌅","Morning Rush Dominates",
                    f"Peak demand is at {ph}:00 — NYC commuters drive highest ridership","#F97316"))
    elif 17 <= ph <= 19:
        out.append(("🌆","Evening Rush Peak",
                    f"Peak demand at {ph}:00 — evening commute fuels trip volume","#F97316"))
    else:
        out.append(("🌙","Off-Peak Pattern",
                    f"Unusual peak at {ph}:00 — possible event or nightlife cluster","#8B5CF6"))

    tz = str(kpis_in.get("top_zone",""))
    if any(k in tz for k in ("Airport","JFK","LaGuardia","EWR")):
        out.append(("✈️","Airport Demand Spike",
                    f"{tz[:30]} leads — airport travel consistently drives volume","#3B82F6"))
    elif "Midtown" in tz:
        out.append(("🏙️","Midtown Hotspot",
                    f"{tz[:30]} is #1 — business district generates steady demand","#F7C948"))
    else:
        out.append(("📍","Top Zone",
                    f"{tz[:30]} leads in total trip volume","#10B981"))

    cp = kpis_in.get("credit_pct", 0)
    if cp > 70:
        out.append(("💳","Digital-First City",
                    f"{cp:.0f}% of trips paid by card — NYC trending cashless","#10B981"))

    if "dow" in df_in.columns and len(df_in) > 100:
        we = df_in[df_in["dow"] >= 5]
        wd = df_in[df_in["dow"] < 5]
        if len(we) > 0 and len(wd) > 0:
            we_avg = len(we) / max(we["dow"].nunique(), 1)
            wd_avg = len(wd) / max(wd["dow"].nunique(), 1)
            if we_avg > wd_avg * 1.08:
                out.append(("🎉","Weekend Surge",
                            "Weekends outpace weekdays — leisure trips boost revenue","#8B5CF6"))
            else:
                out.append(("💼","Weekday Dominance",
                            "Weekdays drive more trips — commuter demand leads","#3B82F6"))

    return out[:4]


# ═════════════════════════════════════════════════════════════════════════════
# Page 1 — Dashboard
# ═════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown('<div class="page-title">TaxiWise Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">NYC Yellow Taxi · {yrs_str} · {kpis["total_trips"]:,} trips</div>',
                unsafe_allow_html=True)

    # KPIs
    _kpi_row([
        ("🚖", f"{kpis['total_trips']:,}",      "Total Trips",    ""),
        ("💰", f"${kpis['avg_fare']:.2f}",       "Avg Fare",       "per trip"),
        ("📍", f"{kpis['avg_distance']:.1f} mi", "Avg Distance",   "per trip"),
        ("⏱️",f"{kpis['avg_duration']:.1f} min","Avg Duration",   "per trip"),
        ("⚡", f"{kpis['peak_hour']}:00",         "Peak Hour",      "most demand"),
        ("🗺️", f"{kpis['active_zones']}",        "Active Zones",   "pickup areas"),
    ], top_idx=0)

    _kpi_row([
        ("💳", f"{kpis['credit_pct']:.1f}%",    "Credit Card",    "of payments"),
        ("🏆", kpis['top_zone'][:28],             "Busiest Zone",   ""),
        ("💵", f"${kpis['total_revenue']:,.0f}", "Total Revenue",  "gross fares"),
    ], top_idx=2)

    # AI Auto-Insights
    _section("🧠 AI Auto-Insights")
    insights = _auto_insights(df, kpis)
    cols_ins = st.columns(len(insights))
    for col, (icon, title, detail, color) in zip(cols_ins, insights):
        with col:
            st.markdown(
                f'<div style="background:rgba(26,29,39,.8);border-top:3px solid {color};'
                f'border-radius:12px;padding:14px 16px;height:100%">'
                f'<div style="font-size:1.4rem;margin-bottom:6px">{icon}</div>'
                f'<div style="font-weight:700;color:#FAFAFA;font-size:.88rem;margin-bottom:4px">{title}</div>'
                f'<div style="color:#9CA3AF;font-size:.75rem;line-height:1.5">{detail}</div>'
                f'</div>', unsafe_allow_html=True,
            )

    st.markdown("---")

    # Filters
    with st.expander("🎛️  Filters", expanded=False):
        fa, fb, fc = st.columns(3)
        with fa:
            boros = ["All"] + sorted(df["pickup_borough"].dropna().unique().tolist()) \
                    if "pickup_borough" in df.columns else ["All"]
            sel_boro = st.selectbox("Borough", boros, key="db_boro")
        with fb:
            avail_m = sorted(int(m) for m in df["month"].dropna().unique() if 1<=int(m)<=12)
            avail_m_lbl = [_MONS[m-1] for m in avail_m]
            sel_m_lbl = st.multiselect("Month", avail_m_lbl, default=avail_m_lbl, key="db_mon")
            sel_months = [avail_m[avail_m_lbl.index(l)] for l in sel_m_lbl] if sel_m_lbl else avail_m
        with fc:
            hr_range = st.slider("Hour range", 0, 23, (0,23), key="db_hr")

    fdf = df.copy()
    if sel_boro != "All" and "pickup_borough" in fdf.columns:
        fdf = fdf[fdf["pickup_borough"] == sel_boro]
    fdf = fdf[fdf["month"].isin(sel_months)]
    fdf = fdf[(fdf["hour"] >= hr_range[0]) & (fdf["hour"] <= hr_range[1])]

    if fdf.empty:
        st.warning("No trips match the current filters.")
        return

    _section("Demand Patterns")
    c1, c2 = st.columns(2)
    with c1: _pchart(charts.trips_by_hour(fdf), h=320)
    with c2: _pchart(charts.trips_by_dow(fdf),  h=320)

    _section("Trends & Geography")
    c3, c4 = st.columns(2)
    with c3: _pchart(charts.monthly_trend(fdf), h=320)
    with c4: _pchart(charts.top_zones(fdf, top_n=10), h=320)

    _section("Demand Heatmap (Hour × Day)")
    _pchart(charts.demand_heatmap(fdf), h=300)

    if "pickup_borough" in fdf.columns:
        _section("Borough Flow")
        _pchart(charts.borough_flow(fdf), h=320)

    # Year comparison summary
    _section("Year-over-Year Summary")
    yoy = df_all.groupby("year").agg(
        Trips=("fare_amount","count"),
        Fare=("fare_amount","mean"),
        Distance=("trip_distance","mean"),
        Duration=("trip_duration_min","mean"),
    ).reset_index().rename(columns={"year":"Year"})
    yoy["Year"]     = yoy["Year"].astype(str)
    yoy["Trips"]    = yoy["Trips"].apply(lambda x: f"{x:,}")
    yoy["Fare"]     = yoy["Fare"].apply(lambda x: f"${x:.2f}")
    yoy["Distance"] = yoy["Distance"].apply(lambda x: f"{x:.1f} mi")
    yoy["Duration"] = yoy["Duration"].apply(lambda x: f"{x:.1f} min")
    st.dataframe(yoy, use_container_width=True, hide_index=True)

    c5, c6 = st.columns(2)
    with c5: _pchart(charts.yearly_trip_comparison(df_all), h=300)
    with c6: _pchart(charts.yearly_fare_trend(df_all),      h=300)


# ═════════════════════════════════════════════════════════════════════════════
# Page 2 — AI Prediction
# ═════════════════════════════════════════════════════════════════════════════
def page_ai_prediction():
    st.markdown('<div class="page-title">AI Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Real-time demand forecasting · Generalization to unseen inputs · Explainable AI</div>',
                unsafe_allow_html=True)

    with st.spinner("Loading AI model …"):
        payload = load_regression_model()

    m          = payload["metrics"]
    model_name = payload["model_name"]
    model_obj  = payload["model"]
    feat_cols  = payload["feature_cols"]
    y_test     = payload["y_test"]

    labels = _zone_label_list()

    # ── Layout: left form | right map+results ────────────────────────────────
    col_l, col_r = st.columns([1, 1.35], gap="large")

    with col_l:
        st.markdown('<div class="sec">⚙️ Input Parameters</div>', unsafe_allow_html=True)

        zone_lbl = st.selectbox("Pickup Zone", labels, index=0, key="ap_zone")
        loc_id   = int(zone_lbl.split("— ID ")[-1])
        defs     = _zone_defaults(loc_id)

        tc1, tc2 = st.columns(2)
        with tc1:
            hour = st.slider("Hour", 0, 23, _now_hour, key="ap_hour",
                             help="0=midnight · 8=morning rush · 18=evening rush")
        with tc2:
            year_sel = st.selectbox("Year", _YEAR_LIST, index=_year_idx, key="ap_year")

        dc1, dc2 = st.columns(2)
        with dc1:
            dow_sel  = st.selectbox("Day of Week", _DOW, index=_now_dow, key="ap_dow")
            dow_num  = _DOW.index(dow_sel)
        with dc2:
            mon_sel  = st.selectbox("Month", _MON, index=_now_mon - 1, key="ap_month")
            mon_num  = _MON.index(mon_sel) + 1

        st.markdown('<div style="margin-top:12px;font-size:.82rem;font-weight:600;color:#9CA3AF">TRIP STATISTICS <span style="font-weight:400">(editable)</span></div>',
                    unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        with sc1:
            avg_fare = st.number_input("Avg Fare ($)",       1.0, 500.0, float(round(defs["fare"],2)), 0.5, key="ap_fare")
            avg_dist = st.number_input("Distance (mi)",      0.1, 100.0, float(round(defs["dist"],2)), 0.1, key="ap_dist")
        with sc2:
            avg_dur  = st.number_input("Duration (min)",     1.0, 300.0, float(round(defs["dur"],1)),  1.0, key="ap_dur")
            hist_cnt = st.number_input("Historical Demand",  0,  500000, int(defs["hist"]),            100, key="ap_hist")
        pax = st.slider("Passengers", 1, 6, 1, key="ap_pax")

        # ── Build features + predict ─────────────────────────────────────────
        features = {
            "pickup_location_id":    float(loc_id),
            "pickup_hour":           float(hour),
            "pickup_day_of_week":    float(dow_num),
            "pickup_month":          float(mon_num),
            "historical_trip_count": float(hist_cnt),
            "avg_fare_amount":       float(avg_fare),
            "avg_trip_distance":     float(avg_dist),
            "avg_trip_duration":     float(avg_dur),
            "year":                  float(year_sel),
        }
        pred = predict_regression(payload, features)

        # Confidence range (RF trees)
        ci_lo = ci_hi = None
        if hasattr(model_obj, "estimators_"):
            X_raw   = np.array([[features[f] for f in feat_cols]], dtype=float)
            t_preds = np.maximum([e.predict(X_raw)[0] for e in model_obj.estimators_], 0)
            ci_lo   = float(np.percentile(t_preds, 10))
            ci_hi   = float(np.percentile(t_preds, 90))

        # Demand level
        p25 = float(np.percentile(y_test, 25))
        p75 = float(np.percentile(y_test, 75))
        p90 = float(np.percentile(y_test, 90))
        if pred < p25:
            level, lclr, lemoji = "Low",       "#6B7280", "🟤"
        elif pred < p75:
            level, lclr, lemoji = "Medium",    "#F7C948", "🟡"
        elif pred < p90:
            level, lclr, lemoji = "High",      "#F97316", "🟠"
        else:
            level, lclr, lemoji = "Very High", "#EF4444", "🔴"

        # Revenue estimate
        rev_hr  = pred * avg_fare * 0.70
        rev_day = rev_hr * 8

        # ── AI Driver Assistant ──────────────────────────────────────────────
        zp = _zone_preds(hour, dow_num, mon_num)
        if not zp.empty:
            best_row  = zp.nlargest(1, "predicted_demand").iloc[0]
            best_name = str(best_row.get("Zone", f"Zone {best_row['PULocationID']}"))
            best_dem  = float(best_row["predicted_demand"])
            best_rev  = float(best_row.get("Revenue est ($/hr)", best_dem * avg_fare * 0.7))
            best_boro = str(best_row.get("Borough", ""))
            same_zone = best_row["PULocationID"] == loc_id

            st.markdown(f"""
            <div class="ai-assist">
              <div class="ai-hdr">🤖 AI Driver Assistant</div>
              <div class="ai-zone">{"✅ You're in the right place!" if same_zone else f"→ {best_name}"}</div>
              <div class="ai-detail">
                {"Your zone is currently the hottest in NYC." if same_zone else f"Borough: {best_boro}"}
                <br>📈 Demand: <b style="color:#F7C948">{best_dem:.0f} trips/hr</b>
                &nbsp;·&nbsp; 💵 Rev est: <b style="color:#10B981">${best_rev:.2f}/hr</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        # Map
        st.markdown('<div class="sec">🌡️ Live Demand Map</div>', unsafe_allow_html=True)
        if not zp.empty:
            fig_map = _build_map(zp, sel_id=loc_id, height=340)
            st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": True},
                            key="ap_map")
        else:
            st.info("Map data not available.")

        # Prediction card
        hist_rows = demand[(demand["PULocationID"]==loc_id) & (demand["hour"]==hour)]
        hist_avg  = float(hist_rows["trip_count"].mean()) if len(hist_rows) > 0 else 0.0
        diff_pct  = ((pred - hist_avg) / max(hist_avg, 1)) * 100
        arr  = "▲" if diff_pct >= 0 else "▼"
        clrd = "#10B981" if diff_pct >= 0 else "#EF4444"
        conf = (max(0, 100-(ci_hi-ci_lo)/max(pred,1)*50)) if ci_lo is not None else None
        ci_html = (f'<div style="font-size:.76rem;color:#9CA3AF;margin-top:6px">'
                   f'Range: <b style="color:#FAFAFA">{ci_lo:.0f}–{ci_hi:.0f}</b> trips'
                   f'{"  ·  Confidence: "+str(int(conf))+"%" if conf else ""}</div>'
                   ) if ci_lo is not None else ""

        st.markdown(f"""
        <div class="pred-main">
          <div class="pred-number">{pred:.0f}</div>
          <div class="pred-unit">predicted trips / hour</div>
          <div style="margin-top:10px">
            <span style="background:{lclr};color:#fff;font-weight:700;
                   padding:4px 18px;border-radius:20px;font-size:.86rem">
              {lemoji} {level} Demand
            </span>
          </div>
          <div style="margin-top:10px;font-size:.78rem;color:#9CA3AF">
            Historical avg {hour:02d}:00 → <b style="color:#FAFAFA">{hist_avg:.0f} trips</b>
            &nbsp;<span style="color:{clrd}">{arr} {abs(diff_pct):.1f}%</span>
          </div>
          {ci_html}
        </div>
        """, unsafe_allow_html=True)

        # Revenue card
        st.markdown(f"""
        <div class="rev-card">
          <div style="font-size:.75rem;color:#9CA3AF;font-weight:600;margin-bottom:8px">💵 REVENUE ESTIMATE (70% driver share)</div>
          <div style="display:flex;gap:20px">
            <div>
              <div style="font-size:1.5rem;font-weight:800;color:#10B981">${rev_hr:.2f}</div>
              <div style="font-size:.7rem;color:#6B7280">per hour</div>
            </div>
            <div>
              <div style="font-size:1.5rem;font-weight:800;color:#10B981">${rev_day:.2f}</div>
              <div style="font-size:.7rem;color:#6B7280">8-hour shift</div>
            </div>
            <div>
              <div style="font-size:1.5rem;font-weight:800;color:#F7C948">${avg_fare:.2f}</div>
              <div style="font-size:.7rem;color:#6B7280">avg fare/trip</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Smart alerts
        if level == "Very High":
            st.markdown('<div class="alert-extreme">🔴 Extreme Demand — Surge pricing likely active. Head here now for maximum earnings.</div>',
                        unsafe_allow_html=True)
        elif level == "High":
            st.markdown('<div class="alert-high">🟠 High Demand Alert — Strong pickup opportunities. Position yourself in this zone.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-ok">🟢 Normal Conditions — Stable demand expected for this zone and time.</div>',
                        unsafe_allow_html=True)

    # ── Explainability (full width) ──────────────────────────────────────────
    st.markdown("---")
    _section("🧠 Why This Prediction? — Explainable AI")

    def _ins(icon, title, detail, color):
        return (f'<div class="insight"><span>{icon}</span> <b>{title}</b>'
                f'<div class="insight-detail">{detail}</div></div>')

    ins_html = ""
    if hour in range(7,10):
        ins_html += _ins("🌅","Morning Rush Hour",
            f"{hour}:00 — highest commuter demand window (7–9)", "#F97316")
    elif hour in range(17,20):
        ins_html += _ins("🌆","Evening Rush Hour",
            f"{hour}:00 — peak evening commute (17–19)", "#F97316")
    elif hour >= 22 or hour < 3:
        ins_html += _ins("🌙","Night Hours",
            f"{hour}:00 — lower demand, longer trips on average", "#6B7280")
    else:
        ins_html += _ins("☀️","Standard Hours",
            f"{hour}:00 — average activity window", "#3B82F6")

    ins_html += _ins("📅","Weekday" if dow_num < 5 else "Weekend",
        f"{dow_sel} — {'commuter patterns dominate' if dow_num<5 else 'leisure & nightlife trips increase'}",
        "#3B82F6" if dow_num<5 else "#8B5CF6")

    zone_pct = float(np.mean(demand["zone_total_trips"] <= hist_cnt) * 100)
    zone_info = zones[zones["LocationID"] == loc_id]
    zname = zone_info["Zone"].iloc[0] if not zone_info.empty else f"Zone {loc_id}"
    if zone_pct >= 80:
        ins_html += _ins("📍","High-Demand Zone",
            f"{zname} — Top {100-int(zone_pct)}% by historical volume", "#10B981")
    elif zone_pct <= 20:
        ins_html += _ins("📍","Low-Demand Zone",
            f"{zname} — {int(zone_pct)}th percentile, below average activity", "#6B7280")
    else:
        ins_html += _ins("📍","Average Zone",
            f"{zname} — {int(zone_pct)}th percentile", "#3B82F6")

    dist_med = float(demand["avg_distance"].median())
    if avg_dist > dist_med * 1.5:
        ins_html += _ins("🛣️","Long Trip",
            f"{avg_dist:.1f} mi — {((avg_dist/dist_med)-1)*100:.0f}% above median ({dist_med:.1f} mi)", "#F7C948")
    elif avg_dist < dist_med * 0.5:
        ins_html += _ins("🛣️","Short Trip",
            f"{avg_dist:.1f} mi — below median ({dist_med:.1f} mi)", "#6B7280")
    else:
        ins_html += _ins("🛣️","Typical Distance",
            f"{avg_dist:.1f} mi — near median ({dist_med:.1f} mi)", "#3B82F6")

    seasons = {
        (12,1,2):("❄️","Winter",  "Cold months — weather may suppress demand","#6B7280"),
        (3,4,5): ("🌸","Spring",  "Balanced demand — pleasant conditions","#3B82F6"),
        (6,7,8): ("☀️","Summer",  "Tourism & leisure boost trip volume","#F97316"),
        (9,10,11):("🍂","Autumn", "Steady demand — average seasonal pattern","#8B5CF6"),
    }
    for mgrp,(icon,nm,det,clr) in seasons.items():
        if mon_num in mgrp:
            ins_html += _ins(icon, f"{nm} — {_MONS[mon_num-1]}", det, clr)
            break

    if pax >= 3:
        ins_html += _ins("👥","Group Trip",
            f"{pax} passengers — group rides often shorter distances", "#8B5CF6")

    st.markdown(f'<div style="columns:2;column-gap:14px">{ins_html}</div>',
                unsafe_allow_html=True)

    # ── Relocation Simulator ─────────────────────────────────────────────────
    with st.expander("🚗  Relocation Simulator — Should I move zones?"):
        r1, r2 = st.columns(2)
        with r1:
            tgt_lbl = st.selectbox("Target Zone", labels,
                                   index=min(1, len(labels)-1), key="rs_tgt")
            tgt_id  = int(tgt_lbl.split("— ID ")[-1])
        with r2:
            rs_yr = st.selectbox("Year", _YEAR_LIST, index=_year_idx, key="rs_yr")

        td = _zone_defaults(tgt_id)
        tgt_f = {
            "pickup_location_id":    float(tgt_id),
            "pickup_hour":           float(hour),
            "pickup_day_of_week":    float(dow_num),
            "pickup_month":          float(mon_num),
            "historical_trip_count": td["hist"],
            "avg_fare_amount":       td["fare"],
            "avg_trip_distance":     td["dist"],
            "avg_trip_duration":     td["dur"],
            "year":                  float(rs_yr),
        }
        tgt_pred = predict_regression(payload, tgt_f)
        d_abs = tgt_pred - pred
        d_pct = (d_abs / max(pred, 1)) * 100
        cur_rev_rs = pred * avg_fare * 0.7
        tgt_rev_rs = tgt_pred * td["fare"] * 0.7

        if d_pct > 20:   rc,rt = "#10B981","✅ Strongly Recommended"
        elif d_pct > 5:  rc,rt = "#F7C948","⚡ Recommended"
        elif d_pct > -5: rc,rt = "#3B82F6","ℹ️ Neutral"
        else:            rc,rt = "#EF4444","⚠️ Not Recommended"

        tgt_zname = zones[zones["LocationID"]==tgt_id]["Zone"].iloc[0] \
                    if tgt_id in zones["LocationID"].values else f"Zone {tgt_id}"

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1A1D27,#252836);
             border:1.5px solid {rc};border-radius:14px;padding:18px 22px">
          <b style="color:#FAFAFA">Moving → {tgt_zname}</b>
          <div style="display:flex;gap:24px;margin-top:10px;flex-wrap:wrap">
            <div><div style="color:#9CA3AF;font-size:.73rem">Demand Δ</div>
              <div style="font-size:1.4rem;font-weight:800;color:#F7C948">
                {"▲" if d_abs>=0 else "▼"} {abs(d_abs):.0f} trips/hr ({d_pct:+.1f}%)
              </div></div>
            <div><div style="color:#9CA3AF;font-size:.73rem">Revenue Δ/hr</div>
              <div style="font-size:1.4rem;font-weight:800;color:{"#10B981" if tgt_rev_rs>=cur_rev_rs else "#EF4444"}">
                {"▲" if tgt_rev_rs>=cur_rev_rs else "▼"} ${abs(tgt_rev_rs-cur_rev_rs):.2f}
              </div></div>
            <div><div style="color:#9CA3AF;font-size:.73rem">Recommendation</div>
              <div style="font-size:1rem;font-weight:700;color:{rc}">{rt}</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        _kpi_row([
            ("📍", f"{pred:.0f}",     f"Current · {zname[:18]}", "trips/hr"),
            ("🎯", f"{tgt_pred:.0f}", f"Target · {tgt_zname[:18]}", "trips/hr"),
            ("💰", f"${cur_rev_rs:.2f}", "Current revenue/hr", "70% share"),
            ("💵", f"${tgt_rev_rs:.2f}", "Target revenue/hr",  "70% share"),
        ])

    # ── What If? Scenario ────────────────────────────────────────────────────
    with st.expander("🎯  What If? Scenario Simulator"):
        st.markdown('<div style="color:#9CA3AF;font-size:.82rem;margin-bottom:12px">'
                    'Change one variable and see how demand shifts.</div>',
                    unsafe_allow_html=True)
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            wi_hour = st.slider("What if Hour?", 0, 23, hour, key="wi_hour")
        with wc2:
            wi_dow_lbl = st.selectbox("What if Day?", _DOW, index=dow_num, key="wi_dow")
            wi_dow = _DOW.index(wi_dow_lbl)
        with wc3:
            wi_year = st.selectbox("What if Year?", _YEAR_LIST,
                                   index=_YEAR_LIST.index(year_sel), key="wi_year")

        wi_f = {**features,
                "pickup_hour": float(wi_hour),
                "pickup_day_of_week": float(wi_dow),
                "year": float(wi_year)}
        wi_pred = predict_regression(payload, wi_f)
        wi_d = wi_pred - pred
        wi_p = (wi_d / max(pred,1)) * 100

        wca, wcb = st.columns(2)
        with wca:
            st.markdown(f"""
            <div style="background:#1A1D27;border:1px solid rgba(255,255,255,.06);
                 border-radius:12px;padding:16px;text-align:center">
              <div style="color:#9CA3AF;font-size:.75rem;margin-bottom:6px">CURRENT SCENARIO</div>
              <div style="font-size:2rem;font-weight:800;color:#F7C948">{pred:.0f}</div>
              <div style="color:#9CA3AF;font-size:.78rem">trips/hr · {dow_sel[:3]} {hour:02d}:00 · {year_sel}</div>
            </div>""", unsafe_allow_html=True)
        with wcb:
            clr_wi = "#10B981" if wi_d >= 0 else "#EF4444"
            st.markdown(f"""
            <div style="background:#1A1D27;border:1px solid {clr_wi}40;
                 border-radius:12px;padding:16px;text-align:center">
              <div style="color:#9CA3AF;font-size:.75rem;margin-bottom:6px">WHAT IF SCENARIO</div>
              <div style="font-size:2rem;font-weight:800;color:{clr_wi}">{wi_pred:.0f}</div>
              <div style="color:#9CA3AF;font-size:.78rem">trips/hr · {wi_dow_lbl[:3]} {wi_hour:02d}:00 · {wi_year}</div>
              <div style="font-size:.82rem;color:{clr_wi};margin-top:4px">
                {"▲" if wi_d>=0 else "▼"} {abs(wi_d):.0f} trips ({wi_p:+.1f}%)
              </div>
            </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Page 3 — AI Insights
# ═════════════════════════════════════════════════════════════════════════════
def page_ai_insights():
    import plotly.graph_objects as go

    st.markdown('<div class="page-title">AI Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Model performance · Feature analysis · Clustering · Regression comparison</div>',
                unsafe_allow_html=True)

    tab_perf, tab_clust, tab_regr = st.tabs(
        ["⚙️  Model Performance", "🔵  Clustering", "📉  Regression"]
    )

    # ── Model Performance ─────────────────────────────────────────────────────
    with tab_perf:
        with st.spinner("Loading regression model …"):
            payload = load_regression_model()
        with st.spinner("Loading XGBoost model …"):
            _, xgb_met, xgb_fi, xgb_yte, xgb_ypred = load_xgb_model()

        mn   = payload["model_name"]
        m    = payload["metrics"]
        am   = payload.get("all_metrics", {})

        _section("Best Model Metrics")
        _kpi_row([
            ("🤖", mn,                 "Active Model",    "best by R²"),
            ("📉", f"{m['mae']:.2f}",  "MAE",             "avg trips error"),
            ("📊", f"{m['rmse']:.2f}", "RMSE",            "root mean sq. error"),
            ("📈", f"{m['r2']:.3f}",   "R² Score",        "variance explained"),
            ("🏋️",f"{payload['n_train']:,}", "Train rows", "all years"),
        ], top_idx=3)

        r2v = m["r2"]
        if r2v > 0.85:  badge = "🟢 Excellent — explains >85% of demand variance"
        elif r2v > 0.70: badge = "🟡 Good — solid predictive power"
        else:            badge = "🔴 Fair — more data would help"
        st.info(badge)

        # All-models comparison (visual cards, no heavy table)
        if am:
            _section("All Models Comparison")
            all_m_data = {**am}
            if xgb_met:
                all_m_data["XGBoost"] = xgb_met

            mcols = st.columns(len(all_m_data))
            for col, (mname, mmet) in zip(mcols, all_m_data.items()):
                is_best = mname == mn
                border  = "rgba(247,201,72,.35)" if is_best else "rgba(255,255,255,.06)"
                with col:
                    st.markdown(f"""
                    <div style="background:#1A1D27;border:1.5px solid {border};
                         border-radius:14px;padding:16px;text-align:center">
                      <div style="font-size:.82rem;font-weight:700;color:#FAFAFA;margin-bottom:10px">
                        {mname} {"✅" if is_best else ""}
                      </div>
                      <div style="font-size:1.2rem;font-weight:800;color:#F7C948">{mmet["r2"]:.3f}</div>
                      <div style="font-size:.68rem;color:#9CA3AF">R²</div>
                      <div style="margin-top:8px;font-size:.85rem;color:#9CA3AF">
                        MAE {mmet["mae"]:.2f} · RMSE {mmet["rmse"]:.2f}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            pc1, pc2 = st.columns(2)
            with pc1: _pchart(reg.chart_metrics_bar(am), h=300)
            with pc2: _pchart(reg.chart_r2_bar(am),      h=300)

        _section("Actual vs Predicted")
        dc1, dc2 = st.columns(2)
        with dc1:
            _pchart(reg.chart_actual_vs_pred(payload["y_test"], payload["y_pred"], mn), h=320)
        with dc2:
            if payload.get("feature_importance") is not None:
                _pchart(reg.chart_feature_importance(payload["feature_importance"], mn), h=320)
            else:
                st.info("Feature importance not available for Linear Regression.")

    # ── Clustering ────────────────────────────────────────────────────────────
    with tab_clust:
        feat_map = clust.available_features(demand)
        if not feat_map:
            st.error("No numeric features for clustering.")
        else:
            cc1, cc2, cc3 = st.columns([2.5, 1, 1])
            with cc1:
                sel_f = st.multiselect("Features (2–5)", list(feat_map.keys()),
                                       default=["trip_count","avg_fare","avg_distance"],
                                       format_func=lambda x: feat_map[x], key="cl_feats")
            with cc2:
                k         = st.slider("Clusters K", 2, 8, 3, key="cl_k")
            with cc3:
                normalize = st.checkbox("Normalize (StandardScaler)", True, key="cl_norm")

            if len(sel_f) < 2:
                st.warning("Select at least 2 features.")
            else:
                with st.spinner("Running KMeans …"):
                    labels_k, X_proc, inertia = clust.run_kmeans(demand, sel_f, k, normalize)

                use_pca = len(sel_f) > 2
                if use_pca:
                    X_2d, vr = clust.apply_pca(X_proc)
                    xl = f"PC1 ({vr[0]*100:.1f}% var)"
                    yl = f"PC2 ({vr[1]*100:.1f}% var)"
                    title = f"KMeans K={k} — PCA 2D"
                else:
                    X_2d = X_proc
                    xl, yl = feat_map[sel_f[0]], feat_map[sel_f[1]]
                    title = f"KMeans K={k}"

                kc1, kc2 = st.columns([2, 1])
                with kc1:
                    _pchart(clust.chart_scatter(X_2d, labels_k, xl, yl, title), h=380)
                with kc2:
                    _section("Elbow Method")
                    with st.spinner("Computing elbow …"):
                        ks_n, in_n = clust.compute_elbow(demand, sel_f, normalize=True)
                        ks_r, in_r = clust.compute_elbow(demand, sel_f, normalize=False)
                    _pchart(clust.chart_elbow(ks_n, in_n, in_r), h=280)

                st.markdown(
                    f'<div class="banner">K={k} · Inertia: {inertia:,.0f} · '
                    f'{"Normalized ✅" if normalize else "Raw ⚠️"}</div>',
                    unsafe_allow_html=True)

                _section("Cluster Statistics")
                ds = demand[sel_f].dropna().copy()
                ds["Cluster"] = labels_k
                st.dataframe(
                    ds.groupby("Cluster")[sel_f].mean().round(2)
                    .rename(index=lambda i: f"Cluster {i}")
                    .rename(columns=feat_map),
                    use_container_width=True,
                )

    # ── Regression ────────────────────────────────────────────────────────────
    with tab_regr:
        _section("Feature Selection")
        rc1, rc2 = st.columns([3,1])
        with rc1:
            sel_reg = st.multiselect("Features", list(reg.REGRESSION_FEATURES.keys()),
                                     default=list(reg.REGRESSION_FEATURES.keys()),
                                     format_func=lambda x: reg.REGRESSION_FEATURES[x],
                                     key="rg_feats")
        with rc2:
            st.markdown('<div style="margin-top:2rem;color:#9CA3AF;font-size:.8rem">'
                        '<b>Target:</b> trip_count</div>', unsafe_allow_html=True)

        if not sel_reg:
            st.warning("Select at least one feature.")
        else:
            with st.spinner("Training models …"):
                out = reg.get_regression_results(tuple(sorted(sel_reg)))
            results = out["results"]
            y_te    = out["y_te"]
            best_n  = max(results, key=lambda x: results[x]["r2"])
            best_r  = results[best_n]

            _kpi_row([
                ("🏆", best_n,                     "Best Model",   "by R²"),
                ("📉", f"{best_r['mae']:.2f}",      "Best MAE",     "trips"),
                ("📈", f"{best_r['r2']:.3f}",       "Best R²",      ""),
                ("🏋️",f"{out['n_train']:,}",        "Train rows",   ""),
            ], top_idx=2)

            rc3, rc4 = st.columns(2)
            with rc3: _pchart(reg.chart_metrics_bar(results), h=310)
            with rc4: _pchart(reg.chart_r2_bar(results),      h=310)

            _section("Actual vs Predicted")
            tabs_m = st.tabs([f"📊 {mn}" for mn in results])
            for tab, (mname, res) in zip(tabs_m, results.items()):
                with tab:
                    _pchart(reg.chart_actual_vs_pred(y_te, res["y_pred"], mname), h=320)

            fi_list = [(mn, r) for mn, r in results.items() if "feature_importance" in r]
            if fi_list:
                _section("Feature Importance")
                fi_cols = st.columns(len(fi_list))
                for col, (mname, res) in zip(fi_cols, fi_list):
                    with col:
                        _pchart(reg.chart_feature_importance(res["feature_importance"], mname), h=310)


# ═════════════════════════════════════════════════════════════════════════════
# Page 4 — Forecasting
# ═════════════════════════════════════════════════════════════════════════════
def page_forecasting():
    import plotly.graph_objects as go

    st.markdown('<div class="page-title">Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Next 24h · Weekly · Monthly forecasts · Animated demand map · Future year predictions</div>',
                unsafe_allow_html=True)

    with st.spinner("Loading model …"):
        payload = load_regression_model()

    # Parameters
    _section("Forecast Parameters")
    fp1, fp2, fp3, fp4 = st.columns(4)
    with fp1:
        fc_lbl = st.selectbox("Zone", _zone_label_list(), key="fc_zone")
        fc_id  = int(fc_lbl.split("— ID ")[-1])
    with fp2:
        fc_dow_lbl = st.selectbox("Day of Week", _DOW, index=_now_dow, key="fc_dow")
        fc_dow     = _DOW.index(fc_dow_lbl)
    with fp3:
        fc_mon_lbl = st.selectbox("Month", _MON, index=_now_mon - 1, key="fc_month")
        fc_month   = _MON.index(fc_mon_lbl) + 1
    with fp4:
        fc_year = st.selectbox("Year", _YEAR_LIST, index=_year_idx, key="fc_year")

    fd = _zone_defaults(fc_id)

    def _fp(hour, dow, month, year):
        return predict_regression(payload, {
            "pickup_location_id":    float(fc_id),
            "pickup_hour":           float(hour),
            "pickup_day_of_week":    float(dow),
            "pickup_month":          float(month),
            "historical_trip_count": fd["hist"],
            "avg_fare_amount":       fd["fare"],
            "avg_trip_distance":     fd["dist"],
            "avg_trip_duration":     fd["dur"],
            "year":                  float(year),
        })

    _DRK = dict(template="plotly_dark", paper_bgcolor="#1A1D27",
                plot_bgcolor="#1A1D27", font=dict(color="#FAFAFA"), height=360)

    tab_h, tab_d, tab_m, tab_map = st.tabs(
        ["⏰  24-Hour", "📅  Day-of-Week", "🗓️  Monthly", "🗺️  Animated Map"]
    )

    with tab_h:
        hrs  = list(range(24))
        preds_h = [_fp(h, fc_dow, fc_month, fc_year) for h in hrs]
        pk_h    = hrs[int(np.argmax(preds_h))]
        fig_h   = go.Figure()
        fig_h.add_trace(go.Bar(
            x=hrs, y=preds_h,
            marker_color=["#EF4444" if h==pk_h else "#F7C948" for h in hrs],
            text=[f"{p:.0f}" if h==pk_h else "" for h,p in zip(hrs,preds_h)],
            textposition="outside",
        ))
        fig_h.add_trace(go.Scatter(x=hrs, y=preds_h, mode="lines",
            line=dict(color="rgba(247,201,72,.35)", width=2, dash="dot"), showlegend=False))
        fig_h.update_layout(title=f"24h — {fc_lbl.split(' (')[0]} · {fc_dow_lbl} · {fc_mon_lbl} {fc_year}",
                            xaxis_title="Hour", yaxis_title="Trips/hr",
                            xaxis=dict(tickmode="linear",tick0=0,dtick=2), **_DRK)
        _pchart(fig_h)
        _kpi_row([
            ("⚡", f"{max(preds_h):.0f}",       f"Peak ({pk_h}:00)",     "trips/hr"),
            ("📉", f"{min(preds_h):.0f}",       "Lowest",               "trips/hr"),
            ("📊", f"{np.mean(preds_h):.0f}",   "Daily Average",        "trips/hr"),
            ("💵", f"${max(preds_h)*fd['fare']*0.7:.2f}", "Peak Revenue est", "per hour"),
        ])

    with tab_d:
        preds_d = [_fp(18, d, fc_month, fc_year) for d in range(7)]
        pk_d    = int(np.argmax(preds_d))
        fig_d   = go.Figure(go.Bar(
            x=_DOW, y=preds_d,
            marker_color=["#EF4444" if i==pk_d else "#3B82F6" for i in range(7)],
            text=[f"{p:.0f}" for p in preds_d], textposition="outside",
        ))
        fig_d.update_layout(title=f"Day-of-Week — {fc_lbl.split(' (')[0]} · 18:00 · {fc_mon_lbl} {fc_year}",
                            xaxis_title="Day", yaxis_title="Trips/hr", **_DRK)
        _pchart(fig_d)
        _kpi_row([
            ("🏆", _DOW[pk_d],              "Busiest Day",  "at 18:00"),
            ("⚡", f"{max(preds_d):.0f}",   "Peak Demand",  "trips/hr"),
            ("📊", f"{np.mean(preds_d):.0f}","Weekly Avg",  "trips/hr"),
            ("📉", f"{min(preds_d):.0f}",   "Quietest",     "trips/hr"),
        ])

    with tab_m:
        preds_m = [_fp(18, fc_dow, mn, fc_year) for mn in range(1,13)]
        pk_m    = int(np.argmax(preds_m))
        fig_m   = go.Figure()
        fig_m.add_trace(go.Scatter(
            x=_MONS, y=preds_m, mode="lines+markers",
            line=dict(color="#3B82F6", width=2.5),
            marker=dict(color=["#EF4444" if i==pk_m else "#3B82F6" for i in range(12)],
                        size=[13 if i==pk_m else 7 for i in range(12)]),
            fill="tozeroy", fillcolor="rgba(59,130,246,.07)",
        ))
        fig_m.update_layout(title=f"Monthly — {fc_lbl.split(' (')[0]} · {fc_dow_lbl} · 18:00 · {fc_year}",
                            xaxis_title="Month", yaxis_title="Trips/hr", **_DRK)
        _pchart(fig_m)
        _kpi_row([
            ("🏆", _MONS[pk_m],              "Busiest Month", "at 18:00"),
            ("⚡", f"{max(preds_m):.0f}",    "Peak Demand",   "trips/hr"),
            ("📊", f"{np.mean(preds_m):.0f}","Annual Avg",    "trips/hr"),
            ("📉", f"{min(preds_m):.0f}",    "Quietest",      "trips/hr"),
        ])

    with tab_map:
        st.markdown('<div class="sec">🌡️ Animated Demand Map — Drag the slider to travel through the day</div>',
                    unsafe_allow_html=True)
        anim_dow_lbl = st.selectbox("Day", _DOW, index=_now_dow, key="am_dow")
        anim_dow     = _DOW.index(anim_dow_lbl)
        anim_mon_lbl = st.selectbox("Month", _MON, index=_now_mon - 1, key="am_mon")
        anim_mon     = _MON.index(anim_mon_lbl) + 1
        anim_year    = st.selectbox("Year", _YEAR_LIST, index=_year_idx, key="am_year")

        anim_hour = st.slider("Hour of Day", 0, 23, _now_hour, key="am_hour",
                              help="Move the slider to watch demand shift across NYC")

        with st.spinner(f"Computing demand map for {anim_hour:02d}:00 …"):
            zp_anim = _zone_preds(anim_hour, anim_dow, anim_mon)

        if not zp_anim.empty:
            fig_anim = _build_map(zp_anim, height=500)
            fig_anim.update_layout(
                title=dict(text=f"NYC Demand — {anim_dow_lbl} {anim_hour:02d}:00 · {anim_mon_lbl} {anim_year}",
                           font=dict(color="#FAFAFA"), x=0.01, y=0.97),
            )
            st.plotly_chart(fig_anim, use_container_width=True,
                            config={"displayModeBar": True}, key="anim_map")

            am_top = zp_anim.nlargest(5, "predicted_demand")
            _section("Top 5 Zones at This Hour")
            cards_html = ""
            clss = ["r1","r2","r3","",""]
            emjs = ["🥇","🥈","🥉","4.","5."]
            for i,(_, row) in enumerate(am_top.iterrows()):
                zn   = str(row.get("Zone",""))
                bo   = str(row.get("Borough",""))
                pd_  = float(row["predicted_demand"])
                af   = float(row.get("avg_fare",0))
                rv   = float(row.get("Revenue est ($/hr)", pd_*af*0.7))
                dl   = str(row.get("Demand Level",""))
                cards_html += f"""
                <div class="zone-card {clss[i]}">
                  <div class="zone-name">{emjs[i]} {zn}
                    <span class="zone-badge">{bo}</span>
                  </div>
                  <div class="zone-stats">
                    🔮 {pd_:.0f} trips/hr &nbsp;·&nbsp;
                    💰 ${af:.2f} avg fare &nbsp;·&nbsp;
                    💵 ${rv:.2f}/hr est. rev &nbsp;·&nbsp;
                    {dl}
                  </div>
                </div>"""
            st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div class="banner">💡 Forecasts use the regression model (2023–2026 training data). '
        'Future year predictions (2027–2035) extrapolate based on learned temporal patterns. '
        'Revenue = Predicted Trips × Avg Fare × 70% driver share.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────
_ROUTES = {
    "dashboard":   page_dashboard,
    "ai_pred":     page_ai_prediction,
    "ai_insights": page_ai_insights,
    "forecasting": page_forecasting,
}
_ROUTES[page_key]()
