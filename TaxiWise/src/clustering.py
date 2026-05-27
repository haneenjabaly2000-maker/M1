"""
TaxiWise — Clustering & PCA
KMeans with optional StandardScaler, Elbow Method, and PCA visualization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#1A1D27",
    plot_bgcolor="#1A1D27",
    font=dict(color="#FAFAFA"),
)
_COLORS = [
    "#F7C948", "#3B82F6", "#10B981", "#EF4444",
    "#A855F7", "#F97316", "#06B6D4", "#84CC16",
]

CLUSTER_FEATURES = {
    "trip_count":       "Trip Count",
    "avg_fare":         "Avg Fare ($)",
    "avg_distance":     "Avg Distance (mi)",
    "avg_duration":     "Avg Duration (min)",
    "zone_total_trips": "Zone Total Trips",
    "hour":             "Hour of Day",
    "dow":              "Day of Week",
    "month":            "Month",
    "avg_tip":          "Avg Tip ($)",
}


def available_features(demand: pd.DataFrame) -> dict[str, str]:
    """Return feature key→label pairs that exist in demand."""
    return {k: v for k, v in CLUSTER_FEATURES.items() if k in demand.columns}


def run_kmeans(
    demand: pd.DataFrame,
    features: list[str],
    k: int,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fit KMeans. Returns (labels, X_processed, inertia).
    X_processed is the scaled (or raw) matrix used for plotting.
    """
    X = demand[features].dropna().values.astype(float)
    if normalize:
        X = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X)
    return labels, X, float(km.inertia_)


def compute_elbow(
    demand: pd.DataFrame,
    features: list[str],
    normalize: bool,
    k_max: int = 10,
) -> tuple[list[int], list[float]]:
    """Return (k_values, inertias) for K=1..k_max."""
    X = demand[features].dropna().values.astype(float)
    if normalize:
        X = StandardScaler().fit_transform(X)
    ks, inertias = [], []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        ks.append(k)
        inertias.append(float(km.inertia_))
    return ks, inertias


def apply_pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project X to 2 PCA components. Returns (X_2d, explained_variance_ratio)."""
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    return X_2d, pca.explained_variance_ratio_


# ── Charts ────────────────────────────────────────────────────────────────────

def chart_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
) -> go.Figure:
    df_p = pd.DataFrame({
        x_label: X[:, 0],
        y_label: X[:, 1],
        "Cluster": labels.astype(str),
    })
    fig = px.scatter(
        df_p, x=x_label, y=y_label, color="Cluster",
        title=title, color_discrete_sequence=_COLORS,
    )
    fig.update_layout(**_DARK, legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


def chart_elbow(
    ks: list[int],
    inertias_norm: list[float],
    inertias_raw: list[float] | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=inertias_norm, mode="lines+markers",
        name="With StandardScaler",
        line=dict(color="#F7C948", width=2),
        marker=dict(size=8),
    ))
    if inertias_raw:
        fig.add_trace(go.Scatter(
            x=ks, y=inertias_raw, mode="lines+markers",
            name="Without Scaling (Raw)",
            line=dict(color="#3B82F6", width=2),
            marker=dict(size=8),
        ))
    fig.update_layout(
        title="Elbow Method — Inertia (WCSS) vs K",
        xaxis_title="K (Number of Clusters)",
        yaxis_title="Inertia (WCSS)",
        xaxis=dict(tickmode="linear", dtick=1),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        **_DARK,
    )
    return fig
