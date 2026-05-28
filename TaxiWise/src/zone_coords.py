"""
Approximate centroid coordinates for NYC taxi zones.
Uses a golden-ratio (Fibonacci) spread within each borough's bounding box.
Visually representative on a map — not geodetically exact.
"""
from __future__ import annotations

_PHI = (1 + 5 ** 0.5) / 2  # golden ratio ≈ 1.618034

_BOROUGH_BOXES: dict[str, dict[str, tuple[float, float]]] = {
    "Manhattan":     {"lat": (40.700, 40.882), "lon": (-74.020, -73.910)},
    "Brooklyn":      {"lat": (40.570, 40.739), "lon": (-74.042, -73.833)},
    "Queens":        {"lat": (40.542, 40.800), "lon": (-73.962, -73.700)},
    "Bronx":         {"lat": (40.785, 40.915), "lon": (-73.933, -73.755)},
    "Staten Island": {"lat": (40.477, 40.651), "lon": (-74.259, -74.034)},
    "EWR":           {"lat": (40.686, 40.706), "lon": (-74.200, -74.168)},
}
_DEFAULT_BOX: dict[str, tuple[float, float]] = {
    "lat": (40.650, 40.850),
    "lon": (-74.050, -73.850),
}


def get_zone_coord(location_id: int, borough: str) -> tuple[float, float]:
    """Return (lat, lon) for a zone using golden-ratio spread within borough bounds."""
    box = _BOROUGH_BOXES.get(borough, _DEFAULT_BOX)
    lat_min, lat_max = box["lat"]
    lon_min, lon_max = box["lon"]
    t = (location_id * _PHI) % 1.0
    r = (location_id * (_PHI ** 2)) % 1.0
    return (
        lat_min + t * (lat_max - lat_min),
        lon_min + r * (lon_max - lon_min),
    )
