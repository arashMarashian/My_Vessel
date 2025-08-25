from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import math, datetime as dt
import requests


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    # initial course from point1 to point2
    to_rad = math.radians
    to_deg = math.degrees
    dlon = to_rad(lon2 - lon1)
    lat1, lat2 = to_rad(lat1), to_rad(lat2)
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    brng = (to_deg(math.atan2(y, x)) + 360.0) % 360.0
    return brng

def angle_diff_deg(a, b) -> float:
    d = (a - b + 540.0) % 360.0 - 180.0
    return d

def great_circle_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    to_rad = math.radians
    dlat = to_rad(lat2 - lat1)
    dlon = to_rad(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(to_rad(lat1)) * math.cos(to_rad(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def estimate_times(path_ll: List[Tuple[float, float]], target_speed_kn: float, depart: dt.datetime) -> List[dt.datetime]:
    """ETA at each waypoint assuming target speed between points (simple, used to time-align env)."""
    times = [depart]
    if len(path_ll) < 2:
        return times
    for i in range(1, len(path_ll)):
        lat1, lon1 = path_ll[i - 1]
        lat2, lon2 = path_ll[i]
        nm = great_circle_km(lat1, lon1, lat2, lon2) * 0.539957
        dt_h = nm / max(1e-6, target_speed_kn)
        times.append(times[-1] + dt.timedelta(hours=dt_h))
    return times

def openmeteo_marine(lat: float, lon: float, when: dt.datetime) -> Dict[str, float]:
    """
    Fetch hourly marine variables from Open-Meteo for the given point/time.
    Returns dict with wind_speed (m/s), wind_dir (deg), wave_height (m).
    """
    base = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wave_height,wind_speed_10m,wind_direction_10m",
        "start_date": when.date().isoformat(),
        "end_date": when.date().isoformat(),
        "timezone": "UTC",
    }
    r = requests.get(base, params=params, timeout=15)
    r.raise_for_status()
    js = r.json()
    times = js["hourly"]["time"]
    ws = js["hourly"]["wind_speed_10m"]
    wd = js["hourly"]["wind_direction_10m"]
    wh = js["hourly"]["wave_height"]
    key = when.replace(minute=0, second=0, microsecond=0).isoformat(timespec="hours")
    if key in times:
        i = times.index(key)
    else:
        diffs = []
        for t in times:
            hh = dt.datetime.fromisoformat(t)
            diffs.append(abs((hh - when).total_seconds()))
        i = diffs.index(min(diffs))
    return {"wind_speed": float(ws[i]), "wind_dir": float(wd[i]), "wave_height": float(wh[i])}

def sample_env_along_route(
    path_ll: List[Tuple[float, float]],
    depart_iso: str,
    target_speed_kn: float,
) -> List[Dict[str, float]]:
    """
    Return list of per-waypoint dicts: wind_speed (m/s), wind_dir (deg), wind_angle_diff (deg), wave_height (m)
    """
    if not path_ll:
        return []
    depart = dt.datetime.fromisoformat(depart_iso.replace("Z", "+00:00")).astimezone(dt.timezone.utc).replace(tzinfo=None)
    times = estimate_times(path_ll, target_speed_kn, depart)
    out = []
    for i, (latlon, t) in enumerate(zip(path_ll, times)):
        lat, lon = latlon
        met = openmeteo_marine(lat, lon, t)
        if i < len(path_ll) - 1:
            lat2, lon2 = path_ll[i + 1]
        else:
            lat2, lon2 = path_ll[i - 1]
        course = bearing_deg(lat, lon, lat2, lon2)
        wind_angle_diff = angle_diff_deg(course, met["wind_dir"])
        out.append(
            {
                "wind_speed": met["wind_speed"],
                "wind_dir": met["wind_dir"],
                "wind_angle_diff": wind_angle_diff,
                "wave_height": met["wave_height"],
            }
        )
    return out
