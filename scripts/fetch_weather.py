#!/usr/bin/env python3
"""Fetch archived Open-Meteo forecasts along the 2025 BWSC route and interpolate to ZP rows."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
API = "https://historical-forecast-api.open-meteo.com/v1/forecast"
TZ = "Australia/Darwin"

HOURLY = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "temperature_80m", "temperature_120m", "temperature_180m", "pressure_msl", "surface_pressure",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
    "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m", "wind_gusts_10m",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance", "global_tilted_irradiance", "terrestrial_radiation",
    "shortwave_radiation_instant", "direct_radiation_instant", "diffuse_radiation_instant", "direct_normal_irradiance_instant", "global_tilted_irradiance_instant", "terrestrial_radiation_instant",
    "evapotranspiration", "et0_fao_evapotranspiration", "precipitation", "precipitation_probability", "rain", "showers", "snowfall", "snow_depth", "weather_code",
    "freezing_level_height", "visibility", "soil_temperature_0cm", "soil_temperature_6cm", "soil_temperature_18cm", "soil_temperature_54cm",
    "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm",
    "uv_index", "uv_index_clear_sky", "is_day", "sunshine_duration", "wet_bulb_temperature_2m", "vapour_pressure_deficit", "cape", "lifted_index",
    "convective_inhibition", "boundary_layer_height",
]
LEVELS = (1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30)
for prefix in ("temperature", "relative_humidity", "dew_point", "cloud_cover", "wind_speed", "wind_direction", "geopotential_height"):
    HOURLY.extend(f"{prefix}_{level}hPa" for level in LEVELS)

DAILY = [
    "weather_code", "temperature_2m_max", "temperature_2m_mean", "temperature_2m_min",
    "apparent_temperature_max", "apparent_temperature_mean", "apparent_temperature_min", "uv_index_max", "uv_index_clear_sky_max",
    "sunrise", "sunset", "daylight_duration", "sunshine_duration", "moonrise", "moonset", "moon_phase",
    "rain_sum", "showers_sum", "snowfall_sum", "precipitation_sum", "precipitation_hours", "precipitation_probability_max",
    "precipitation_probability_mean", "precipitation_probability_min", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration", "cape_mean", "cape_max", "cape_min", "cloud_cover_mean", "cloud_cover_max", "cloud_cover_min",
    "dew_point_2m_mean", "dew_point_2m_max", "dew_point_2m_min", "et0_fao_evapotranspiration_sum", "growing_degree_days_base_0_limit_50",
    "leaf_wetness_probability_mean", "relative_humidity_2m_mean", "relative_humidity_2m_max", "relative_humidity_2m_min", "snowfall_water_equivalent_sum",
    "pressure_msl_mean", "pressure_msl_max", "pressure_msl_min", "surface_pressure_mean", "surface_pressure_max", "surface_pressure_min", "updraft_max",
    "visibility_mean", "visibility_min", "visibility_max", "wind_gusts_10m_mean", "wind_speed_10m_mean", "wind_gusts_10m_min", "wind_speed_10m_min",
    "wet_bulb_temperature_2m_mean", "wet_bulb_temperature_2m_max", "wet_bulb_temperature_2m_min", "vapour_pressure_deficit_max",
]

MINUTELY_15 = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation", "rain", "snowfall",
    "snowfall_height", "freezing_level_height", "sunshine_duration", "weather_code", "wind_speed_10m", "wind_speed_80m",
    "wind_direction_10m", "wind_direction_80m", "wind_gusts_10m", "visibility", "cape", "lightning_potential", "is_day",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance", "global_tilted_irradiance",
    "terrestrial_radiation", "shortwave_radiation_instant", "direct_radiation_instant", "diffuse_radiation_instant",
    "direct_normal_irradiance_instant", "global_tilted_irradiance_instant", "terrestrial_radiation_instant",
]


def fetch(params: dict[str, str], excluded: set[str]) -> dict:
    params = params.copy()
    rate_limit_retries = 0
    while True:
        allowed = {"hourly": HOURLY, "daily": DAILY, "minutely_15": MINUTELY_15}
        for group, variables in allowed.items():
            if group in params:
                params[group] = ",".join(x for x in params[group].split(",") if x in variables and x not in excluded)
        url = API + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "YATA-MPC/0.1 (research data preparation)"})
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                payload = response.read()
            data = json.loads(payload)
            return {"data": data, "url": url, "bytes": len(payload)}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                rate_limit_retries += 1
                if rate_limit_retries > 5:
                    raise RuntimeError(f"Open-Meteo remained rate-limited after {rate_limit_retries - 1} retries: {detail}") from exc
                wait_s = max(65, int(exc.headers.get("Retry-After", "0") or 0))
                print(f"Open-Meteo rate limit reached; waiting {wait_s}s before retry {rate_limit_retries}/5.", file=sys.stderr, flush=True)
                time.sleep(wait_s)
                continue
            # API errors name an invalid variable. Remove once globally and repeat so the final selected set is explicit.
            match = re.search(r"(?:invalid String value|Invalid value for parameter)[^A-Za-z0-9_]*([A-Za-z][A-Za-z0-9_]*)", detail, re.I)
            if exc.code == 400 and match:
                variable = match.group(1)
                if variable in excluded or variable not in HOURLY + DAILY + MINUTELY_15:
                    raise RuntimeError(f"Open-Meteo rejected request: {detail}") from exc
                excluded.add(variable)
                print(f"API does not accept {variable}; excluding it from the complete route query.", file=sys.stderr, flush=True)
                continue
            raise RuntimeError(f"Open-Meteo HTTP {exc.code}: {detail}") from exc


def save_gz_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, separators=(",", ":"))


def interp_arrays(values: np.ndarray, nl: np.ndarray, nr: np.ndarray, tl: np.ndarray, tr: np.ndarray,
                  sw: np.ndarray, tw: np.ndarray, name: str, categorical: bool) -> np.ndarray:
    if categorical:
        nidx = np.where(sw < 0.5, nl, nr)
        tidx = np.where(tw < 0.5, tl, tr)
        return values[nidx, tidx]
    if "wind_direction" in name:
        rad = np.deg2rad(values % 360.0)
        vals = (rad[nl, tl], rad[nr, tl], rad[nl, tr], rad[nr, tr])
        weights = ((1 - sw) * (1 - tw), sw * (1 - tw), (1 - sw) * tw, sw * tw)
        x = np.zeros(len(nl)); y = np.zeros(len(nl)); total = np.zeros(len(nl))
        for v, w in zip(vals, weights):
            good = np.isfinite(v)
            x[good] += np.cos(v[good]) * w[good]
            y[good] += np.sin(v[good]) * w[good]
            total[good] += w[good]
        return np.where(total > 0, (np.rad2deg(np.arctan2(y, x)) + 360) % 360, np.nan)
    vals = (values[nl, tl], values[nr, tl], values[nl, tr], values[nr, tr])
    weights = ((1 - sw) * (1 - tw), sw * (1 - tw), (1 - sw) * tw, sw * tw)
    out = np.zeros(len(nl)); total = np.zeros(len(nl))
    for v, w in zip(vals, weights):
        good = np.isfinite(v)
        out[good] += v[good] * w[good]
        total[good] += w[good]
    return np.divide(out, total, out=np.full(len(nl), np.nan), where=total > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing-km", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--refresh", action="store_true", help="re-fetch even when cached route points exist")
    parser.add_argument("--probe-only", action="store_true", help="fetch/cache the first route batch and report API coverage")
    args = parser.parse_args()
    if args.spacing_km <= 0 or args.batch_size < 1:
        parser.error("spacing-km must be positive and batch-size at least 1")
    geo = pd.read_csv(ROOT / "data/processed/race_days_distance_geo.csv", low_memory=False)
    route = pd.read_csv(ROOT / "data/input/route/route_100m_dem.csv")
    max_dist = float(geo.distance_from_darwin_m.max())
    nodes = np.unique(np.r_[np.arange(0, math.floor(max_dist / (args.spacing_km * 1000)) * args.spacing_km * 1000 + 1, args.spacing_km * 1000), max_dist])
    # Ensure the last whole-spacing node plus the endpoint both occur.
    node_lat = np.interp(nodes, route.distance_m, route.lat)
    node_lon = np.interp(nodes, route.distance_m, route.lon)
    request_points = [{"point_id": int(i), "distance_m": float(d), "latitude": float(lat), "longitude": float(lon)} for i, (d, lat, lon) in enumerate(zip(nodes, node_lat, node_lon))]
    raw_dir = ROOT / "data/weather_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    excluded: set[str] = {"precipitable_water"}  # Historical Forecast currently rejects this variable.
    hourly_records, minutely_records, daily_records, response_meta = [], [], [], []
    # Earlier 12-location batches sampled at 10 km; map those cached locations by route distance.
    def distance_cache(distance_m: float) -> Path:
        key = int(round(distance_m))
        return raw_dir / f"historical_forecast_route_distance_{key:08d}m.json.gz"

    if not args.refresh:
        for legacy_path in sorted(raw_dir.glob("historical_forecast_route_batch_*.json.gz")):
            match = re.search(r"batch_(\d+)", legacy_path.name)
            if not match:
                continue
            legacy_no = int(match.group(1))
            with gzip.open(legacy_path, "rt", encoding="utf-8") as f:
                legacy_result = json.load(f)
            locs = legacy_result.get("data", [])
            locs = locs if isinstance(locs, list) else [locs]
            for j, loc in enumerate(locs):
                point_id = (legacy_no - 1) * 12 + j
                legacy_distance_m = point_id * 10000.0
                cache_path = distance_cache(legacy_distance_m)
                if point_id < 284 and not cache_path.exists():
                    save_gz_json(cache_path, {"data": loc, "cached_from": legacy_path.name, "legacy_spacing_km": 10})
        # Migrate any point caches left by an interrupted run before the distance-based cache was introduced.
        for legacy_path in sorted(raw_dir.glob("historical_forecast_route_point_*.json.gz")):
            match = re.search(r"point_(\d+)", legacy_path.name)
            if not match:
                continue
            point_id = int(match.group(1))
            cache_path = distance_cache(point_id * 10000.0)
            if point_id < 180 and not cache_path.exists():
                with gzip.open(legacy_path, "rt", encoding="utf-8") as f:
                    save_gz_json(cache_path, json.load(f))
    point_caches = {p["point_id"]: distance_cache(p["distance_m"]) for p in request_points}
    missing_points = [p for p in request_points if args.refresh or not point_caches[p["point_id"]].exists()]
    query_count = math.ceil(len(missing_points) / args.batch_size)
    for batch_no, offset in enumerate(range(0, len(missing_points), args.batch_size), start=1):
        batch = missing_points[offset:offset + args.batch_size]
        result_path = raw_dir / f"historical_forecast_route_supplement_{batch_no:03d}_p{batch[0]['point_id']:04d}_{batch[-1]['point_id']:04d}.json.gz"
        if result_path.exists() and not args.refresh:
            with gzip.open(result_path, "rt", encoding="utf-8") as f:
                result = json.load(f)
        else:
            params = {
                "latitude": ",".join(f"{p['latitude']:.6f}" for p in batch),
                "longitude": ",".join(f"{p['longitude']:.6f}" for p in batch),
                "hourly": ",".join(HOURLY), "daily": ",".join(DAILY),
                "start_date": "2025-08-24", "end_date": "2025-08-29", "timezone": TZ,
                "temperature_unit": "celsius", "wind_speed_unit": "ms", "precipitation_unit": "mm",
                "tilt": "0", "azimuth": "0",
            }
            result = fetch(params, excluded)
            save_gz_json(result_path, result)
        data = result["data"]
        locations = data if isinstance(data, list) else [data]
        if len(locations) != len(batch):
            raise ValueError(f"API returned {len(locations)} locations for {len(batch)} route points, batch {batch_no}")
        for point, loc in zip(batch, locations):
            save_gz_json(point_caches[point["point_id"]], {"data": loc, "cached_from": result_path.name})
        print(f"Historical Forecast route batch {batch_no}/{query_count} ({min(offset + len(batch), len(missing_points))}/{len(missing_points)} missing points fetched; {args.spacing_km:g} km route spacing)", flush=True)
        if args.probe_only:
            print(json.dumps({"probe_points": len(batch), "hourly_variables_returned": list(locations[0].get("hourly", {}).keys()),
                              "daily_variables_returned": list(locations[0].get("daily", {}).keys()),
                              "excluded": sorted(excluded), "raw_cache": str(result_path)}, indent=2))
            return
        time.sleep(6)

    # Open-Meteo also exposes a 15-minute series. In Australia it is documented as hourly values
    # interpolated to 15-minute timestamps, but keep it as a separate source series for completeness.
    def minutely_cache(point_id: int) -> Path:
        return raw_dir / f"historical_forecast_route_minutely_15_point_{point_id:04d}.json.gz"

    missing_minutely = [p for p in request_points if args.refresh or not minutely_cache(p["point_id"]).exists()]
    minutely_batch_count = math.ceil(len(missing_minutely) / args.batch_size)
    for batch_no, offset in enumerate(range(0, len(missing_minutely), args.batch_size), start=1):
        batch = missing_minutely[offset:offset + args.batch_size]
        result_path = raw_dir / f"historical_forecast_route_minutely_15_supplement_{batch_no:03d}_p{batch[0]['point_id']:04d}_{batch[-1]['point_id']:04d}.json.gz"
        if result_path.exists() and not args.refresh:
            with gzip.open(result_path, "rt", encoding="utf-8") as f:
                result = json.load(f)
        else:
            params = {
                "latitude": ",".join(f"{p['latitude']:.6f}" for p in batch),
                "longitude": ",".join(f"{p['longitude']:.6f}" for p in batch),
                "minutely_15": ",".join(MINUTELY_15),
                "start_date": "2025-08-24", "end_date": "2025-08-29", "timezone": TZ,
                "temperature_unit": "celsius", "wind_speed_unit": "ms", "precipitation_unit": "mm",
                "tilt": "0", "azimuth": "0",
            }
            result = fetch(params, excluded)
            save_gz_json(result_path, result)
        data = result["data"]
        locations = data if isinstance(data, list) else [data]
        if len(locations) != len(batch):
            raise ValueError(f"API returned {len(locations)} locations for {len(batch)} 15-minute route points, batch {batch_no}")
        for point, loc in zip(batch, locations):
            if "minutely_15" not in loc or "time" not in loc["minutely_15"]:
                raise ValueError(f"No 15-minute data for route point {point['point_id']}: {loc.keys()}")
            save_gz_json(minutely_cache(point["point_id"]), {"data": loc, "cached_from": result_path.name})
        print(f"Historical Forecast 15-minute route batch {batch_no}/{minutely_batch_count} ({min(offset + len(batch), len(missing_minutely))}/{len(missing_minutely)} missing points fetched)", flush=True)
        time.sleep(6)

    # Assemble dense hourly, 15-minute, and daily route tables in route order from per-point caches.
    for point in request_points:
        with gzip.open(point_caches[point["point_id"]], "rt", encoding="utf-8") as f:
            loc = json.load(f)["data"]
        if "hourly" not in loc or "time" not in loc["hourly"]:
            raise ValueError(f"No hourly forecast for route point {point['point_id']}: {loc.keys()}")
        hourly = loc["hourly"]
        for j, iso in enumerate(hourly["time"]):
            rec = {"point_id": point["point_id"], "route_distance_m": point["distance_m"], "request_latitude": point["latitude"],
                   "request_longitude": point["longitude"], "resolved_latitude": loc.get("latitude"), "resolved_longitude": loc.get("longitude"),
                   "resolved_elevation_m": loc.get("elevation"), "hour_local": iso}
            for name, arr in hourly.items():
                if name != "time":
                    rec[name] = arr[j] if j < len(arr) else None
            hourly_records.append(rec)
        daily = loc.get("daily", {})
        for j, iso in enumerate(daily.get("time", [])):
            rec = {"point_id": point["point_id"], "route_distance_m": point["distance_m"], "request_latitude": point["latitude"],
                   "request_longitude": point["longitude"], "resolved_latitude": loc.get("latitude"), "resolved_longitude": loc.get("longitude"),
                   "resolved_elevation_m": loc.get("elevation"), "date_local": iso}
            for name, arr in daily.items():
                if name != "time":
                    rec[name] = arr[j] if j < len(arr) else None
            daily_records.append(rec)
        with gzip.open(minutely_cache(point["point_id"]), "rt", encoding="utf-8") as f:
            minute_loc = json.load(f)["data"]
        minutely = minute_loc["minutely_15"]
        for j, iso in enumerate(minutely["time"]):
            rec = {"point_id": point["point_id"], "route_distance_m": point["distance_m"], "request_latitude": point["latitude"],
                   "request_longitude": point["longitude"], "resolved_latitude": minute_loc.get("latitude"),
                   "resolved_longitude": minute_loc.get("longitude"), "resolved_elevation_m": minute_loc.get("elevation"),
                   "time_local": iso}
            for name, arr in minutely.items():
                if name != "time":
                    rec[name] = arr[j] if j < len(arr) else None
            minutely_records.append(rec)
        response_meta.append({"point_id": point["point_id"], "distance_m": point["distance_m"], "request_latitude": point["latitude"],
                              "request_longitude": point["longitude"], "resolved_latitude": loc.get("latitude"), "resolved_longitude": loc.get("longitude"),
                              "resolved_elevation_m": loc.get("elevation"), "timezone": loc.get("timezone"),
                              "utc_offset_seconds": loc.get("utc_offset_seconds"), "hourly_units": loc.get("hourly_units", {}),
                              "daily_units": loc.get("daily_units", {}), "minutely_15_units": minute_loc.get("minutely_15_units", {}),
                              "minutely_15_resolved_latitude": minute_loc.get("latitude"),
                              "minutely_15_resolved_longitude": minute_loc.get("longitude"), "response_keys": list(loc.keys())})

    hdf = pd.DataFrame(hourly_records).sort_values(["point_id", "hour_local"]).reset_index(drop=True)
    mdf = pd.DataFrame(minutely_records).sort_values(["point_id", "time_local"]).reset_index(drop=True)
    ddf = pd.DataFrame(daily_records).sort_values(["point_id", "date_local"]).reset_index(drop=True)
    hdf.to_csv(ROOT / "data/processed/weather_route_hourly.csv.gz", index=False, compression="gzip", encoding="utf-8")
    mdf.to_csv(ROOT / "data/processed/weather_route_minutely_15.csv.gz", index=False, compression="gzip", encoding="utf-8")
    ddf.to_csv(ROOT / "data/processed/weather_route_daily.csv.gz", index=False, compression="gzip", encoding="utf-8")
    # Weather interpolation positions along the route and through local time.
    distances = geo.distance_from_darwin_m.to_numpy(dtype=float)
    nr = np.clip(np.searchsorted(nodes, distances, side="right"), 1, len(nodes) - 1)
    nl = nr - 1
    sw = np.divide(distances - nodes[nl], nodes[nr] - nodes[nl], out=np.zeros(len(distances)), where=(nodes[nr] != nodes[nl]))
    local_naive = pd.to_datetime(geo.timestamp_local, utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    time_ns = local_naive.astype("int64").to_numpy()
    hour_labels = hdf[hdf.point_id == 0].hour_local.to_numpy()
    htime = pd.to_datetime(pd.Series(hour_labels)).astype("int64").to_numpy()
    tr = np.clip(np.searchsorted(htime, time_ns, side="right"), 1, len(htime) - 1)
    tl = tr - 1
    tw = np.divide(time_ns - htime[tl], htime[tr] - htime[tl], out=np.zeros(len(time_ns), dtype=float), where=(htime[tr] != htime[tl]))
    weather_columns = {
        "day": geo.day.to_numpy(), "timestamp_local": geo.timestamp_local.to_numpy(), "distance_from_darwin_m": distances,
        "weather_time_lower_local": hour_labels[tl], "weather_time_upper_local": hour_labels[tr], "weather_time_weight_upper": tw,
    }
    minutely_labels = mdf[mdf.point_id == 0].time_local.to_numpy()
    minutely_times = pd.to_datetime(pd.Series(minutely_labels)).astype("int64").to_numpy()
    minutely_tr = np.clip(np.searchsorted(minutely_times, time_ns, side="right"), 1, len(minutely_times) - 1)
    minutely_tl = minutely_tr - 1
    minutely_tw = np.divide(time_ns - minutely_times[minutely_tl], minutely_times[minutely_tr] - minutely_times[minutely_tl],
                            out=np.zeros(len(time_ns), dtype=float), where=(minutely_times[minutely_tr] != minutely_times[minutely_tl]))
    weather_columns["weather_15min_time_lower_local"] = minutely_labels[minutely_tl]
    weather_columns["weather_15min_time_upper_local"] = minutely_labels[minutely_tr]
    weather_columns["weather_15min_time_weight_upper"] = minutely_tw
    weather_columns["weather_route_node_lower_id"] = nl
    weather_columns["weather_route_node_upper_id"] = nr
    weather_columns["weather_route_node_lower_km"] = nodes[nl] / 1000
    weather_columns["weather_route_node_upper_km"] = nodes[nr] / 1000
    weather_columns["weather_route_weight_upper"] = sw
    metadata = pd.DataFrame(response_meta).sort_values("point_id").reset_index(drop=True)
    for side, ids in (("lower", nl), ("upper", nr)):
        for coord in ("request_latitude", "request_longitude", "resolved_latitude", "resolved_longitude"):
            weather_columns[f"weather_{side}_{coord}"] = metadata[coord].to_numpy()[ids]
    static_columns = {"point_id", "route_distance_m", "request_latitude", "request_longitude", "resolved_latitude", "resolved_longitude", "resolved_elevation_m", "hour_local"}
    variables = [c for c in hdf.columns if c not in static_columns]
    minutely_static = {"point_id", "route_distance_m", "request_latitude", "request_longitude", "resolved_latitude", "resolved_longitude", "resolved_elevation_m", "time_local"}
    minutely_variables = [c for c in mdf.columns if c not in minutely_static]
    daily_static = {"point_id", "route_distance_m", "request_latitude", "request_longitude", "resolved_latitude", "resolved_longitude", "resolved_elevation_m", "date_local"}
    daily_variables = [c for c in ddf.columns if c not in daily_static]

    def missing_profile(frame: pd.DataFrame, names: list[str]) -> dict:
        missing = frame[names].isna().sum()
        return {"all_null_variables": missing[missing == len(frame)].index.tolist(),
                "variables_with_missing_values": {k: int(v) for k, v in missing.items() if v > 0}}

    variable_availability = {
        "hourly": missing_profile(hdf, variables),
        "minutely_15": missing_profile(mdf, minutely_variables),
        "daily": missing_profile(ddf, daily_variables),
    }
    node_hourly: dict[str, np.ndarray] = {}
    for name in variables:
        node_hourly[name] = hdf.pivot(index="point_id", columns="hour_local", values=name).sort_index().to_numpy(dtype=float)
        categorical = name in {"weather_code", "is_day"}
        weather_columns[name] = interp_arrays(node_hourly[name], nl, nr, tl, tr, sw, tw, name, categorical)
    node_minutely: dict[str, np.ndarray] = {}
    for name in minutely_variables:
        node_minutely[name] = mdf.pivot(index="point_id", columns="time_local", values=name).sort_index().to_numpy(dtype=float)
        categorical = name in {"weather_code", "is_day"}
        weather_columns[f"minutely_15_{name}"] = interp_arrays(node_minutely[name], nl, nr, minutely_tl, minutely_tr, sw, minutely_tw, name, categorical)
    # Daily aggregates remain constant in time and are interpolated spatially; times are local calendar dates.
    ddates = pd.to_datetime(ddf.date_local).dt.date.to_numpy()
    row_dates = local_naive.dt.date.to_numpy()
    day_order = sorted(set(ddates))
    day_lookup = {d: i for i, d in enumerate(day_order)}
    day_idx = np.array([day_lookup[d] for d in row_dates], dtype=int)
    day_interp_meta = {}
    for name in daily_variables:
        mat = ddf.pivot(index="point_id", columns="date_local", values=name).sort_index().to_numpy()
        out_name = "daily_" + name
        if name in {"sunrise", "sunset", "moonrise", "moonset"}:
            # Convert local timestamp strings to seconds from local midnight for spatially meaningful interpolation.
            parsed = pd.to_datetime(mat.ravel(), errors="coerce")
            secs = (parsed.hour * 3600 + parsed.minute * 60 + parsed.second).to_numpy(dtype=float).reshape(mat.shape)
            mat = secs
        mat = mat.astype(float)
        lv, rv = mat[nl, day_idx], mat[nr, day_idx]
        w = sw
        if name in {"weather_code", "wind_direction_10m_dominant"}:
            if "wind_direction" in name:
                a = np.deg2rad(lv); b = np.deg2rad(rv)
                val = (np.rad2deg(np.arctan2((1-w)*np.sin(a)+w*np.sin(b), (1-w)*np.cos(a)+w*np.cos(b)))+360)%360
            else:
                val = np.where(w < 0.5, lv, rv)
        else:
            den = (1-w)*np.isfinite(lv) + w*np.isfinite(rv)
            val = np.divide((1-w)*np.nan_to_num(lv) + w*np.nan_to_num(rv), den, out=np.full(len(geo), np.nan), where=den > 0)
        if name in {"sunrise", "sunset", "moonrise", "moonset"}:
            day0 = pd.to_datetime(row_dates.astype(str)).to_numpy(dtype="datetime64[ns]")
            seconds = np.rint(np.nan_to_num(val, nan=0)).astype(np.int64)
            val_dt = day0 + seconds.astype("timedelta64[s]")
            formatted = pd.to_datetime(val_dt).strftime("%Y-%m-%dT%H:%M:%S")
            weather_columns[out_name] = [formatted[i] if np.isfinite(val[i]) else None for i in range(len(val))]
        else:
            weather_columns[out_name] = val
    weather_columns["weather_source"] = np.repeat("Open-Meteo Historical Forecast API; model-grid output interpolated in space/time", len(geo))
    weather_columns["weather_temporal_source_resolution"] = np.repeat(
        "hourly series plus minutely_15 series; linear interpolation to 5-second rows (categorical nearest); regional 15-minute values may be hourly-model interpolation", len(geo))
    weather_columns["weather_spatial_source_spacing_km"] = np.repeat(args.spacing_km, len(geo))
    weather_columns["weather_gti_assumed_tilt_degrees"] = np.zeros(len(geo), dtype=int)
    weather_columns["weather_gti_assumed_azimuth_degrees"] = np.zeros(len(geo), dtype=int)
    weather = pd.DataFrame(weather_columns)
    out = pd.concat([geo.reset_index(drop=True), weather.drop(columns=["day", "timestamp_local", "distance_from_darwin_m"])], axis=1)
    out.to_csv(ROOT / "data/processed/race_days_distance_geo_weather.csv", index=False, encoding="utf-8-sig")
    out.to_csv(ROOT / "data/processed/race_days_distance_geo_weather.csv.gz", index=False, compression="gzip", encoding="utf-8")
    report = {
        "source_api": API, "api_documentation": "https://open-meteo.com/en/docs/historical-forecast-api", "timezone": TZ,
        "dates_local": ["2025-08-24", "2025-08-29"], "route_weather_nodes": len(nodes), "requested_spacing_km": args.spacing_km,
        "telemetry_rows_with_weather": len(out), "hourly_model_values": len(variables), "daily_model_values": len(daily_variables),
        "minutely_15_model_values": len(minutely_variables), "minutely_15_timestamps_per_route_node": len(minutely_labels),
        "rejected_or_known_unavailable_variables": sorted(excluded), "hourly_variables_returned": variables, "daily_variables_returned": daily_variables,
        "minutely_15_variables_returned": minutely_variables, "variable_availability": variable_availability,
        "api_response_location_metadata": response_meta, "model_selection": "Open-Meteo default/best-match historical forecast; response model name is not consistently exposed per point",
        "limitations": ["Coordinates resolve to forecast-model grid cells; returned grid coordinates are preserved.",
                        "Hourly weather is linearly interpolated to 5-second telemetry timestamps; this is a model reconstruction, not an on-vehicle sensor reading.",
                        "Weather values are interpolated along the official route at the requested route-node spacing; 20 km route-node spacing can smooth shorter-scale local weather variation, and pressure-level/daily data can be null where models lack coverage.",
                        "Open-Meteo documents 15-minute native data for Central Europe and North America; in Australia the returned 15-minute series is interpolated from hourly weather and does not add independent temporal measurements.",
                        "GTI is returned only for horizontal plane parameters (tilt=0°, azimuth=0°); this is not YATA's unknown installed panel plane.",
                        "Historical Forecast precipitable_water is not accepted by the endpoint and was omitted."]
    }
    (ROOT / "reports/weather_coverage.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"weather_rows": len(out), "nodes": len(nodes), "hourly_variables": len(variables), "minutely_15_variables": len(minutely_variables), "daily_variables": len(daily_variables),
                      "output": str(ROOT / "data/processed/race_days_distance_geo_weather.csv"), "excluded": sorted(excluded)}, indent=2))


if __name__ == "__main__":
    main()
