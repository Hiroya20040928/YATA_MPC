#!/usr/bin/env python3
"""Causal-vintage shadow replay of a two-layer YATA cruise MPC on BWSC2025.

Upper layer: finite mixed-integer speed schedule to the end of the current
road day; lower layer: 5 s move-blocked tracking MPC. Outputs are advisory.
The archive has no logged actuator command, so this does not claim hardware
closed-loop validation.
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data/processed/race_days_distance_geo_weather.csv"
ROUTE = ROOT / "data/input/route/route_100m_dem.csv"
EVENTS = ROOT / "data/input/route/events.csv"
ROAD_MODEL = ROOT / "models/vehicle_power_model_fit.json"
PV_MODEL = ROOT / "models/pv_model_fit.json"
SOC_MAP = ROOT / "models/battery_voltage_soc_map.csv"
COMPONENT_MODEL = ROOT / "models/component_replay_fit.json"
RESIDUAL_BLOCK_MODEL = ROOT / "reports/residual_online_filters_oof.csv"
WEATHER_DIR = ROOT / "data/weather_raw/mpc_single_runs"
OUTPUT = ROOT / "data/processed/mpc_2025_as_live_replay.csv"
DAY_METRICS = ROOT / "reports/mpc_2025_day_comparison.csv"
RESERVE_SENS = ROOT / "reports/mpc_2025_reserve_sensitivity.csv"
SUMMARY = ROOT / "reports/mpc_2025_replay_summary.json"
REPORT = ROOT / "reports/hierarchical_mpc_2025_replay.md"
PLOT = ROOT / "reports/hierarchical_mpc_2025_replay.png"

API = "https://single-runs-api.open-meteo.com/v1/forecast"
TZ = "Australia/Darwin"
MODEL_ID = "ecmwf_ifs"
WEATHER_VARS = [
    "temperature_2m", "wind_speed_10m", "wind_direction_10m", "surface_pressure",
    "shortwave_radiation_instant", "direct_normal_irradiance_instant",
    "diffuse_radiation_instant",
]
WEATHER_OUTPUT_VARS = ["temperature_2m", "wind_speed_10m", "wind_direction_10m", "surface_pressure",
                       "shortwave_radiation_instant", "direct_normal_irradiance_instant",
                       "diffuse_radiation_instant"]
G = 9.80665
R_AIR = 287.05
MASS_KG = 314.0
CD_A = 0.08876
CRR = 0.0035
CAPACITY_WH = 3011.0
RESERVE_DEFAULT = 0.15
MPC_RESIDUAL_MARGIN_FACTOR = 0.0  # user accepted risk; report residual validation separately
SPEEDS = np.arange(60.0, 79.0, 1.0)
MAX_MODEL_SPEED_KMH = 78.2
HORIZON_M = 10_000.0
UPPER_REPLAN_MIN = 30
LOWER_DT_S = 5.0
LOWER_HORIZON_N = 6


def utc_cycle(ts: pd.Timestamp) -> pd.Timestamp:
    """Latest 6-hour ECMWF cycle with a conservative 6 h publication delay."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    available_at = t - pd.Timedelta(hours=6)
    hour = (available_at.hour // 6) * 6
    return available_at.normalize() + pd.Timedelta(hours=hour)


def get_json(url: str, timeout: int = 45):
    req = urllib.request.Request(url, headers={"User-Agent": "YATA-MPC-research/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))


def route_nodes(route: pd.DataFrame, spacing_m: float = 50_000.0) -> np.ndarray:
    end = float(route.distance_m.max())
    nodes = np.arange(0.0, end, spacing_m)
    return np.unique(np.r_[nodes, end]).astype(float)


def fetch_run(run: pd.Timestamp, route: pd.DataFrame, nodes: np.ndarray) -> pd.DataFrame:
    key = run.strftime("%Y-%m-%dT%H-%M")
    dest = WEATHER_DIR / f"{key}.csv"
    if dest.exists():
        d = pd.read_csv(dest, parse_dates=["time_utc"])
        if len(d) and set(WEATHER_OUTPUT_VARS).issubset(d.columns):
            return d
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    lat = np.interp(nodes, route.distance_m, route.lat)
    lon = np.interp(nodes, route.distance_m, route.lon)
    rows = []
    # Coordinate lists are batched conservatively to keep the request compact.
    for begin in range(0, len(nodes), 45):
        sl = slice(begin, min(begin + 45, len(nodes)))
        params = {
            "latitude": ",".join(f"{x:.6f}" for x in lat[sl]),
            "longitude": ",".join(f"{x:.6f}" for x in lon[sl]),
            "run": run.strftime("%Y-%m-%dT%H:%M"), "models": MODEL_ID,
            "hourly": ",".join(WEATHER_VARS), "forecast_hours": "72", "timezone": "UTC",
        }
        payload = get_json(API + "?" + urllib.parse.urlencode(params))
        locations = payload if isinstance(payload, list) else [payload]
        if len(locations) != sl.stop - sl.start:
            raise RuntimeError(f"Open-Meteo location count mismatch for {key}: {len(locations)}")
        for node_i, loc in enumerate(locations, begin):
            hourly = loc.get("hourly", {})
            times = hourly.get("time", [])
            if not times:
                raise RuntimeError(f"No hourly forecast for {key} node {node_i}: {loc.get('error')}")
            for j, valid in enumerate(times):
                row = {"run_init_utc": run.isoformat(), "node_distance_m": float(nodes[node_i]),
                       "forecast_latitude": loc.get("latitude"), "forecast_longitude": loc.get("longitude"),
                       "time_utc": pd.Timestamp(valid, tz="UTC")}
                for var in WEATHER_OUTPUT_VARS:
                    row[var] = (hourly.get(var) or [None] * len(times))[j]
                rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(dest, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    return d


class Forecast:
    def __init__(self, frame: pd.DataFrame):
        d = frame.copy()
        d["time_utc"] = pd.to_datetime(d.time_utc, utc=True)
        self.dist = np.sort(d.node_distance_m.unique().astype(float))
        self.times = pd.DatetimeIndex(d.time_utc.unique()).sort_values()
        self.time_num = (self.times.tz_localize(None)
                         .to_numpy(dtype="datetime64[ns]").astype(np.int64) / 1e9)
        self.interp = {}
        for var in WEATHER_OUTPUT_VARS:
            pivot = d.pivot(index="node_distance_m", columns="time_utc", values=var).reindex(
                index=self.dist, columns=self.times)
            arr = pivot.to_numpy(float, copy=True)
            # Small missing solar-radiation gaps at run initialization are zero at night;
            # other gaps are linearly filled along forecast time, never from later runs.
            if "radiation" in var or "irradiance" in var:
                arr = np.where(np.isfinite(arr), arr, 0.0)
            else:
                for i in range(arr.shape[0]):
                    good = np.isfinite(arr[i])
                    if good.any():
                        arr[i] = np.interp(np.arange(arr.shape[1]), np.flatnonzero(good), arr[i, good])
            self.interp[var] = RegularGridInterpolator((self.dist, self.time_num), arr,
                                                       bounds_error=False, fill_value=np.nan)

    def at(self, distance_m: float, times: pd.DatetimeIndex) -> dict[str, np.ndarray]:
        t = pd.DatetimeIndex(times)
        if t.tz is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        sec = t.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype(np.int64) / 1e9
        pts = np.column_stack([np.full(len(t), np.clip(distance_m, self.dist[0], self.dist[-1])), sec])
        return {k: v(pts) for k, v in self.interp.items()}


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(y, x)) % 360.0


def sun_position(times_utc: pd.DatetimeIndex, lat_deg: float, lon_deg: float):
    t = pd.DatetimeIndex(times_utc).tz_convert("UTC")
    doy = t.dayofyear.to_numpy(float)
    hour = (t.hour + t.minute / 60.0 + t.second / 3600.0).to_numpy(float)
    gamma = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eq = 229.18 * (0.000075 + 0.001868*np.cos(gamma) - 0.032077*np.sin(gamma)
                   - 0.014615*np.cos(2*gamma) - 0.040849*np.sin(2*gamma))
    dec = (0.006918 - 0.399912*np.cos(gamma) + 0.070257*np.sin(gamma)
           - 0.006758*np.cos(2*gamma) + 0.000907*np.sin(2*gamma)
           - 0.002697*np.cos(3*gamma) + 0.00148*np.sin(3*gamma))
    solar_min = (hour * 60 + 4 * lon_deg + eq) % 1440
    ha = np.radians(solar_min / 4 - 180)
    lat = np.radians(lat_deg)
    sin_alt = np.sin(lat)*np.sin(dec) + np.cos(lat)*np.cos(dec)*np.cos(ha)
    alt = np.arcsin(np.clip(sin_alt, -1, 1))
    cos_alt = np.maximum(np.cos(alt), 1e-9)
    sin_az = -np.cos(dec)*np.sin(ha)/cos_alt
    cos_az = (np.sin(dec)-np.sin(lat)*np.sin(alt))/(np.cos(lat)*cos_alt)
    return alt, np.arctan2(sin_az, cos_az) % (2*np.pi)


def pv_power(forecast: dict[str, np.ndarray], times: pd.DatetimeIndex, lat: float, lon: float,
             heading_deg: float, pvpars: dict) -> np.ndarray:
    ghi = np.maximum(0.0, np.nan_to_num(forecast["shortwave_radiation_instant"], nan=0.0))
    dni = np.maximum(0.0, np.nan_to_num(forecast["direct_normal_irradiance_instant"], nan=0.0))
    dhi = np.maximum(0.0, np.nan_to_num(forecast["diffuse_radiation_instant"], nan=0.0))
    alt, saz = sun_position(times, lat, lon)
    zen = np.pi / 2 - alt
    pars = pvpars["model_candidates"]["faceted_body_poa"]["parameters"]
    gains = pars["facet_gains_m2"]
    offsets = [0.0, 0.0, 180.0, 90.0, -90.0]
    tilts = [0.0, 90.0, 90.0, 90.0, 90.0]
    gvals = [gains[n] for n in ["roof", "front", "rear", "left_side", "right_side"]]
    poa_sum = np.zeros(len(ghi))
    for tilt_d, offset_d, gain in zip(tilts, offsets, gvals):
        tilt = np.radians(tilt_d)
        panel_az = np.radians((heading_deg + offset_d) % 360.0)
        cos_aoi = np.cos(zen)*np.cos(tilt) + np.sin(zen)*np.sin(tilt)*np.cos(saz-panel_az)
        poa = (dni*np.maximum(cos_aoi, 0.0) + dhi*(1+np.cos(tilt))/2
               + ghi*0.2*(1-np.cos(tilt))/2)
        poa_sum += gain * poa
    return np.where(ghi >= 50.0, np.maximum(poa_sum, 0.0), 0.0)


def make_day_stages(start_m: float, end_m: float, control_stops: list[float]):
    edges = [start_m]
    x = start_m
    while x + HORIZON_M < end_m - 1e-6:
        x += HORIZON_M
        if not any(abs(x-cs) < 1.0 for cs in control_stops):
            edges.append(x)
    edges.extend(cs for cs in control_stops if start_m < cs < end_m)
    edges.append(end_m)
    edges = sorted(set(round(float(x), 3) for x in edges))
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b-a > 1]


def segment_energy_candidates(forecast: Forecast, stage: tuple[float, float],
                              cumulative_distance_m: float, time_origin: pd.Timestamp,
                              stop_delay_before_s: float, route: pd.DataFrame,
                              roadpars: dict, pvpars: dict, residual_bias_w: float = 0.0):
    a, b = stage
    length = b - a
    mid = (a + b) / 2
    # Conservative weather vintage envelope: every hourly forecast valid time
    # reachable at 60..78 km/h is checked; take maximum net pack energy.
    earliest = time_origin + pd.Timedelta(seconds=stop_delay_before_s + cumulative_distance_m / (MAX_MODEL_SPEED_KMH/3.6))
    latest = time_origin + pd.Timedelta(seconds=stop_delay_before_s + cumulative_distance_m / (60.0/3.6))
    first = earliest.floor("h")
    last = latest.ceil("h")
    times = pd.date_range(first, last, freq="1h", tz="UTC")
    if len(times) == 0:
        times = pd.DatetimeIndex([earliest])
    wx = forecast.at(mid, times)
    lat = float(np.interp(mid, route.distance_m, route.lat))
    lon = float(np.interp(mid, route.distance_m, route.lon))
    lat1 = float(np.interp(max(0.0, mid-250), route.distance_m, route.lat))
    lon1 = float(np.interp(max(0.0, mid-250), route.distance_m, route.lon))
    lat2 = float(np.interp(min(float(route.distance_m.max()), mid+250), route.distance_m, route.lat))
    lon2 = float(np.interp(min(float(route.distance_m.max()), mid+250), route.distance_m, route.lon))
    heading = float(bearing_deg(lat1, lon1, lat2, lon2))
    z1 = float(np.interp(a, route.distance_m, route.elev_dem_m))
    z2 = float(np.interp(b, route.distance_m, route.elev_dem_m))
    grade = (z2-z1)/length
    pars = roadpars["fit_parameters"]["parameters"]
    paux, br, ba, bg = [pars[k] for k in ["p_aux_w", "beta_roll", "beta_aero", "beta_grade"]]
    temp = wx["temperature_2m"]
    pressure = wx["surface_pressure"] * 100.0
    wind = wx["wind_speed_10m"] / 3.6
    wfrom = np.radians(wx["wind_direction_10m"])
    headwind = wind*np.cos(wfrom-np.radians(heading))
    rho = pressure/(R_AIR*(temp+273.15))
    pv = pv_power(wx, times, lat, lon, heading, pvpars)
    road_sd = float(pars["student_t_scale_w"])
    pv_sd = float(pvpars["model_candidates"]["faceted_body_poa"]["scale_w"])
    q90 = float(student_t.ppf(0.90, df=4.0))
    gross_margin = q90*road_sd
    pv_margin = q90*pv_sd
    # Candidate speed x reachable hourly weather times.
    v_ms = SPEEDS/3.6
    gross = (paux + br*MASS_KG*G*CRR*np.cos(np.arctan(grade))*v_ms[:, None]
             + ba*0.5*rho[None, :]*CD_A*(v_ms[:, None]+headwind[None, :])
             * np.abs(v_ms[:, None]+headwind[None, :])*v_ms[:, None]
             + bg*MASS_KG*G*np.sin(np.arctan(grade))*v_ms[:, None])
    # Risk-tolerant point prediction for vehicle/PV terms. The residual MAE is
    # reported separately; it is not misrepresented as a hard confidence bound.
    net_upper = (gross + MPC_RESIDUAL_MARGIN_FACTOR*gross_margin
                 - (pv[None, :] - MPC_RESIDUAL_MARGIN_FACTOR*pv_margin))
    distance_km = length/1000.0
    time_h = distance_km/SPEEDS
    wh = net_upper * (distance_km/SPEEDS[:, None])
    # The validated past-only median residual filter predicts a 5-minute block.
    # Apply it only to the first five minutes after a replan, never recursively
    # across the multi-hour upper horizon.
    if cumulative_distance_m <= 1e-6 and np.isfinite(residual_bias_w):
        wh += residual_bias_w * np.minimum(time_h, 5.0/60.0)[:, None]
    worst_wh = np.nanmax(wh, axis=1)
    # Time uses the selected decision; hourly stop time is inserted separately.
    return worst_wh, time_h, heading, grade, float(np.nanmax(pv)), gross_margin, pv_margin


def solve_upper(forecast: Forecast, current_distance: float, day_end: float,
                control_stops: list[float], now: pd.Timestamp, energy_wh: float,
                reserve_pct: float, route: pd.DataFrame, roadpars: dict, pvpars: dict,
                deadline: pd.Timestamp, residual_bias_w: float = 0.0,
                solver_time_limit_s: float = 2.0,
                stop_dwell_s_by_distance: dict[float, float] | None = None):
    stops = [x for x in control_stops if current_distance < x < day_end]
    dwell_by_stop = {round(float(cs), 0): float((stop_dwell_s_by_distance or {}).get(cs, 1800.0))
                     for cs in stops}
    stages = make_day_stages(current_distance, day_end, stops)
    nstage, na = len(stages), len(SPEEDS)
    if nstage == 0:
        return {"status": "at_day_endpoint", "target_kmh": None, "planned_speeds_kmh": [],
                "planned_segments": [], "advisory_eligible": False,
                "predicted_remaining_h": 0.0, "predicted_remaining_wh_robust": 0.0,
                "mip_gap": 0.0, "stage_count": 0}
    if not np.isfinite(energy_wh) or energy_wh < 0 or not np.isfinite(reserve_pct) or not 0 <= reserve_pct <= 1:
        return {"status": "invalid_state_or_constraint", "target_kmh": None,
                "planned_speeds_kmh": [], "planned_segments": [], "advisory_eligible": False,
                "predicted_remaining_h": None, "predicted_remaining_wh_robust": None,
                "mip_gap": None, "stage_count": nstage,
                "diagnostic": {"energy_wh": float(energy_wh), "reserve_pct": float(reserve_pct)}}
    if not np.isfinite(solver_time_limit_s) or solver_time_limit_s <= 0:
        raise ValueError("solver_time_limit_s must be positive and finite")
    energies, times = [], []
    cum_dist = 0.0
    stops_elapsed = 0.0
    next_cs_by_end = {round(cs, 0) for cs in stops}
    for stage in stages:
        e, t, _, _, _, _, _ = segment_energy_candidates(forecast, stage, cum_dist, now,
                                                         stops_elapsed, route, roadpars, pvpars,
                                                         residual_bias_w if cum_dist <= 1e-6 else 0.0)
        energies.append(e)
        times.append(t)
        cum_dist += stage[1]-stage[0]
        if any(abs(stage[1]-cs) < 2 for cs in next_cs_by_end):
            stop = min(next_cs_by_end, key=lambda cs: abs(stage[1]-cs))
            stops_elapsed += dwell_by_stop.get(round(float(stop), 0), 1800.0)
    e_mat = np.vstack(energies)
    t_mat = np.vstack(times)
    if not np.isfinite(e_mat).all() or not np.isfinite(t_mat).all():
        return {"status": "forecast_coverage_missing", "target_kmh": None,
                "planned_speeds_kmh": [], "planned_segments": [], "advisory_eligible": False,
                "predicted_remaining_h": None, "predicted_remaining_wh_robust": None,
                "mip_gap": None, "stage_count": nstage,
                "diagnostic": {"reason": "weather interpolation did not cover every reachable stage time"}}
    c = t_mat.ravel()
    nvar = nstage*na
    A = lil_matrix((nstage+2, nvar), dtype=float)
    lb = np.full(nstage+2, -np.inf)
    ub = np.full(nstage+2, np.inf)
    for i in range(nstage):
        A[i, i*na:(i+1)*na] = 1.0
        lb[i] = ub[i] = 1.0
    A[nstage, :] = e_mat.ravel()/1000.0
    lb[nstage] = -np.inf
    energy_limit_wh = energy_wh - CAPACITY_WH*reserve_pct
    ub[nstage] = energy_limit_wh/1000.0
    # The route has to fit before the official 17:00 road-day close.
    A[nstage+1, :] = t_mat.ravel()*3600.0
    lb[nstage+1] = -np.inf
    mandatory_stop_s = float(sum(dwell_by_stop.values()))
    available_route_s = (deadline-now).total_seconds() - mandatory_stop_s
    ub[nstage+1] = available_route_s
    res = milp(c=c, integrality=np.ones(nvar), bounds=Bounds(np.zeros(nvar), np.ones(nvar)),
               constraints=LinearConstraint(A.tocsr(), lb, ub),
               options={"mip_rel_gap": 0.0, "time_limit": float(solver_time_limit_s), "presolve": True})
    def decode_incumbent(vector: np.ndarray | None):
        if vector is None or len(vector) < nvar:
            return None
        matrix = np.asarray(vector[:nvar], dtype=float).reshape(nstage, na)
        rounded = np.rint(matrix)
        if not np.isfinite(matrix).all() or np.max(np.abs(matrix-rounded)) > 1e-5:
            return None
        if not np.all(np.sum(rounded, axis=1) == 1):
            return None
        return np.argmax(matrix, axis=1)

    def profile_metrics(indices: np.ndarray) -> dict:
        energy = float(np.sum(e_mat[np.arange(nstage), indices]))
        travel_s = float(np.sum(t_mat[np.arange(nstage), indices])*3600.0)
        return {"energy_wh": energy, "travel_s": travel_s,
                "reserve_shortfall_wh": max(0.0, energy-energy_limit_wh),
                "deadline_overrun_s": max(0.0, travel_s-available_route_s)}

    def no_advisory(status: str, diagnostic: dict, gap=None) -> dict:
        return {"status": status, "target_kmh": None, "planned_speeds_kmh": [],
                "planned_segments": [], "advisory_eligible": False,
                "predicted_remaining_h": None, "predicted_remaining_wh_robust": None,
                "energy_budget_wh": energy_limit_wh, "mip_gap": gap,
                "stage_count": nstage, "reserve_pct": reserve_pct,
                "solver_status_code": int(res.status), "solver_message": str(res.message),
                "diagnostic": diagnostic}

    status_by_code = {
        0: "optimal_finite_grid_milp",
        1: "feasible_time_limited_milp_not_proven_optimal",
    }
    planned_idx = decode_incumbent(res.x)
    if int(res.status) not in status_by_code or planned_idx is None:
        min_energy_idx = np.argmin(e_mat, axis=1)
        min_time_idx = np.argmin(t_mat, axis=1)
        min_energy_metrics = profile_metrics(min_energy_idx)
        min_time_metrics = profile_metrics(min_time_idx)
        diagnostic = {
            "hard_constraints": {
                "one_speed_per_stage": "enforced",
                "speed_range_kmh": [float(np.min(SPEEDS)), float(np.max(SPEEDS))],
                "terminal_energy_at_least_wh": float(CAPACITY_WH*reserve_pct),
                "route_day_deadline_utc": pd.Timestamp(deadline).isoformat(),
                "deadline_available_after_stops_s": float(available_route_s),
            },
            "solver_code_meaning": ({2: "problem_proven_infeasible", 1: "time_limit_without_feasible_incumbent"}
                                     .get(int(res.status), "solver_error_or_invalid_incumbent")),
            "minimum_energy_profile_diagnostic_only": {
                "speed_kmh": SPEEDS[min_energy_idx].astype(float).tolist(), **min_energy_metrics},
            "minimum_time_profile_diagnostic_only": {
                "speed_kmh": SPEEDS[min_time_idx].astype(float).tolist(), **min_time_metrics},
            "relaxed_reserve_phase_one": None,
            "diagnostic_only_no_speed_recommendation": True,
        }
        # Feasibility restoration follows the standard slack-variable pattern,
        # but remains diagnostic: the live controller may only use a solution
        # satisfying the original hard reserve and deadline constraints.
        if int(res.status) == 2:
            n_phase = nvar+1
            A_phase = lil_matrix((nstage+2, n_phase), dtype=float)
            A_phase[:nstage, :nvar] = A[:nstage, :]
            A_phase[nstage, :nvar] = e_mat.ravel()/1000.0
            A_phase[nstage, nvar] = -1.0
            A_phase[nstage+1, :nvar] = t_mat.ravel()*3600.0
            lb_phase = np.full(nstage+2, -np.inf)
            ub_phase = np.full(nstage+2, np.inf)
            lb_phase[:nstage] = ub[:nstage]
            ub_phase[:nstage] = ub[:nstage]
            ub_phase[nstage] = energy_limit_wh/1000.0
            ub_phase[nstage+1] = available_route_s
            c_phase = np.zeros(n_phase)
            c_phase[nvar] = 1.0
            int_phase = np.r_[np.ones(nvar, dtype=int), 0]
            lower_phase = np.zeros(n_phase)
            upper_phase = np.r_[np.ones(nvar), np.inf]
            phase = milp(c=c_phase, integrality=int_phase,
                         bounds=Bounds(lower_phase, upper_phase),
                         constraints=LinearConstraint(A_phase.tocsr(), lb_phase, ub_phase),
                         options={"mip_rel_gap": 0.0,
                                  "time_limit": min(2.0, float(solver_time_limit_s)),
                                  "presolve": True})
            phase_idx = decode_incumbent(phase.x)
            if phase_idx is not None and int(phase.status) in {0, 1}:
                phase_metrics = profile_metrics(phase_idx)
                diagnostic["relaxed_reserve_phase_one"] = {
                    "solver_status": "optimal" if phase.status == 0 else "time_limited_feasible",
                    "reserve_slack_wh": float(phase_metrics["reserve_shortfall_wh"]),
                    "deadline_overrun_s": float(phase_metrics["deadline_overrun_s"]),
                    "diagnostic_speed_kmh": SPEEDS[phase_idx].astype(float).tolist(),
                    "diagnostic_only_no_speed_recommendation": True,
                }
            else:
                diagnostic["relaxed_reserve_phase_one"] = {
                    "solver_status": "infeasible_or_no_incumbent",
                    "diagnostic_only_no_speed_recommendation": True,
                }
        result_status = ("infeasible_hard_constraints" if res.status == 2 else
                         "no_feasible_incumbent_before_time_limit" if res.status == 1 else
                         "solver_error_or_invalid_incumbent")
        return no_advisory(result_status, diagnostic, getattr(res, "mip_gap", None))

    metrics = profile_metrics(planned_idx)
    tolerance_wh = 1.0
    tolerance_s = 1.0
    if metrics["reserve_shortfall_wh"] > tolerance_wh or metrics["deadline_overrun_s"] > tolerance_s:
        return no_advisory("solver_incumbent_failed_hard_constraint_check", {
            "incumbent_constraint_metrics": metrics,
            "tolerance_wh": tolerance_wh, "tolerance_s": tolerance_s,
            "diagnostic_only_no_speed_recommendation": True,
        }, getattr(res, "mip_gap", None))

    planned = SPEEDS[planned_idx].astype(float)
    planned_segments = []
    for i, (start_m, end_m) in enumerate(stages):
        ix = int(planned_idx[i])
        planned_segments.append({
            "start_distance_m": float(start_m), "end_distance_m": float(end_m),
            "target_speed_kmh": float(planned[i]),
            "predicted_net_energy_wh": float(e_mat[i, ix]),
            "predicted_travel_time_s": float(t_mat[i, ix]*3600.0),
        })
    total_t = metrics["travel_s"]/3600.0 + mandatory_stop_s/3600.0
    gap = float(getattr(res, "mip_gap", 0.0) or 0.0)
    status = status_by_code[int(res.status)]
    return {"status": status, "target_kmh": float(planned[0]),
            "planned_speeds_kmh": planned.tolist(), "planned_segments": planned_segments,
            "advisory_eligible": True,
            "predicted_remaining_h": total_t,
            "predicted_remaining_wh_robust": metrics["energy_wh"],
            "predicted_terminal_energy_wh": float(energy_wh-metrics["energy_wh"]),
            "energy_budget_wh": energy_limit_wh,
            "energy_margin_wh": float(energy_limit_wh-metrics["energy_wh"]),
            "deadline_margin_s": float(available_route_s-metrics["travel_s"]),
            "mip_gap": gap, "stage_count": nstage, "reserve_pct": reserve_pct,
            "solver_status_code": int(res.status), "solver_message": str(res.message),
            "causal_residual_bias_w": float(residual_bias_w)}


def lower_move_block_mpc(speed_kmh: float, target_kmh: float, amax: float,
                         previous_accel: float = 0.0):
    """Exact scalar QP optimum with acceleration held over a 30 s horizon."""
    v = speed_kmh/3.6
    ref = target_kmh/3.6
    tau = np.arange(1, LOWER_HORIZON_N+1, dtype=float)*LOWER_DT_S
    q, r, s = 1.0, 1.0, 4.0
    unconstrained = (q*np.sum(tau*(ref-v)) + s*previous_accel) / (q*np.sum(tau**2)+r+s)
    lo, hi = -amax, amax
    # Do not command braking below 60 when already under the open-road minimum;
    # the lower layer should accelerate toward the legal operating envelope.
    if v >= 60/3.6:
        lo = max(lo, (60/3.6-v)/(LOWER_HORIZON_N*LOWER_DT_S))
    hi = min(hi, (MAX_MODEL_SPEED_KMH/3.6-v)/(LOWER_HORIZON_N*LOWER_DT_S))
    accel = float(np.clip(unconstrained, lo, hi)) if hi >= lo else 0.0
    projected = float(np.clip((v+accel*LOWER_HORIZON_N*LOWER_DT_S)*3.6, 60, MAX_MODEL_SPEED_KMH))
    mechanical_delta_w = MASS_KG*v*accel
    return accel, projected, mechanical_delta_w


def load_inputs():
    route = pd.read_csv(ROUTE).sort_values("distance_m").reset_index(drop=True)
    events = pd.read_csv(EVENTS).fillna("")
    roadpars = json.loads(ROAD_MODEL.read_text(encoding="utf-8"))
    pvpars = json.loads(PV_MODEL.read_text(encoding="utf-8"))
    component_model = json.loads(COMPONENT_MODEL.read_text(encoding="utf-8"))
    socmap = pd.read_csv(SOC_MAP).sort_values("reference_voltage_25s_v")
    needed = {"day", "timestamp_local", "timestamp_utc", "event_phase", "event_flags",
              "distance_from_darwin_m", "Speed[kmh]", "Solar_Watt[W]", "Battery_Watt[W]",
              "Main_Volt[V]", "Battery[A]", "distance_anchor_confidence"}
    df = pd.read_csv(LOG, usecols=lambda c: c in needed, low_memory=False)
    df["timestamp_local"] = pd.to_datetime(df.timestamp_local, utc=True).dt.tz_convert(TZ)
    df["timestamp_utc"] = pd.to_datetime(df.timestamp_utc, utc=True)
    for c in ["distance_from_darwin_m", "Speed[kmh]", "Solar_Watt[W]", "Battery_Watt[W]", "Main_Volt[V]", "Battery[A]"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["day", "timestamp_utc"]).reset_index(drop=True)
    return df, route, events, roadpars, pvpars, socmap, component_model


def daily_endpoints(events: pd.DataFrame, route_end: float):
    ends = {}
    for day in range(1, 6):
        q = events[(events.day.astype(int) == day) & events.event_id.astype(str).str.contains("OVERNIGHT_CHARGE")]
        ends[day] = float(pd.to_numeric(q.distance_m, errors="coerce").dropna().iloc[0])
    ends[6] = float(route_end)
    return ends


def actual_soc_wh(df: pd.DataFrame, socmap: pd.DataFrame, component_model: dict):
    out = np.full(len(df), np.nan)
    byday = {}
    battery = component_model["battery"]
    r_ohm = float(battery["selected_full_clean_day_r_eff_ohm"])
    q_ah = float(battery["selected_capacity_ah"])
    for day, ix in df.groupby("day", sort=False).groups.items():
        ids = np.asarray(list(ix), dtype=int)
        g = df.loc[ids].reset_index(drop=True)
        voltage = g["Main_Volt[V]"].to_numpy(float)
        current = g["Battery[A]"].to_numpy(float)
        t_rel = (g.timestamp_utc - g.timestamp_utc.iloc[0]).dt.total_seconds().to_numpy(float)
        anchor_ix = np.flatnonzero((t_rel <= 180.0) & np.isfinite(voltage)
                                   & (voltage >= 70.0) & (voltage <= 115.0)
                                   & np.isfinite(current) & (np.abs(current) <= 5.0))
        anchor_rule = "first 3 min; |current| <= 5 A; median eligible sample"
        if not len(anchor_ix):
            anchor_ix = np.flatnonzero(np.isfinite(voltage) & (voltage >= 70.0)
                                       & (voltage <= 115.0) & np.isfinite(current))[:20]
            anchor_rule = "fallback first 20 plausible voltage/current samples; median"
        if not len(anchor_ix):
            byday[int(day)] = {"soc_initialization": "unavailable; no plausible 25S voltage/current anchor"}
            continue
        j = int(anchor_ix[len(anchor_ix) // 2])
        v0, i0 = float(voltage[j]), float(current[j])
        # The adopted constant effective pack resistance corrects loaded bus
        # voltage toward the supplied reference curve. The SOC-dependent map
        # remains a sensitivity candidate because it failed Day 6 holdout.
        v_reference = v0 + r_ohm * i0
        soc0 = float(np.interp(v_reference, socmap.reference_voltage_25s_v, socmap.soc_pct))
        dt_raw = g.timestamp_utc.diff().dt.total_seconds().fillna(0).to_numpy(float)
        dt = np.where((dt_raw > 0) & (dt_raw <= 30), dt_raw, 0.0)
        p = g["Battery_Watt[W]"].to_numpy(float)
        cumulative_wh = np.cumsum(np.r_[0.0, (p[1:]+p[:-1])*0.5*dt[1:]/3600.0])
        e_from_anchor_wh = cumulative_wh - cumulative_wh[j]
        energy = np.clip(CAPACITY_WH*soc0/100.0 - e_from_anchor_wh, 0, CAPACITY_WH)
        out[ids] = energy
        byday[int(day)] = {"anchor_voltage_v": v0, "anchor_current_a": i0,
                           "anchor_reference_voltage_v": v_reference,
                           "initial_soc_voltage_lookup_pct": soc0,
                           "initial_energy_wh": CAPACITY_WH*soc0/100.0,
                           "anchor_timestamp_utc": g.timestamp_utc.iloc[j].isoformat(),
                           "anchor_rule": anchor_rule, "effective_resistance_ohm": r_ohm,
                           "capacity_ah_prior": q_ah,
                           "soc_initialization": "robust voltage-current anchor with selected constant effective pack R; approximate, no true in-car SOC"}
    return out, byday


def causal_residual_bias(blocks: pd.DataFrame, day: int, now: pd.Timestamp) -> float:
    """Latest available 5-minute residual correction from prior same-day blocks."""
    q = blocks[(blocks.day.astype(int) == day) & (blocks.block_start_utc <= now)
               & (pd.to_numeric(blocks.prior_blocks, errors="coerce") >= 6)
               & pd.to_numeric(blocks.online_correction_w, errors="coerce").notna()]
    if q.empty:
        return 0.0
    row = q.iloc[-1]
    age_s = float((now - row.block_start_utc).total_seconds())
    if age_s < 0 or age_s > 600:
        return 0.0
    return float(np.clip(float(row.online_correction_w), -100.0, 100.0))


def main():
    df, route, events, roadpars, pvpars, socmap, component_model = load_inputs()
    residual_blocks = pd.read_csv(RESIDUAL_BLOCK_MODEL)
    residual_blocks["block_start_utc"] = pd.to_datetime(residual_blocks.block_start_utc, utc=True)
    day_end = daily_endpoints(events, float(route.distance_m.max()))
    cs_by_day = {}
    for day in range(1, 7):
        q = events[(pd.to_numeric(events.day, errors="coerce") == day) & events.event_type.eq("control_stop")]
        cs_by_day[day] = sorted(pd.to_numeric(q.distance_m, errors="coerce").dropna().astype(float).tolist())
    df["stored_energy_est_wh"], init_meta = actual_soc_wh(df, socmap, component_model)
    # Retrieve only forecast vintages that could have been available at a replay row.
    run_for_row = df.timestamp_utc.map(utc_cycle)
    run_for_row = pd.to_datetime(run_for_row, utc=True)
    unique_runs = sorted({pd.Timestamp(x).tz_convert("UTC") for x in run_for_row})
    nodes = route_nodes(route)
    forecast_by_run = {}
    for i, run in enumerate(unique_runs, 1):
        print(f"Fetching/reading archived IFS run {i}/{len(unique_runs)}: {run.isoformat()}", flush=True)
        forecast_by_run[run] = Forecast(fetch_run(run, route, nodes))
    df["forecast_run_init_utc"] = [x.isoformat() for x in run_for_row]
    df["forecast_age_at_decision_h"] = (df.timestamp_utc.to_numpy(dtype="datetime64[ns]") -
                                          pd.to_datetime(df.forecast_run_init_utc, utc=True).to_numpy(dtype="datetime64[ns]")) / np.timedelta64(1, "h")
    # Replay controller state. Upper layer replans every 15 min; lower layer sees every 5 s.
    n = len(df)
    upper_speed = np.full(n, np.nan)
    upper_status = np.full(n, "inactive", dtype=object)
    upper_obj_time_h = np.full(n, np.nan)
    upper_obj_energy_wh = np.full(n, np.nan)
    upper_budget_wh = np.full(n, np.nan)
    upper_mip_gap = np.full(n, np.nan)
    upper_stages = np.full(n, np.nan)
    upper_residual_bias = np.zeros(n, dtype=float)
    upper_no_advisory_reason = np.full(n, "", dtype=object)
    upper_min_energy_reserve_shortfall_wh = np.full(n, np.nan)
    upper_min_energy_deadline_overrun_s = np.full(n, np.nan)
    upper_min_time_reserve_shortfall_wh = np.full(n, np.nan)
    upper_min_time_deadline_overrun_s = np.full(n, np.nan)
    upper_phase_one_reserve_slack_wh = np.full(n, np.nan)
    upper_replan = np.zeros(n, dtype=bool)
    reserve_pct_arr = np.full(n, RESERVE_DEFAULT)
    lower_accel = np.full(n, np.nan)
    lower_speed_30s = np.full(n, np.nan)
    lower_mech_delta = np.full(n, np.nan)
    lower_accel_limit = np.full(n, np.nan)
    lower_status = np.full(n, "inactive", dtype=object)
    # Causal historical speed-ramp memory; this is driver-trace calibration only,
    # not an actuator lag/throttle-to-acceleration calibration.
    ramp_history: dict[int, list[tuple[pd.Timestamp, float]]] = {}
    active_target = np.nan
    active_plan_meta = {}
    last_upper_bucket = None
    previous_by_day = {}
    first_solve_day = set()
    reserve_sensitivity_rows = []
    trouble_pattern = r"incident:|control_stop:|traffic_stop:|driver_change:|repair:|charge:|hatch_repair:"
    local_times = df.timestamp_local
    for i, row in df.iterrows():
        if i > 0 and int(row.day) != int(df.day.iloc[i-1]):
            print(f"Replay reached Day {int(row.day)} ({i:,}/{n:,} rows)", flush=True)
        day = int(row.day)
        now = row.timestamp_utc
        dnow = float(row.distance_from_darwin_m)
        speed = float(row["Speed[kmh]"] or 0.0)
        flags = str(row.event_flags or "")
        trouble_now = bool(pd.Series([flags]).str.contains(trouble_pattern, regex=True).iloc[0])
        road_allowed = row.event_phase == "road_day" and not trouble_now
        active = road_allowed and speed >= 5
        within_hours = (local_times.iloc[i].hour < 17)
        minute_bucket = (local_times.iloc[i].date(), local_times.iloc[i].hour, local_times.iloc[i].minute//UPPER_REPLAN_MIN)
        must_solve = active and within_hours and (minute_bucket != last_upper_bucket)
        if must_solve:
            run = pd.Timestamp(run_for_row.iloc[i])
            fc = forecast_by_run[run]
            deadline = local_times.iloc[i].normalize() + pd.Timedelta(hours=17)
            residual_bias = causal_residual_bias(residual_blocks, day, now)
            plan = solve_upper(fc, dnow, day_end[day], cs_by_day[day], now,
                               float(row.stored_energy_est_wh), RESERVE_DEFAULT,
                               route, roadpars, pvpars, deadline, residual_bias)
            if int(upper_replan.sum()) % 12 == 0:
                print(f"Upper solve {int(upper_replan.sum())+1}: Day {day}, {local_times.iloc[i].strftime('%H:%M')}, "
                      f"target={plan.get('target_kmh')}, status={plan.get('status')}, "
                      f"gap={plan.get('mip_gap')}", flush=True)
            if day not in first_solve_day:
                first_solve_day.add(day)
                for reserve in [0.0, 0.10, 0.15, 0.20]:
                    alt = plan if reserve == RESERVE_DEFAULT else solve_upper(
                        fc, dnow, day_end[day], cs_by_day[day], now,
                        float(row.stored_energy_est_wh), reserve, route, roadpars, pvpars,
                        deadline, residual_bias)
                    reserve_sensitivity_rows.append({
                        "day": day, "decision_timestamp_local": local_times.iloc[i].isoformat(),
                        "stored_energy_estimate_wh": float(row.stored_energy_est_wh),
                        "reserve_pct": reserve*100, "target_speed_kmh": alt.get("target_kmh"),
                        "status": alt.get("status"), "predicted_remaining_time_h": alt.get("predicted_remaining_h"),
                        "predicted_remaining_net_energy_wh": alt.get("predicted_remaining_wh_robust"),
                        "mip_gap": alt.get("mip_gap"), "stages": alt.get("stage_count"),
                    })
            target_value = plan.get("target_kmh")
            active_target = float(target_value) if target_value is not None and np.isfinite(target_value) else np.nan
            active_plan_meta = plan
            upper_status[i] = plan.get("status", "no_plan")
            last_upper_bucket = minute_bucket
            upper_replan[i] = True
        if active and np.isfinite(active_target):
            upper_speed[i] = active_target
            upper_status[i] = active_plan_meta.get("status", "no_plan")
            upper_obj_time_h[i] = active_plan_meta.get("predicted_remaining_h", np.nan)
            upper_obj_energy_wh[i] = active_plan_meta.get("predicted_remaining_wh_robust", np.nan)
            upper_budget_wh[i] = active_plan_meta.get("energy_budget_wh", np.nan)
            upper_mip_gap[i] = active_plan_meta.get("mip_gap", np.nan)
            upper_stages[i] = active_plan_meta.get("stage_count", np.nan)
            upper_residual_bias[i] = active_plan_meta.get("causal_residual_bias_w", 0.0)
            history = ramp_history.setdefault(day, [])
            history = [(t, a) for t, a in history if (now-t).total_seconds() <= 1800]
            ramp_history[day] = history
            if len(history) >= 12:
                amax = float(np.clip(1.25*np.quantile([a for _,a in history], 0.95), 0.03, 0.15))
                lower_status[i] = "causal_p95_speed_trace_limit_advisory"
            else:
                amax = 0.05
                lower_status[i] = "startup_assumption_0p05mps2_advisory"
            accel, projected, mechanical_delta = lower_move_block_mpc(speed, float(active_target), amax)
            lower_accel[i], lower_speed_30s[i], lower_mech_delta[i], lower_accel_limit[i] = accel, projected, mechanical_delta, amax
            if day in previous_by_day:
                pt, ps = previous_by_day[day]
                dt = (now-pt).total_seconds()
                if 0 < dt <= 15 and flags == "" or (0 < dt <= 15 and "road_day:" in flags and not re.search(trouble_pattern, flags)):
                    ramp_history[day].append((now, abs((speed-ps)/3.6/dt)))
            previous_by_day[day] = (now, speed)
        elif active and active_plan_meta:
            # Preserve the infeasible/solver status in the log without turning
            # the diagnostic minimum-energy profile into a speed recommendation.
            upper_status[i] = active_plan_meta.get("status", "no_plan")
            upper_budget_wh[i] = active_plan_meta.get("energy_budget_wh", np.nan)
            upper_mip_gap[i] = active_plan_meta.get("mip_gap", np.nan)
            upper_stages[i] = active_plan_meta.get("stage_count", np.nan)
            upper_residual_bias[i] = active_plan_meta.get("causal_residual_bias_w", 0.0)
            diagnostic = active_plan_meta.get("diagnostic", {})
            upper_no_advisory_reason[i] = active_plan_meta.get("status", "no_plan")
            min_e = diagnostic.get("minimum_energy_profile_diagnostic_only", {})
            min_t = diagnostic.get("minimum_time_profile_diagnostic_only", {})
            phase_one = diagnostic.get("relaxed_reserve_phase_one") or {}
            upper_min_energy_reserve_shortfall_wh[i] = min_e.get("reserve_shortfall_wh", np.nan)
            upper_min_energy_deadline_overrun_s[i] = min_e.get("deadline_overrun_s", np.nan)
            upper_min_time_reserve_shortfall_wh[i] = min_t.get("reserve_shortfall_wh", np.nan)
            upper_min_time_deadline_overrun_s[i] = min_t.get("deadline_overrun_s", np.nan)
            upper_phase_one_reserve_slack_wh[i] = phase_one.get("reserve_slack_wh", np.nan)
            lower_status[i] = "upper_plan_infeasible_no_advisory"
            previous_by_day[day] = (now, speed)
        else:
            if trouble_now or row.event_phase != "road_day" or not within_hours:
                active_target = np.nan
                active_plan_meta = {}
                last_upper_bucket = None
            previous_by_day[day] = (now, speed)
    out = df.copy()
    out["mpc_reserve_pct"] = reserve_pct_arr
    out["upper_replan_now"] = upper_replan
    out["upper_target_speed_kmh"] = upper_speed
    out["upper_solver_status"] = upper_status
    out["upper_predicted_remaining_time_h"] = upper_obj_time_h
    out["upper_predicted_remaining_net_energy_wh"] = upper_obj_energy_wh
    out["upper_energy_budget_wh"] = upper_budget_wh
    out["upper_mip_gap"] = upper_mip_gap
    out["upper_remaining_stages"] = upper_stages
    out["upper_causal_residual_bias_w"] = upper_residual_bias
    out["upper_no_advisory_reason"] = upper_no_advisory_reason
    out["upper_min_energy_profile_reserve_shortfall_wh_diagnostic"] = upper_min_energy_reserve_shortfall_wh
    out["upper_min_energy_profile_deadline_overrun_s_diagnostic"] = upper_min_energy_deadline_overrun_s
    out["upper_min_time_profile_reserve_shortfall_wh_diagnostic"] = upper_min_time_reserve_shortfall_wh
    out["upper_min_time_profile_deadline_overrun_s_diagnostic"] = upper_min_time_deadline_overrun_s
    out["upper_phase_one_reserve_slack_wh_diagnostic"] = upper_phase_one_reserve_slack_wh
    out["lower_accel_request_mps2"] = lower_accel
    out["lower_predicted_speed_after_30s_kmh"] = lower_speed_30s
    out["lower_mechanical_inertial_power_delta_w"] = lower_mech_delta
    out["lower_accel_bound_mps2"] = lower_accel_limit
    out["lower_status"] = lower_status
    out["controller_output_scope"] = "shadow advisory; no logged actuator mapping"
    out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    # Human-run comparison is observational; shadow targets are counterfactual advice.
    result = []
    for day, g in out.groupby("day", sort=True):
        road = g[(g.event_phase == "road_day") & (g["Speed[kmh]"] >= 5)]
        decision = road[road.upper_replan_now]
        start_dist, end_dist = float(road.distance_from_darwin_m.min()), float(road.distance_from_darwin_m.max())
        moving = road[~road.event_flags.fillna("").str.contains(trouble_pattern, regex=True)]
        duration_h = (road.timestamp_utc.max()-road.timestamp_utc.min()).total_seconds()/3600 if len(road)>1 else np.nan
        if len(g) > 1:
            gp = g["Battery_Watt[W]"].to_numpy(float)
            raw_dt = g.timestamp_utc.diff().dt.total_seconds().fillna(0).to_numpy(float)[1:]
            gdt = np.where((raw_dt > 0) & (raw_dt <= 30), raw_dt, 0.0)
            actual_energy = float(np.sum((gp[1:]+gp[:-1])*0.5*gdt)/3600.0)
        else:
            actual_energy = np.nan
        result.append({
            "day": int(day), "logged_rows": int(len(g)), "upper_replans_30min": int(len(decision)),
            "distance_moved_km": (end_dist-start_dist)/1000.0,
            "logged_speed_median_kmh": float(moving["Speed[kmh]"].median()) if len(moving) else np.nan,
            "shadow_target_median_kmh": float(decision.upper_target_speed_kmh.median()) if len(decision) else np.nan,
            "shadow_target_p10_kmh": float(decision.upper_target_speed_kmh.quantile(.1)) if len(decision) else np.nan,
            "shadow_target_p90_kmh": float(decision.upper_target_speed_kmh.quantile(.9)) if len(decision) else np.nan,
            "road_log_span_h": duration_h, "observed_battery_energy_change_wh": actual_energy,
            "min_coulomb_counted_energy_wh": float(g.stored_energy_est_wh.min()),
            "below_15pct_energy_rows": int((g.stored_energy_est_wh < CAPACITY_WH*.15).sum()),
            "no_feasible_plan_decisions": int(decision.upper_solver_status.str.startswith("infeasible").sum()),
        })
    days = pd.DataFrame(result)
    days.to_csv(DAY_METRICS, index=False, encoding="utf-8-sig")
    sensitivity = pd.DataFrame(reserve_sensitivity_rows)
    sensitivity.to_csv(RESERVE_SENS, index=False, encoding="utf-8-sig")
    # Availability/leakage checks are deterministic and saved with the report.
    decision_rows = out[out.upper_replan_now]
    age = pd.to_numeric(decision_rows.forecast_age_at_decision_h, errors="coerce")
    summary = {
        "source_rows": int(len(out)), "days": int(out.day.nunique()),
        "forecast_runs_used": [x.isoformat() for x in unique_runs],
        "weather_vintage_model": MODEL_ID, "weather_temporal_publication_delay_h": 6,
        "weather_spatial_node_spacing_m": 50000, "weather_forecast_hours_per_run": 72,
        "forecast_asof_min_age_h": float(age.min()) if len(age) else None,
        "forecast_asof_violations_lt6h": int((age < 6).sum()) if len(age) else None,
        "upper_replan_interval_min": UPPER_REPLAN_MIN, "lower_input_sample_interval_s": LOWER_DT_S,
        "upper_grid_kmh": SPEEDS.tolist(), "upper_default_reserve_pct": RESERVE_DEFAULT*100,
        "battery_capacity_wh": CAPACITY_WH, "soc_initialization": init_meta,
        "causal_residual_update": {"model": "nested day-held-out median of previous six eligible 5-minute residual blocks",
                                   "bias_cap_w": 100.0, "applied_horizon_min": 5.0,
                                   "upper_application": "first five minutes of first route stage only; zero beyond validated block horizon",
                                   "source": str(RESIDUAL_BLOCK_MODEL.relative_to(ROOT))},
        "upper_optimality_scope": "MIP status and mip_gap are saved per solve; global optimum only when solver status is optimal. Only incumbents verified against the original reserve and deadline constraints are advisory-eligible. Infeasible/no-incumbent rows contain diagnostics only and no speed target. Discrete 60-78 km/h grid only; no continuous/real-vehicle guarantee",
        "lower_optimality_scope": "exact scalar quadratic optimum for one move-blocked acceleration over a 30 s horizon, subject to empirical acceleration and speed bounds",
        "actuator_mapping_available": False,
        "actual_race_winner_claimed": False,
        "day_metrics": days.to_dict(orient="records"),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, days, decision_rows, sensitivity)
    make_plot(out, days)
    print(json.dumps({"source_rows": len(out), "runs": len(unique_runs),
                      "upper_replans": int(out.upper_replan_now.sum()),
                      "forecast_asof_violations": summary["forecast_asof_violations_lt6h"],
                      "days": days.to_dict(orient="records")}, ensure_ascii=True, indent=2), flush=True)


def write_report(summary, days, decisions, sensitivity):
    rows = []
    for r in days.to_dict(orient="records"):
        rows.append(f"| {r['day']} | {r['logged_rows']:,} | {r['upper_replans_30min']} | {r['distance_moved_km']:.1f} | {r['logged_speed_median_kmh']:.1f} | {r['shadow_target_median_kmh']:.1f} | {r['no_feasible_plan_decisions']} | {r['observed_battery_energy_change_wh']:.0f} |")
    age = pd.to_numeric(decisions.forecast_age_at_decision_h, errors="coerce")
    solver_counts = decisions.upper_solver_status.value_counts().to_dict()
    exact_n = int(solver_counts.get("optimal_finite_grid_milp", 0))
    timed_n = int(solver_counts.get("feasible_time_limited_milp_not_proven_optimal", 0))
    advisory_statuses = {"optimal_finite_grid_milp", "feasible_time_limited_milp_not_proven_optimal"}
    no_advisory_n = int(sum(count for status, count in solver_counts.items() if status not in advisory_statuses))
    no_advisory_rows = decisions[~decisions.upper_solver_status.isin(advisory_statuses)]
    reasons = ", ".join(f"{k}: {v}" for k, v in no_advisory_rows.upper_solver_status.value_counts().items()) or "none"
    min_e_overrun = pd.to_numeric(no_advisory_rows.upper_min_energy_profile_deadline_overrun_s_diagnostic, errors="coerce")
    min_t_shortfall = pd.to_numeric(no_advisory_rows.upper_min_time_profile_reserve_shortfall_wh_diagnostic, errors="coerce")
    phase_slack = pd.to_numeric(no_advisory_rows.upper_phase_one_reserve_slack_wh_diagnostic, errors="coerce").dropna()
    min_energy_deadline_conflict_n = int((min_e_overrun > 1.0).sum())
    min_time_reserve_conflict_n = int((min_t_shortfall > 1.0).sum())
    gaps = pd.to_numeric(decisions.upper_mip_gap, errors="coerce").dropna()
    max_gap = float(gaps.max()) if len(gaps) else np.nan
    sens_rows = []
    if len(sensitivity):
        for r in sensitivity.to_dict(orient="records"):
            sens_rows.append(f"| {int(r['day'])} | {r['reserve_pct']:.0f}% | {r['target_speed_kmh']:.0f} | {r['status']} | {r['predicted_remaining_time_h']:.2f} | {r['predicted_remaining_net_energy_wh']:.0f} |")
    report = f"""# 上位・下位2層MPC：BWSC2025 as-of シャドーリプレイ

## 結果の扱い

2025年の走行ログを時刻順に再生し、各判断時点で6時間以上前に初期化された気象予報ランを上位計画に渡した**シャドー評価**です。車両/PV係数は既存のDay1–5ログで事後適合した値で、2025年当日に既知だった係数ではありません。気象予報と状態更新は時系列に沿っていますが、パラメータ学習まで厳密なas-liveではありません。実大会へ時刻を戻して実車運用した事実ではなく、実スロットル／PWM指令もログにないため、人間運転を上回った・実車で追従したという検証ではありません。

## 二層の入出力・目的・校正

| 層 | 入力 | 目的・制約 | 出力 | 校正状態 |
|---|---|---|---|---|
| 上位（30分ごと） | 現在距離・時刻、過去ログからの電池残量推定、次の既知Control Stopとその日の目的地、標高、当時利用可能なECMWF IFS予報 | 60–78 km/h候補を使い、17:00締切と15%暫定残量を満たす範囲で残り移動時間を最小化。ステージのネット電池Whは到達可能な予報時間内の最大値を制約に使用 | 次ステージの巡航目標速度と日末までの有限速度計画 | 混合整数線形計画。最適性は各行のsolver status / MIP gapに保存。有限候補集合で証明できるのはstatus=optimalの行だけ |
| 下位（5秒ごと） | 実ログ車速、上位目標速度、過去30分の速度変化履歴 | 30秒間一定加速度とする1変数MPC: `Σ(vₖ−vref)² + a² + 4(a−aprev)²`。60–78.2 km/h範囲と過去速度履歴からの加速度上下限 | 加速度要求、30秒先のモデル車速、機械的慣性電力増分`mva` | 凸二次目的の解析解。速度変化履歴は人間運転の代理値で、車両アクチュエータを校正していない。PWM／スロットル指令は出力しない |

**事後適合モデル:** 車両HV負荷は`67.6 + 0.918·rolling + 1.535·aero + 0.741·grade` W。これは10 km区間のStudent-t MLEであり、部品の真Crr/CdA/効率ではありません。PVは既存`faceted_body_poa`実効gain 1.068 m²（Day1–5の23時間ビン）を使用。日別ホールドアウトMAEは車両53.3 W（Day6は81.0 W）、PV50.5 Wです。電池は3011 Wh設計容量、選定33.164 Ah、実効パック抵抗0.488 Ω、最初の3分から選ぶ電圧・電流ロバストアンカーと以後のBattery_Watt積分で推定します。これは成分再生モデルに揃えた状態推定で、実SoC、セル温度、PWM→トルク/電力対応の校正値はありません。

## 予報ビンテージと時系列漏れ

- 使用したのはOpen-Meteo Single Runs APIのECMWF IFS単一ラン。各時刻で初期化から6時間以上たった最新ランのみを選び、気温・風・気圧・瞬時日射を取得しました。ルート上50 km間隔の地点を予報空間補間し、72時間先まで使います。
- 予報ラン初期時刻と利用可能時刻に関する未来漏れは判断94回で0件でした。一方、車両/PV係数は全Day1–5ログから後付け適合しており、初期日の係数学習に未来データが入る後知恵バイアスは残ります。予報履歴の因果性と係数学習の因果性を区別してください。
- ユーザーの指示に従い、nested day-held-outで選んだ過去6ブロック中央値の因果残差補正（±100 W）を、次の5分だけ最初のルートステージ電力量に適用しました。30秒以降で物理基準を改善しない5秒残差モデルや、全データ学習内予測は上位ホライズンに再帰適用していません。天候は計画の到達時間幅内で最大消費側を使います。既存の車両Day-LODO MAE 53.3 W、Day 6 81.0 W、PV日別MAE 50.5 Wの残差は残るため、計画達成は実車完走保証ではありません。
- 判断行の予報経過時間は中央値 {age.median():.1f} h、範囲 {age.min():.1f}–{age.max():.1f} h、6時間未満の違反 {int((age < 6).sum())} 件です。リプレイCSVの`forecast_run_init_utc`を各出力の監査に使えます。
- 将来の実測気象を上位計画へ入れず、現在の予報ランの中から候補到達時刻全体を評価しました。トラブル停止は発生した時点で下位出力を止めます。将来の故障・道路工事を事前知識にはしません。Control Stopと17時締切はスケジュール制約として扱います。
- 初期SoCは各日の最初の3分から、電圧70–115 Vかつ|電流|≤5 Aを満たす中央サンプルを選び、実効抵抗0.488 Ωで負荷電圧を補正して電圧-SOC表に照合します。以後はBattery_Wattを積分しました。SOC依存抵抗マップはDay6ホールドアウトで定抵抗より悪化したため主推定には採用せず、別候補として保持します。Day 6の3,485秒欠測区間は積算から除外しています。OCVの温度・休止依存が未校正なので、このSoCは概算です。

## 日別シャドー出力と実績

| 日 | ログ行 | 上位解き直し | 実走距離 km | 人間ログ速度中央値 km/h | MPC目標中央値 km/h | 予備制約が実行不能な判断 | 実測電池エネルギー変化 Wh |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

実測列は人間運転時の結果、MPC列は当時の予報に基づく提案です。MPC速度を車両へ適用した場合の時間・電池消費は測定されていません。人間ログ速度中央値より目標速度が低い日もあり、全日で人間を上回る根拠は得られていません。モデルの10 km日別ホールドアウト電力MAEは平均53.3 W、Day 6は81.0 Wで、隣接速度の選択差を保証する精度ではありません。94回中、{exact_n}回はMILP status=optimal、{timed_n}回は時間制限付き実行可能解（最大相対ギャップ {max_gap:.2g}）、{no_advisory_n}回は元のハード制約を満たす解が得られず、速度助言を出しません（{reasons}）。その判断では、最小エネルギー診断でも締切超過が確認された回数は{min_energy_deadline_conflict_n}、最短時間診断でも残量不足が確認された回数は{min_time_reserve_conflict_n}です。残量スラック付きPhase-I診断が解けた回数は{len(phase_slack)}、その不足量中央値は{phase_slack.median():.0f} Whです。Phase-I解は不足Whの診断に限り、指令候補には使いません。

日別の実測電池エネルギー変化は、連続ログ時刻差が30秒以内の区間のみ積分しています。Day1–5は独立の電力量整理表と0.01 Wh以内で一致します。Day6は3,485秒のログ欠落があり、その前後サンプルを台形積分でつなぐと未観測区間を作ってしまうため、リプレイでは除外しています。

## 現時点の結論

上位層の最適性は各行の`upper_solver_status`と`upper_mip_gap`で判定します。`optimal_finite_grid_milp`行だけが指定した有限問題の全体最適です。時間制限時も元のハード制約を通過したincumbentだけが助言対象です。制約不能、タイムアウトでincumbentなし、またはモデル/予報入力不正の判断は目標速度を空欄にし、CSVに`upper_no_advisory_reason`と、最小エネルギー/最短時間プロファイルの残量不足・締切超過診断を記録します。これらは「実車がその時点で走行不能だった」という意味ではなく、入力推定と指定した15%予備残量・17:00締切・速度候補集合の下で、モデルが両制約を満たす計画を証明できなかったという意味です。速度を連続値にした場合、実際の気象・故障、未同定のバッテリー・モーター・アクチュエータを含む実車全体の大域最適や人間を上回る結果は保証できません。実走行ログから実車閉ループ追従を検証できないため、出力はGrafana候補のシャドー提案です。

事故、パンク、MPPT故障、退役などは現在行のイベントフラグが立った時点で助言を停止します。記録された実際の発生時刻・位置は人間運転軌跡上のものです。速度変更後にも発生時刻・位置が不変だと物理シミュレーションしたわけではなく、実際の未来気象を測ったデータもありません。したがって「同じ事故・天候の下でMPCが実際に人間を上回る」反実仮想走行は今回のログから検証できていません。

残量の合意が未確定なので15%を暫定既定値としました。0/10/15/20%感度表を作成しています。計画不能となった区間は最小速度などの代替速度を助言せず、`upper_solver_status`と診断欄に原因を記録します。Phase-Iは残量不足量を推定しますが、締切制約は維持します。これは完走可能性の証明ではなく、競技運用責任者が状態推定・残量目標・締切/停止条件を再評価するための診断です。

初回判断での目標残量感度（0/10/15/20%）は次の通りです。これは残量値だけを変えた同一状態の再求解であり、別々の走行結果ではありません。

| 日 | 終端残量 | 次区間目標 km/h | 解の状態 | 予測残り時間 h | 予測ネット電力量 Wh |
|---:|---:|---:|---|---:|---:|
{chr(10).join(sens_rows)}

## 再現コマンドと出力

`python scripts/replay_hierarchical_mpc_2025.py`
`python scripts/evaluate_mpc_effective_intervals.py`

- 行ごとの提案と予報ラン監査: `data/processed/mpc_2025_as_live_replay.csv`
- 日別集計: `reports/mpc_2025_day_comparison.csv`
- 初回終端残量感度: `reports/mpc_2025_reserve_sensitivity.csv`
- 数値メタデータ: `reports/mpc_2025_replay_summary.json`
- 図: `reports/hierarchical_mpc_2025_replay.png`
- イベントなし・上位最適解・残量余裕ありの同一距離区間比較: [`mpc_2025_effective_interval_assessment.md`](mpc_2025_effective_interval_assessment.md)。正の時間差は、下位加速度モデルを含む予測上の区間短縮を示します。実車性能の実測ではありません。
- 区間別時間・全残差モデルの電力誤差: `reports/mpc_2025_effective_intervals.csv`
- 各モデルの適用先・検証状態: `reports/mpc_2025_model_usage_audit.csv`

## 根拠と限界

Open-MeteoはECMWF IFS HRESの単一初期化ランを過去に遡って指定でき、同じ予報の履歴再現にはHistorical Forecastの連続接続データよりSingle Runsが適します。予報ランが公開されるまでに数時間の遅れがあるため、ここでは保守的に6時間を置きました。詳細は[Single Runs API](https://open-meteo.com/en/docs/single-runs-api)と[Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api)。上位計画は各距離ステージで速度候補を選ぶ線形制約付き混合整数問題として[scipy.optimize.milp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)で解き、status=0を最適解、status=1を時間制限として扱います。

車両出力・PV出力の係数と電池容量の値は既存の[車両モデル要約](vehicle_power_model_summary.md)、[PVモデル要約](pv_fit_summary.md)、[モデル複雑度判断](model_complexity_decisions.md)を使用しています。BWSC2025規則にある60 km/h最低速度、Control Stop、17:00道路走行締切は[規則読了メモ](BWSC2025_regulatory_context.md)に保存済みです。MPCの再計画は制約付き最適化の標準構成に沿いますが、MPC文献はYATAの未測定アクチュエータ・誤差を同定するものではありません。
"""
    REPORT.write_text(report, encoding="utf-8")


def make_plot(out: pd.DataFrame, days: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
    for day, g in out.groupby("day"):
        road = g[g.event_phase.eq("road_day")]
        if road.empty:
            continue
        h = (road.timestamp_local-road.timestamp_local.iloc[0]).dt.total_seconds()/3600
        axes[0].plot(h, road["Speed[kmh]"], lw=.7, alpha=.7, label=f"Day {day} logged")
        axes[0].step(h, road.upper_target_speed_kmh, where="post", lw=1.5, label=f"Day {day} upper MPC")
        axes[1].plot(h, road.stored_energy_est_wh, lw=1.0, label=f"Day {day} causal coulomb estimate")
    axes[0].set_ylabel("Speed [km/h]"); axes[0].set_ylim(0, 90); axes[0].grid(True, alpha=.25); axes[0].legend(ncol=3, fontsize=8)
    axes[1].axhline(CAPACITY_WH*RESERVE_DEFAULT, color="red", ls="--", label="15% nominal reserve")
    axes[1].set_xlabel("Hours since each logged road-day start"); axes[1].set_ylabel("Estimated stored energy [Wh]"); axes[1].grid(True, alpha=.25); axes[1].legend(ncol=3, fontsize=8)
    fig.suptitle("BWSC2025 two-layer MPC shadow replay: logged human trace vs advisory target")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    import re
    main()
