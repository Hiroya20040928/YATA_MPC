#!/usr/bin/env python3
"""Local HTTP input port for 1 Hz YATA telemetry and route weather forecasts."""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from uuid import uuid4
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from yata_mpc.telemetry import LatestInputPort, TelemetryError, _utc  # noqa: E402
from yata_mpc.controller import LowerSpeedMPC, UpperReplanGate  # noqa: E402
from yata_mpc.battery_state import BatteryEnergyEstimator  # noqa: E402
import replay_hierarchical_mpc_2025 as replay  # noqa: E402

PORT = LatestInputPort()
BATTERY = BatteryEnergyEstimator(ROOT)
UPPER_PLAN: dict | None = None
UPPER_SCHEDULE: dict | None = None
UPPER_PLAN_STATE: dict = {"status": "upper_plan_missing", "advisory_eligible": False}
LOWER_CONTROLLER: LowerSpeedMPC | None = None
LAST_ADVICE: dict | None = None
PLAN_LOCK = threading.RLock()
UPPER_GATE = UpperReplanGate(ROOT / "models/mpc_operating_config.json")
UPPER_WORKER_BUSY = False
UPPER_WORKER_LOCK = threading.Lock()
UPPER_CONFIG = json.loads((ROOT / "models/mpc_operating_config.json").read_text(encoding="utf-8"))
UPPER_TRIGGER_COOLDOWN_UNTIL = 0.0


class FittedModelPreview:
    """Evaluate the fitted vehicle and fixed-body PV models at the newest point."""

    def __init__(self, route_path: Path):
        self.route = pd.read_csv(route_path).sort_values("distance_m").reset_index(drop=True)
        self.road = json.loads((ROOT / "models/vehicle_power_model_fit.json").read_text(encoding="utf-8"))
        self.pv = json.loads((ROOT / "models/pv_model_fit.json").read_text(encoding="utf-8"))
        self.mass = 314.0
        self.cda = 0.08876
        self.crr = 0.0035

    def route_at(self, distance_m: float):
        distances = self.route.distance_m.to_numpy(float)
        z = self.route.elev_dem_m.to_numpy(float)
        lat = self.route.lat.to_numpy(float)
        lon = self.route.lon.to_numpy(float)
        d = float(np.clip(distance_m, distances[0], distances[-1]))
        lo, hi = max(distances[0], d - 250.0), min(distances[-1], d + 250.0)
        elev_lo, elev_hi = float(np.interp(lo, self.route.distance_m, z)), float(np.interp(hi, self.route.distance_m, z))
        lat_lo, lon_lo = float(np.interp(lo, self.route.distance_m, lat)), float(np.interp(lo, self.route.distance_m, lon))
        lat_hi, lon_hi = float(np.interp(hi, self.route.distance_m, lat)), float(np.interp(hi, self.route.distance_m, lon))
        return {"latitude": float(np.interp(d, self.route.distance_m, lat)),
            "longitude": float(np.interp(d, self.route.distance_m, lon)),
            "grade": (elev_hi - elev_lo) / max(1.0, hi - lo),
            "heading_deg": float(replay.bearing_deg(lat_lo, lon_lo, lat_hi, lon_hi))}

    def predict(self, snapshot: dict) -> dict:
        wx = snapshot["weather"]
        pos = snapshot["position"]
        geo = self.route_at(snapshot["route_distance_m"])
        pars = self.road["fit_parameters"]["parameters"]
        v = float(snapshot["speed_kmh"]) / 3.6
        temp_k = float(wx["temperature_2m_c"]) + 273.15
        rho = float(wx["surface_pressure_pa"]) / (287.05 * temp_k)
        headwind = float(wx["wind_speed_10m_ms"]) * math.cos(math.radians(
            float(wx["wind_direction_10m_deg"]) - geo["heading_deg"]))
        relative_air = v + headwind
        grade = geo["grade"]
        roll_w = self.mass * 9.80665 * self.crr * math.cos(math.atan(grade)) * v
        aero_w = 0.5 * rho * self.cda * relative_air * abs(relative_air) * v
        grade_w = self.mass * 9.80665 * grade * v
        vehicle_w = (float(pars["p_aux_w"]) + float(pars["beta_roll"]) * roll_w
            + float(pars["beta_aero"]) * aero_w + float(pars["beta_grade"]) * grade_w)
        forecast = {"shortwave_radiation_instant": np.array([max(0.0, wx["ghi_w_m2"])]),
            "direct_normal_irradiance_instant": np.array([max(0.0, wx["dni_w_m2"])]),
            "diffuse_radiation_instant": np.array([max(0.0, wx["dhi_w_m2"])])}
        t = pd.DatetimeIndex([pd.Timestamp(snapshot["timestamp_utc"])])
        pv_w = float(replay.pv_power(forecast, t, float(pos["latitude_deg"]),
            float(pos["longitude_deg"]), geo["heading_deg"], self.pv)[0])
        measured_pv_w = float(snapshot["pv_power_w"])
        return {"predicted_vehicle_load_w": vehicle_w,
            "predicted_pv_w": pv_w,
            "measured_pv_w": measured_pv_w,
            "predicted_battery_power_required_w": vehicle_w - pv_w,
            "measured_battery_power_w": float(snapshot["hv_bus"]["battery_power_w"]),
            "battery_power_residual_w": float(snapshot["hv_bus"]["battery_power_w"]) - (vehicle_w - pv_w),
            "route_grade_pct": 100.0 * grade, "route_heading_deg": geo["heading_deg"],
            "vehicle_model_student_t_scale_w": float(self.road["fit_parameters"]["parameters"]["student_t_scale_w"]),
            "pv_model_student_t_scale_w": float(self.pv["model_candidates"]["faceted_body_poa"]["scale_w"]),
            "prediction_scope": "fitted 2025 fixed-body PV + effective road-load model; advisory preview"}


PREVIEW = FittedModelPreview(ROOT / "data/input/route/route_100m_dem.csv")


def _forecast_model_frame(grid: dict) -> pd.DataFrame:
    points = grid.get("points", [])
    if not isinstance(points, list) or len(points) < 2:
        raise TelemetryError("full route/weather grid is missing")
    raw = pd.DataFrame(points)
    required = {"distance_m", "time_utc", "temperature_2m_c", "wind_speed_10m_ms",
        "wind_direction_10m_deg", "surface_pressure_pa", "ghi_w_m2", "dni_w_m2", "dhi_w_m2"}
    if not required.issubset(raw.columns):
        raise TelemetryError(f"route/weather grid missing fields: {sorted(required - set(raw.columns))}")
    return pd.DataFrame({
        "node_distance_m": pd.to_numeric(raw.distance_m, errors="coerce"),
        "time_utc": pd.to_datetime(raw.time_utc, utc=True, errors="coerce"),
        "temperature_2m": pd.to_numeric(raw.temperature_2m_c, errors="coerce"),
        "wind_speed_10m": pd.to_numeric(raw.wind_speed_10m_ms, errors="coerce") * 3.6,
        "wind_direction_10m": pd.to_numeric(raw.wind_direction_10m_deg, errors="coerce"),
        "surface_pressure": pd.to_numeric(raw.surface_pressure_pa, errors="coerce") / 100.0,
        "shortwave_radiation_instant": pd.to_numeric(raw.ghi_w_m2, errors="coerce"),
        "direct_normal_irradiance_instant": pd.to_numeric(raw.dni_w_m2, errors="coerce"),
        "diffuse_radiation_instant": pd.to_numeric(raw.dhi_w_m2, errors="coerce"),
    })


def _solve_live_upper(snapshot: dict, schedule: dict, grid: dict, battery: dict) -> dict:
    if snapshot["position"].get("route_id") != schedule["route_id"] or grid.get("route_id") != schedule["route_id"]:
        return {"status": "route_id_mismatch", "advisory_eligible": False,
                "diagnostic": {"telemetry_route_id": snapshot["position"].get("route_id"),
                    "forecast_route_id": grid.get("route_id"), "schedule_route_id": schedule["route_id"]}}
    if not battery.get("ready"):
        return {"status": "battery_state_uninitialized", "advisory_eligible": False,
                "diagnostic": battery}
    energy_wh = battery.get("energy_lower_2sigma_wh")
    if energy_wh is None or not math.isfinite(float(energy_wh)):
        return {"status": "battery_energy_bound_invalid", "advisory_eligible": False}
    frame = _forecast_model_frame(grid)
    forecast = replay.Forecast(frame)
    stops = [float(x["distance_m"]) for x in schedule["control_stops"]]
    dwell = {float(x["distance_m"]): float(x["dwell_s"]) for x in schedule["control_stops"]}
    now = pd.Timestamp(snapshot["timestamp_utc"])
    result = replay.solve_upper(forecast, float(snapshot["route_distance_m"]),
        float(schedule["day_end_distance_m"]), stops, now, float(energy_wh),
        float(schedule["reserve_pct"])/100.0, PREVIEW.route, PREVIEW.road,
        PREVIEW.pv, pd.Timestamp(schedule["deadline_utc"]), residual_bias_w=0.0,
        solver_time_limit_s=float(UPPER_CONFIG.get("upper_solver_time_limit_s", 5.0)),
        stop_dwell_s_by_distance=dwell)
    return result


def _upper_worker(snapshot: dict, schedule: dict, grid: dict, battery: dict):
    global UPPER_PLAN, UPPER_PLAN_STATE, UPPER_WORKER_BUSY, UPPER_GATE
    try:
        result = _solve_live_upper(snapshot, schedule, grid, battery)
        with PLAN_LOCK:
            if UPPER_SCHEDULE is None or UPPER_SCHEDULE["schedule_id"] != schedule["schedule_id"]:
                return
            if not result.get("advisory_eligible"):
                UPPER_PLAN = None
                UPPER_GATE.failed()
                UPPER_PLAN_STATE = {"status": result.get("status", "upper_solver_failed"),
                    "advisory_eligible": False, "diagnostic": result.get("diagnostic"),
                    "solver_message": result.get("solver_message"),
                    "replan_reason": UPPER_PLAN_STATE.get("replan_reason")}
                return
            issued = pd.Timestamp(snapshot["timestamp_utc"])
            valid_until = min(issued + pd.Timedelta(seconds=float(UPPER_CONFIG["upper_replan_s"])),
                              pd.Timestamp(schedule["valid_until_utc"]))
            plan = {"schema_version": "1.0", "plan_id": f"live-{uuid4().hex[:16]}",
                "issued_at_utc": issued.isoformat(), "valid_until_utc": valid_until.isoformat(),
                "solver_status": result["status"], "advisory_eligible": True,
                "segments": result["planned_segments"],
                "energy_budget_wh": result.get("energy_budget_wh"),
                "predicted_remaining_wh_robust": result.get("predicted_remaining_wh_robust"),
                "energy_margin_wh": result.get("energy_margin_wh"),
                "deadline_margin_s": result.get("deadline_margin_s"),
                "reserve_pct": float(schedule["reserve_pct"]),
                "schedule_id": schedule["schedule_id"]}
            UPPER_PLAN = plan
            UPPER_PLAN_STATE = {"status": result["status"], "advisory_eligible": True,
                "replan_reason": UPPER_PLAN_STATE.get("replan_reason"),
                "stage_count": result.get("stage_count"), "mip_gap": result.get("mip_gap"),
                "energy_margin_wh": result.get("energy_margin_wh"),
                "deadline_margin_s": result.get("deadline_margin_s"),
                "solver_message": result.get("solver_message")}
    except Exception as exc:
        with PLAN_LOCK:
            UPPER_PLAN = None
            UPPER_GATE.failed()
            UPPER_PLAN_STATE = {"status": "upper_solver_runtime_error",
                "advisory_eligible": False, "detail": str(exc)}
    finally:
        with UPPER_WORKER_LOCK:
            UPPER_WORKER_BUSY = False


def _maybe_start_upper_solve(snapshot: dict) -> dict:
    global UPPER_WORKER_BUSY, UPPER_TRIGGER_COOLDOWN_UNTIL, UPPER_PLAN_STATE, UPPER_PLAN
    with PLAN_LOCK:
        schedule = UPPER_SCHEDULE
        if schedule is None:
            return {"status": "race_schedule_missing", "advisory_eligible": False}
        now = pd.Timestamp(snapshot["timestamp_utc"]).to_pydatetime()
        if snapshot["position"].get("route_id") != schedule["route_id"]:
            UPPER_PLAN = None
            UPPER_PLAN_STATE = {"status": "race_schedule_route_id_mismatch", "advisory_eligible": False}
            return dict(UPPER_PLAN_STATE)
        valid_from, valid_until = _utc(schedule["valid_from_utc"]), _utc(schedule["valid_until_utc"])
        if now < valid_from or now > valid_until:
            UPPER_PLAN = None
            UPPER_PLAN_STATE = {"status": "race_schedule_not_current", "advisory_eligible": False}
            return {"status": "race_schedule_not_current", "advisory_eligible": False}
        selected = None
        if UPPER_PLAN:
            matches = [seg for seg in UPPER_PLAN["segments"]
                if float(seg["start_distance_m"]) <= float(snapshot["route_distance_m"]) < float(seg["end_distance_m"])]
            selected = matches[0] if matches else None
        tracking_error = (float(snapshot["speed_kmh"]) - float(selected["target_speed_kmh"])) if selected else 0.0
        battery = BATTERY.snapshot()
        remaining_plan_wh = None
        if UPPER_PLAN:
            dnow = float(snapshot["route_distance_m"])
            remaining_parts = []
            for seg in UPPER_PLAN["segments"]:
                start, end = float(seg["start_distance_m"]), float(seg["end_distance_m"])
                if end <= dnow:
                    continue
                if "predicted_net_energy_wh" not in seg:
                    remaining_parts = []
                    break
                fraction_left = 1.0 if dnow <= start else (end-dnow)/(end-start)
                remaining_parts.append(float(seg["predicted_net_energy_wh"])*fraction_left)
            if remaining_parts:
                remaining_plan_wh = float(sum(remaining_parts))
            elif dnow < float(schedule["day_end_distance_m"]):
                remaining_plan_wh = math.inf
        reserve_floor_wh = replay.CAPACITY_WH*float(schedule["reserve_pct"])/100.0
        lower_energy = battery.get("energy_lower_2sigma_wh")
        reserve_breached = (lower_energy is not None and remaining_plan_wh is not None and
            float(lower_energy) < reserve_floor_wh + remaining_plan_wh - 1.0)
        due, reason = UPPER_GATE.due(now, float(snapshot["route_distance_m"]),
            snapshot.get("forecast_issue_time_utc"), "road_day", tracking_error, reserve_breached)
        if not due:
            return {"status": UPPER_PLAN_STATE.get("status", "upper_plan_missing"),
                "advisory_eligible": UPPER_PLAN is not None, "replan_reason": "not_due"}
        if time.monotonic() < UPPER_TRIGGER_COOLDOWN_UNTIL:
            return {"status": UPPER_PLAN_STATE.get("status", "upper_mpc_replan_in_progress"),
                "advisory_eligible": UPPER_PLAN is not None, "replan_reason": "trigger_cooldown"}
        grid = PORT.forecast_grid()
        if grid is None:
            return {"status": "route_forecast_missing", "advisory_eligible": False}
        with UPPER_WORKER_LOCK:
            if UPPER_WORKER_BUSY:
                return {"status": "upper_mpc_replan_in_progress", "advisory_eligible": False,
                        "replan_reason": reason}
            UPPER_WORKER_BUSY = True
        UPPER_GATE.attempted(now, float(snapshot["route_distance_m"]), snapshot.get("forecast_issue_time_utc"))
        UPPER_TRIGGER_COOLDOWN_UNTIL = time.monotonic() + 30.0
        # Do not continue advising from a trajectory optimized against older
        # state or forecast inputs while a due receding-horizon solve runs.
        UPPER_PLAN = None
        UPPER_PLAN_STATE = {"status": "upper_mpc_replan_in_progress", "advisory_eligible": False,
                            "replan_reason": reason}
        worker = threading.Thread(target=_upper_worker,
            args=(dict(snapshot), dict(schedule), grid, battery), daemon=True)
        worker.start()
        return dict(UPPER_PLAN_STATE)


class Handler(BaseHTTPRequestHandler):
    server_version = "YATA-MPC-Input/1.0"

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args), flush=True)

    def _reply(self, status: int, body: dict):
        raw = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        global UPPER_PLAN, UPPER_PLAN_STATE, UPPER_GATE, LOWER_CONTROLLER, LAST_ADVICE
        if self.path == "/api/health":
            with PLAN_LOCK:
                upper_state = dict(UPPER_PLAN_STATE)
                has_schedule = UPPER_SCHEDULE is not None
            return self._reply(200, {"ok": True, "service": "yata-mpc-live-input",
                "schema_version": "1.0", "upper_planner": upper_state,
                "active_race_schedule": has_schedule})
        if self.path == "/v1/telemetry/latest":
            snap = PORT.snapshot()
            return self._reply(200 if snap.get("ready") else 503, snap)
        if self.path == "/v1/controller/input":
            snap = PORT.snapshot()
            # This is the stable pull contract for the controller. It contains
            # the newest accepted measured state and weather at its current
            # route distance and timestamp.
            if snap.get("ready"):
                try:
                    snap["predictions"] = PREVIEW.predict(snap)
                    snap["battery_state"] = BATTERY.snapshot()
                except Exception as exc:
                    return self._reply(503, {"ready": False, "reason": "fitted_model_preview_failed", "detail": str(exc)})
            return self._reply(200 if snap.get("ready") else 503, snap)
        if self.path == "/v1/controller/advice":
            snap = PORT.snapshot()
            if not snap.get("ready"):
                if snap.get("reason") in {"event_phase_suppresses_mpc", "event_flag_suppresses_mpc"}:
                    with PLAN_LOCK:
                        UPPER_PLAN = None
                        UPPER_PLAN_STATE = {"status": snap["reason"], "advisory_eligible": False}
                        UPPER_GATE = UpperReplanGate(ROOT / "models/mpc_operating_config.json")
                return self._reply(503, snap)
            planner_state = _maybe_start_upper_solve(snap)
            with PLAN_LOCK:
                plan = UPPER_PLAN
                if plan is None:
                    return self._reply(503, {"ready": False,
                        "reason": planner_state.get("status", "upper_plan_missing"),
                        "upper_planner": planner_state})
                now = pd.Timestamp(snap["timestamp_utc"]).to_pydatetime()
                if now < _utc(plan["issued_at_utc"]) or now > _utc(plan["valid_until_utc"]):
                    return self._reply(503, {"ready": False, "reason": "upper_plan_not_current"})
                candidates = [x for x in plan["segments"] if x["start_distance_m"] <= snap["route_distance_m"] < x["end_distance_m"]]
                if not candidates:
                    return self._reply(503, {"ready": False, "reason": "upper_plan_route_gap"})
                seg = candidates[0]
                if plan["solver_status"] not in {"optimal_finite_grid_milp", "feasible_time_limited_milp_not_proven_optimal"}:
                    return self._reply(503, {"ready": False, "reason": "upper_plan_solver_not_advisory_eligible",
                        "solver_status": plan["solver_status"]})
                if LOWER_CONTROLLER is None:
                    LOWER_CONTROLLER = LowerSpeedMPC(ROOT / "models/mpc_operating_config.json",
                        ROOT / "models/m2096_inferred_actuator_envelope.csv")
                advice_key = (snap["sequence"], plan["plan_id"], snap.get("forecast_issue_time_utc"))
                if LAST_ADVICE is None or LAST_ADVICE.get("key") != advice_key:
                    predictions = PREVIEW.predict(snap)
                    if snap.get("previous_sample_gap_s") is not None and snap["previous_sample_gap_s"] > 2.5:
                        LOWER_CONTROLLER.reset()
                    lower = LOWER_CONTROLLER.step(snap["speed_kmh"], seg["target_speed_kmh"])
                    LAST_ADVICE = {"key": advice_key, "sequence": snap["sequence"], "lower": lower, "predictions": predictions,
                        "upper_plan_id": plan["plan_id"], "upper_solver_status": plan["solver_status"]}
                if not LAST_ADVICE["lower"].get("advisory_eligible", False):
                    return self._reply(503, {"ready": False,
                        "reason": "lower_mpc_infeasible_no_advisory",
                        "timestamp_utc": snap["timestamp_utc"],
                        "route_distance_m": snap["route_distance_m"],
                        "event": snap["event"], "upper_plan_state": dict(UPPER_PLAN_STATE), **LAST_ADVICE,
                        "battery_state": BATTERY.snapshot()})
                response = {"ready": True, "timestamp_utc": snap["timestamp_utc"],
                    "route_distance_m": snap["route_distance_m"], "event": snap["event"],
                    "upper_plan_state": dict(UPPER_PLAN_STATE), **LAST_ADVICE}
                response["battery_state"] = BATTERY.snapshot()
                return self._reply(200, response)
        return self._reply(404, {"error": "not_found"})

    def do_POST(self):
        if self.path not in {"/v1/telemetry", "/v1/forecast/route", "/v1/race-schedule", "/v1/upper-plan"}:
            return self._reply(404, {"error": "not_found"})
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 2_000_000:
            return self._reply(413, {"error": "body_size_invalid"})
        try:
            payload = json.loads(self.rfile.read(length))
            if self.path == "/v1/telemetry":
                result = PORT.ingest_telemetry(payload)
                result["battery_state"] = BATTERY.ingest(payload)
            elif self.path == "/v1/forecast/route":
                result = PORT.ingest_route_weather(payload)
            elif self.path == "/v1/race-schedule":
                result = validate_race_schedule(payload)
            else:
                result = validate_upper_plan(payload)
            return self._reply(202, result)
        except (json.JSONDecodeError, TelemetryError) as exc:
            return self._reply(400, {"error": "validation_error", "detail": str(exc)})
        except Exception as exc:
            return self._reply(500, {"error": "internal_error", "detail": str(exc)})


def validate_race_schedule(schedule: dict) -> dict:
    global UPPER_SCHEDULE, UPPER_PLAN, UPPER_PLAN_STATE, UPPER_GATE, UPPER_TRIGGER_COOLDOWN_UNTIL
    required = {"schema_version", "schedule_id", "route_id", "valid_from_utc", "valid_until_utc",
                "day_end_distance_m", "deadline_utc", "reserve_pct", "control_stops"}
    if not isinstance(schedule, dict) or not required.issubset(schedule):
        raise TelemetryError("race schedule requires schema_version, identifiers, validity, day endpoint, deadline, reserve_pct, and control_stops")
    if schedule["schema_version"] != "1.0":
        raise TelemetryError("race schedule schema_version must be '1.0'")
    if not all(isinstance(schedule[k], str) and schedule[k].strip() for k in ("schedule_id", "route_id")):
        raise TelemetryError("schedule_id and route_id must be non-empty strings")
    valid_from = _utc(schedule["valid_from_utc"])
    valid_until = _utc(schedule["valid_until_utc"])
    deadline = _utc(schedule["deadline_utc"])
    if valid_until <= valid_from or (valid_until-valid_from).total_seconds() > 36*3600:
        raise TelemetryError("race schedule validity must be positive and at most 36 hours")
    if not valid_from < deadline <= valid_until:
        raise TelemetryError("deadline_utc must be after valid_from and no later than valid_until")
    try:
        day_end = float(schedule["day_end_distance_m"])
        reserve_pct = float(schedule["reserve_pct"])
    except (TypeError, ValueError) as exc:
        raise TelemetryError("day_end_distance_m and reserve_pct must be numeric") from exc
    route_end = float(PREVIEW.route.distance_m.max())
    if not math.isfinite(day_end) or not 0 < day_end <= route_end:
        raise TelemetryError(f"day_end_distance_m must be within loaded route [0, {route_end:g}]")
    if not math.isfinite(reserve_pct) or not 0 <= reserve_pct <= 100:
        raise TelemetryError("reserve_pct must be in [0, 100]")
    if not isinstance(schedule["control_stops"], list):
        raise TelemetryError("control_stops must be an array")
    stops = []
    for item in schedule["control_stops"]:
        if not isinstance(item, dict) or not {"distance_m", "dwell_s"}.issubset(item):
            raise TelemetryError("each control stop requires distance_m and dwell_s")
        try:
            distance, dwell = float(item["distance_m"]), float(item["dwell_s"])
        except (TypeError, ValueError) as exc:
            raise TelemetryError("control stop distance_m and dwell_s must be numeric") from exc
        if not math.isfinite(distance) or not 0 < distance < day_end:
            raise TelemetryError("control stop distance must be finite and strictly before day_end_distance_m")
        if not math.isfinite(dwell) or not 0 <= dwell <= 7200:
            raise TelemetryError("control stop dwell_s must be in [0, 7200]")
        stops.append({"distance_m": distance, "dwell_s": dwell})
    stops.sort(key=lambda x: x["distance_m"])
    if any(a["distance_m"] == b["distance_m"] for a, b in zip(stops, stops[1:])):
        raise TelemetryError("control stop distances must be unique")
    normalized = {**schedule, "valid_from_utc": valid_from.isoformat(),
        "valid_until_utc": valid_until.isoformat(), "deadline_utc": deadline.isoformat(),
        "day_end_distance_m": day_end, "reserve_pct": reserve_pct, "control_stops": stops}
    with PLAN_LOCK:
        UPPER_SCHEDULE = normalized
        UPPER_PLAN = None
        UPPER_PLAN_STATE = {"status": "race_schedule_accepted_waiting_for_live_replan",
                            "advisory_eligible": False}
        UPPER_GATE = UpperReplanGate(ROOT / "models/mpc_operating_config.json")
        UPPER_TRIGGER_COOLDOWN_UNTIL = 0.0
    return {"accepted": True, "schedule_id": normalized["schedule_id"],
        "route_id": normalized["route_id"], "control_stops": len(stops),
        "valid_from_utc": normalized["valid_from_utc"], "valid_until_utc": normalized["valid_until_utc"]}


def validate_upper_plan(plan: dict) -> dict:
    global UPPER_PLAN, UPPER_PLAN_STATE
    eligible_statuses = {"optimal_finite_grid_milp", "feasible_time_limited_milp_not_proven_optimal"}
    if isinstance(plan, dict) and (plan.get("solver_status") not in eligible_statuses
            or plan.get("advisory_eligible") is not True):
        with PLAN_LOCK:
            UPPER_PLAN = None
            UPPER_PLAN_STATE = {"status": plan.get("solver_status", "invalid_external_plan"),
                                "advisory_eligible": False}
        raise TelemetryError("only an explicitly eligible, constraint-feasible MPC solution can be accepted")
    required = {"schema_version", "schedule_id", "plan_id", "issued_at_utc", "valid_until_utc",
                "solver_status", "advisory_eligible", "energy_budget_wh",
                "predicted_remaining_wh_robust", "energy_margin_wh", "deadline_margin_s",
                "reserve_pct", "segments"}
    if not isinstance(plan, dict) or not required.issubset(plan):
        raise TelemetryError("upper plan requires schema_version, plan_id, issue/expiry, solver status, eligibility, and segments")
    if plan["schema_version"] != "1.0":
        raise TelemetryError("upper plan schema_version must be '1.0'")
    if UPPER_SCHEDULE is None or plan["schedule_id"] != UPPER_SCHEDULE["schedule_id"]:
        raise TelemetryError("upper plan schedule_id must match the currently active race schedule")
    issued, valid = _utc(plan["issued_at_utc"]), _utc(plan["valid_until_utc"])
    if valid <= issued or (valid - issued).total_seconds() > 300:
        raise TelemetryError("upper plan validity must be positive and at most 300 seconds")
    if issued < _utc(UPPER_SCHEDULE["valid_from_utc"]) or valid > _utc(UPPER_SCHEDULE["valid_until_utc"]):
        raise TelemetryError("upper plan must fit within the active race-schedule validity")
    numeric_plan_values = [float(plan[k]) for k in ("energy_budget_wh", "predicted_remaining_wh_robust",
        "energy_margin_wh", "deadline_margin_s", "reserve_pct")]
    if not all(map(math.isfinite, numeric_plan_values)):
        raise TelemetryError("upper plan energy/time margins and reserve must be finite")
    energy_budget, predicted_wh, energy_margin, deadline_margin, plan_reserve = numeric_plan_values
    if energy_budget < 0 or not 0 <= plan_reserve <= 100 or energy_margin < -1.0 or deadline_margin < -1.0:
        with PLAN_LOCK:
            UPPER_PLAN = None
            UPPER_PLAN_STATE = {"status": "external_plan_failed_hard_constraint_check", "advisory_eligible": False}
        raise TelemetryError("upper plan must have nonnegative energy budget and pass energy/deadline constraints")
    if abs(plan_reserve-float(UPPER_SCHEDULE["reserve_pct"])) > 1e-6:
        raise TelemetryError("upper plan reserve_pct must match the active race schedule")
    if abs((energy_budget-predicted_wh)-energy_margin) > 1.0:
        with PLAN_LOCK:
            UPPER_PLAN = None
            UPPER_PLAN_STATE = {"status": "external_plan_energy_accounting_mismatch", "advisory_eligible": False}
        raise TelemetryError("energy_budget_wh - predicted_remaining_wh_robust must equal energy_margin_wh within 1 Wh")
    if not isinstance(plan["segments"], list) or not plan["segments"]:
        raise TelemetryError("upper plan segments must be a non-empty array")
    if any(not isinstance(seg, dict) or not {"start_distance_m", "end_distance_m", "target_speed_kmh",
            "predicted_net_energy_wh", "predicted_travel_time_s"}.issubset(seg) for seg in plan["segments"]):
        raise TelemetryError("every segment must include distance bounds, target speed, predicted energy, and travel time")
    normalized = dict(plan)
    normalized["segments"] = sorted(plan["segments"], key=lambda x: float(x["start_distance_m"]))
    last_end = -1.0
    total_segment_energy = 0.0
    latest_state = PORT.snapshot()
    if not latest_state.get("ready"):
        raise TelemetryError("a fresh telemetry state is required before accepting an upper plan")
    current_distance = float(latest_state["route_distance_m"])
    previous_end = None
    total_segment_time_s = 0.0
    for i, seg in enumerate(normalized["segments"]):
        start, end, target, energy, travel_s = [float(seg[k]) for k in (
            "start_distance_m", "end_distance_m", "target_speed_kmh", "predicted_net_energy_wh", "predicted_travel_time_s")]
        if (not all(map(math.isfinite, (start, end, target, energy, travel_s))) or start < 0
                or end <= start or start < last_end or not 60 <= target <= 78.2):
            raise TelemetryError("upper plan segment bounds overlap or target is outside 60..78.2 km/h")
        if travel_s < 0:
            raise TelemetryError("predicted_travel_time_s cannot be negative")
        if i == 0 and (start > current_distance + 1.0 or end <= current_distance):
            raise TelemetryError("upper plan must cover the current route distance")
        if previous_end is not None and abs(start-previous_end) > 1.0:
            raise TelemetryError("upper plan segments must be contiguous")
        total_segment_energy += energy
        total_segment_time_s += travel_s
        last_end = end
        previous_end = end
    if normalized["segments"][-1]["end_distance_m"] < float(UPPER_SCHEDULE["day_end_distance_m"])-1.0:
        raise TelemetryError("upper plan must extend through the active day endpoint")
    if abs(total_segment_energy-predicted_wh) > 1.0:
        raise TelemetryError("predicted_remaining_wh_robust must equal the sum of segment net energy within 1 Wh")
    expected_budget = float(BATTERY.snapshot().get("energy_lower_2sigma_wh", math.nan)) - replay.CAPACITY_WH*plan_reserve/100.0
    if not math.isfinite(expected_budget) or abs(expected_budget-energy_budget) > 5.0:
        raise TelemetryError("energy_budget_wh must match latest lower-two-sigma battery energy minus reserve within 5 Wh")
    current_time = pd.Timestamp(latest_state["timestamp_utc"])
    remaining_stop_s = sum(float(x["dwell_s"]) for x in UPPER_SCHEDULE["control_stops"]
                           if float(x["distance_m"]) > current_distance)
    expected_deadline_margin = (pd.Timestamp(UPPER_SCHEDULE["deadline_utc"])-current_time).total_seconds() - remaining_stop_s - total_segment_time_s
    if abs(expected_deadline_margin-deadline_margin) > 5.0:
        raise TelemetryError("deadline_margin_s is inconsistent with active schedule and segment travel times")
    with PLAN_LOCK:
        UPPER_PLAN = normalized
        UPPER_PLAN_STATE = {"status": normalized["solver_status"],
                            "advisory_eligible": True, "source": "external_upper_plan"}
    return {"accepted": True, "plan_id": normalized["plan_id"], "segments": len(normalized["segments"]),
            "issued_at_utc": issued.isoformat(), "valid_until_utc": valid.isoformat()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="loopback by default; set an authenticated private interface only for deployment")
    parser.add_argument("--port", type=int, default=8125)
    parser.add_argument("--route", type=Path, default=ROOT / "data/input/route/route_100m_dem.csv",
                        help="route CSV with distance_m, lat, lon, elev_dem_m; default is BWSC2025 replay route")
    args = parser.parse_args()
    global PREVIEW
    PREVIEW = FittedModelPreview(args.route)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"YATA MPC live input listening at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
