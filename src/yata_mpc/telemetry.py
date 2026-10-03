"""Validated 1 Hz telemetry and route-weather input port for YATA MPC."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from threading import RLock
from typing import Any


class TelemetryError(ValueError):
    pass


def _utc(value: str) -> datetime:
    try:
        out = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise TelemetryError("timestamp_utc must be RFC3339 with a timezone") from exc
    if out.tzinfo is None:
        raise TelemetryError("timestamp_utc must include UTC offset")
    return out.astimezone(timezone.utc)


def validate_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Validate required live fields and the ZP sign / sensor consistency rules."""
    if not isinstance(sample, dict) or sample.get("schema_version") != "1.0":
        raise TelemetryError("schema_version must be '1.0'")
    for key in ("vehicle_id", "sequence", "timestamp_utc", "position", "speed_kmh", "hv_bus", "pv_power_w", "event"):
        if key not in sample:
            raise TelemetryError(f"missing required field: {key}")
    if not isinstance(sample["vehicle_id"], str) or not sample["vehicle_id"].strip():
        raise TelemetryError("vehicle_id must be a non-empty string")
    if isinstance(sample["sequence"], bool) or not isinstance(sample["sequence"], int) or sample["sequence"] < 0:
        raise TelemetryError("sequence must be a non-negative integer")
    _utc(sample["timestamp_utc"])
    pos = sample["position"]
    bus = sample["hv_bus"]
    event = sample["event"]
    if not isinstance(pos, dict) or not isinstance(bus, dict) or not isinstance(event, dict):
        raise TelemetryError("position, hv_bus, and event must be objects")
    required_pos = ("latitude_deg", "longitude_deg", "horizontal_accuracy_m", "route_distance_m", "route_id")
    required_bus = ("voltage_v", "battery_current_a", "battery_power_w")
    for key in required_pos:
        if key not in pos:
            raise TelemetryError(f"missing position.{key}")
    if not isinstance(pos["route_id"], str) or not pos["route_id"].strip():
        raise TelemetryError("position.route_id must be a non-empty string")
    for key in required_bus:
        if key not in bus:
            raise TelemetryError(f"missing hv_bus.{key}")
    values = {
        "latitude_deg": (-90, 90, pos["latitude_deg"]),
        "longitude_deg": (-180, 180, pos["longitude_deg"]),
        "horizontal_accuracy_m": (0, 1000, pos["horizontal_accuracy_m"]),
        "route_distance_m": (0, 10_000_000, pos["route_distance_m"]),
        "speed_kmh": (0, 200, sample["speed_kmh"]),
        "voltage_v": (1, 150, bus["voltage_v"]),
        "battery_current_a": (-200, 200, bus["battery_current_a"]),
        "battery_power_w": (-20_000, 20_000, bus["battery_power_w"]),
        "pv_power_w": (0, 10_000, sample["pv_power_w"]),
    }
    for key, (lo, hi, value) in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not lo <= value <= hi:
            raise TelemetryError(f"{key} must be finite and in [{lo}, {hi}]")
    if abs(float(bus["voltage_v"]) * float(bus["battery_current_a"]) - float(bus["battery_power_w"])) > max(10.0, 0.03 * abs(float(bus["battery_power_w"]))):
        raise TelemetryError("battery_power_w is inconsistent with voltage_v * battery_current_a")
    if event.get("phase") not in {"road_day", "control_stop", "traffic_stop", "repair", "overnight", "other"}:
        raise TelemetryError("event.phase is invalid")
    if not isinstance(event.get("flags"), list) or len(event["flags"]) > 32:
        raise TelemetryError("event.flags must be a list of at most 32 strings")
    if any(not isinstance(x, str) for x in event["flags"]):
        raise TelemetryError("every event.flags item must be a string")
    if float(pos["horizontal_accuracy_m"]) > 50.0:
        sample = dict(sample)
        sample["_input_quality_warning"] = "gps_accuracy_over_50m"
    return sample


@dataclass(frozen=True)
class RouteWeatherPoint:
    distance_m: float
    time_utc: datetime
    temperature_2m_c: float
    wind_speed_10m_ms: float
    wind_direction_10m_deg: float
    surface_pressure_pa: float
    ghi_w_m2: float
    dni_w_m2: float
    dhi_w_m2: float


class LatestInputPort:
    """Thread-safe latest-state port. The controller pulls one coherent snapshot per tick."""

    def __init__(self, stale_after_s: float = 2.5, max_weather_gap_h: float = 3.0):
        self.stale_after_s = stale_after_s
        self.max_weather_gap_h = max_weather_gap_h
        self._lock = RLock()
        self._sample: dict[str, Any] | None = None
        self._sample_time: datetime | None = None
        self._weather: list[RouteWeatherPoint] = []
        self._weather_issue: datetime | None = None
        self._weather_route_id: str | None = None
        self._last_sequence = -1
        self._last_sequence_gap = 0
        self._last_sample_gap_s: float | None = None

    def ingest_telemetry(self, raw: dict[str, Any]) -> dict[str, Any]:
        sample = validate_sample(raw)
        stamp = _utc(sample["timestamp_utc"])
        with self._lock:
            if sample["sequence"] <= self._last_sequence:
                raise TelemetryError("sequence must strictly increase")
            if self._sample_time is not None and stamp <= self._sample_time:
                raise TelemetryError("timestamp_utc must strictly increase")
            self._last_sequence_gap = max(0, sample["sequence"] - self._last_sequence - 1) if self._last_sequence >= 0 else 0
            self._last_sample_gap_s = (stamp - self._sample_time).total_seconds() if self._sample_time is not None else None
            self._sample = sample
            self._sample_time = stamp
            self._last_sequence = sample["sequence"]
        return {"accepted": True, "sequence": sample["sequence"], "timestamp_utc": stamp.isoformat(),
                "sequence_gap": self._last_sequence_gap, "sample_gap_s": self._last_sample_gap_s}

    def ingest_route_weather(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Accept a route-distance/time forecast grid; do not mix it into raw telemetry."""
        try:
            issue = _utc(payload["issue_time_utc"])
            route_id = payload["route_id"]
            if not isinstance(route_id, str) or not route_id.strip():
                raise TelemetryError("route_id must be a non-empty string")
            rows = payload["points"]
            if not isinstance(rows, list) or len(rows) < 2:
                raise TelemetryError("points must contain at least two route/time forecast rows")
            points = []
            for row in rows:
                points.append(RouteWeatherPoint(
                    float(row["distance_m"]), _utc(row["time_utc"]),
                    float(row["temperature_2m_c"]), float(row["wind_speed_10m_ms"]),
                    float(row["wind_direction_10m_deg"]), float(row["surface_pressure_pa"]),
                    float(row["ghi_w_m2"]), float(row["dni_w_m2"]), float(row["dhi_w_m2"]),
                ))
                if not all(isfinite(v) for v in (points[-1].distance_m, points[-1].temperature_2m_c,
                    points[-1].wind_speed_10m_ms, points[-1].wind_direction_10m_deg,
                    points[-1].surface_pressure_pa, points[-1].ghi_w_m2, points[-1].dni_w_m2,
                    points[-1].dhi_w_m2)):
                    raise TelemetryError("forecast values must be finite")
            grid_keys = [(p.distance_m, p.time_utc) for p in points]
            if len(set(grid_keys)) != len(grid_keys):
                raise TelemetryError("route-weather grid must not contain duplicate distance/time pairs")
            if any(p.distance_m < 0 or p.wind_speed_10m_ms < 0 or p.surface_pressure_pa <= 0 for p in points):
                raise TelemetryError("forecast route distance, wind, or pressure is outside its valid range")
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, TelemetryError):
                raise
            raise TelemetryError(f"invalid route-weather payload: {exc}") from exc
        with self._lock:
            if self._weather_issue is not None and issue < self._weather_issue:
                raise TelemetryError("forecast issue_time_utc cannot move backwards")
            self._weather = sorted(points, key=lambda p: (p.distance_m, p.time_utc))
            self._weather_issue = issue
            self._weather_route_id = route_id
        return {"accepted": True, "route_id": route_id, "issue_time_utc": issue.isoformat(), "points": len(points)}

    def snapshot(self, now: datetime | None = None) -> dict[str, Any]:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        with self._lock:
            sample, stamp = self._sample, self._sample_time
            weather, issue, weather_route_id = list(self._weather), self._weather_issue, self._weather_route_id
            sequence_gap, sample_gap_s = self._last_sequence_gap, self._last_sample_gap_s
        if sample is None or stamp is None:
            return {"ready": False, "reason": "no_telemetry"}
        t = _utc(sample["timestamp_utc"])
        age = (now - stamp).total_seconds()
        if age < -1 or age > self.stale_after_s:
            return {"ready": False, "reason": "telemetry_stale", "telemetry_age_s": age,
                    "sequence": sample["sequence"]}
        if sample["event"]["phase"] != "road_day":
            return {"ready": False, "reason": "event_phase_suppresses_mpc", "event": sample["event"],
                    "telemetry_age_s": age, "sequence": sample["sequence"]}
        trouble_flags = {"incident", "control_stop", "traffic_stop", "driver_change",
            "repair", "charge", "hatch_repair", "fault", "emergency", "emergency_stop"}
        flagged_event_types = {
            flag.strip().lower().replace("-", "_").replace(" ", "_").split(":", 1)[0]
            for flag in sample["event"]["flags"]
        }
        if flagged_event_types & trouble_flags:
            return {"ready": False, "reason": "event_flag_suppresses_mpc", "event": sample["event"],
                    "telemetry_age_s": age, "sequence": sample["sequence"]}
        quality = sample.get("quality", {})
        if any(quality.get(name) is False for name in ("gps_valid", "hv_valid", "pv_valid")):
            return {"ready": False, "reason": "sensor_quality_invalid", "telemetry_age_s": age,
                    "sequence": sample["sequence"]}
        position = sample["position"]
        if position["route_id"] != weather_route_id:
            return {"ready": False, "reason": "forecast_route_id_mismatch",
                    "telemetry_route_id": position["route_id"], "forecast_route_id": weather_route_id,
                    "telemetry_age_s": age, "sequence": sample["sequence"]}
        if position["horizontal_accuracy_m"] > 50:
            return {"ready": False, "reason": "gps_accuracy_low", "telemetry_age_s": age,
                    "sequence": sample["sequence"]}
        if not weather or issue is None:
            return {"ready": False, "reason": "route_forecast_missing", "telemetry_age_s": age,
                    "sequence": sample["sequence"]}
        forecast_age = (t - issue).total_seconds()
        if forecast_age < 0 or forecast_age > 12 * 3600:
            return {"ready": False, "reason": "route_forecast_issue_stale_or_future",
                    "telemetry_age_s": age, "forecast_age_s": forecast_age,
                    "forecast_issue_time_utc": issue.isoformat(), "sequence": sample["sequence"]}
        d = float(position["route_distance_m"])
        by_distance: dict[float, list[RouteWeatherPoint]] = {}
        for p in weather:
            by_distance.setdefault(p.distance_m, []).append(p)
        ds = sorted(by_distance)
        lo = max((x for x in ds if x <= d), default=None)
        hi = min((x for x in ds if x >= d), default=None)
        if lo is None or hi is None or d - lo > 30_000 or hi - d > 30_000:
            return {"ready": False, "reason": "route_forecast_spatial_coverage_gap", "telemetry_age_s": age,
                    "forecast_issue_time_utc": issue.isoformat(), "sequence": sample["sequence"]}
        lower = self._at_time(by_distance[lo], t)
        upper = self._at_time(by_distance[hi], t)
        if lower is None or upper is None:
            return {"ready": False, "reason": "route_forecast_time_coverage_gap", "telemetry_age_s": age,
                    "forecast_issue_time_utc": issue.isoformat(), "sequence": sample["sequence"]}
        frac = 0.0 if hi == lo else (d - lo) / (hi - lo)
        weather_now = {key: lower[key] + frac * (upper[key] - lower[key]) for key in lower}
        direction_delta = ((upper["wind_direction_10m_deg"] - lower["wind_direction_10m_deg"] + 180) % 360) - 180
        weather_now["wind_direction_10m_deg"] = (lower["wind_direction_10m_deg"] + frac * direction_delta) % 360
        return {"ready": True, "sequence": sample["sequence"], "timestamp_utc": stamp.isoformat(),
                "telemetry_age_s": age, "telemetry_sequence_gap": sequence_gap,
                "previous_sample_gap_s": sample_gap_s,
                "quality_warning": (sequence_gap > 0 or (sample_gap_s is not None and sample_gap_s > 2.5)),
                "route_distance_m": d,
                "position": position, "speed_kmh": sample["speed_kmh"], "hv_bus": sample["hv_bus"],
                "pv_power_w": sample["pv_power_w"], "event": sample["event"],
                "sensors": sample.get("sensors", {}), "weather": weather_now,
                "forecast_issue_time_utc": issue.isoformat(), "forecast_age_s": (t - issue).total_seconds()}

    def forecast_grid(self) -> dict[str, Any] | None:
        """Return an immutable copy of the latest full route/time weather grid."""
        with self._lock:
            weather, issue, route_id = list(self._weather), self._weather_issue, self._weather_route_id
        if issue is None or not weather:
            return None
        return {"route_id": route_id, "issue_time_utc": issue.isoformat(), "points": [
            {"distance_m": p.distance_m, "time_utc": p.time_utc.isoformat(),
             "temperature_2m_c": p.temperature_2m_c,
             "wind_speed_10m_ms": p.wind_speed_10m_ms,
             "wind_direction_10m_deg": p.wind_direction_10m_deg,
             "surface_pressure_pa": p.surface_pressure_pa,
             "ghi_w_m2": p.ghi_w_m2, "dni_w_m2": p.dni_w_m2,
             "dhi_w_m2": p.dhi_w_m2}
            for p in weather]}

    def _at_time(self, points: list[RouteWeatherPoint], target: datetime) -> dict[str, float] | None:
        points = sorted(points, key=lambda p: p.time_utc)
        before = [p for p in points if p.time_utc <= target]
        after = [p for p in points if p.time_utc >= target]
        if not before or not after:
            return None
        a, b = before[-1], after[0]
        if (target - a.time_utc).total_seconds() > self.max_weather_gap_h * 3600 or (b.time_utc - target).total_seconds() > self.max_weather_gap_h * 3600:
            return None
        f = 0.0 if a.time_utc == b.time_utc else (target - a.time_utc).total_seconds() / (b.time_utc - a.time_utc).total_seconds()
        keys = ("temperature_2m_c", "wind_speed_10m_ms", "wind_direction_10m_deg", "surface_pressure_pa", "ghi_w_m2", "dni_w_m2", "dhi_w_m2")
        out = {key: getattr(a, key) + f * (getattr(b, key) - getattr(a, key)) for key in keys}
        direction_delta = ((b.wind_direction_10m_deg - a.wind_direction_10m_deg + 180) % 360) - 180
        out["wind_direction_10m_deg"] = (a.wind_direction_10m_deg + f * direction_delta) % 360
        return out
