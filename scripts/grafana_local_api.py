#!/usr/bin/env python3
"""Read-only local API that streams the BWSC2025 MPC shadow replay to Grafana."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
REPLAY_CSV = ROOT / "data/processed/mpc_2025_as_live_replay.csv"
COMPONENT_CSV = ROOT / "data/processed/full_race_component_replay_5s.csv"
ROUTE_CSV = ROOT / "data/input/route/route_100m_dem.csv"
EVENTS_CSV = ROOT / "data/input/route/events.csv"
BATTERY_CAPACITY_WH = 3011.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def route_position(distance_m: float, distances: list[float], route: list[dict[str, str]]) -> dict[str, float]:
    distance_m = max(distances[0], min(distances[-1], distance_m))
    right = bisect.bisect_left(distances, distance_m)
    if right <= 0:
        left = right = 0
    elif right >= len(distances):
        left = right = len(distances) - 1
    else:
        left = right - 1
    d0, d1 = distances[left], distances[right]
    fraction = 0.0 if d1 == d0 else (distance_m - d0) / (d1 - d0)
    p0, p1 = route[left], route[right]
    return {
        "latitude": float(p0["lat"]) + fraction * (float(p1["lat"]) - float(p0["lat"])),
        "longitude": float(p0["lon"]) + fraction * (float(p1["lon"]) - float(p0["lon"])),
        "elevation_m": float(p0["elev_dem_m"]) + fraction * (float(p1["elev_dem_m"]) - float(p0["elev_dem_m"])),
    }


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def safe_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


class ReplayStore:
    def __init__(self, rate: float, start_index: int, refresh_s: float) -> None:
        replay = read_csv(REPLAY_CSV)
        components = read_csv(COMPONENT_CSV)
        if len(replay) != len(components):
            raise RuntimeError(f"Replay/component row mismatch: {len(replay)} != {len(components)}")
        for index, (a, b) in enumerate(zip(replay, components)):
            if a.get("day") != b.get("day") or a.get("timestamp_utc") != b.get("timestamp_utc"):
                raise RuntimeError(f"Replay/component alignment failed at row {index}")
        self.rows: list[dict[str, str]] = replay
        self.components: list[dict[str, str]] = components
        self.route = read_csv(ROUTE_CSV)
        self.route_distances = [float(row["distance_m"]) for row in self.route]
        self.events = read_csv(EVENTS_CSV)
        self.rate = rate
        self.start_index = max(0, min(start_index, len(self.rows) - 1))
        self.refresh_s = refresh_s
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.latest_index = self.start_index
        self.latest_access = self.started
        self.route_geojson = self.make_route_geojson()

    def index(self) -> int:
        elapsed = max(0.0, time.monotonic() - self.started)
        step = int(elapsed * self.rate / self.refresh_s)
        index = min(len(self.rows) - 1, self.start_index + step)
        with self.lock:
            self.latest_index = index
            self.latest_access = time.monotonic()
        return index

    def point(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        component = self.components[index]
        distance = number(row.get("distance_from_darwin_m")) or 0.0
        location = route_position(distance, self.route_distances, self.route)
        energy = number(row.get("stored_energy_est_wh"))
        forecast_age = number(row.get("forecast_age_at_decision_h"))
        current = number(row.get("Battery[A]"))
        battery_w = number(row.get("Battery_Watt[W]"))
        upper_status = (row.get("upper_solver_status") or "inactive").strip()
        lower_status = (row.get("lower_status") or "inactive").strip()
        flags = (row.get("event_flags") or "").strip()
        event_phase = (row.get("event_phase") or "").strip()
        target = number(row.get("upper_target_speed_kmh"))
        reason = "正常"
        severity = "ok"
        if "infeasible" in upper_status.lower() or "fallback" in upper_status.lower():
            severity = "warning"
            reason = "上位MPCにハード制約を満たす計画なし。速度助言を停止し、人の判断へ"
        elif "feasible_time_limited" in upper_status.lower() or "not_proven_optimal" in upper_status.lower():
            severity = "warning"
            reason = "上位MPCは実行可能解を返したが最適性未証明"
        elif upper_status.lower() not in {"inactive", "optimal_finite_grid_milp"}:
            severity = "warning"
            reason = f"上位MPC状態を確認: {upper_status}"
        if event_phase and event_phase not in {"road_day", "moving"}:
            severity = "warning"
            reason = f"走行フェーズ: {event_phase}"
        if flags and not flags.startswith("road_day:"):
            severity = "critical" if severity == "ok" else severity
            reason = flags
        if energy is not None and energy / BATTERY_CAPACITY_WH <= 0.15:
            severity = "critical"
            reason = "推定残量が暫定15%リザーブ以下"
        if forecast_age is not None and forecast_age > 12.0 and severity == "ok":
            severity = "warning"
            reason = "気象予報データが12時間超経過"

        return {
            "time": iso_now(),
            "replay_index": index + 1,
            "replay_total_rows": len(self.rows),
            "replay_rate_x": self.rate,
            "replay_day": int(row.get("day") or 0),
            "race_time_local": row.get("timestamp_local"),
            "event_phase": event_phase,
            "event_flags": flags,
            "distance_km": distance / 1000.0,
            "distance_confidence": row.get("distance_anchor_confidence"),
            **location,
            "speed_kmh": number(row.get("Speed[kmh]")),
            "target_speed_kmh": target,
            "energy_wh": energy,
            "soc_estimate_pct": None if energy is None else 100.0 * energy / BATTERY_CAPACITY_WH,
            "battery_voltage_v": number(row.get("Main_Volt[V]")),
            "battery_current_a": current,
            "battery_power_w": battery_w,
            "pv_power_w": number(row.get("Solar_Watt[W]")),
            "air_temp_c": number(component.get("temperature_2m")),
            "irradiance_w_m2": number(component.get("shortwave_radiation_instant")),
            "wind_speed_ms": number(component.get("wind_speed_10m")),
            "route_grade_pct": number(component.get("route_grade_smoothed_pct")),
            "forecast_age_h": forecast_age,
            "forecast_run_init_utc": row.get("forecast_run_init_utc"),
            "mpc_solver_status": upper_status,
            "mpc_replan_now": safe_bool(row.get("upper_replan_now")),
            "mpc_mip_gap": number(row.get("upper_mip_gap")),
            "mpc_remaining_time_h": number(row.get("upper_predicted_remaining_time_h")),
            "mpc_remaining_net_energy_wh": number(row.get("upper_predicted_remaining_net_energy_wh")),
            "mpc_energy_budget_wh": number(row.get("upper_energy_budget_wh")),
            "mpc_reserve_pct": number(row.get("mpc_reserve_pct")),
            "mpc_residual_bias_w": number(row.get("upper_causal_residual_bias_w")),
            "lower_accel_request_mps2": number(row.get("lower_accel_request_mps2")),
            "lower_predicted_speed_30s_kmh": number(row.get("lower_predicted_speed_after_30s_kmh")),
            "lower_status": lower_status,
            "measured_load_proxy_w": number(component.get("measured_load_proxy_w")),
            "predicted_vehicle_load_w": number(component.get("predicted_vehicle_load_w")),
            "vehicle_load_error_w": number(component.get("vehicle_load_error_measured_minus_model_w")),
            "predicted_pv_w": number(component.get("predicted_pv_w")),
            "pv_error_w": number(component.get("pv_error_measured_minus_model_w")),
            "severity": severity,
            "operator_note": reason,
            "controller_scope": row.get("controller_output_scope"),
        }

    def series(self, index: int, limit: int = 720) -> list[dict[str, Any]]:
        start = max(0, index - limit + 1)
        now_ms = int(time.time() * 1000)
        synthetic_step_ms = self.refresh_s * 1000.0 / self.rate
        rows: list[dict[str, Any]] = []
        last_target: float | None = None
        for prior in range(start - 1, -1, -1):
            last_target = number(self.rows[prior].get("upper_target_speed_kmh"))
            if last_target is not None:
                break
        for i in range(start, index + 1):
            row = self.rows[i]
            component = self.components[i]
            candidate_target = number(row.get("upper_target_speed_kmh"))
            if candidate_target is not None:
                last_target = candidate_target
            age = index - i
            rows.append({
                "time_ms": int(now_ms - age * synthetic_step_ms),
                "speed_kmh": number(row.get("Speed[kmh]")),
                "target_speed_kmh": last_target,
                "battery_power_w": number(row.get("Battery_Watt[W]")),
                "pv_power_w": number(row.get("Solar_Watt[W]")),
                "battery_voltage_v": number(row.get("Main_Volt[V]")),
                "battery_current_a": number(row.get("Battery[A]")),
                "energy_wh": number(row.get("stored_energy_est_wh")),
                "air_temp_c": number(component.get("temperature_2m")),
                "irradiance_w_m2": number(component.get("shortwave_radiation_instant")),
                "wind_speed_ms": number(component.get("wind_speed_10m")),
                "measured_load_proxy_w": number(component.get("measured_load_proxy_w")),
                "predicted_vehicle_load_w": number(component.get("predicted_vehicle_load_w")),
                "predicted_pv_w": number(component.get("predicted_pv_w")),
            })
        return rows

    def alerts(self, index: int) -> list[dict[str, Any]]:
        p = self.point(index)
        return [{
            "severity": p["severity"],
            "status": "監視中" if p["severity"] == "ok" else "要確認",
            "item": p["operator_note"],
            "day": p["replay_day"],
            "race_time_local": p["race_time_local"],
            "distance_km": p["distance_km"],
            "solver": p["mpc_solver_status"],
            "event_flags": p["event_flags"],
        }]

    def decisions(self, index: int) -> list[dict[str, str]]:
        p = self.point(index)
        fields = [
            ("レース日 / フェーズ", f"Day {p['replay_day']} / {p['event_phase']}", "BWSC2025 実測ログ"),
            ("大会ログ時刻", str(p["race_time_local"] or "—"), "現地時刻; 再生時計ではありません"),
            ("累積距離", f"{p['distance_km']:.1f} km", f"アンカー信頼度: {p['distance_confidence'] or '不明'}"),
            ("上位MPC目標", f"{p['target_speed_kmh']:.1f} km/h" if p["target_speed_kmh"] is not None else "未出力", "シャドー提案; 実車指令ではありません"),
            ("上位ソルバー", str(p["mpc_solver_status"]), f"MIP gap: {p['mpc_mip_gap'] if p['mpc_mip_gap'] is not None else '—'}"),
            ("下位MPC状態", str(p["lower_status"]), f"加速度要求: {p['lower_accel_request_mps2'] if p['lower_accel_request_mps2'] is not None else '—'} m/s²"),
            ("下位予測速度", f"{p['lower_predicted_speed_30s_kmh']:.1f} km/h" if p["lower_predicted_speed_30s_kmh"] is not None else "未出力", "30秒先"),
            ("気象予報経過", f"{p['forecast_age_h']:.1f} h" if p["forecast_age_h"] is not None else "不明", f"run: {p['forecast_run_init_utc'] or '—'} UTC"),
            ("残りネット電力量予測", f"{p['mpc_remaining_net_energy_wh']:.0f} Wh" if p["mpc_remaining_net_energy_wh"] is not None else "未出力", f"暫定reserve {100*(p['mpc_reserve_pct'] or 0):.0f}%"),
        ]
        return [{"項目": key, "値": value, "補足": note} for key, value, note in fields]

    def make_route_geojson(self) -> dict[str, Any]:
        coordinates = [[float(row["lon"]), float(row["lat"])] for row in self.route[::5]]
        if coordinates[-1] != [float(self.route[-1]["lon"]), float(self.route[-1]["lat"])]:
            coordinates.append([float(self.route[-1]["lon"]), float(self.route[-1]["lat"])])
        features: list[dict[str, Any]] = [{
            "type": "Feature",
            "properties": {
                "class": "BWSC2025 official route reconstruction",
                "name": "Darwin → Adelaide",
                "stroke": "#F4A340",
                "stroke-width": 3,
            },
            "geometry": {"type": "LineString", "coordinates": coordinates},
        }]
        for event in self.events:
            distance = number(event.get("distance_m"))
            if distance is None or distance < self.route_distances[0] or distance > self.route_distances[-1]:
                continue
            pos = route_position(distance, self.route_distances, self.route)
            features.append({
                "type": "Feature",
                "properties": {
                    "class": event.get("event_type"),
                    "name": event.get("event_label"),
                    "day": event.get("day"),
                    "distance_km": round(distance / 1000.0, 1),
                    "marker-color": "#C44055" if "incident" in str(event.get("event_type")) else "#4E9A77",
                    "marker-size": "small",
                },
                "geometry": {"type": "Point", "coordinates": [pos["longitude"], pos["latitude"]]},
            })
        return {"type": "FeatureCollection", "features": features}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--rate", type=float, default=1.0, help="replay speed multiplier")
    parser.add_argument("--start-index", type=int, default=159, help="zero-based row; default begins on-road")
    parser.add_argument("--refresh-seconds", type=float, default=5.0, help="telemetry sampling interval in source log")
    args = parser.parse_args()
    if args.rate <= 0 or args.refresh_seconds <= 0:
        parser.error("--rate and --refresh-seconds must be positive")
    store = ReplayStore(args.rate, args.start_index, args.refresh_seconds)

    class Handler(BaseHTTPRequestHandler):
        server_version = "YATA-MPC-Local/1.0"

        def end_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:3000")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.end_headers()

        def respond(self, payload: Any, content_type: str = "application/json; charset=utf-8") -> None:
            data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            index = store.index()
            if path == "/api/health":
                self.respond({"status": "ok", "rows": len(store.rows), "replay_index": index + 1, "source_mode": "historical_shadow_replay"})
            elif path == "/api/current":
                self.respond([store.point(index)])
            elif path == "/api/series":
                self.respond(store.series(index))
            elif path == "/api/alerts":
                self.respond(store.alerts(index))
            elif path == "/api/decision":
                self.respond(store.decisions(index))
            elif path == "/api/route.geojson":
                self.respond(store.route_geojson, "application/geo+json; charset=utf-8")
            elif path == "/api/events":
                self.respond(store.events)
            else:
                self.send_error(404, "Not found")

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"YATA MPC local replay API listening on http://{args.host}:{args.port}; {len(store.rows)} rows; rate={args.rate:g}x; start={args.start_index}", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
