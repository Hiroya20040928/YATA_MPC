"""Runtime lower-layer MPC and event-driven upper-layer refresh policy."""
from __future__ import annotations

import bisect
import json
from math import isfinite
from pathlib import Path
from typing import Any


class LowerSpeedMPC:
    """One-move, constrained speed-tracking MPC using fitted operating config."""

    def __init__(self, config_path: Path, envelope_path: Path):
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        rows = []
        with envelope_path.open(encoding="utf-8-sig") as f:
            import csv
            rows = list(csv.DictReader(f))
        self.speeds = [float(r["speed_kmh"]) for r in rows]
        self.accel_caps = [float(r["inferred_acceleration_cap_mps2"]) for r in rows]
        self.decel_caps = [float(r["inferred_deceleration_cap_mps2"]) for r in rows]
        self.previous_accel = 0.0

    def _interpolate(self, values: list[float], speed_kmh: float) -> float:
        j = bisect.bisect_left(self.speeds, speed_kmh)
        if j <= 0:
            return values[0]
        if j >= len(self.speeds):
            return values[-1]
        f = (speed_kmh - self.speeds[j - 1]) / (self.speeds[j] - self.speeds[j - 1])
        return values[j - 1] + f * (values[j] - values[j - 1])

    def reset(self) -> None:
        self.previous_accel = 0.0

    def step(self, speed_kmh: float, target_kmh: float) -> dict[str, Any]:
        dt = float(self.config["lower_update_s"])
        horizon_s = float(self.config["lower_prediction_horizon_s"])
        speed_step = float(self.config.get("lower_speed_advisory_resolution_kmh", 1.0))
        speed_kmh, target_kmh = float(speed_kmh), float(target_kmh)
        if not all(map(isfinite, (speed_kmh, target_kmh))):
            return {"status": "invalid_state_or_target", "advisory_eligible": False,
                "target_speed_kmh": None, "acceleration_advisory_mps2": None,
                "predicted_speed_after_horizon_kmh": None,
                "reason": "speed and target must be finite"}
        target = round(target_kmh / speed_step) * speed_step
        v, ref = speed_kmh / 3.6, target / 3.6
        n = max(1, round(horizon_s / dt))
        horizon = n * dt
        sum_tau = dt * n * (n + 1) / 2
        sum_tau2 = dt * dt * n * (n + 1) * (2 * n + 1) / 6
        track_weight = 1.0
        accel_weight = float(self.config["acceleration_weight"])
        move_weight = float(self.config["move_rate_weight"])
        numerator = track_weight * sum_tau * (ref - v) + move_weight * self.previous_accel
        denominator = track_weight * sum_tau2 + accel_weight + move_weight
        raw = numerator / denominator
        actuator_accel_cap = self._interpolate(self.accel_caps, speed_kmh)
        actuator_decel_cap = self._interpolate(self.decel_caps, speed_kmh)
        a_hi = actuator_accel_cap
        a_lo = -actuator_decel_cap
        a_lo = max(a_lo, (60.0 / 3.6 - v) / horizon)
        a_hi = min(a_hi, (78.2 / 3.6 - v) / horizon)
        if a_lo > a_hi:
            return {"status": "infeasible_no_advisory", "advisory_eligible": False,
                "target_speed_kmh": None, "acceleration_advisory_mps2": None,
                "predicted_speed_after_horizon_kmh": None,
                "prediction_horizon_s": horizon, "update_interval_s": dt,
                "feasible_acceleration_interval_mps2": [],
                "acceleration_bounds_mps2": {"minimum_required": a_lo, "maximum_allowed": a_hi},
                "actuator_acceleration_cap_mps2": actuator_accel_cap,
                "actuator_deceleration_cap_mps2": actuator_decel_cap,
                "reason": "actuator and horizon-end speed constraints have no common acceleration"}
        accel = min(a_hi, max(a_lo, raw))
        # Report the model trajectory itself; clipping would hide a constraint
        # mismatch and make the displayed prediction differ from the QP.
        projected = (v + accel * horizon) * 3.6
        if projected < 60.0 - 1e-7 or projected > 78.2 + 1e-7:
            return {"status": "terminal_speed_constraint_check_failed", "advisory_eligible": False,
                "target_speed_kmh": None, "acceleration_advisory_mps2": None,
                "predicted_speed_after_horizon_kmh": projected,
                "prediction_horizon_s": horizon, "update_interval_s": dt,
                "feasible_acceleration_interval_mps2": [a_lo, a_hi],
                "reason": "optimized trajectory failed the terminal-speed hard-constraint check"}
        self.previous_accel = accel
        return {"status": "optimal_scalar_qp", "advisory_eligible": True,
            "target_speed_kmh": target, "acceleration_advisory_mps2": accel,
            "predicted_speed_after_horizon_kmh": projected, "prediction_horizon_s": horizon,
            "update_interval_s": dt,
            "feasible_acceleration_interval_mps2": [a_lo, a_hi],
            "actuator_acceleration_cap_mps2": actuator_accel_cap,
            "actuator_deceleration_cap_mps2": actuator_decel_cap,
            "scope": "speed advisory from nested 2025-tuned effective response envelope; no PWM output"}


class UpperReplanGate:
    """Multi-rate upper MPC refresh gate; caller supplies new forecast/state each tick."""

    def __init__(self, config_path: Path):
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.last_replan_time = None
        self.last_replan_distance_m = None
        self.last_forecast_issue = None

    def due(self, now, distance_m: float, forecast_issue: str | None,
            event_phase: str, tracking_error_kmh: float = 0.0,
            energy_contract_breached: bool = False) -> tuple[bool, str]:
        if event_phase != "road_day":
            return False, "event_suppresses_planning"
        if self.last_replan_time is None:
            return True, "initial_plan"
        if (now - self.last_replan_time).total_seconds() >= float(self.config["upper_replan_s"]):
            return True, "periodic_refresh"
        if forecast_issue and forecast_issue != self.last_forecast_issue:
            return True, "forecast_issue_changed"
        if self.last_replan_distance_m is not None and distance_m - self.last_replan_distance_m >= 10_000:
            return True, "route_advanced_10km"
        if abs(tracking_error_kmh) > 3.0:
            return True, "tracking_contract_breach"
        if energy_contract_breached:
            return True, "energy_contract_breach"
        return False, "hold_upper_plan"

    def committed(self, now, distance_m: float, forecast_issue: str | None):
        self.last_replan_time = now
        self.last_replan_distance_m = float(distance_m)
        self.last_forecast_issue = forecast_issue

    def attempted(self, now, distance_m: float, forecast_issue: str | None):
        """Throttle repeated failed solves while retaining the same due policy."""
        self.committed(now, distance_m, forecast_issue)

    def failed(self) -> None:
        """Make a failed solve due again after the caller's retry cooldown."""
        self.last_replan_time = None
        self.last_replan_distance_m = None
        self.last_forecast_issue = None
