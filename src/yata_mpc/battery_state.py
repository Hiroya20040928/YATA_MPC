"""Live pack-energy estimate from existing 25S voltage curve and signed ZP power."""
from __future__ import annotations

import bisect
import csv
import json
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class BatteryEnergyEstimator:
    """Coulomb/energy estimator; reports provenance and uncertainty, not true BMS SoC."""

    def __init__(self, root: Path):
        model = json.loads((root / "models/component_replay_fit.json").read_text(encoding="utf-8"))["battery"]
        self.capacity_wh = float(model["user_theoretical_pack_energy_wh"])
        self.r_ohm = float(model["selected_full_clean_day_r_eff_ohm"])
        with (root / "models/battery_voltage_soc_map.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.voltages = [float(r["reference_voltage_25s_v"]) for r in rows]
        self.soc = [float(r["soc_pct"]) for r in rows]
        self._lock = RLock()
        self._first_time: datetime | None = None
        self._anchor_time: datetime | None = None
        self._last_time: datetime | None = None
        self._last_power: float | None = None
        self._energy_wh: float | None = None
        self._uncertainty_wh: float | None = None
        self._anchor_type = "uninitialized"
        self._startup: list[dict[str, Any]] = []
        self._gap_count = 0

    def _energy_from_voltage(self, voltage_v: float, current_a: float) -> float:
        # ZP convention: positive current discharges the pack; add I*R to
        # terminal voltage to estimate the reference/OCV voltage.
        ref_v = voltage_v + self.r_ohm * current_a
        pct = float(np.interp(ref_v, self.voltages, self.soc))
        return self.capacity_wh * pct / 100.0

    def _integrate_rows(self, rows: list[dict[str, Any]], anchor: dict[str, Any]) -> tuple[float, int]:
        energy = self._energy_from_voltage(anchor["voltage_v"], anchor["current_a"])
        gaps = 0
        previous = anchor
        for row in rows:
            if row["timestamp"] <= anchor["timestamp"]:
                continue
            dt = (row["timestamp"] - previous["timestamp"]).total_seconds()
            if 0 < dt <= 2.5:
                energy -= 0.5 * (previous["power_w"] + row["power_w"]) * dt / 3600.0
            elif dt > 2.5:
                gaps += 1
            previous = row
        return min(self.capacity_wh, max(0.0, energy)), gaps

    def ingest(self, sample: dict[str, Any]) -> dict[str, Any]:
        t = _parse_time(sample["timestamp_utc"])
        bus = sample["hv_bus"]
        row = {"timestamp": t, "voltage_v": float(bus["voltage_v"]),
            "current_a": float(bus["battery_current_a"]), "power_w": float(bus["battery_power_w"])}
        with self._lock:
            if self._first_time is None:
                self._first_time = t
            if self._last_time is not None:
                dt = (t - self._last_time).total_seconds()
                if 0 < dt <= 2.5 and self._energy_wh is not None:
                    self._energy_wh -= 0.5 * (float(self._last_power) + row["power_w"]) * dt / 3600.0
                    self._energy_wh = min(self.capacity_wh, max(0.0, self._energy_wh))
                elif dt > 2.5:
                    self._gap_count += 1
                    if self._uncertainty_wh is not None:
                        self._uncertainty_wh = min(self.capacity_wh, self._uncertainty_wh + abs(row["power_w"]) * dt / 3600.0)
            if self._energy_wh is None:
                # The first usable voltage is an explicit low-confidence
                # fallback until a robust 3-minute low-current anchor exists.
                self._energy_wh = self._energy_from_voltage(row["voltage_v"], row["current_a"])
                self._uncertainty_wh = 0.15 * self.capacity_wh
                self._anchor_time = t
                self._anchor_type = "loaded_voltage_fallback"
            if self._first_time is not None and (t - self._first_time).total_seconds() <= 180:
                self._startup.append(row)
                good = [x for x in self._startup if abs(x["current_a"]) <= 5 and 70 <= x["voltage_v"] <= 115]
                if len(good) >= 10:
                    anchor = good[len(good) // 2]
                    corrected, gaps = self._integrate_rows(self._startup, anchor)
                    self._energy_wh = corrected
                    self._anchor_time = anchor["timestamp"]
                    self._anchor_type = "first_3min_low_current_median"
                    self._uncertainty_wh = 0.08 * self.capacity_wh + gaps * 10.0
                    self._gap_count = gaps
            self._last_time, self._last_power = t, row["power_w"]
            return self._snapshot_unlocked()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict[str, Any]:
        if self._energy_wh is None:
            return {"ready": False, "status": "waiting_for_valid_voltage"}
        uncertainty = float(self._uncertainty_wh or 0.0)
        energy = float(self._energy_wh)
        return {"ready": True, "energy_remaining_wh": energy,
            "soc_estimate_pct": 100.0 * energy / self.capacity_wh,
            "energy_uncertainty_wh": uncertainty,
            "energy_lower_2sigma_wh": max(0.0, energy - 2 * uncertainty),
            "capacity_wh_prior": self.capacity_wh, "effective_resistance_ohm": self.r_ohm,
            "anchor_type": self._anchor_type,
            "anchor_time_utc": self._anchor_time.isoformat() if self._anchor_time else None,
            "unbridged_sample_gaps": self._gap_count,
            "state_quality": "low" if self._anchor_type == "loaded_voltage_fallback" else "estimated",
            "is_bms_soc": False}
