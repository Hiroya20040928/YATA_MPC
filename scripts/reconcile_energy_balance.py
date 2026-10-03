#!/usr/bin/env python3
"""Reintegrate ZP power samples and compare with the team's derived energy note."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEAM_SUMMARY = {
    1: {"pv_wh": 6319.47, "battery_extraction_wh": 2796.30, "battery_charge_wh": 1895.15, "load_wh": 7220.62},
    2: {"pv_wh": 6004.59, "battery_extraction_wh": 2346.02, "battery_charge_wh": 1814.16, "load_wh": 6536.46},
    3: {"pv_wh": 6249.54, "battery_extraction_wh": 1643.07, "battery_charge_wh": 1744.91, "load_wh": 6147.70},
    4: {"pv_wh": 7221.32, "battery_extraction_wh": 1640.44, "battery_charge_wh": 2231.39, "load_wh": 6630.73},
    5: {"pv_wh": 5713.40, "battery_extraction_wh": 2160.15, "battery_charge_wh": 1983.23, "load_wh": 5890.33},
}


def integrate_wh(power_w: np.ndarray, time_s: np.ndarray) -> float:
    return float(np.trapezoid(power_w, time_s) / 3600.0)


def main() -> None:
    path = ROOT / "data/processed/race_days_merged.csv"
    df = pd.read_csv(path, usecols=["day", "timestamp_utc", "Solar_Watt[W]", "Battery_Watt[W]"])
    df["timestamp_utc"] = pd.to_datetime(df.timestamp_utc, utc=True)
    out = []
    for day, group in df.groupby("day", sort=True):
        group = group.sort_values("timestamp_utc")
        t = group.timestamp_utc.diff().dt.total_seconds().fillna(0).cumsum().to_numpy(dtype=float)
        pv = pd.to_numeric(group["Solar_Watt[W]"], errors="coerce").to_numpy(dtype=float)
        batt = pd.to_numeric(group["Battery_Watt[W]"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(t) & np.isfinite(pv) & np.isfinite(batt)
        t, pv, batt = t[valid], pv[valid], batt[valid]
        # Signed battery power is positive for extraction and negative for charge, per the attached team energy-balance note.
        load = pv + batt
        calc = {"day": int(day), "rows": int(len(group)), "start_utc": group.timestamp_utc.iloc[0].isoformat(), "end_utc": group.timestamp_utc.iloc[-1].isoformat(),
                "logged_pv_wh": integrate_wh(pv, t), "logged_battery_signed_wh": integrate_wh(batt, t),
                "logged_battery_extraction_wh": integrate_wh(np.maximum(batt, 0), t),
                "logged_battery_charge_wh": integrate_wh(np.maximum(-batt, 0), t),
                "logged_load_proxy_wh": integrate_wh(load, t)}
        if int(day) in TEAM_SUMMARY:
            ref = TEAM_SUMMARY[int(day)]
            for key, value in ref.items():
                calc[f"team_note_{key}"] = value
                calc[f"difference_{key}_wh"] = calc["logged_pv_wh"] - value if key == "pv_wh" else (
                    calc["logged_battery_extraction_wh"] - value if key == "battery_extraction_wh" else (
                        calc["logged_battery_charge_wh"] - value if key == "battery_charge_wh" else calc["logged_load_proxy_wh"] - value))
        out.append(calc)
    result = pd.DataFrame(out)
    result.to_csv(ROOT / "reports/energy_balance_reconciliation.csv", index=False, encoding="utf-8-sig")
    report = {
        "method": "Trapezoidal integration of every source power row using UTC elapsed seconds; battery extraction is max(Battery_Watt,0), charging is max(-Battery_Watt,0), and load proxy is Solar_Watt + Battery_Watt.",
        "team_reference": "Attached BWSC2025_energy_balance_notes.pdf; its numbers are derived summaries and are compared here rather than treated as raw measurements.",
        "days": out,
        "interpretation": "Differences can arise from source-note time-window boundaries, missing/interpolated samples in the original graph, sign conventions, or dropped/duplicated log rows. Check the per-day values before fitting a load model.",
    }
    (ROOT / "reports/energy_balance_reconciliation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
