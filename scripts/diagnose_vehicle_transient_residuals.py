#!/usr/bin/env python3
"""Quantify how transient acceleration and suspected regen affect row residuals."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fit_vehicle_power_model as features  # noqa: E402

RAW = ROOT / "data/processed/race_days_distance_geo_weather.csv"
REPLAY = ROOT / "data/processed/full_race_component_replay_5s.csv"
OUT = ROOT / "reports/vehicle_transient_residual_diagnostics.csv"
SUMMARY = ROOT / "reports/vehicle_transient_residual_diagnostics.json"


def main() -> None:
    raw = pd.read_csv(RAW, low_memory=False)
    raw = features.make_features(raw)
    replay = pd.read_csv(REPLAY, usecols=[
        "day", "timestamp_local", "vehicle_load_error_measured_minus_model_w",
        "road_model_comparison_eligible", "Battery_Watt[W]", "Battery[A]", "Speed[kmh]",
        "event_flags",
    ])
    raw["timestamp_local"] = pd.to_datetime(raw.timestamp_local, utc=True)
    replay["timestamp_local"] = pd.to_datetime(replay.timestamp_local, utc=True)
    data = raw.merge(replay, on=["day", "timestamp_local"], how="inner",
                     suffixes=("_raw", "_replay"), validate="one_to_one")
    data = data.sort_values(["day", "timestamp_local"]).reset_index(drop=True)
    data["inertial_basis_w"] = 314.0 * data.speed_mps * data.acceleration_smoothed_mps2
    data["stable_cruise_proxy"] = data.acceleration_smoothed_mps2.abs().le(0.075)
    rows: list[dict] = []

    eligible = data[data.road_model_comparison_eligible].copy()
    for day in sorted(eligible.day.unique()):
        day_all = eligible[eligible.day.eq(day)]
        for stable, group in day_all.groupby("stable_cruise_proxy", dropna=False):
            err = group.vehicle_load_error_measured_minus_model_w.to_numpy(float)
            rows.append({"diagnostic": "steady_vs_all", "day": int(day),
                         "category": "stable_proxy" if stable else "dynamic_proxy",
                         "n": int(len(group)), "fraction_of_day_eligible": float(len(group) / len(day_all)),
                         "mae_w": float(np.mean(np.abs(err))), "bias_measured_minus_model_w": float(np.mean(err)),
                         "p90_abs_error_w": float(np.quantile(np.abs(err), 0.9))})

        # Cross-correlation of local residual with acceleration-derived inertial
        # power. Use the full 5 s sequence and reject pairs across logging gaps.
        seq = data[data.day.eq(day)].sort_values("timestamp_local").copy()
        err = seq.vehicle_load_error_measured_minus_model_w.to_numpy(float)
        basis = seq.inertial_basis_w.to_numpy(float)
        times = (seq.timestamp_local - seq.timestamp_local.iloc[0]).dt.total_seconds().to_numpy(float)
        eligible_mask = seq.road_model_comparison_eligible.to_numpy(bool)
        best = {"lag_samples": None, "lag_seconds": None, "correlation": None, "n": 0}
        lag_results = []
        for lag in range(-6, 7):
            if lag > 0:
                left, right = slice(lag, None), slice(None, -lag)
            elif lag < 0:
                left, right = slice(None, lag), slice(-lag, None)
            else:
                left = right = slice(None)
            e, a = err[left], basis[right]
            dt = times[left] - times[right]
            ok = (eligible_mask[left] & eligible_mask[right] & np.isfinite(e) & np.isfinite(a)
                  & (np.abs(dt - lag * 5.0) <= 0.5))
            if int(ok.sum()) < 30:
                continue
            corr = float(np.corrcoef(e[ok], a[ok])[0, 1])
            result = {"lag_samples": lag, "lag_seconds": lag * 5, "correlation": corr, "n": int(ok.sum())}
            lag_results.append(result)
            if best["correlation"] is None or abs(corr) > abs(best["correlation"]):
                best = result
        rows.append({"diagnostic": "inertial_correlation", "day": int(day),
                     "category": "max_abs_corr_lag_within_plus_minus_30s",
                     "n": int(best["n"]), "fraction_of_day_eligible": np.nan,
                     "mae_w": np.nan, "bias_measured_minus_model_w": np.nan,
                     "p90_abs_error_w": np.nan, "best_lag_samples": best["lag_samples"],
                     "best_lag_seconds": best["lag_seconds"], "correlation": best["correlation"]})

    d6 = eligible[eligible.day.eq(6)]
    neg = d6[d6["Battery_Watt[W]_raw"].le(-1000.0)]
    summary = {
        "source": str(RAW.relative_to(ROOT)),
        "replay": str(REPLAY.relative_to(ROOT)),
        "eligible_rows": int(len(eligible)),
        "stable_cruise_proxy": "abs(smoothed acceleration) <= 0.075 m/s^2; speed-derived proxy, not a validated sensor",
        "all_eligible_mae_w": float(np.mean(np.abs(eligible.vehicle_load_error_measured_minus_model_w))),
        "stable_proxy_mae_w": float(np.mean(np.abs(eligible.loc[eligible.stable_cruise_proxy, "vehicle_load_error_measured_minus_model_w"]))),
        "stable_proxy_rows": int(eligible.stable_cruise_proxy.sum()),
        "day6_moving_rows": int(len(d6)),
        "day6_battery_below_minus_1000w_rows": int(len(neg)),
        "day6_battery_below_minus_1000w_min_w": float(neg["Battery_Watt[W]_raw"].min()) if len(neg) else None,
        "day6_battery_below_minus_1000w_median_speed_kmh": float(neg["Speed[kmh]_raw"].median()) if len(neg) else None,
        "day6_battery_below_minus_1000w_median_current_a": float(neg["Battery[A]_raw"].median()) if len(neg) else None,
        "day6_negative_power_interpretation": "May include regenerative braking during deceleration and/or the logged ZP anomaly; available channels do not separate these causes.",
        "day_lag_results": [
            {key: row[key] for key in ["day", "best_lag_samples", "best_lag_seconds", "correlation", "n"]}
            for row in rows if row["diagnostic"] == "inertial_correlation"
        ],
    }
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
