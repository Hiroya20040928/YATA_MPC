#!/usr/bin/env python3
"""Compare constant pack R with the corrected lab-derived R(SOC) map by race day."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import refine_component_models as component  # noqa: E402

INPUT = ROOT / "data/processed/race_days_distance_geo_weather.csv"
CONSTANT_CV = ROOT / "reports/battery_voltage_model_cv.csv"
OUT_CV = ROOT / "reports/battery_soc_resistance_candidate_cv.csv"
OUT_JSON = ROOT / "models/battery_soc_resistance_candidate_cv.json"
Q_AH = 33.164
DF = 4.0


def metric(y: np.ndarray, pred: np.ndarray) -> dict:
    e = np.asarray(y, float) - np.asarray(pred, float)
    return {"n": int(len(e)), "mae_v": float(np.mean(np.abs(e))),
            "rmse_v": float(np.sqrt(np.mean(e * e))), "bias_measured_minus_model_v": float(np.mean(e)),
            "p90_abs_error_v": float(np.quantile(np.abs(e), .9))}


def rows_for(frame: pd.DataFrame, r: float | dict) -> pd.DataFrame:
    state = component.battery_anchor_and_state(frame, Q_AH, r, max_gap_s=75.0)
    return component.battery_validation_rows(state, minutely_input=True)


def robust_nll(residual: np.ndarray) -> float:
    residual = np.asarray(residual, float)
    residual = residual[np.isfinite(residual)]
    if len(residual) < 10:
        return 1e12
    scale = max(float(np.median(np.abs(residual - np.median(residual))) * 1.4826), .05)
    return float(-np.sum(student_t.logpdf(residual / scale, df=DF) - np.log(scale)))


def main() -> None:
    component.make_lab_battery_maps()
    cols = ["day", "timestamp_local", "timestamp_utc", "event_phase", "event_flags",
            "Battery[A]", "Main_Volt[V]", "Speed[kmh]"]
    raw = pd.read_csv(INPUT, usecols=lambda c: c in cols, low_memory=False)
    raw["day"] = pd.to_numeric(raw.day, errors="coerce").astype(int)
    raw["timestamp_utc"] = pd.to_datetime(raw.timestamp_utc, utc=True, errors="coerce")
    raw["timestamp_local"] = pd.to_datetime(raw.timestamp_local, errors="coerce")
    raw["minute_utc"] = raw.timestamp_utc.dt.floor("min")
    minute_frames = {}
    for day, group in raw.groupby("day", sort=True):
        minute_frames[int(day)] = group.groupby(["day", "minute_utc"], as_index=False).agg(
            timestamp_utc=("timestamp_utc", "first"), timestamp_local=("timestamp_local", "first"),
            **{"Battery[A]": ("Battery[A]", "median"), "Main_Volt[V]": ("Main_Volt[V]", "median"),
               "Speed[kmh]": ("Speed[kmh]", "median"), "event_phase": ("event_phase", "first"),
               "event_flags": ("event_flags", "first")})

    const_cv = pd.read_csv(CONSTANT_CV)
    const_cv = const_cv[np.isclose(const_cv.capacity_ah, Q_AH)]
    results = []
    for held in [1, 2, 3, 4, 5, 6]:
        train_days = [d for d in [1, 2, 3, 4, 5] if d != held] if held <= 5 else [1, 2, 3, 4, 5]
        train_days = [d for d in train_days if d in minute_frames]
        if held not in minute_frames:
            continue
        train = [minute_frames[d] for d in train_days]
        test = minute_frames[held]
        crows = const_cv[const_cv.held_out_day.eq(held)]
        r_const = float(crows.r_eff_ohm.iloc[0]) if len(crows) else float(const_cv.r_eff_ohm.mean())

        def map_objective(scale: float) -> float:
            residuals = []
            for dayframe in train:
                scored = rows_for(dayframe, {"scale": float(scale)})
                if len(scored):
                    residuals.extend((scored.measured_voltage_v - scored.predicted_voltage_v).to_numpy(float))
            return robust_nll(np.asarray(residuals))

        optimum = minimize_scalar(map_objective, bounds=(.05, 3.0), method="bounded", options={"xatol": 1e-3})
        r_map = {"scale": float(optimum.x)}
        train_map_residuals, train_const_residuals = [], []
        for dayframe in train:
            a = rows_for(dayframe, r_map)
            c = rows_for(dayframe, r_const)
            if len(a): train_map_residuals.extend((a.measured_voltage_v - a.predicted_voltage_v).to_numpy(float))
            if len(c): train_const_residuals.extend((c.measured_voltage_v - c.predicted_voltage_v).to_numpy(float))
        offset_map = float(np.median(train_map_residuals)) if train_map_residuals else 0.0
        offset_const = float(np.median(train_const_residuals)) if train_const_residuals else 0.0

        const_test = rows_for(test, r_const)
        map_test = rows_for(test, r_map)
        for name, scored, offset, detail in [
            ("constant_R", const_test, 0.0, {"r_ohm": r_const}),
            ("constant_R_plus_train_offset", const_test, offset_const, {"r_ohm": r_const, "offset_v": offset_const}),
            ("lab_R_soc_scaled", map_test, 0.0, {"scale": r_map["scale"]}),
            ("lab_R_soc_scaled_plus_train_offset", map_test, offset_map, {"scale": r_map["scale"], "offset_v": offset_map}),
        ]:
            pred = scored.predicted_voltage_v.to_numpy(float) + offset
            results.append({"held_out_day": held, "model": name,
                            **metric(scored.measured_voltage_v.to_numpy(float), pred),
                            "parameters_json": json.dumps(detail, sort_keys=True),
                            "train_offset_from_measured_minus_predicted_v": offset,
                            "test_rows_1min": int(len(scored))})

    cv = pd.DataFrame(results)
    cv.to_csv(OUT_CV, index=False, encoding="utf-8-sig")
    clean = cv[cv.held_out_day.le(5)]
    summaries = clean.groupby("model").agg(mean_day_mae_v=("mae_v", "mean"),
        mean_day_rmse_v=("rmse_v", "mean"), mean_day_bias_v=("bias_measured_minus_model_v", "mean"),
        folds=("held_out_day", "nunique")).to_dict(orient="index")
    base = summaries["constant_R"]["mean_day_mae_v"]
    best_name = min(summaries, key=lambda k: summaries[k]["mean_day_mae_v"])
    result = {"capacity_ah_fixed_from_10a_discharge_test": Q_AH,
              "resistance_map_source": "8 A vs 10 A lab discharge curves; voltage axis sorted by ascending SOC; endpoints 0 and 100% excluded because differential resistance is negative there.",
              "constant_R_baseline_clean_lodo": summaries["constant_R"],
              "candidate_summaries_clean_lodo": summaries,
              "best_cv_model": best_name,
              "improvement_over_constant_R_mae_v": base - summaries[best_name]["mean_day_mae_v"],
              "decision": "Only adopt an SOC-dependent map if it improves held-day MAE materially and its scale is stable. A fitted voltage offset is a predictive discrepancy term, not a change to sensor calibration; do not interpret it as a physical voltage correction without a reference meter.",
              "folds": results,
              "limitations": ["Voltage-only SOC is not independent truth; each day is initialized from its own bus voltage.",
                              "The 8 A and 10 A tests have different discharge energy and may differ in temperature, cutoff, or cell state.",
                              "The 5-second source is reduced to one-minute medians for CV; folds, not rows, are the independent unit."]}
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"outputs": [str(OUT_CV), str(OUT_JSON)], "summaries_clean_lodo": summaries,
                      "best_model": best_name, "improvement_v": base - summaries[best_name]["mean_day_mae_v"],
                      "folds": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
