#!/usr/bin/env python3
"""Day-blocked evaluation of a short-horizon inertial road-load candidate.

This is a model-selection analysis, not a sensor calibration. It compares a
steady road-load model with a model that adds signed kinetic-power terms.
The 5-second speed derivative is noisy, so the fit uses 1-minute means and
keeps whole race days out for validation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fit_vehicle_power_model as features  # noqa: E402

INPUT = ROOT / "data/processed/race_days_distance_geo_weather.csv"
OUT_CV = ROOT / "reports/vehicle_inertial_candidate_cv.csv"
OUT_JSON = ROOT / "models/vehicle_inertial_candidate_cv.json"
MASS_KG = features.MASS_NOMINAL_KG
DF = 4.0
TERMS = ["roll_basis_w", "aero_basis_w", "grade_basis_w"]
ACC_TERMS = ["accel_positive_w", "accel_negative_w"]


def metric(y: np.ndarray, pred: np.ndarray) -> dict:
    e = np.asarray(y, float) - np.asarray(pred, float)
    return {"n": int(len(e)), "mae_w": float(np.mean(np.abs(e))),
            "rmse_w": float(np.sqrt(np.mean(e * e))),
            "bias_measured_minus_model_w": float(np.mean(e)),
            "p90_abs_error_w": float(np.quantile(np.abs(e), .9))}


def fit(frame: pd.DataFrame, aux_w: float, use_accel: bool) -> dict:
    cols = TERMS + (ACC_TERMS if use_accel else [])
    X = frame[cols].to_numpy(float)
    y = frame.net_power_w.to_numpy(float) - aux_w

    def nll(p):
        mu = X @ p[:-1]
        scale = max(float(p[-1]), 1.0)
        return float(-np.sum(student_t.logpdf((y - mu) / scale, df=DF) - np.log(scale)))

    fit_result = minimize(nll, np.r_[np.ones(len(cols)), 120.0], method="L-BFGS-B",
                          bounds=[(0., 5.)] * len(cols) + [(10., 1500.)],
                          options={"maxiter": 1000, "ftol": 1e-10})
    return {"coefficients": dict(zip(cols, map(float, fit_result.x[:-1]))),
            "student_t_scale_w": float(fit_result.x[-1]),
            "success": bool(fit_result.success), "message": str(fit_result.message)}


def predict(frame: pd.DataFrame, model: dict, aux_w: float) -> np.ndarray:
    y = np.full(len(frame), aux_w, dtype=float)
    for name, beta in model["coefficients"].items():
        y += beta * frame[name].to_numpy(float)
    return y


def main() -> None:
    cols = ["day", "timestamp_local", "timestamp_utc", "event_phase", "event_flags",
            "Solar_Watt[W]", "Battery_Watt[W]", "Speed[kmh]", "distance_from_darwin_m",
            "distance_anchor_confidence", "route_grade_smoothed_pct", "temperature_2m",
            "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
    d = pd.read_csv(INPUT, usecols=lambda c: c in cols, low_memory=False)
    d["timestamp_local"] = pd.to_datetime(d.timestamp_local, errors="coerce")
    d["timestamp_utc"] = pd.to_datetime(d.timestamp_utc, utc=True, errors="coerce")
    d["day"] = pd.to_numeric(d.day, errors="coerce").astype(int)
    d = features.make_features(d)
    d["net_power_w"] = pd.to_numeric(d["Solar_Watt[W]"], errors="coerce") + pd.to_numeric(d["Battery_Watt[W]"], errors="coerce")
    d["accel_positive_w"] = MASS_KG * np.maximum(d.acceleration_smoothed_mps2, 0.) * d.speed_mps
    d["accel_negative_w"] = MASS_KG * np.minimum(d.acceleration_smoothed_mps2, 0.) * d.speed_mps

    # Total stopped load already includes the user-confirmed 21 W weak-system draw.
    aux_w = float(d.loc[d["Speed[kmh]"].eq(0) & d.event_phase.isin(["road_day", "charge"])
                        & d.net_power_w.between(0., 250.), "net_power_w"].median())
    flags = d.event_flags.fillna("").astype(str)
    exclude = flags.str.contains(r"incident:|control_stop:|traffic_stop:|driver_change:|repair:|charge:|D6_ZP_ANOMALY|D6_RETIRE", regex=True)
    for start, end in [("2025-08-28 10:31:00", "2025-08-28 10:37:00"),
                       ("2025-08-28 11:51:00", "2025-08-28 12:50:00")]:
        exclude |= d.day.eq(5) & d.timestamp_local.between(start, end)
    eligible = (d.event_phase.eq("road_day") & ~exclude & d["Speed[kmh]"].between(45., 90.)
                & d.distance_anchor_confidence.isin(["high", "medium"])
                & d.net_power_w.between(100., 2600.)
                & d.acceleration_smoothed_mps2.between(-.50, .50)
                & d[["net_power_w", *TERMS, *ACC_TERMS]].notna().all(axis=1))
    q = d.loc[eligible].copy()
    q["minute_local"] = q.timestamp_local.dt.floor("min")
    aggregate_cols = ["net_power_w", *TERMS, *ACC_TERMS]
    minute = q.groupby(["day", "minute_local"], as_index=False).agg(
        **{c: (c, "mean") for c in aggregate_cols}, n_5s=("net_power_w", "size"))
    minute = minute[minute.n_5s.ge(6)].reset_index(drop=True)
    q = q.merge(minute[["day", "minute_local", "n_5s"]], on=["day", "minute_local"], how="inner")

    fold_rows, row_fold_rows = [], []
    predictions = []
    for held in sorted(int(v) for v in minute.day.unique()):
        if held == 6:
            train_days = [d0 for d0 in [1, 2, 3, 4, 5] if d0 in minute.day.unique()]
        else:
            train_days = [d0 for d0 in [1, 2, 3, 4, 5] if d0 != held and d0 in minute.day.unique()]
        tr, te = minute[minute.day.isin(train_days)], minute[minute.day.eq(held)]
        if held <= 5 and (tr.empty or te.empty):
            continue
        for name, use_accel in [("steady", False), ("signed_inertia", True)]:
            model = fit(tr, aux_w, use_accel)
            pred_minute = predict(te, model, aux_w)
            fold_rows.append({"held_out_day": held, "model": name, **metric(te.net_power_w.to_numpy(), pred_minute),
                              "train_minute_bins": len(tr), "test_minute_bins": len(te),
                              "coefficients_json": json.dumps(model["coefficients"], sort_keys=True)})
            row_test = q[q.day.eq(held)]
            pred_row = predict(row_test, model, aux_w)
            row_fold_rows.append({"held_out_day": held, "model": name, **metric(row_test.net_power_w.to_numpy(), pred_row),
                                  "test_5s_rows": len(row_test)})
            if held <= 5:
                predictions.append({"day": held, "model": name, "y": te.net_power_w.to_numpy(), "pred": pred_minute})

    cv = pd.DataFrame(fold_rows)
    row_cv = pd.DataFrame(row_fold_rows)
    cv["validation_grain"] = "1-minute mean; one race day held out"
    row_cv["validation_grain"] = "eligible 5-second rows; one race day held out; descriptive due autocorrelation"
    all_cv = pd.concat([cv, row_cv], ignore_index=True, sort=False)
    all_cv.to_csv(OUT_CV, index=False, encoding="utf-8-sig")
    summaries = {}
    for grain, table in [("one_minute", cv[cv.held_out_day.le(5)]), ("five_second", row_cv[row_cv.held_out_day.le(5)])]:
        summaries[grain] = table.groupby("model").agg(mean_day_mae_w=("mae_w", "mean"),
            mean_day_rmse_w=("rmse_w", "mean"), mean_day_bias_w=("bias_measured_minus_model_w", "mean"),
            folds=("held_out_day", "nunique")).to_dict(orient="index")
    result = {"scope": "BWSC2025 road cruise candidate; complete days 1-5 used in LODO; Day6 held out from all clean-day fitting",
              "auxiliary_total_w": aux_w, "eligible_5s_rows": len(q), "eligible_1minute_bins": len(minute),
              "minute_bin_filter": "at least 6 eligible 5-second rows per local minute",
              "candidate": "adds separate nonnegative coefficients for positive and negative acceleration mechanical power m*a*v; uses provided 314 kg floor-mass proxy",
              "baseline": "same 1-minute sample and Student-t fit, with no inertial terms",
              "summary_clean_lodo": summaries, "folds_minute": cv.to_dict(orient="records"),
              "folds_5s_descriptive": row_cv.to_dict(orient="records"),
              "decision": "adopt only if one-minute leave-one-day-out mean MAE improves and signed-inertia coefficients are stable in sign/magnitude; otherwise retain baseline and flag derivative as unidentifiable/noisy",
              "limitations": ["5-second logged speed is integer km/h and differentiated after smoothing; acceleration remains noisy.",
                              "Day folds protect temporal leakage, but source points remain serially correlated.",
                              "Coefficients are effective electrical-bus terms, not standalone motor efficiency or true vehicle mass.",
                              "The 5-second MAE is descriptive; independent evidence grain is the held-out day and 1-minute aggregation."]}
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"files": [str(OUT_CV), str(OUT_JSON)], "summary_clean_lodo": summaries,
                      "folds_minute": cv.to_dict(orient="records"), "folds_5s": row_cv.to_dict(orient="records")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
