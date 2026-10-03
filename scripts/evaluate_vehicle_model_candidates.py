#!/usr/bin/env python3
"""Compare bounded, interpretable 10 km road-load candidate models by held-out day."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/vehicle_energy_segments_10km.csv"
FIT = ROOT / "models/vehicle_power_model_fit.json"
OUT = ROOT / "reports/vehicle_model_candidate_cv.csv"
SUMMARY = ROOT / "reports/vehicle_model_candidate_cv.json"
DF = 4.0


def features(d: pd.DataFrame, candidate: str) -> tuple[np.ndarray, list[str]]:
    base = ["roll_w", "aero_w"]
    if candidate == "integrated_grade":
        cols = base + ["integrated_grade_w"]
    elif candidate == "plus_speed":
        cols = base + ["grade_endpoint_w", "speed_kmh"]
    elif candidate == "plus_speed2":
        cols = base + ["grade_endpoint_w", "speed_kmh", "speed_sq_100"]
    else:
        cols = base + ["grade_endpoint_w"]
    return d[cols].to_numpy(float), cols


def fit(x: np.ndarray, y: np.ndarray, p_aux: float) -> dict:
    def objective(z: np.ndarray) -> float:
        beta, scale = z[:-1], z[-1]
        residual = (y - p_aux - x @ beta) / scale
        return float(-np.sum(student_t.logpdf(residual, df=DF) - np.log(scale)))

    init = np.r_[np.ones(x.shape[1]), 50.0]
    bounds = [(0.0, 5.0)] * x.shape[1] + [(10.0, 500.0)]
    opt = minimize(objective, init, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 1000, "ftol": 1e-11})
    return {"beta": opt.x[:-1], "scale": float(opt.x[-1]), "success": bool(opt.success)}


def predict(x: np.ndarray, beta: np.ndarray, p_aux: float) -> np.ndarray:
    return p_aux + x @ beta


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    e = y - p
    return {"n": int(len(e)), "mae_w": float(np.mean(np.abs(e))),
            "rmse_w": float(np.sqrt(np.mean(e ** 2))), "bias_w": float(np.mean(e))}


def main() -> None:
    d = pd.read_csv(INPUT)
    model = json.loads(FIT.read_text(encoding="utf-8"))
    p_aux = float(model["p_aux_w_stationary_median"])
    d["integrated_grade_w"] = d["grade_wh"] * 3600.0 / d["time_s"]
    d["speed_sq_100"] = d["speed_kmh"] ** 2 / 100.0
    clean = d[d.day.isin([1, 2, 3, 4, 5])].copy().reset_index(drop=True)
    candidates = ["baseline", "integrated_grade", "plus_speed", "plus_speed2"]
    rows: list[dict] = []
    for candidate in candidates:
        oof = np.full(len(clean), np.nan)
        fold_params = []
        for day in [1, 2, 3, 4, 5]:
            train = clean[clean.day.ne(day)]
            test = clean[clean.day.eq(day)]
            x_train, cols = features(train, candidate)
            x_test, _ = features(test, candidate)
            fitted = fit(x_train, train.measured_net_w.to_numpy(float), p_aux)
            idx = clean.index[clean.day.eq(day)]
            oof[idx] = predict(x_test, fitted["beta"], p_aux)
            fold_params.append({"held_out_day": day, "features": cols,
                                "coefficients": [float(v) for v in fitted["beta"]],
                                "scale_w": fitted["scale"], "fit_success": fitted["success"],
                                **metrics(test.measured_net_w.to_numpy(float), oof[idx])})
        all_metrics = metrics(clean.measured_net_w.to_numpy(float), oof)
        rows.append({"candidate": candidate, **all_metrics,
                     "day1_5_mean_fold_mae_w": float(np.mean([r["mae_w"] for r in fold_params])),
                     "day6_metrics": None, "folds_json": json.dumps(fold_params)})

    # Day 6 is reported as a diagnostic holdout, but it is anomalous and is not
    # used for candidate selection or refitting.
    day6 = d[d.day.eq(6)].copy()
    clean_all = d[d.day.isin([1, 2, 3, 4, 5])].copy()
    for candidate, row in zip(candidates, rows):
        x_train, _ = features(clean_all, candidate)
        x_test, _ = features(day6, candidate)
        fitted = fit(x_train, clean_all.measured_net_w.to_numpy(float), p_aux)
        m6 = metrics(day6.measured_net_w.to_numpy(float), predict(x_test, fitted["beta"], p_aux))
        row["day6_metrics"] = json.dumps(m6)
        row["full_fit_coefficients"] = json.dumps([float(v) for v in fitted["beta"]])
    pd.DataFrame(rows).drop(columns=["folds_json"]).to_csv(OUT, index=False, encoding="utf-8-sig")
    payload = {
        "source": str(INPUT.relative_to(ROOT)),
        "target": "10 km segment mean HV power (Solar_Watt + Battery_Watt)",
        "validation": "Leave-one-day-out on Days 1-5; Day 6 anomaly is diagnostic only.",
        "p_aux_w": p_aux,
        "selection_note": "Candidate coefficients are effective predictive terms, not identified physical truth. Prefer only changes that improve held-out days without relying on measured target features.",
        "results": rows,
    }
    SUMMARY.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(f"{row['candidate']}: LODO MAE={row['mae_w']:.2f} W, RMSE={row['rmse_w']:.2f} W, "
              f"bias={row['bias_w']:+.2f} W; Day6={row['day6_metrics']}")


if __name__ == "__main__":
    main()
