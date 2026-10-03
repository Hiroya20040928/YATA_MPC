#!/usr/bin/env python3
"""Fit constrained MAE-optimal residual blends on validation, score untouched tails."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from evaluate_residual_method_sweep import ROOT, FEATURES, prepare_data, specs, day_splits, fit_score

OUT_ROWS = ROOT / "reports/residual_stacking_oof_predictions.csv"
OUT_JSON = ROOT / "reports/residual_stacking_summary.json"
OUT_VAL = ROOT / "reports/residual_stacking_validation_predictions.csv"
KEYS = ["day", "timestamp_utc"]
BASE_MODEL_COLS = ["HGB", "ARX", "Tree", "Filter", "Zero"]


def load_correction(path: str, name: str, col: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / path, low_memory=False)
    d["timestamp_utc"] = pd.to_datetime(d.timestamp_utc, utc=True, errors="coerce")
    return d[KEYS + [col]].drop_duplicates(KEYS).rename(columns={col: name})


def metric(y: np.ndarray, p: np.ndarray) -> dict:
    e = np.asarray(y, float) - np.asarray(p, float)
    return {"n": int(len(e)), "mae_w": float(np.mean(np.abs(e))),
            "rmse_w": float(np.sqrt(np.mean(e**2))),
            "p90_abs_error_w": float(np.quantile(np.abs(e), .9)),
            "bias_w": float(np.mean(e))}


def simplex_mae_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve min ||y-Xw||_1 subject to w>=0 and sum(w)=1."""
    n, k = x.shape
    c = np.r_[np.zeros(k), np.ones(n)]
    upper = np.vstack([np.c_[x, -np.eye(n)], np.c_[-x, -np.eye(n)]])
    rhs = np.r_[y, -y]
    equality = np.r_[np.ones(k), np.zeros(n)][None, :]
    solved = linprog(c, A_ub=upper, b_ub=rhs, A_eq=equality, b_eq=[1.0],
                     bounds=[(0, 1)] * k + [(0, None)] * n, method="highs")
    if not solved.success:
        raise RuntimeError(f"blend optimization failed: {solved.message}")
    return solved.x[:k]


def main() -> None:
    data = prepare_data().reset_index(drop=True)
    summary = json.loads((ROOT / "reports/residual_method_sweep_summary.json").read_text(encoding="utf-8"))
    winner_name = summary["global_winner"]["configuration"]
    winner = next(s for s in specs() if s["name"] == winner_name)
    val_chunks = []
    test_chunks = []
    for day, frame in data.groupby("day", sort=True):
        frame = frame.reset_index(drop=True)
        tr, va, te = day_splits(frame)
        x = frame[FEATURES].to_numpy(float)
        y = frame.target_residual_w.to_numpy(float)
        model, _ = fit_score(winner, x[tr], y[tr], x[va], y[va])
        v = frame.iloc[va][KEYS + ["target_residual_w"]].copy()
        v["HGB"] = model.predict(x[va])
        v["validation_scope"] = "day6_stress_validation" if day == 6 else "clean_validation"
        val_chunks.append(v)
        t = pd.read_csv(ROOT / "reports/residual_method_sweep_oof_predictions.csv", low_memory=False)
        t["timestamp_utc"] = pd.to_datetime(t.timestamp_utc, utc=True, errors="coerce")
        t = t[t.day.eq(day)].copy()
        test_chunks.append(t)
    val = pd.concat(val_chunks, ignore_index=True)
    test = pd.concat(test_chunks, ignore_index=True)
    test["HGB"] = test.sweep_prediction_w
    for path, name, col in [
        ("data/processed/full_race_residual_arx_5s.csv", "ARX", "arx_correction_w"),
        ("data/processed/full_race_residual_online_tree_5s.csv", "Tree", "tree_correction_w"),
        ("data/processed/full_race_residual_prequential_5s.csv", "Filter", "residual_correction_w"),
    ]:
        val = val.merge(load_correction(path, name, col), on=KEYS, how="left", validate="one_to_one")
        test = test.merge(load_correction(path, name, col), on=KEYS, how="left", validate="one_to_one")
    val["Zero"] = 0.0
    test["Zero"] = 0.0
    val["Bias"] = val.groupby("day").target_residual_w.transform("median")
    test = test.merge(val.groupby("day", as_index=False).target_residual_w.median()
                      .rename(columns={"target_residual_w": "Bias"}), on="day", how="left", validate="many_to_one")
    candidates = BASE_MODEL_COLS + ["Bias"]
    clean_val = val[val.day.le(5)].dropna(subset=candidates)
    global_w = simplex_mae_weights(clean_val[candidates].to_numpy(float), clean_val.target_residual_w.to_numpy(float))
    val["global_blend_correction_w"] = val[candidates].to_numpy(float) @ global_w
    val["global_blend_error_w"] = val.target_residual_w - val.global_blend_correction_w
    test["global_blend_correction_w"] = test[candidates].to_numpy(float) @ global_w
    per_day_weights = {}
    per_day_bias = {}
    for day in range(1, 6):
        va = val[val.day.eq(day)].dropna(subset=candidates)
        te_mask = test.day.eq(day)
        w = simplex_mae_weights(va[candidates].to_numpy(float), va.target_residual_w.to_numpy(float))
        val.loc[val.day.eq(day), "daywise_blend_correction_w"] = val.loc[val.day.eq(day), candidates].to_numpy(float) @ w
        val.loc[val.day.eq(day), "daywise_blend_error_w"] = val.loc[val.day.eq(day), "target_residual_w"] - val.loc[val.day.eq(day), "daywise_blend_correction_w"]
        test.loc[te_mask, "daywise_blend_correction_w"] = test.loc[te_mask, candidates].to_numpy(float) @ w
        per_day_weights[str(day)] = {name: float(value) for name, value in zip(candidates, w)}
        per_day_bias[str(day)] = float(val.loc[val.day.eq(day), "daywise_blend_error_w"].median())
    # Day 6 uses weights fit only on clean-day validation residuals.
    test.loc[test.day.eq(6), "daywise_blend_correction_w"] = test.loc[test.day.eq(6), "global_blend_correction_w"]
    global_bias = float(clean_val.target_residual_w.sub(clean_val[candidates].to_numpy(float) @ global_w).median())
    test["global_bias_adjusted_correction_w"] = test.global_blend_correction_w + global_bias
    test["daywise_bias_adjusted_correction_w"] = test.daywise_blend_correction_w
    for day, bias in per_day_bias.items():
        test.loc[test.day.eq(int(day)), "daywise_bias_adjusted_correction_w"] += bias
    test.loc[test.day.eq(6), "daywise_bias_adjusted_correction_w"] += global_bias
    global_weights = {name: float(value) for name, value in zip(candidates, global_w)}
    test["global_blend_error_w"] = test.target_residual_w - test.global_blend_correction_w
    test["daywise_blend_error_w"] = test.target_residual_w - test.daywise_blend_correction_w
    test.to_csv(OUT_ROWS, index=False, encoding="utf-8-sig")
    val.to_csv(OUT_VAL, index=False, encoding="utf-8-sig")

    results = {}
    for scope, rows in [("clean_days_1_5", test[test.day.le(5)]), ("day6_stress", test[test.day.eq(6)])]:
        results[scope] = {"n": int(len(rows)), "physical_baseline": metric(rows.target_residual_w, np.zeros(len(rows))),
                          "HGB": metric(rows.target_residual_w, rows.sweep_prediction_w),
                          "ARX": metric(rows.target_residual_w, rows.ARX),
                          "Tree": metric(rows.target_residual_w, rows.Tree),
                          "Filter": metric(rows.target_residual_w, rows.Filter),
                          "global_validation_blend": metric(rows.target_residual_w, rows.global_blend_correction_w),
                          "daywise_validation_blend": metric(rows.target_residual_w, rows.daywise_blend_correction_w),
                          "global_blend_plus_validation_median_error": metric(rows.target_residual_w, rows.global_bias_adjusted_correction_w),
                          "daywise_blend_plus_validation_median_error": metric(rows.target_residual_w, rows.daywise_bias_adjusted_correction_w)}
    payload = {"selection": "MAE-optimal convex blend (weights nonnegative and sum to one), fit on middle validation block; final per-day 20% remains untouched",
               "candidate_corrections": candidates, "global_weights_clean_day_validation": global_weights,
               "daywise_weights_clean_day_validation": per_day_weights,
               "global_blend_median_validation_error_w": global_bias,
               "daywise_blend_median_validation_error_w": per_day_bias,
               "day6_uses_global_clean_day_weights": True, "results": results,
               "interpretation": "Any blend is selected before the final test block. This is one-step conditional scoring with observed past residuals and logged future covariates.",
               "outputs": [str(OUT_ROWS), str(OUT_VAL), str(OUT_JSON)]}
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
