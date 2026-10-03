#!/usr/bin/env python3
"""Test per-day adaptive algorithm selection on untouched final day segments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_residual_method_sweep import (
    ROOT, OUT_CSV, FEATURES, prepare_data, specs, day_splits, fit_score, metric,
)

OUT_ROWS = ROOT / "reports/residual_method_sweep_daywise_test.csv"
OUT_SUMMARY = ROOT / "reports/residual_method_sweep_daywise_summary.json"


def main() -> None:
    data = prepare_data()
    score_table = pd.read_csv(OUT_CSV, low_memory=False)
    all_specs = {s["name"]: s for s in specs()}
    rows, predictions = [], []
    for day, frame in data.groupby("day", sort=True):
        frame = frame.reset_index(drop=True)
        i_train, i_val, i_test = day_splits(frame)
        config_row = (score_table[score_table.day.eq(day)]
                      .sort_values(["mae_w", "configuration"]).iloc[0])
        spec = all_specs[str(config_row.configuration)]
        x = frame[FEATURES].to_numpy(float)
        y = frame.target_residual_w.to_numpy(float)
        fit, test = np.concatenate([i_train, i_val]), i_test
        model, test_score = fit_score(spec, x[fit], y[fit], x[test], y[test])
        base = metric(y[test], np.zeros(len(test)))
        rows.append({"day": int(day), "selected_family": spec["family"],
                     "selected_configuration": spec["name"],
                     "selection_validation_mae_w": float(config_row.mae_w),
                     "outer_test_baseline": base, "outer_test_model": test_score})
        out = frame.iloc[test][["day", "timestamp_utc"]].copy()
        out["target_residual_w"] = y[test]
        out["prediction_w"] = model.predict(x[test])
        out["corrected_error_w"] = out.target_residual_w - out.prediction_w
        out["selected_family"] = spec["family"]
        out["selected_configuration"] = spec["name"]
        predictions.append(out)
    prediction_table = pd.concat(predictions, ignore_index=True)
    prediction_table.to_csv(OUT_ROWS, index=False, encoding="utf-8-sig")
    report_rows = []
    for scope, subset in [("clean_days_1_5", prediction_table[prediction_table.day.le(5)]),
                          ("day6_stress", prediction_table[prediction_table.day.eq(6)])]:
        if len(subset):
            e = subset.corrected_error_w.to_numpy(float)
            b = subset.target_residual_w.to_numpy(float)
            report_rows.append({"scope": scope, "n": int(len(e)),
                "baseline_mae_w": float(np.mean(np.abs(b))),
                "adaptive_selector_mae_w": float(np.mean(np.abs(e))),
                "baseline_p90_w": float(np.quantile(np.abs(b), .9)),
                "adaptive_selector_p90_w": float(np.quantile(np.abs(e), .9)),
                "days_won": int(sum(r["day"] <= 5 and r["outer_test_model"]["mae_w"] < r["outer_test_baseline"]["mae_w"] for r in rows)) if scope == "clean_days_1_5" else None})
    summary = {"selection": "for each day, choose among the full method/configuration grid using that day's 20% validation segment; refit on the first 80%; score only the final untouched 20%",
               "per_day_results": rows, "aggregate": report_rows,
               "interpretation": "This is a within-day adaptive selector. Its test blocks remain untouched, but its chosen algorithm may differ by day and need not transfer to a future route/day."}
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
