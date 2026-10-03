#!/usr/bin/env python3
"""Compare available residual learners on identical untouched sweep timestamps."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "reports/residual_method_sweep_oof_predictions.csv"
TCN = ROOT / "reports/residual_tcn_oof_predictions.csv"
OUT_CSV = ROOT / "reports/residual_common_test_comparison.csv"
OUT_JSON = ROOT / "reports/residual_common_test_comparison.json"
KEYS = ["day", "timestamp_utc"]


def load_model(path: str, name: str, prediction_col: str) -> pd.DataFrame:
    frame = pd.read_csv(ROOT / path, low_memory=False)
    frame["timestamp_utc"] = pd.to_datetime(frame.timestamp_utc, utc=True, errors="coerce")
    out = frame[KEYS + [prediction_col]].copy()
    out = out.rename(columns={prediction_col: name})
    return out.drop_duplicates(KEYS)


def score(y: np.ndarray, correction: np.ndarray) -> dict:
    e = np.asarray(y, float) - np.asarray(correction, float)
    return {"n": int(len(e)), "mae_w": float(np.mean(np.abs(e))),
            "rmse_w": float(np.sqrt(np.mean(e**2))),
            "p90_abs_error_w": float(np.quantile(np.abs(e), .9)),
            "bias_w": float(np.mean(e))}


def main() -> None:
    base = pd.read_csv(SWEEP, low_memory=False)
    base["timestamp_utc"] = pd.to_datetime(base.timestamp_utc, utc=True, errors="coerce")
    comparison = base.rename(columns={"sweep_prediction_w": "HGB_full_feature_sweep"})
    comparison = comparison[KEYS + ["target_residual_w", "HGB_full_feature_sweep", "sweep_scope"]]
    comparison["physical_baseline"] = 0.0
    sources = [
        ("data/processed/full_race_residual_arx_5s.csv", "causal_ARX", "arx_correction_w"),
        ("data/processed/full_race_residual_online_tree_5s.csv", "online_tree", "tree_correction_w"),
        ("data/processed/full_race_residual_prequential_5s.csv", "adaptive_filter", "residual_correction_w"),
        ("data/processed/full_race_residual_online_kernel_5s.csv", "KRR_plus_adaptive_bias", "krr_plus_tuned_bias_correction_w"),
        ("data/processed/full_race_residual_gp_online_hybrid_5s.csv", "GP_plus_online_filter", "hybrid_total_correction_w"),
        ("data/processed/full_race_residual_online_filters_5s.csv", "filter_family", "filter_correction_w"),
    ]
    for path, name, col in sources:
        mod = load_model(path, name, col)
        comparison = comparison.merge(mod, on=KEYS, how="left", validate="one_to_one")
    try:
        tcn = load_model("reports/residual_tcn_oof_predictions.csv", "TCN", "tcn_correction_w")
        comparison = comparison.merge(tcn, on=KEYS, how="left", validate="one_to_one")
    except FileNotFoundError:
        pass

    metrics = []
    model_cols = [c for c in comparison.columns if c not in KEYS + ["target_residual_w", "sweep_scope"]]
    for col in model_cols:
        valid = comparison[col].notna()
        for scope, mask in [("clean_days_1_5", comparison.day.le(5)),
                            ("day6_stress", comparison.day.eq(6))]:
            rows = comparison.loc[valid & mask]
            metrics.append({"model": col, "scope": scope, **score(rows.target_residual_w, rows[col])})
    pd.DataFrame(metrics).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    payload = {"scope": "Exact same timestamps as the full-feature sweep's untouched final 20% per day",
               "comparison_rows": int(len(comparison)),
               "metrics": metrics,
               "per_day": {col: {str(int(day)): score(rows.target_residual_w, rows[col])
                                 for day, rows in comparison[comparison[col].notna()].groupby("day", sort=True)}
                           for col in model_cols}}
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
