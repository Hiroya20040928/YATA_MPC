#!/usr/bin/env python3
"""Independently recompute headline residual-model scores from saved OOF CSVs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/residual_learning_independent_audit.json"


def metrics(error: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(error, dtype=float)
    error = error[np.isfinite(error)]
    return {
        "n": int(error.size),
        "mae_w": float(np.mean(np.abs(error))),
        "p90_abs_error_w": float(np.quantile(np.abs(error), 0.9)),
        "bias_w": float(np.mean(error)),
    }


def paired(frame: pd.DataFrame, observed: str, prediction: str) -> dict:
    return metrics(frame[observed].to_numpy(float) - frame[prediction].to_numpy(float))


def per_day(frame: pd.DataFrame, observed: str, prediction: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for day, rows in frame.groupby("day", sort=True):
        result[str(int(day))] = paired(rows, observed, prediction)
    return result


def main() -> None:
    result: dict = {"recomputed_from_saved_out_of_fold_csv": True}

    five_min = pd.read_csv(ROOT / "reports/residual_online_kernel_oof.csv", low_memory=False)
    five_min = five_min[five_min.day.le(5)]
    result["five_minute_clean_days_1_5"] = {
        "physical_baseline": paired(five_min, "measured_w", "predicted_w"),
        "krr_plus_adaptive_median": paired(five_min, "measured_w", "krr_plus_tuned_bias_prediction_w"),
        "krr_plus_kalman": paired(five_min, "measured_w", "krr_plus_kalman_prediction_w"),
    }

    one_step = pd.read_csv(ROOT / "data/processed/full_race_residual_arx_5s.csv", low_memory=False)
    result["five_second_one_step"] = {}
    for scope, rows in [("clean_days_1_5", one_step[one_step.day.le(5)]),
                        ("day6_stress", one_step[one_step.day.eq(6)])]:
        result["five_second_one_step"][scope] = {
            "physical_baseline": paired(rows, "measured_w", "predicted_w"),
            "past_only_arx": paired(rows, "measured_w", "arx_prediction_w"),
            "past_only_arx_warm_state": paired(rows[rows.arx_prior_samples.ge(120)], "measured_w", "arx_prediction_w"),
        }

    tree = pd.read_csv(ROOT / "data/processed/full_race_residual_online_tree_5s.csv", low_memory=False,
                       usecols=["day", "timestamp_utc", "measured_w", "predicted_w", "tree_prediction_w", "tree_prior_samples"])
    arx = pd.read_csv(ROOT / "data/processed/full_race_residual_arx_5s.csv", low_memory=False,
                      usecols=["day", "timestamp_utc", "arx_prediction_w"])
    tree = tree.merge(arx, on=["day", "timestamp_utc"], how="left", validate="one_to_one")
    warm_tree = tree[tree.tree_prior_samples.ge(360)]
    result["tree_one_step"] = {
        "clean_days_1_5_all_rows": {
            "physical_baseline": paired(tree[tree.day.le(5)], "measured_w", "predicted_w"),
            "nonlinear_tree": paired(tree[tree.day.le(5)], "measured_w", "tree_prediction_w"),
        },
        "same_360_row_warm_support_days_1_5": {
            "n": int(len(warm_tree[warm_tree.day.le(5)])),
            "physical_baseline": paired(warm_tree[warm_tree.day.le(5)], "measured_w", "predicted_w"),
            "nonlinear_tree": paired(warm_tree[warm_tree.day.le(5)], "measured_w", "tree_prediction_w"),
            "causal_arx": paired(warm_tree[warm_tree.day.le(5)], "measured_w", "arx_prediction_w"),
        },
    }

    tree_horizon = pd.read_csv(ROOT / "reports/residual_tree_multihorizon_forecasts.csv", low_memory=False)
    result["tree_conditional_horizons"] = {}
    for h, rows in tree_horizon.groupby("horizon_s", sort=True):
        result["tree_conditional_horizons"][f"{int(h)}_s"] = {
            "clean_days_1_5": {
                "physical_baseline": paired(rows[rows.day.le(5)], "measured_future_w", "baseline_future_w"),
                "tree_rollout": paired(rows[rows.day.le(5)], "measured_future_w", "tree_future_w"),
            },
            "day6_stress": {
                "physical_baseline": paired(rows[rows.day.eq(6)], "measured_future_w", "baseline_future_w"),
                "tree_rollout": paired(rows[rows.day.eq(6)], "measured_future_w", "tree_future_w"),
            },
        }

    horizon = pd.read_csv(ROOT / "reports/residual_multihorizon_arx_forecasts.csv", low_memory=False)
    result["conditional_horizons"] = {}
    for h, horizon_rows in horizon.groupby("horizon_s", sort=True):
        result["conditional_horizons"][f"{int(h)}_s"] = {}
        for scope, rows in [("clean_days_1_5", horizon_rows[horizon_rows.day.le(5)]),
                            ("day6_stress", horizon_rows[horizon_rows.day.eq(6)])]:
            result["conditional_horizons"][f"{int(h)}_s"][scope] = {
                "physical_baseline": paired(rows, "measured_future_w", "baseline_future_w"),
                "past_only_arx": paired(rows, "measured_future_w", "arx_future_w"),
            }

    common = tree_horizon.merge(
        horizon[["day", "horizon_s", "target_utc", "arx_future_w"]],
        on=["day", "horizon_s", "target_utc"], how="inner", validate="one_to_one",
    )
    result["same_targets_tree_vs_arx"] = {}
    for h, rows in common.groupby("horizon_s", sort=True):
        clean_rows = rows[rows.day.le(5)]
        result["same_targets_tree_vs_arx"][f"{int(h)}_s"] = {
            "clean_days_1_5_n": int(len(clean_rows)),
            "tree_rollout": paired(clean_rows, "measured_future_w", "tree_future_w"),
            "causal_arx": paired(clean_rows, "measured_future_w", "arx_future_w"),
            "tree_rollout_by_day": per_day(clean_rows, "measured_future_w", "tree_future_w"),
            "causal_arx_by_day": per_day(clean_rows, "measured_future_w", "arx_future_w"),
        }

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
