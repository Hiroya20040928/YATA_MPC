#!/usr/bin/env python3
"""Audit a simple residual forecast blend on matched tree/ARX horizons.

The tree and ARX predictions are already generated prequentially. Blend weights
are selected leave-one-day-out on clean days, then scored on the held-out day.
Day 6 is evaluated with a weight selected only from Days 1-5.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TREE_PATH = ROOT / "reports/residual_tree_multihorizon_forecasts.csv"
ARX_PATH = ROOT / "reports/residual_multihorizon_arx_forecasts.csv"
OUT = ROOT / "reports/residual_horizon_blend_audit.json"
WEIGHTS = np.linspace(0.0, 1.0, 21)  # tree share; ARX share is 1 - weight


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def score(rows: pd.DataFrame, prediction: str) -> dict[str, float | int]:
    y = rows.measured_future_w.to_numpy(float)
    e = y - rows[prediction].to_numpy(float)
    return {"n": int(len(e)), "mae_w": mae(y, rows[prediction].to_numpy(float)),
            "p90_abs_error_w": float(np.quantile(np.abs(e), .9)),
            "bias_w": float(np.mean(e))}


def main() -> None:
    tree = pd.read_csv(TREE_PATH, low_memory=False)
    arx = pd.read_csv(ARX_PATH, low_memory=False)
    common = tree.merge(arx[["day", "horizon_s", "target_utc", "arx_future_w"]],
                        on=["day", "horizon_s", "target_utc"], how="inner",
                        validate="one_to_one")
    common["ensemble_future_w"] = np.nan
    selected: dict[str, dict] = {}
    summary: dict[str, dict] = {}
    for horizon, rows in common.groupby("horizon_s", sort=True):
        clean = rows[rows.day.le(5)]
        fold_details = {}
        for held_day in sorted(clean.day.unique()):
            train = clean[clean.day.ne(held_day)]
            test = clean[clean.day.eq(held_day)]
            y_train = train.measured_future_w.to_numpy(float)
            tree_train = train.tree_future_w.to_numpy(float)
            arx_train = train.arx_future_w.to_numpy(float)
            w = min(WEIGHTS, key=lambda x: mae(y_train, x * tree_train + (1-x) * arx_train))
            idx = common.index[(common.day.eq(held_day)) & (common.horizon_s.eq(horizon))]
            common.loc[idx, "ensemble_future_w"] = (
                w * common.loc[idx, "tree_future_w"] + (1-w) * common.loc[idx, "arx_future_w"])
            fold_details[str(int(held_day))] = {"tree_weight_selected_from_other_days": float(w),
                                                "test_tree": score(test, "tree_future_w"),
                                                "test_arx": score(test, "arx_future_w")}
        # Day 6 is a prospective stress evaluation: its blend is calibrated on all clean days.
        d6 = rows[rows.day.eq(6)]
        if len(d6):
            y_train = clean.measured_future_w.to_numpy(float)
            tree_train = clean.tree_future_w.to_numpy(float)
            arx_train = clean.arx_future_w.to_numpy(float)
            w6 = min(WEIGHTS, key=lambda x: mae(y_train, x * tree_train + (1-x) * arx_train))
            idx = common.index[(common.day.eq(6)) & (common.horizon_s.eq(horizon))]
            common.loc[idx, "ensemble_future_w"] = (
                w6 * common.loc[idx, "tree_future_w"] + (1-w6) * common.loc[idx, "arx_future_w"])
        else:
            w6 = None
        scored_clean = common[(common.horizon_s.eq(horizon)) & common.day.le(5)]
        summary[str(int(horizon))] = {
            "matched_clean_targets_n": int(len(clean)),
            "tree": score(clean, "tree_future_w"),
            "arx": score(clean, "arx_future_w"),
            "leave_one_day_out_blend": score(scored_clean, "ensemble_future_w"),
            "folds": fold_details,
            "day6_weight_selected_from_days_1_5": w6,
            "day6_tree": score(d6, "tree_future_w") if len(d6) else None,
            "day6_arx": score(d6, "arx_future_w") if len(d6) else None,
            "day6_blend": score(common[(common.horizon_s.eq(horizon)) & common.day.eq(6)],
                                 "ensemble_future_w") if len(d6) else None,
        }
    OUT.write_text(json.dumps({"method": "leave-one-clean-day-out blend weight; 21 weights from 0 to 1; Day 6 weight trained on Days 1-5",
                               "matched_targets_across_all_horizons": int(len(common)),
                               "results": summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUT)
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
