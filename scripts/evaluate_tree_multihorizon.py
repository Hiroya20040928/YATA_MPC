#!/usr/bin/env python3
"""Roll a past-only tree residual model forward without future residual updates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_online_tree_residual import FEATURES, MIN_HISTORY, MODELS, estimator, impute

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/full_race_residual_online_tree_5s.csv"
OUT = ROOT / "reports/residual_tree_multihorizon_forecasts.csv"
OUT_SUMMARY = ROOT / "reports/residual_tree_multihorizon_summary.json"
OUT_REPORT = ROOT / "reports/residual_tree_multihorizon_report.md"
HORIZONS_S = [5, 30, 60, 300]
RESET_GAP_S = 600.0
REFIT_STEP_S = 12


def parse_configuration(value: str):
    model_name, window_text = value.rsplit("_w", maxsplit=1)
    spec = next(spec for spec in MODELS if spec[0] == model_name)
    return spec, int(window_text)


def contiguous(times: np.ndarray, first: int, last: int) -> bool:
    if last <= first:
        return True
    delta = np.diff(times[first:last + 1])
    return bool(np.all(np.isfinite(delta)) and np.all(np.abs(delta - 5.0) <= 1.0))


def forecast_day(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.sort_values("timestamp_utc").copy().reset_index(drop=True)
    g["timestamp_utc"] = pd.to_datetime(g.timestamp_utc, utc=True, errors="coerce")
    times = (g.timestamp_utc - g.timestamp_utc.iloc[0]).dt.total_seconds().to_numpy(float)
    x_all = g[FEATURES].to_numpy(float)
    residual = g.baseline_error_w.to_numpy(float)
    configs = g.tree_selected_model.astype(str).to_numpy()
    day = int(g.day.iloc[0])
    rows = []
    stream_start = 0
    last_origin = -1
    for i in range(len(g)):
        if i > 0 and times[i] - times[i - 1] > RESET_GAP_S:
            stream_start = i
            last_origin = -1
        n = i - stream_start
        if n < MIN_HISTORY or n % 300 != 0 or i == last_origin:
            continue
        if configs[i] == "warmup_physics" or not np.isfinite(residual[i]):
            continue
        spec, window = parse_configuration(configs[i])
        train_start = max(stream_start, i + 1 - window)
        x_train = x_all[train_start:i + 1]
        y_train = residual[train_start:i + 1]
        if len(y_train) < 120 or not np.isfinite(y_train).all():
            continue
        x_train, _ = impute(x_train, x_train[:1])
        model = estimator(spec, seed=0)
        model.fit(x_train, y_train)
        last_residual = float(residual[i])
        max_steps = max(HORIZONS_S) // 5
        predictions = {}
        for step in range(1, max_steps + 1):
            target = i + step
            if target >= len(g) or not contiguous(times, i, target):
                break
            xrow = x_all[target].copy()
            xrow[0] = last_residual
            _, xrow = impute(x_train, xrow.reshape(1, -1))
            last_residual = float(model.predict(xrow)[0])
            horizon = step * 5
            if horizon in HORIZONS_S:
                predictions[horizon] = (last_residual, target)
        for horizon, (resid_pred, target) in predictions.items():
            rows.append({
                "day": day,
                "horizon_s": horizon,
                "origin_utc": g.timestamp_utc.iloc[i].isoformat(),
                "target_utc": g.timestamp_utc.iloc[target].isoformat(),
                "measured_future_w": float(g.measured_w.iloc[target]),
                "baseline_future_w": float(g.predicted_w.iloc[target]),
                "tree_residual_forecast_w": resid_pred,
                "tree_future_w": float(g.predicted_w.iloc[target]) + resid_pred,
                "selected_model": configs[i],
                "training_rows": int(len(y_train)),
            })
        last_origin = i
    return pd.DataFrame(rows)


def metrics(frame: pd.DataFrame, observed: str, prediction: str) -> dict:
    err = frame[observed].to_numpy(float) - frame[prediction].to_numpy(float)
    return {"n": int(len(err)), "mae_w": float(np.mean(np.abs(err))),
            "p90_abs_error_w": float(np.quantile(np.abs(err), 0.9)),
            "bias_w": float(np.mean(err))}


def main() -> None:
    data = pd.read_csv(INPUT, low_memory=False)
    forecasts = pd.concat([forecast_day(group.copy()) for _, group in data.groupby("day", sort=True)], ignore_index=True)
    forecasts.to_csv(OUT, index=False, encoding="utf-8-sig")
    summary = {
        "validation": "tree configuration comes from past-only nested one-step tuning; at each origin the fitted tree uses data available through the current measured residual; future residual measurements are withheld and the model residual is recursively fed forward",
        "future_inputs": "logged future speed, inertial proxy, route/weather features are treated as known; errors in MPC speed, weather and route forecasts are excluded",
        "origin_period_seconds": 1500,
        "minimum_history_rows": MIN_HISTORY,
        "horizons": {},
    }
    for horizon, group in forecasts.groupby("horizon_s", sort=True):
        summary["horizons"][f"{int(horizon)}_s"] = {}
        for scope, rows in [("clean_days_1_5", group[group.day.le(5)]), ("day6_stress", group[group.day.eq(6)])]:
            summary["horizons"][f"{int(horizon)}_s"][scope] = {
                "physical_baseline": metrics(rows, "measured_future_w", "baseline_future_w"),
                "tree_residual_rollout": metrics(rows, "measured_future_w", "tree_future_w"),
            }
    summary["outputs"] = [str(OUT), str(OUT_SUMMARY)]
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Tree residual rollout by prediction horizon", "",
             "At each origin, the model uses residual measurements through the origin. After that, it receives no residual updates. Predicted residuals are recursively fed into the next step. Future speed, route and weather inputs are taken from the logged path and treated as exact; this is a conditional test, not a complete MPC forecast.", "",
             "| Horizon | Day1–5 n | Baseline MAE | Tree rollout MAE | Baseline P90 | Tree P90 | Day6 baseline→tree MAE | Day6 baseline→tree P90 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for horizon, values in summary["horizons"].items():
        clean = values["clean_days_1_5"]
        stress = values["day6_stress"]
        lines.append(f"| {horizon.removesuffix('_s')} s | {clean['physical_baseline']['n']} | {clean['physical_baseline']['mae_w']:.1f} W | {clean['tree_residual_rollout']['mae_w']:.1f} W | {clean['physical_baseline']['p90_abs_error_w']:.1f} W | {clean['tree_residual_rollout']['p90_abs_error_w']:.1f} W | {stress['physical_baseline']['mae_w']:.1f}→{stress['tree_residual_rollout']['mae_w']:.1f} W | {stress['physical_baseline']['p90_abs_error_w']:.1f}→{stress['tree_residual_rollout']['p90_abs_error_w']:.1f} W |")
    lines += ["", "Origins are sampled every 25 minutes after warm-up. The forecast windows overlap, so the rows are not independent samples; day-level differences and tail behavior matter more than nominal row counts.", "",
              f"Detailed forecasts: `{OUT.relative_to(ROOT)}`.", ""]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
