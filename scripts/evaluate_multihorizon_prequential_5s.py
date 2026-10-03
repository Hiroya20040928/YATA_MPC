#!/usr/bin/env python3
"""Tune causal residual filters specifically for 5 s, 30 s, 60 s and 5 min leads."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/full_race_residual_corrected_5s.csv"
OUT_CSV = ROOT / "reports/residual_multihorizon_prequential_5s.csv"
OUT_JSON = ROOT / "reports/residual_multihorizon_prequential_5s_summary.json"
OUT_REPORT = ROOT / "reports/residual_multihorizon_prequential_5s_report.md"
HORIZONS_S = [5, 30, 60, 300]
MIN_HISTORY = 120
MIN_VALIDATION_HISTORY = 20
TUNE_EVERY = 60
INNER_HOLDOUT = 60
GAP_RESET_S = 600
CAPS_W = [100.0, 300.0, 500.0, 1000.0]
CONFIGS = ([ ("zero", 0.0) ]
           + [ ("median", w) for w in (12, 60, 300) ]
           + [ ("persistence", phi) for phi in (0.25, 0.5, 0.75, 1.0) ]
           + [ ("ar1", n) for n in (60, 120, 300) ])


def forecast_h(past: np.ndarray, kind: str, parameter: float, steps: int, cap: float) -> float:
    y = np.asarray(past, dtype=float)
    if kind == "zero" or len(y) == 0:
        return 0.0
    if kind == "median":
        return float(np.clip(np.median(y[-int(parameter):]), -cap, cap))
    if kind == "persistence":
        state = float(y[-1])
        for _ in range(steps):
            state = float(np.clip(parameter * state, -cap, cap))
        return state
    lookback = min(int(parameter), len(y) - 1)
    if lookback < 20:
        return float(np.clip(np.median(y[-min(12, len(y)):]), -cap, cap))
    target = y[-lookback:]
    lag = y[-lookback-1:-1]
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(lag)), lag]), target, rcond=None)[0]
    beta[1] = np.clip(beta[1], -0.25, 0.98)
    state = float(y[-1])
    for _ in range(steps):
        state = float(np.clip(beta[0] + beta[1] * state, -cap, cap))
    return state


def tune_horizon(history: list[float], future_offset: int) -> tuple[str, float, float, float]:
    """Tune for targets that are already observed, using only past forecast origins."""
    end = len(history) - future_offset
    start = max(MIN_VALIDATION_HISTORY, end - INNER_HOLDOUT)
    if end <= start:
        return "zero", 0.0, 1000.0, float("nan")
    y = np.asarray(history, dtype=float)
    candidates = []
    for kind, parameter in CONFIGS:
        predictions = [forecast_h(y[:j], kind, parameter, future_offset + 1, 1000.0)
                       for j in range(start, end)]
        targets = y[start + future_offset:end + future_offset]
        for cap in CAPS_W:
            clipped = np.clip(predictions, -cap, cap)
            mae = float(np.mean(np.abs(targets - clipped)))
            complexity = {"zero": 0, "median": 1, "persistence": 2, "ar1": 3}[kind]
            smoothness = -parameter if kind == "median" else parameter
            candidates.append((mae, complexity, smoothness, cap, kind, parameter))
    candidates.sort()
    mae, _, _, cap, kind, parameter = candidates[0]
    return kind, float(parameter), float(cap), mae


def eval_day(frame: pd.DataFrame, horizon_s: int) -> list[dict]:
    g = frame.sort_values("timestamp_utc").reset_index(drop=True)
    g["timestamp_utc"] = pd.to_datetime(g.timestamp_utc, utc=True, errors="coerce")
    measured = g.measured_w.to_numpy(float)
    physical = g.predicted_w.to_numpy(float)
    residual = measured - physical
    timestamps = g.timestamp_utc.to_numpy()
    future_offset = horizon_s // 5 - 1
    history: list[float] = []
    chosen = ("zero", 0.0, 1000.0, float("nan"))
    last_tune_n = -TUNE_EVERY
    previous_time = None
    forecasts = []
    for origin in range(len(g)):
        now = pd.Timestamp(timestamps[origin])
        if previous_time is not None and (now - previous_time).total_seconds() > GAP_RESET_S:
            history = []
            chosen = ("zero", 0.0, 1000.0, float("nan"))
            last_tune_n = -TUNE_EVERY
        previous_time = now
        n = len(history)
        if n >= MIN_HISTORY and n - last_tune_n >= TUNE_EVERY:
            chosen = tune_horizon(history, future_offset)
            last_tune_n = n
        target = origin + future_offset
        contiguous = target < len(g) and (pd.Timestamp(timestamps[target]) - now).total_seconds() == future_offset * 5
        if n >= MIN_HISTORY and contiguous:
            kind, parameter, cap, inner_mae = chosen
            correction = forecast_h(np.asarray(history), kind, parameter, future_offset + 1, cap)
            forecasts.append({"day": int(g.day.iloc[origin]), "horizon_s": horizon_s,
                              "origin_index": origin, "target_index": target,
                              "origin_utc": now.isoformat(), "target_utc": pd.Timestamp(timestamps[target]).isoformat(),
                              "measured_w": float(measured[target]), "baseline_prediction_w": float(physical[target]),
                              "filter_prediction_w": float(physical[target] + correction),
                              "filter_correction_w": correction, "selected_filter": kind,
                              "selected_parameter": parameter, "selected_cap_w": cap,
                              "inner_validation_mae_w": inner_mae, "prior_samples": n})
        # The current row's residual is available only to future prediction origins.
        history.append(float(residual[origin]))
    return forecasts


def score(g: pd.DataFrame) -> dict:
    base_error = g.measured_w.to_numpy(float) - g.baseline_prediction_w.to_numpy(float)
    model_error = g.measured_w.to_numpy(float) - g.filter_prediction_w.to_numpy(float)
    return {"n_forecasts": int(len(g)), "baseline_mae_w": float(np.mean(np.abs(base_error))),
            "filter_mae_w": float(np.mean(np.abs(model_error))),
            "improvement_pct": float(100 * (np.mean(np.abs(base_error)) - np.mean(np.abs(model_error))) / np.mean(np.abs(base_error))),
            "baseline_p90_abs_error_w": float(np.quantile(np.abs(base_error), .9)),
            "filter_p90_abs_error_w": float(np.quantile(np.abs(model_error), .9)),
            "filter_bias_w": float(np.mean(model_error))}


def main() -> None:
    data = pd.read_csv(INPUT, low_memory=False)
    forecasts = pd.DataFrame([r for h in HORIZONS_S for _, g in data.groupby("day", sort=True)
                              for r in eval_day(g.copy(), h)])
    forecasts.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    summary = {}
    for h, g in forecasts.groupby("horizon_s", sort=True):
        summary[str(int(h))] = {"clean_days_1_5": score(g[g.day.le(5)]),
                                "day6_stress": score(g[g.day.eq(6)]),
                                "prior_inner_mae_w": float(g[g.day.le(5)].inner_validation_mae_w.mean()),
                                "next_target_mae_w": score(g[g.day.le(5)])["filter_mae_w"]}
    choices = {str(int(h)): {f"{kind}({p:g}),cap={cap:g}W": int(n) for (kind, p, cap), n in
                             g[g.day.le(5)][["selected_filter", "selected_parameter", "selected_cap_w"]].value_counts().items()}
               for h, g in forecasts.groupby("horizon_s", sort=True)}
    payload = {"validation": "horizon-specific nested prequential evaluation; each filter is tuned on earlier origins whose target is already observed; current/future target is never used for tuning",
               "forecast_semantics": "Future physical baseline uses logged future speed/road inputs, while learned residual corrections propagate without intervening residual measurements. This isolates model error conditional on the logged speed trajectory.",
               "horizon_results": summary, "selected_configuration_counts": choices,
               "filter_candidates": {"median_samples": [12, 60, 300], "persistence_phi": [0.25, 0.5, 0.75, 1.0],
                                      "ar1_lookback_samples": [60, 120, 300], "correction_cap_w": CAPS_W},
               "update_every_samples": TUNE_EVERY, "minimum_history_samples": MIN_HISTORY,
               "outputs": [str(OUT_CSV), str(OUT_JSON)]}
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Horizon-specific online residual models", "",
             "For each horizon, filter and cap selection uses only earlier forecast origins whose matching target power has already been observed. The chosen model is then propagated to future target rows without intervening residual measurements. Future physical baseline power uses logged future speed/route inputs, so this is conditional on the recorded trajectory.", "",
             "| Horizon | Day1–5 n | Baseline MAE (W) | Filter MAE (W) | Improvement | Baseline P90 (W) | Filter P90 (W) | Day6 MAE base→filter (W) | Day6 P90 base→filter (W) |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for h in HORIZONS_S:
        c, d6 = summary[str(h)]["clean_days_1_5"], summary[str(h)]["day6_stress"]
        lines.append(f"| {h} s | {c['n_forecasts']} | {c['baseline_mae_w']:.1f} | {c['filter_mae_w']:.1f} | {c['improvement_pct']:+.1f}% | {c['baseline_p90_abs_error_w']:.1f} | {c['filter_p90_abs_error_w']:.1f} | {d6['baseline_mae_w']:.1f}→{d6['filter_mae_w']:.1f} | {d6['baseline_p90_abs_error_w']:.1f}→{d6['filter_p90_abs_error_w']:.1f} |")
    lines += ["", "These tests report residual prediction only. Replaying measured future speeds does not verify future-weather or driver-command prediction, and block autocorrelation means the forecast count is not the number of independent experiments.", "",
              f"Outputs: `{OUT_CSV.relative_to(ROOT)}`, `{OUT_JSON.relative_to(ROOT)}`.", ""]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
