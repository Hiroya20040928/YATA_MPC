#!/usr/bin/env python3
"""Measure how the causal 5-second residual state decays across forecast horizons."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/full_race_residual_prequential_5s.csv"
OUT_CSV = ROOT / "reports/residual_prequential_horizon_metrics.csv"
OUT_JSON = ROOT / "reports/residual_prequential_horizon_summary.json"
OUT_REPORT = ROOT / "reports/residual_prequential_horizon_report.md"
HORIZONS_S = [5, 30, 60, 300]


def forecast_h(past: np.ndarray, kind: str, parameter: float, horizon_steps: int, cap: float) -> float:
    y = np.asarray(past, dtype=float)
    if len(y) == 0:
        return 0.0
    if kind == "median":
        state = float(np.median(y[-int(parameter):]))
        return float(np.clip(state, -cap, cap))
    if kind == "ewma":
        x = y[-120:]
        ages = np.arange(len(x) - 1, -1, -1, dtype=float)
        weights = float(parameter) * (1.0 - float(parameter)) ** ages
        state = float(np.dot(weights / weights.sum(), x))
        return float(np.clip(state, -cap, cap))
    if kind == "persistence":
        state = float(y[-1])
        for _ in range(horizon_steps):
            state = float(np.clip(float(parameter) * state, -cap, cap))
        return state
    lookback = min(int(parameter), len(y) - 1)
    if lookback < 20:
        return float(np.clip(np.median(y[-min(6, len(y)):]), -cap, cap))
    target = y[-lookback:]
    lag = y[-lookback-1:-1]
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(lag)), lag]), target, rcond=None)[0]
    beta[1] = np.clip(beta[1], -0.25, 0.98)
    state = float(y[-1])
    for _ in range(horizon_steps):
        state = float(np.clip(beta[0] + beta[1] * state, -cap, cap))
    return state


def metrics(frame: pd.DataFrame) -> dict:
    e0 = frame.measured_future_w.to_numpy(float) - frame.baseline_future_w.to_numpy(float)
    e1 = frame.measured_future_w.to_numpy(float) - frame.filter_future_w.to_numpy(float)
    return {"n_forecasts": int(len(e0)),
            "baseline_mae_w": float(np.mean(np.abs(e0))),
            "filter_mae_w": float(np.mean(np.abs(e1))),
            "improvement_pct": float(100 * (np.mean(np.abs(e0)) - np.mean(np.abs(e1))) / np.mean(np.abs(e0))),
            "baseline_p90_abs_error_w": float(np.quantile(np.abs(e0), .9)),
            "filter_p90_abs_error_w": float(np.quantile(np.abs(e1), .9)),
            "filter_bias_w": float(np.mean(e1))}


def main() -> None:
    data = pd.read_csv(INPUT, low_memory=False)
    data["timestamp_utc"] = pd.to_datetime(data.timestamp_utc, utc=True, errors="coerce")
    forecast_rows = []
    for day, g in data.groupby("day", sort=True):
        g = g.sort_values("timestamp_utc").reset_index(drop=True)
        y = (g.measured_w - g.predicted_w).to_numpy(float)
        times = g.timestamp_utc.to_numpy()
        for horizon_s in HORIZONS_S:
            offset = horizon_s // 5 - 1
            if offset < 0:
                raise ValueError("Horizons must be at least one sample interval")
            for i in range(1, len(g) - offset):
                if int(g.prior_samples.iloc[i]) < 120:
                    continue
                j = i + offset
                if pd.Timedelta(times[j] - times[i]).total_seconds() != offset * 5:
                    continue
                kind = str(g.selected_filter.iloc[i])
                parameter = float(g.selected_filter_parameter.iloc[i])
                cap = float(g.selected_correction_cap_w.iloc[i])
                # At origin i, the last observed residual is i-1. Propagate only this state.
                correction = forecast_h(y[:i], kind, parameter, offset + 1, cap)
                forecast_rows.append({"day": int(day), "origin_utc": times[i], "target_utc": times[j],
                                      "horizon_s": horizon_s, "measured_future_w": float(g.measured_w.iloc[j]),
                                      "baseline_future_w": float(g.predicted_w.iloc[j]),
                                      "filter_future_w": float(g.predicted_w.iloc[j]) + correction,
                                      "residual_forecast_w": correction, "selected_filter": kind})
    forecasts = pd.DataFrame(forecast_rows)
    forecasts.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    summary = {}
    for horizon, h in forecasts.groupby("horizon_s", sort=True):
        summary[str(int(horizon))] = {"clean_days_1_5": metrics(h[h.day.le(5)]),
                                      "day6_stress": metrics(h[h.day.eq(6)])}
    payload = {"horizon_metrics": summary,
               "forecast_semantics": "At each origin the learned residual state uses completed telemetry only; future physical baseline power uses the logged future speed/road inputs, isolating residual prediction error. No residual measurements between origin and target are used.",
               "outputs": [str(OUT_CSV), str(OUT_JSON)]}
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Prequential residual horizon decay", "",
             "The residual state at each forecast origin uses only already completed samples. It is propagated without any intervening telemetry update. The physical baseline at the target uses the logged speed and route inputs, so this measures residual-correction decay conditional on the observed driving profile.", "",
             "| Horizon | Day1–5 forecasts | Baseline MAE (W) | Filter MAE (W) | Improvement | Baseline P90 (W) | Filter P90 (W) | Day6 MAE baseline→filter (W) | Day6 P90 baseline→filter (W) |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for horizon, result in summary.items():
        c, d6 = result["clean_days_1_5"], result["day6_stress"]
        lines.append(f"| {horizon} s | {c['n_forecasts']} | {c['baseline_mae_w']:.1f} | {c['filter_mae_w']:.1f} | {c['improvement_pct']:+.1f}% | {c['baseline_p90_abs_error_w']:.1f} | {c['filter_p90_abs_error_w']:.1f} | {d6['baseline_mae_w']:.1f}→{d6['filter_mae_w']:.1f} | {d6['baseline_p90_abs_error_w']:.1f}→{d6['filter_p90_abs_error_w']:.1f} |")
    lines += ["", "The 5-second result is not a multi-minute accuracy guarantee. This horizon test holds the residual estimate fixed between measurements and therefore shows how quickly the usable online correction decays.", "",
              f"Outputs: `{OUT_CSV.relative_to(ROOT)}`, `{OUT_JSON.relative_to(ROOT)}`.", ""]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
