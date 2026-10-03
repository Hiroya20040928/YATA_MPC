#!/usr/bin/env python3
"""Evaluate horizon-specific causal ARX residual rollouts on logged speed paths."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/full_race_residual_corrected_5s.csv"
OUT_CSV = ROOT / "reports/residual_multihorizon_arx_forecasts.csv"
OUT_JSON = ROOT / "reports/residual_multihorizon_arx_summary.json"
OUT_REPORT = ROOT / "reports/residual_multihorizon_arx_report.md"
HORIZONS = [5, 30, 60, 300]
MIN_HISTORY = 120
MIN_INNER_HISTORY = 30
INNER_HOLDOUT = 60
TUNE_EVERY = 60
RESET_GAP_S = 10.0
MASS_PROXY_KG = 314.0
ALPHAS = [0.2, 0.35, 0.5]
WINDOWS = [60, 120, 300]
CAPS = [100.0, 300.0, 500.0, 1000.0]


def inertia_features(g: pd.DataFrame, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    speed = pd.to_numeric(g["Speed[kmh]"], errors="coerce").to_numpy(float) / 3.6
    timestamps = pd.to_datetime(g.timestamp_utc, utc=True, errors="coerce")
    delta_s = timestamps.diff().dt.total_seconds().to_numpy(float)
    smooth = np.full(len(g), np.nan)
    accel = np.zeros(len(g))
    for i in range(len(g)):
        if not np.isfinite(speed[i]):
            continue
        if i == 0 or not np.isfinite(delta_s[i]) or delta_s[i] > RESET_GAP_S or not np.isfinite(smooth[i-1]):
            smooth[i] = speed[i]
        else:
            smooth[i] = alpha * speed[i] + (1.0-alpha) * smooth[i-1]
            accel[i] = (smooth[i] - smooth[i-1]) / max(1.0, delta_s[i])
    p = MASS_PROXY_KG * smooth * np.clip(accel, -0.5, 0.5)
    return np.maximum(p, 0.0), np.minimum(p, 0.0)


def fit_beta(history: list[dict], xp: np.ndarray, xn: np.ndarray, window: int) -> np.ndarray | None:
    h = history[-window:]
    if len(h) < MIN_INNER_HISTORY:
        return None
    y = np.asarray([r["residual_w"] for r in h], float)
    lag = np.asarray([r["previous_residual_w"] for r in h], float)
    ix = np.asarray([r["index"] for r in h], int)
    X = np.column_stack([np.ones(len(h)), lag, xp[ix], xn[ix]])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    penalty = np.diag([1e-8, 0.02, 0.03, 0.03])
    for _ in range(4):
        err = y - X @ beta
        center = float(np.median(err))
        weights = np.minimum(1.0, 100.0 / np.maximum(np.abs(err-center), 1e-8))
        beta = np.linalg.solve(X.T @ (weights[:, None] * X) + penalty, X.T @ (weights*y))
        beta[1] = np.clip(beta[1], -0.25, 0.98)
        beta[2:] = np.clip(beta[2:], 0.0, 2.0)
    return beta


def rollout(beta: np.ndarray, residual_before_origin: float, xp: np.ndarray, xn: np.ndarray,
            target_indices: list[int], cap: float) -> float:
    state = float(residual_before_origin)
    for idx in target_indices:
        state = float(np.clip(beta[0] + beta[1]*state + beta[2]*xp[idx] + beta[3]*xn[idx], -cap, cap))
    return state


def choose(history: list[dict], horizon_steps: int,
           feature_cache: dict[float, tuple[np.ndarray, np.ndarray]]) -> tuple[float, int, float, float]:
    end = len(history) - horizon_steps + 1
    start = max(MIN_INNER_HISTORY, end - INNER_HOLDOUT)
    if end <= start:
        return 0.35, 120, 1000.0, float("nan")
    candidates = []
    for alpha in ALPHAS:
        xp, xn = feature_cache[alpha]
        for window in WINDOWS:
            predictions_by_cap = {cap: [] for cap in CAPS}
            targets = []
            for origin in range(start, end):
                target_local = origin + horizon_steps - 1
                if target_local >= len(history):
                    continue
                t0 = pd.Timestamp(history[origin]["time"])
                t1 = pd.Timestamp(history[target_local]["time"])
                if (t1 - t0).total_seconds() != (horizon_steps - 1) * 5:
                    continue
                beta = fit_beta(history[:origin], xp, xn, window)
                if beta is None:
                    continue
                ix = [int(history[k]["index"]) for k in range(origin, target_local + 1)]
                prior = float(history[origin-1]["residual_w"]) if origin else 0.0
                targets.append(float(history[target_local]["residual_w"]))
                for cap in CAPS:
                    predictions_by_cap[cap].append(rollout(beta, prior, xp, xn, ix, cap))
            for cap, preds in predictions_by_cap.items():
                if preds and len(preds) == len(targets):
                    mae = float(np.mean(np.abs(np.asarray(targets) - np.asarray(preds))))
                    candidates.append((mae, alpha, window, cap))
    if not candidates:
        return 0.35, 120, 1000.0, float("nan")
    candidates.sort()
    mae, alpha, window, cap = candidates[0]
    return float(alpha), int(window), float(cap), mae


def evaluate_day(frame: pd.DataFrame, horizon_s: int) -> list[dict]:
    g = frame.sort_values("timestamp_utc").copy().reset_index(drop=True)
    g["timestamp_utc"] = pd.to_datetime(g.timestamp_utc, utc=True, errors="coerce")
    measured, physical = g.measured_w.to_numpy(float), g.predicted_w.to_numpy(float)
    residual = measured - physical
    times = g.timestamp_utc.to_numpy()
    feature_cache = {a: inertia_features(g, a) for a in ALPHAS}
    steps = horizon_s // 5
    q = steps - 1
    history: list[dict] = []
    selected = (0.35, 120, 1000.0, float("nan"))
    last_tune_n = -TUNE_EVERY
    previous_time = None
    output = []
    for origin in range(len(g)):
        now = pd.Timestamp(times[origin])
        if previous_time is not None and (now - previous_time).total_seconds() > RESET_GAP_S:
            history = []
            selected = (0.35, 120, 1000.0, float("nan"))
            last_tune_n = -TUNE_EVERY
        previous_time = now
        n = len(history)
        if n >= MIN_HISTORY and n - last_tune_n >= TUNE_EVERY:
            selected = choose(history, steps, feature_cache)
            last_tune_n = n
        target = origin + q
        if target < len(g) and n >= MIN_HISTORY and (pd.Timestamp(times[target]) - now).total_seconds() == q*5:
            alpha, window, cap, inner = selected
            xp, xn = feature_cache[alpha]
            beta = fit_beta(history, xp, xn, window)
            if beta is not None:
                target_indices = list(range(origin, target+1))
                prior = float(history[-1]["residual_w"])
                correction = rollout(beta, prior, xp, xn, target_indices, cap)
                output.append({"day": int(g.day.iloc[origin]), "horizon_s": horizon_s,
                               "origin_utc": now.isoformat(), "target_utc": pd.Timestamp(times[target]).isoformat(),
                               "measured_future_w": float(measured[target]), "baseline_future_w": float(physical[target]),
                               "arx_future_w": float(physical[target] + correction), "arx_residual_forecast_w": correction,
                               "selected_alpha": alpha, "selected_window": window, "selected_cap_w": cap,
                               "inner_validation_mae_w": inner, "prior_samples": n})
        previous_residual = float(history[-1]["residual_w"]) if history else 0.0
        history.append({"index": origin, "time": pd.Timestamp(times[origin]).isoformat(),
                        "residual_w": float(residual[origin]), "previous_residual_w": previous_residual})
    return output


def metrics(g: pd.DataFrame) -> dict:
    e0 = g.measured_future_w.to_numpy(float) - g.baseline_future_w.to_numpy(float)
    e1 = g.measured_future_w.to_numpy(float) - g.arx_future_w.to_numpy(float)
    return {"n_forecasts": int(len(g)), "baseline_mae_w": float(np.mean(np.abs(e0))),
            "arx_mae_w": float(np.mean(np.abs(e1))),
            "improvement_pct": float(100 * (np.mean(np.abs(e0))-np.mean(np.abs(e1))) / np.mean(np.abs(e0))),
            "baseline_p90_abs_error_w": float(np.quantile(np.abs(e0), .9)),
            "arx_p90_abs_error_w": float(np.quantile(np.abs(e1), .9)),
            "arx_bias_w": float(np.mean(e1))}


def main() -> None:
    data = pd.read_csv(INPUT, low_memory=False)
    forecasts = pd.DataFrame([r for h in HORIZONS for _, g in data.groupby("day", sort=True)
                              for r in evaluate_day(g.copy(), h)])
    forecasts.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    summary = {}
    for horizon, g in forecasts.groupby("horizon_s", sort=True):
        summary[str(int(horizon))] = {"clean_days_1_5": metrics(g[g.day.le(5)]),
                                      "day6_stress": metrics(g[g.day.eq(6)]),
                                      "mean_inner_validation_mae_w": float(g[g.day.le(5)].inner_validation_mae_w.mean())}
    payload = {"validation": "horizon-specific nested causal ARX; prior residual and signed acceleration terms are fitted on past completed rows; the current residual is never used before prediction",
               "forecast_semantics": "The future inertial input is calculated from the logged speed path. In MPC the corresponding input would come from the candidate speed trajectory; no residual measurement between origin and target is used.",
               "horizon_results": summary, "update_every_samples": TUNE_EVERY,
               "minimum_history_samples": MIN_HISTORY, "reset_gap_seconds": RESET_GAP_S,
               "mass_proxy_kg": MASS_PROXY_KG, "selected_grids": {"ewma_speed_alpha": ALPHAS, "window_samples": WINDOWS, "residual_cap_w": CAPS},
               "outputs": [str(OUT_CSV), str(OUT_JSON)]}
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Horizon-specific causal ARX residual evaluation", "",
             "At each origin, the predictor is trained only on earlier completed 5-second rows. It propagates a residual state with a lagged residual and a signed inertial input derived from the logged speed trajectory. Horizon-specific parameters are selected from past origins whose target residual is already observed. The logged future speed is a conditional input, not a speed forecast.", "",
             "| Horizon | Day1–5 n | Baseline MAE (W) | ARX MAE (W) | Improvement | Baseline P90 (W) | ARX P90 (W) | Day6 MAE baseline→ARX (W) | Day6 P90 baseline→ARX (W) |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for h in HORIZONS:
        c, d6 = summary[str(h)]["clean_days_1_5"], summary[str(h)]["day6_stress"]
        lines.append(f"| {h} s | {c['n_forecasts']} | {c['baseline_mae_w']:.1f} | {c['arx_mae_w']:.1f} | {c['improvement_pct']:+.1f}% | {c['baseline_p90_abs_error_w']:.1f} | {c['arx_p90_abs_error_w']:.1f} | {d6['baseline_mae_w']:.1f}→{d6['arx_mae_w']:.1f} | {d6['baseline_p90_abs_error_w']:.1f}→{d6['arx_p90_abs_error_w']:.1f} |")
    lines += ["", "Future speed inputs come from the recorded telemetry and are treated as known. That makes this a conditional residual-power test; it does not include error in the future speed plan, weather forecast, or route condition. The inertial basis uses a 314 kg design proxy and does not identify loaded mass or motor efficiency.", "",
              f"Outputs: `{OUT_CSV.relative_to(ROOT)}`, `{OUT_JSON.relative_to(ROOT)}`.", ""]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
