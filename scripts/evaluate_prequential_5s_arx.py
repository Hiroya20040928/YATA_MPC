#!/usr/bin/env python3
"""Past-only ARX residual predictor combining power persistence and inertia."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/full_race_residual_corrected_5s.csv"
AR_FILTER = ROOT / "data/processed/full_race_residual_prequential_5s.csv"
OUT_ROWS = ROOT / "data/processed/full_race_residual_arx_5s.csv"
OUT_DAY = ROOT / "reports/residual_arx_5s_by_day.csv"
OUT_JSON = ROOT / "reports/residual_arx_5s_summary.json"
OUT_REPORT = ROOT / "reports/residual_arx_5s_report.md"
OUT_MODEL = ROOT / "models/vehicle_residual_arx_5s_candidate.json"
MASS_PROXY_KG = 314.0
MIN_HISTORY = 120
INNER_HOLDOUT = 60
TUNE_EVERY = 60
GAP_RESET_S = 600.0
ALPHAS = [0.2, 0.35, 0.5]
WINDOWS = [60, 120, 300]
CAPS = [100.0, 300.0, 500.0, 1000.0]


def causal_inertia(frame: pd.DataFrame, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    speed = pd.to_numeric(frame["Speed[kmh]"], errors="coerce").to_numpy(float) / 3.6
    timestamps = pd.to_datetime(frame.timestamp_utc, utc=True, errors="coerce")
    delta_s = timestamps.diff().dt.total_seconds().to_numpy(float)
    smooth = np.full(len(frame), np.nan)
    acceleration = np.zeros(len(frame))
    for i in range(len(frame)):
        if not np.isfinite(speed[i]):
            continue
        if i == 0 or not np.isfinite(delta_s[i]) or delta_s[i] > GAP_RESET_S or not np.isfinite(smooth[i - 1]):
            smooth[i] = speed[i]
        else:
            smooth[i] = alpha * speed[i] + (1.0 - alpha) * smooth[i - 1]
            dt = max(1.0, delta_s[i])
            acceleration[i] = (smooth[i] - smooth[i - 1]) / dt
    power = MASS_PROXY_KG * smooth * np.clip(acceleration, -0.5, 0.5)
    return np.maximum(power, 0.0), np.minimum(power, 0.0)


def fit_huber_arx(history: list[dict], xp_all: np.ndarray, xn_all: np.ndarray,
                  target_index: int, window: int) -> float:
    h = history[-window:]
    if len(h) < 30:
        return 0.0
    y = np.asarray([r["error_w"] for r in h], float)
    lag = np.asarray([r["previous_error_w"] for r in h], float)
    indices = np.asarray([r["index"] for r in h], int)
    X = np.column_stack([np.ones(len(h)), lag, xp_all[indices], xn_all[indices]])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    penalty = np.diag([1e-8, 0.02, 0.03, 0.03])
    for _ in range(4):
        r = y - X @ beta
        center = float(np.median(r))
        scale = max(30.0, 1.4826 * float(np.median(np.abs(r - center))))
        weights = np.minimum(1.0, 100.0 / np.maximum(np.abs(r - center), 1e-8))
        beta = np.linalg.solve(X.T @ (weights[:, None] * X) + penalty, X.T @ (weights * y))
        beta[1] = np.clip(beta[1], -0.25, 0.98)
        beta[2:] = np.clip(beta[2:], 0.0, 2.0)
    current = np.asarray([1.0, float(history[-1]["error_w"]), xp_all[target_index], xn_all[target_index]])
    return float(current @ beta)


def tune(history: list[dict], feature_cache: dict[float, tuple[np.ndarray, np.ndarray]]) -> tuple[float, int, float, float]:
    start = max(30, len(history) - INNER_HOLDOUT)
    candidates = []
    for alpha in ALPHAS:
        xp, xn = feature_cache[alpha]
        for window in WINDOWS:
            pred, target = [], []
            for j in range(start, len(history)):
                target_idx = int(history[j]["index"])
                pred.append(fit_huber_arx(history[:j], xp, xn, target_idx, window))
                target.append(float(history[j]["error_w"]))
            for cap in CAPS:
                mae = float(np.mean(np.abs(np.asarray(target) - np.clip(pred, -cap, cap))))
                candidates.append((mae, alpha, window, cap))
    candidates.sort()
    mae, alpha, window, cap = candidates[0]
    return alpha, int(window), float(cap), mae


def evaluate_day(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.sort_values("timestamp_utc").copy().reset_index(drop=True)
    g["timestamp_utc"] = pd.to_datetime(g.timestamp_utc, utc=True, errors="coerce")
    feature_cache = {a: causal_inertia(g, a) for a in ALPHAS}
    measured, baseline = g.measured_w.to_numpy(float), g.predicted_w.to_numpy(float)
    history: list[dict] = []
    alpha, window, cap, inner = 0.35, 120, 300.0, float("nan")
    last_tune_n = -TUNE_EVERY
    previous_time = None
    predictions, corrections, selected_alpha, selected_window, selected_cap, inner_scores, prior_counts = [], [], [], [], [], [], []
    for i, row in enumerate(g.itertuples(index=False)):
        now = row.timestamp_utc
        if previous_time is not None and (now - previous_time).total_seconds() > GAP_RESET_S:
            history = []
            alpha, window, cap, inner = 0.35, 120, 300.0, float("nan")
            last_tune_n = -TUNE_EVERY
        previous_time = now
        n = len(history)
        prior_counts.append(n)
        if n >= MIN_HISTORY and n - last_tune_n >= TUNE_EVERY:
            alpha, window, cap, inner = tune(history, feature_cache)
            last_tune_n = n
        xp, xn = feature_cache[alpha]
        correction = fit_huber_arx(history, xp, xn, i, window) if n >= MIN_HISTORY else 0.0
        correction = float(np.clip(correction, -cap, cap))
        corrections.append(correction)
        predictions.append(baseline[i] + correction)
        selected_alpha.append(alpha)
        selected_window.append(window)
        selected_cap.append(cap)
        inner_scores.append(inner)
        previous_error = history[-1]["error_w"] if history else 0.0
        history.append({"index": i, "error_w": float(measured[i] - baseline[i]),
                        "previous_error_w": float(previous_error)})
    g["arx_prediction_w"] = predictions
    g["arx_correction_w"] = corrections
    g["selected_accel_alpha"] = selected_alpha
    g["selected_window_samples"] = selected_window
    g["selected_cap_w"] = selected_cap
    g["arx_inner_mae_w"] = inner_scores
    g["arx_prior_samples"] = prior_counts
    return g


def metric(g: pd.DataFrame, pred: str) -> dict:
    e = g.measured_w.to_numpy(float) - g[pred].to_numpy(float)
    return {"n": int(len(e)), "mae_w": float(np.mean(np.abs(e))),
            "p90_abs_error_w": float(np.quantile(np.abs(e), .9)), "bias_w": float(np.mean(e))}


def main() -> None:
    source = pd.read_csv(INPUT, low_memory=False)
    result = pd.concat([evaluate_day(g.copy()) for _, g in source.groupby("day", sort=True)], ignore_index=True)
    ar_filter = pd.read_csv(AR_FILTER, low_memory=False, usecols=["day", "timestamp_utc", "residual_prediction_w"])
    result["timestamp_utc"] = pd.to_datetime(result.timestamp_utc, utc=True, errors="coerce")
    ar_filter["timestamp_utc"] = pd.to_datetime(ar_filter.timestamp_utc, utc=True, errors="coerce")
    result = result.merge(ar_filter, on=["day", "timestamp_utc"], how="left", validate="one_to_one")
    result.to_csv(OUT_ROWS, index=False, encoding="utf-8-sig")
    clean, day6 = result[result.day.le(5)], result[result.day.eq(6)]
    base, ar, arx, base6, arx6 = [metric(g, col) for g, col in [
        (clean, "predicted_w"), (clean, "residual_prediction_w"), (clean, "arx_prediction_w"),
        (day6, "predicted_w"), (day6, "arx_prediction_w")]]
    day_rows = []
    for day, g in result.groupby("day", sort=True):
        for name, col in [("physics_baseline", "predicted_w"), ("past_only_5s_filter", "residual_prediction_w"),
                          ("causal_arx", "arx_prediction_w")]:
            day_rows.append({"day": int(day), "model": name, **metric(g, col)})
    pd.DataFrame(day_rows).to_csv(OUT_DAY, index=False, encoding="utf-8-sig")
    wins = int(sum(pd.DataFrame(day_rows).query("day <= 5 and model == 'causal_arx'").set_index("day").mae_w <
                   pd.DataFrame(day_rows).query("day <= 5 and model == 'physics_baseline'").set_index("day").mae_w))
    warm_wins = 0
    for day in sorted(clean.day.unique()):
        g = result[(result.day.eq(day)) & result.arx_prior_samples.ge(MIN_HISTORY)]
        if len(g) and metric(g, "arx_prediction_w")["mae_w"] < metric(g, "predicted_w")["mae_w"]:
            warm_wins += 1
    improvement = 100 * (base["mae_w"] - arx["mae_w"]) / base["mae_w"]
    tail = arx["p90_abs_error_w"] <= base["p90_abs_error_w"] and arx6["p90_abs_error_w"] <= base6["p90_abs_error_w"]
    active = clean[(clean.arx_prior_samples.ge(MIN_HISTORY)) & clean.arx_inner_mae_w.notna()]
    gap = {"active_samples": int(len(active)), "inner_mae_w": float(active.arx_inner_mae_w.mean()),
           "next_sample_mae_w": float(np.mean(np.abs(active.measured_w - active.arx_prediction_w)))}
    choices = {f"accel_alpha={a:g},window={w},cap={c:g}W": int(n) for (a, w, c), n in
               result[(result.day.le(5)) & result.arx_prior_samples.ge(MIN_HISTORY)][
                   ["selected_accel_alpha", "selected_window_samples", "selected_cap_w"]].value_counts().items()}
    adopted = improvement >= 10 and wins >= 4 and warm_wins >= 4 and tail
    summary = {"validation": "nested past-only 5-second one-step predictions; candidate combines lagged residual error with causal signed inertial power",
               "mass_basis_kg": MASS_PROXY_KG,
               "clean_days_1_5": {"physics_baseline": base, "past_only_residual_filter": ar,
                                  "causal_arx": arx, "improvement_pct": improvement,
                                  "day_wins": wins, "warm_state_day_wins": warm_wins},
               "day6_stress": {"baseline": base6, "causal_arx": arx6},
               "inner_to_next_sample": gap, "selected_configuration_counts": choices,
               "adopted_to_shadow": bool(adopted), "production_ready": False,
               "warning": "314 kg is a design-mass proxy, not confirmed loaded mass. Coefficients are predictive and do not identify actual inertia or efficiency.",
               "outputs": [str(OUT_ROWS), str(OUT_DAY), str(OUT_JSON)]}
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_REPORT.write_text("\n".join([
        "# Causal ARX residual candidate for 5-second forecasts", "",
        "The prediction combines the previous measured power residual with current causal signed inertial power. It fits a robust regularized linear model on prior rows only. Speed smoothing, lookback and correction cap are selected every 60 completed samples using a 60-sample inner one-step holdout.", "",
        f"Day1–5 MAE: physics {base['mae_w']:.1f} W; residual-only AR filter {ar['mae_w']:.1f} W; causal ARX {arx['mae_w']:.1f} W. ARX P90 {base['p90_abs_error_w']:.1f}→{arx['p90_abs_error_w']:.1f} W; day wins {wins}/5, warm-state wins {warm_wins}/5.",
        f"Day6 stress MAE {base6['mae_w']:.1f}→{arx6['mae_w']:.1f} W; P90 {base6['p90_abs_error_w']:.1f}→{arx6['p90_abs_error_w']:.1f} W.",
        f"Inner MAE {gap['inner_mae_w']:.1f} W, next-sample MAE {gap['next_sample_mae_w']:.1f} W over {gap['active_samples']} updates.",
        f"Selected configurations: `{choices}`.", "",
        "This is one-step prediction only. The 314 kg feature scaling is a design-mass proxy, not a measurement of driver-loaded mass.", "",
        f"Outputs: `{OUT_ROWS.relative_to(ROOT)}`, `{OUT_DAY.relative_to(ROOT)}`, `{OUT_JSON.relative_to(ROOT)}`.", ""
    ]), encoding="utf-8")
    if adopted:
        OUT_MODEL.write_text(json.dumps({"model_type": "causal_online_arx_residual_candidate",
                                         "stage": "shadow_candidate_not_connected_to_replay_grafana_or_commands",
                                         "production_ready": False, "metrics": summary["clean_days_1_5"],
                                         "day6_stress": summary["day6_stress"], "validation": summary["validation"],
                                         "selected_configuration_counts": choices, "warning": summary["warning"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"clean_days_1_5": summary["clean_days_1_5"], "day6_stress": summary["day6_stress"],
                      "inner_to_next_sample": gap, "selected_configuration_counts": choices,
                      "adopted_to_shadow": adopted, "outputs": summary["outputs"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
