#!/usr/bin/env python3
"""Test a causal acceleration-power residual term with nested online selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/full_race_residual_corrected_5s.csv"
OUT_ROWS = ROOT / "data/processed/full_race_residual_inertial_5s.csv"
OUT_DAY = ROOT / "reports/residual_inertial_5s_by_day.csv"
OUT_SUMMARY = ROOT / "reports/residual_inertial_5s_summary.json"
OUT_REPORT = ROOT / "reports/residual_inertial_5s_report.md"
OUT_MODEL = ROOT / "models/vehicle_residual_inertial_5s_candidate.json"
MASS_PROXY_KG = 314.0
MIN_HISTORY = 120
INNER_HOLDOUT = 60
TUNE_EVERY = 60
GAP_RESET_S = 600.0
ALPHAS = [0.2, 0.35, 0.5]
WINDOWS = [120, 300]
CAPS_W = [100.0, 300.0, 500.0, 1000.0]


def causal_inertial_features(frame: pd.DataFrame, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    g = frame.sort_values("timestamp_utc").copy()
    speed = pd.to_numeric(g["Speed[kmh]"], errors="coerce").to_numpy(float) / 3.6
    timestamps = pd.to_datetime(g.timestamp_utc, utc=True, errors="coerce")
    delta_s = timestamps.diff().dt.total_seconds().to_numpy(float)
    smoothed = np.full(len(g), np.nan)
    accel = np.zeros(len(g))
    last = None
    for i in range(len(g)):
        if not np.isfinite(speed[i]):
            continue
        if last is None or not np.isfinite(delta_s[i]) or delta_s[i] > GAP_RESET_S:
            smoothed[i] = speed[i]
            accel[i] = 0.0
        else:
            smoothed[i] = alpha * speed[i] + (1.0 - alpha) * smoothed[i - 1]
            dt = max(delta_s[i], 1.0)
            accel[i] = (smoothed[i] - smoothed[i - 1]) / dt
        last = i
    accel = np.clip(accel, -0.5, 0.5)
    inertia = MASS_PROXY_KG * smoothed * accel
    return np.maximum(inertia, 0.0), np.minimum(inertia, 0.0)


def fit_huber(y: np.ndarray, x_pos: np.ndarray, x_neg: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(y)), x_pos, x_neg])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    penalty = np.diag([1e-8, 0.03, 0.03])
    for _ in range(4):
        residual = y - X @ beta
        center = float(np.median(residual))
        sigma = max(30.0, 1.4826 * float(np.median(np.abs(residual - center))))
        weight = np.minimum(1.0, 100.0 / np.maximum(np.abs(residual - center), 1e-8))
        lhs = X.T @ (weight[:, None] * X) + penalty
        rhs = X.T @ (weight * y)
        beta = np.linalg.solve(lhs, rhs)
        beta[1:] = np.clip(beta[1:], 0.0, 2.0)
    return beta


def predict_at(history: list[dict], x_pos_all: np.ndarray, x_neg_all: np.ndarray,
               target_index: int, window: int) -> float:
    h = history[-window:]
    if len(h) < 20:
        return 0.0
    y = np.asarray([r["error_w"] for r in h], float)
    indices = np.asarray([r["index"] for r in h], dtype=int)
    xp = x_pos_all[indices]
    xn = x_neg_all[indices]
    beta = fit_huber(y, xp, xn)
    return float(beta[0] + beta[1] * x_pos_all[target_index] + beta[2] * x_neg_all[target_index])


def tune(history: list[dict], feature_cache: dict[float, tuple[np.ndarray, np.ndarray]]) -> tuple[float, int, float, float]:
    start = max(30, len(history) - INNER_HOLDOUT)
    candidates = []
    for alpha in ALPHAS:
        xp_all, xn_all = feature_cache[alpha]
        for window in WINDOWS:
            predictions, targets = [], []
            for j in range(start, len(history)):
                target_index = int(history[j]["index"])
                correction = predict_at(history[:j], xp_all, xn_all, target_index, window)
                predictions.append(correction)
                targets.append(float(history[j]["error_w"]))
            for cap in CAPS_W:
                if predictions:
                    mae = float(np.mean(np.abs(np.asarray(targets) - np.clip(predictions, -cap, cap))))
                    candidates.append((mae, alpha, window, cap))
    if not candidates:
        return 0.35, 120, 300.0, float("nan")
    candidates.sort()
    mae, alpha, window, cap = candidates[0]
    return alpha, int(window), float(cap), mae


def evaluate_day(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.sort_values("timestamp_utc").copy().reset_index(drop=True)
    g["timestamp_utc"] = pd.to_datetime(g.timestamp_utc, utc=True, errors="coerce")
    feature_cache = {a: causal_inertial_features(g, a) for a in ALPHAS}
    measured = g.measured_w.to_numpy(float)
    base = g.predicted_w.to_numpy(float)
    history: list[dict] = []
    alpha, window, cap, inner = 0.35, 120, 300.0, float("nan")
    last_tune_n = -TUNE_EVERY
    previous_time = None
    pred_values, corrections, selected_alpha, selected_window, selected_cap, inner_values, prior_counts = [], [], [], [], [], [], []
    for i, row in enumerate(g.itertuples(index=False)):
        now = row.timestamp_utc
        if previous_time is not None and (now - previous_time).total_seconds() > GAP_RESET_S:
            history = []
            last_tune_n = -TUNE_EVERY
            alpha, window, cap, inner = 0.35, 120, 300.0, float("nan")
        previous_time = now
        n = len(history)
        prior_counts.append(n)
        if n >= MIN_HISTORY and n - last_tune_n >= TUNE_EVERY:
            alpha, window, cap, inner = tune(history, feature_cache)
            last_tune_n = n
        xp, xn = feature_cache[alpha]
        correction = predict_at(history, xp, xn, i, window) if n >= MIN_HISTORY else 0.0
        correction = float(np.clip(correction, -cap, cap))
        corrections.append(correction)
        pred_values.append(base[i] + correction)
        selected_alpha.append(alpha)
        selected_window.append(window)
        selected_cap.append(cap)
        inner_values.append(inner)
        # Record only data known after the current prediction.
        history.append({"error_w": float(measured[i] - base[i]), "index": i})
    g["inertial_prediction_w"] = pred_values
    g["inertial_correction_w"] = corrections
    g["selected_accel_smoothing_alpha"] = selected_alpha
    g["selected_inertial_window"] = selected_window
    g["selected_inertial_cap_w"] = selected_cap
    g["inertial_inner_mae_w"] = inner_values
    g["inertial_prior_samples"] = prior_counts
    return g


def metric(g: pd.DataFrame, pred: str) -> dict:
    e = g.measured_w.to_numpy(float) - g[pred].to_numpy(float)
    return {"n": int(len(e)), "mae_w": float(np.mean(np.abs(e))),
            "p90_abs_error_w": float(np.quantile(np.abs(e), .9)), "bias_w": float(np.mean(e))}


def main() -> None:
    data = pd.read_csv(INPUT, low_memory=False)
    outputs = [evaluate_day(g.copy()) for _, g in data.groupby("day", sort=True)]
    result = pd.concat(outputs, ignore_index=True)
    result.to_csv(OUT_ROWS, index=False, encoding="utf-8-sig")
    rows = []
    for day, g in result.groupby("day", sort=True):
        for name, col in [("physics_baseline", "predicted_w"), ("causal_inertial_residual", "inertial_prediction_w")]:
            rows.append({"day": int(day), "model": name, **metric(g, col),
                         "baseline_mae_w": metric(g, "predicted_w")["mae_w"]})
    days = pd.DataFrame(rows)
    days.to_csv(OUT_DAY, index=False, encoding="utf-8-sig")
    clean = result[result.day.le(5)]
    d6 = result[result.day.eq(6)]
    base, model, base6, model6 = metric(clean, "predicted_w"), metric(clean, "inertial_prediction_w"), metric(d6, "predicted_w"), metric(d6, "inertial_prediction_w")
    perday = days[(days.day.le(5)) & days.model.eq("causal_inertial_residual")]
    baseline_days = days[(days.day.le(5)) & days.model.eq("physics_baseline")].set_index("day")
    wins = int(sum(perday.set_index("day").mae_w < baseline_days.mae_w))
    warm_wins = 0
    for day in sorted(clean.day.unique()):
        g = result[(result.day.eq(day)) & result.inertial_prior_samples.ge(MIN_HISTORY)]
        if len(g) and metric(g, "inertial_prediction_w")["mae_w"] < metric(g, "predicted_w")["mae_w"]:
            warm_wins += 1
    active = clean[(clean.inertial_prior_samples.ge(MIN_HISTORY)) & clean.inertial_inner_mae_w.notna()]
    gap = {"active_samples": int(len(active)), "mean_inner_mae_w": float(active.inertial_inner_mae_w.mean()),
           "next_sample_mae_w": float(np.mean(np.abs(active.measured_w - active.inertial_prediction_w)))}
    improvement = 100 * (base["mae_w"] - model["mae_w"]) / base["mae_w"]
    tail_pass = model["p90_abs_error_w"] <= base["p90_abs_error_w"] and model6["p90_abs_error_w"] <= base6["p90_abs_error_w"]
    adopted = improvement >= 10 and wins >= 4 and warm_wins >= 4 and tail_pass
    choices = {f"EWMA alpha={a:g},window={w},cap={c:g}W": int(n) for (a, w, c), n in
               result[(result.day.le(5)) & result.inertial_prior_samples.ge(MIN_HISTORY)][
                   ["selected_accel_smoothing_alpha", "selected_inertial_window", "selected_inertial_cap_w"]].value_counts().items()}
    summary = {"validation": "nested past-only 5-second one-step evaluation; acceleration is backward-looking, settings refit every five minutes from prior one-step errors",
               "mass_basis_kg": MASS_PROXY_KG, "acceleration_feature": "m_proxy * causal_EWMA_speed * causal_EWMA_acceleration; acceleration clipped to ±0.5 m/s²",
               "clean_days_1_5": {"baseline": base, "causal_inertial_residual": model,
                                  "improvement_pct": improvement, "day_wins": wins, "warm_state_day_wins": warm_wins},
               "day6_stress": {"baseline": base6, "causal_inertial_residual": model6},
               "inner_to_next_sample_gap": gap, "selected_configuration_counts": choices,
               "adopted_to_shadow": bool(adopted), "production_ready": False,
               "warning": "The 314 kg mass is a prior model proxy, not the confirmed actual mass with driver. The fitted residual term is predictive and does not identify true mass or drivetrain efficiency.",
               "outputs": [str(OUT_ROWS), str(OUT_DAY), str(OUT_SUMMARY)]}
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_REPORT.write_text("\n".join([
        "# Causal acceleration residual candidate", "",
        "This candidate uses a backward-looking exponentially smoothed speed derivative to form signed inertial power bases. A robust bounded linear fit is trained on completed rows only; smoothing time constant, training window and correction cap are selected every five minutes from past one-step errors. This avoids the centered smoothing's future leakage in the earlier descriptive inertial analysis.", "",
        f"Day1–5 baseline MAE {base['mae_w']:.1f} W → candidate {model['mae_w']:.1f} W ({improvement:+.1f}%), P90 {base['p90_abs_error_w']:.1f}→{model['p90_abs_error_w']:.1f} W; full-day wins {wins}/5 and warm-state wins {warm_wins}/5.",
        f"Day6 stress MAE {base6['mae_w']:.1f}→{model6['mae_w']:.1f} W; P90 {base6['p90_abs_error_w']:.1f}→{model6['p90_abs_error_w']:.1f} W.",
        f"Prior inner MAE averaged {gap['mean_inner_mae_w']:.1f} W and next one-step MAE {gap['next_sample_mae_w']:.1f} W over {gap['active_samples']} samples.",
        f"Selected settings: `{choices}`.", "",
        "The 314 kg basis is a prior proxy rather than confirmed loaded mass. Its coefficient is predictive, not an identified physical mass or motor efficiency. The 5-second one-step result does not establish multi-step MPC accuracy.", "",
        f"Outputs: `{OUT_ROWS.relative_to(ROOT)}`, `{OUT_DAY.relative_to(ROOT)}`, `{OUT_SUMMARY.relative_to(ROOT)}`.", ""
    ]), encoding="utf-8")
    if adopted:
        OUT_MODEL.write_text(json.dumps({"model_type": "causal_online_inertial_residual_candidate",
                                         "stage": "shadow_candidate_not_connected_to_replay_grafana_or_commands",
                                         "production_ready": False,
                                         "metrics": summary["clean_days_1_5"], "day6_stress": summary["day6_stress"],
                                         "selected_configuration_counts": choices,
                                         "validation": summary["validation"], "warning": summary["warning"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"clean_days_1_5": summary["clean_days_1_5"], "day6_stress": summary["day6_stress"],
                      "inner_to_next_sample_gap": gap, "selected_configuration_counts": choices,
                      "adopted_to_shadow": adopted, "outputs": summary["outputs"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
