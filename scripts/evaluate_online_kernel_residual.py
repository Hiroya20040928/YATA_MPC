#!/usr/bin/env python3
"""Prequential, nested tuning of a nonlinear same-day residual learner.

At each 5-minute block, hyperparameters are selected only from one-step
walk-forward predictions on completed earlier blocks. The final prediction is
then made for the current block. Days 1-5 are summarized separately; Day 6 is
anomaly stress data only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports/residual_block_oof.csv"
OUT_BLOCKS = ROOT / "reports/residual_online_kernel_oof.csv"
OUT_DAYS = ROOT / "reports/residual_online_kernel_by_day.csv"
OUT_SUMMARY = ROOT / "reports/residual_online_kernel_summary.json"
OUT_REPORT = ROOT / "reports/residual_online_kernel_report.md"
OUT_MODEL = ROOT / "models/vehicle_residual_online_kernel_candidate.json"
ROWS_IN = ROOT / "data/processed/full_race_residual_corrected_5s.csv"
ROWS_OUT = ROOT / "data/processed/full_race_residual_online_kernel_5s.csv"
FEATURES = ["speed_kmh", "relative_air_speed_mps", "grade_pct", "temperature_c", "air_density_kgm3"]
LENGTHS = [0.5, 1.0, 2.0]
ALPHAS = [0.1, 1.0, 10.0, 100.0]
TRAIN_WINDOWS = [12, 24, 36, 48]
MAX_WINDOW = max(TRAIN_WINDOWS)
INNER_HOLDOUT = 12
MIN_INNER_TRAIN = 8
MIN_HISTORY = 14
BIAS_WINDOWS = [3, 6, 9]
CORRECTION_CAPS_W = [30.0, 50.0, 75.0, 100.0, 150.0]
HUBER_DELTAS = [("none", None), ("30W", 30.0), ("60W", 60.0), ("100W", 100.0)]
KALMAN_Q_RATIOS = [0.001, 0.01, 0.05, 0.1, 0.5]
DEFAULT_BIAS_WINDOW = 6
DEFAULT_CORRECTION_CAP_W = 100.0
KRR_CAP_W = 100.0
GAP_RESET_S = 600.0


def fit_krr(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
            length_scale: float, alpha: float, huber_delta_w: float | None = None) -> np.ndarray:
    """RBF kernel ridge with fold-local scaling and response centering."""
    mu = np.nanmean(train_x, axis=0)
    scale = np.nanstd(train_x, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    z = (train_x - mu) / scale
    zt = (test_x - mu) / scale
    ymu = float(np.mean(train_y))
    yscale = max(float(np.std(train_y)), 20.0)
    y = (train_y - ymu) / yscale
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2)
    kernel = np.exp(-0.5 * d2 / length_scale**2)
    kernel.flat[::len(kernel) + 1] += alpha
    cross_d2 = ((zt[:, None, :] - z[None, :, :]) ** 2).sum(axis=2)
    cross = np.exp(-0.5 * cross_d2 / length_scale**2)
    weights = np.linalg.solve(kernel + alpha * np.eye(len(kernel)), y)
    if huber_delta_w is not None:
        delta = max(float(huber_delta_w) / yscale, 1e-6)
        for _ in range(4):
            residual = y - kernel @ weights
            robust_weight = np.minimum(1.0, delta / np.maximum(np.abs(residual), 1e-9))
            weights = np.linalg.solve(robust_weight[:, None] * kernel + alpha * np.eye(len(kernel)),
                                      robust_weight * y)
    return ymu + yscale * (cross @ weights)


def tune_from_past(hist: pd.DataFrame) -> tuple[float, float, int, str, float, int]:
    """Tune RBF length, regularization, lookback and robust loss using prior one-step errors."""
    x = hist[FEATURES].to_numpy(float)
    y = hist.baseline_error_w.to_numpy(float)
    start = max(MIN_INNER_TRAIN, len(hist) - INNER_HOLDOUT)
    validation = list(range(start, len(hist)))
    if len(validation) < 6:
        return 1.0, 100.0, MAX_WINDOW, "none", float("nan"), 0
    scores = []
    for ell in LENGTHS:
        for alpha in ALPHAS:
            for window in TRAIN_WINDOWS:
                for delta_label, delta_w in HUBER_DELTAS:
                    errors = []
                    for j in validation:
                        train_start = max(0, j - window)
                        if j - train_start < MIN_INNER_TRAIN:
                            continue
                        pred = fit_krr(x[train_start:j], y[train_start:j], x[j:j + 1], ell, alpha, delta_w)[0]
                        errors.append(abs(y[j] - pred))
                    if errors:
                        # Tie-break toward plain least squares, stronger regularization,
                        # smoother features, and a longer history.
                        complexity = 0 if delta_label == "none" else 1
                        scores.append((float(np.mean(errors)), complexity, -alpha, -ell,
                                       -window, ell, alpha, window, delta_label))
    if not scores:
        return 1.0, 100.0, MAX_WINDOW, "none", float("nan"), 0
    scores.sort()
    mae, _, _, _, _, ell, alpha, window, delta_label = scores[0]
    return ell, alpha, window, delta_label, mae, len(validation)


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 20.0
    center = float(np.median(values))
    return max(20.0, 1.4826 * float(np.median(np.abs(values - center))))


def kalman_bias_forecast(history_innovations: np.ndarray, q_over_r: float) -> float:
    """Robust local-level Kalman one-step forecast from completed residual innovations."""
    y = np.asarray(history_innovations, dtype=float)
    if len(y) == 0:
        return 0.0
    sigma = robust_sigma(y[-24:])
    r = sigma**2
    q = q_over_r * r
    state, variance = 0.0, r
    for obs in y:
        pred_var = variance + q
        gain = pred_var / (pred_var + r)
        innovation = float(np.clip(obs - state, -3.0 * sigma, 3.0 * sigma))
        state = state + gain * innovation
        variance = (1.0 - gain) * pred_var
    return float(state)


def tune_kalman_from_past(hist: pd.DataFrame) -> tuple[float, float, float]:
    """Choose process/measurement ratio and cap from prequential errors in past history."""
    start = max(MIN_HISTORY, len(hist) - INNER_HOLDOUT)
    candidates = []
    for q_ratio in KALMAN_Q_RATIOS:
        for cap in CORRECTION_CAPS_W:
            errors = []
            for j in range(start, len(hist)):
                prior = hist.innovation_w.iloc[:j].to_numpy(float)
                if len(prior) < MIN_HISTORY:
                    continue
                offset = kalman_bias_forecast(prior, q_ratio)
                correction = float(np.clip(hist.krr_prediction_w.iloc[j] + offset, -cap, cap))
                errors.append(abs(hist.baseline_error_w.iloc[j] - correction))
            if errors:
                candidates.append((float(np.mean(errors)), q_ratio, cap))
    if not candidates:
        return 0.01, DEFAULT_CORRECTION_CAP_W, float("nan")
    # On ties, favor slower adaptation and a smaller bounded correction.
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][1], candidates[0][2], candidates[0][0]


def evaluate_day(frame: pd.DataFrame, day: int) -> pd.DataFrame:
    g = frame[frame.day.eq(day)].sort_values("block_start_utc").copy().reset_index(drop=True)
    history: list[dict] = []
    innovation_history: list[tuple[pd.Timestamp, float]] = []
    krr_preds, combo_preds, kalman_preds = [], [], []
    selected_ell, selected_alpha, selected_train_window, selected_huber = [], [], [], []
    selected_bias_window, selected_cap, combo_inner_mae = [], [], []
    selected_kalman_qr, selected_kalman_cap, kalman_inner_mae = [], [], []
    inner_mae, history_n = [], []
    for row in g.itertuples(index=False):
        now = row.block_start_utc
        if history and (now - history[-1]["time"]).total_seconds() > GAP_RESET_S:
            history = []
            innovation_history = []
        hist = pd.DataFrame(history)
        n_hist = len(hist)
        history_n.append(n_hist)
        pred = 0.0
        ell, alpha, train_window, huber_label, val_mae, n_val = 1.0, 100.0, MAX_WINDOW, "none", np.nan, 0
        if n_hist >= MIN_HISTORY:
            recent = hist.tail(MAX_WINDOW).reset_index(drop=True)
            ell, alpha, train_window, huber_label, val_mae, n_val = tune_from_past(recent)
            fit_hist = recent.tail(train_window)
            huber_delta_w = dict(HUBER_DELTAS)[huber_label]
            pred = float(fit_krr(fit_hist[FEATURES].to_numpy(float),
                                 fit_hist.baseline_error_w.to_numpy(float),
                                 np.asarray([[getattr(row, c) for c in FEATURES]], dtype=float),
                                 ell, alpha, huber_delta_w)[0])
        pred = float(np.clip(pred, -KRR_CAP_W, KRR_CAP_W))
        bias_window, cap_w, bias_cv_mae = DEFAULT_BIAS_WINDOW, DEFAULT_CORRECTION_CAP_W, np.nan
        combo_start = max(MIN_HISTORY, len(hist) - INNER_HOLDOUT)
        combo_scores = []
        for w in BIAS_WINDOWS:
            for candidate_cap in CORRECTION_CAPS_W:
                errors = []
                for j in range(combo_start, len(hist)):
                    prior_innovations = hist.innovation_w.iloc[max(0, j - w):j].to_numpy(float)
                    if len(prior_innovations) < w:
                        continue
                    validation_correction = float(np.clip(
                        hist.krr_prediction_w.iloc[j] + np.median(prior_innovations),
                        -candidate_cap, candidate_cap))
                    errors.append(abs(hist.baseline_error_w.iloc[j] - validation_correction))
                if errors:
                    # Prefer a longer median window and smaller cap when scores tie.
                    combo_scores.append((float(np.mean(errors)), -w, candidate_cap, w))
        if combo_scores:
            bias_cv_mae, _, cap_w, bias_window = min(combo_scores)
        prior_innov = [e for _, e in innovation_history[-bias_window:]]
        online = float(np.median(prior_innov)) if len(prior_innov) >= bias_window else 0.0
        total = float(np.clip(pred + online, -cap_w, cap_w))
        kalman_qr, kalman_cap, kalman_cv_mae = tune_kalman_from_past(hist)
        kalman_offset = kalman_bias_forecast(hist.innovation_w.to_numpy(float), kalman_qr) if n_hist >= MIN_HISTORY else 0.0
        kalman_total = float(np.clip(pred + kalman_offset, -kalman_cap, kalman_cap))
        krr_preds.append(pred)
        combo_preds.append(total)
        kalman_preds.append(kalman_total)
        selected_ell.append(ell)
        selected_alpha.append(alpha)
        selected_train_window.append(train_window)
        selected_huber.append(huber_label)
        selected_bias_window.append(bias_window)
        selected_cap.append(cap_w)
        combo_inner_mae.append(bias_cv_mae)
        selected_kalman_qr.append(kalman_qr)
        selected_kalman_cap.append(kalman_cap)
        kalman_inner_mae.append(kalman_cv_mae)
        inner_mae.append(val_mae)
        # The current target is appended only after the prediction is recorded.
        innovation = float(row.baseline_error_w - pred)
        innovation_history.append((now, innovation))
        history.append({"time": now, "baseline_error_w": float(row.baseline_error_w),
                        "krr_prediction_w": pred, "innovation_w": innovation,
                        **{c: float(getattr(row, c)) for c in FEATURES}})
    g["krr_correction_w"] = krr_preds
    g["krr_plus_tuned_bias_correction_w"] = combo_preds
    g["krr_prediction_w"] = g.predicted_w + g.krr_correction_w
    g["krr_plus_tuned_bias_prediction_w"] = g.predicted_w + g.krr_plus_tuned_bias_correction_w
    g["krr_plus_kalman_correction_w"] = kalman_preds
    g["krr_plus_kalman_prediction_w"] = g.predicted_w + g.krr_plus_kalman_correction_w
    g["selected_length_scale"] = selected_ell
    g["selected_alpha"] = selected_alpha
    g["selected_train_window"] = selected_train_window
    g["selected_huber_delta"] = selected_huber
    g["selected_bias_window"] = selected_bias_window
    g["selected_correction_cap_w"] = selected_cap
    g["inner_bias_pipeline_mae_w"] = combo_inner_mae
    g["selected_kalman_q_over_r"] = selected_kalman_qr
    g["selected_kalman_correction_cap_w"] = selected_kalman_cap
    g["inner_kalman_pipeline_mae_w"] = kalman_inner_mae
    g["inner_walkforward_mae_w"] = inner_mae
    g["prior_blocks"] = history_n
    return g


def score(frame: pd.DataFrame, pred_col: str) -> dict:
    e = frame.measured_w.to_numpy(float) - frame[pred_col].to_numpy(float)
    return {"n_blocks": int(len(e)), "mae_w": float(np.mean(np.abs(e))),
            "p90_abs_error_w": float(np.quantile(np.abs(e), .9)),
            "bias_w": float(np.mean(e))}


def main() -> None:
    d = pd.read_csv(INPUT, parse_dates=["block_start_utc"])
    outputs = [evaluate_day(d, int(day)) for day in sorted(d.day.unique())]
    oof = pd.concat(outputs, ignore_index=True)
    oof.to_csv(OUT_BLOCKS, index=False, encoding="utf-8-sig")
    results = []
    for day, g in oof.groupby("day", sort=True):
        warm = g.prior_blocks.ge(MIN_HISTORY)
        for model, pred_col in [("physics_baseline", "predicted_w"),
                                ("prequential_kernel", "krr_prediction_w"),
                                ("prequential_kernel_plus_adaptive_bias", "krr_plus_tuned_bias_prediction_w"),
                                ("prequential_kernel_plus_kalman_bias", "krr_plus_kalman_prediction_w"),
                                ("nested_selected_online_rule", "online_predicted_w")]:
            full = score(g, pred_col)
            paired = score(g.loc[warm], pred_col) if warm.any() else {"mae_w": None, "p90_abs_error_w": None, "n_blocks": 0}
            results.append({"day": int(day), "model": model, **full,
                            "warm_state_n": paired["n_blocks"], "warm_state_mae_w": paired["mae_w"],
                            "baseline_mae_all_blocks_w": score(g, "predicted_w")["mae_w"],
                            "baseline_mae_warm_state_w": score(g.loc[warm], "predicted_w")["mae_w"] if warm.any() else None,
                            "inner_tuning_updates": int(g.selected_alpha.notna().sum() - (g.prior_blocks < MIN_HISTORY).sum()),
                            "hyperparameter_choices": ";".join(
                                f"ell={ell:g},alpha={alpha:g},lookback={window},huber={huber}×{n}"
                                for (ell, alpha, window, huber), n in
                                pd.DataFrame({"ell": g.selected_length_scale,
                                              "alpha": g.selected_alpha,
                                              "window": g.selected_train_window,
                                              "huber": g.selected_huber_delta})
                                .loc[g.prior_blocks.ge(MIN_HISTORY)]
                                .value_counts().items()),
                            "bias_pipeline_choices": ";".join(
                                f"median={window},cap={cap:g}×{n}"
                                for (window, cap), n in
                                pd.DataFrame({"window": g.selected_bias_window,
                                              "cap": g.selected_correction_cap_w})
                                .loc[g.prior_blocks.ge(MIN_HISTORY)]
                                .value_counts().items())})
    day_scores = pd.DataFrame(results)
    day_scores.to_csv(OUT_DAYS, index=False, encoding="utf-8-sig")
    clean = oof[oof.day.le(5)].copy()
    selected = oof[oof.day.le(5)].copy()
    # The previous nested selection procedure is evaluated on the same block grain.
    base = score(clean, "predicted_w")
    summary_methods = []
    for model, pred_col in [("prequential_kernel", "krr_prediction_w"),
                            ("prequential_kernel_plus_adaptive_bias", "krr_plus_tuned_bias_prediction_w"),
                            ("prequential_kernel_plus_kalman_bias", "krr_plus_kalman_prediction_w"),
                            ("nested_selected_online_rule", "online_predicted_w")]:
        pred = score(clean, pred_col)
        per_day = day_scores[(day_scores.day.le(5)) & day_scores.model.eq(model)]
        summary_methods.append({"model": model, "pooled_mae_w": pred["mae_w"],
                                "pooled_p90_abs_error_w": pred["p90_abs_error_w"], "pooled_bias_w": pred["bias_w"],
                                "baseline_mae_w": base["mae_w"],
                                "improvement_pct": 100 * (base["mae_w"] - pred["mae_w"]) / base["mae_w"],
                                "clean_day_wins": int((per_day.mae_w < per_day.baseline_mae_all_blocks_w).sum()),
                                "macro_day_mae_w": float(per_day.mae_w.mean()),
                                "warm_state_day_wins": int((per_day.warm_state_mae_w < per_day.baseline_mae_warm_state_w).sum())})
    day6 = day_scores[day_scores.day.eq(6)].to_dict(orient="records")
    choices = oof[(oof.day.le(5)) & oof.prior_blocks.ge(MIN_HISTORY)][
        ["selected_length_scale", "selected_alpha", "selected_train_window", "selected_huber_delta"]]
    choice_counts = {f"ell={ell:g},alpha={alpha:g},lookback={window},huber={huber}": int(n)
                     for (ell, alpha, window, huber), n in choices.value_counts().items()}
    bias_choices = oof[(oof.day.le(5)) & oof.prior_blocks.ge(MIN_HISTORY)][
        ["selected_bias_window", "selected_correction_cap_w"]]
    bias_choice_counts = {f"median={window},cap={cap:g}W": int(n)
                          for (window, cap), n in bias_choices.value_counts().items()}
    kalman_choices = oof[(oof.day.le(5)) & oof.prior_blocks.ge(MIN_HISTORY)][
        ["selected_kalman_q_over_r", "selected_kalman_correction_cap_w"]]
    kalman_choice_counts = {f"q_over_r={ratio:g},cap={cap:g}W": int(n)
                            for (ratio, cap), n in kalman_choices.value_counts().items()}
    d6_scores = day_scores[day_scores.day.eq(6)].set_index("model")
    d6_base_p90 = float(d6_scores.loc["physics_baseline", "p90_abs_error_w"])
    for m in summary_methods:
        if m["model"] == "nested_selected_online_rule":
            m["day6_p90_w"] = float(d6_scores.loc[m["model"], "p90_abs_error_w"])
            m["tail_guard_pass"] = bool(m["pooled_p90_abs_error_w"] <= base["p90_abs_error_w"]
                                         and m["day6_p90_w"] <= d6_base_p90)
        else:
            m["day6_p90_w"] = float(d6_scores.loc[m["model"], "p90_abs_error_w"])
            m["tail_guard_pass"] = bool(m["pooled_p90_abs_error_w"] <= base["p90_abs_error_w"]
                                         and m["day6_p90_w"] <= d6_base_p90)
    adopted = any(m["improvement_pct"] >= 10.0 and m["clean_day_wins"] >= 4
                  and m["warm_state_day_wins"] >= 4 and m["tail_guard_pass"]
                  for m in summary_methods if m["model"] != "nested_selected_online_rule")
    active_clean = clean[clean.prior_blocks.ge(MIN_HISTORY)].copy()
    tuning_gap = {}
    for label, inner_col, pred_col in [
        ("adaptive_median", "inner_bias_pipeline_mae_w", "krr_plus_tuned_bias_prediction_w"),
        ("kalman", "inner_kalman_pipeline_mae_w", "krr_plus_kalman_prediction_w")]:
        active = active_clean.dropna(subset=[inner_col])
        tuning_gap[label] = {"active_future_blocks": int(len(active)),
                             "mean_prior_inner_cv_mae_w": float(active[inner_col].mean()),
                             "mean_next_block_abs_error_w": float(np.mean(np.abs(active.measured_w - active[pred_col]))),
                             "note": "Inner score uses only preceding blocks; outer score uses the next untouched block."}
    row_data = pd.read_csv(ROWS_IN, low_memory=False)
    row_data["timestamp_utc"] = pd.to_datetime(row_data.timestamp_utc, utc=True, errors="coerce")
    row_data["block_start_utc"] = row_data.timestamp_utc.dt.floor("5min")
    row_lookup = oof[["day", "block_start_utc", "krr_plus_tuned_bias_correction_w",
                      "krr_plus_kalman_correction_w", "selected_bias_window", "selected_correction_cap_w",
                      "selected_kalman_q_over_r", "selected_kalman_correction_cap_w"]]
    row_data = row_data.merge(row_lookup, on=["day", "block_start_utc"], how="left", validate="many_to_one")
    row_data["krr_online_prediction_w"] = row_data.predicted_w + row_data.krr_plus_tuned_bias_correction_w
    row_data["krr_online_error_w"] = row_data.measured_w - row_data.krr_online_prediction_w
    row_data["krr_kalman_online_prediction_w"] = row_data.predicted_w + row_data.krr_plus_kalman_correction_w
    row_data["krr_kalman_online_error_w"] = row_data.measured_w - row_data.krr_kalman_online_prediction_w
    row_data.to_csv(ROWS_OUT, index=False, encoding="utf-8-sig")
    row_level = {}
    for label, condition in [("clean_days_1_5", row_data.day.le(5)), ("day6_anomaly_stress", row_data.day.eq(6))]:
        scored = row_data.loc[condition].dropna(subset=["krr_online_prediction_w"])
        eb = scored.measured_w - scored.predicted_w
        ec = scored.krr_online_error_w
        ek = scored.krr_kalman_online_error_w
        row_level[label] = {"n_rows": int(len(scored)),
                            "baseline_mae_w": float(np.mean(np.abs(eb))),
                            "corrected_mae_w": float(np.mean(np.abs(ec))),
                            "baseline_p90_abs_error_w": float(np.quantile(np.abs(eb), .9)),
                            "corrected_p90_abs_error_w": float(np.quantile(np.abs(ec), .9)),
                            "baseline_bias_w": float(np.mean(eb)),
                            "corrected_bias_w": float(np.mean(ec)),
                            "kalman_corrected_mae_w": float(np.mean(np.abs(ek))),
                            "kalman_corrected_p90_abs_error_w": float(np.quantile(np.abs(ek), .9)),
                            "kalman_corrected_bias_w": float(np.mean(ek))}
    payload = {"validation": "prequential nested forward validation; inner model/parameter selection uses only completed past 5-minute residual blocks; outer test is each future block; base predictions are day-held-out physics-model outputs",
               "input_blocks_clean_days": int(clean.shape[0]), "clean_days": [1, 2, 3, 4, 5],
               "day6_role": "anomaly stress only", "feature_set": FEATURES,
               "hyperparameter_grid": {"rbf_length_scale_standardized": LENGTHS, "alpha": ALPHAS,
                                       "huber_delta_w": [label for label, _ in HUBER_DELTAS],
                                       "lookback_blocks": TRAIN_WINDOWS,
                                       "innovation_median_window_blocks": BIAS_WINDOWS,
                                       "total_correction_cap_w": CORRECTION_CAPS_W,
                                       "kalman_q_over_r": KALMAN_Q_RATIOS},
               "tuning": {"inner_validation_blocks": INNER_HOLDOUT,
                          "minimum_history_blocks": MIN_HISTORY, "history_reset_gap_seconds": GAP_RESET_S},
               "bias_combination": {"type": "past-only selected median of completed KRR innovation residuals",
                                    "default_window_blocks": DEFAULT_BIAS_WINDOW,
                                    "default_cap_w": DEFAULT_CORRECTION_CAP_W},
               "physical_baseline": base, "models": summary_methods,
               "online_hyperparameter_choice_counts": choice_counts,
               "online_bias_pipeline_choice_counts": bias_choice_counts,
               "online_kalman_pipeline_choice_counts": kalman_choice_counts,
               "inner_to_next_block_generalization": tuning_gap,
               "day6_anomaly_stress": day6,
               "row_level_5s_results": row_level,
               "adoption_gate": "at least 10% pooled clean-day MAE improvement, at least 4/5 full-day wins, at least 4/5 warm-state wins, and no pooled-clean or Day6-stress P90 worsening; project engineering gate, not external standard",
               "adopted": bool(adopted), "production_ready": False,
               "interpretation": "This nested walk-forward protocol directly tests tuning overfit on future blocks. Day-level sample count remains five, route/weather distributions shift by day, and race-weather hindcast inputs are retrospectively known; cross-event performance is not established."}
    OUT_SUMMARY.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# BWSC2025 online kernel residual learner", "",
             "## Validation design", "",
             "Each 5-minute block is predicted using completed residual blocks only. Before predicting, an RBF kernel-ridge learner selects length scale, regularization, 12/24/36/48-block lookback and squared-error versus Huber loss (δ=30/60/100 W) from expanding one-step forecasts inside the prior 12 blocks (minimum 8 earlier training blocks), then refits using the selected history. Two causal residual-state corrections are compared: a past-only median (window 3/6/9 blocks) and a robust local-level Kalman filter whose process/measurement variance ratio (0.001/0.01/0.05/0.1/0.5) and total correction cap (30/50/75/100/150 W) are selected from prior one-step errors. Kalman updates clip past innovations at three robust standard deviations. The current target is excluded from fitting and parameter choice. Histories reset after gaps over 10 minutes.",
             "", "## Day 1–5 results", "",
             "| Model | pooled MAE (W) | improvement vs physics | P90 (W) | day wins | warm-state wins |",
             "|---|---:|---:|---:|---:|---:|"]
    for m in summary_methods:
        lines.append(f"| {m['model']} | {m['pooled_mae_w']:.1f} | {m['improvement_pct']:+.1f}% | {m['pooled_p90_abs_error_w']:.1f} | {m['clean_day_wins']}/5 | {m['warm_state_day_wins']}/5 |")
    lines += ["", "## Hyperparameter and bias diagnostics", "",
              f"Selected RBF/Huber configurations over clean-day prequential updates: `{choice_counts}`.",
              f"Selected innovation window/cap configurations: `{bias_choice_counts}`.",
              f"Selected Kalman process/measurement ratio and cap configurations: `{kalman_choice_counts}`.",
              "", "The time-ordered holdout directly tests whether model selection on prior residuals generalizes to later blocks. It avoids selecting one global configuration on the same outer-fold outcomes. Repeated blocks remain autocorrelated, so daily paired results and P90 are retained; the five race days are not treated as hundreds of independent days.",
              "", f"For the median-corrected model, prior inner-validation MAE averaged {tuning_gap['adaptive_median']['mean_prior_inner_cv_mae_w']:.1f} W and the next untouched-block MAE averaged {tuning_gap['adaptive_median']['mean_next_block_abs_error_w']:.1f} W over {tuning_gap['adaptive_median']['active_future_blocks']} blocks. For the Kalman model the corresponding values were {tuning_gap['kalman']['mean_prior_inner_cv_mae_w']:.1f} W and {tuning_gap['kalman']['mean_next_block_abs_error_w']:.1f} W. The inner-to-next gap makes tuning optimism visible; adoption is based on the outer next-block scores.",
              "", "Day 6 is reported only as anomaly stress and never selects the model. The inputs are retrospective race-weather values rather than archived forecasts; forecast error is not represented. This method is not connected to the model replay, Grafana, or commands.",
              "", f"Applying the median correction to the original 5-second rows gives clean-day MAE {row_level['clean_days_1_5']['baseline_mae_w']:.1f}→{row_level['clean_days_1_5']['corrected_mae_w']:.1f} W; the Kalman version gives {row_level['clean_days_1_5']['kalman_corrected_mae_w']:.1f} W. On Day6, P90 is {row_level['day6_anomaly_stress']['baseline_p90_abs_error_w']:.1f} W for baseline, {row_level['day6_anomaly_stress']['corrected_p90_abs_error_w']:.1f} W for median, and {row_level['day6_anomaly_stress']['kalman_corrected_p90_abs_error_w']:.1f} W for Kalman. Full row output: `{ROWS_OUT.relative_to(ROOT)}`.",
              "", "## Current decision", "",
              "A candidate is promoted to shadow evaluation only if it meets the project gate of ≥10% pooled clean-day MAE improvement, at least four of five day wins, at least four of five warm-state wins, and no pooled-clean or Day 6 stress-set P90 worsening. This is a provisional engineering rule. Passing it does not establish BWSC2027 transfer or satisfy the separate 23.8 W speed-decision error gate.",
              "", "Outputs: `residual_online_kernel_oof.csv`, `residual_online_kernel_by_day.csv`, `residual_online_kernel_summary.json`, this report, and the full five-second corrected log.", ""]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    if adopted:
        best = min((m for m in summary_methods if m["model"] != "nested_selected_online_rule"), key=lambda m: m["pooled_mae_w"])
        OUT_MODEL.write_text(json.dumps({"model_type": best["model"],
                                         "stage": "shadow_candidate_not_connected_to_replay_grafana_or_commands",
                                         "production_ready": False, "selected_hyperparameters": choice_counts,
                                         "features": FEATURES, "lookback_blocks": TRAIN_WINDOWS,
                                         "huber_delta_candidates_w": [label for label, _ in HUBER_DELTAS],
                                         "history_reset_gap_seconds": GAP_RESET_S,
                                         "correction_cap_w": CORRECTION_CAPS_W,
                                         "best_candidate_oof_metrics": best,
                                         "online_hyperparameter_choice_counts": choice_counts,
                                         "online_bias_pipeline_choice_counts": bias_choice_counts,
                                         "online_kalman_pipeline_choice_counts": kalman_choice_counts,
                                         "validation_scope": payload["validation"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"baseline": base, "models": summary_methods,
                      "hyperparameter_choice_counts": choice_counts, "day6": day6,
                      "adopted_to_shadow": adopted,
                      "row_level_5s": row_level, "tuning_gap": tuning_gap,
                      "outputs": [str(OUT_BLOCKS), str(OUT_DAYS), str(OUT_SUMMARY), str(OUT_REPORT), str(ROWS_OUT)]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
