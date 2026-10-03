#!/usr/bin/env python3
"""Past-only one-step nonlinear residual models for a BWSC2025 shadow study."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/full_race_residual_corrected_5s.csv"
COMPONENTS = ROOT / "data/processed/full_race_component_replay_5s.csv"
OUT_ROWS = ROOT / "data/processed/full_race_residual_online_tree_5s.csv"
OUT_DAYS = ROOT / "reports/residual_online_tree_by_day.csv"
OUT_SUMMARY = ROOT / "reports/residual_online_tree_summary.json"
OUT_REPORT = ROOT / "reports/residual_online_tree_report.md"

MASS_PROXY_KG = 314.0
MIN_HISTORY = 360
TUNE_EVERY = 300
INNER_HOLDOUT = 60
REFIT_EVERY = 12
MAX_WINDOW = 1200
RESET_GAP_S = 600.0

FEATURES = ["previous_residual_w", "speed_kmh", "inertial_power_w",
            "grade_pct", "wind_speed_mps", "temperature_c",
            "air_density_kgm3", "distance_km"]
WINDOWS = [300, 600, 1200]
MODELS = [
    ("hgb_leaf7_iter40", "hgb", 7, 40),
    ("hgb_leaf15_iter40", "hgb", 15, 40),
    ("hgb_leaf7_iter80", "hgb", 7, 80),
    ("extra_depth4_leaf30", "extra", 4, 30),
    ("extra_depth6_leaf50", "extra", 6, 50),
]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.sort_values("timestamp_utc").copy().reset_index(drop=True)
    g["timestamp_utc"] = pd.to_datetime(g.timestamp_utc, utc=True, errors="coerce")
    speed = pd.to_numeric(g["Speed[kmh]"], errors="coerce").to_numpy(float) / 3.6
    delta_s = g.timestamp_utc.diff().dt.total_seconds().to_numpy(float)
    smoothed = np.full(len(g), np.nan)
    acceleration = np.zeros(len(g))
    previous_residual = np.zeros(len(g))
    baseline_error = pd.to_numeric(g.baseline_error_w, errors="coerce").to_numpy(float)
    for i in range(len(g)):
        gap = i == 0 or not np.isfinite(delta_s[i]) or delta_s[i] > RESET_GAP_S
        if gap or not np.isfinite(speed[i]):
            smoothed[i] = speed[i]
        else:
            smoothed[i] = 0.35 * speed[i] + 0.65 * smoothed[i - 1]
            acceleration[i] = (smoothed[i] - smoothed[i - 1]) / max(1.0, delta_s[i])
            previous_residual[i] = baseline_error[i - 1] if np.isfinite(baseline_error[i - 1]) else 0.0
        if not np.isfinite(smoothed[i]):
            smoothed[i] = speed[i]
    g["previous_residual_w"] = previous_residual
    g["speed_kmh"] = pd.to_numeric(g["Speed[kmh]"], errors="coerce")
    g["inertial_power_w"] = MASS_PROXY_KG * smoothed * np.clip(acceleration, -0.5, 0.5)
    g["grade_pct"] = pd.to_numeric(g["route_grade_smoothed_pct"], errors="coerce")
    g["wind_speed_mps"] = pd.to_numeric(g["wind_speed_10m"], errors="coerce")
    g["temperature_c"] = pd.to_numeric(g["temperature_2m"], errors="coerce")
    # Standard-atmosphere density proxy; temperature is the only ambient field needed.
    temp_k = g["temperature_c"].to_numpy(float) + 273.15
    g["air_density_kgm3"] = 1.225 * 288.15 / np.maximum(temp_k, 1.0)
    g["distance_km"] = pd.to_numeric(g.distance_from_darwin_m, errors="coerce") / 1000.0
    return g


def estimator(model_spec: tuple[str, str, int, int], seed: int):
    name, kind, complexity, strength = model_spec
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.05,
            max_iter=strength, max_leaf_nodes=complexity,
            l2_regularization=10.0, min_samples_leaf=30,
            random_state=seed,
        )
    return ExtraTreesRegressor(
        n_estimators=60, max_depth=complexity,
        min_samples_leaf=strength, max_features=0.8,
        random_state=seed, n_jobs=1,
    )


def impute(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(train, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    return np.where(np.isfinite(train), train, medians), np.where(np.isfinite(other), other, medians)


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def evaluate_day(frame: pd.DataFrame) -> pd.DataFrame:
    g = build_features(frame)
    x_all = g[FEATURES].to_numpy(float)
    y_all = pd.to_numeric(g.baseline_error_w, errors="coerce").to_numpy(float)
    baseline = pd.to_numeric(g.predicted_w, errors="coerce").to_numpy(float)
    measured = pd.to_numeric(g.measured_w, errors="coerce").to_numpy(float)
    model_correction = np.zeros(len(g))
    prior_counts = np.zeros(len(g), dtype=int)
    selected = ["warmup_physics"] * len(g)
    inner_scores = np.full(len(g), np.nan)
    fitted = None
    fitted_name = "warmup_physics"
    fitted_window = 0
    last_tune = -TUNE_EVERY
    last_time = None
    stream_start = 0
    current_inner_score = np.nan
    current_tune_block = -1
    tune_blocks = np.full(len(g), -1, dtype=int)
    for i in range(len(g)):
        now = g.timestamp_utc.iloc[i]
        if last_time is not None and (now - last_time).total_seconds() > RESET_GAP_S:
            fitted = None
            fitted_name = "warmup_physics"
            fitted_window = 0
            last_tune = -TUNE_EVERY
            stream_start = i
            current_inner_score = np.nan
            current_tune_block = -1
        last_time = now
        n = i - stream_start
        prior_counts[i] = n
        if not np.isfinite(y_all[i]):
            continue
        if n >= MIN_HISTORY and (n - last_tune >= TUNE_EVERY):
            val_start_rel = max(120, n - INNER_HOLDOUT)
            train_end_rel = val_start_rel
            scores = []
            for model_spec in MODELS:
                for window in WINDOWS:
                    train_start_rel = max(0, train_end_rel - window)
                    if train_end_rel - train_start_rel < min(120, window):
                        continue
                    train_start = stream_start + train_start_rel
                    train_end = stream_start + train_end_rel
                    val_start = stream_start + val_start_rel
                    xt, xv = impute(x_all[train_start:train_end], x_all[val_start:i])
                    yt = y_all[train_start:train_end]
                    if not len(xv) or not np.isfinite(yt).all():
                        continue
                    m = estimator(model_spec, seed=0)
                    m.fit(xt, yt)
                    pred = m.predict(xv)
                    scores.append((mae(y_all[val_start:i], pred), model_spec[0], window, model_spec))
            if scores:
                best_score, fitted_name, fitted_window, best_spec = min(scores, key=lambda row: (row[0], row[1], row[2]))
                current_inner_score = best_score
                current_tune_block = i
            last_tune = n
        tune_blocks[i] = current_tune_block
        inner_scores[i] = current_inner_score
        if n >= MIN_HISTORY and fitted_name != "warmup_physics" and (fitted is None or n % REFIT_EVERY == 0):
            train_start = max(stream_start, i - fitted_window)
            xt, _ = impute(x_all[train_start:i], x_all[i:i + 1])
            active_spec = next(spec for spec in MODELS if spec[0] == fitted_name)
            fitted = estimator(active_spec, seed=0)
            fitted.fit(xt, y_all[train_start:i])
        if n >= MIN_HISTORY and fitted is not None:
            train_start = max(stream_start, i - fitted_window)
            _, xnow = impute(x_all[train_start:i], x_all[i:i + 1])
            model_correction[i] = float(fitted.predict(xnow)[0])
            selected[i] = f"{fitted_name}_w{fitted_window}"

    g["tree_correction_w"] = model_correction
    g["tree_prediction_w"] = baseline + model_correction
    g["tree_prior_samples"] = prior_counts
    g["tree_selected_model"] = selected
    g["tree_inner_mae_w"] = inner_scores
    g["tree_tune_block"] = tune_blocks
    return g


def metrics(frame: pd.DataFrame, pred_col: str) -> dict:
    err = frame.measured_w.to_numpy(float) - frame[pred_col].to_numpy(float)
    return {"n": int(len(err)), "mae_w": float(np.mean(np.abs(err))),
            "p90_abs_error_w": float(np.quantile(np.abs(err), .9)),
            "bias_w": float(np.mean(err))}


def main() -> None:
    residual = pd.read_csv(INPUT, low_memory=False)
    source = pd.read_csv(COMPONENTS, low_memory=False,
                         usecols=["day", "timestamp_utc", "temperature_2m", "wind_speed_10m", "route_grade_smoothed_pct"])
    residual = residual.merge(source, on=["day", "timestamp_utc"], how="left", validate="one_to_one")
    result = pd.concat([evaluate_day(g.copy()) for _, g in residual.groupby("day", sort=True)], ignore_index=True)
    result.to_csv(OUT_ROWS, index=False, encoding="utf-8-sig")

    day_rows = []
    for day, rows in result.groupby("day", sort=True):
        for label, subset in [("all", rows), ("warm_state", rows[rows.tree_prior_samples.ge(MIN_HISTORY)])]:
            day_rows.append({"day": int(day), "scope": label,
                             "baseline_mae_w": metrics(subset, "predicted_w")["mae_w"],
                             "tree_mae_w": metrics(subset, "tree_prediction_w")["mae_w"],
                             "baseline_p90_w": metrics(subset, "predicted_w")["p90_abs_error_w"],
                             "tree_p90_w": metrics(subset, "tree_prediction_w")["p90_abs_error_w"],
                             "n": int(len(subset))})
    pd.DataFrame(day_rows).to_csv(OUT_DAYS, index=False, encoding="utf-8-sig")

    clean, day6 = result[result.day.le(5)], result[result.day.eq(6)]
    warm_clean = clean[clean.tree_prior_samples.ge(MIN_HISTORY)]
    warm_day6 = day6[day6.tree_prior_samples.ge(MIN_HISTORY)]
    active = clean[clean.tree_tune_block.ge(0)]
    per_tune_rows = []
    for block_id, block in active.groupby("tree_tune_block", sort=True):
        per_tune_rows.append({"block": int(block_id), "inner_mae_w": float(block.tree_inner_mae_w.iloc[0]),
                              "outer_mae_w": metrics(block, "tree_prediction_w")["mae_w"]})
    per_tune = pd.DataFrame(per_tune_rows)
    summary = {
        "validation": "past-only nested prequential one-step replay; model family and training window are selected on the previous 60 rows every 300 completed rows; fitted estimator refreshes every 12 rows using completed rows only",
        "feature_set": FEATURES,
        "candidate_models": ["HistGradientBoostingRegressor (leaf count 7/15, 40/80 iterations)",
                             "ExtraTreesRegressor (depth 4/6, minimum leaf 30/50)"],
        "training_windows_rows": WINDOWS,
        "minimum_history_rows": MIN_HISTORY,
        "retune_every_rows": TUNE_EVERY,
        "model_refit_every_rows": REFIT_EVERY,
        "inner_holdout_rows": INNER_HOLDOUT,
        "history_reset_gap_seconds": RESET_GAP_S,
        "inertial_mass_proxy_kg": MASS_PROXY_KG,
        "clean_days_1_5": {
            "baseline": metrics(clean, "predicted_w"),
            "tree_candidate": metrics(clean, "tree_prediction_w"),
            "warm_state_baseline": metrics(warm_clean, "predicted_w"),
            "warm_state_tree": metrics(warm_clean, "tree_prediction_w"),
            "inner_to_next_block": {"n_blocks": int(len(per_tune)),
                                    "mean_inner_mae_w": float(per_tune.inner_mae_w.mean()),
                                    "mean_untouched_outer_block_mae_w": float(per_tune.outer_mae_w.mean())},
        },
        "day6_stress": {"baseline": metrics(day6, "predicted_w"), "tree_candidate": metrics(day6, "tree_prediction_w"),
                         "warm_state_baseline": metrics(warm_day6, "predicted_w"), "warm_state_tree": metrics(warm_day6, "tree_prediction_w")},
        "day_metrics": day_rows,
        "selected_configurations": {str(k): int(v) for k, v in
                                    result[result.tree_prior_samples.ge(MIN_HISTORY)].tree_selected_model.value_counts().items()},
        "production_ready": False,
        "interpretation": "This is a logged-speed-conditioned one-step residual candidate. It does not establish multi-step MPC accuracy or cross-event transfer.",
        "outputs": [str(OUT_ROWS), str(OUT_DAYS), str(OUT_SUMMARY)],
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_REPORT.write_text("\n".join([
        "# Online nonlinear residual candidate (5-second one-step)", "",
        "The candidate compares shallow histogram gradient boosting and extremely randomized trees. At each 300-row update, model family and history window are selected using a 60-row inner holdout made entirely from completed data. The selected model is refit every 12 rows using only residuals that had already been observed. It resets after a logging gap over ten minutes.", "",
        f"Day1–5 MAE: baseline {summary['clean_days_1_5']['baseline']['mae_w']:.1f} W; tree candidate {summary['clean_days_1_5']['tree_candidate']['mae_w']:.1f} W. P90 {summary['clean_days_1_5']['baseline']['p90_abs_error_w']:.1f}→{summary['clean_days_1_5']['tree_candidate']['p90_abs_error_w']:.1f} W.",
        f"Warm-state MAE: baseline {summary['clean_days_1_5']['warm_state_baseline']['mae_w']:.1f} W; tree candidate {summary['clean_days_1_5']['warm_state_tree']['mae_w']:.1f} W. Day6 stress MAE {summary['day6_stress']['baseline']['mae_w']:.1f}→{summary['day6_stress']['tree_candidate']['mae_w']:.1f} W.",
        f"Mean inner MAE {summary['clean_days_1_5']['inner_to_next_block']['mean_inner_mae_w']:.1f} W; following untouched outer-block MAE {summary['clean_days_1_5']['inner_to_next_block']['mean_untouched_outer_block_mae_w']:.1f} W.", "",
        "Feature inputs include logged speed, causal inertial power using the 314 kg proxy, route grade, distance, wind and ambient temperature. These are predictive features only; they do not identify motor efficiency, true mass or sensor bias. Future MPC inputs and forecast errors are not simulated.", "",
        f"Outputs: `{OUT_ROWS.relative_to(ROOT)}`, `{OUT_DAYS.relative_to(ROOT)}`, `{OUT_SUMMARY.relative_to(ROOT)}`.", ""
    ]), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
