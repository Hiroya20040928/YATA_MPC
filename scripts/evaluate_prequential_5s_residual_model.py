#!/usr/bin/env python3
"""Nested prequential evaluation of one-step 5-second residual filters."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/full_race_residual_corrected_5s.csv"
OUT_ROWS = ROOT / "data/processed/full_race_residual_prequential_5s.csv"
OUT_SUMMARY = ROOT / "reports/residual_prequential_5s_summary.json"
OUT_DAYS = ROOT / "reports/residual_prequential_5s_by_day.csv"
OUT_REPORT = ROOT / "reports/residual_prequential_5s_report.md"
OUT_MODEL = ROOT / "models/vehicle_residual_prequential_5s_candidate.json"

MIN_HISTORY = 120  # 10 minutes at 5-second sampling
TUNE_EVERY = 60    # reselect using completed data every five minutes
INNER_HOLDOUT = 60
GAP_RESET_S = 600.0
CAPS_W = [50.0, 100.0, 200.0, 300.0, 500.0, 1000.0]
CONFIGS = ([ ("median", w) for w in (3, 6, 12, 24) ]
           + [ ("ewma", a) for a in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75) ]
           + [ ("persistence", phi) for phi in (0.25, 0.5, 0.75, 1.0) ]
           + [ ("ar1", n) for n in (60, 120, 300) ])


def forecast(history: np.ndarray, kind: str, parameter: float) -> float:
    y = np.asarray(history, dtype=float)
    if len(y) == 0:
        return 0.0
    if kind == "median":
        return float(np.median(y[-int(parameter):]))
    if kind == "ewma":
        y = y[-120:]
        ages = np.arange(len(y) - 1, -1, -1, dtype=float)
        weights = float(parameter) * (1.0 - float(parameter)) ** ages
        weights /= weights.sum()
        return float(np.dot(weights, y))
    if kind == "persistence":
        return float(parameter) * float(y[-1])
    # Fit an intercept and AR(1) coefficient using completed prior pairs only.
    lookback = min(int(parameter), len(y) - 1)
    if lookback < 20:
        return float(np.median(y[-min(6, len(y)):]))
    target = y[-lookback:]
    lag = y[-lookback-1:-1]
    design = np.column_stack([np.ones(len(lag)), lag])
    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    beta[1] = np.clip(beta[1], -0.25, 0.98)
    return float(beta[0] + beta[1] * y[-1])


def tune(history: list[float]) -> tuple[str, float, float, float]:
    start = max(20, len(history) - INNER_HOLDOUT)
    candidates = []
    for kind, parameter in CONFIGS:
        for cap in CAPS_W:
            errors = []
            for j in range(start, len(history)):
                estimate = forecast(np.asarray(history[:j]), kind, parameter)
                estimate = float(np.clip(estimate, -cap, cap))
                errors.append(abs(history[j] - estimate))
            if errors:
                complexity = {"median": 0, "persistence": 1, "ewma": 2, "ar1": 3}[kind]
                smoothness = -parameter if kind in ("median", "ar1") else parameter
                candidates.append((float(np.mean(errors)), complexity, smoothness, cap,
                                   kind, parameter))
    if not candidates:
        return "median", 6.0, 300.0, float("nan")
    candidates.sort()
    mae, _, _, cap, kind, parameter = candidates[0]
    return kind, float(parameter), float(cap), mae


def evaluate_day(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.sort_values("timestamp_utc").copy().reset_index(drop=True)
    g["timestamp_utc"] = pd.to_datetime(g.timestamp_utc, utc=True, errors="coerce")
    errors: list[float] = []
    corrections, predictions, chosen, parameters, caps, inner_scores = [], [], [], [], [], []
    prior_counts = []
    selected = ("median", 6.0, 300.0, float("nan"))
    last_tune_n = -TUNE_EVERY
    previous_time = None
    for i, row in enumerate(g.itertuples(index=False)):
        now = row.timestamp_utc
        if previous_time is not None and (now - previous_time).total_seconds() > GAP_RESET_S:
            errors = []
            selected = ("median", 6.0, 300.0, float("nan"))
            last_tune_n = -TUNE_EVERY
        previous_time = now
        n = len(errors)
        prior_counts.append(n)
        if n >= MIN_HISTORY and n - last_tune_n >= TUNE_EVERY:
            selected = tune(errors)
            last_tune_n = n
        kind, parameter, cap, inner_mae = selected
        correction = forecast(np.asarray(errors), kind, parameter) if n >= MIN_HISTORY else 0.0
        correction = float(np.clip(correction, -cap, cap))
        corrections.append(correction)
        predictions.append(float(row.predicted_w) + correction)
        chosen.append(kind)
        parameters.append(parameter)
        caps.append(cap)
        inner_scores.append(inner_mae)
        # One-step target enters the state only after the prediction is fixed.
        errors.append(float(row.measured_w) - float(row.predicted_w))
    g["residual_prediction_w"] = predictions
    g["residual_correction_w"] = corrections
    g["residual_error_w"] = g.measured_w - g.residual_prediction_w
    g["selected_filter"] = chosen
    g["selected_filter_parameter"] = parameters
    g["selected_correction_cap_w"] = caps
    g["prior_inner_mae_w"] = inner_scores
    g["prior_samples"] = prior_counts
    return g


def metric(g: pd.DataFrame, pred: str) -> dict:
    e = g.measured_w.to_numpy(float) - g[pred].to_numpy(float)
    return {"n": int(len(e)), "mae_w": float(np.mean(np.abs(e))),
            "rmse_w": float(np.sqrt(np.mean(e**2))),
            "p90_abs_error_w": float(np.quantile(np.abs(e), .9)),
            "bias_w": float(np.mean(e))}


def main() -> None:
    source = pd.read_csv(INPUT, low_memory=False)
    outputs = [evaluate_day(g.copy()) for _, g in source.groupby("day", sort=True)]
    result = pd.concat(outputs, ignore_index=True)
    result.to_csv(OUT_ROWS, index=False, encoding="utf-8-sig")
    day_rows = []
    for day, g in result.groupby("day", sort=True):
        warm = g.prior_samples.ge(MIN_HISTORY)
        for model, col in [("physics_baseline", "predicted_w"), ("past_only_5s_filter", "residual_prediction_w")]:
            all_score = metric(g, col)
            warm_score = metric(g.loc[warm], col) if warm.any() else {"mae_w": None, "n": 0}
            day_rows.append({"day": int(day), "model": model, **all_score,
                             "warm_state_n": warm_score["n"], "warm_state_mae_w": warm_score["mae_w"],
                             "baseline_mae_w": metric(g, "predicted_w")["mae_w"]})
    day_scores = pd.DataFrame(day_rows)
    day_scores.to_csv(OUT_DAYS, index=False, encoding="utf-8-sig")
    clean = result[result.day.le(5)].copy()
    warm_clean = clean[clean.prior_samples.ge(MIN_HISTORY)]
    base = metric(clean, "predicted_w")
    fitted = metric(clean, "residual_prediction_w")
    wins = day_scores[(day_scores.day.le(5)) & day_scores.model.eq("past_only_5s_filter")]
    base_day = day_scores[(day_scores.day.le(5)) & day_scores.model.eq("physics_baseline")].set_index("day")
    win_count = int(sum(wins.set_index("day").mae_w < base_day.mae_w))
    warm_win_count = 0
    for day in sorted(clean.day.unique()):
        g = result[(result.day.eq(day)) & result.prior_samples.ge(MIN_HISTORY)]
        if len(g) and metric(g, "residual_prediction_w")["mae_w"] < metric(g, "predicted_w")["mae_w"]:
            warm_win_count += 1
    d6 = result[result.day.eq(6)]
    d6_base = metric(d6, "predicted_w")
    d6_model = metric(d6, "residual_prediction_w")
    tail_guard = fitted["p90_abs_error_w"] <= base["p90_abs_error_w"] and d6_model["p90_abs_error_w"] <= d6_base["p90_abs_error_w"]
    improvement = 100 * (base["mae_w"] - fitted["mae_w"]) / base["mae_w"]
    adopted = improvement >= 10 and win_count >= 4 and warm_win_count >= 4 and tail_guard
    active = clean[(clean.prior_samples.ge(MIN_HISTORY)) & clean.prior_inner_mae_w.notna()]
    gap = {"active_samples": int(len(active)), "mean_prior_inner_mae_w": float(active.prior_inner_mae_w.mean()),
           "mean_next_sample_abs_error_w": float(np.mean(np.abs(active.measured_w - active.residual_prediction_w)))}
    active_config = result[(result.day.le(5)) & result.prior_samples.ge(MIN_HISTORY)]
    choices = {f"{kind}({p:g}),cap={cap:g}W": int(n) for (kind, p, cap), n in
               active_config[["selected_filter", "selected_filter_parameter", "selected_correction_cap_w"]].value_counts().items()}
    summary = {"validation": "past-only nested one-step evaluation; tuning is repeated every 60 completed samples; current 5-second target is never used before its prediction",
               "sample_period_seconds": 5, "retune_period_samples": TUNE_EVERY,
               "minimum_history_samples": MIN_HISTORY, "gap_reset_seconds": GAP_RESET_S,
               "filter_candidates": {"median_samples": [3, 6, 12, 24], "ewma_alpha": [0.05, 0.1, 0.2, 0.35, 0.5, 0.75],
                                      "persistence_phi": [0.25, 0.5, 0.75, 1.0], "ar1_lookback_samples": [60, 120, 300],
                                      "correction_caps_w": CAPS_W},
               "clean_days_1_5": {"baseline": base, "past_only_filter": fitted,
                                  "improvement_pct": improvement, "day_wins": win_count,
                                  "warm_state_day_wins": warm_win_count,
                                  "warm_state_baseline": metric(warm_clean, "predicted_w"),
                                  "warm_state_filter": metric(warm_clean, "residual_prediction_w")},
               "day6_stress": {"baseline": d6_base, "past_only_filter": d6_model},
               "inner_to_next_sample_gap": gap, "selected_configuration_counts": choices,
               "day_metrics": day_rows, "adopted_to_shadow": bool(adopted), "production_ready": False,
               "interpretation": "This predicts only the next 5-second residual using prior telemetry. It is not a multi-step MPC forecast; residual persistence was observed to decay strongly with horizon.",
               "outputs": [str(OUT_ROWS), str(OUT_DAYS), str(OUT_SUMMARY)]}
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# BWSC2025 past-only 5-second residual prediction", "",
             "At each row, the predictor uses only earlier 5-second telemetry. It chooses among rolling median, EWMA, persistence and AR(1) filters every 60 completed samples, scoring candidate choices on the preceding 60 one-step forecasts. The current measured power enters the state only after its prediction is recorded. State resets after a gap over ten minutes.", "",
             "## Results", "",
             "| Data | Baseline MAE (W) | Residual filter MAE (W) | Improvement | P90 baseline → filter (W) |",
             "|---|---:|---:|---:|---:|"]
    lines.append(f"| Day1–5 all rows (n={base['n']}) | {base['mae_w']:.1f} | {fitted['mae_w']:.1f} | {improvement:+.1f}% | {base['p90_abs_error_w']:.1f} → {fitted['p90_abs_error_w']:.1f} |")
    lines.append(f"| Day6 stress (n={d6_base['n']}) | {d6_base['mae_w']:.1f} | {d6_model['mae_w']:.1f} | — | {d6_base['p90_abs_error_w']:.1f} → {d6_model['p90_abs_error_w']:.1f} |")
    lines += ["", f"Day wins: {win_count}/5; warm-state wins: {warm_win_count}/5. Prior inner MAE averaged {gap['mean_prior_inner_mae_w']:.1f} W; next untouched-sample MAE averaged {gap['mean_next_sample_abs_error_w']:.1f} W over {gap['active_samples']} scored updates.",
              f"Selected settings: `{choices}`.", "",
              "The 5-second residual autocorrelation is useful for receding one-step correction but falls toward zero over about one minute. This result cannot be interpreted as a long-horizon MPC accuracy claim. The project shadow gate also requires no pooled clean-day or Day6 P90 worsening; passing it does not establish BWSC2027 transfer or meet the 23.8 W speed-decision threshold.",
              "", f"Outputs: `{OUT_ROWS.relative_to(ROOT)}`, `{OUT_DAYS.relative_to(ROOT)}`, `{OUT_SUMMARY.relative_to(ROOT)}`.", ""]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    if adopted:
        OUT_MODEL.write_text(json.dumps({"model_type": "past_only_selected_5s_residual_filter",
                                         "stage": "shadow_candidate_not_connected_to_replay_grafana_or_commands",
                                         "production_ready": False, "clean_days_1_5": summary["clean_days_1_5"],
                                         "day6_stress": summary["day6_stress"],
                                         "selected_configuration_counts": choices,
                                         "validation": summary["validation"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"clean_days_1_5": summary["clean_days_1_5"], "day6_stress": summary["day6_stress"],
                      "inner_to_next_sample_gap": gap, "selected_configuration_counts": choices,
                      "adopted_to_shadow": adopted, "outputs": summary["outputs"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
