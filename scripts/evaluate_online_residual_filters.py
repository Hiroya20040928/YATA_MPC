#!/usr/bin/env python3
"""Compare causal filters on the prequential kernel model's past innovations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports/residual_online_kernel_oof.csv"
ROWS_IN = ROOT / "data/processed/full_race_residual_online_kernel_5s.csv"
OUT_BLOCKS = ROOT / "reports/residual_online_filters_oof.csv"
OUT_DAYS = ROOT / "reports/residual_online_filters_by_day.csv"
OUT_SUMMARY = ROOT / "reports/residual_online_filters_summary.json"
OUT_REPORT = ROOT / "reports/residual_online_filters_report.md"
ROWS_OUT = ROOT / "data/processed/full_race_residual_online_filters_5s.csv"
OUT_MODEL = ROOT / "models/vehicle_residual_online_filters_candidate.json"

MIN_HISTORY = 14
INNER_HOLDOUT = 12
GAP_RESET_S = 600.0
CAPS = [30.0, 50.0, 75.0, 100.0, 150.0]
CONFIGS = ([ ("median", w) for w in (3, 6, 9) ]
           + [ ("ewma", a) for a in (0.1, 0.2, 0.35, 0.5, 0.75) ]
           + [ ("theil_sen", w) for w in (6, 12, 24) ])


def filter_offset(innovations: np.ndarray, kind: str, parameter: float) -> float:
    x = np.asarray(innovations, dtype=float)
    if len(x) == 0:
        return 0.0
    if kind == "median":
        return float(np.median(x[-int(parameter):]))
    if kind == "ewma":
        x = x[-24:]
        age = np.arange(len(x) - 1, -1, -1, dtype=float)
        weights = float(parameter) * (1.0 - float(parameter)) ** age
        weights /= weights.sum()
        return float(np.dot(weights, x))
    # Robust one-step trend extrapolation using the Theil-Sen slope.
    x = x[-int(parameter):]
    if len(x) < 3:
        return float(np.median(x))
    t = np.arange(len(x), dtype=float)
    slopes = [(x[j] - x[i]) / (t[j] - t[i])
              for i in range(len(x) - 1) for j in range(i + 1, len(x))]
    slope = float(np.median(slopes))
    intercept = float(np.median(x - slope * t))
    return intercept + slope * len(x)


def select_filter(history: list[dict]) -> tuple[str, float, float, float]:
    """Select filter and correction cap using only inner forecasts in history."""
    start = max(MIN_HISTORY, len(history) - INNER_HOLDOUT)
    scores = []
    for kind, parameter in CONFIGS:
        for cap in CAPS:
            errors = []
            for j in range(start, len(history)):
                prior = np.asarray([h["innovation_w"] for h in history[:j]], dtype=float)
                offset = filter_offset(prior, kind, parameter)
                correction = float(np.clip(history[j]["krr_correction_w"] + offset, -cap, cap))
                errors.append(abs(history[j]["baseline_error_w"] - correction))
            if errors:
                # Prefer the simpler median on ties, then a longer history/smoother EWMA.
                complexity = {"median": 0, "ewma": 1, "theil_sen": 2}[kind]
                smoothness = -parameter if kind == "median" else parameter if kind == "ewma" else -parameter
                scores.append((float(np.mean(errors)), complexity, smoothness, cap,
                               kind, parameter, len(errors)))
    if not scores:
        return "median", 6.0, 100.0, float("nan")
    scores.sort()
    mae, _, _, cap, kind, parameter, _ = scores[0]
    return kind, float(parameter), float(cap), mae


def evaluate_day(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.sort_values("block_start_utc").to_dict(orient="records")
    history: list[dict] = []
    predicted, corrections, kinds, parameters, caps, inner_scores = [], [], [], [], [], []
    prior_counts = []
    previous_time = None
    for row in rows:
        now = row["block_start_utc"]
        if previous_time is not None and (now - previous_time).total_seconds() > GAP_RESET_S:
            history = []
        previous_time = now
        n = len(history)
        prior_counts.append(n)
        kind, parameter, cap, inner_mae = select_filter(history)
        offset = filter_offset(np.asarray([h["innovation_w"] for h in history]), kind, parameter) if n >= MIN_HISTORY else 0.0
        correction = float(np.clip(float(row["krr_correction_w"]) + offset, -cap, cap))
        corrections.append(correction)
        predicted.append(float(row["predicted_w"]) + correction)
        kinds.append(kind)
        parameters.append(parameter)
        caps.append(cap)
        inner_scores.append(inner_mae)
        # Append this block's innovation only after the current prediction is fixed.
        innovation = float(row["baseline_error_w"]) - float(row["krr_correction_w"])
        history.append({"baseline_error_w": float(row["baseline_error_w"]),
                        "krr_correction_w": float(row["krr_correction_w"]),
                        "innovation_w": innovation})
    result = pd.DataFrame(rows)
    result["filter_prediction_w"] = predicted
    result["filter_correction_w"] = corrections
    result["selected_filter"] = kinds
    result["selected_filter_parameter"] = parameters
    result["selected_filter_cap_w"] = caps
    result["filter_inner_mae_w"] = inner_scores
    result["filter_prior_blocks"] = prior_counts
    return result


def score(g: pd.DataFrame, pred_col: str) -> dict:
    error = g.measured_w.to_numpy(float) - g[pred_col].to_numpy(float)
    return {"n_blocks": int(len(error)), "mae_w": float(np.mean(np.abs(error))),
            "p90_abs_error_w": float(np.quantile(np.abs(error), .9)), "bias_w": float(np.mean(error))}


def main() -> None:
    source = pd.read_csv(INPUT, parse_dates=["block_start_utc"])
    result = pd.concat([evaluate_day(g.copy()) for _, g in source.groupby("day", sort=True)], ignore_index=True)
    result.to_csv(OUT_BLOCKS, index=False, encoding="utf-8-sig")
    day_rows = []
    for day, g in result.groupby("day", sort=True):
        warm = g.filter_prior_blocks.ge(MIN_HISTORY)
        for name, col in [("physics_baseline", "predicted_w"),
                          ("prequential_kernel", "krr_prediction_w"),
                          ("adaptive_median", "krr_plus_tuned_bias_prediction_w"),
                          ("adaptive_kalman", "krr_plus_kalman_prediction_w"),
                          ("adaptive_filter_selection", "filter_prediction_w")]:
            metric = score(g, col)
            warm_metric = score(g.loc[warm], col) if warm.any() else {"mae_w": None, "n_blocks": 0}
            day_rows.append({"day": int(day), "model": name, **metric,
                             "warm_state_mae_w": warm_metric["mae_w"], "warm_state_n": warm_metric["n_blocks"],
                             "baseline_mae_all_blocks_w": score(g, "predicted_w")["mae_w"],
                             "baseline_mae_warm_state_w": score(g.loc[warm], "predicted_w")["mae_w"] if warm.any() else None})
    day_scores = pd.DataFrame(day_rows)
    day_scores.to_csv(OUT_DAYS, index=False, encoding="utf-8-sig")
    clean = result[result.day.le(5)].copy()
    baseline = score(clean, "predicted_w")
    methods = []
    for name, col in [("prequential_kernel", "krr_prediction_w"),
                      ("adaptive_median", "krr_plus_tuned_bias_prediction_w"),
                      ("adaptive_kalman", "krr_plus_kalman_prediction_w"),
                      ("adaptive_filter_selection", "filter_prediction_w")]:
        metric = score(clean, col)
        days = day_scores[(day_scores.day.le(5)) & day_scores.model.eq(name)]
        methods.append({"model": name, "pooled_mae_w": metric["mae_w"],
                        "pooled_p90_abs_error_w": metric["p90_abs_error_w"], "pooled_bias_w": metric["bias_w"],
                        "improvement_pct": 100 * (baseline["mae_w"] - metric["mae_w"]) / baseline["mae_w"],
                        "clean_day_wins": int((days.mae_w < days.baseline_mae_all_blocks_w).sum()),
                        "warm_state_day_wins": int((days.warm_state_mae_w < days.baseline_mae_warm_state_w).sum()),
                        "macro_day_mae_w": float(days.mae_w.mean())})
    d6 = day_scores[day_scores.day.eq(6)].set_index("model")
    d6_base_p90 = float(d6.loc["physics_baseline", "p90_abs_error_w"])
    for m in methods:
        m["day6_p90_w"] = float(d6.loc[m["model"], "p90_abs_error_w"])
        m["tail_guard_pass"] = bool(m["pooled_p90_abs_error_w"] <= baseline["p90_abs_error_w"] and m["day6_p90_w"] <= d6_base_p90)
    chosen = result[(result.day.le(5)) & result.filter_prior_blocks.ge(MIN_HISTORY)]
    config_counts = {f"{kind}({parameter:g}),cap={cap:g}W": int(n)
                     for (kind, parameter, cap), n in
                     chosen[["selected_filter", "selected_filter_parameter", "selected_filter_cap_w"]].value_counts().items()}
    active = clean[(clean.filter_prior_blocks.ge(MIN_HISTORY)) & clean.filter_inner_mae_w.notna()]
    tuning_gap = {"active_future_blocks": int(len(active)),
                  "mean_prior_inner_cv_mae_w": float(active.filter_inner_mae_w.mean()),
                  "mean_next_block_abs_error_w": float(np.mean(np.abs(active.measured_w - active.filter_prediction_w)))}
    adopted = any(m["model"] == "adaptive_filter_selection" and m["improvement_pct"] >= 10.0
                  and m["clean_day_wins"] >= 4 and m["warm_state_day_wins"] >= 4 and m["tail_guard_pass"] for m in methods)
    row_data = pd.read_csv(ROWS_IN, low_memory=False)
    row_data["timestamp_utc"] = pd.to_datetime(row_data.timestamp_utc, utc=True, errors="coerce")
    row_data["block_start_utc"] = row_data.timestamp_utc.dt.floor("5min")
    lookup = result[["day", "block_start_utc", "filter_correction_w", "selected_filter", "selected_filter_parameter", "selected_filter_cap_w"]]
    row_data = row_data.merge(lookup, on=["day", "block_start_utc"], how="left", validate="many_to_one")
    row_data["filter_online_prediction_w"] = row_data.predicted_w + row_data.filter_correction_w
    row_data["filter_online_error_w"] = row_data.measured_w - row_data.filter_online_prediction_w
    row_data.to_csv(ROWS_OUT, index=False, encoding="utf-8-sig")
    row_level = {}
    for name, condition in [("clean_days_1_5", row_data.day.le(5)), ("day6_anomaly_stress", row_data.day.eq(6))]:
        scored = row_data.loc[condition].dropna(subset=["filter_online_prediction_w"])
        base_err = scored.measured_w - scored.predicted_w
        filt_err = scored.filter_online_error_w
        row_level[name] = {"n_rows": int(len(scored)), "baseline_mae_w": float(np.mean(np.abs(base_err))),
                           "filter_mae_w": float(np.mean(np.abs(filt_err))),
                           "baseline_p90_abs_error_w": float(np.quantile(np.abs(base_err), .9)),
                           "filter_p90_abs_error_w": float(np.quantile(np.abs(filt_err), .9)),
                           "filter_bias_w": float(np.mean(filt_err))}
    payload = {"validation": "nested prequential block-forward evaluation; filter choice and cap use prior residual one-step errors only; each current block remains untouched until scored",
               "clean_days": [1, 2, 3, 4, 5], "day6_role": "anomaly stress only",
               "filter_candidates": {"median_windows": [3, 6, 9], "ewma_alpha": [0.1, 0.2, 0.35, 0.5, 0.75],
                                      "theil_sen_lookback": [6, 12, 24], "total_correction_cap_w": CAPS},
               "inner_holdout_blocks": INNER_HOLDOUT, "minimum_history_blocks": MIN_HISTORY,
               "models": methods, "selected_configuration_counts": config_counts,
               "inner_to_next_block_generalization": tuning_gap, "day6": d6.reset_index().to_dict(orient="records"),
               "row_level_5s": row_level, "adopted_to_shadow": bool(adopted), "production_ready": False,
               "interpretation": "The outer score tests the entire past-only selection procedure. It does not establish cross-event transfer; all 2025 observations are retrospective hindcasts.",
               "references": ["Kalman (1960), https://doi.org/10.1115/1.3662552",
                             "Lanius and Gather (2010), https://doi.org/10.1016/j.csda.2009.10.009"]}
    OUT_SUMMARY.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# BWSC2025 past-only residual filter comparison", "",
             "## Method", "",
             "At each 5-minute block, the filter family and correction cap are selected from one-step forecasts on earlier completed blocks only. Candidates are rolling medians, exponentially weighted averages, and a Theil–Sen robust local trend. The current observation is appended only after its prediction is recorded. This follows the state-estimation and robust online-filtering method families in Kalman (1960) and Lanius & Gather (2010); the specific candidate grid and clipping limits are engineering choices, not values prescribed by those papers.",
             "", "## Day 1–5 next-block results", "",
             "| Model | MAE (W) | improvement vs physics | P90 (W) | day wins | warm-state wins | Day6 P90 (W) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for m in methods:
        lines.append(f"| {m['model']} | {m['pooled_mae_w']:.1f} | {m['improvement_pct']:+.1f}% | {m['pooled_p90_abs_error_w']:.1f} | {m['clean_day_wins']}/5 | {m['warm_state_day_wins']}/5 | {m['day6_p90_w']:.1f} |")
    lines += ["", f"Selected configurations on warm clean-day updates: `{config_counts}`.",
              f"The filter selector's prior inner MAE averaged {tuning_gap['mean_prior_inner_cv_mae_w']:.1f} W; the next untouched block MAE averaged {tuning_gap['mean_next_block_abs_error_w']:.1f} W over {tuning_gap['active_future_blocks']} blocks.",
              "", f"At the original 5-second grain, clean-day MAE was {row_level['clean_days_1_5']['baseline_mae_w']:.1f} W baseline and {row_level['clean_days_1_5']['filter_mae_w']:.1f} W after filtering; Day6 P90 was {row_level['day6_anomaly_stress']['baseline_p90_abs_error_w']:.1f}→{row_level['day6_anomaly_stress']['filter_p90_abs_error_w']:.1f} W.",
              "", "A shadow candidate must improve pooled clean MAE by at least 10%, win at least four of five days and warm-state days, and avoid worsening pooled-clean and Day6 block P90. This project gate is an engineering criterion, not an external standard. Passing it still does not meet the separate 23.8 W speed-decision threshold or establish BWSC2027 performance.",
              "", "References: [Kalman (1960)](https://doi.org/10.1115/1.3662552); [Lanius & Gather (2010)](https://doi.org/10.1016/j.csda.2009.10.009).", "",
              f"Outputs: `{OUT_BLOCKS.relative_to(ROOT)}`, `{OUT_DAYS.relative_to(ROOT)}`, `{OUT_SUMMARY.relative_to(ROOT)}`, `{ROWS_OUT.relative_to(ROOT)}`.", ""]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    if adopted:
        winner = next(m for m in methods if m["model"] == "adaptive_filter_selection")
        OUT_MODEL.write_text(json.dumps({"model_type": "past_only_selected_online_residual_filter",
                                         "stage": "shadow_candidate_not_connected_to_replay_grafana_or_commands",
                                         "production_ready": False, "best_candidate_oof_metrics": winner,
                                         "selected_configuration_counts": config_counts,
                                         "validation_scope": payload["validation"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"baseline": baseline, "models": methods, "selected_configuration_counts": config_counts,
                      "inner_to_next_block_generalization": tuning_gap, "row_level_5s": row_level,
                      "adopted_to_shadow": adopted,
                      "outputs": [str(OUT_BLOCKS), str(OUT_DAYS), str(OUT_SUMMARY), str(OUT_REPORT), str(ROWS_OUT)]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
