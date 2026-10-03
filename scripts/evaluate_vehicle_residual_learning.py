#!/usr/bin/env python3
"""Evaluate causal, bounded residual adaptation on BWSC2025 race telemetry.

The existing physical vehicle model is re-fitted with one race day held out at
a time. This script aggregates eligible rows into 5-minute blocks, then
forecasts each block using only residuals from earlier blocks on the same day.
Day 6 is reported as a stress test and is never used to tune or fit the
correction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/processed/race_days_distance_geo_weather.csv"
REPLAY = ROOT / "data/processed/full_race_component_replay_5s.csv"
SEGMENTS = ROOT / "data/processed/vehicle_energy_segments_10km.csv"
OUT_CV = ROOT / "reports/residual_online_cv.csv"
OUT_BLOCKS = ROOT / "reports/residual_block_oof.csv"
OUT_ROW_REPLAY = ROOT / "data/processed/full_race_residual_corrected_5s.csv"
OUT_SUMMARY = ROOT / "reports/residual_online_summary.json"
OUT_REPORT = ROOT / "reports/residual_learning_report.md"
OUT_MODEL = ROOT / "models/vehicle_residual_online.json"
BLOCK = "5min"
MIN_ROWS_PER_BLOCK = 20
MAX_HISTORY_GAP_MIN = 10.0
BIAS_CAP_W = 100.0
RIDGE_ALPHA = 10.0
RIDGE_WINDOW_BLOCKS = 24
RIDGE_MIN_HISTORY = 12
sys.path.insert(0, str(ROOT / "scripts"))
import fit_vehicle_power_model as vehicle_features  # noqa: E402
import fit_vehicle_energy_segments as segment_fit  # noqa: E402


def score(frame: pd.DataFrame, pred_col: str) -> dict:
    err = frame.measured_w.to_numpy(float) - frame[pred_col].to_numpy(float)
    return {
        "n_blocks": int(len(err)),
        "mae_w": float(np.mean(np.abs(err))) if len(err) else None,
        "rmse_w": float(np.sqrt(np.mean(err**2))) if len(err) else None,
        "p90_abs_error_w": float(np.quantile(np.abs(err), 0.9)) if len(err) else None,
        "bias_measured_minus_model_w": float(np.mean(err)) if len(err) else None,
    }


def evaluate_candidate(blocks: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows: list[dict] = []
    for day, group in blocks.groupby("day", sort=True):
        g = group.sort_values("block_start_utc").copy().reset_index(drop=True)
        history: list[tuple[pd.Timestamp, float, np.ndarray]] = []
        corrections: list[float] = []
        prior_counts: list[int] = []
        for row in g.itertuples(index=False):
            now = row.block_start_utc
            if history and (now - history[-1][0]).total_seconds() > MAX_HISTORY_GAP_MIN * 60.0:
                history = []
            prior_counts.append(len(history))
            corr = 0.0
            if mode == "last_error" and history:
                corr = float(np.clip(history[-1][1], -BIAS_CAP_W, BIAS_CAP_W))
            elif mode == "median_prev3" and len(history) >= 3:
                corr = float(np.clip(np.median([e for _, e, _ in history[-3:]]),
                                     -BIAS_CAP_W, BIAS_CAP_W))
            elif mode == "median_prev6" and len(history) >= 6:
                corr = float(np.clip(np.median([e for _, e, _ in history[-6:]]),
                                     -BIAS_CAP_W, BIAS_CAP_W))
            elif mode in {"median_prev9", "median_prev12"}:
                window = int(mode.rsplit("prev", 1)[1])
                if len(history) >= window:
                    corr = float(np.clip(np.median([e for _, e, _ in history[-window:]]),
                                         -BIAS_CAP_W, BIAS_CAP_W))
            elif mode in {"ewma_alpha20", "ewma_alpha40"} and len(history) >= 3:
                alpha = 0.2 if mode == "ewma_alpha20" else 0.4
                # Rebuild from the completed history so a gap reset is naturally honored.
                smoothed = history[0][1]
                for _, past_error, _ in history[1:]:
                    smoothed = alpha * past_error + (1.0 - alpha) * smoothed
                corr = float(np.clip(smoothed, -BIAS_CAP_W, BIAS_CAP_W))
            elif mode == "ridge_prev12" and len(history) >= RIDGE_MIN_HISTORY:
                past = history[-RIDGE_WINDOW_BLOCKS:]
                x_hist = np.vstack([x for _, _, x in past])
                y_hist = np.asarray([e for _, e, _ in past], dtype=float)
                x_mean = x_hist.mean(axis=0)
                x_std = x_hist.std(axis=0)
                x_std[x_std < 1e-8] = 1.0
                x_scaled = (x_hist - x_mean) / x_std
                x_now = (row.residual_features - x_mean) / x_std
                y_mean = float(y_hist.mean())
                beta = np.linalg.solve(x_scaled.T @ x_scaled + RIDGE_ALPHA * np.eye(x_scaled.shape[1]),
                                       x_scaled.T @ (y_hist - y_mean))
                corr = float(np.clip(y_mean + x_now @ beta, -BIAS_CAP_W, BIAS_CAP_W))
            corrections.append(corr)
            history.append((now, float(row.baseline_error_w), row.residual_features))
        g["online_correction_w"] = corrections
        g["prior_blocks"] = prior_counts
        g["online_predicted_w"] = g.predicted_w + g.online_correction_w
        g["baseline_error_w"] = g.measured_w - g.predicted_w
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else blocks.assign(
        online_correction_w=[], online_predicted_w=[])


def main() -> None:
    raw_cols = ["day", "timestamp_utc", "timestamp_local", "event_phase", "event_flags",
                "Solar_Watt[W]", "Battery_Watt[W]", "Speed[kmh]", "distance_from_darwin_m",
                "distance_anchor_confidence", "route_elevation_dem_m", "route_grade_smoothed_pct",
                "temperature_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
    d = pd.read_csv(RAW, usecols=lambda c: c in raw_cols, low_memory=False)
    d["timestamp_utc"] = pd.to_datetime(d.timestamp_utc, utc=True, errors="coerce")
    d = vehicle_features.make_features(d)
    d["measured_w"] = (pd.to_numeric(d["Solar_Watt[W]"], errors="coerce")
                       + pd.to_numeric(d["Battery_Watt[W]"], errors="coerce"))
    replay_mask = pd.read_csv(REPLAY, usecols=["day", "timestamp_utc", "road_model_comparison_eligible"])
    replay_mask["timestamp_utc"] = pd.to_datetime(replay_mask.timestamp_utc, utc=True, errors="coerce")
    replay_mask["road_model_comparison_eligible"] = replay_mask.road_model_comparison_eligible.astype(str).str.lower().eq("true")
    d = d.merge(replay_mask, on=["day", "timestamp_utc"], how="left", validate="one_to_one")
    segments = pd.read_csv(SEGMENTS)
    d["predicted_w"] = np.nan
    fold_metadata = {}
    for day in sorted(int(v) for v in d.day.dropna().unique() if 1 <= int(v) <= 6):
        train_days = [v for v in (1, 2, 3, 4, 5) if v != day] if day <= 5 else [1, 2, 3, 4, 5]
        train_segments = segments[segments.day.isin(train_days)].copy()
        stop = d[(d.day.isin(train_days)) & d["Speed[kmh]"].eq(0)
                 & d.event_phase.isin(["road_day", "charge"]) & d.measured_w.between(0, 250)].measured_w
        p_aux = float(stop.median())
        train_segments.attrs["p_aux_w"] = p_aux
        fitted = segment_fit.fit_mle(train_segments)
        p = fitted["parameters"]
        idx = d.day.eq(day)
        d.loc[idx, "predicted_w"] = (
            p["p_aux_w"] + p["beta_roll"] * d.loc[idx, "roll_basis_w"]
            + p["beta_aero"] * d.loc[idx, "aero_basis_w"]
            + p["beta_grade"] * d.loc[idx, "grade_basis_w"])
        fold_metadata[str(day)] = {"train_days": train_days, "train_stop_rows": int(len(stop)),
                                   "fold_p_aux_w": p_aux, "coefficients": p}
    d["error_w"] = d.measured_w - d.predicted_w
    eligible = (d.road_model_comparison_eligible.fillna(False)
                & d.day.between(1, 6)
                & d[["timestamp_utc", "measured_w", "predicted_w", "error_w"]].notna().all(axis=1))
    d = d.loc[eligible].copy()
    d["block_start_utc"] = d.timestamp_utc.dt.floor(BLOCK)
    blocks = (d.groupby(["day", "block_start_utc"], as_index=False)
              .agg(measured_w=("measured_w", "mean"),
                   predicted_w=("predicted_w", "mean"),
                   baseline_error_w=("error_w", "mean"),
                   eligible_rows=("error_w", "size"),
                   speed_kmh=("Speed[kmh]", "mean"),
                   relative_air_speed_mps=("relative_air_speed_mps", "mean"),
                   grade_pct=("route_grade_smoothed_pct", "mean"),
                   temperature_c=("temperature_2m", "mean"),
                   air_density_kgm3=("air_density_kgm3", "mean"),
                   distance_m=("distance_from_darwin_m", "mean")))
    feature_cols = ["speed_kmh", "relative_air_speed_mps", "grade_pct", "temperature_c", "air_density_kgm3"]
    blocks["residual_features"] = list(blocks[feature_cols].to_numpy(float))
    blocks = blocks.loc[blocks.eligible_rows.ge(MIN_ROWS_PER_BLOCK)].copy()

    # Select/check rules only on clean days. Day 6 is an anomaly stress set.
    clean = blocks[blocks.day.le(5)].copy()
    d6 = blocks[blocks.day.eq(6)].copy()
    # Pre-declared robust-memory alternatives test whether a different adaptation
    # time constant improves held-out days without introducing new physical inputs.
    modes = ["last_error", "median_prev3", "median_prev6", "median_prev9",
             "median_prev12", "ewma_alpha20", "ewma_alpha40", "ridge_prev12"]
    clean_rows = []
    day_rows = []
    predictions: dict[str, pd.DataFrame] = {}
    for mode in modes:
        adapted = evaluate_candidate(clean, mode)
        predictions[mode] = adapted
        for day, g in adapted.groupby("day", sort=True):
            scored = g.copy()
            # Fair paired comparison: score both models on the same blocks where
            # the adaptive model has enough prior history to issue a correction.
            # Zero-valued estimates after warm-up are valid estimates, so also
            # track the first eligible block by its prior-block count below.
            min_prior = {"last_error": 1, "median_prev3": 3, "median_prev6": 6,
                         "median_prev9": 9, "median_prev12": 12,
                         "ewma_alpha20": 3, "ewma_alpha40": 3,
                         "ridge_prev12": RIDGE_MIN_HISTORY}[mode]
            active = scored.prior_blocks.ge(min_prior)
            paired = scored.loc[active]
            clean_rows.append({"model": mode, "scope": "clean_days_1_5_all_blocks",
                               **score(scored, "predicted_w"),
                               "adapted_mae_w": score(scored, "online_predicted_w")["mae_w"],
                               "adapted_rmse_w": score(scored, "online_predicted_w")["rmse_w"],
                               "correction_blocks": int(active.sum()),
                               "warmup_excluded_paired_baseline_mae_w": score(paired, "predicted_w")["mae_w"],
                               "warmup_excluded_adapted_mae_w": score(paired, "online_predicted_w")["mae_w"]})
            day_rows.append({"day": int(day), "model": mode,
                             "blocks": int(len(scored)),
                             "correction_blocks": int(active.sum()),
                             "all_block_baseline_mae_w": score(scored, "predicted_w")["mae_w"],
                             "all_block_adapted_mae_w": score(scored, "online_predicted_w")["mae_w"],
                             "all_block_baseline_p90_abs_error_w": score(scored, "predicted_w")["p90_abs_error_w"],
                             "all_block_adapted_p90_abs_error_w": score(scored, "online_predicted_w")["p90_abs_error_w"],
                             "paired_baseline_mae_w": score(paired, "predicted_w")["mae_w"],
                             "paired_adapted_mae_w": score(paired, "online_predicted_w")["mae_w"],
                             "baseline_bias_w": score(scored, "predicted_w")["bias_measured_minus_model_w"],
                             "adapted_bias_w": score(scored, "online_predicted_w")["bias_measured_minus_model_w"]})

    # Day 6 is never used for selection or correction-rule fitting. Apply the
    # already-fixed rules only as an anomaly-response stress diagnostic.
    day6_rows = []
    for mode in modes:
        adapted6 = evaluate_candidate(d6, mode) if not d6.empty else d6.copy()
        for day, g in adapted6.groupby("day", sort=True):
            day6_rows.append({"day": int(day), "model": mode, "blocks": int(len(g)),
                              "baseline_mae_w": score(g, "predicted_w")["mae_w"],
                              "adapted_mae_w": score(g, "online_predicted_w")["mae_w"],
                              "baseline_p90_abs_error_w": score(g, "predicted_w")["p90_abs_error_w"],
                              "adapted_p90_abs_error_w": score(g, "online_predicted_w")["p90_abs_error_w"],
                              "baseline_bias_w": score(g, "predicted_w")["bias_measured_minus_model_w"],
                              "adapted_bias_w": score(g, "online_predicted_w")["bias_measured_minus_model_w"]})

    summary_frame = pd.DataFrame(clean_rows)
    day_frame = pd.DataFrame(day_rows)
    out = day_frame.merge(pd.DataFrame(day6_rows), how="outer", on=["day", "model"], suffixes=("_clean", "_d6"))
    out.to_csv(OUT_CV, index=False, encoding="utf-8-sig")
    candidate_summary = []
    for mode, g in day_frame.groupby("model", sort=False):
        n_all = int(g.blocks.sum())
        n_paired = int(g.correction_blocks.sum())
        candidate_summary.append({
            "model": mode,
            "blocks": n_all,
            "baseline_mae_w": float(np.average(g.all_block_baseline_mae_w, weights=g.blocks)),
            "adapted_mae_w": float(np.average(g.all_block_adapted_mae_w, weights=g.blocks)),
            "paired_baseline_mae_w": float(np.average(g.paired_baseline_mae_w, weights=g.correction_blocks)),
            "paired_adapted_mae_w": float(np.average(g.paired_adapted_mae_w, weights=g.correction_blocks)),
            "paired_day_wins": int((g.paired_adapted_mae_w < g.paired_baseline_mae_w).sum()),
            "paired_blocks": n_paired,
        })
    best = min(candidate_summary, key=lambda r: r["adapted_mae_w"])
    # Nested day selection: for each outer test day, pick the correction rule
    # using only the other four days, then score it on the untouched test day.
    nested_rows = []
    clean_days = sorted(int(v) for v in clean.day.unique())
    for test_day in clean_days:
        selection_train = day_frame[day_frame.day.ne(test_day)]
        train_scores = selection_train.groupby("model").all_block_adapted_mae_w.mean()
        selected_rule = str(train_scores.idxmin())
        test_row = day_frame[(day_frame.day.eq(test_day)) & day_frame.model.eq(selected_rule)].iloc[0]
        nested_rows.append({"held_out_day": test_day, "selected_rule_from_other_days": selected_rule,
                            "blocks": int(test_row.blocks),
                            "baseline_mae_w": float(test_row.all_block_baseline_mae_w),
                            "adapted_mae_w": float(test_row.all_block_adapted_mae_w),
                            "baseline_p90_abs_error_w": float(test_row.all_block_baseline_p90_abs_error_w),
                            "adapted_p90_abs_error_w": float(test_row.all_block_adapted_p90_abs_error_w),
                            "paired_baseline_mae_w": float(test_row.paired_baseline_mae_w),
                            "paired_adapted_mae_w": float(test_row.paired_adapted_mae_w),
                            "correction_blocks": int(test_row.correction_blocks)})
    nested_n = sum(r["blocks"] for r in nested_rows)
    baseline_mae = sum(r["baseline_mae_w"] * r["blocks"] for r in nested_rows) / nested_n
    adapted_mae = sum(r["adapted_mae_w"] * r["blocks"] for r in nested_rows) / nested_n
    improvement_pct = 100.0 * (baseline_mae - adapted_mae) / baseline_mae if baseline_mae else 0.0
    per_day_wins = sum(r["paired_adapted_mae_w"] < r["paired_baseline_mae_w"] for r in nested_rows)
    selected_counts = pd.Series([r["selected_rule_from_other_days"] for r in nested_rows]).value_counts()
    deployment_rule = str(selected_counts.index[0])
    deployable = bool(improvement_pct >= 10.0 and per_day_wins >= 4)
    nested_detail = []
    for row in nested_rows:
        g = predictions[row["selected_rule_from_other_days"]]
        g = g[g.day.eq(row["held_out_day"])].copy()
        g["selected_rule"] = row["selected_rule_from_other_days"]
        g["validation_scope"] = "nested_day_holdout"
        g["corrected_error_w"] = g.measured_w - g.online_predicted_w
        nested_detail.append(g)
    day6_detail = evaluate_candidate(d6, deployment_rule) if not d6.empty else d6.copy()
    if not day6_detail.empty:
        day6_detail["selected_rule"] = deployment_rule
        day6_detail["validation_scope"] = "day6_anomaly_stress_only"
        day6_detail["corrected_error_w"] = day6_detail.measured_w - day6_detail.online_predicted_w
        nested_detail.append(day6_detail)
    detail = pd.concat(nested_detail, ignore_index=True)
    detail_cols = ["day", "block_start_utc", "distance_m", "eligible_rows", "speed_kmh",
                   "relative_air_speed_mps", "grade_pct", "temperature_c", "air_density_kgm3",
                   "measured_w", "predicted_w", "baseline_error_w", "online_correction_w",
                   "online_predicted_w", "corrected_error_w", "prior_blocks", "selected_rule",
                   "validation_scope"]
    detail[detail_cols].to_csv(OUT_BLOCKS, index=False, encoding="utf-8-sig")
    selected_day6_stress = next((r for r in day6_rows if r["model"] == deployment_rule), None)
    nested_scope = detail[detail.validation_scope.eq("nested_day_holdout")]
    nested_p90 = {
        "baseline_abs_error_w": float(np.quantile(np.abs(nested_scope.baseline_error_w.to_numpy(float)), 0.9)),
        "corrected_abs_error_w": float(np.quantile(np.abs(nested_scope.corrected_error_w.to_numpy(float)), 0.9)),
    }
    d["block_start_utc"] = d.timestamp_utc.dt.floor(BLOCK)
    correction_lookup = detail[["day", "block_start_utc", "online_correction_w",
                                "selected_rule", "validation_scope"]]
    full = d.merge(correction_lookup, on=["day", "block_start_utc"], how="left", validate="many_to_one")
    full["corrected_prediction_w"] = full.predicted_w + full.online_correction_w
    full["baseline_error_w"] = full.measured_w - full.predicted_w
    full["corrected_error_w"] = full.measured_w - full.corrected_prediction_w
    eligible_full = full[full.road_model_comparison_eligible.fillna(False)
                         & full.measured_w.notna() & full.predicted_w.notna()]
    row_level_results = {}
    for label, selection in [("clean_days_1_5", eligible_full.day.le(5)),
                             ("day6_anomaly_stress", eligible_full.day.eq(6))]:
        selected_rows = eligible_full.loc[selection]
        grp = selected_rows.dropna(subset=["corrected_prediction_w"])
        row_level_results[label] = {"n_eligible_rows": int(len(selected_rows)),
                                    "n_scored_rows": int(len(grp)),
                                    "baseline_mae_w": float(np.mean(np.abs(grp.baseline_error_w))),
                                    "corrected_mae_w": float(np.mean(np.abs(grp.corrected_error_w))),
                                    "baseline_p90_abs_error_w": float(np.quantile(np.abs(grp.baseline_error_w), 0.9)),
                                    "corrected_p90_abs_error_w": float(np.quantile(np.abs(grp.corrected_error_w), 0.9)),
                                    "baseline_bias_w": float(grp.baseline_error_w.mean()),
                                    "corrected_bias_w": float(grp.corrected_error_w.mean())}
    full_cols = ["day", "timestamp_utc", "timestamp_local", "event_phase", "event_flags",
                 "distance_from_darwin_m", "Speed[kmh]", "distance_anchor_confidence",
                 "road_model_comparison_eligible", "measured_w", "predicted_w", "online_correction_w",
                 "corrected_prediction_w", "baseline_error_w", "corrected_error_w",
                 "selected_rule", "validation_scope"]
    full[full_cols].to_csv(OUT_ROW_REPLAY, index=False, encoding="utf-8-sig")
    payload = {
        "source": str(RAW.relative_to(ROOT)),
        "eligibility_mask_source": str(REPLAY.relative_to(ROOT)),
        "base_model": "The existing 10 km constrained Student-t model is re-fitted for each held-out day. The stationary auxiliary median is recomputed from training days only, excluding the held-out day.",
        "base_folds": fold_metadata,
        "target": "5-minute mean measured HV load minus the existing out-of-day vehicle prediction",
        "validation": "Causal within-day walk-forward. Days 1-5 are clean evaluation days; Day 6 is a fixed anomaly stress set and never tunes the method.",
        "eligibility": "Existing road_model_comparison_eligible flag; each 5-minute UTC block requires at least 20 eligible 5-second rows.",
        "methods": {"last_error": "next-block additive bias equals previous block residual, clipped to ±100 W",
                    "median_prev3": "next-block additive bias equals median of up to 3 prior eligible blocks, clipped to ±100 W; requires 3 prior blocks",
                    "median_prev6": "next-block additive bias equals median of up to 6 prior eligible blocks, clipped to ±100 W; requires 6 prior blocks",
                    "median_prev9": "next-block additive bias equals median of 9 prior eligible blocks, clipped to ±100 W; requires 9 prior blocks",
                    "median_prev12": "next-block additive bias equals median of 12 prior eligible blocks, clipped to ±100 W; requires 12 prior blocks",
                    "ewma_alpha20": "next-block additive bias is EWMA(alpha=0.2) of prior eligible block residuals; requires 3 prior blocks; clipped to ±100 W",
                    "ewma_alpha40": "next-block additive bias is EWMA(alpha=0.4) of prior eligible block residuals; requires 3 prior blocks; clipped to ±100 W",
                    "ridge_prev12": f"ridge regression of prior residuals on speed, relative air speed, route grade, ambient temperature and density; alpha={RIDGE_ALPHA:g}, up to {RIDGE_WINDOW_BLOCKS} prior blocks, minimum {RIDGE_MIN_HISTORY}; output clipped to ±100 W",
                    "reset": f"History resets if the gap between eligible blocks exceeds {MAX_HISTORY_GAP_MIN:g} minutes."},
        "clean_day_results": clean_rows,
        "clean_day_fold_results": day_rows,
        "all_clean_days_candidate_summary_descriptive_only": candidate_summary,
        "nested_outer_day_results": nested_rows,
        "day6_stress_results": day6_rows,
        "row_level_5s_results": row_level_results,
        "best_by_all_clean_days_descriptive_only": best["model"],
        "nested_deployment_rule_consensus": deployment_rule,
        "nested_clean_baseline_mae_w": baseline_mae,
        "nested_clean_adapted_mae_w": adapted_mae,
        "nested_clean_mae_improvement_pct": improvement_pct,
        "nested_clean_paired_day_wins": int(per_day_wins),
        "nested_clean_p90_abs_error_w": nested_p90,
        "selected_for_shadow_candidate": deployable,
        "production_ready": False,
        "adoption_gate": "Nested leave-one-day-out rule selection: at least 10% weighted clean-day MAE improvement and a paired warm-state win on at least 4 of 5 clean days. This is a project engineering gate, not an external standard.",
        "interpretation": "This is a bounded online residual-bias correction, not a recovered physical coefficient or a globally trained black-box map. It may compensate persistent same-day model bias after warm-up only.",
    }
    OUT_SUMMARY.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if deployable:
        OUT_MODEL.write_text(json.dumps({
            "model_type": "causal bounded online residual bias",
            "stage": "shadow_candidate_not_connected_to_replay_or_grafana",
            "selected_rule": deployment_rule,
            "block_minutes": 5,
            "minimum_eligible_rows_per_block": MIN_ROWS_PER_BLOCK,
            "max_history_gap_minutes": MAX_HISTORY_GAP_MIN,
            "bias_cap_w": BIAS_CAP_W,
            "scope": "same-day next 5-minute eligible road-power prediction only; reset after gaps; not integrated with race replay, Grafana, or actuator commands",
            "evidence": {"nested_clean_days_1_5_baseline_mae_w": baseline_mae,
                         "nested_clean_days_1_5_corrected_mae_w": adapted_mae,
                         "improvement_pct": improvement_pct,
                         "paired_day_wins": int(per_day_wins),
                         "nested_clean_p90_abs_error_before_after_w": nested_p90,
                         "row_level_5s_clean_mae_before_after_w": {
                             "baseline": row_level_results["clean_days_1_5"]["baseline_mae_w"],
                             "corrected": row_level_results["clean_days_1_5"]["corrected_mae_w"]},
                         "outer_fold_rule_selections": nested_rows},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    elif OUT_MODEL.exists():
        OUT_MODEL.unlink()

    report_lines = [
        "# BWSC2025: オンライン残差補正の評価",
        "",
        "## 結論",
        "",
        (f"日単位の入れ子交差検証では、各評価日のルールを残り4日だけで選びました。外側評価の5分平均MAEは "
         f"{baseline_mae:.1f} Wから{adapted_mae:.1f} Wへ {improvement_pct:.1f}%改善、絶対誤差P90は "
         f"{nested_p90['baseline_abs_error_w']:.1f} Wから{nested_p90['corrected_abs_error_w']:.1f} Wへ変化し、ウォーム状態の対応比較で {per_day_wins}/5日改善しました。"),
        f"外側foldで最も多く選ばれたルールは `{deployment_rule}` です。全Day1–5で再集計した記述的な候補最小MAEは `{best['model']}` の {best['adapted_mae_w']:.1f} Wですが、同一データで選んだ値なので性能保証値ではありません。",
        ("採用ゲートを満たすため、同日内の次5分区間に限る±100 W制限付き残差補正を、シャドー検証用の暫定候補として保存しました。現在の全レース再生・Grafana・速度指令には接続していません。"
         if deployable else
         "事前に定めた採用ゲートを満たさないため、残差補正は採用せず診断値として保管します。"),
        "この結果はレース当日中の持続的な偏りを追う補正の検証であり、車両の物理係数を真値へコミットした結果ではありません。",
        "実測は単一大会のDay 1–6のみで、Day 6は異常ホールドアウトです。未知大会・異なる経路に対する最高精度や保証性能を、この標本から主張することはできません。",
        "",
        "## 検証設計",
        "",
        "- 元ログ43,963行から既存の`road_model_comparison_eligible`マスクだけを使いました。対象日を除いて走行抵抗モデルを再推定し、停止時補助負荷の中央値も訓練日だけから計算してから、実測負荷・予測・残差を5分UTC区間へ集約しました。20行未満の区間は除外しました。",
        "- 各区間は、その区間より前に完了した同日区間の残差だけから補正します。適格区間の間隔が10分を超えれば履歴をリセットします。",
        "- 比較候補は前区間残差、直近3/6/9/12区間の中央値、EWMA（α=0.2/0.4）、および速度・相対風速・勾配・気温・密度から残差を予測する小規模Ridge回帰です。補正幅はいずれも±100 Wに制限しました。Day 1–5のみで候補を比較し、Day 6は記録済み異常のストレス評価に限定しました。",
        "- 集計誤差の時間相関が残るため、5分区間を独立標本とは見なしません。日別成績とウォームアップ後の対応比較を併記します。",
        "",
        "## 日別比較（最良候補）",
        "",
        "| 評価Day | 他4日で選んだルール | 補正可能区間 | 補正前MAE W | 補正後MAE W |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in nested_rows:
        report_lines.append(f"| {row['held_out_day']} | {row['selected_rule_from_other_days']} | {row['correction_blocks']} | {row['paired_baseline_mae_w']:.1f} | {row['paired_adapted_mae_w']:.1f} |")
    if selected_day6_stress is not None:
        report_lines += ["", "Day 6の異常ストレス確認では、" + deployment_rule +
                         f"の適用でMAEが{selected_day6_stress['baseline_mae_w']:.1f} Wから{selected_day6_stress['adapted_mae_w']:.1f} Wへ下がりましたが、"
                         f"絶対誤差P90は{selected_day6_stress['baseline_p90_abs_error_w']:.1f} Wから{selected_day6_stress['adapted_p90_abs_error_w']:.1f} Wへ変化しました。"
                         "異常があると補正が異常残差を吸収し得るため、異常フラグ時は補正更新を停止する必要があります。これは通常日の精度改善を示す根拠には数えません。"]
    row_clean = row_level_results["clean_days_1_5"]
    row_d6 = row_level_results["day6_anomaly_stress"]
    report_lines += ["", "## 5秒行レベルへの遡及適用", "",
                     f"補正可能な5分区間に属したDay 1–5の{row_clean['n_scored_rows']:,}行ではMAEが{row_clean['baseline_mae_w']:.1f} Wから{row_clean['corrected_mae_w']:.1f} W、"
                     f"P90は{row_clean['baseline_p90_abs_error_w']:.1f} Wから{row_clean['corrected_p90_abs_error_w']:.1f} Wへ変化しました。"
                     f"Day 6の{row_d6['n_scored_rows']:,}行ではMAEが{row_d6['baseline_mae_w']:.1f} Wから{row_d6['corrected_mae_w']:.1f} W、P90が{row_d6['baseline_p90_abs_error_w']:.1f} Wから{row_d6['corrected_p90_abs_error_w']:.1f} Wでした。"
                     "これは実測を含むブロック残差を用いた次区間予測の検証です。5秒の瞬時制御精度が達成されたという意味ではありません。"]
    report_lines += ["", "## 運用上の扱い", "",
                     "- 補正係数は観測済み残差の遅行中央値です。物理的なCdA、Crr、モーター効率、勾配係数に還元・再解釈しません。",
                     "- 初回の履歴不足中は基準モデルを使います。残差が急変したら原因が未確定なため、上限で抑えた補正値と残差品質フラグを表示します。",
                     "- Day 6は異常日なので補正ルールの学習・選択に使いません。Day 6で補正が誤差を下げても、それは異常を隠す可能性があり通常性能の証拠ではありません。",
                     "- 5分集約の改善は、5秒指令精度や未知ルート・BWSC2027への転移を意味しません。最適速度の自動指令は、別途予測帯と速度選択の安定性が検証されるまで保留します。",
                     "",
                     "## 生成物", "",
                     f"- `{OUT_CV.relative_to(ROOT)}`: 日別、補正前後のウォークフォワード比較",
                     f"- `{OUT_BLOCKS.relative_to(ROOT)}`: 全評価区間の補正前後残差と選択fold",
                     f"- `{OUT_ROW_REPLAY.relative_to(ROOT)}`: 全適格5秒行と補正列（短すぎる区間は補正値なし）",
                     f"- `{OUT_SUMMARY.relative_to(ROOT)}`: 評価条件・日別係数・採否・集計指標",
                     (f"- `{OUT_MODEL.relative_to(ROOT)}`: 採用ゲートを満たした場合のみ出力される運用ルール" if deployable else "- 運用モデルは出力していません。"),
                     ""]
    OUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps({"descriptive_best": best["model"], "deployment_rule": deployment_rule,
                      "nested_baseline_mae_w": baseline_mae,
                      "nested_adapted_mae_w": adapted_mae, "improvement_pct": improvement_pct,
                      "paired_day_wins": per_day_wins, "adopted": deployable,
                      "nested_outer_day_results": nested_rows,
                      "clean_day_rows": day_rows, "day6_stress": day6_rows,
                      "outputs": [str(OUT_CV), str(OUT_BLOCKS), str(OUT_ROW_REPLAY), str(OUT_SUMMARY), str(OUT_REPORT),
                                  str(OUT_MODEL) if deployable else None]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
