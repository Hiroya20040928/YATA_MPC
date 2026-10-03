#!/usr/bin/env python3
"""Check whether a fixed cross-day GP and same-day bias adaptation complement each other."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_crossday_residual_models as crossday  # noqa: E402
import fit_vehicle_power_model as vehicle_features  # noqa: E402

IN = ROOT / "reports/residual_crossday_oof.csv"
OUT = ROOT / "reports/residual_gp_online_hybrid.csv"
SUMMARY = ROOT / "reports/residual_gp_online_hybrid_summary.json"
BLOCKS = ROOT / "reports/residual_gp_online_hybrid_block_oof.csv"
ROWS_IN = ROOT / "data/processed/full_race_residual_corrected_5s.csv"
ROWS_OUT = ROOT / "data/processed/full_race_residual_gp_online_hybrid_5s.csv"
MODEL_OUT = ROOT / "models/vehicle_residual_hybrid_candidate.json"
CAP = 100.0


def add_online_error_correction(frame: pd.DataFrame, gp_col: str) -> pd.DataFrame:
    rows = []
    for day, group in frame.groupby("day", sort=True):
        g = group.sort_values("block_start_utc").copy().reset_index(drop=True)
        history: list[tuple[pd.Timestamp, float]] = []
        corrections = []
        prior_counts = []
        for row in g.itertuples(index=False):
            now = row.block_start_utc
            if history and (now - history[-1][0]).total_seconds() > 600:
                history = []
            prior_counts.append(len(history))
            online = float(np.median([e for _, e in history[-6:]])) if len(history) >= 6 else 0.0
            corrections.append(online)
            # Residual after the cross-day GP is observable once this block is complete.
            innovation = float(row.residual_w - getattr(row, gp_col))
            history.append((now, innovation))
        g["gp_online_bias_w"] = corrections
        g["prior_blocks"] = prior_counts
        g["hybrid_total_correction_w"] = np.clip(g[gp_col] + g.gp_online_bias_w, -CAP, CAP)
        g["hybrid_prediction_w"] = g.predicted_w + g.hybrid_total_correction_w
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def metric(y: np.ndarray, p: np.ndarray) -> dict:
    e = y - p
    return {"n": int(len(e)), "mae_w": float(np.mean(np.abs(e))),
            "p90_abs_error_w": float(np.quantile(np.abs(e), .9)), "bias_w": float(e.mean())}


def main() -> None:
    oof = pd.read_csv(IN, parse_dates=["block_start_utc"])
    comparisons = []
    details = []
    for noise in [60, 100]:
        gp_col = f"gpr_rbf_noise{noise}_correction_w"
        pred_col = f"gpr_rbf_noise{noise}_prediction_w"
        h = add_online_error_correction(oof, gp_col)
        for day, g in h.groupby("day", sort=True):
            warm = g.prior_blocks.ge(6)
            for method, pcol, selection in [
                ("physics_baseline", "predicted_w", g),
                (f"gp_noise{noise}", pred_col, g),
                (f"gp_noise{noise}_plus_median6", "hybrid_prediction_w", g),
            ]:
                m = metric(selection.measured_w.to_numpy(float), selection[pcol].to_numpy(float))
                mw = metric(g.loc[warm, "measured_w"].to_numpy(float), g.loc[warm, pcol].to_numpy(float)) if warm.any() else None
                comparisons.append({"gp_noise_sd_w": noise, "day": int(day), "method": method,
                                    "n_blocks": m["n"], "mae_w": m["mae_w"], "p90_abs_error_w": m["p90_abs_error_w"],
                                    "bias_w": m["bias_w"], "warmup_blocks": int((~warm).sum()),
                                    "warm_state_n": mw["n"] if mw else 0,
                                    "warm_state_mae_w": mw["mae_w"] if mw else None})
        h["gp_noise_sd_w"] = noise
        details.append(h)
    all_detail = pd.concat(details, ignore_index=True)
    day_scores = pd.DataFrame(comparisons)
    day_scores.to_csv(OUT, index=False, encoding="utf-8-sig")

    # Day 6 is an anomaly stress evaluation only; it never tunes either method.
    data_cols = ["day", "timestamp_utc", "timestamp_local", "event_phase", "event_flags", "Solar_Watt[W]",
                 "Battery_Watt[W]", "Speed[kmh]", "distance_from_darwin_m", "distance_anchor_confidence",
                 "route_elevation_dem_m", "route_grade_smoothed_pct", "temperature_2m", "surface_pressure",
                 "wind_speed_10m", "wind_direction_10m"]
    d = pd.read_csv(crossday.RAW, usecols=lambda c: c in data_cols, low_memory=False)
    d["timestamp_utc"] = pd.to_datetime(d.timestamp_utc, utc=True, errors="coerce")
    d = vehicle_features.make_features(d)
    d["measured_w"] = pd.to_numeric(d["Solar_Watt[W]"], errors="coerce") + pd.to_numeric(d["Battery_Watt[W]"], errors="coerce")
    d["roll_basis_w"] = vehicle_features.MASS_NOMINAL_KG * vehicle_features.G * vehicle_features.CRR_NOMINAL * np.cos(np.arctan(d.grade_fraction)) * d.speed_mps
    d["grade_basis_w"] = vehicle_features.MASS_NOMINAL_KG * vehicle_features.G * np.sin(np.arctan(d.grade_fraction)) * d.speed_mps
    segments = pd.read_csv(crossday.SEGMENTS)
    mask = pd.read_csv(crossday.REPLAY, usecols=["day", "timestamp_utc", "road_model_comparison_eligible"])
    mask["timestamp_utc"] = pd.to_datetime(mask.timestamp_utc, utc=True, errors="coerce")
    mask["road_model_comparison_eligible"] = mask.road_model_comparison_eligible.astype(str).str.lower().eq("true")
    test6 = crossday.fit_predict_blocks(d, segments, mask, 6, [1, 2, 3, 4, 5])
    labels6 = [crossday.fit_predict_blocks(d, segments, mask, label_day,
                                           [day for day in [1, 2, 3, 4, 5] if day != label_day])
               for label_day in [1, 2, 3, 4, 5]]
    train6 = pd.concat(labels6, ignore_index=True)
    day6_details = []
    for noise in [60, 100]:
        correction, sd = crossday.fit_gp(train6, test6, float(noise))
        test6 = test6.copy()
        test6[f"gpr_rbf_noise{noise}_correction_w"] = correction
        test6[f"gpr_rbf_noise{noise}_prediction_w"] = test6.predicted_w + correction
        test6[f"gpr_rbf_noise{noise}_predictive_sd_w"] = sd
        h6 = add_online_error_correction(test6, f"gpr_rbf_noise{noise}_correction_w")
        h6["gp_noise_sd_w"] = noise
        day6_details.append(h6)
        for method, pcol in [("physics_baseline", "predicted_w"), (f"gp_noise{noise}", f"gpr_rbf_noise{noise}_prediction_w"),
                              (f"gp_noise{noise}_plus_median6", "hybrid_prediction_w")]:
            m = metric(h6.measured_w.to_numpy(float), h6[pcol].to_numpy(float))
            comparisons.append({"gp_noise_sd_w": noise, "day": 6, "method": method,
                                "n_blocks": m["n"], "mae_w": m["mae_w"], "p90_abs_error_w": m["p90_abs_error_w"],
                                "bias_w": m["bias_w"], "warmup_blocks": int(h6.prior_blocks.lt(6).sum()),
                                "warm_state_n": int(h6.prior_blocks.ge(6).sum()),
                                "warm_state_mae_w": metric(h6.loc[h6.prior_blocks.ge(6), "measured_w"].to_numpy(float),
                                                             h6.loc[h6.prior_blocks.ge(6), pcol].to_numpy(float))["mae_w"]})
    day_scores = pd.DataFrame(comparisons)
    day_scores.to_csv(OUT, index=False, encoding="utf-8-sig")
    detail_all = pd.concat([all_detail, *day6_details], ignore_index=True)
    detail_all.to_csv(BLOCKS, index=False, encoding="utf-8-sig")
    clean = day_scores[day_scores.day.le(5)]
    summary = []
    for noise in [60, 100]:
        s = clean[clean.gp_noise_sd_w.eq(noise)]
        base = s[s.method.eq("physics_baseline")].set_index("day")
        for method in [f"gp_noise{noise}", f"gp_noise{noise}_plus_median6"]:
            comp = s[s.method.eq(method)].set_index("day")
            wins = int((comp.mae_w < base.mae_w).sum())
            summary.append({"method": method, "pooled_mae_w": float(np.average(comp.mae_w, weights=comp.n_blocks)),
                            "macro_day_mae_w": float(comp.mae_w.mean()), "days_won_vs_physics": wins,
                            "baseline_pooled_mae_w": float(np.average(base.mae_w, weights=base.n_blocks)),
                            "improvement_pct": 100 * (np.average(base.mae_w, weights=base.n_blocks)
                                                       - np.average(comp.mae_w, weights=comp.n_blocks))
                            / np.average(base.mae_w, weights=base.n_blocks)})
    # Apply the conservative GP-noise=100 candidate to the corresponding outer-fold
    # base predictions at the original eligible 5-second rows.
    block_lookup = detail_all[detail_all.gp_noise_sd_w.eq(100)][
        ["day", "block_start_utc", "hybrid_total_correction_w", "prior_blocks"]].copy()
    raw_rows = pd.read_csv(ROWS_IN, low_memory=False)
    raw_rows["timestamp_utc"] = pd.to_datetime(raw_rows.timestamp_utc, utc=True, errors="coerce")
    raw_rows["block_start_utc"] = raw_rows.timestamp_utc.dt.floor("5min")
    raw_rows = raw_rows.merge(block_lookup, on=["day", "block_start_utc"], how="left", validate="many_to_one")
    raw_rows["hybrid_prediction_w"] = raw_rows.predicted_w + raw_rows.hybrid_total_correction_w
    raw_rows["hybrid_error_w"] = raw_rows.measured_w - raw_rows.hybrid_prediction_w
    raw_rows.to_csv(ROWS_OUT, index=False, encoding="utf-8-sig")
    row_metrics = {}
    for name, cond in [("clean_days_1_5", raw_rows.day.le(5)), ("day6_anomaly_stress", raw_rows.day.eq(6))]:
        scored = raw_rows.loc[cond].dropna(subset=["hybrid_prediction_w"])
        baseline_error = scored.measured_w - scored.predicted_w
        corrected_error = scored.hybrid_error_w
        row_metrics[name] = {"n_rows": int(len(scored)),
                             "baseline_mae_w": float(np.mean(np.abs(baseline_error))),
                             "hybrid_mae_w": float(np.mean(np.abs(corrected_error))),
                             "baseline_p90_abs_error_w": float(np.quantile(np.abs(baseline_error), .9)),
                             "hybrid_p90_abs_error_w": float(np.quantile(np.abs(corrected_error), .9)),
                             "baseline_bias_w": float(np.mean(baseline_error)),
                             "hybrid_bias_w": float(np.mean(corrected_error))}
    # The best exploratory hybrid is only 0.4 W below the simpler online median6
    # result and was one of several variants tried; do not promote it as validated.
    chosen = next(x for x in summary if x["method"] == "gp_noise100_plus_median6")
    payload = {"validation": "outer clean-day holdout on previously cross-fitted GP predictions; online median6 corrects only prior same-day GP innovations; day6 is a fixed anomaly stress set",
               "clean_results": summary,
               "day6_stress": day_scores[day_scores.day.eq(6)].to_dict(orient="records"),
               "row_level_5s_results_conservative_noise100": row_metrics,
               "comparison_to_simpler_online_median6": {"hybrid_noise100_mae_w": chosen["pooled_mae_w"],
                                                         "simple_online_median6_mae_w": 57.06502184906484,
                                                         "incremental_gain_w": 57.06502184906484 - chosen["pooled_mae_w"]},
               "adopted": False,
               "interpretation": "The conservative fixed GP plus online residual-median variant passes the preliminary MAE/day-win gate versus physics, but improves the simpler online median6 model by less than 1 W. Multiple variants were explored on only five clean day folds, so selection bias and uncertainty are substantial. Keep this as a research candidate; do not replace the simpler shadow rule or connect to MPC/Grafana until an independent race or a pre-registered holdout confirms it and speed-decision resolution is met."}
    SUMMARY.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    MODEL_OUT.write_text(json.dumps({"model_type": "fixed RBF residual GP plus causal same-day median6 innovation bias",
                                     "stage": "research_candidate_not_connected_to_replay_grafana_or_commands",
                                     "production_ready": False, "gp_signal_sd_w": 60.0,
                                     "gp_noise_sd_w": 100.0, "gp_length_scale_standardized": 1.5,
                                     "gp_features": crossday.FEATURES, "online_history_blocks": 6,
                                     "block_minutes": 5, "minimum_eligible_rows_per_block": 20,
                                     "maximum_history_gap_minutes": 10, "total_correction_cap_w": 100,
                                     "validation": payload["comparison_to_simpler_online_median6"],
                                     "selection_caveat": "One of multiple residual-model alternatives explored on the same five clean day folds; outer-CV score is exploratory and selection-biased.",
                                     "deployment_gate_met": False}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
