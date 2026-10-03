#!/usr/bin/env python3
"""Test past-only blending of location-free and route-distance 5-minute models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/residual_online_kernel_oof.csv"
DIST = ROOT / "reports/residual_online_kernel_distance_oof.csv"
ROWS_IN = ROOT / "data/processed/full_race_residual_corrected_5s.csv"
OUT_OOF = ROOT / "reports/residual_online_kernel_distance_blend_oof.csv"
OUT_DAYS = ROOT / "reports/residual_online_kernel_distance_blend_by_day.csv"
OUT_JSON = ROOT / "reports/residual_online_kernel_distance_blend_summary.json"
OUT_REPORT = ROOT / "reports/residual_online_kernel_distance_blend_report.md"
OUT_ROWS = ROOT / "data/processed/full_race_residual_online_kernel_distance_blend_5s.csv"
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]  # weight on distance-enabled model
MIN_HISTORY = 14
HISTORY_WINDOW = 12
GAP_RESET_S = 600.0


def evaluate_day(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("block_start_utc").copy().reset_index(drop=True)
    history: list[dict] = []
    weight, inner_mae = 0.0, float("nan")
    prior_counts, chosen, inner_values, prediction = [], [], [], []
    previous_time = None
    for row in g.itertuples(index=False):
        now = row.block_start_utc
        if previous_time is not None and (now - previous_time).total_seconds() > GAP_RESET_S:
            history, weight, inner_mae = [], 0.0, float("nan")
        previous_time = now
        prior_counts.append(len(history))
        if len(history) >= MIN_HISTORY:
            recent = history[-HISTORY_WINDOW:]
            scores = []
            for w in WEIGHTS:
                errors = [abs(h["measured_w"] - ((1.0 - w) * h["base_prediction_w"] + w * h["distance_prediction_w"])) for h in recent]
                scores.append((float(np.mean(errors)), w, w))
            inner_mae, _, weight = min(scores)
        pred = (1.0 - weight) * float(row.base_prediction_w) + weight * float(row.distance_prediction_w)
        prediction.append(pred)
        chosen.append(weight)
        inner_values.append(inner_mae)
        history.append({"measured_w": float(row.measured_w), "base_prediction_w": float(row.base_prediction_w),
                        "distance_prediction_w": float(row.distance_prediction_w)})
    g["selected_distance_weight"] = chosen
    g["blend_prediction_w"] = prediction
    g["blend_prior_blocks"] = prior_counts
    g["blend_inner_mae_w"] = inner_values
    return g


def metric(g: pd.DataFrame, col: str) -> dict:
    e = g.measured_w.to_numpy(float) - g[col].to_numpy(float)
    return {"n_blocks": int(len(e)), "mae_w": float(np.mean(np.abs(e))),
            "p90_abs_error_w": float(np.quantile(np.abs(e), .9)), "bias_w": float(np.mean(e))}


def main() -> None:
    a = pd.read_csv(BASE, parse_dates=["block_start_utc"])
    b = pd.read_csv(DIST, parse_dates=["block_start_utc"])
    cols = ["day", "block_start_utc", "krr_plus_tuned_bias_prediction_w"]
    data = a.merge(b[cols], on=["day", "block_start_utc"], how="inner", validate="one_to_one", suffixes=("_base", "_distance"))
    data = data.rename(columns={"krr_plus_tuned_bias_prediction_w_base": "base_prediction_w",
                                "krr_plus_tuned_bias_prediction_w_distance": "distance_prediction_w"})
    result = pd.concat([evaluate_day(g.copy()) for _, g in data.groupby("day", sort=True)], ignore_index=True)
    result.to_csv(OUT_OOF, index=False, encoding="utf-8-sig")
    day_rows = []
    for day, g in result.groupby("day", sort=True):
        for name, col in [("physics_baseline", "predicted_w"),
                          ("adaptive_median", "base_prediction_w"),
                          ("distance_median", "distance_prediction_w"),
                          ("adaptive_distance_blend", "blend_prediction_w")]:
            day_rows.append({"day": int(day), "model": name, **metric(g, col)})
    methods = []
    clean = result[result.day.le(5)]
    baseline = metric(clean, "predicted_w")
    for name, col in [("adaptive_median", "base_prediction_w"),
                      ("distance_median", "distance_prediction_w"),
                      ("adaptive_distance_blend", "blend_prediction_w")]:
        m = metric(clean, col)
        days = []
        for day, g in clean.groupby("day"):
            days.append(metric(g, col)["mae_w"] < metric(g, "predicted_w")["mae_w"])
        methods.append({"model": name, **m,
                        "improvement_pct_vs_physics": 100 * (baseline["mae_w"] - m["mae_w"]) / baseline["mae_w"],
                        "clean_day_wins": int(sum(days))})
    d6 = result[result.day.eq(6)]
    d6_metrics = {name: metric(d6, col) for name, col in [("physics", "predicted_w"),
                                                          ("adaptive_median", "base_prediction_w"),
                                                          ("distance_median", "distance_prediction_w"),
                                                          ("adaptive_blend", "blend_prediction_w")]}
    choice = {f"distance_weight={w:g}": int(n) for w, n in result[(result.day.le(5)) & result.blend_prior_blocks.ge(MIN_HISTORY)].selected_distance_weight.value_counts().items()}
    active = result[(result.day.le(5)) & result.blend_prior_blocks.ge(MIN_HISTORY) & result.blend_inner_mae_w.notna()]
    gap = {"active_blocks": int(len(active)), "mean_prior_inner_mae_w": float(active.blend_inner_mae_w.mean()),
           "next_block_mae_w": float(np.mean(np.abs(active.measured_w - active.blend_prediction_w)))}
    adopted = any(m["model"] == "adaptive_distance_blend" and m["improvement_pct_vs_physics"] >= 10
                  and m["clean_day_wins"] >= 4
                  and m["p90_abs_error_w"] <= baseline["p90_abs_error_w"]
                  and d6_metrics["adaptive_blend"]["p90_abs_error_w"] <= d6_metrics["physics"]["p90_abs_error_w"] for m in methods)
    rows = pd.read_csv(ROWS_IN, low_memory=False)
    rows["timestamp_utc"] = pd.to_datetime(rows.timestamp_utc, utc=True, errors="coerce")
    rows["block_start_utc"] = rows.timestamp_utc.dt.floor("5min")
    lookup = result[["day", "block_start_utc", "blend_prediction_w"]].copy()
    lookup["blend_correction_w"] = lookup.blend_prediction_w - result.predicted_w.to_numpy(float)
    rows = rows.merge(lookup[["day", "block_start_utc", "blend_correction_w"]], on=["day", "block_start_utc"], how="left", validate="many_to_one")
    rows["blend_online_prediction_w"] = rows.predicted_w + rows.blend_correction_w
    rows["blend_online_error_w"] = rows.measured_w - rows.blend_online_prediction_w
    rows.to_csv(OUT_ROWS, index=False, encoding="utf-8-sig")
    row_metrics = {}
    for label, cond in [("clean_days_1_5", rows.day.le(5)), ("day6", rows.day.eq(6))]:
        g = rows.loc[cond].dropna(subset=["blend_online_prediction_w"])
        base_e, blend_e = g.measured_w - g.predicted_w, g.blend_online_error_w
        row_metrics[label] = {"n_rows": int(len(g)), "baseline_mae_w": float(np.mean(np.abs(base_e))),
                             "blend_mae_w": float(np.mean(np.abs(blend_e))),
                             "baseline_p90_abs_error_w": float(np.quantile(np.abs(base_e), .9)),
                             "blend_p90_abs_error_w": float(np.quantile(np.abs(blend_e), .9))}
    payload = {"models": methods, "physical_baseline": baseline, "day6_stress": d6_metrics,
               "selected_distance_weight_counts": choice, "inner_to_next_block": gap,
               "row_level_5s": row_metrics, "adopted_to_shadow": bool(adopted), "production_ready": False,
               "validation": "past-only nested ensemble weight selection from prior 12 completed blocks; current target is excluded",
               "outputs": [str(OUT_OOF), str(OUT_DAYS), str(OUT_JSON), str(OUT_ROWS)]}
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(day_rows).to_csv(OUT_DAYS, index=False, encoding="utf-8-sig")
    OUT_REPORT.write_text("\n".join([
        "# Route-distance model blend evaluation", "",
        "The distance-enabled and location-free prequential kernel models are blended with a weight selected from the prior 12 completed blocks. The current target is excluded from weight selection.", "",
        "| Model | Day1–5 MAE (W) | P90 (W) | improvement vs physics |",
        "|---|---:|---:|---:|"] + [f"| {m['model']} | {m['mae_w']:.1f} | {m['p90_abs_error_w']:.1f} | {m['improvement_pct_vs_physics']:+.1f}% |" for m in methods] + [
        "", f"Selected distance weights: `{choice}`. Prior inner MAE {gap['mean_prior_inner_mae_w']:.1f} W; next-block MAE {gap['next_block_mae_w']:.1f} W over {gap['active_blocks']} blocks.",
        f"Day6 P90: physics {d6_metrics['physics']['p90_abs_error_w']:.1f} W, adaptive median {d6_metrics['adaptive_median']['p90_abs_error_w']:.1f} W, distance blend {d6_metrics['adaptive_blend']['p90_abs_error_w']:.1f} W.",
        f"5-second row MAE: physics {row_metrics['clean_days_1_5']['baseline_mae_w']:.1f} W, blended correction {row_metrics['clean_days_1_5']['blend_mae_w']:.1f} W.", "",
        "The route feature is evaluated as a predictive input. Its fitted effect does not identify a physical cause such as road surface or wind exposure.", "",
        f"Outputs: `{OUT_OOF.relative_to(ROOT)}`, `{OUT_JSON.relative_to(ROOT)}`, `{OUT_ROWS.relative_to(ROOT)}`.", ""
    ]), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
