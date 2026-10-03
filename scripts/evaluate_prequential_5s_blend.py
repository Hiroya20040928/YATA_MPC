#!/usr/bin/env python3
"""Past-only ensemble weighting of the causal 5-second and inertial predictors."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AR_PATH = ROOT / "data/processed/full_race_residual_prequential_5s.csv"
INERTIA_PATH = ROOT / "data/processed/full_race_residual_inertial_5s.csv"
OUT_ROWS = ROOT / "data/processed/full_race_residual_prequential_blend_5s.csv"
OUT_DAYS = ROOT / "reports/residual_prequential_blend_5s_by_day.csv"
OUT_JSON = ROOT / "reports/residual_prequential_blend_5s_summary.json"
OUT_REPORT = ROOT / "reports/residual_prequential_blend_5s_report.md"
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]  # weight on the AR/persistence predictor
MIN_HISTORY = 120
INNER_HOLDOUT = 60
TUNE_EVERY = 60
GAP_RESET_S = 600


def evaluate_day(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.sort_values("timestamp_utc").copy().reset_index(drop=True)
    g["timestamp_utc"] = pd.to_datetime(g.timestamp_utc, utc=True, errors="coerce")
    history: list[dict] = []
    alpha, inner_mae = 1.0, float("nan")
    last_tune_n = -TUNE_EVERY
    previous_time = None
    weights, predictions, prior_n, inner = [], [], [], []
    for row in g.itertuples(index=False):
        now = row.timestamp_utc
        if previous_time is not None and (now - previous_time).total_seconds() > GAP_RESET_S:
            history = []
            alpha, inner_mae, last_tune_n = 1.0, float("nan"), -TUNE_EVERY
        previous_time = now
        n = len(history)
        prior_n.append(n)
        if n >= MIN_HISTORY and n - last_tune_n >= TUNE_EVERY:
            val = history[-INNER_HOLDOUT:]
            scores = []
            for weight in WEIGHTS:
                pred = np.asarray([weight * h["ar_prediction_w"] + (1.0 - weight) * h["inertial_prediction_w"] for h in val])
                target = np.asarray([h["measured_w"] for h in val])
                scores.append((float(np.mean(np.abs(target - pred))), -weight, weight))
            inner_mae, _, alpha = min(scores)
            last_tune_n = n
        prediction = alpha * float(row.residual_prediction_w) + (1.0 - alpha) * float(row.inertial_prediction_w)
        predictions.append(prediction)
        weights.append(alpha)
        inner.append(inner_mae)
        history.append({"measured_w": float(row.measured_w),
                        "ar_prediction_w": float(row.residual_prediction_w),
                        "inertial_prediction_w": float(row.inertial_prediction_w)})
    g["selected_ar_weight"] = weights
    g["blended_prediction_w"] = predictions
    g["blend_prior_samples"] = prior_n
    g["blend_inner_mae_w"] = inner
    return g


def metric(g: pd.DataFrame, col: str) -> dict:
    e = g.measured_w.to_numpy(float) - g[col].to_numpy(float)
    return {"n": int(len(e)), "mae_w": float(np.mean(np.abs(e))),
            "p90_abs_error_w": float(np.quantile(np.abs(e), .9)), "bias_w": float(np.mean(e))}


def main() -> None:
    ar = pd.read_csv(AR_PATH, low_memory=False)
    inertia = pd.read_csv(INERTIA_PATH, low_memory=False,
                          usecols=["day", "timestamp_utc", "inertial_prediction_w"])
    data = ar.merge(inertia, on=["day", "timestamp_utc"], how="inner", validate="one_to_one")
    result = pd.concat([evaluate_day(g.copy()) for _, g in data.groupby("day", sort=True)], ignore_index=True)
    result.to_csv(OUT_ROWS, index=False, encoding="utf-8-sig")
    day_rows = []
    for day, g in result.groupby("day", sort=True):
        for name, col in [("physics_baseline", "predicted_w"),
                          ("past_only_5s_filter", "residual_prediction_w"),
                          ("causal_inertial", "inertial_prediction_w"),
                          ("adaptive_blend", "blended_prediction_w")]:
            day_rows.append({"day": int(day), "model": name, **metric(g, col)})
    day_scores = pd.DataFrame(day_rows)
    day_scores.to_csv(OUT_DAYS, index=False, encoding="utf-8-sig")
    clean = result[result.day.le(5)]
    base, ar_metric, inertia_metric, blended = [metric(clean, c) for c in ["predicted_w", "residual_prediction_w", "inertial_prediction_w", "blended_prediction_w"]]
    d6 = result[result.day.eq(6)]
    d6_base, d6_blend = metric(d6, "predicted_w"), metric(d6, "blended_prediction_w")
    perday = day_scores[(day_scores.day.le(5)) & day_scores.model.eq("adaptive_blend")].set_index("day")
    base_day = day_scores[(day_scores.day.le(5)) & day_scores.model.eq("physics_baseline")].set_index("day")
    wins = int(sum(perday.mae_w < base_day.mae_w))
    warm_wins = 0
    for day in sorted(clean.day.unique()):
        g = result[(result.day.eq(day)) & result.blend_prior_samples.ge(MIN_HISTORY)]
        if len(g) and metric(g, "blended_prediction_w")["mae_w"] < metric(g, "predicted_w")["mae_w"]:
            warm_wins += 1
    improvement = 100 * (base["mae_w"] - blended["mae_w"]) / base["mae_w"]
    tail_pass = blended["p90_abs_error_w"] <= base["p90_abs_error_w"] and d6_blend["p90_abs_error_w"] <= d6_base["p90_abs_error_w"]
    adopted = improvement >= 10 and wins >= 4 and warm_wins >= 4 and tail_pass
    active = clean[(clean.blend_prior_samples.ge(MIN_HISTORY)) & clean.blend_inner_mae_w.notna()]
    gap = {"active_samples": int(len(active)), "prior_inner_mae_w": float(active.blend_inner_mae_w.mean()),
           "next_sample_mae_w": float(np.mean(np.abs(active.measured_w - active.blended_prediction_w)))}
    weight_counts = {f"ar_weight={float(w):g}": int(n) for w, n in result[(result.day.le(5)) & result.blend_prior_samples.ge(MIN_HISTORY)].selected_ar_weight.value_counts().items()}
    summary = {"validation": "past-only convex blending; the blend weight is selected every 60 samples using errors from the prior 60 completed samples",
               "clean_days_1_5": {"physics_baseline": base, "5s_filter": ar_metric,
                                  "causal_inertial": inertia_metric, "adaptive_blend": blended,
                                  "improvement_pct": improvement, "day_wins": wins,
                                  "warm_state_day_wins": warm_wins},
               "day6_stress": {"baseline": d6_base, "adaptive_blend": d6_blend},
               "inner_to_next_sample": gap, "selected_ar_weight_counts": weight_counts,
               "adopted_to_shadow": bool(adopted), "production_ready": False,
               "interpretation": "A successful blend is still a one-step 5-second predictor; it does not imply multi-step MPC readiness.",
               "outputs": [str(OUT_ROWS), str(OUT_DAYS), str(OUT_JSON)]}
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_REPORT.write_text("\n".join([
        "# Past-only blend of online 5-second residual predictors", "",
        "At each five-minute update, the blend weight is selected from the preceding 60 completed samples. Both component predictors were generated causally on the same race log. The current target is appended only after its prediction is fixed.", "",
        f"Day1–5 MAE: physics {base['mae_w']:.1f} W; AR/persistence {ar_metric['mae_w']:.1f} W; causal inertial {inertia_metric['mae_w']:.1f} W; selected blend {blended['mae_w']:.1f} W. Blend P90 {base['p90_abs_error_w']:.1f}→{blended['p90_abs_error_w']:.1f} W; Day6 P90 {d6_base['p90_abs_error_w']:.1f}→{d6_blend['p90_abs_error_w']:.1f} W.",
        f"Prior blend-selection MAE averaged {gap['prior_inner_mae_w']:.1f} W; next-sample MAE averaged {gap['next_sample_mae_w']:.1f} W. Selected AR weights: `{weight_counts}`.", "",
        "This evaluates only the next 5-second sample. The same controller cannot carry this correction into longer horizons without a separate forecast model.", "",
        f"Outputs: `{OUT_ROWS.relative_to(ROOT)}`, `{OUT_DAYS.relative_to(ROOT)}`, `{OUT_JSON.relative_to(ROOT)}`.", ""
    ]), encoding="utf-8")
    print(json.dumps({"baseline": base, "5s_filter": ar_metric, "causal_inertial": inertia_metric,
                      "adaptive_blend": blended, "day6": summary["day6_stress"],
                      "inner_to_next_sample": gap, "selected_ar_weight_counts": weight_counts,
                      "adopted_to_shadow": adopted, "outputs": summary["outputs"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
