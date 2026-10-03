#!/usr/bin/env python3
"""Nested chronological meta-stacking audit for BWSC2025 one-step residuals."""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from evaluate_residual_method_sweep import ROOT, prepare_data

KEYS = ["day", "timestamp_utc"]
OUT_JSON = ROOT / "reports/residual_meta_stacking_summary.json"
OUT_CSV = ROOT / "reports/residual_meta_stacking_test.csv"
OUT_VAL = ROOT / "reports/residual_meta_stacking_validation.csv"
OUT_FULLFIT = ROOT / "data/processed/full_race_residual_risk_committed_5s.csv"
OUT_MODEL = ROOT / "models/vehicle_residual_meta_maxfit.joblib"
OUT_MANIFEST = ROOT / "models/vehicle_residual_meta_maxfit_manifest.json"


def load_base(path: str, name: str, col: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / path, low_memory=False)
    d["timestamp_utc"] = pd.to_datetime(d.timestamp_utc, utc=True, errors="coerce")
    return d[KEYS + [col]].drop_duplicates(KEYS).rename(columns={col: name})


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    e = np.asarray(y, float) - np.asarray(p, float)
    return {"n": int(len(e)), "mae_w": float(np.mean(np.abs(e))),
            "rmse_w": float(np.sqrt(np.mean(e**2))),
            "p90_abs_error_w": float(np.quantile(np.abs(e), .9)),
            "bias_w": float(np.mean(e))}


def candidate_specs() -> list[dict]:
    return [
        {"family": "ridge", "name": "ridge_a10", "params": {"alpha": 10.0}},
        {"family": "ridge", "name": "ridge_a100", "params": {"alpha": 100.0}},
        {"family": "hgb", "name": "hgb_l7", "params": {"max_leaf_nodes": 7, "max_iter": 150, "min_samples_leaf": 40}},
        {"family": "hgb", "name": "hgb_l15", "params": {"max_leaf_nodes": 15, "max_iter": 200, "min_samples_leaf": 40}},
        {"family": "extra", "name": "extra_d8_l30", "params": {"max_depth": 8, "min_samples_leaf": 30, "n_estimators": 150}},
        {"family": "extra", "name": "extra_d12_l30", "params": {"max_depth": 12, "min_samples_leaf": 30, "n_estimators": 150}},
        {"family": "xgb", "name": "xgb_d3", "params": {"max_depth": 3, "n_estimators": 200, "min_child_weight": 20, "learning_rate": .03, "subsample": .8, "colsample_bytree": .85, "reg_lambda": 20}},
        {"family": "cat", "name": "cat_d4", "params": {"depth": 4, "iterations": 250, "learning_rate": .03, "l2_leaf_reg": 20}},
    ]


def make_model(spec: dict):
    p = spec["params"]
    f = spec["family"]
    if f == "ridge":
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), Ridge(**p))
    elif f == "hgb":
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              HistGradientBoostingRegressor(loss="absolute_error", random_state=41,
                                                            l2_regularization=10.0, **p))
    elif f == "extra":
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              ExtraTreesRegressor(random_state=41, n_jobs=4, max_features=.9, **p))
    elif f == "xgb":
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              XGBRegressor(objective="reg:squarederror", eval_metric="mae", n_jobs=4,
                                           random_state=41, verbosity=0, **p))
    elif f == "cat":
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              CatBoostRegressor(loss_function="MAE", random_seed=41, thread_count=4,
                                                verbose=False, allow_writing_files=False, **p))
    else:
        raise ValueError(f)
    return model


def main() -> None:
    val = pd.read_csv(ROOT / "reports/residual_stacking_validation_predictions.csv", low_memory=False)
    test = pd.read_csv(ROOT / "reports/residual_stacking_oof_predictions.csv", low_memory=False)
    for d in (val, test):
        d["timestamp_utc"] = pd.to_datetime(d.timestamp_utc, utc=True, errors="coerce")
    test["HGB"] = test.sweep_prediction_w
    for path, name, col in [
        ("data/processed/full_race_residual_arx_5s.csv", "ARX", "arx_correction_w"),
        ("data/processed/full_race_residual_online_tree_5s.csv", "Tree", "tree_correction_w"),
        ("data/processed/full_race_residual_prequential_5s.csv", "Filter", "residual_correction_w"),
    ]:
        if name not in val.columns:
            val = val.merge(load_base(path, name, col), on=KEYS, how="left", validate="one_to_one")
        if name not in test.columns:
            test = test.merge(load_base(path, name, col), on=KEYS, how="left", validate="one_to_one")

    data = prepare_data().reset_index(drop=True)
    context = ["speed_kmh", "distance_km", "elapsed_s", "headwind_mps", "crosswind_mps",
               "route_grade_smoothed_pct", "residual_lag_1", "residual_lag_3", "residual_lag_6",
               "residual_lag_12", "shortwave_radiation", "global_tilted_irradiance",
               "temperature_2m", "wind_speed_10m"]
    basecols = ["HGB", "ARX", "Tree", "Filter"]
    keep = KEYS + context
    val = val.merge(data[keep], on=KEYS, how="left", validate="one_to_one")
    test = test.merge(data[keep], on=KEYS, how="left", validate="one_to_one")
    meta_features = basecols + context

    # Reserve the latter 30% of each middle validation block for selecting the
    # meta-model configuration; the final 20% remains untouched throughout.
    val["meta_role"] = ""
    for day, g in val.groupby("day", sort=True):
        idx = g.sort_values("timestamp_utc").index.to_numpy()
        cut = max(1, int(math.floor(.70 * len(idx))))
        val.loc[idx[:cut], "meta_role"] = "meta_fit"
        val.loc[idx[cut:], "meta_role"] = "meta_select"
    fit = val[(val.day.le(5)) & (val.meta_role.eq("meta_fit"))].dropna(subset=meta_features)
    select = val[(val.day.le(5)) & (val.meta_role.eq("meta_select"))].dropna(subset=meta_features)
    yfit = fit.target_residual_w.to_numpy(float)
    ysel = select.target_residual_w.to_numpy(float)
    leaderboard = []
    for spec in candidate_specs():
        model = make_model(spec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(fit[meta_features].to_numpy(float), yfit)
        p = model.predict(select[meta_features].to_numpy(float))
        per_day = []
        for day, g in select.assign(pred=p).groupby("day"):
            per_day.append(metrics(g.target_residual_w.to_numpy(float), g.pred.to_numpy(float))["mae_w"])
        leaderboard.append({"name": spec["name"], "spec": spec,
                            "mean_day_select_mae_w": float(np.mean(per_day)),
                            "select_mae_w": float(np.mean(np.abs(ysel - p)))})
    leaderboard.sort(key=lambda x: x["mean_day_select_mae_w"])
    winner = next(s for s in candidate_specs() if s["name"] == leaderboard[0]["name"])
    clean_val = val[(val.day.le(5)) & val[meta_features].notna().all(axis=1)]
    final = make_model(winner)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    final.fit(clean_val[meta_features].to_numpy(float), clean_val.target_residual_w.to_numpy(float))
    test["meta_prediction_w"] = final.predict(test[meta_features].to_numpy(float))
    test["meta_error_w"] = test.target_residual_w - test.meta_prediction_w
    test.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    val.to_csv(OUT_VAL, index=False, encoding="utf-8-sig")
    results = {}
    for scope, rows in [("clean_days_1_5", test[test.day.le(5)]), ("day6_stress", test[test.day.eq(6)])]:
        results[scope] = {
            "physical_baseline": metrics(rows.target_residual_w.to_numpy(float), np.zeros(len(rows))),
            "HGB": metrics(rows.target_residual_w.to_numpy(float), rows.HGB.to_numpy(float)),
            "global_convex_blend": metrics(rows.target_residual_w.to_numpy(float), rows.global_blend_correction_w.to_numpy(float)),
            "meta_model": metrics(rows.target_residual_w.to_numpy(float), rows.meta_prediction_w.to_numpy(float)),
        }
    payload = {
        "validation_design": "Within each day, first 70% of the middle validation block fits the meta-model; its last 30% selects configuration; final 20% untouched tail is scored once.",
        "meta_features": meta_features,
        "selected": winner,
        "selection_leaderboard": leaderboard,
        "results": results,
        "outputs": [str(OUT_CSV), str(OUT_VAL), str(OUT_JSON)],
    }

    # User-authorized maximum-commit fit: refit the selected stack on every
    # available labeled row. These same-row scores are in-sample fit only.
    full = data[KEYS + ["target_residual_w", "measured_w", "predicted_w"] + context].copy()
    for path, name, col in [
        ("data/processed/full_race_residual_maxfit_5s.csv", "HGB", "committed_residual_correction_w"),
        ("data/processed/full_race_residual_arx_5s.csv", "ARX", "arx_correction_w"),
        ("data/processed/full_race_residual_online_tree_5s.csv", "Tree", "tree_correction_w"),
        ("data/processed/full_race_residual_prequential_5s.csv", "Filter", "residual_correction_w"),
    ]:
        full = full.merge(load_base(path, name, col), on=KEYS, how="left", validate="one_to_one")
    full["target_residual_w"] = pd.to_numeric(full.target_residual_w, errors="coerce")
    # The committed inference pipeline imputes unavailable weather/history
    # fields. Preserve every labeled measurement row in the full-data fit.
    full = full.dropna(subset=["target_residual_w", "measured_w", "predicted_w"]).copy()
    committed = make_model(winner)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        committed.fit(full[meta_features].to_numpy(float), full.target_residual_w.to_numpy(float))
    full["risk_committed_residual_correction_w"] = committed.predict(full[meta_features].to_numpy(float))
    full["risk_committed_prediction_w"] = full.predicted_w + full.risk_committed_residual_correction_w
    full["risk_committed_corrected_error_w"] = full.measured_w - full.risk_committed_prediction_w
    full["fit_scope"] = "all available BWSC2025 rows; in-sample, authorized maximum-fit"
    full.to_csv(OUT_FULLFIT, index=False, encoding="utf-8-sig")
    joblib.dump(committed, OUT_MODEL)
    full_metrics = {
        "physical_baseline": metrics(full.target_residual_w.to_numpy(float), np.zeros(len(full))),
        "risk_committed_meta_fit": metrics(full.target_residual_w.to_numpy(float),
                                            full.risk_committed_residual_correction_w.to_numpy(float)),
        "selected_hgb_full_fit": metrics(full.target_residual_w.to_numpy(float), full.HGB.to_numpy(float)),
        "n_rows": int(len(full)),
        "scope": "in-sample reconstruction score; not a forward performance estimate",
    }
    payload["all_data_risk_commit"] = {
        "estimator": winner,
        "training_rows": int(len(full)),
        "features": meta_features,
        "in_sample_metrics": full_metrics,
        "data_csv": str(OUT_FULLFIT),
        "model": str(OUT_MODEL),
        "manifest": str(OUT_MANIFEST),
    }
    manifest = {
        "purpose": "Maximum BWSC2025 historical residual fit; not yet a validated forward MPC model",
        "family": winner,
        "features": meta_features,
        "training_rows": int(len(full)),
        "training_scope": "all available BWSC2025 rows including Day6",
        "base_prediction_inputs": {
            "HGB": "full-data HistGradientBoosting residual correction",
            "ARX": "causal online ARX residual correction",
            "Tree": "causal online tree residual correction",
            "Filter": "causal residual filter correction",
        },
        "validation": payload["results"],
        "all_data_in_sample_metrics": full_metrics,
        "important_limit": "All-data fit uses training labels. The exact measured-channel oracle is excluded from this predictor. Historical route, speed, and weather are conditional inputs; this does not validate future forecast/MPC errors.",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["outputs"] += [str(OUT_FULLFIT), str(OUT_MODEL), str(OUT_MANIFEST)]
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
