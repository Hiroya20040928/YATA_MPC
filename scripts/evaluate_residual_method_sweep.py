#!/usr/bin/env python3
"""Broad residual-model sweep and full-data fit for BWSC2025.

Validation is chronological inside each race day: 60% fit, 20% model selection,
20% untouched one-step test. Only past residual features are used. The sweep
also writes an explicitly labeled measurement-oracle upper bound and a final
model refit on all available rows; the latter's training score is not an OOF
score.
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
RESIDUAL = ROOT / "data/processed/full_race_residual_corrected_5s.csv"
RAW = ROOT / "data/processed/race_days_distance_geo_weather.csv"
OUT_CSV = ROOT / "reports/residual_method_sweep_validation.csv"
OUT_JSON = ROOT / "reports/residual_method_sweep_summary.json"
OUT_REPORT = ROOT / "reports/residual_method_sweep_report.md"
OUT_PRED = ROOT / "data/processed/full_race_residual_maxfit_5s.csv"
OUT_MODEL = ROOT / "models/vehicle_residual_maxfit.joblib"
OUT_MANIFEST = ROOT / "models/vehicle_residual_maxfit_manifest.json"

KEYS = ["day", "timestamp_utc"]
RAW_FEATURES = [
    "route_latitude", "route_longitude", "route_elevation_dem_m",
    "route_grade_smoothed_pct", "temperature_2m", "relative_humidity_2m",
    "dew_point_2m", "apparent_temperature", "temperature_80m", "temperature_120m",
    "temperature_180m", "pressure_msl", "surface_pressure", "cloud_cover",
    "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "wind_speed_10m",
    "wind_speed_80m", "wind_speed_120m", "wind_speed_180m", "wind_direction_10m",
    "wind_direction_80m", "wind_direction_120m", "wind_direction_180m",
    "wind_gusts_10m", "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "direct_normal_irradiance", "global_tilted_irradiance", "terrestrial_radiation",
    "shortwave_radiation_instant", "direct_radiation_instant",
    "diffuse_radiation_instant", "direct_normal_irradiance_instant",
    "global_tilted_irradiance_instant", "terrestrial_radiation_instant",
    "cloud_cover_15min", "wind_speed_10m_15min", "wind_direction_10m_15min",
    "wind_gusts_10m_15min", "shortwave_radiation_15min", "direct_radiation_15min",
    "diffuse_radiation_15min", "global_tilted_irradiance_15min", "visibility",
    "uv_index", "is_day", "sunshine_duration", "wet_bulb_temperature_2m",
    "vapour_pressure_deficit", "cape", "boundary_layer_height", "precipitation",
    "rain", "showers", "weather_code", "weather_spatial_source_spacing_km",
]

FEATURES: list[str] = []


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    return {"n": int(len(e)), "mae_w": float(np.mean(np.abs(e))),
            "rmse_w": float(np.sqrt(np.mean(e**2))),
            "p90_abs_error_w": float(np.quantile(np.abs(e), .9)),
            "bias_w": float(np.mean(e))}


def prepare_data() -> pd.DataFrame:
    residual = pd.read_csv(RESIDUAL, low_memory=False)
    raw = pd.read_csv(RAW, usecols=lambda c: c in set(KEYS + RAW_FEATURES + [
        "Speed[kmh]", "Solar_Watt[W]", "Battery_Watt[W]", "distance_from_darwin_m",
        "timestamp_local", "wind_direction_10m", "wind_speed_10m",
        "route_grade_smoothed_pct", "weather_source", "weather_temporal_source_resolution",
        "weather_spatial_source_spacing_km",
    ]), low_memory=False)
    for f in (residual, raw):
        f["timestamp_utc"] = pd.to_datetime(f.timestamp_utc, utc=True, errors="coerce")
    g = residual.merge(raw, on=KEYS, how="left", suffixes=("", "_raw"), validate="one_to_one")
    if g["Solar_Watt[W]"].isna().any() or g["Battery_Watt[W]"].isna().any():
        raise ValueError("Residual rows failed to match measured PV/battery channels")
    g = g.sort_values(["day", "timestamp_utc"]).reset_index(drop=True)
    g["target_residual_w"] = pd.to_numeric(g.measured_w, errors="coerce") - pd.to_numeric(g.predicted_w, errors="coerce")
    g["distance_km"] = pd.to_numeric(g.distance_from_darwin_m, errors="coerce") / 1000.0
    g["speed_kmh"] = pd.to_numeric(g["Speed[kmh]"], errors="coerce")
    g["speed_mps"] = g.speed_kmh / 3.6
    g["elapsed_s"] = (g.timestamp_utc - g.groupby("day").timestamp_utc.transform("min")).dt.total_seconds()
    # Route tangent and wind decomposition are available from the known route/weather path.
    lat = pd.to_numeric(g.route_latitude, errors="coerce")
    lon = pd.to_numeric(g.route_longitude, errors="coerce")
    dlat = lat.groupby(g.day).diff().fillna(0.0) * 111320.0
    dlon = lon.groupby(g.day).diff().fillna(0.0) * 111320.0 * np.cos(np.deg2rad(lat))
    heading = np.arctan2(dlon, dlat)
    wind_dir = np.deg2rad(pd.to_numeric(g.wind_direction_10m, errors="coerce"))
    wind_speed = pd.to_numeric(g.wind_speed_10m, errors="coerce")
    rel = wind_dir - heading
    g["headwind_mps"] = wind_speed * np.cos(rel)
    g["crosswind_mps"] = wind_speed * np.sin(rel)
    for col in ["wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m"]:
        if col in g:
            angle = np.deg2rad(pd.to_numeric(g[col], errors="coerce"))
            g[col + "_sin"] = np.sin(angle)
            g[col + "_cos"] = np.cos(angle)
    g["time_of_day_sin"] = np.sin(2 * np.pi * (g.elapsed_s % 86400.0) / 86400.0)
    g["time_of_day_cos"] = np.cos(2 * np.pi * (g.elapsed_s % 86400.0) / 86400.0)
    g["speed_delta_mps"] = g.groupby("day").speed_mps.diff() / 5.0
    g["inertial_power_proxy_w"] = 314.0 * g.speed_mps * g.speed_delta_mps.clip(-0.5, 0.5)

    # Reset causal residual history after a long logging gap; no current/future
    # residual enters any feature at its prediction timestamp.
    delta_s = g.groupby("day").timestamp_utc.diff().dt.total_seconds()
    g["stream_id"] = ((delta_s.isna()) | delta_s.gt(600.0)).groupby(g.day).cumsum()
    group = g.groupby(["day", "stream_id"], sort=False)["target_residual_w"]
    for lag in [1, 2, 3, 6, 12, 24, 60, 120]:
        name = f"residual_lag_{lag}"
        g[name] = group.shift(lag)
        FEATURES.append(name)
    for window in [3, 6, 12, 24, 60, 120]:
        shifted = group.shift(1).groupby([g.day, g.stream_id], sort=False)
        for stat in ["mean", "median", "std"]:
            name = f"residual_{stat}_{window}"
            rolling = shifted.rolling(window=window, min_periods=max(1, window // 4))
            value = getattr(rolling, stat)().reset_index(level=[0, 1], drop=True)
            g[name] = value.reindex(g.index)
            FEATURES.append(name)
    for field in [c for c in RAW_FEATURES if c in g.columns]:
        if pd.api.types.is_numeric_dtype(g[field]):
            FEATURES.append(field)
    FEATURES.extend(["speed_kmh", "distance_km", "elapsed_s", "headwind_mps",
                     "crosswind_mps", "speed_delta_mps", "inertial_power_proxy_w",
                     "time_of_day_sin", "time_of_day_cos"])
    # Exclude repeated columns introduced by the raw join and fields that are
    # direct measurements of the response or quality/provenance labels.
    FEATURES[:] = list(dict.fromkeys(FEATURES))
    return g


def specs() -> list[dict[str, Any]]:
    result = []
    for alpha in [10.0, 100.0]:
        result.append({"family": "ridge", "name": f"ridge_a{alpha:g}", "scale": True,
                       "params": {"alpha": alpha}})
    for leaves, iters, leafsize in [(7, 100, 30), (15, 200, 30), (31, 300, 50)]:
        result.append({"family": "hist_gradient_boosting", "name": f"hgb_l{leaves}_i{iters}", "scale": False,
                       "params": {"max_leaf_nodes": leaves, "max_iter": iters,
                                  "min_samples_leaf": leafsize, "learning_rate": 0.05,
                                  "l2_regularization": 10.0}})
    for depth, leaf, trees in [(5, 30, 100), (10, 30, 100), (None, 60, 150)]:
        result.append({"family": "extra_trees", "name": f"extra_d{depth}_leaf{leaf}", "scale": False,
                       "params": {"max_depth": depth, "min_samples_leaf": leaf, "n_estimators": trees}})
    for depth, leaf, trees in [(8, 30, 150), (None, 60, 200)]:
        result.append({"family": "random_forest", "name": f"rf_d{depth}_leaf{leaf}", "scale": False,
                       "params": {"max_depth": depth, "min_samples_leaf": leaf, "n_estimators": trees}})
    for depth, trees, child in [(3, 250, 20), (5, 400, 30)]:
        result.append({"family": "xgboost", "name": f"xgb_d{depth}_n{trees}", "scale": False,
                       "params": {"max_depth": depth, "n_estimators": trees, "min_child_weight": child,
                                  "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8,
                                  "reg_lambda": 20.0, "reg_alpha": 1.0}})
    for depth, trees in [(4, 350), (6, 500), (8, 650)]:
        result.append({"family": "catboost", "name": f"cat_d{depth}_n{trees}", "scale": False,
                       "params": {"depth": depth, "iterations": trees, "learning_rate": 0.03,
                                  "l2_leaf_reg": 20.0}})
    for layers, alpha in [((32, 16), 1.0), ((64, 32), 10.0)]:
        result.append({"family": "mlp", "name": f"mlp_{'x'.join(map(str,layers))}_a{alpha:g}", "scale": True,
                       "params": {"hidden_layer_sizes": layers, "alpha": alpha, "max_iter": 350}})
    return result


def model_pipeline(spec: dict[str, Any]) -> Pipeline:
    p = spec["params"]
    family = spec["family"]
    if family == "ridge":
        estimator = Ridge(alpha=p["alpha"])
    elif family == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(loss="absolute_error", random_state=17, **p)
    elif family == "extra_trees":
        estimator = ExtraTreesRegressor(random_state=17, n_jobs=4, max_features=0.9, **p)
    elif family == "random_forest":
        estimator = RandomForestRegressor(random_state=17, n_jobs=4, max_features=0.9, **p)
    elif family == "xgboost":
        estimator = XGBRegressor(objective="reg:squarederror", eval_metric="mae", n_jobs=4,
                                 random_state=17, verbosity=0, **p)
    elif family == "catboost":
        estimator = CatBoostRegressor(loss_function="MAE", random_seed=17, thread_count=4,
                                      verbose=False, allow_writing_files=False, **p)
    elif family == "mlp":
        estimator = MLPRegressor(random_state=17, early_stopping=False, tol=1e-4, **p)
    else:
        raise ValueError(f"unknown family: {family}")
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if spec["scale"]:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def fit_score(spec: dict[str, Any], x_train: np.ndarray, y_train: np.ndarray,
              x_eval: np.ndarray, y_eval: np.ndarray) -> tuple[Pipeline, dict]:
    model = model_pipeline(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)
    return model, metric(y_eval, model.predict(x_eval))


def day_splits(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(g)
    i1 = max(120, int(math.floor(n * .60)))
    i2 = max(i1 + 60, int(math.floor(n * .80)))
    i2 = min(i2, n - 20)
    return np.arange(0, i1), np.arange(i1, i2), np.arange(i2, n)


def main() -> None:
    data = prepare_data()
    x_all = data[FEATURES].to_numpy(float)
    y_all = data.target_residual_w.to_numpy(float)
    candidate_specs = specs()
    validation_rows = []
    day_cache: dict[int, dict[str, Any]] = {}
    for day, index in data.groupby("day", sort=True).groups.items():
        pos = np.asarray(list(index), dtype=int)
        frame = data.loc[pos].reset_index(drop=True)
        x = x_all[pos]
        y = y_all[pos]
        i_train, i_val, i_test = day_splits(frame)
        day_cache[int(day)] = {"pos": pos, "frame": frame, "x": x, "y": y,
                               "train": i_train, "val": i_val, "test": i_test}
        for spec in candidate_specs:
            model, val_metric = fit_score(spec, x[i_train], y[i_train], x[i_val], y[i_val])
            validation_rows.append({"day": int(day), "family": spec["family"],
                                    "configuration": spec["name"], **val_metric})
    validation = pd.DataFrame(validation_rows)
    clean_validation = validation[validation.day.le(5)]
    leaderboard = (clean_validation.groupby(["family", "configuration"], as_index=False)
                   .agg(mean_day_validation_mae_w=("mae_w", "mean"),
                        median_day_validation_mae_w=("mae_w", "median"),
                        mean_day_validation_p90_w=("p90_abs_error_w", "mean")))
    family_best = (leaderboard.sort_values(["family", "mean_day_validation_mae_w"])
                   .groupby("family", as_index=False).first())
    winner_row = family_best.sort_values("mean_day_validation_mae_w").iloc[0]
    winner_family = str(winner_row.family)
    winner_name = str(winner_row.configuration)
    winner = next(s for s in candidate_specs if s["name"] == winner_name)

    # Final untouched last-20%-of-day test; family/configuration was selected
    # only on the prior validation portions of Days 1-5.
    test_rows = []
    prediction_chunks = []
    for day, cache in day_cache.items():
        x, y = cache["x"], cache["y"]
        fit_idx = np.concatenate([cache["train"], cache["val"]])
        test_idx = cache["test"]
        model, test_metric = fit_score(winner, x[fit_idx], y[fit_idx], x[test_idx], y[test_idx])
        scope = "clean_day_outer_test" if day <= 5 else "day6_stress_test"
        test_rows.append({"day": day, "family": winner_family, "configuration": winner_name,
                          "scope": scope, **test_metric,
                          "baseline_mae_w": float(np.mean(np.abs(y[test_idx]))),
                          "baseline_p90_w": float(np.quantile(np.abs(y[test_idx]), .9))})
        frame = cache["frame"].iloc[test_idx].copy()
        frame["sweep_prediction_w"] = model.predict(x[test_idx])
        frame["sweep_corrected_error_w"] = frame.target_residual_w - frame.sweep_prediction_w
        frame["sweep_scope"] = scope
        prediction_chunks.append(frame[["day", "timestamp_utc", "target_residual_w", "sweep_prediction_w",
                                        "sweep_corrected_error_w", "sweep_scope"]])

    # Refit the validation-selected configuration on every available labeled row.
    committed = model_pipeline(winner)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        committed.fit(x_all, y_all)
    fitted_residual = committed.predict(x_all)
    data["committed_residual_correction_w"] = fitted_residual
    data["committed_prediction_w"] = data.predicted_w + fitted_residual
    data["committed_corrected_error_w"] = data.measured_w - data.committed_prediction_w
    # Explicit target-leakage/oracle ceiling: measured gross output is defined as
    # PV + signed battery channel, so this is an identity, not a forecast model.
    data["measured_oracle_prediction_w"] = pd.to_numeric(data["Solar_Watt[W]"], errors="coerce") + pd.to_numeric(data["Battery_Watt[W]"], errors="coerce")
    data["oracle_identity_error_w"] = data.measured_w - data.measured_oracle_prediction_w
    data["oracle_residual_correction_w"] = data.measured_oracle_prediction_w - data.predicted_w
    data["oracle_corrected_error_w"] = data.measured_w - (data.predicted_w + data.oracle_residual_correction_w)

    joblib.dump(committed, OUT_MODEL)
    data[["day", "timestamp_utc", "event_phase", "distance_from_darwin_m", "Speed[kmh]",
          "measured_w", "predicted_w", "target_residual_w", "committed_residual_correction_w",
          "committed_prediction_w", "committed_corrected_error_w", "measured_oracle_prediction_w",
          "oracle_identity_error_w", "oracle_residual_correction_w", "oracle_corrected_error_w"]].to_csv(
              OUT_PRED, index=False, encoding="utf-8-sig")
    validation.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(test_rows).to_csv(ROOT / "reports/residual_method_sweep_outer_test_by_day.csv",
                                   index=False, encoding="utf-8-sig")
    pd.concat(prediction_chunks, ignore_index=True).to_csv(
        ROOT / "reports/residual_method_sweep_oof_predictions.csv", index=False, encoding="utf-8-sig")

    selected_clean_tests = pd.DataFrame(test_rows)
    selected_clean_tests = selected_clean_tests[selected_clean_tests.day.le(5)]
    selected_d6 = pd.DataFrame(test_rows)
    selected_d6 = selected_d6[selected_d6.day.eq(6)]
    in_sample = metric(y_all, fitted_residual)
    oracle_metric = metric(y_all, data.oracle_residual_correction_w.to_numpy(float))
    oracle_corrected_metric = metric(np.zeros(len(data)), data.oracle_corrected_error_w.to_numpy(float))
    clean_outer = selected_clean_tests
    summary = {
        "selection": "global family/configuration selected by mean chronological validation MAE across Days 1-5; final test is the last 20% of each day and was not used for selection",
        "prediction_semantics": "one-step conditional prediction; current speed, route and archived weather fields are treated as known; residual lags use only earlier measurements",
        "rows": int(len(data)), "rows_by_day": {str(int(k)): int(v) for k, v in data.day.value_counts().sort_index().items()},
        "features": FEATURES,
        "candidate_count": int(len(candidate_specs)),
        "candidate_families": sorted(validation.family.unique().tolist()),
        "global_winner": {"family": winner_family, "configuration": winner_name,
                          "mean_day_validation_mae_w": float(winner_row.mean_day_validation_mae_w),
                          "mean_day_validation_p90_w": float(winner_row.mean_day_validation_p90_w)},
        "family_leaderboard_by_clean_validation": family_best.sort_values("mean_day_validation_mae_w").to_dict(orient="records"),
        "selected_model_outer_test_clean_days_1_5": {
            "n_days": int(len(clean_outer)),
            "n_rows": int(clean_outer.n.sum()),
            "mean_day_baseline_mae_w": float(clean_outer.baseline_mae_w.mean()),
            "mean_day_model_mae_w": float(clean_outer.mae_w.mean()),
            "mean_day_baseline_p90_w": float(clean_outer.baseline_p90_w.mean()),
            "mean_day_model_p90_w": float(clean_outer.p90_abs_error_w.mean()),
            "per_day": clean_outer.to_dict(orient="records")},
        "selected_model_day6_stress_outer_test": selected_d6.to_dict(orient="records"),
        "full_data_commit_fit_in_sample_not_oof": in_sample,
        "measurement_oracle_not_a_forecast": {"construction": "Solar_Watt[W] + Battery_Watt[W] equals measured_w by definition",
                                               "residual_correction_matches_target": oracle_metric,
                                               "corrected_output_error": oracle_corrected_metric,
                                               "measured_channel_identity_error": metric(data.measured_w.to_numpy(float), data.measured_oracle_prediction_w.to_numpy(float)),
                                               "valid_for_future_prediction": False},
        "data_hash_sha256": hashlib.sha256(RESIDUAL.read_bytes()).hexdigest(),
        "outputs": [str(OUT_CSV), str(OUT_PRED), str(OUT_MODEL)],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MANIFEST.write_text(json.dumps({"model": str(OUT_MODEL), "selected_family": winner_family,
        "selected_configuration": winner_name, "feature_names": FEATURES,
        "fit_rows": int(len(data)), "fit_days": [int(x) for x in sorted(data.day.unique())],
        "training_target": "measured_w - predicted_w", "inference_horizon": "one 5-second step",
        "uses_previous_observed_residuals": True,
        "uses_logged_speed_and_archived_weather_as_conditional_inputs": True,
        "measured Solar_Watt and Battery_Watt are excluded from the model features": True,
        "validation_summary": summary["selected_model_outer_test_clean_days_1_5"],
        "limitations": ["Full-data in-sample fit is optimistic and is not a validation metric.",
                        "Future speed/weather forecast errors are not represented.",
                        "Model selection is restricted to the supplied BWSC2025 logs."]}, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# BWSC2025 残差モデル全候補スイープと全データ再フィット", "",
             "## 結果", "",
             f"Day1–5の各日の60%で学習、次の20%で全候補を選択し、最後の20%を一度だけ評価した。選択したモデルは **{winner_family} / {winner_name}**。",
             f"未使用テスト区間の平均日別MAEは物理ベース {summary['selected_model_outer_test_clean_days_1_5']['mean_day_baseline_mae_w']:.1f} W、残差補正後 {summary['selected_model_outer_test_clean_days_1_5']['mean_day_model_mae_w']:.1f} W。",
             f"Days1–5で選んだままのDay6ストレス評価は: {json.dumps(summary['selected_model_day6_stress_outer_test'], ensure_ascii=False)}", "",
             f"全{len(candidate_specs)}構成を検証し、最終選択モデルを16,242行すべてに再フィットした学習内残差はMAE {in_sample['mae_w']:.2f} W、P90 {in_sample['p90_abs_error_w']:.2f} W。これは同じ行を使って学習・評価した値なので、性能証拠ではなく学習内適合度である。", "",
             "## 評価の切り分け", "",
             "- 予測モデル特徴量は速度、走行距離、ルート勾配・位置、Open-Meteo由来の複数高度の風・気温、雲量、短波・直達・散乱・傾斜面日射、日射角の風成分、5秒間隔の過去残差ラグ／移動統計量である。",
             "- 現在時刻のSolar_Watt・Battery_Wattは応答値を構成する実測列なので、予測モデルの特徴量から除外した。",
             "- 別枠の測定オラクルはSolar_Watt + Battery_Wattをそのまま用い、定義上の測定値を完全再構成する。残差ゼロだが、将来予測器として無効である。",
             "- 最終学習内スコアは全日全行のフィット結果である。未使用の時系列テスト指標は上記の最後20%だけである。Day6はモデル選択に使っていない。",
             "- 速度と天候はログ上の値を既知として条件づけた。MPCで生成する未来速度・天気予報の誤差は含まない。", "",
             "## 出力", "",
             f"- 検証スコア全候補: `{OUT_CSV.relative_to(ROOT)}`",
             "- 最終20%の行別予測: `reports/residual_method_sweep_oof_predictions.csv`",
             f"- 全行再フィットと測定オラクル列: `{OUT_PRED.relative_to(ROOT)}`",
             f"- シリアライズ済み学習器: `{OUT_MODEL.relative_to(ROOT)}`",
             f"- 特徴量・モデル・検証条件: `{OUT_MANIFEST.relative_to(ROOT)}`", ""]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"winner": summary["global_winner"],
                      "clean_outer": summary["selected_model_outer_test_clean_days_1_5"],
                      "day6": summary["selected_model_day6_stress_outer_test"],
                      "full_fit_in_sample": in_sample, "oracle": oracle_metric,
                      "outputs": summary["outputs"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
