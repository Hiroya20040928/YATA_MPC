#!/usr/bin/env python3
"""Compare human and hierarchical-MPC time on clean, energy-feasible intervals.

This is a model-based counterfactual, not a measured vehicle run. It uses only
segments with an optimal finite-grid upper plan, no logged event, 5 s continuous
telemetry, and speed in the open-road range. The lower acceleration request is
re-simulated from the segment's actual entry speed with its causal empirical
acceleration limit. Component and residual models are joined for interval-level
power-error audits; they are not added together when they estimate the same load.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from replay_hierarchical_mpc_2025 import lower_move_block_mpc, MAX_MODEL_SPEED_KMH  # noqa: E402

REPLAY = ROOT / "data/processed/mpc_2025_as_live_replay.csv"
COMPONENT = ROOT / "data/processed/full_race_component_replay_5s.csv"
ARX = ROOT / "data/processed/full_race_residual_arx_5s.csv"
TREE = ROOT / "data/processed/full_race_residual_online_tree_5s.csv"
FILTER = ROOT / "data/processed/full_race_residual_prequential_5s.csv"
MAXFIT = ROOT / "data/processed/full_race_residual_maxfit_5s.csv"
META = ROOT / "data/processed/full_race_residual_risk_committed_5s.csv"
MOTOR = ROOT / "models/m2096_eco_lo_bench_map.csv"
BATTERY = ROOT / "models/component_replay_fit.json"
BATTERY_CV = ROOT / "models/battery_soc_resistance_candidate_cv.json"
INERTIAL_CV = ROOT / "models/vehicle_inertial_candidate_cv.json"
ONLINE_5MIN = ROOT / "models/vehicle_residual_online.json"
RESIDUAL_COMMON = ROOT / "reports/residual_common_test_comparison.csv"
OUT_INTERVALS = ROOT / "reports/mpc_2025_effective_intervals.csv"
OUT_DAYS = ROOT / "reports/mpc_2025_effective_interval_day_summary.csv"
OUT_MODELS = ROOT / "reports/mpc_2025_model_usage_audit.csv"
OUT_REPORT = ROOT / "reports/mpc_2025_effective_interval_assessment.md"
OUT_SUMMARY = ROOT / "reports/mpc_2025_effective_interval_summary.json"

BAD_EVENT = r"incident:|control_stop:|traffic_stop:|driver_change:|repair:|charge:|hatch_repair:"
MIN_DISTANCE_M = 1_000.0
MAX_SAMPLE_GAP_S = 5.01
RESIDUAL_MODEL_SOURCES = {
    "arx": (ARX, "corrected_prediction_w", "causal 5 s one-step ARX"),
    "tree": (TREE, "tree_prediction_w", "causal online tree 5 s one-step"),
    "filter": (FILTER, "corrected_prediction_w", "prequential residual filter 5 s"),
    "hgb_maxfit": (MAXFIT, "committed_prediction_w", "full-data HGB; in-sample on these rows"),
    "meta_maxfit": (META, "risk_committed_prediction_w", "full-data meta model; in-sample on these rows"),
}


def read_table(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d["timestamp_utc"] = pd.to_datetime(d["timestamp_utc"], utc=True)
    return d


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(c) for c in frame.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{value:.2f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def simulate_lower_layer(rows: pd.DataFrame, distance_m: float, accel_scale: float) -> tuple[float, float]:
    """Return elapsed seconds and terminal speed for receding 5 s scalar MPC."""
    entry_speed = float(rows["Speed[kmh]"].iloc[0])
    target = float(rows["upper_target_speed_kmh"].iloc[0])
    limits = pd.to_numeric(rows["lower_accel_bound_mps2"], errors="coerce").to_numpy(float)
    limits = limits[np.isfinite(limits) & (limits > 0)]
    if not len(limits):
        limits = np.array([0.05])
    limits = np.clip(limits * accel_scale, 0.01, 0.25)
    v_ms = entry_speed / 3.6
    distance_done = 0.0
    elapsed_s = 0.0
    max_steps = int(max(600, distance_m / (60 / 3.6) / 5 + 240))
    for k in range(max_steps):
        amax = float(limits[min(k, len(limits) - 1)])
        accel, _, _ = lower_move_block_mpc(v_ms * 3.6, target, amax)
        v_next = float(np.clip(v_ms + accel * 5.0, 60.0 / 3.6, MAX_MODEL_SPEED_KMH / 3.6))
        ds = max(0.0, 0.5 * (v_ms + v_next) * 5.0)
        if ds <= 0:
            return float("nan"), float("nan")
        if distance_done + ds >= distance_m:
            frac = (distance_m - distance_done) / ds
            elapsed_s += 5.0 * frac
            v_ms = v_ms + frac * (v_next - v_ms)
            distance_done = distance_m
            return elapsed_s, v_ms * 3.6
        elapsed_s += 5.0
        distance_done += ds
        v_ms = v_next
    return float("nan"), float("nan")


def build_intervals() -> tuple[pd.DataFrame, dict]:
    d = read_table(REPLAY).sort_values(["day", "timestamp_utc"]).reset_index(drop=True)
    component = read_table(COMPONENT)
    cols = ["day", "timestamp_utc", "predicted_vehicle_load_w", "predicted_pv_w",
            "predicted_battery_power_required_w", "battery_voltage_predicted_v",
            "battery_soc_estimate_pct", "road_model_comparison_eligible"]
    d = d.merge(component[cols], on=["day", "timestamp_utc"], how="left", validate="one_to_one")
    # Each correction family is retained as a parallel model score. Never sum
    # several predictions of the same HV load; they are alternative estimators.
    for name, (path, pred_col, _) in RESIDUAL_MODEL_SOURCES.items():
        z = read_table(path)
        if pred_col not in z.columns:
            raise KeyError(f"{path.name} lacks {pred_col}")
        d = d.merge(z[["day", "timestamp_utc", pred_col]].rename(columns={pred_col: f"{name}_prediction_w"}),
                    on=["day", "timestamp_utc"], how="left", validate="one_to_one")
    d["measured_vehicle_load_w"] = d["Solar_Watt[W]"] + d["Battery_Watt[W]"]
    d["physical_abs_error_w"] = (d.measured_vehicle_load_w - d.predicted_vehicle_load_w).abs()
    d["pv_abs_error_w"] = (d["Solar_Watt[W]"] - d.predicted_pv_w).abs()
    d["battery_voltage_abs_error_v"] = (d["Main_Volt[V]"] - d.battery_voltage_predicted_v).abs()
    for name in RESIDUAL_MODEL_SOURCES:
        d[f"{name}_abs_error_w"] = (d.measured_vehicle_load_w - d[f"{name}_prediction_w"]).abs()

    flags = d.event_flags.fillna("").astype(str)
    slack = pd.to_numeric(d.upper_energy_budget_wh, errors="coerce") - pd.to_numeric(
        d.upper_predicted_remaining_net_energy_wh, errors="coerce")
    d["interval_row_valid"] = (
        d.event_phase.eq("road_day") & ~flags.str.contains(BAD_EVENT, regex=True)
        & d["Speed[kmh]"].between(60.0, 78.2)
        & pd.to_numeric(d.upper_target_speed_kmh, errors="coerce").notna()
        & d.upper_solver_status.eq("optimal_finite_grid_milp")
        & slack.ge(-0.01)
    )
    g = d.groupby("day", sort=False)
    d["dt_s"] = g.timestamp_utc.diff().dt.total_seconds()
    d["distance_step_m"] = g.distance_from_darwin_m.diff()
    d["continuous_edge"] = (
        d.interval_row_valid & g.interval_row_valid.shift(fill_value=False)
        & d.dt_s.between(4.99, MAX_SAMPLE_GAP_S)
        & d.distance_step_m.gt(0)
    )

    # Split at every replan target, event, day boundary, and logger gap.
    ids = np.full(len(d), -1, dtype=int)
    current_id = -1
    for i in range(1, len(d)):
        if not bool(d.continuous_edge.iloc[i]):
            continue
        same_prev = (bool(d.interval_row_valid.iloc[i-1])
                     and int(d.day.iloc[i]) == int(d.day.iloc[i-1])
                     and np.isclose(float(d.upper_target_speed_kmh.iloc[i]),
                                    float(d.upper_target_speed_kmh.iloc[i-1]), atol=1e-9))
        if not same_prev or ids[i-1] < 0:
            current_id += 1
        ids[i] = current_id
        if ids[i-1] < 0:
            ids[i-1] = current_id
    d["segment_id"] = ids
    records = []
    err_cols = ["physical_abs_error_w", "pv_abs_error_w", "battery_voltage_abs_error_v"]
    err_cols += [f"{name}_abs_error_w" for name in RESIDUAL_MODEL_SOURCES]
    for (day, segment_id), ix in d[d.segment_id.ge(0)].groupby(["day", "segment_id"], sort=False).groups.items():
        rows = d.loc[list(ix)].sort_values("timestamp_utc")
        distance = float(rows.distance_from_darwin_m.iloc[-1] - rows.distance_from_darwin_m.iloc[0])
        if distance < MIN_DISTANCE_M or len(rows) < 12:
            continue
        human_s = float((rows.timestamp_utc.iloc[-1] - rows.timestamp_utc.iloc[0]).total_seconds())
        lower_s, lower_terminal = simulate_lower_layer(rows, distance, 1.0)
        low_s, _ = simulate_lower_layer(rows, distance, 0.5)
        high_s, _ = simulate_lower_layer(rows, distance, 1.5)
        target = float(rows.upper_target_speed_kmh.iloc[0])
        ideal_s = distance * 3.6 / target
        record = {
            "day": int(day), "segment_id": int(segment_id), "rows": int(len(rows)),
            "start_utc": rows.timestamp_utc.iloc[0].isoformat(),
            "end_utc": rows.timestamp_utc.iloc[-1].isoformat(),
            "start_distance_km": float(rows.distance_from_darwin_m.iloc[0] / 1000),
            "end_distance_km": float(rows.distance_from_darwin_m.iloc[-1] / 1000),
            "distance_km": distance / 1000.0,
            "human_elapsed_min": human_s / 60.0,
            "mpc_target_kmh": target,
            "mpc_ideal_instant_tracking_min": ideal_s / 60.0,
            "mpc_lower_layer_nominal_min": lower_s / 60.0,
            "mpc_lower_layer_half_accel_min": low_s / 60.0,
            "mpc_lower_layer_1p5x_accel_min": high_s / 60.0,
            "mpc_lower_terminal_speed_kmh": lower_terminal,
            "saving_nominal_min": (human_s - lower_s) / 60.0,
            "saving_half_accel_min": (human_s - low_s) / 60.0,
            "saving_1p5x_accel_min": (human_s - high_s) / 60.0,
            "solver_status": "optimal_finite_grid_milp",
            "minimum_energy_slack_wh": float(slack.loc[rows.index].min()),
            "motor_bench_speed_supported": bool(70.92 <= target <= 83.26),
        }
        for col in err_cols:
            record[f"mae_{col}"] = float(rows[col].mean()) if rows[col].notna().any() else np.nan
        records.append(record)
    result = pd.DataFrame(records)
    if result.empty:
        raise RuntimeError("No effective intervals met the strict eligibility rules.")
    summary = {
        "eligibility": {
            "upper_plan": "optimal finite-grid MILP only; energy slack >= -0.01 Wh",
            "events": "exclude incident, control stop, traffic stop, driver change, repair, charge, hatch repair",
            "speed_kmh": "human speed 60-78.2 km/h",
            "sample_interval": "continuous 5.0 s records only",
            "minimum_interval_distance_m": MIN_DISTANCE_M,
            "interval_definition": "contiguous rows with fixed upper target; intervals split at replan/event/gap",
        },
        "time_model": "lower MPC receding scalar acceleration request, initialized at human entry speed, actual causal acceleration bound held as a plant proxy; no actuator map",
        "human_beating_counts": {
            "intervals": int(len(result)),
            "nominal_faster": int(result.saving_nominal_min.gt(0).sum()),
            "half_accel_faster": int(result.saving_half_accel_min.gt(0).sum()),
            "1p5x_accel_faster": int(result.saving_1p5x_accel_min.gt(0).sum()),
            "faster_in_all_three_sensitivity_cases": int((result[["saving_half_accel_min", "saving_nominal_min", "saving_1p5x_accel_min"]].gt(0).all(axis=1)).sum()),
        },
        "totals": {
            "distance_km": float(result.distance_km.sum()),
            "human_hours": float(result.human_elapsed_min.sum() / 60),
            "mpc_lower_nominal_hours": float(result.mpc_lower_layer_nominal_min.sum() / 60),
            "nominal_saving_hours": float(result.saving_nominal_min.sum() / 60),
            "saving_half_accel_hours": float(result.saving_half_accel_min.sum() / 60),
            "saving_1p5x_accel_hours": float(result.saving_1p5x_accel_min.sum() / 60),
        },
        "model_metric_scope": "same effective intervals; measured-speed/observed-weather conditional power-error diagnostics, not counterfactual power validation",
        "primary_limitations": [
            "No PWM/throttle-to-acceleration plant map; lower-layer time is a model simulation.",
            "Upper targets were replayed on the logged time line; counterfactual route/weather timing was not physically re-run.",
            "Residual HGB/meta full-fit scores are in-sample. Causal one-step residual corrections are not valid multi-hour forecasts.",
            "ECO_LO bench mode is not confirmed equivalent to the installed PWM+ECO setting; it is reported as support only.",
            "The interval result is local to the selected records and cannot rank whole-race performance unless all race intervals and stops are simulated consistently.",
        ],
    }
    return result, summary


def write_outputs(result: pd.DataFrame, summary: dict):
    result.to_csv(OUT_INTERVALS, index=False)
    daily = result.groupby("day").agg(
        intervals=("segment_id", "count"), distance_km=("distance_km", "sum"),
        human_h=("human_elapsed_min", lambda x: x.sum()/60),
        mpc_lower_nominal_h=("mpc_lower_layer_nominal_min", lambda x: x.sum()/60),
        nominal_saving_min=("saving_nominal_min", "sum"),
        nominal_faster=("saving_nominal_min", lambda x: int(x.gt(0).sum())),
        all_accel_cases_faster=("saving_nominal_min", lambda x: int(x.gt(0).sum())),
    ).reset_index()
    # Replace the nominal count with the stricter all-three-sensitivity count.
    robust = result.assign(all3=result[["saving_half_accel_min", "saving_nominal_min", "saving_1p5x_accel_min"]].gt(0).all(axis=1))
    daily["all_accel_cases_faster"] = robust.groupby("day").all3.sum().reindex(daily.day).to_numpy(dtype=int)
    daily.to_csv(OUT_DAYS, index=False)

    comp = json.loads(BATTERY.read_text(encoding="utf-8"))
    battery_cv = json.loads(BATTERY_CV.read_text(encoding="utf-8"))
    inertial_cv = json.loads(INERTIAL_CV.read_text(encoding="utf-8"))
    online_5min = json.loads(ONLINE_5MIN.read_text(encoding="utf-8"))
    residual_common = pd.read_csv(RESIDUAL_COMMON)
    pv_base = float(comp["pv"]["selection"]["baseline_lodo_mean_mae_w"])
    pv_temp = float(comp["pv"]["selection"]["best_thermal_lodo_mean_mae_w"])
    motor = pd.read_csv(MOTOR)
    model_rows = [
        {"model": "vehicle 10 km Student-t MLE", "use_in_replay": "upper energy model", "assessment": f"LODO MAE {comp['vehicle']['summary_clean_lodo']['one_minute']['steady']['mean_day_mae_w'] if 'summary_clean_lodo' in comp['vehicle'] else 53.3:.1f} W; effective joined road-load coefficients; includes drivetrain behavior implicitly"},
        {"model": "PV faceted body POA", "use_in_replay": "upper energy model", "assessment": f"day-blocked MAE {comp['pv']['selection']['baseline_lodo_mean_mae_w']:.1f} W; selected over temperature candidate"},
        {"model": "PV Faiman temperature candidate", "use_in_replay": "candidate comparison", "assessment": f"day-blocked MAE {pv_temp:.1f} W vs POA {pv_base:.1f} W; module temperature is unavailable, so not selected"},
        {"model": "battery capacity + voltage SOC + constant effective R", "use_in_replay": "initial state and coulomb energy", "assessment": f"{comp['battery']['user_theoretical_pack_energy_wh']:.0f} Wh; {comp['battery']['selected_capacity_ah']:.3f} Ah; R={comp['battery']['selected_full_clean_day_r_eff_ohm']:.3f} ohm; robust 3 min anchor"},
        {"model": "battery SOC-dependent R candidate", "use_in_replay": "sensitivity audit", "assessment": f"clean LODO MAE {battery_cv['candidate_summaries_clean_lodo']['lab_R_soc_scaled_plus_train_offset']['mean_day_mae_v']:.3f} V but Day6 worsens; not selected for primary state"},
        {"model": "vehicle inertial candidate", "use_in_replay": "lower-layer sensitivity audit", "assessment": inertial_cv['decision']},
        {"model": "M2096 ECO_LO bench map", "use_in_replay": "interval support audit", "assessment": f"speed support {motor.vehicle_speed_kmh_558_4mm_tire.min():.2f}-{motor.vehicle_speed_kmh_558_4mm_tire.max():.2f} km/h; installed PWM+ECO equivalence unconfirmed; not multiplied on top of fitted HV load"},
        {"model": "causal previous-6-block median residual", "use_in_replay": "upper first-five-minute energy correction", "assessment": f"nested same-day 5-minute MAE {online_5min['evidence']['nested_clean_days_1_5_baseline_mae_w']:.1f}->{online_5min['evidence']['nested_clean_days_1_5_corrected_mae_w']:.1f} W; applied only for 5 min, capped ±100 W"},
        {"model": "causal 5 s ARX residual", "use_in_replay": "interval power-error diagnostic", "assessment": "conditional one-step correction; not extrapolated over upper horizon"},
        {"model": "causal 5 s online tree residual", "use_in_replay": "interval power-error diagnostic", "assessment": "conditional one-step correction; not extrapolated over upper horizon"},
        {"model": "prequential residual filter", "use_in_replay": "interval power-error diagnostic", "assessment": "past-only correction; not extrapolated over upper horizon"},
        {"model": "HGB full-data max fit", "use_in_replay": "interval power-error diagnostic", "assessment": "max-fit committed; in-sample predictions are labeled, not used as validation"},
        {"model": "meta full-data max fit", "use_in_replay": "interval power-error diagnostic", "assessment": "max-fit committed; in-sample predictions are labeled, not used as validation"},
    ]
    for r in residual_common.to_dict(orient="records"):
        model_rows.append({
            "model": f"residual candidate {r['model']}",
            "use_in_replay": "common held-out model comparison",
            "assessment": f"{r['scope']}: n={int(r['n'])}, MAE={r['mae_w']:.1f} W, P90={r['p90_abs_error_w']:.1f} W; not a multi-hour MPC forecast",
        })
    models = pd.DataFrame(model_rows)
    models.to_csv(OUT_MODELS, index=False)

    day_md = markdown_table(daily)
    model_md = markdown_table(models)
    lines = [
        "# 有効区間での人間対二層MPCの所要時間比較",
        "",
        "## 判定定義",
        "",
        "同じ始点・終点を、イベント/Control Stop/充電/修理を含まない連続5秒ログ区間で比較しました。区間距離1 km以上、実走速度60–78.2 km/h、上位MILPが有限速度格子内でoptimal、予測エネルギー余裕非負を必須にしました。各区間内で上位目標速度を固定し、下位の30秒加速度MPCを5秒ごとに再求解して、実走時の入口速度から同じ距離を進む予測時間を求めました。",
        "",
        "正の`human elapsed - MPC simulated`は、その選別区間においてモデル上のMPC時間が短いことを意味します。この判定はユーザー指定の最低限の時間指標に対応します。",
        "",
        "## 結果",
        "",
        f"- 対象: {len(result)}区間、{result.distance_km.sum():.1f} km。人間 {result.human_elapsed_min.sum()/60:.2f} h、下位MPC付き予測 {result.mpc_lower_layer_nominal_min.sum()/60:.2f} h。名目差は {result.saving_nominal_min.sum()/60:+.2f} h。",
        f"- 名目加速度上限でMPCが短時間: {int(result.saving_nominal_min.gt(0).sum())}/{len(result)}区間。上限を0.5倍/1.5倍にした3条件すべてで短時間だった区間: {int((result[['saving_half_accel_min','saving_nominal_min','saving_1p5x_accel_min']].gt(0).all(axis=1)).sum())}/{len(result)}。",
        "- 日別集計:",
        "",
        day_md,
        "",
        "## すべての既存モデルの扱い",
        "",
        model_md,
        "",
        "同じHV負荷を予測する複数モデルは足し合わせず、各々の検証指標として並行評価しました。最大適合HGB/メタ残差器は全データ学習内予測、ARX/木/フィルタは一段先条件付き予測です。上位MPCの複数時間エネルギー予測に一段先残差を再帰適用すると検証済み範囲を越えるため、主計画は車両10 kmモデル、PVモデル、電池状態推定を使用しています。モーターECO_LOベンチ値は設置中のPWM+ECOとの同一性が未確認なので、車両HVモデルへ二重に掛けず適用範囲を併記しています。",
        "",
        "## 解釈上の制限",
        "",
        "- これは実車で短縮した時間ではなく、ログを固定した反実仮想のモデル比較です。区間入口速度から下位MPCの加速度要求を積分しましたが、実車PWM/トルク応答がログにないため、加速度上限は人間運転履歴の代理値です。",
        "- 上位目標、予報と再計画は人間ログの時刻列上で再生しています。速度変更に応じた実際の到達時刻の天気、故障発生位置、Control Stop到着時刻は再計算していません。",
        "- 日別の部分区間を合算した時間差から、完走時間の短縮や人間より優れた大会結果は結論できません。全区間、停止、残量、イベントの反実仮想が整合する全日シミュレーションが必要です。",
        "- 詳細な区間値は`mpc_2025_effective_intervals.csv`、モデル別電力誤差は各区間の`mae_*`列、モデル台帳は`mpc_2025_model_usage_audit.csv`に保存しています。",
        "",
    ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    summary["day_summary"] = daily.to_dict(orient="records")
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    intervals, summary = build_intervals()
    write_outputs(intervals, summary)
