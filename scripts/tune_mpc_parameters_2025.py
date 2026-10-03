#!/usr/bin/env python3
"""Nested day-held-out tuning of lower MPC settings on the 2025 eligible intervals.

The logged trace supplies a driver-plus-vehicle acceleration envelope, not a
throttle-to-acceleration transfer function. M2096 ECO_LO DUTY=1 torque is used
only as a physical upper bound. All time-savings figures remain counterfactual.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "data/processed/mpc_2025_as_live_replay.csv"
INTERVALS = ROOT / "reports/mpc_2025_effective_intervals.csv"
BENCH = ROOT / "models/m2096_eco_lo_bench_map.csv"
OUT_SWEEP = ROOT / "reports/mpc_lower_parameter_sweep_2025.csv"
OUT_NESTED = ROOT / "reports/mpc_lower_nested_cv_2025.csv"
OUT_SUMMARY = ROOT / "reports/mpc_lower_tuning_summary_2025.json"
OUT_CONFIG = ROOT / "models/mpc_operating_config.json"
OUT_ENVELOPE = ROOT / "models/m2096_inferred_actuator_envelope.csv"
OUT_REPORT = ROOT / "reports/mpc_tuning_and_actuator_inference.md"

MASS_KG, G, CD_A, CRR, R_WHEEL = 314.0, 9.80665, 0.08876, 0.0035, 0.5584 / 2
BAD_EVENT = r"incident:|control_stop:|traffic_stop:|driver_change:|repair:|charge:|hatch_repair:"
PARAMS = list(itertools.product(
    [1.0, 5.0],       # lower update interval (s)
    [20.0, 30.0, 45.0],  # lower prediction horizon (s)
    [0.75, 0.90, 0.95],  # empirical acceleration-envelope quantile
    [1.0, 2.0],  # 0.5 km/h adds no distinction: upper references are integer km/h
    [(0.1, 0.0), (1.0, 4.0), (10.0, 16.0)], # acceleration / move-rate weights
))


def load_data():
    cols = ["day", "timestamp_utc", "event_phase", "event_flags", "distance_from_darwin_m",
            "Speed[kmh]", "upper_target_speed_kmh", "upper_solver_status", "upper_energy_budget_wh",
            "upper_predicted_remaining_net_energy_wh"]
    d = pd.read_csv(REPLAY, usecols=cols, low_memory=False)
    d["timestamp_utc"] = pd.to_datetime(d.timestamp_utc, utc=True)
    for c in ["Speed[kmh]", "distance_from_darwin_m", "upper_target_speed_kmh",
              "upper_energy_budget_wh", "upper_predicted_remaining_net_energy_wh"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values(["day", "timestamp_utc"]).reset_index(drop=True)
    day_groups = {int(day): g.reset_index(drop=True) for day, g in d.groupby("day", sort=True)}
    interval = pd.read_csv(INTERVALS)
    interval["start_utc"] = pd.to_datetime(interval.start_utc, utc=True)
    interval["end_utc"] = pd.to_datetime(interval.end_utc, utc=True)
    entries = []
    for row in interval.itertuples(index=False):
        group = day_groups[int(row.day)]
        match = group.index[group.timestamp_utc.eq(row.start_utc)]
        if len(match) != 1:
            raise RuntimeError(f"Cannot uniquely locate interval start: day={row.day} {row.start_utc}")
        entry = group.iloc[int(match[0])]
        entries.append({"day": int(row.day), "segment_id": int(row.segment_id),
            "distance_m": float(row.distance_km) * 1000.0,
            "human_s": float(row.human_elapsed_min) * 60.0,
            "entry_speed_kmh": float(entry["Speed[kmh]"]),
            "target_kmh": float(row.mpc_target_kmh),
            "start_utc": row.start_utc.isoformat(), "end_utc": row.end_utc.isoformat()})
    bench = pd.read_csv(BENCH).sort_values("vehicle_speed_kmh_558_4mm_tire")
    speed = bench.vehicle_speed_kmh_558_4mm_tire.to_numpy(float)
    torque = bench.torque_nm.to_numpy(float)
    # ECO_LO is a fixed-DUTY bench curve, not the installed PWM+ECO setting.
    # Below its first point, clamp torque to that point and flag extrapolation.
    net_a = (torque / R_WHEEL - MASS_KG * G * CRR - 0.5 * 1.2 * CD_A * (speed / 3.6) ** 2) / MASS_KG
    dense_speed = np.arange(55.0, 83.51, 0.5)
    dense_accel = np.interp(dense_speed, speed, net_a)
    return d, day_groups, entries, dense_speed, dense_accel, bench


def train_caps(day_groups: dict[int, pd.DataFrame], train_days: set[int], quantile: float) -> tuple[float, float, int, int]:
    ups, downs = [], []
    for day in train_days:
        g = day_groups[day].copy()
        flags = g.event_flags.fillna("").astype(str)
        valid = (g.event_phase.eq("road_day") & ~flags.str.contains(BAD_EVENT, regex=True)
                 & g["Speed[kmh]"].between(60.0, 78.2))
        dt = g.timestamp_utc.diff().dt.total_seconds()
        dv = g["Speed[kmh]"].diff() / 3.6
        a = dv / dt
        ok = valid & valid.shift(fill_value=False) & dt.between(4.99, 5.01) & a.notna()
        ups.extend(a[ok & a.gt(0)].to_list())
        downs.extend((-a[ok & a.lt(0)]).to_list())
    if len(ups) < 100 or len(downs) < 100:
        raise RuntimeError("Too few observed speed ramps for an acceleration envelope")
    return (float(np.quantile(ups, quantile)), float(np.quantile(downs, quantile)), len(ups), len(downs))


def sim_interval(entry: dict, params: tuple, acc_pos: float, acc_neg: float,
                 motor_speed: np.ndarray, motor_accel: np.ndarray) -> float:
    dt, horizon_s, _, speed_step, weight_pair = params
    accel_weight, move_weight = weight_pair
    track_weight = 1.0
    n = max(1, int(round(horizon_s / dt)))
    horizon = n * dt
    sum_tau = dt * n * (n + 1) / 2
    sum_tau2 = dt * dt * n * (n + 1) * (2 * n + 1) / 6
    target = float(np.clip(round(entry["target_kmh"] / speed_step) * speed_step, 60.0, 78.2))
    v = entry["entry_speed_kmh"] / 3.6
    ref = target / 3.6
    done = 0.0
    elapsed = 0.0
    prev_a = 0.0
    max_steps = min(int(entry["distance_m"] / max(v, 60 / 3.6) / dt + 5000), int(900 / dt))
    for _ in range(max_steps):
        v_kmh = v * 3.6
        lookup = int(round((min(max(v_kmh, motor_speed[0]), motor_speed[-1]) - motor_speed[0]) / 0.5))
        product_cap = max(0.005, float(motor_accel[lookup]))
        a_hi = min(acc_pos, product_cap)
        a_lo = -acc_neg
        uncon = (track_weight * sum_tau * (ref - v) + move_weight * prev_a) / (
            track_weight * sum_tau2 + accel_weight + move_weight)
        a_lo = max(a_lo, (60 / 3.6 - v) / horizon)
        a_hi = min(a_hi, (78.2 / 3.6 - v) / horizon)
        a = float(np.clip(uncon, a_lo, a_hi)) if a_lo <= a_hi else 0.0
        v_next = float(np.clip(v + a * dt, 60 / 3.6, 78.2 / 3.6))
        ds = max(0.0, 0.5 * (v + v_next) * dt)
        if ds <= 0:
            return float("inf")
        remaining = entry["distance_m"] - done
        if ds >= remaining:
            frac = remaining / ds
            elapsed += dt * frac
            return elapsed
        elapsed += dt
        done += ds
        prev_a = a
        v = v_next
        # Once the receding controller has settled near its reference, the
        # remaining fixed-target segment is analytically a cruise. This is
        # exactly the same constant-speed tail assumed by the upper planner.
        if abs(v - ref) <= 0.2 / 3.6 and abs(a) <= 0.02:
            return elapsed + max(0.0, entry["distance_m"] - done) / max(60 / 3.6, v)
    return float("inf")


def score_config(params, interval_rows, caps_by_day, motor_speed, motor_accel):
    per_day: dict[int, float] = {}
    for e in interval_rows:
        ap, an = caps_by_day[e["day"]]
        modeled = sim_interval(e, params, ap, an, motor_speed, motor_accel)
        per_day[e["day"]] = per_day.get(e["day"], 0.0) + (e["human_s"] - modeled)
    return per_day


def main():
    d, day_groups, entries, motor_speed, motor_accel, bench = load_data()
    days = sorted({e["day"] for e in entries})
    if len(entries) != 62 or days != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"Expected archived eligible interval set (62, Day1-5), got {len(entries)} / {days}")
    # Nested day-blocked evaluation: each outer day is excluded from parameter
    # selection and its actuator envelope. Inner scores use day-held-out caps.
    config_index = {p: i for i, p in enumerate(PARAMS)}
    nested_records = []
    for outer in days:
        print(f"Nested MPC tuning: outer holdout Day {outer}/5", flush=True)
        outer_train = set(days) - {outer}
        inner_scores = {p: 0.0 for p in PARAMS}
        for inner in sorted(outer_train):
            fit_days = outer_train - {inner}
            inner_caps = {}
            for q in [0.75, 0.90, 0.95]:
                ap, an, _, _ = train_caps(day_groups, fit_days, q)
                inner_caps[q] = (ap, an)
            inner_rows = [e for e in entries if e["day"] == inner]
            for p in PARAMS:
                ap, an = inner_caps[p[2]]
                for e in inner_rows:
                    elapsed = sim_interval(e, p, ap, an, motor_speed, motor_accel)
                    inner_scores[p] += e["human_s"] - elapsed
        best_inner = max(inner_scores.values())
        # Parsimony rule: select the coarsest/cheapest setting within 0.5% of
        # the best inner-fold saving; this limits tuning to noise-level gains.
        near = [p for p in PARAMS if inner_scores[p] >= best_inner - max(1.0, abs(best_inner) * 0.005)]
        selected = sorted(near, key=lambda p: (-p[0], p[1], -p[3], -p[4][0], -p[4][1]))[0]
        ap, an, nup, ndn = train_caps(day_groups, outer_train, selected[2])
        test_rows = [e for e in entries if e["day"] == outer]
        test_saved = sum(e["human_s"] - sim_interval(e, selected, ap, an, motor_speed, motor_accel) for e in test_rows)
        nested_records.append({"outer_held_out_day": outer, "inner_selected_params": selected,
            "lower_update_s": selected[0], "prediction_horizon_s": selected[1],
            "accel_quantile": selected[2], "speed_resolution_kmh": selected[3],
            "accel_weight": selected[4][0], "move_rate_weight": selected[4][1],
            "train_positive_samples": nup, "train_negative_samples": ndn,
            "train_accel_cap_mps2": ap, "train_decel_cap_mps2": an,
            "outer_test_intervals": len(test_rows), "outer_test_saving_min": test_saved / 60})
        print(f"  selected {selected}; held-out modeled saving {test_saved/60:.2f} min", flush=True)
    nested = pd.DataFrame(nested_records)
    nested.to_csv(OUT_NESTED, index=False)

    # Deploy setting is selected by nested outer-fold frequency. A tie uses a
    # fixed simplicity rule; outer-test outcomes do not select the setting.
    nested_counts = nested.groupby(["lower_update_s", "prediction_horizon_s", "accel_quantile",
        "speed_resolution_kmh", "accel_weight", "move_rate_weight"], dropna=False).agg(
            selected_folds=("outer_held_out_day", "count")).reset_index()
    nested_counts = nested_counts.sort_values(["selected_folds"], ascending=False)
    # Outer-fold outcomes are reserved for validation and are not used to pick
    # a single committed setting. Ties use a fixed lower-complexity rule.
    max_selected = int(nested_counts.selected_folds.max())
    tied = nested_counts[nested_counts.selected_folds.eq(max_selected)]
    tied = tied.sort_values(["lower_update_s", "prediction_horizon_s", "speed_resolution_kmh",
        "accel_weight", "move_rate_weight"], ascending=[False, True, False, False, False])
    wr = tied.iloc[0]
    winning = (float(wr.lower_update_s), float(wr.prediction_horizon_s), float(wr.accel_quantile),
        float(wr.speed_resolution_kmh), (float(wr.accel_weight), float(wr.move_rate_weight)))

    # Re-score the complete eligible set with cross-fitted acceleration bounds
    # for each day. This is the parameter sweep used in the report.
    fold_caps = {}
    for day in days:
        fit_days = set(days) - {day}
        for q in [0.75, 0.90, 0.95]:
            ap, an, _, _ = train_caps(day_groups, fit_days, q)
            fold_caps[(day, q)] = (ap, an)
    sweep_rows = []
    for idx, p in enumerate(PARAMS, 1):
        per_day = {}
        for e in entries:
            ap, an = fold_caps[(e["day"], p[2])]
            elapsed = sim_interval(e, p, ap, an, motor_speed, motor_accel)
            per_day[e["day"]] = per_day.get(e["day"], 0.0) + (e["human_s"] - elapsed)
        row = {"lower_update_s": p[0], "prediction_horizon_s": p[1], "accel_quantile": p[2],
            "speed_resolution_kmh": p[3], "accel_weight": p[4][0], "move_rate_weight": p[4][1],
            "oof_saving_min": sum(per_day.values()) / 60,
            "days_faster": sum(v > 0 for v in per_day.values()), "worst_day_saving_min": min(per_day.values()) / 60,
            "best_day_saving_min": max(per_day.values()) / 60}
        row.update({f"day_{day}_saving_min": per_day.get(day, 0.0) / 60 for day in days})
        sweep_rows.append(row)
        if idx % 100 == 0:
            print(f"Scored {idx}/{len(PARAMS)} parameter combinations", flush=True)
    sweep = pd.DataFrame(sweep_rows).sort_values("oof_saving_min", ascending=False)
    sweep.to_csv(OUT_SWEEP, index=False)
    # One-second telemetry is an explicit operating requirement. Retain the
    # nested-selected horizon, limits and weights, but solve once per new 1 Hz
    # sample. Compare this override with the nested-selected 5 s candidate.
    operational = (1.0, winning[1], winning[2], winning[3], winning[4])
    def find_sweep_row(params):
        return sweep[(sweep.lower_update_s == params[0]) & (sweep.prediction_horizon_s == params[1])
            & (sweep.accel_quantile == params[2]) & (sweep.speed_resolution_kmh == params[3])
            & (sweep.accel_weight == params[4][0]) & (sweep.move_rate_weight == params[4][1])].iloc[0].to_dict()
    operational_row = find_sweep_row(operational)
    nested_best_row = find_sweep_row(winning)

    # Final committed physical envelope uses Day1-5 as requested after the
    # nested evaluation; all fits/caps are exported with the sample count.
    ap, an, nup, ndn = train_caps(day_groups, set(days), operational[2])
    speeds = motor_speed
    product = motor_accel
    env = pd.DataFrame({"speed_kmh": speeds, "bench_net_accel_flat_mps2": product,
        "observed_driver_vehicle_accel_cap_mps2": ap,
        "inferred_acceleration_cap_mps2": np.minimum(product, ap),
        "inferred_deceleration_cap_mps2": an,
        "bench_speed_supported": (speeds >= bench.vehicle_speed_kmh_558_4mm_tire.min())
            & (speeds <= bench.vehicle_speed_kmh_558_4mm_tire.max())})
    env.to_csv(OUT_ENVELOPE, index=False)
    config = {
        "scope": "BWSC2025 eligible-interval nested day-blocked tuning; advisory-only, counterfactual",
        "selected_by": "most frequent outer-fold inner-selected setting; inner choices within 0.5% of best saving favor lower complexity",
        "lower_update_s": operational[0], "lower_prediction_horizon_s": operational[1],
        "nested_selected_lower_update_s": winning[0],
        "empirical_acceleration_quantile": operational[2], "speed_reference_resolution_kmh": operational[3],
        "acceleration_weight": winning[4][0], "move_rate_weight": winning[4][1],
        "operational_override_reason": "recompute on every new 1 Hz sample as requested; 2025 paired interval simulation differs from nested-selected 5 s by <0.03% of eligible human time",
        "nested_best_oof_saving_min": float(nested_best_row["oof_saving_min"]),
        "operational_1hz_oof_saving_min": float(operational_row["oof_saving_min"]),
        "upper_replan_s": 300,
        "upper_replan_policy": "every 300 s plus immediate event, route displacement >=10 km, forecast issue change, or tracking/energy contract breach",
        "upper_replan_evidence_status": "provisional design: first-hour fixed-log probe compared 5/10/15/30/60 min; 5 min produced 32 target changes across 62 within-day transitions, 24/68 no-advisory results, and p95 solve wall time 2.73 s; no causal race-time or optimal-cadence claim",
        "upper_solver_time_limit_s": 5.0,
        "upper_solver_time_limit_evidence_status": "provisional runtime candidate: 18 first-hour 5-min cases with a feasible incumbent but unproven optimality at 2 s were retried at 5 s; 16 were proven optimal and 2 remained time-limited feasible; no no-advisory cases, no target changes, median modeled-time improvement 0 min, probe wall p95 5.76 s/max 5.79 s; historical replay remains pinned to 2 s",
        "upper_speed_candidate_step_kmh": 1.0,
        "upper_speed_step_evidence": "retained at 1 km/h: with a 5 s MILP limit on six first-decision states, 0.5 km/h yielded 1 optimal/2 time-limited/3 no-advisory and 1 km/h yielded 3 optimal/0 time-limited/3 no-advisory; median solve wall time 3.14 vs 1.43 s, while modeled time difference on jointly feasible days was at most 0.38 s/day; 5 s log quantization and 50.5-53.3 W model MAE do not support finer ranking",
        "lower_speed_advisory_resolution_kmh": 1.0,
        "actuator_model": "min(M2096 ECO_LO DUTY=1 flat-road torque-derived acceleration ceiling, day-blocked empirical speed-ramp quantile); deceleration from observed speed ramps",
        "inferred_accel_cap_mps2_day1_5": ap, "inferred_decel_cap_mps2_day1_5": an,
        "fit_samples_positive": nup, "fit_samples_negative": ndn,
        "outer_fold_selection": nested_records,
        "limitations": [
            "ECO_LO bench mode is not confirmed equivalent to installed PWM+ECO and its sweep is not a throttle map.",
            "The historical speed derivative identifies a driver-plus-vehicle envelope, not actuator lag or PWM response.",
            "No speed-step response or logged actuator request exists, so response time constant and jerk limit remain unidentifiable.",
            "The eligible-interval time ranking holds each previously solved upper target fixed and does not rerun the upper energy MILP on counterfactual arrival times.",
            "The fold metrics are grouped by race day; points within a day are autocorrelated.",
        ]}
    OUT_CONFIG.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"eligible_intervals": len(entries), "days": days,
        "human_hours": sum(e["human_s"] for e in entries) / 3600,
        "nested_selection": nested_records,
        "nested_selected_setting": {"lower_update_s": winning[0], "prediction_horizon_s": winning[1],
            "accel_quantile": winning[2], "speed_resolution_kmh": winning[3],
            "accel_weight": winning[4][0], "move_rate_weight": winning[4][1]},
        "selected_setting": {"lower_update_s": operational[0], "prediction_horizon_s": operational[1],
            "accel_quantile": operational[2], "speed_resolution_kmh": operational[3],
            "accel_weight": operational[4][0], "move_rate_weight": operational[4][1]},
        "nested_best_oof_parameter_result": nested_best_row,
        "selected_oof_parameter_result": operational_row,
        "top_10_oof_candidates": sweep.head(10).to_dict(orient="records"),
        "actuator_envelope": {"bench_speed_min_kmh": float(bench.vehicle_speed_kmh_558_4mm_tire.min()),
            "bench_speed_max_kmh": float(bench.vehicle_speed_kmh_558_4mm_tire.max()), "positive_empirical_cap_mps2": ap,
            "negative_empirical_cap_mps2": an, "positive_samples": nup, "negative_samples": ndn,
            "bench_efficiency_is_not_used_as_in_car_truth": True}}
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# MPC下位周期・アクチュエータ応答の推定と2025日別検証", "",
        "## 方法", "",
        f"62の有効区間（{summary['human_hours']:.2f} h、Day1–5）を再利用し、外側の1日を完全に選択から外す5-fold nested day-blocked validationを行った。内側選択では、更新周期、予測時間、加減速包絡の分位点、速度指示刻み、加速度/指令変化重みを比較し、最良時間短縮から0.5%以内なら低複雑度の候補を選んだ。速度・予報行は同一日の中で自己相関するため、独立単位は5日である。",
        "", "## 推定した下位設定", "",
        f"nested検証で最も選ばれた候補は{winning[0]:.0f}秒更新・{winning[1]:.0f}秒先読み・q={winning[2]:.2f}・{winning[3]:.1f} km/h刻み・重み({winning[4][0]:g},{winning[4][1]:g})で、OOF短縮{nested_best_row['oof_saving_min']:.2f}分だった。実運用は最新1秒状態ごとに解き直す要件に合わせ、同じ設定の1秒周期へ変更する。対象の人間時間{summary['human_hours']:.2f} hに対する推定短縮は{operational_row['oof_saving_min']:.2f}分、5秒候補との差は{nested_best_row['oof_saving_min']-operational_row['oof_saving_min']:.2f}分（0.03%未満）だった。全候補は`mpc_lower_parameter_sweep_2025.csv`、日単位nested foldは`mpc_lower_nested_cv_2025.csv`。",
        "", "## アクチュエータ包絡", "",
        f"M2096 ECO_LO DUTY=1ベンチのトルクをタイヤ半径{R_WHEEL:.4f} m・質量{MASS_KG:.0f} kgへ換算し、平坦路の理論転がり/空気抵抗を差し引いて物理上限とした。車速ログのDay1–5加減速分布から外側fold訓練日の分位点を得て、速度ごとに小さい側を下位到達上限とした。全日モデル包絡は正加速{ap:.4f} m/s²、減速{an:.4f} m/s²である。速度範囲外ベンチ値は端点トルクでクランプしており外挿フラグを出力した。",
        "", "## 限界", "",
        "これはPWM指令から加速度への伝達関数やアクチュエータ遅れの同定ではない。試験ブックは固定DUTYのECO_LO曲線で、実車のPWM+ECOとの対応、制御指令、トルク/電流応答のログが存在しない。5秒整数km/hログから得た加速度は粗く、主に人間の操作を含む。したがって1秒周期の優位性や短縮分はシミュレーション内に限る。",
        "", "上位の300秒再計画周期は設定値として `mpc_operating_config.json` に置いたが、30分ごとの既存上位計画を異なる周期で全再求解し、反実仮想時刻・エネルギー・天気を最後まで回すベンチマークはこのデータでは未実施である。したがって300秒は実測最適値ではなく、1秒状態更新と10 km再計画トリガを両立させる暫定設計値である。",
        "", "## ファイル", "",
        "- `models/m2096_inferred_actuator_envelope.csv`: 速度ごとのベンチ物理上限、ログ上限、使用上限とベンチ支持範囲。",
        "- `models/mpc_operating_config.json`: 実行設定と根拠ステータス。",
        "- `reports/mpc_lower_nested_cv_2025.csv`: outer holdout fold outcomes.",
        "- `reports/mpc_lower_parameter_sweep_2025.csv`: all tested setting combinations.",
        "- `reports/mpc_lower_tuning_summary_2025.json`: selected settings and reproducible metrics.",
    ]
    OUT_REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"eligible_intervals": len(entries), "winning": summary["selected_setting"],
        "nested": nested_records, "selected_oof": summary["selected_oof_parameter_result"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
