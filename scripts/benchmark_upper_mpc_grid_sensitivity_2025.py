#!/usr/bin/env python3
"""Benchmark upper-MPC grid/space choices at one as-of state per race day.

This is a reproducible parameter sensitivity study, not a closed-loop race
counterfactual: it re-solves the existing as-of upper MILP at the first
eligible decision of each day and reports solver/plan metrics.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replay_hierarchical_mpc_2025 as replay  # noqa: E402

OUTPUT = ROOT / "reports/mpc_upper_grid_sensitivity_2025.csv"
SUMMARY = ROOT / "reports/mpc_upper_grid_sensitivity_2025.json"
CADENCE_COUNTS = ROOT / "reports/mpc_upper_cadence_counts_2025.csv"
SPEED_STEPS_KMH = (0.5, 1.0, 2.0)
STAGE_LENGTHS_KM = (3.0, 5.0, 10.0, 20.0)
RESERVE_PCT = 0.15


def main() -> None:
    replay_rows = pd.read_csv(replay.OUTPUT, low_memory=False,
                              parse_dates=["timestamp_utc", "timestamp_local"])
    race, route, events, road, pv, socmap, component = replay.load_inputs()
    day_end = replay.daily_endpoints(events, float(route.distance_m.max()))
    stops = {
        day: sorted(pd.to_numeric(events[
            (pd.to_numeric(events.day, errors="coerce") == day)
            & events.event_type.eq("control_stop")].distance_m,
            errors="coerce").dropna().astype(float).tolist())
        for day in range(1, 7)
    }
    energy, _ = replay.actual_soc_wh(race, socmap, component)
    race["energy_for_solve_wh"] = energy
    decision_rows = replay_rows[replay_rows.upper_replan_now].groupby("day", sort=True).head(1)
    cadence_rows: list[dict] = []
    trouble_pattern = r"incident:|control_stop:|traffic_stop:|driver_change:|repair:|charge:|hatch_repair:"
    trouble_re = re.compile(trouble_pattern)
    for period_min in (5, 10, 15, 30, 60):
        counts = {}
        for day, group in replay_rows.groupby("day", sort=True):
            last_bucket = None
            count = 0
            for _, local_time, phase, flags, speed in group[
                    ["day", "timestamp_local", "event_phase", "event_flags", "Speed[kmh]"]
                    ].itertuples(index=False, name=None):
                road_allowed = phase == "road_day" and not bool(trouble_re.search(str(flags or "")))
                if road_allowed and float(speed or 0.0) >= 5 and local_time.hour < 17:
                    bucket = (local_time.date(), local_time.hour, local_time.minute // period_min)
                    if bucket != last_bucket:
                        count += 1
                        last_bucket = bucket
                elif not road_allowed or local_time.hour >= 17:
                    last_bucket = None
            counts[int(day)] = count
        cadence_rows.append({"upper_replan_interval_min": period_min,
                             "total_as_if_live_solves": sum(counts.values()),
                             "by_day": json.dumps(counts, separators=(",", ":"))})
    pd.DataFrame(cadence_rows).to_csv(CADENCE_COUNTS, index=False, encoding="utf-8-sig")
    forecast_cache: dict[pd.Timestamp, replay.Forecast] = {}
    raw_runs: dict[pd.Timestamp, pd.DataFrame] = {}
    route_nodes = replay.route_nodes(route)
    original_speeds = replay.SPEEDS.copy()
    original_stage = replay.HORIZON_M
    records: list[dict] = []
    try:
        for stage_km in STAGE_LENGTHS_KM:
            replay.HORIZON_M = stage_km * 1000.0
            for speed_step in SPEED_STEPS_KMH:
                replay.SPEEDS = np.arange(60.0, 78.0 + speed_step * 0.25, speed_step)
                setting = f"{stage_km:g}km_stage_{speed_step:g}kmh_grid"
                print(f"setting {setting}", flush=True)
                for _, decision in decision_rows.iterrows():
                    day = int(decision.day)
                    now = pd.Timestamp(decision.timestamp_utc)
                    run = pd.Timestamp(decision.forecast_run_init_utc)
                    if run not in forecast_cache:
                        raw_runs[run] = replay.fetch_run(run, route, route_nodes)
                        forecast_cache[run] = replay.Forecast(raw_runs[run])
                    fc = forecast_cache[run]
                    local = pd.Timestamp(decision.timestamp_local)
                    deadline = local.normalize() + pd.Timedelta(hours=17)
                    idx = race.index[(race.day.astype(int) == day)
                                     & (race.timestamp_utc == now)]
                    if not len(idx):
                        idx = race.index[(race.day.astype(int) == day)
                                         & (race.timestamp_utc >= now)].take([0])
                    row = race.loc[idx[0]]
                    t0 = time.perf_counter()
                    plan = replay.solve_upper(
                        fc, float(decision.distance_from_darwin_m), day_end[day],
                        stops[day], now, float(row.energy_for_solve_wh), RESERVE_PCT,
                        route, road, pv, deadline, 0.0)
                    elapsed = time.perf_counter() - t0
                    actual_remaining = float((replay_rows[
                        (replay_rows.day == day) & (replay_rows.event_phase == "road_day")
                        & (replay_rows["Speed[kmh]"] >= 5)].timestamp_utc.max() - now
                    ).total_seconds() / 3600.0)
                    records.append({
                        "day": day, "decision_timestamp_utc": now.isoformat(),
                        "current_distance_m": float(decision.distance_from_darwin_m),
                        "initial_energy_estimate_wh": float(row.energy_for_solve_wh),
                        "reserve_pct": 100.0 * RESERVE_PCT,
                        "stage_length_km": stage_km, "speed_grid_step_kmh": speed_step,
                        "speed_candidates": int(len(replay.SPEEDS)),
                        "solver_status": plan["status"], "mip_gap": plan.get("mip_gap"),
                        "first_stage_target_kmh": plan.get("target_kmh"),
                        "predicted_remaining_time_h": plan.get("predicted_remaining_h"),
                        "predicted_remaining_energy_wh": plan.get("predicted_remaining_wh_robust"),
                        "stage_count": plan.get("stage_count"),
                        "solve_wall_s": elapsed,
                        "logged_human_time_remaining_h": actual_remaining,
                        "model_time_delta_vs_logged_h": (actual_remaining - float(plan["predicted_remaining_h"])
                            if plan.get("predicted_remaining_h") is not None else np.nan),
                        "metric_scope": "single first-decision re-solve; not a closed-loop human-beating result",
                    })
    finally:
        replay.SPEEDS = original_speeds
        replay.HORIZON_M = original_stage

    result = pd.DataFrame(records)
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    summary_rows = []
    for (stage_km, speed_step), group in result.groupby(["stage_length_km", "speed_grid_step_kmh"]):
        summary_rows.append({
            "stage_length_km": float(stage_km), "speed_grid_step_kmh": float(speed_step),
            "decision_states": int(len(group)),
            "optimal": int(group.solver_status.eq("optimal_finite_grid_milp").sum()),
            "time_limited_feasible": int(group.solver_status.eq("feasible_time_limited_milp_not_proven_optimal").sum()),
            "no_advisory": int((~group.solver_status.isin({"optimal_finite_grid_milp", "feasible_time_limited_milp_not_proven_optimal"})).sum()),
            "median_solver_wall_s": float(group.solve_wall_s.median()),
            "p95_solver_wall_s": float(group.solve_wall_s.quantile(.95)),
            "median_predicted_remaining_h": float(group.predicted_remaining_time_h.median()),
            "mean_model_time_delta_vs_logged_h": float(group.model_time_delta_vs_logged_h.mean()),
            "day_results": group[["day", "solver_status", "first_stage_target_kmh",
                                  "predicted_remaining_time_h", "predicted_remaining_energy_wh",
                                  "solve_wall_s"]].to_dict("records"),
        })
    metadata = {
        "source": str(replay.OUTPUT.relative_to(ROOT)),
        "sample": "first eligible upper-MPC decision from each BWSC2025 day (6 states)",
        "reserve_pct_fixed": 100.0 * RESERVE_PCT,
        "upper_grid_steps_kmh": list(SPEED_STEPS_KMH),
        "stage_lengths_km": list(STAGE_LENGTHS_KM),
        "time_limit_s_per_milp": 2.0,
        "interpretation": [
            "This is parameter sensitivity on six as-of states, not cross-validated closed-loop race performance.",
            "The race-day initial energy estimate and all fitted model parameters are uncertain.",
            "A lower predicted remaining time is a model objective value, not measured time savings.",
            "Day 6 human time ends at the logged wind-related retirement, while the MILP plans to route end; those times are not comparable.",
        ],
        "summary": summary_rows,
    }
    SUMMARY.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT}", flush=True)
    print(f"wrote {SUMMARY}", flush=True)
    print(f"wrote {CADENCE_COUNTS}", flush=True)


if __name__ == "__main__":
    main()
