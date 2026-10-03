#!/usr/bin/env python3
"""Re-solve first-hour 5-minute cases that were time-limited at 2 s, with 5 s."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replay_hierarchical_mpc_2025 as replay  # noqa: E402

INPUT = ROOT / "reports/mpc_upper_replan_period_probe_2025.csv"
OUTPUT = ROOT / "reports/mpc_upper_solver_budget_probe_2025.csv"
SUMMARY = ROOT / "reports/mpc_upper_solver_budget_probe_2025.json"
TIME_LIMIT_S = 5.0


def main() -> None:
    base = pd.read_csv(INPUT, low_memory=False, parse_dates=["decision_timestamp_utc"])
    base = base[(base.period_min == 5)
                & base.status.eq("feasible_time_limited_milp_not_proven_optimal")].copy()
    log = pd.read_csv(replay.OUTPUT, low_memory=False,
                      parse_dates=["timestamp_utc", "timestamp_local"])
    race, route, events, road, pv, _, _ = replay.load_inputs()
    endpoints = replay.daily_endpoints(events, float(route.distance_m.max()))
    control_stops = {
        day: sorted(pd.to_numeric(events[
            (pd.to_numeric(events.day, errors="coerce") == day)
            & events.event_type.eq("control_stop")].distance_m,
            errors="coerce").dropna().astype(float).tolist())
        for day in range(1, 7)
    }
    forecasts: dict[pd.Timestamp, replay.Forecast] = {}
    nodes = replay.route_nodes(route)
    records: list[dict] = []

    for _, candidate in base.iterrows():
        day = int(candidate.day)
        now = pd.Timestamp(candidate.decision_timestamp_utc)
        ix = log.index[(log.day.astype(int) == day) & (log.timestamp_utc == now)]
        if not len(ix):
            ix = log.index[(log.day.astype(int) == day)
                           & (log.timestamp_utc >= now)].take([0])
        decision = log.loc[ix[0]]
        run = pd.Timestamp(decision.forecast_run_init_utc)
        if run not in forecasts:
            forecasts[run] = replay.Forecast(replay.fetch_run(run, route, nodes))
        now_local = now.tz_convert(replay.TZ)
        deadline = now_local.normalize() + pd.Timedelta(hours=17)
        started = time.perf_counter()
        plan = replay.solve_upper(
            forecasts[run], float(candidate.distance_m), endpoints[day],
            control_stops[day], now, float(candidate.current_energy_est_wh),
            replay.RESERVE_DEFAULT, route, road, pv, deadline, 0.0,
            solver_time_limit_s=TIME_LIMIT_S)
        wall_s = time.perf_counter() - started
        old_h = float(candidate.predicted_remaining_h)
        new_h = float(plan["predicted_remaining_h"]) if plan.get("predicted_remaining_h") is not None else float("nan")
        new_target = float(plan["target_kmh"]) if plan.get("target_kmh") is not None else float("nan")
        records.append({
            "day": day, "decision_timestamp_utc": now.isoformat(),
            "distance_m": float(candidate.distance_m),
            "base_time_limit_s": 2.0, "probe_time_limit_s": TIME_LIMIT_S,
            "base_status": candidate.status, "probe_status": plan["status"],
            "base_mip_gap": float(candidate.mip_gap), "probe_mip_gap": plan.get("mip_gap"),
            "base_target_kmh": float(candidate.target_kmh),
            "probe_target_kmh": new_target,
            "base_predicted_remaining_h": old_h,
            "probe_predicted_remaining_h": new_h,
            "predicted_time_improvement_min": (old_h-new_h)*60.0,
            "probe_predicted_remaining_wh": plan.get("predicted_remaining_wh_robust"),
            "stage_count": plan.get("stage_count"), "probe_wall_s": wall_s,
        })
        print(f"Day {day} solve complete", flush=True)

    result = pd.DataFrame(records)
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    summary = {
        "sample": "all first-hour 5-minute cases whose 2 s solve returned a feasible time-limited plan",
        "base_time_limit_s": 2.0, "probe_time_limit_s": TIME_LIMIT_S,
        "cases": int(len(result)),
        "cases_proven_optimal_at_5s": int(result.probe_status.eq("optimal_finite_grid_milp").sum()),
        "cases_still_time_limited_at_5s": int(result.probe_status.eq("feasible_time_limited_milp_not_proven_optimal").sum()),
        "cases_without_feasible_plan_at_5s": int((~result.probe_status.isin({"optimal_finite_grid_milp", "feasible_time_limited_milp_not_proven_optimal"})).sum()),
        "target_changed_cases": int((result.base_target_kmh != result.probe_target_kmh).sum()),
        "median_predicted_time_improvement_min": float(result.predicted_time_improvement_min.median()),
        "p95_probe_wall_s": float(result.probe_wall_s.quantile(.95)),
        "max_probe_wall_s": float(result.probe_wall_s.max()),
        "limitations": [
            "The candidate states are fixed to human-run telemetry; this is not a counterfactual race rollout.",
            "A lower finite-grid objective does not establish lower real race time because model and actuator mismatch remain.",
            "The probe tests solver budget only for cases already feasible but unproven at 2 s; infeasible cases are excluded.",
        ],
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT}", flush=True)
    print(f"wrote {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
