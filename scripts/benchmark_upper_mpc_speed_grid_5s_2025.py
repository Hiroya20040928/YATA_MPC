#!/usr/bin/env python3
"""Compare 0.5 and 1 km/h upper speed grids with a 5 s MILP budget."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replay_hierarchical_mpc_2025 as replay  # noqa: E402

OUTPUT = ROOT / "reports/mpc_upper_speed_grid_5s_probe_2025.csv"
SUMMARY = ROOT / "reports/mpc_upper_speed_grid_5s_probe_2025.json"
GRID_STEPS_KMH = (0.5, 1.0)
STAGE_LENGTH_KM = 10.0
RESERVE_PCT = 0.15
TIME_LIMIT_S = 5.0


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
    decisions = replay_rows[replay_rows.upper_replan_now].groupby("day", sort=True).head(1)
    cache: dict[pd.Timestamp, replay.Forecast] = {}
    nodes = replay.route_nodes(route)
    old_speeds, old_stage = replay.SPEEDS.copy(), replay.HORIZON_M
    rows: list[dict] = []
    try:
        replay.HORIZON_M = STAGE_LENGTH_KM * 1000.0
        for step in GRID_STEPS_KMH:
            replay.SPEEDS = np.arange(60.0, 78.0 + step * 0.25, step)
            for _, decision in decisions.iterrows():
                day, now = int(decision.day), pd.Timestamp(decision.timestamp_utc)
                run = pd.Timestamp(decision.forecast_run_init_utc)
                if run not in cache:
                    cache[run] = replay.Forecast(replay.fetch_run(run, route, nodes))
                ix = race.index[(race.day.astype(int) == day)
                                & (race.timestamp_utc == now)]
                if not len(ix):
                    ix = race.index[(race.day.astype(int) == day)
                                    & (race.timestamp_utc >= now)].take([0])
                state = race.loc[ix[0]]
                local = pd.Timestamp(decision.timestamp_local)
                deadline = local.normalize() + pd.Timedelta(hours=17)
                started = time.perf_counter()
                plan = replay.solve_upper(
                    cache[run], float(decision.distance_from_darwin_m), day_end[day],
                    stops[day], now, float(state.energy_for_solve_wh), RESERVE_PCT,
                    route, road, pv, deadline, 0.0,
                    solver_time_limit_s=TIME_LIMIT_S)
                wall = time.perf_counter() - started
                rows.append({
                    "day": day, "decision_timestamp_utc": now.isoformat(),
                    "stage_length_km": STAGE_LENGTH_KM,
                    "speed_grid_step_kmh": step,
                    "speed_candidates": int(len(replay.SPEEDS)),
                    "solver_time_limit_s": TIME_LIMIT_S,
                    "status": plan["status"], "mip_gap": plan.get("mip_gap"),
                    "first_target_kmh": plan.get("target_kmh"),
                    "predicted_remaining_h": plan.get("predicted_remaining_h"),
                    "predicted_remaining_wh": plan.get("predicted_remaining_wh_robust"),
                    "stage_count": plan.get("stage_count"), "solve_wall_s": wall,
                })
            print(f"speed grid {step:g} km/h complete", flush=True)
    finally:
        replay.SPEEDS, replay.HORIZON_M = old_speeds, old_stage

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    summaries = []
    for step, group in result.groupby("speed_grid_step_kmh", sort=True):
        summaries.append({
            "speed_grid_step_kmh": float(step),
            "decision_states": int(len(group)),
            "optimal": int(group.status.eq("optimal_finite_grid_milp").sum()),
            "time_limited_feasible": int(group.status.eq("feasible_time_limited_milp_not_proven_optimal").sum()),
            "no_advisory": int((~group.status.isin({"optimal_finite_grid_milp", "feasible_time_limited_milp_not_proven_optimal"})).sum()),
            "median_solve_wall_s": float(group.solve_wall_s.median()),
            "p95_solve_wall_s": float(group.solve_wall_s.quantile(.95)),
            "max_solve_wall_s": float(group.solve_wall_s.max()),
            "median_predicted_remaining_h": float(group.predicted_remaining_h.median()),
            "day_results": group[["day", "status", "first_target_kmh",
                                  "predicted_remaining_h", "predicted_remaining_wh",
                                  "solve_wall_s"]].to_dict("records"),
        })
    pivot = result.pivot(index="day", columns="speed_grid_step_kmh",
                         values="predicted_remaining_h")
    summary = {
        "sample": "first eligible upper decision from each day; fixed as-of state/forecast",
        "stage_length_km": STAGE_LENGTH_KM, "reserve_pct": RESERVE_PCT * 100,
        "solver_time_limit_s": TIME_LIMIT_S,
        "interpretation": [
            "This is a finite-grid sensitivity comparison, not closed-loop race-time evidence.",
            "A half-km/h grid adds candidate variables, while speed and power logs/model residuals do not validate that resolution as physically superior.",
            "Infeasible reserve/deadline and solver failures remain diagnostic no-advisory states, not feasible speed recommendations.",
        ],
        "summary": summaries,
        "per_day_predicted_time_delta_h_0p5_minus_1p0": {
            str(int(day)): float(pivot.loc[day, 0.5] - pivot.loc[day, 1.0])
            for day in pivot.index
        },
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT}", flush=True)
    print(f"wrote {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
