#!/usr/bin/env python3
"""Probe upper-MPC recalculation periods on the first race hour of each day."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replay_hierarchical_mpc_2025 as replay  # noqa: E402

OUTPUT = ROOT / "reports/mpc_upper_replan_period_probe_2025.csv"
SUMMARY = ROOT / "reports/mpc_upper_replan_period_probe_2025.json"
PERIODS_MIN = (5, 10, 15, 30, 60)
TROUBLE = re.compile(r"incident:|control_stop:|traffic_stop:|driver_change:|repair:|charge:|hatch_repair:")


def main() -> None:
    log = pd.read_csv(replay.OUTPUT, low_memory=False,
                      parse_dates=["timestamp_utc", "timestamp_local"])
    race, route, events, road, pv, socmap, component = replay.load_inputs()
    endpoints = replay.daily_endpoints(events, float(route.distance_m.max()))
    control_stops = {
        day: sorted(pd.to_numeric(events[
            (pd.to_numeric(events.day, errors="coerce") == day)
            & events.event_type.eq("control_stop")].distance_m,
            errors="coerce").dropna().astype(float).tolist())
        for day in range(1, 7)
    }
    energy, _ = replay.actual_soc_wh(race, socmap, component)
    race["energy_for_solve_wh"] = energy
    runs: dict[pd.Timestamp, replay.Forecast] = {}
    nodes = replay.route_nodes(route)
    records: list[dict] = []

    for day, g in log.groupby("day", sort=True):
        g = g[(g.event_phase == "road_day") & (g["Speed[kmh]"] >= 5)
              & (g.timestamp_local.dt.hour < 17)
              & ~g.event_flags.fillna("").astype(str).str.contains(TROUBLE)]
        if g.empty:
            continue
        start = g.timestamp_utc.iloc[0]
        first_hour = g[g.timestamp_utc <= start + pd.Timedelta(hours=1)]
        for period_min in PERIODS_MIN:
            elapsed_bucket = ((first_hour.timestamp_utc - start).dt.total_seconds()
                              // (period_min * 60)).astype(int)
            samples = first_hour.loc[~elapsed_bucket.duplicated(keep="first").to_numpy()]
            previous_target = None
            for _, decision in samples.iterrows():
                now = pd.Timestamp(decision.timestamp_utc)
                run = pd.Timestamp(decision.forecast_run_init_utc)
                if run not in runs:
                    runs[run] = replay.Forecast(replay.fetch_run(run, route, nodes))
                ix = race.index[(race.day.astype(int) == int(day))
                                & (race.timestamp_utc == now)]
                if not len(ix):
                    ix = race.index[(race.day.astype(int) == int(day))
                                    & (race.timestamp_utc >= now)].take([0])
                state = race.loc[ix[0]]
                deadline = pd.Timestamp(decision.timestamp_local).normalize() + pd.Timedelta(hours=17)
                t0 = time.perf_counter()
                plan = replay.solve_upper(
                    runs[run], float(decision.distance_from_darwin_m), endpoints[int(day)],
                    control_stops[int(day)], now, float(state.energy_for_solve_wh),
                    replay.RESERVE_DEFAULT, route, road, pv, deadline, 0.0)
                elapsed = time.perf_counter() - t0
                target = plan.get("target_kmh")
                records.append({
                    "day": int(day), "period_min": period_min,
                    "decision_timestamp_utc": now.isoformat(),
                    "elapsed_from_first_decision_s": float((now-start).total_seconds()),
                    "distance_m": float(decision.distance_from_darwin_m),
                    "current_energy_est_wh": float(state.energy_for_solve_wh),
                    "status": plan["status"], "mip_gap": plan.get("mip_gap"),
                    "target_kmh": target, "target_changed_from_previous": (
                        bool(target != previous_target) if previous_target is not None else False),
                    "predicted_remaining_h": plan.get("predicted_remaining_h"),
                    "predicted_remaining_energy_wh": plan.get("predicted_remaining_wh_robust"),
                    "stage_count": plan.get("stage_count"), "solve_wall_s": elapsed,
                })
                previous_target = target
            print(f"Day {int(day)} period sweep complete", flush=True)

    result = pd.DataFrame(records)
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    groups = []
    for period, group in result.groupby("period_min", sort=True):
        groups.append({
            "period_min": int(period), "solves": int(len(group)),
            "target_changes_after_first": int(group.target_changed_from_previous.sum()),
            "optimal": int(group.status.eq("optimal_finite_grid_milp").sum()),
            "time_limited_feasible": int(group.status.eq("feasible_time_limited_milp_not_proven_optimal").sum()),
            "no_advisory": int((~group.status.isin({"optimal_finite_grid_milp", "feasible_time_limited_milp_not_proven_optimal"})).sum()),
            "median_solve_s": float(group.solve_wall_s.median()),
            "p95_solve_s": float(group.solve_wall_s.quantile(.95)),
            "day_target_changes": group.groupby("day").target_changed_from_previous.sum().astype(int).to_dict(),
        })
    summary = {
        "sample": "first hour of each day, using logged state/energy and as-of archived weather",
        "solver_grid_kmh": replay.SPEEDS.tolist(), "stage_length_km": replay.HORIZON_M/1000,
        "reserve_pct": replay.RESERVE_DEFAULT*100, "periods": groups,
        "limitations": [
            "This is not a counterfactual closed-loop race: timestamps and logged state are fixed to the human run.",
            "No PWM response is observed, and forecast/vehicle model parameters were fitted after the race.",
            "Target changes measure replanning sensitivity, not a causal lap-time improvement.",
        ],
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT}", flush=True)
    print(f"wrote {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
