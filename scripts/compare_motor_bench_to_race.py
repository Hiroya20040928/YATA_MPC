#!/usr/bin/env python3
"""Compare the supplied one-dimensional ECO_LO bench sweep with race operation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "models/m2096_eco_lo_bench_map.csv"
RACE = ROOT / "data/processed/full_race_component_replay_5s.csv"
SPEED_CURVE = ROOT / "reports/speed_power_prediction_curve.csv"
OUT_FIG = ROOT / "reports/motor_bench_vs_race.png"
OUT_CSV = ROOT / "reports/motor_bench_road_comparison.csv"
OUT_JSON = ROOT / "reports/motor_bench_road_comparison.json"
P_AUX_W = 67.6


def main() -> None:
    bench = pd.read_csv(BENCH).sort_values("vehicle_speed_kmh_558_4mm_tire")
    race = pd.read_csv(RACE)
    race = race[(race.day <= 5) & race.road_model_comparison_eligible].copy()
    race["gross_bus_load_w"] = race["Solar_Watt[W]"] + race["Battery_Watt[W]"]
    race["bus_load_less_stopped_aux_w"] = race.gross_bus_load_w - P_AUX_W
    race["speed_bin_kmh"] = np.round(race["Speed[kmh]"].astype(float)).astype(int)
    grouped = race.groupby("speed_bin_kmh").agg(
        n=("gross_bus_load_w", "size"),
        raw_median_gross_w=("gross_bus_load_w", "median"),
        raw_median_less_aux_w=("bus_load_less_stopped_aux_w", "median"),
        raw_p25_less_aux_w=("bus_load_less_stopped_aux_w", lambda x: x.quantile(.25)),
        raw_p75_less_aux_w=("bus_load_less_stopped_aux_w", lambda x: x.quantile(.75)),
    ).reset_index().rename(columns={"speed_bin_kmh": "speed_kmh"})
    grouped = grouped[grouped.n >= 10].copy()
    bench_at_race_speed = np.interp(grouped.speed_kmh, bench.vehicle_speed_kmh_558_4mm_tire,
                                    bench.input_power_w, left=np.nan, right=np.nan)
    # numpy interpolation clamps by default; force unsupported points back to NaN.
    lo, hi = bench.vehicle_speed_kmh_558_4mm_tire.min(), bench.vehicle_speed_kmh_558_4mm_tire.max()
    bench_at_race_speed[(grouped.speed_kmh < lo) | (grouped.speed_kmh > hi)] = np.nan
    grouped["bench_duty1_input_w_if_speed_overlaps"] = bench_at_race_speed
    grouped.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    curve = pd.read_csv(SPEED_CURVE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    ax = axes[0]
    hx = ax.hexbin(race["Speed[kmh]"], race["bus_load_less_stopped_aux_w"], gridsize=40,
                   mincnt=1, bins="log", cmap="magma", linewidths=0)
    sc = ax.scatter(bench.vehicle_speed_kmh_558_4mm_tire, bench.input_power_w,
                    c=bench.total_efficiency_pct, cmap="viridis", edgecolors="white",
                    linewidths=.7, s=60, zorder=4, label="ECO_LO bench (DUTY=1)")
    ax.plot(bench.vehicle_speed_kmh_558_4mm_tire, bench.input_power_w,
            color="#56B4E9", alpha=.7, lw=1.2, zorder=3)
    ax.axvspan(57.626, 78.172, color="#2ca02c", alpha=.08, label="10 km training speed support")
    ax.plot(curve.speed_kmh, curve.model_gross_vehicle_power_w - P_AUX_W, color="#f28e2b", lw=2.1,
            label="Race MLE curve less stopped auxiliary baseline")
    ax.set(xlabel="Vehicle speed (km/h)", ylabel="Input power less 67.6 W stopped baseline (W)",
           title="Race telemetry and fixed-DUTY bench sweep")
    ax.set_xlim(45, 88)
    ax.set_ylim(-500, max(3300, float(np.nanpercentile(bench.input_power_w, 100)) + 100))
    ax.legend(loc="upper left", fontsize=8)
    cb = fig.colorbar(hx, ax=ax, pad=.02)
    cb.set_label("Race rows per hexagon (log scale)")
    cbar = fig.colorbar(sc, ax=ax, pad=.11)
    cbar.set_label("Bench total efficiency (%)")

    ax2 = axes[1]
    points = ax2.scatter(bench.mechanical_output_w, bench.input_power_w,
                         c=bench.total_efficiency_pct, cmap="viridis", s=65,
                         edgecolors="white", linewidths=.7)
    ax2.plot(bench.mechanical_output_w, bench.input_power_w, color="#56B4E9", lw=1.2)
    for i in [0, 7, 8, 15, len(bench) - 1]:
        p = bench.iloc[i]
        ax2.annotate(f"{p.vehicle_speed_kmh_558_4mm_tire:.0f} km/h",
                     (p.mechanical_output_w, p.input_power_w),
                     xytext=(-8, 5) if i == 0 else (5, 4),
                     textcoords="offset points", fontsize=8,
                     ha="right" if i == 0 else "left")
    ax2.plot([0, bench.mechanical_output_w.max()], [0, bench.mechanical_output_w.max()],
             color="gray", ls="--", lw=1, label="100% efficiency")
    ax2.set(xlabel="Bench mechanical output (W)", ylabel="Bench motor/controller input (W)",
            title="ECO_LO characterization is one DUTY=1 sweep")
    ax2.legend(fontsize=8)
    fig.colorbar(points, ax=ax2, pad=.02, label="Bench total efficiency (%)")
    fig.suptitle("M2096 ECO_LO bench data are not an in-car PWM+ECO efficiency map", y=1.02)
    fig.text(.5, .005,
             "Race gross bus load includes stopped auxiliaries; plotted points subtract 67.6 W as a comparison only. "
             "Bench sweep changes load at fixed DUTY=1; no race throttle/PWM-to-bench mapping is available.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .98))
    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "bench_mode": "ECO_LO",
        "bench_duty": 1.0,
        "bench_points": int(len(bench)),
        "bench_vehicle_speed_range_kmh": [float(lo), float(hi)],
        "bench_input_power_range_w": [float(bench.input_power_w.min()), float(bench.input_power_w.max())],
        "bench_torque_range_nm": [float(bench.torque_nm.min()), float(bench.torque_nm.max())],
        "race_clean_rows_days1_5": int(len(race)),
        "race_speed_support_kmh": [float(race["Speed[kmh]"].min()), float(race["Speed[kmh]"].max())],
        "auxiliary_comparison_w": P_AUX_W,
        "model_disposition": "Do not impose the bench curve as an in-car efficiency map. It is a fixed-DUTY load sweep, only partially overlaps the fitted race speed support, and its PWM/ECO state equivalence is unknown.",
        "speed_rows": grouped.astype(object).where(pd.notna(grouped), None).to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "speed_rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
