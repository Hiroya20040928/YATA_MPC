#!/usr/bin/env python3
"""Plot the current BWSC2025 speed-to-power model at all-race typical conditions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fit_vehicle_power_model as features  # noqa: E402

INPUT = ROOT / "data/processed/race_days_distance_geo_weather.csv"
REPLAY = ROOT / "data/processed/full_race_component_replay_5s.csv"
VEHICLE = ROOT / "models/vehicle_power_model_fit.json"
COMPONENT = ROOT / "models/component_replay_fit.json"
OUT_FIG = ROOT / "reports/speed_power_prediction.png"
OUT_CURVE = ROOT / "reports/speed_power_prediction_curve.csv"
OUT_CONTEXT = ROOT / "reports/speed_power_prediction_context.json"
OUT_NOTE = ROOT / "reports/speed_power_prediction_notes.md"
TZ = "Australia/Darwin"


def main() -> None:
    cols = ["day", "timestamp_local", "timestamp_utc", "event_phase", "event_flags",
            "Speed[kmh]", "Solar_Watt[W]", "Battery_Watt[W]", "distance_from_darwin_m",
            "distance_anchor_confidence", "temperature_2m", "surface_pressure",
            "wind_speed_10m", "wind_direction_10m", "route_grade_smoothed_pct"]
    d = pd.read_csv(INPUT, usecols=lambda c: c in cols, low_memory=False)
    d["day"] = pd.to_numeric(d.day, errors="coerce").astype(int)
    d["timestamp_local"] = pd.to_datetime(d.timestamp_local, errors="coerce")
    d["timestamp_utc"] = pd.to_datetime(d.timestamp_utc, utc=True, errors="coerce")
    d = features.make_features(d)
    d["gross_measured_w"] = (pd.to_numeric(d["Solar_Watt[W]"], errors="coerce")
                              + pd.to_numeric(d["Battery_Watt[W]"], errors="coerce"))

    component = json.loads(COMPONENT.read_text(encoding="utf-8"))
    fit = json.loads(VEHICLE.read_text(encoding="utf-8"))
    p = fit["fit_parameters"]["parameters"]
    component_vehicle = component["vehicle"]
    fold_coefficients = component_vehicle["row_replay_coefficients"]
    replay = pd.read_csv(REPLAY, usecols=["day", "timestamp_utc", "predicted_pv_w", "pv_model_comparison_eligible"], low_memory=False)
    replay["day"] = pd.to_numeric(replay.day, errors="coerce").astype(int)
    replay["timestamp_utc"] = pd.to_datetime(replay.timestamp_utc, utc=True, errors="coerce")
    d = d.merge(replay, on=["day", "timestamp_utc"], how="left", validate="one_to_one")

    flags = d.event_flags.fillna("").astype(str)
    flagged = flags.str.contains(r"control_stop:|traffic_stop:|driver_change:|incident:|repair:|charge:|D6_ZP_ANOMALY|D6_RETIRE", regex=True)
    local_ts = pd.to_datetime(d.timestamp_local, errors="coerce", utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    for start, end in [("2025-08-28 10:31:00", "2025-08-28 10:37:00"),
                       ("2025-08-28 11:51:00", "2025-08-28 12:50:00")]:
        flagged |= d.day.eq(5) & local_ts.between(start, end)
    moving = (d.event_phase.eq("road_day") & pd.to_numeric(d["Speed[kmh]"], errors="coerce").between(45, 90)
              & d.distance_anchor_confidence.isin(["high", "medium"]))
    clean = moving & ~flagged & d.gross_measured_w.notna()
    x = d.loc[clean].copy()

    # Race-level median conditions from all clean moving rows (Days 1-6).
    representative = {
        "population": "All BWSC2025 clean road_day rows, speed 45-90 km/h, high/medium distance-anchor confidence, Days 1-6; event/anomaly windows excluded.",
        "rows": int(len(x)),
        "temperature_2m_median_c": float(x.temperature_2m.median()),
        "surface_pressure_median_hpa": float(x.surface_pressure.median()),
        "air_density_median_kgm3": float(x.air_density_kgm3.median()),
        "wind_speed_10m_median_kmh": float(x.wind_speed_10m.median()),
        "wind_direction_10m_median_deg_from": float(x.wind_direction_10m.median()),
        "along_route_headwind_component_median_mps": float((x.relative_air_speed_mps - x.speed_mps).median()),
        "route_grade_median_pct": float(x.route_grade_smoothed_pct.median()),
        "auxiliary_bus_load_w": float(p["p_aux_w"]),
    }
    solar_sample = x[x.pv_model_comparison_eligible.astype(bool) & pd.to_numeric(x.predicted_pv_w, errors="coerce").notna()]
    pv_rep = float(solar_sample.predicted_pv_w.median()) if len(solar_sample) else 0.0
    representative["modelled_pv_power_median_w_daylight_moving"] = pv_rep
    representative["pv_population_rows"] = int(len(solar_sample))

    # Reference curve at median conditions. Acceleration is zero: this is a
    # steady-cruise curve; gross consumption is Solar_Watt + Battery_Watt.
    speed_grid = np.arange(0., 101., 1.)
    v = speed_grid / 3.6
    grade = representative["route_grade_median_pct"] / 100.0
    theta = np.arctan(grade)
    rho = representative["air_density_median_kgm3"]
    headwind = representative["along_route_headwind_component_median_mps"]
    u = v + headwind
    roll = features.MASS_NOMINAL_KG * features.G * features.CRR_NOMINAL * np.cos(theta) * v
    aero = .5 * rho * (u * np.abs(u)) * features.CD_A_NOMINAL_M2 * v
    grade_w = features.MASS_NOMINAL_KG * features.G * np.sin(theta) * v

    def curve_for(param: dict) -> np.ndarray:
        return (float(param.get("p_aux_w", p["p_aux_w"]))
                + float(param["beta_roll"]) * roll
                + float(param["beta_aero"]) * aero
                + float(param["beta_grade"]) * grade_w)

    model_gross = curve_for(p)
    model_battery = model_gross - pv_rep
    fold_curves = np.stack([curve_for(z) for day, z in fold_coefficients.items() if int(day) <= 5])
    fold_min, fold_max = np.min(fold_curves, axis=0), np.max(fold_curves, axis=0)

    # Standardize each measured point to the race-median conditions using the
    # fitted road model, then display both raw and adjusted distributions.
    local_pred = (p["p_aux_w"] + p["beta_roll"] * x.roll_basis_w.to_numpy(float)
                  + p["beta_aero"] * x.aero_basis_w.to_numpy(float)
                  + p["beta_grade"] * x.grade_basis_w.to_numpy(float))
    speeds = pd.to_numeric(x["Speed[kmh]"], errors="coerce").to_numpy(float)
    rep_at_rows = np.interp(speeds, speed_grid, model_gross)
    x["model_standardized_gross_w"] = x.gross_measured_w.to_numpy(float) - local_pred + rep_at_rows
    x["model_residual_after_standardization_w"] = x.model_standardized_gross_w - rep_at_rows
    x["speed_bin_kmh"] = np.rint(speeds).astype(int)
    binned_rows = []
    for speed, group in x.groupby("speed_bin_kmh", sort=True):
        raw = group.gross_measured_w.dropna().to_numpy(float)
        std = group.model_standardized_gross_w.dropna().to_numpy(float)
        binned_rows.append({"speed_kmh": int(speed), "n": len(group),
                            "raw_median_w": float(np.median(raw)),
                            "raw_q25_w": float(np.quantile(raw, .25)), "raw_q75_w": float(np.quantile(raw, .75)),
                            "standardized_median_w": float(np.median(std)),
                            "standardized_q25_w": float(np.quantile(std, .25)), "standardized_q75_w": float(np.quantile(std, .75)),
                            "standardized_residual_median_w": float(np.median(group.model_residual_after_standardization_w))})
    bins = pd.DataFrame(binned_rows)
    curve = pd.DataFrame({"speed_kmh": speed_grid, "model_gross_vehicle_power_w": model_gross,
                          "model_net_battery_power_w_at_median_pv": model_battery,
                          "day_lodo_fold_min_gross_w": fold_min, "day_lodo_fold_max_gross_w": fold_max})
    curve = curve.merge(bins, on="speed_kmh", how="left")
    curve.to_csv(OUT_CURVE, index=False, encoding="utf-8-sig")

    battery_energy_wh = float(component["battery"]["user_theoretical_pack_energy_wh"])
    summary_speeds = [50, 60, 70, 80, 85]
    summary_rows = []
    for speed in summary_speeds:
        row = bins[bins.speed_kmh.eq(speed)]
        summary_rows.append({"speed_kmh": speed,
                             "model_domain": "supported" if 57.63 <= speed <= 78.17 else "extrapolation",
                             "gross_model_w": float(np.interp(speed, speed_grid, model_gross)),
                             "pv_representative_w": pv_rep,
                             "battery_net_model_w": float(np.interp(speed, speed_grid, model_battery)),
                             "nominal_energy_hours_no_reserve": (battery_energy_wh / np.interp(speed, speed_grid, model_battery)
                                                                  if np.interp(speed, speed_grid, model_battery) > 0 else None),
                             "raw_sample_n": int(row.iloc[0].n) if len(row) else 0,
                             "raw_median_w": float(row.iloc[0].raw_median_w) if len(row) else None,
                             "standardized_median_w": float(row.iloc[0].standardized_median_w) if len(row) else None})

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax, residual_ax) = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                                          gridspec_kw={"height_ratios": [3.2, 1]}, constrained_layout=True)
    hb = ax.hexbin(x.loc[:, "Speed[kmh]"], x.gross_measured_w, gridsize=55, mincnt=1,
                   bins="log", cmap="magma", linewidths=0, alpha=.86)
    ax.fill_between(speed_grid, fold_min, fold_max, color="#5876a3", alpha=.13,
                    label="Day-LODO coefficient envelope (sensitivity, not CI)")
    b = bins[bins.n.ge(10)]
    ax.fill_between(b.speed_kmh, b.raw_q25_w, b.raw_q75_w, color="#e07a47", alpha=.14,
                    label="Raw telemetry IQR (n ≥ 10)")
    ax.plot(b.speed_kmh, b.raw_median_w, color="#d65f32", marker="o", ms=3, lw=1.25,
            label="Raw median")
    ax.fill_between(b.speed_kmh, b.standardized_q25_w, b.standardized_q75_w,
                    color="#1686a1", alpha=.10, label="Model-standardized telemetry IQR (n ≥ 10)")
    ax.plot(b.speed_kmh, b.standardized_median_w, color="#1686a1", marker="o", ms=3, lw=1.5,
            label="Model-standardized median")
    # The 10 km segment training samples span mean speeds 57.63-78.17 km/h.
    fit_support = (speed_grid >= 57.63) & (speed_grid <= 78.17)
    ax.plot(speed_grid[fit_support], model_gross[fit_support], color="#214e82", lw=2.9,
            label="YATA_MPC 10 km MLE gross-power model at representative conditions")
    ax.plot(speed_grid[~fit_support], model_gross[~fit_support], color="#214e82", lw=2.2, ls="--",
            label="Model extrapolation outside 57.6–78.2 km/h training support")
    ax.plot(speed_grid, model_battery, color="#6d5aa7", lw=1.7, ls=":",
            label=f"Battery power = gross − race-median PV ({pv_rep:.0f} W)")
    ax.axvspan(57.63, 78.17, color="#214e82", alpha=.025, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_ylabel("HV power (W)")
    ax.set_title("BWSC2025 speed vs. gross consumption at race-representative conditions")
    ax.legend(loc="upper left", ncol=2, fontsize=8, framealpha=.95)
    cbar = fig.colorbar(hb, ax=ax, pad=.012)
    cbar.set_label("Raw clean moving observations per hexbin (log scale)")
    summary_lines = []
    for speed in [60, 70, 80, 85]:
        row = bins[bins.speed_kmh.eq(speed)]
        if len(row):
            n = int(row.iloc[0].n)
            domain_tag = "" if 57.63 <= speed <= 78.17 else " (extrap)"
            summary_lines.append(f"{speed} km/h{domain_tag}: model {np.interp(speed, speed_grid, model_gross):.0f} W; "
                                 f"raw median {row.iloc[0].raw_median_w:.0f} W; "
                                 f"standardized {row.iloc[0].standardized_median_w:.0f} W; n={n}")
        elif speed == 85:
            summary_lines.append(f"85 km/h (extrap): model {np.interp(speed, speed_grid, model_gross):.0f} W; no clean samples")
    if summary_lines:
        ax.text(.995, .985, "\n".join(summary_lines), transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, bbox={"boxstyle": "round,pad=.35", "facecolor": "white", "edgecolor": "#9aa4b2", "alpha": .94})

    residual_ax.axhline(0, color="#30343b", lw=1)
    residual_ax.plot(b.speed_kmh, b.standardized_residual_median_w, color="#6d5aa7", marker="o", ms=3, lw=1.3)
    model_at_bins = np.interp(b.speed_kmh.to_numpy(float), speed_grid, model_gross)
    residual_low = b.standardized_q25_w.to_numpy(float) - model_at_bins
    residual_high = b.standardized_q75_w.to_numpy(float) - model_at_bins
    residual_ax.fill_between(b.speed_kmh, residual_low, residual_high,
                             color="#6d5aa7", alpha=.12)
    residual_ax.set_xlabel("Vehicle speed (km/h)")
    residual_ax.set_ylabel("Standardized median − model (W)")
    residual_ax.set_xticks(np.arange(0, 101, 10))
    residual_limit = max(100., float(np.nanmax(np.abs(np.r_[residual_low, residual_high,
                                                              b.standardized_residual_median_w.to_numpy(float)]))) * 1.08)
    residual_ax.set_ylim(-residual_limit, residual_limit)
    residual_ax.text(.01, .97,
                     "Steady cruise curve assumes 0 acceleration; raw spread includes acceleration/braking/regen. IQR is observed spread, not CI.",
                     transform=residual_ax.transAxes, ha="left", va="top", fontsize=8)
    fig.savefig(OUT_FIG, dpi=190, bbox_inches="tight")
    plt.close(fig)

    out_context = {
        "representative_conditions": representative,
        "model": {"source": str(VEHICLE.relative_to(ROOT)), "parameters": p,
                  "fit_training_mean_speed_range_kmh": [57.626, 78.172], "outside_domain_style": "dashed extrapolation",
                  "assumed_acceleration_mps2": 0.0, "mass_kg": features.MASS_NOMINAL_KG,
                  "cd_a_m2": features.CD_A_NOMINAL_M2, "crr": features.CRR_NOMINAL,
                  "auxiliary_load_included_w": float(p["p_aux_w"])},
        "data": {"all_race_rows": int(len(d)), "clean_moving_rows": int(len(x)),
                 "flagged_or_excluded_moving_rows": int((moving & flagged).sum()),
                 "standardization_formula": "measured gross power - full-fit model at the row's logged weather/grade + full-fit model at race-median conditions for that row's speed",
                 "gross_power_definition": "Solar_Watt + Battery_Watt; signed Battery_Watt is positive on discharge.",
                 "pv_net_power_definition": "Gross vehicle power minus median modelled PV power among moving daylight rows; an illustrative energy-balance curve, not a battery voltage limit calculation."},
        "outputs": [str(OUT_FIG.relative_to(ROOT)), str(OUT_CURVE.relative_to(ROOT))],
    }
    OUT_CONTEXT.write_text(json.dumps(out_context, indent=2, ensure_ascii=False), encoding="utf-8")
    condition_rows = [
        ("Road rows", representative["rows"], "clean moving rows, Days 1–6"),
        ("Ambient temperature", representative["temperature_2m_median_c"], "°C median"),
        ("Surface pressure", representative["surface_pressure_median_hpa"], "hPa median"),
        ("Air density", representative["air_density_median_kgm3"], "kg/m³ median"),
        ("Wind speed", representative["wind_speed_10m_median_kmh"], "km/h median"),
        ("Wind direction", representative["wind_direction_10m_median_deg_from"], "degrees FROM median"),
        ("Along-route headwind component", representative["along_route_headwind_component_median_mps"], "m/s median; negative is tailwind"),
        ("Route grade", representative["route_grade_median_pct"], "% median"),
        ("Auxiliary load", representative["auxiliary_bus_load_w"], "W; includes 21 W weak systems"),
        ("PV power", pv_rep, "W median of modeled moving-daylight output"),
        ("Nominal pack energy", battery_energy_wh, "Wh; user basis, 3011 Wh"),
    ]
    note = "# BWSC2025 speed–power prediction at race-representative conditions\n\n"
    note += "![Speed–power curve](speed_power_prediction.png)\n\n"
    note += "The solid curve is the current 10 km segment MLE gross HV load model evaluated at representative BWSC2025 conditions. The 2025 telemetry points use `Solar_Watt + Battery_Watt`; the purple dotted curve subtracts median modeled daylight PV. This is a steady-cruise model (zero acceleration), not a closed-loop simulation.\n\n"
    note += "## Representative conditions\n\n| Condition | Value | Basis |\n|---|---:|---|\n"
    for name, value, unit in condition_rows:
        formatted = f"{value:,}" if isinstance(value, int) else f"{value:.3f}"
        note += f"| {name} | {formatted} | {unit} |\n"
    note += "\n## Predictions\n\n| Speed (km/h) | Model domain | Gross vehicle power (W) | Net battery power at median PV (W) | Raw sample n | Raw median (W) | Standardized median (W) | Idealized 3011 Wh depletion time (h) |\n|---:|---|---:|---:|---:|---:|---:|---:|\n"
    for row in summary_rows:
        time = f"{row['nominal_energy_hours_no_reserve']:.2f}" if row["nominal_energy_hours_no_reserve"] is not None else "surplus at median PV"
        raw = f"{row['raw_median_w']:.1f}" if row["raw_median_w"] is not None else "—"
        std = f"{row['standardized_median_w']:.1f}" if row["standardized_median_w"] is not None else "—"
        note += f"| {row['speed_kmh']} | {row['model_domain']} | {row['gross_model_w']:.1f} | {row['battery_net_model_w']:.1f} | {row['raw_sample_n']} | {raw} | {std} | {time} |\n"
    note += "\nThe depletion-time column is only a gross nominal-energy ratio (3011 Wh divided by positive net power). It assumes the entire theoretical pack energy is usable and ignores reserve SoC, voltage/current limits, temperature, discharge efficiency, and changing solar/weather. Negative battery power means modeled PV exceeds modeled vehicle demand under the chosen median conditions; it is not a charging-limit prediction.\n\n"
    note += "The actual telemetry IQR includes acceleration, braking/regeneration, route grade and measurement/model error; it is observed spread, not a confidence interval. Condition-standardized telemetry subtracts the full-fit model at each row's logged conditions and adds its prediction at median conditions, so it is a model adjustment rather than independent validation. The 10 km model was trained on segment mean speeds from 57.6 to 78.2 km/h; solid portions are within that support and dashed portions are extrapolations. The 50, 80 and 85 km/h table rows are outside training support; 85 km/h has no clean plotted telemetry samples. The adjacent ±2.5 km/h decision-resolution gate remains failed (53.3 W 10 km LODO MAE vs. 23.8 W internal gate). See `BWSC2025_component_model_iteration.md` for validation and limits.\n"
    OUT_NOTE.write_text(note, encoding="utf-8")
    print(json.dumps({"figure": str(OUT_FIG), "curve": str(OUT_CURVE), "context": str(OUT_CONTEXT),
                      "note": str(OUT_NOTE),
                      "representative": representative,
                      "model_power_at_kmh": {str(s): float(np.interp(s, speed_grid, model_gross)) for s in [50, 60, 70, 80, 85]},
                      "model_battery_at_kmh": {str(s): float(np.interp(s, speed_grid, model_battery)) for s in [50, 60, 70, 80, 85]},
                      "clean_moving_rows": len(x), "flagged_moving_rows": int((moving & flagged).sum()),
                      "speed_summary": summary_rows},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
