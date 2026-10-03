#!/usr/bin/env python3
"""Fit a first road-load model to BWSC2025 route segments.

Power is Solar_Watt + Battery_Watt: the direct HV-bus balance proxy confirmed
by the team. Segment aggregation avoids treating 5-second samples as
independent evidence. Coefficients are effective calibration terms, not
separately identified true mass, Crr, CdA, or motor efficiency.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fit_vehicle_power_model as features  # noqa: E402

INPUT = ROOT / "data/processed/race_days_distance_geo_weather.csv"
OUT_SEGMENTS = ROOT / "data/processed/vehicle_energy_segments_10km.csv"
OUT_MODEL = ROOT / "models/vehicle_power_model_fit.json"
OUT_CV = ROOT / "reports/vehicle_power_cv_by_day.csv"
OUT_FIG = ROOT / "reports/vehicle_power_model_diagnostics.png"
OUT_REPORT = ROOT / "reports/vehicle_power_model_summary.md"
MASS_FLOOR_KG = 314.0  # 234 kg car + regulatory 80 kg occupant/ballast floor
BIN_M = 10_000.0
G = 9.80665
DF = 4.0


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    e = np.asarray(y, float) - np.asarray(pred, float)
    return {"n": int(len(e)), "mae_w": float(np.mean(np.abs(e))),
            "rmse_w": float(np.sqrt(np.mean(e**2))),
            "bias_measured_minus_model_w": float(np.mean(e)),
            "p90_abs_error_w": float(np.quantile(np.abs(e), .9))}


def fit_mle(frame: pd.DataFrame) -> dict:
    cols = ["roll_w", "aero_w", "grade_endpoint_w"]
    X, y = frame[cols].to_numpy(float), frame.measured_net_w.to_numpy(float)
    p0 = float(frame.attrs["p_aux_w"])
    def nll(p):
        mu = p0 + X @ p[:3]
        scale = p[3]
        return float(-np.sum(student_t.logpdf((y - mu) / scale, df=DF) - np.log(scale)))
    # p_aux is measured from stopped telemetry and remains fixed in the road fit.
    fit = minimize(nll, [1., 1., 1., 60.], method="L-BFGS-B",
                   bounds=[(0, 5), (0, 5), (0, 5), (10, 500)],
                   options={"maxiter": 1000, "ftol": 1e-11})
    return {"success": bool(fit.success), "message": str(fit.message),
            "parameters": {"p_aux_w": p0, **{k: float(v) for k, v in zip(
                ["beta_roll", "beta_aero", "beta_grade", "student_t_scale_w"], fit.x)}},
            "negative_log_likelihood": float(fit.fun), "df": DF, "n": len(frame)}


def predict(frame: pd.DataFrame, model: dict) -> np.ndarray:
    p = model["parameters"]
    return (p["p_aux_w"] + p["beta_roll"] * frame.roll_w.to_numpy(float)
            + p["beta_aero"] * frame.aero_w.to_numpy(float)
            + p["beta_grade"] * frame.grade_endpoint_w.to_numpy(float))


def main() -> None:
    cols = ["day", "timestamp_local", "event_phase", "event_flags", "Solar_Watt[W]",
            "Battery_Watt[W]", "Speed[kmh]", "distance_from_darwin_m",
            "distance_anchor_confidence", "route_elevation_dem_m", "route_grade_smoothed_pct",
            "temperature_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
    d = pd.read_csv(INPUT, usecols=lambda c: c in cols, low_memory=False)
    d = features.make_features(d)
    # Five-second HV balance and stopped-road load. The measured 21 W weak-system
    # draw is a component of this total; do not add it a second time.
    d["net_power_w"] = pd.to_numeric(d["Solar_Watt[W]"], errors="coerce") + pd.to_numeric(d["Battery_Watt[W]"], errors="coerce")
    stop = d[(d["Speed[kmh]"].eq(0)) & d.event_phase.isin(["road_day", "charge"])
             & d.net_power_w.between(0, 250)].net_power_w.dropna()
    p_aux = float(stop.median())
    d["p_aux_w"] = p_aux
    d["next_timestamp"] = d.groupby("day", sort=False).timestamp_local.shift(-1)
    d["dt_s"] = (d.next_timestamp - d.timestamp_local).dt.total_seconds()
    d["next_distance_m"] = d.groupby("day", sort=False).distance_from_darwin_m.shift(-1)
    d["ds_m"] = d.next_distance_m - d.distance_from_darwin_m
    d["next_elevation_m"] = d.groupby("day", sort=False).route_elevation_dem_m.shift(-1)
    d["e_net_wh"] = d.net_power_w * d.dt_s / 3600
    grade = pd.to_numeric(d.route_grade_smoothed_pct, errors="coerce") / 100
    theta = np.arctan(grade)
    d["e_roll_wh"] = MASS_FLOOR_KG * G * features.CRR_NOMINAL * np.cos(theta) * d.ds_m / 3600
    # Signed relative-air-speed work with weather direction defined as FROM.
    u = d.relative_air_speed_mps
    d["e_aero_wh"] = .5 * d.air_density_kgm3 * features.CD_A_NOMINAL_M2 * u * u.abs() * d.speed_mps * d.dt_s / 3600
    d["e_grade_wh"] = MASS_FLOOR_KG * G * (d.next_elevation_m - d.route_elevation_dem_m) / d.dt_s * d.dt_s / 3600
    flags = d.event_flags.fillna("").astype(str)
    exclude = flags.str.contains(r"control_stop:|traffic_stop:|driver_change:|D5_HATCH_FAILURE|D5_PUNCTURE|D4_LOW_BATTERY_STOP|D6_ZP_ANOMALY|D6_RETIRE", regex=True)
    ts = d.timestamp_local
    for start, end in [("2025-08-28 10:31:00", "2025-08-28 10:37:00"),
                       ("2025-08-28 11:51:00", "2025-08-28 12:50:00")]:
        exclude |= (d.day.eq(5) & ts.between(start, end))
    d["exclude_segment_fit"] = exclude
    good = (d.event_phase.eq("road_day") & ~exclude & d["Speed[kmh]"].ge(45)
            & d.dt_s.between(2, 15) & d.ds_m.gt(0)
            & d.distance_anchor_confidence.isin(["high", "medium"])
            & d[["net_power_w", "air_density_kgm3", "relative_air_speed_mps",
                 "route_elevation_dem_m", "next_elevation_m"]].notna().all(axis=1))
    q = d.loc[good].copy()
    q["bin_id"] = (q.distance_from_darwin_m // BIN_M).astype(int)
    seg = q.groupby(["day", "bin_id"], as_index=False).agg(
        net_wh=("e_net_wh", "sum"), roll_wh=("e_roll_wh", "sum"), aero_wh=("e_aero_wh", "sum"),
        grade_wh=("e_grade_wh", "sum"), time_s=("dt_s", "sum"), distance_m=("ds_m", "sum"),
        speed_kmh=("Speed[kmh]", "mean"), rows=("dt_s", "size"),
        start_m=("distance_from_darwin_m", "min"), end_m=("distance_from_darwin_m", "max"),
        elev_start_m=("route_elevation_dem_m", "first"), elev_end_m=("next_elevation_m", "last"))
    seg = seg[(seg.distance_m >= BIN_M * .75) & (seg.rows >= 5) & (seg.time_s > 0)].copy()
    seg["measured_net_w"] = seg.net_wh * 3600 / seg.time_s
    seg["roll_w"] = seg.roll_wh * 3600 / seg.time_s
    seg["aero_w"] = seg.aero_wh * 3600 / seg.time_s
    seg["grade_endpoint_w"] = MASS_FLOOR_KG * G * (seg.elev_end_m - seg.elev_start_m) / seg.time_s
    seg["predicted_net_w"] = np.nan
    seg["residual_measured_minus_model_w"] = np.nan
    seg.attrs["p_aux_w"] = p_aux
    train = seg[seg.day.isin([1, 2, 3, 4, 5])].copy()
    train.attrs["p_aux_w"] = p_aux
    model = fit_mle(train)
    motor_prior = features.build_motor_map()
    beta = model["parameters"]
    eta_reference = motor_prior["efficiency_at_1000w_input_pct"] / 100.0
    conditional_equivalents = {
        "assumed_efficiency": eta_reference,
        "effective_cd_a_m2_if_eco_lo_1kw_efficiency_applies": features.CD_A_NOMINAL_M2 * beta["beta_aero"] * eta_reference,
        "effective_crr_if_eco_lo_1kw_efficiency_applies": features.CRR_NOMINAL * beta["beta_roll"] * eta_reference,
        "grade_equivalent_mass_kg_if_eco_lo_1kw_efficiency_applies": MASS_FLOOR_KG * beta["beta_grade"] * eta_reference,
        "warning": "Conditional diagnostics only. The grade-equivalent mass falls below vehicle dry mass, proving at least one grade/route/regen/measurement assumption is not physical; do not use these as true fitted parameters."
    }
    train["pred"] = predict(train, model)
    full_metrics = metrics(train.measured_net_w, train.pred)
    cv_rows = []
    for day in [1, 2, 3, 4, 5]:
        tr = seg[seg.day.isin([d0 for d0 in [1, 2, 3, 4, 5] if d0 != day])].copy()
        te = seg[seg.day.eq(day)].copy()
        tr.attrs["p_aux_w"] = p_aux
        f = fit_mle(tr)
        cv_rows.append({"held_out_day": day, **metrics(te.measured_net_w, predict(te, f)),
                        "train_segments": len(tr), "test_segments": len(te)})
    d6 = seg[seg.day.eq(6)].copy()
    heldout_d6 = {**metrics(d6.measured_net_w, predict(d6, model)), "day": 6,
                  "segments": len(d6), "used_in_fit": False}
    # Additionally compare distance resolutions using only the same clean run samples.
    resolution_cv = []
    for km in [2.5, 5., 10.]:
        if km == 10:
            candidate = seg.copy()
        else:
            qq = q.copy()
            qq["bin_id"] = (qq.distance_from_darwin_m // (km * 1000)).astype(int)
            candidate = qq.groupby(["day", "bin_id"], as_index=False).agg(
                net_wh=("e_net_wh", "sum"), roll_wh=("e_roll_wh", "sum"), aero_wh=("e_aero_wh", "sum"),
                grade_wh=("e_grade_wh", "sum"), time_s=("dt_s", "sum"), distance_m=("ds_m", "sum"),
                rows=("dt_s", "size"), start_m=("distance_from_darwin_m", "min"), end_m=("distance_from_darwin_m", "max"),
                elev_start_m=("route_elevation_dem_m", "first"), elev_end_m=("next_elevation_m", "last"))
            candidate = candidate[(candidate.distance_m >= km * 1000 * .75) & (candidate.rows >= 5) & (candidate.time_s > 0)].copy()
            candidate["measured_net_w"] = candidate.net_wh * 3600 / candidate.time_s
            candidate["roll_w"] = candidate.roll_wh * 3600 / candidate.time_s
            candidate["aero_w"] = candidate.aero_wh * 3600 / candidate.time_s
            candidate["grade_endpoint_w"] = MASS_FLOOR_KG * G * (candidate.elev_end_m-candidate.elev_start_m)/candidate.time_s
        errs=[]
        for day in [1,2,3,4,5]:
            tr=candidate[candidate.day.ne(day)&candidate.day.isin([1,2,3,4,5])].copy(); te=candidate[candidate.day.eq(day)].copy()
            if te.empty or tr.empty: continue
            tr.attrs["p_aux_w"]=p_aux
            f=fit_mle(tr); errs.extend(np.abs(te.measured_net_w.to_numpy()-predict(te,f)).tolist())
        resolution_cv.append({"segment_width_km": km, "segment_count": int(len(candidate)),
                              "clean_day_lodo_mae_w": float(np.mean(errs)) if errs else None})
    seg.loc[seg.day.isin([1,2,3,4,5]), "predicted_net_w"] = predict(seg[seg.day.isin([1,2,3,4,5])], model)
    seg.loc[seg.day.eq(6), "predicted_net_w"] = predict(d6, model)
    seg["residual_measured_minus_model_w"] = seg.measured_net_w - seg.predicted_net_w
    seg.to_csv(OUT_SEGMENTS, index=False, encoding="utf-8-sig")
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(OUT_CV, index=False, encoding="utf-8-sig")
    payload = {
        "target": "Solar_Watt[W] + Battery_Watt[W]; direct HV bus balance proxy",
        "fit_method": "Student-t (df=4) maximum likelihood on 10 km route-distance segment means",
        "features": {"rolling": "314 kg * g * 0.0035 * cos(route grade) * distance",
                     "aerodynamic": "0.5 * modeled air density * 0.08876 m2 * signed relative-air-speed-squared * distance",
                     "grade": "314 kg * g * endpoint DEM elevation change"},
        "p_aux_w_stationary_median": p_aux, "weak_system_measured_w_component": 21.0,
        "fit_days": [1,2,3,4,5], "held_out_anomaly_day": 6,
        "segment_count_fit": len(train), "segment_count_day6": len(d6),
        "fit_parameters": model, "fit_metrics_days1to5": full_metrics,
        "leave_one_clean_day_out": cv_rows, "day6_fully_held_out_metrics": heldout_d6,
        "resolution_comparison": resolution_cv,
        "design_priors": {"car_mass_without_driver_kg": 234.0, "regulatory_occupant_mass_floor_kg": 80.0,
                          "combined_floor_proxy_kg": MASS_FLOOR_KG, "cd_a_m2": features.CD_A_NOMINAL_M2,
                          "crr": features.CRR_NOMINAL, "tire_outer_diameter_m": features.TIRE_DIAMETER_M,
                          "wheel_count": 3, "gear_ratio": 1.0},
        "motor_bench_reference": motor_prior,
        "conditional_physical_equivalents_at_bench_efficiency": conditional_equivalents,
        "interpretation": "beta_roll, beta_aero, beta_grade absorb errors in design priors, DEM grade, modeled wind, installed motor/controller efficiency and ZP bias. They cannot be reported as true Crr/CdA/mass/efficiency. beta_grade materially differs from physical unity; it is evidence that route grade/energy/regeneration/measurement are not independently identified.",
        "limits": ["Distance is anchor-scaled from speed, not GPS truth; route elevation is DEM-derived.",
                   "The 80 kg occupant is the regulatory floor only; actual weighed driver mass is unknown.",
                   "Stationary 67.6 W is the observed total HV-bus-load proxy; the confirmed 21 W weak-system load is included within it and must not be added again.",
                   "D5 tire replacement and hatch-related interval are excluded from healthy cruise fit; periods remain in original joined CSV and event records.",
                   "The D6 fit is held out because its ZP anomaly was already documented; its positive bias is retained as a model adequacy warning."]}
    (OUT_MODEL).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    colors = {1:"#276FBF",2:"#F28E2B",3:"#59A14F",4:"#B07AA1",5:"#E15759",6:"#777777"}
    for day,g in seg.groupby("day"):
        ax.scatter(g.predicted_net_w,g.measured_net_w,s=34,label=f"Day {day}",color=colors.get(int(day)),alpha=.82)
    lo=float(min(seg.predicted_net_w.min(),seg.measured_net_w.min())); hi=float(max(seg.predicted_net_w.max(),seg.measured_net_w.max()))
    ax.plot([lo,hi],[lo,hi],"k--",lw=1); ax.set(xlabel="Modelled HV net power (W)",ylabel="Measured HV net power (W)",title="10 km cruise energy model; Day 6 held out")
    ax.legend(ncol=3,frameon=True); fig.tight_layout(); fig.savefig(OUT_FIG,dpi=170); plt.close(fig)
    cv_mae=float(np.mean([r["mae_w"] for r in cv_rows]))
    report=f"""# BWSC2025 vehicle energy model, initial MLE\n\nThe fitted target is the team-confirmed direct HV bus balance proxy, `Solar_Watt + Battery_Watt`. The sample is aggregated into **{len(seg)} 10 km segments**. The clean calibration uses Days 1–5 ({len(train)} segments); Day 6 ({len(d6)} segments) is held out because its ZP anomaly was recorded.\n\n## Fit and validation\n\n- Stopped-bus median used for total auxiliary baseline: **{p_aux:.1f} W**. The separately provided 21 W weak-system draw is a component and is included within this baseline; it is not added again.\n- Clean-day fit: MAE **{full_metrics['mae_w']:.1f} W**, RMSE **{full_metrics['rmse_w']:.1f} W**, mean bias **{full_metrics['bias_measured_minus_model_w']:+.1f} W**.\n- Leave-one-clean-day-out mean MAE: **{cv_mae:.1f} W**.\n- Day 6 held-out MAE: **{heldout_d6['mae_w']:.1f} W**, bias **{heldout_d6['bias_measured_minus_model_w']:+.1f} W**.\n- Fitted effective multipliers: rolling {model['parameters']['beta_roll']:.3f}, aero {model['parameters']['beta_aero']:.3f}, endpoint grade {model['parameters']['beta_grade']:.3f}.\n\nThese multipliers are **not true physical parameter estimates**. The grade coefficient differs materially from unity despite a 314 kg floor mass. Since mass, Crr, CdA, motor/controller efficiency, DEM grade, regen behavior, and bus metering are confounded in the available channels, the data do not support fitting each as an independent truth. Treat the coefficient as a correction to the joined route and energy model. The elevated Day 6 residual further means the model is an initial cruise predictor, not yet safe as an unguarded MPC command.\n\n## Day 5 handling\n\nThe documented 10:31–10:37 tire change and 11:51–12:50 hatch-flap/detachment/repair window are excluded from the healthy-road-load fit. Day 5 records, events, and weather remain in the integrated source log. The tire change may alter rolling resistance; hatch loss/flapping can change drag and panel exposure, so neither interval should train a single steady-state model.\n\n## Model evidence and limits\n\nThe road-load terms follow the standard longitudinal balance of rolling, aerodynamic, grade, and inertial forces. This early segment fit omits acceleration energy by restricting it to long, mostly steady 10 km bins. The vehicle uses the supplied theoretical CdA 0.08876 m² and Crr 0.0035 as priors, 234 kg dry vehicle plus the 80 kg regulatory occupant/ballast minimum as a 314 kg floor proxy, and the provided tire diameter. The ECO_LO bench curve is retained as a motor/controller prior, not equated to in-car PWM+ECO operation.\n\nResolution comparison (leave-one-clean-day-out) is in `vehicle_power_model_fit.json`; finer bins increase residual noise. The present 10 km segment model is appropriate only as an aggregate calibration/check, not yet as the MPC's short-horizon power map.\n\nSee `vehicle_power_cv_by_day.csv`, `vehicle_energy_segments_10km.csv`, and `vehicle_power_model_diagnostics.png` for row-level segment predictions and residuals.\n"""
    report = report.replace(
        "The ECO_LO bench curve is retained as a motor/controller prior, not equated to in-car PWM+ECO operation.",
        "The ECO_LO bench curve is retained as a motor/controller prior, not equated to in-car PWM+ECO operation. "
        f"The workbook reports {motor_prior['efficiency_at_1000w_input_pct']:.2f}% at 1 kW input; "
        "the installed serial and PWM+ECO controller-state equivalence are not confirmed. With the supplied "
        "558.4 mm tyre and direct drive, bench motor rpm is converted to vehicle speed in "
        "`models/m2096_eco_lo_bench_map.csv`.")
    report = report.replace(
        "These multipliers are **not true physical parameter estimates**.",
        "If the bench 1 kW ECO_LO efficiency were imposed, the multipliers would imply CdA "
        f"{conditional_equivalents['effective_cd_a_m2_if_eco_lo_1kw_efficiency_applies']:.3f} m², "
        f"Crr {conditional_equivalents['effective_crr_if_eco_lo_1kw_efficiency_applies']:.4f}, and a "
        f"grade-equivalent mass of only {conditional_equivalents['grade_equivalent_mass_kg_if_eco_lo_1kw_efficiency_applies']:.0f} kg. "
        "That last value is below the 234 kg dry vehicle mass and is physically impossible; it diagnoses route-grade/DEM, regeneration, telemetry, or model mismatch. "
        "These multipliers are **not true physical parameter estimates**.")
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"segments":len(seg),"train_segments":len(train),"day6_segments":len(d6),"p_aux_w":p_aux,
                      "fit":full_metrics,"lodo_mean_mae_w":cv_mae,"day6":heldout_d6,
                      "parameters":model["parameters"],"outputs":[str(OUT_SEGMENTS),str(OUT_MODEL),str(OUT_CV),str(OUT_FIG),str(OUT_REPORT)]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
