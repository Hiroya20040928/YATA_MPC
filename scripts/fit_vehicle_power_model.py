#!/usr/bin/env python3
"""Fit an interpretable cruise-power model to BWSC2025 telemetry.

The fitted target is Solar_Watt + Battery_Watt, the measured HV-bus load proxy.
Design values are priors.  The returned component multipliers deliberately remain
effective coefficients because road-load scale and drivetrain efficiency are not
separately identifiable from these telemetry channels.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/race_days_distance_geo_weather.csv"
ROUTE = ROOT / "data/input/route/route_100m_dem.csv"
MOTOR = ROOT / "data/input/design/M2096_characterization_user_supplied.xlsx"
EVENTS = ROOT / "data/input/route/events.csv"
OUT_MODEL = ROOT / "models/vehicle_power_model_fit.json"
OUT_DIAG = ROOT / "reports/vehicle_power_model_diagnostics.png"
OUT_CV = ROOT / "reports/vehicle_power_cv_by_day.csv"
OUT_CURVE = ROOT / "models/m2096_eco_lo_bench_map.csv"

MASS_NOMINAL_KG = 314.0  # 234 kg car + 80 kg regulation occupant/ballast floor
CD_A_NOMINAL_M2 = 0.08876
CRR_NOMINAL = 0.0035
WEAK_AUX_W = 21.0
TIRE_DIAMETER_M = 0.5584
G = 9.80665
R_DRY_AIR = 287.05
STUDENT_DF = 4.0


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2, clockwise from true north."""
    a1, a2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dl = np.deg2rad(lon2 - lon1)
    y = np.sin(dl) * np.cos(a2)
    x = np.cos(a1) * np.sin(a2) - np.sin(a1) * np.cos(a2) * np.cos(dl)
    return (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create wheel-road-load terms with forecast weather and route geometry."""
    d = df.copy()
    d["timestamp_local"] = pd.to_datetime(d["timestamp_local"], errors="coerce")
    d["speed_mps"] = pd.to_numeric(d["Speed[kmh]"], errors="coerce") / 3.6
    d["net_power_w"] = pd.to_numeric(d["Solar_Watt[W]"], errors="coerce") + pd.to_numeric(
        d["Battery_Watt[W]"], errors="coerce"
    )
    # Heading is reconstructed from the official route track, not a vehicle compass.
    route = pd.read_csv(ROUTE)
    s = route["distance_m"].to_numpy(float)
    lat = route["lat"].to_numpy(float)
    lon = route["lon"].to_numpy(float)
    dist = pd.to_numeric(d["distance_from_darwin_m"], errors="coerce").to_numpy(float)
    dminus, dplus = np.maximum(0.0, dist - 100.0), np.minimum(s[-1], dist + 100.0)
    lat1, lat2 = np.interp(dminus, s, lat), np.interp(dplus, s, lat)
    lon1, lon2 = np.interp(dminus, s, lon), np.interp(dplus, s, lon)
    d["route_heading_deg"] = bearing_deg(lat1, lon1, lat2, lon2)

    t_air = pd.to_numeric(d["temperature_2m"], errors="coerce").to_numpy(float)
    pressure_hpa = pd.to_numeric(d["surface_pressure"], errors="coerce").to_numpy(float)
    d["air_density_kgm3"] = pressure_hpa * 100.0 / (R_DRY_AIR * (t_air + 273.15))
    wind = pd.to_numeric(d["wind_speed_10m"], errors="coerce").to_numpy(float) / 3.6
    wind_from = pd.to_numeric(d["wind_direction_10m"], errors="coerce").to_numpy(float)
    # Open-Meteo direction is the direction wind comes FROM; positive cosine is headwind.
    headwind = wind * np.cos(np.deg2rad(wind_from - d["route_heading_deg"].to_numpy(float)))
    v = d["speed_mps"].to_numpy(float)
    u_rel = v + headwind
    d["relative_air_speed_mps"] = u_rel
    rho = d["air_density_kgm3"].to_numpy(float)
    # Signed longitudinal aerodynamic work allows a sufficiently strong tailwind to assist.
    d["aero_basis_w"] = 0.5 * rho * u_rel * np.abs(u_rel) * v * CD_A_NOMINAL_M2

    grade = pd.to_numeric(d["route_grade_smoothed_pct"], errors="coerce").to_numpy(float) / 100.0
    d["grade_fraction"] = grade
    d["roll_basis_w"] = MASS_NOMINAL_KG * G * CRR_NOMINAL * np.cos(np.arctan(grade)) * v
    d["grade_basis_w"] = MASS_NOMINAL_KG * G * np.sin(np.arctan(grade)) * v

    # Smooth the integer-km/h signal before differentiation; acceleration is only used
    # as a stability filter here because 5 s, 1 km/h quantization is too coarse for power.
    acc = np.full(len(d), np.nan)
    for _, idx in d.groupby("day", sort=False).groups.items():
        ids = np.asarray(list(idx), dtype=int)
        gdf = d.loc[ids]
        secs = (gdf["timestamp_local"] - gdf["timestamp_local"].iloc[0]).dt.total_seconds().to_numpy(float)
        sp = gdf["speed_mps"].rolling(7, center=True, min_periods=3).median().to_numpy()
        if len(ids) >= 3:
            dt = np.diff(secs)
            deriv = np.diff(sp) / np.where(dt > 0, dt, np.nan)
            a = np.full(len(ids), np.nan)
            a[1:-1] = (sp[2:] - sp[:-2]) / (secs[2:] - secs[:-2])
            a[0], a[-1] = deriv[0], deriv[-1]
            # A logging gap must not be interpreted as a true acceleration event.
            bad = np.zeros(len(ids), dtype=bool)
            bad[0] = bad[-1] = True
            gap = dt > 15.0
            bad[:-1] |= gap
            bad[1:] |= gap
            a[bad] = np.nan
            acc[ids] = a
    d["acceleration_smoothed_mps2"] = acc
    d["wheel_rpm"] = v / (np.pi * TIRE_DIAMETER_M) * 60.0
    return d


def build_motor_map() -> dict:
    """Export and summarize the closest available 2023 ECO low-speed bench map."""
    xls = pd.ExcelFile(MOTOR)
    df = pd.read_excel(xls, sheet_name="ECO_LO", header=None)
    values = {}
    for r in range(len(df)):
        label = str(df.iat[r, 2]) if pd.notna(df.iat[r, 2]) else ""
        if label in {"Udc(V)", "Ｉdc(A)", "Pin(W)", "Ｎ(rpm)", "Ｔ(N-m)", "Pm(W)", "ηt(%)"}:
            values[label] = pd.to_numeric(df.iloc[r, 3:19], errors="coerce").to_numpy(float)
    speed_index = next(
        r for r in range(len(df))
        if str(df.iat[r, 1]).startswith("BRIDGESTONE")
    )
    speed_row = pd.to_numeric(df.iloc[speed_index, 3:19], errors="coerce").to_numpy(float)
    data = pd.DataFrame(
        {
            "bench_mode": "ECO_LO",
            "bench_reported_vehicle_speed_kmh": speed_row,
            "input_power_w": values["Pin(W)"],
            "motor_rpm": values["Ｎ(rpm)"],
            "vehicle_speed_kmh_558_4mm_tire": values["Ｎ(rpm)"] * np.pi * TIRE_DIAMETER_M * 60.0 / 1000.0,
            "wheel_rpm_558_4mm_tire": values["Ｎ(rpm)"],
            "torque_nm": values["Ｔ(N-m)"],
            "mechanical_output_w": values["Pm(W)"],
            "total_efficiency_pct": values["ηt(%)"],
        }
    )
    data.to_csv(OUT_CURVE, index=False)
    # The sheet has a fixed-command torque-speed curve, not a full 2-D efficiency map.
    return {
        "sheet": "ECO_LO",
        "motor_serial": "M2096D-III-152",
        "controller_serial": "Mitsuba retained controller; not confirmed as installed unit",
        "date": "2023-09-15",
        "measurement_uncertainty_power_pct": 0.05,
        "measurement_uncertainty_torque_pct": 0.1,
        "efficiency_at_800w_input_pct": float(np.interp(800.0, data.input_power_w, data.total_efficiency_pct)),
        "efficiency_at_1000w_input_pct": float(np.interp(1000.0, data.input_power_w, data.total_efficiency_pct)),
        "efficiency_at_1200w_input_pct": float(np.interp(1200.0, data.input_power_w, data.total_efficiency_pct)),
        "speed_range_kmh_as_reported_by_bench": [float(data.bench_reported_vehicle_speed_kmh.min()), float(data.bench_reported_vehicle_speed_kmh.max())],
        "vehicle_speed_range_kmh_at_user_tire_diameter": [float(data.vehicle_speed_kmh_558_4mm_tire.min()), float(data.vehicle_speed_kmh_558_4mm_tire.max())],
        "notes": "Closest ECO bench curve. It is a one-dimensional full-command curve; it is not a PWM-ECO operating surface and is used only as a prior/reference.",
    }


def sample_table(d: pd.DataFrame) -> pd.DataFrame:
    """Restrict to stable, ordinary on-road cruise rows and aggregate by minute."""
    events = pd.read_csv(EVENTS).fillna("")
    d = d.copy()
    d["event_extra"] = d["event_flags"].fillna("").str.contains(
        r"incident:|control_stop:|traffic_stop:|driver_change:|repair:|charge:", regex=True
    )
    # D5's hatch off and repair intervals materially alter CdA; tire replacement is also
    # excluded because installed tire state/pressure was not logged.
    ts = d["timestamp_local"]
    d["manual_exclusion"] = False
    for day, start, end in [
        (5, "2025-08-28 10:31:00", "2025-08-28 10:37:00"),
        (5, "2025-08-28 11:51:00", "2025-08-28 12:50:00"),
    ]:
        d.loc[(d["day"] == day) & (ts >= start) & (ts <= end), "manual_exclusion"] = True
    # MPPT fault is retained: Solar_Watt + Battery_Watt is the direct HV-bus balance
    # proxy, so the PV split changes but the total load remains observed.
    # Known control/traffic/incident rows are removed via event_phase/flags.
    use = (
        d["event_phase"].eq("road_day")
        & ~d["event_extra"]
        & ~d["manual_exclusion"]
        & d["Speed[kmh]"].between(50, 85)
        & d["acceleration_smoothed_mps2"].abs().le(0.075)
        & d["distance_anchor_confidence"].isin(["high", "medium"])
        & d["net_power_w"].between(100.0, 2200.0)
        & d[["aero_basis_w", "roll_basis_w", "grade_basis_w"]].notna().all(axis=1)
    )
    d = d.loc[use].copy()
    d["minute_local"] = d["timestamp_local"].dt.floor("min")
    group_cols = ["day", "minute_local"]
    out = (
        d.groupby(group_cols, as_index=False)
        .agg(
            measured_net_w=("net_power_w", "mean"),
            speed_kmh=("Speed[kmh]", "mean"),
            aero_basis_w=("aero_basis_w", "mean"),
            roll_basis_w=("roll_basis_w", "mean"),
            grade_basis_w=("grade_basis_w", "mean"),
            grade_fraction=("grade_fraction", "mean"),
            n_5s=("net_power_w", "size"),
        )
    )
    out = out.loc[out.n_5s.ge(3)].reset_index(drop=True)
    return out


def fit_mle(x: pd.DataFrame, use_grade: bool = True) -> dict:
    """Student-t MLE for the effective physical component coefficients."""
    names = ["aux_total_w", "beta_roll", "beta_aero"] + (["beta_grade"] if use_grade else [])
    cols = ["roll_basis_w", "aero_basis_w"] + (["grade_basis_w"] if use_grade else [])
    X = x[cols].to_numpy(float)
    y = x["measured_net_w"].to_numpy(float)

    def pred(p):
        return p[0] + X @ p[1:-1]

    def nll(p):
        mu = pred(p)
        scale = max(p[-1], 1.0)
        return float(-np.sum(student_t.logpdf((y - mu) / scale, df=STUDENT_DF) - np.log(scale)))

    n_beta = len(cols)
    init = np.r_[60.0, np.ones(n_beta), 140.0]
    bounds = [(0.0, 300.0)] + [(0.0, 5.0)] * n_beta + [(10.0, 1000.0)]
    fit = minimize(nll, init, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000, "ftol": 1e-10})
    p = fit.x
    return {
        "success": bool(fit.success),
        "message": str(fit.message),
        "parameters": {n: float(v) for n, v in zip(names + ["student_t_scale_w"], p)},
        "df": STUDENT_DF,
        "n": len(x),
        "negative_log_likelihood": float(fit.fun),
    }


def predict(x: pd.DataFrame, fit: dict, use_grade: bool = True) -> np.ndarray:
    p = fit["parameters"]
    y = p["aux_total_w"] + p["beta_roll"] * x.roll_basis_w.to_numpy(float) + p["beta_aero"] * x.aero_basis_w.to_numpy(float)
    if use_grade:
        y += p["beta_grade"] * x.grade_basis_w.to_numpy(float)
    return y


def metrics(y, yhat):
    e = np.asarray(y) - np.asarray(yhat)
    return {
        "n": int(len(e)),
        "mae_w": float(np.mean(np.abs(e))),
        "rmse_w": float(np.sqrt(np.mean(e**2))),
        "bias_measured_minus_model_w": float(np.mean(e)),
        "p90_abs_error_w": float(np.quantile(np.abs(e), 0.9)),
    }


def main():
    d = pd.read_csv(INPUT, low_memory=False)
    d = make_features(d)
    x = sample_table(d)
    if len(x) < 200:
        raise RuntimeError(f"Too few eligible minute bins: {len(x)}")

    fits = {"with_grade": fit_mle(x, True), "flat_road": fit_mle(x, False)}
    cv = []
    for day in sorted(x.day.unique()):
        tr, te = x.loc[x.day != day], x.loc[x.day == day]
        for name, use_grade in [("with_grade", True), ("flat_road", False)]:
            f = fit_mle(tr, use_grade)
            yp = predict(te, f, use_grade)
            cv.append({"held_out_day": int(day), "model": name, **metrics(te.measured_net_w, yp), "train_n": int(len(tr))})
    cvdf = pd.DataFrame(cv)
    cvdf.to_csv(OUT_CV, index=False)
    best_name = (
        cvdf.groupby("model").mae_w.mean().sort_values().index[0]
    )
    use_grade = best_name == "with_grade"
    selected = fits[best_name]
    yhat = predict(x, selected, use_grade)
    full_metrics = metrics(x.measured_net_w, yhat)
    by_day = []
    for day, g in x.assign(pred=yhat).groupby("day"):
        by_day.append({"day": int(day), **metrics(g.measured_net_w, g.pred)})

    motor = build_motor_map()
    param = selected["parameters"]
    eta_ref = motor["efficiency_at_1000w_input_pct"] / 100.0
    effective = {
        "mass_nominal_kg": MASS_NOMINAL_KG,
        "cd_a_theoretical_m2": CD_A_NOMINAL_M2,
        "crr_theoretical": CRR_NOMINAL,
        "auxiliary_measured_weak_load_w": WEAK_AUX_W,
        "motor_bench_reference_efficiency_at_1kw_input": eta_ref,
        "effective_cd_a_if_eta_fixed_to_bench": float(CD_A_NOMINAL_M2 * param["beta_aero"] * eta_ref),
        "effective_crr_if_eta_fixed_to_bench": float(CRR_NOMINAL * param["beta_roll"] * eta_ref),
        "effective_mass_ratio_if_eta_fixed_to_bench": float(param.get("beta_grade", np.nan) * eta_ref),
        "warning": "Effective fitted coefficients combine true road-load parameters, motor/controller efficiency, wind/grade/weather error, and metering bias. They are not independent physical truth estimates.",
    }
    result = {
        "target": "Solar_Watt[W] + Battery_Watt[W] (net HV load proxy)",
        "source_rows": int(len(d)),
        "eligible_minute_bins": int(len(x)),
        "eligible_days": [int(v) for v in sorted(x.day.unique())],
        "aggregation": "1-minute means of stable 5-second race-day telemetry; speed 50-85 km/h, |smoothed acceleration|<=0.075 m/s2; event and anchor-uncertain rows removed",
        "method": "constrained Student-t maximum likelihood, df=4; leave-one-day-out evaluation",
        "selected_model": best_name,
        "fits": fits,
        "selected_full_sample_metrics": full_metrics,
        "leave_one_day_out_mean_mae_w": {str(k): float(v) for k, v in cvdf.groupby("model").mae_w.mean().items()},
        "leave_one_day_out_mean_bias_w": {str(k): float(v) for k, v in cvdf.groupby("model").bias_measured_minus_model_w.mean().items()},
        "by_day_in_sample_metrics": by_day,
        "effective_parameter_interpretation": effective,
        "motor_bench_reference": motor,
        "nominal_constants": {"mass_kg": MASS_NOMINAL_KG, "cd_a_m2": CD_A_NOMINAL_M2, "crr": CRR_NOMINAL, "weak_aux_w": WEAK_AUX_W, "tire_outer_diameter_m": TIRE_DIAMETER_M},
        "known_exclusions": ["control stops, charging/traffic stops, incidents and repairs", "Day5 10:31-10:37 puncture/tire change", "Day5 11:51-12:50 hatch failure/repair; hatch changes aerodynamic shape", "low-confidence distance segments"],
    }
    OUT_MODEL.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(x.measured_net_w, yhat, c=x.day, cmap="viridis", s=12, alpha=0.55)
    lim = [0, max(x.measured_net_w.max(), yhat.max()) * 1.03]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(xlim=lim, ylim=lim, xlabel="Measured HV load proxy [W]", ylabel="Model [W]", title=f"Selected: {best_name}")
    ax[1].boxplot([x.loc[x.day == day, "measured_net_w"].to_numpy() - yhat[x.day.to_numpy() == day] for day in sorted(x.day.unique())], labels=[f"D{day}" for day in sorted(x.day.unique())], showfliers=False)
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set(ylabel="Measured − model [W]", title="In-sample residual by day")
    fig.tight_layout()
    fig.savefig(OUT_DIAG, dpi=170)
    plt.close(fig)
    print(json.dumps({"selected_model": best_name, "full_metrics": full_metrics, "cv_mae": result["leave_one_day_out_mean_mae_w"], "cv_bias": result["leave_one_day_out_mean_bias_w"], "params": selected["parameters"], "effective": effective, "cv_by_day": cvdf.to_dict(orient="records")}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    raise SystemExit("Superseded for model fitting. Run scripts/fit_vehicle_energy_segments.py; this module remains a shared feature and motor-map helper.")
