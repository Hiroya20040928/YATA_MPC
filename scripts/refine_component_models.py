#!/usr/bin/env python3
"""Day-blocked PV, battery and full-race replay for BWSC2025.

PV is the combined panel+MPPT HV output. Battery is a pack-level voltage,
current and coulomb-count model. Road load is replayed at every source sample
using the existing 10 km MLE coefficients, but parameters are refit without
the day being replayed. A 5 s residual is diagnostic, not an independent
training observation or a claim of 5 s predictive accuracy.
"""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fit_pv_power_model_physical as pv  # noqa: E402
import fit_vehicle_energy_segments as vehicle  # noqa: E402
import fit_vehicle_power_model as features  # noqa: E402

INPUT = ROOT / "data/processed/race_days_distance_geo_weather.csv"
SEGMENTS = ROOT / "data/processed/vehicle_energy_segments_10km.csv"
PV_HOURLY = ROOT / "data/processed/pv_model_hourly_diagnostic.csv"
PV_MAP = ROOT / "models/pv_model_fit.json"
VEHICLE_MAP = ROOT / "models/vehicle_power_model_fit.json"
BATTERY_TEST = ROOT / "data/input/design/Battery_discharge_characterization.xlsx"
BATTERY_SOC_MAP = ROOT / "data/input/design/Battery_SOC_voltage_mapping.xlsx"

OUT_REPLAY = ROOT / "data/processed/full_race_component_replay_5s.csv"
OUT_SEGMENT_REPLAY = ROOT / "data/processed/full_race_component_replay_10km.csv"
OUT_PV_FOLDS = ROOT / "reports/pv_temperature_model_cv.csv"
OUT_BATTERY_FOLDS = ROOT / "reports/battery_voltage_model_cv.csv"
OUT_DAILY = ROOT / "reports/component_replay_by_day.csv"
OUT_BATTERY_MAP = ROOT / "models/battery_voltage_soc_map.csv"
OUT_BATTERY_R = ROOT / "models/battery_effective_resistance_map.csv"
OUT_FIT = ROOT / "models/component_replay_fit.json"
OUT_REPORT = ROOT / "reports/BWSC2025_component_model_iteration.md"
OUT_FIG = ROOT / "reports/component_replay_diagnostics.png"

TZ = "Australia/Darwin"
DF = 4.0
Q_CANDIDATES_AH = [31.0, 32.0, 33.0, 33.164, 34.0, 34.597, 35.0, 36.0]
PV_THERMAL_U0 = [20.0, 25.0, 30.0]
PV_THERMAL_U1 = 6.84
BATTERY_SOC_PTS: np.ndarray | None = None
BATTERY_VOLT_PTS: np.ndarray | None = None
BATTERY_R_SOC_PTS: np.ndarray | None = None
BATTERY_R_PTS: np.ndarray | None = None


def score(y, yhat) -> dict:
    a, b = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(a) & np.isfinite(b)
    e = a[ok] - b[ok]
    return {
        "n": int(len(e)),
        "mae": float(np.mean(np.abs(e))) if len(e) else None,
        "rmse": float(np.sqrt(np.mean(e**2))) if len(e) else None,
        "bias_measured_minus_model": float(np.mean(e)) if len(e) else None,
        "p90_abs_error": float(np.quantile(np.abs(e), 0.9)) if len(e) else None,
    }


def temp_cell_c(x: pd.DataFrame, u0: float, u1: float = PV_THERMAL_U1) -> np.ndarray:
    """Faiman-style empirical module-temperature estimate; priors, not fitted here."""
    poa_roof = pv.poa_tools.poa_irradiance(x, 0.0, 0.0)
    wind_mps = pd.to_numeric(x.wind_speed_10m, errors="coerce").to_numpy(float) / 3.6
    ambient = pd.to_numeric(x.temp_c, errors="coerce").to_numpy(float)
    return ambient + np.maximum(poa_roof, 0.0) / (u0 + u1 * np.maximum(wind_mps, 0.0))


def fit_pv_thermal(x: pd.DataFrame, u0: float) -> dict:
    """Fit one temperature multiplier on top of training-fold facet gains."""
    base_fit = pv.fit(x, "faceted_body_poa")
    base = pv.prediction(x, base_fit["parameters"], "faceted_body_poa")
    temp = temp_cell_c(x, u0)
    y = x.pv_w.to_numpy(float)

    def nll(z):
        gamma, scale = z
        pred = base * np.maximum(0.0, 1.0 + gamma * (temp - 25.0))
        return float(-np.sum(student_t.logpdf((y - pred) / scale, df=DF) - np.log(scale)))

    opt = minimize(nll, [-0.002, max(float(np.std(y - base)), 5.0)], method="L-BFGS-B",
                   bounds=[(-0.02, 0.01), (1.0, 600.0)], options={"maxiter": 500, "ftol": 1e-10})
    gamma, scale = map(float, opt.x)
    return {"kind": "faceted_body_poa_faiman_temp", "u0": float(u0), "u1": float(PV_THERMAL_U1),
            "gamma_per_c": gamma, "scale_w": scale, "base_parameters": base_fit["parameters"],
            "success": bool(opt.success)}


def predict_pv_thermal(x: pd.DataFrame, model: dict) -> np.ndarray:
    base = pv.prediction(x, model["base_parameters"], "faceted_body_poa")
    temp = temp_cell_c(x, model["u0"], model["u1"])
    return base * np.maximum(0.0, 1.0 + model["gamma_per_c"] * (temp - 25.0))


def pv_cv(hourly: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    eligible = hourly[hourly.fit_eligible.astype(bool)].dropna(
        subset=["pv_w", "ghi", "temp_c", "route_latitude", "route_longitude",
                "direct_normal_irradiance", "diffuse_radiation", "wind_speed_10m"]
    ).copy()
    candidates = []
    fold_predictions: dict[tuple[int, float], dict] = {}
    folds = sorted(int(d) for d in eligible.day.unique())
    for day in folds:
        train, test = eligible[eligible.day.ne(day)].copy(), eligible[eligible.day.eq(day)].copy()
        base = pv.fit(train, "faceted_body_poa")
        base_pred = pv.prediction(test, base["parameters"], "faceted_body_poa")
        candidates.append({"u0": "baseline_no_temp", "held_out_day": day, **score(test.pv_w, base_pred),
                           "gamma_per_c": 0.0, "model": "faceted_body_poa"})
        for u0 in PV_THERMAL_U0:
            m = fit_pv_thermal(train, u0)
            pred = predict_pv_thermal(test, m)
            candidates.append({"u0": u0, "held_out_day": day, **score(test.pv_w, pred),
                               "gamma_per_c": m["gamma_per_c"], "model": "faceted_body_poa_faiman_temp"})
            fold_predictions[(day, u0)] = m

    cv = pd.DataFrame(candidates)
    means = cv.groupby(["model", "u0"], dropna=False, sort=False).agg(
        mean_day_mae_w=("mae", "mean"), mean_day_rmse_w=("rmse", "mean"),
        mean_day_bias_w=("bias_measured_minus_model", "mean"), n_days=("held_out_day", "nunique")
    ).reset_index()
    base_mae = float(means.loc[means.model.eq("faceted_body_poa"), "mean_day_mae_w"].iloc[0])
    thermal = means[means.model.eq("faceted_body_poa_faiman_temp")].sort_values("mean_day_mae_w")
    chosen_u0 = float(thermal.iloc[0].u0)
    chosen_gamma = [float(fold_predictions[(d, chosen_u0)]["gamma_per_c"]) for d in folds]
    thermal_mae = float(thermal.iloc[0].mean_day_mae_w)
    stable_sign = bool(all(v <= 0 for v in chosen_gamma) or all(v >= 0 for v in chosen_gamma))
    not_at_boundary = bool(all(abs(v) < 0.009 for v in chosen_gamma))
    selected = "faceted_body_poa_faiman_temp" if (
        thermal_mae <= 0.98 * base_mae and stable_sign and not_at_boundary
    ) else "faceted_body_poa"
    final_model = fit_pv_thermal(eligible, chosen_u0) if selected.endswith("faiman_temp") else pv.fit(eligible, "faceted_body_poa")
    final_model["selected_model"] = selected
    final_model["cv_diagnostics"] = {
        "baseline_lodo_mean_mae_w": base_mae,
        "best_thermal_lodo_mean_mae_w": thermal_mae,
        "best_thermal_u0_w_m2c": chosen_u0,
        "u1_ws_w_s_m3c": PV_THERMAL_U1,
        "fold_gamma_per_c": chosen_gamma,
        "gamma_sign_stable": stable_sign,
        "gamma_not_at_bound": not_at_boundary,
        "selection_rule": "Thermal map adopted only if day-blocked MAE improves by at least 2%, fold gamma sign is stable, and no fold is near the imposed bound.",
    }
    return cv, final_model, {"eligible_bins": int(len(eligible)), "means": means.to_dict(orient="records"),
                             "fold_days": folds, "selected_model": selected,
                             **final_model["cv_diagnostics"]}


def pv_fold_models(hourly: pd.DataFrame, selected: str, chosen_u0: float) -> dict[int, dict]:
    eligible = hourly[hourly.fit_eligible.astype(bool)].dropna(
        subset=["pv_w", "ghi", "temp_c", "route_latitude", "route_longitude",
                "direct_normal_irradiance", "diffuse_radiation", "wind_speed_10m"]
    ).copy()
    models = {}
    for day in sorted(int(x) for x in hourly.day.unique()):
        if day == 6:
            train = eligible[eligible.day.ne(6)].copy()
        else:
            train = eligible[eligible.day.ne(day)].copy()
        if selected.endswith("faiman_temp"):
            models[day] = fit_pv_thermal(train, chosen_u0)
        else:
            models[day] = pv.fit(train, "faceted_body_poa")
    return models


def make_lab_battery_maps() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    q = pd.read_excel(BATTERY_SOC_MAP)
    map_df = q[["SOC(%)", "25セル電圧(V)"]].copy().rename(
        columns={"SOC(%)": "soc_pct", "25セル電圧(V)": "reference_voltage_25s_v"})
    map_df = map_df.dropna().sort_values("soc_pct")
    g8 = pd.read_excel(BATTERY_TEST, sheet_name="8Aグラフ", header=None).iloc[4:25, 0:3].copy()
    g10 = pd.read_excel(BATTERY_TEST, sheet_name="10Aグラフ", header=None).iloc[4:25, 0:3].copy()
    g8.columns = ["soc_pct", "capacity_ah", "voltage_2s_6p_v"]
    g10.columns = ["soc_pct", "capacity_ah", "voltage_2s_6p_v"]
    for x in (g8, g10):
        for c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
        x.dropna(inplace=True)
        # np.interp requires an increasing x-axis; the supplied discharge tables
        # are ordered from 100% SOC down to 0% SOC.
        x.sort_values("soc_pct", inplace=True)
    curve = map_df[["soc_pct"]].drop_duplicates().sort_values("soc_pct")
    curve["voltage_8a_25s_v"] = np.interp(curve.soc_pct, g8.soc_pct, g8.voltage_2s_6p_v) * 25.0 / 2.0
    curve["voltage_10a_25s_v"] = np.interp(curve.soc_pct, g10.soc_pct, g10.voltage_2s_6p_v) * 25.0 / 2.0
    curve["differential_resistance_8_to_10a_ohm"] = (
        curve.voltage_8a_25s_v - curve.voltage_10a_25s_v
    ) / 2.0
    curve["resistance_usable_nonnegative"] = (
        curve.differential_resistance_8_to_10a_ohm.between(0.0, 1.5)
        & curve.soc_pct.between(5.0, 95.0)
    )
    curve.to_csv(OUT_BATTERY_R, index=False, encoding="utf-8-sig")
    map_df.to_csv(OUT_BATTERY_MAP, index=False, encoding="utf-8-sig")
    g8_energy = g8.sort_values("capacity_ah")
    g10_energy = g10.sort_values("capacity_ah")
    e8_wh = float(np.trapezoid(g8_energy.voltage_2s_6p_v.to_numpy(float), g8_energy.capacity_ah.to_numpy(float)) * 25.0 / 2.0)
    e10_wh = float(np.trapezoid(g10_energy.voltage_2s_6p_v.to_numpy(float), g10_energy.capacity_ah.to_numpy(float)) * 25.0 / 2.0)
    summary = {
        "pack_basis": "25S6P user-provided configuration; two-series six-parallel lab discharge curves scaled by 25/2 for pack voltage.",
        "lab_capacity_8a_ah": float(pd.to_numeric(pd.read_excel(BATTERY_TEST, sheet_name="8Aグラフ", header=None).iat[1, 1])),
        "lab_capacity_10a_ah": float(pd.to_numeric(pd.read_excel(BATTERY_TEST, sheet_name="10Aグラフ", header=None).iat[1, 1])),
        "design_capacity_ah": 33.0,
        "user_theoretical_pack_energy_wh": 3011.0,
        "lab_8a_integrated_energy_25s6p_wh": e8_wh,
        "lab_10a_integrated_energy_25s6p_wh": e10_wh,
        "lab_10a_energy_minus_user_theoretical_wh": e10_wh - 3011.0,
        "lab_10a_energy_difference_pct": 100.0 * (e10_wh / 3011.0 - 1.0),
        "capacity_selection_note": "The 10 A curve's integrated 25S6P energy matches the user's 3011 Wh design basis. Prefer its measured 33.164 Ah capacity unless voltage-only LODO improves by more than one logged Main_Volt resolution step (0.1 V); per-day initial SOC and OCV-map error otherwise confound capacity.",
        "map_points": int(len(map_df)),
        "map_source_note": "5%-step 25-cell voltage/SoC table from supplied battery materials; treated as empirical reference curve, not certified OCV.",
        "lab_curve_note": "8 A and 10 A discharge-voltage curves are rate-specific; their difference yields only a quasi-static differential-resistance proxy, not pulse-test R0.",
        "positive_r_points": int(curve.resistance_usable_nonnegative.sum()),
        "r_proxy_median_ohm": float(curve.loc[curve.resistance_usable_nonnegative, "differential_resistance_8_to_10a_ohm"].median()),
    }
    global BATTERY_SOC_PTS, BATTERY_VOLT_PTS, BATTERY_R_SOC_PTS, BATTERY_R_PTS
    BATTERY_SOC_PTS = map_df.soc_pct.to_numpy(float)
    BATTERY_VOLT_PTS = map_df.reference_voltage_25s_v.to_numpy(float)
    valid_r = curve[curve.resistance_usable_nonnegative]
    BATTERY_R_SOC_PTS = valid_r.soc_pct.to_numpy(float)
    BATTERY_R_PTS = valid_r.differential_resistance_8_to_10a_ohm.to_numpy(float)
    return map_df, curve, summary


def battery_anchor_and_state(g: pd.DataFrame, q_ah: float, r_ohm: float | dict, max_gap_s: float = 15.0) -> pd.DataFrame:
    d = g.sort_values("timestamp_utc").copy().reset_index(drop=True)
    t = d.timestamp_utc
    dt = t.diff().dt.total_seconds().fillna(0.0).to_numpy(float)
    cur = pd.to_numeric(d["Battery[A]"], errors="coerce").to_numpy(float)
    volt = pd.to_numeric(d["Main_Volt[V]"], errors="coerce").to_numpy(float)
    phase = d.event_phase.fillna("").astype(str).to_numpy()
    t_rel = (t - t.iloc[0]).dt.total_seconds().to_numpy(float)
    anchor_rows = np.flatnonzero((t_rel <= 180.0) & np.isfinite(volt) & (volt >= 70.0) & (volt <= 115.0)
                                 & np.isfinite(cur) & (np.abs(cur) <= 5.0))
    if len(anchor_rows) == 0:
        anchor_rows = np.flatnonzero(np.isfinite(volt) & (volt >= 70.0) & (volt <= 115.0) & np.isfinite(cur))[:20]
    if len(anchor_rows) == 0:
        out = d.copy()
        out["battery_soc_estimate_pct"] = np.nan
        out["battery_voltage_predicted_v"] = np.nan
        out["battery_anchor_voltage_v"] = np.nan
        out["battery_effective_r_ohm"] = np.nan
        return out
    # Use a robust short initial window; this suppresses the Day 3 startup zero/step.
    anchor_i = int(anchor_rows[len(anchor_rows) // 2])
    v0, i0 = float(volt[anchor_i]), float(cur[anchor_i])
    if BATTERY_SOC_PTS is None or BATTERY_VOLT_PTS is None:
        vmap = pd.read_csv(OUT_BATTERY_MAP)
        soc_pts = vmap.soc_pct.to_numpy(float)
        volt_pts = vmap.reference_voltage_25s_v.to_numpy(float)
    else:
        soc_pts, volt_pts = BATTERY_SOC_PTS, BATTERY_VOLT_PTS
    # V = Vref(SOC) - R(SOC)*I; I>0 means discharge, I<0 charge.
    if isinstance(r_ohm, dict):
        if BATTERY_R_SOC_PTS is None or BATTERY_R_PTS is None:
            raise RuntimeError("Load the lab resistance map before fitting an SOC-dependent R candidate.")
        r_scale = float(r_ohm["scale"])
        soc0 = float(np.interp(v0, volt_pts, soc_pts))
        for _ in range(8):
            r0 = r_scale * float(np.interp(soc0, BATTERY_R_SOC_PTS, BATTERY_R_PTS))
            soc0 = float(np.interp(v0 + r0 * i0, volt_pts, soc_pts))
    else:
        r_scale = None
        soc0 = float(np.interp(v0 + float(r_ohm) * i0, volt_pts, soc_pts))
    qstep = cur.copy()
    # Charge coulomb efficiency is not characterized; use unity as a transparent base case.
    qstep[~np.isfinite(qstep)] = 0.0
    da = 0.5 * (qstep[:-1] + qstep[1:]) * np.clip(dt[1:], 0.0, max_gap_s) / 3600.0
    cumulative_ah = np.r_[0.0, np.cumsum(da)]
    cumulative_ah -= cumulative_ah[anchor_i]
    soc = soc0 - cumulative_ah / q_ah * 100.0
    vref = np.interp(np.clip(soc, 0.0, 100.0), soc_pts, volt_pts)
    if r_scale is None:
        r_trace = np.full(len(soc), float(r_ohm), dtype=float)
    else:
        r_trace = r_scale * np.interp(np.clip(soc, 0.0, 100.0), BATTERY_R_SOC_PTS, BATTERY_R_PTS)
    vpred = vref - r_trace * cur
    out = d.copy()
    out["battery_soc_estimate_pct"] = soc
    out["battery_voltage_predicted_v"] = vpred
    out["battery_voltage_reference_v"] = vref
    out["battery_effective_r_ohm"] = r_trace
    out["battery_cumulative_ah_from_anchor"] = cumulative_ah
    out["battery_anchor_voltage_v"] = np.nan
    out.loc[anchor_i, "battery_anchor_voltage_v"] = v0
    out["battery_anchor_soc_pct"] = np.nan
    out.loc[anchor_i, "battery_anchor_soc_pct"] = soc0
    out["battery_model_valid_voltage"] = (
        np.isfinite(volt) & (volt >= 70.0) & (volt <= 115.0) & np.isfinite(cur)
        & (soc >= 0.0) & (soc <= 100.0)
    )
    out["battery_charge_voltage_extrapolation"] = cur < -0.25
    out["battery_event_phase"] = phase
    return out


def battery_validation_rows(d: pd.DataFrame, minutely_input: bool = False) -> pd.DataFrame:
    # One-minute medians avoid counting the 5 s autocorrelation as independent evidence.
    x = d.copy()
    x["battery_current_a"] = pd.to_numeric(x["Battery[A]"], errors="coerce")
    x["battery_voltage_v"] = pd.to_numeric(x["Main_Volt[V]"], errors="coerce")
    x["eligible"] = (x.event_phase.eq("road_day") & pd.to_numeric(x["Speed[kmh]"], errors="coerce").ge(30.0)
                     & x.battery_current_a.ge(0.25) & x.battery_voltage_v.between(70.0, 115.0)
                     & x.battery_model_valid_voltage)
    flags = x.event_flags.fillna("").astype(str)
    x.loc[flags.str.contains(r"incident:|repair:|D5_HATCH_FAILURE|D5_PUNCTURE|D6_ZP_ANOMALY|D6_RETIRE", regex=True), "eligible"] = False
    x["minute_local"] = pd.to_datetime(x.timestamp_local, utc=True).dt.tz_convert(TZ).dt.floor("min")
    if minutely_input:
        out = x[x.eligible].copy()
        out["minute_local"] = pd.to_datetime(out.timestamp_local, utc=True).dt.tz_convert(TZ).dt.floor("min")
        out["current_a"] = out.battery_current_a
        out["measured_voltage_v"] = out.battery_voltage_v
        out["predicted_voltage_v"] = out.battery_voltage_predicted_v
        out["soc_estimate_pct"] = out.battery_soc_estimate_pct
        out["n_5s"] = 1
        return out[["day", "minute_local", "current_a", "measured_voltage_v", "predicted_voltage_v", "soc_estimate_pct", "n_5s"]]
    cols = ["day", "minute_local", "eligible", "battery_current_a", "battery_voltage_v",
            "battery_voltage_predicted_v", "battery_soc_estimate_pct"]
    out = x.groupby(["day", "minute_local"], as_index=False)[cols[2:]].agg(
        eligible=("eligible", "max"), current_a=("battery_current_a", "median"),
        measured_voltage_v=("battery_voltage_v", "median"), predicted_voltage_v=("battery_voltage_predicted_v", "median"),
        soc_estimate_pct=("battery_soc_estimate_pct", "median"), n_5s=("battery_current_a", "size"))
    return out[out.eligible & out.n_5s.ge(6)].copy()


def battery_lodo(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict[int, dict]]:
    days = sorted(int(d) for d in raw.day.unique())
    # Fit on one-minute medians because that is the validation grain; full 5 s
    # rows are replayed only after parameter selection.
    minute_frames = {}
    qraw = raw.copy()
    qraw["minute_utc"] = qraw.timestamp_utc.dt.floor("min")
    qraw["Speed[kmh]"] = pd.to_numeric(qraw["Speed[kmh]"], errors="coerce")
    for day, g in qraw.groupby("day", sort=True):
        minute_frames[int(day)] = g.groupby(["day", "minute_utc"], as_index=False).agg(
            timestamp_utc=("timestamp_utc", "first"), timestamp_local=("timestamp_local", "first"),
            **{"Battery[A]": ("Battery[A]", "median"), "Main_Volt[V]": ("Main_Volt[V]", "median"),
               "Speed[kmh]": ("Speed[kmh]", "median"), "event_phase": ("event_phase", "first"),
               "event_flags": ("event_flags", "first")}
        )
    fold_rows, fold_models = [], {}
    profile_rows = []
    for qah in Q_CANDIDATES_AH:
        maes = []
        for held in [d for d in days if d <= 5]:
            trdays = [d for d in days if d <= 5 and d != held]
            # Optimize a single pack-level resistance with the empirical V-SOC map fixed.
            def objective(r):
                vals = []
                for td in trdays:
                    replay = battery_anchor_and_state(minute_frames[td], qah, float(r), max_gap_s=75.0)
                    x = battery_validation_rows(replay, minutely_input=True)
                    if len(x): vals.extend((x.measured_voltage_v - x.predicted_voltage_v).to_numpy(float))
                if len(vals) < 10:
                    return 1e9
                a = np.asarray(vals)
                scale = max(float(np.median(np.abs(a - np.median(a))) * 1.4826), 0.05)
                return float(-np.sum(student_t.logpdf((a - np.median(a)) / scale, df=DF)))
            fit = minimize_scalar(objective, bounds=(0.0, 0.8), method="bounded", options={"xatol": 1e-4})
            rfit = float(fit.x)
            replay_test = battery_anchor_and_state(minute_frames[held], qah, rfit, max_gap_s=75.0)
            test = battery_validation_rows(replay_test, minutely_input=True)
            metrics = score(test.measured_voltage_v, test.predicted_voltage_v)
            fold_rows.append({"capacity_ah": qah, "held_out_day": held, "r_eff_ohm": rfit, **metrics})
            maes.append(metrics["mae"] if metrics["mae"] is not None else np.inf)
        profile_rows.append({"capacity_ah": qah, "lodo_mean_mae_v": float(np.mean(maes)),
                             "lodo_mean_rmse_v": float(np.mean([r["rmse"] for r in fold_rows if r["capacity_ah"] == qah]))})
    profile = pd.DataFrame(profile_rows).sort_values("lodo_mean_mae_v")
    best = profile.iloc[0]
    # Prefer the measured 10 A test capacity if it is within 1% of the best LODO error.
    q_selected = float(best.capacity_ah)
    q10 = 33.164
    q10_row = profile[np.isclose(profile.capacity_ah, q10)]
    # A capacity-only MAE gain below one logged Main_Volt LSB (0.1 V) is not
    # sufficient evidence to overrule the direct 10 A discharge capacity.
    if len(q10_row) and float(q10_row.iloc[0].lodo_mean_mae_v) <= float(best.lodo_mean_mae_v) + 0.1:
        q_selected = q10
    # Full clean-day fit for the Day 6 held-out replay.
    def final_objective(r):
        vals = []
        for td in [d for d in days if d <= 5]:
            replay = battery_anchor_and_state(minute_frames[td], q_selected, float(r), max_gap_s=75.0)
            x = battery_validation_rows(replay, minutely_input=True)
            if len(x): vals.extend((x.measured_voltage_v - x.predicted_voltage_v).to_numpy(float))
        a = np.asarray(vals)
        scale = max(float(np.median(np.abs(a - np.median(a))) * 1.4826), 0.05)
        return float(-np.sum(student_t.logpdf((a - np.median(a)) / scale, df=DF)))
    final = minimize_scalar(final_objective, bounds=(0.0, 0.8), method="bounded", options={"xatol": 1e-4})
    r_final = float(final.x)
    # Coefficients for each out-of-day row prediction. D6 uses all clean Days 1-5.
    for day in days:
        if day <= 5:
            train_days = [d for d in days if d <= 5 and d != day]
            def fold_objective(r):
                vals = []
                for td in train_days:
                    x = battery_validation_rows(battery_anchor_and_state(minute_frames[td], q_selected, float(r), max_gap_s=75.0), minutely_input=True)
                    if len(x): vals.extend((x.measured_voltage_v - x.predicted_voltage_v).to_numpy(float))
                a = np.asarray(vals)
                scale = max(float(np.median(np.abs(a - np.median(a))) * 1.4826), 0.05)
                return float(-np.sum(student_t.logpdf((a - np.median(a)) / scale, df=DF)))
            ff = minimize_scalar(fold_objective, bounds=(0.0, 0.8), method="bounded", options={"xatol": 1e-4})
            fold_models[day] = {"capacity_ah": q_selected, "r_eff_ohm": float(ff.x)}
        else:
            fold_models[day] = {"capacity_ah": q_selected, "r_eff_ohm": r_final}
    cv = pd.DataFrame(fold_rows)
    profile["selected"] = np.isclose(profile.capacity_ah, q_selected)
    cv.to_csv(OUT_BATTERY_FOLDS, index=False, encoding="utf-8-sig")
    summary = {
        "capacity_profile": profile.to_dict(orient="records"),
        "selected_capacity_ah": q_selected,
        "measured_10a_capacity_ah": 33.164,
        "measured_8a_capacity_ah": 34.597,
        "selected_full_clean_day_r_eff_ohm": r_final,
        "cv_mean_mae_v": float(cv[cv.capacity_ah.eq(q_selected)].mae.mean()),
        "cv_mean_rmse_v": float(cv[cv.capacity_ah.eq(q_selected)].rmse.mean()),
        "cv_mean_bias_v": float(cv[cv.capacity_ah.eq(q_selected)].bias_measured_minus_model.mean()),
        "r_by_day_cv": {str(d): v["r_eff_ohm"] for d, v in fold_models.items()},
        "state_method": "Initialize each separate logged day from its robust first-3-minute 25S voltage/current anchor, then integrate measured battery current by elapsed UTC time. No overnight current is invented.",
        "charge_efficiency": 1.0,
        "charge_efficiency_note": "Unity coulombic charge efficiency is a transparent placeholder; available source has discharge curves but no charge test. Voltage validation is discharge-road-only.",
        "validation_grain": "1-minute medians; 5-second source rows are autocorrelated.",
        "soc_truth_available": False,
    }
    return cv, summary, fold_models


def reuse_battery_cv() -> tuple[pd.DataFrame, dict, dict[int, dict]]:
    """Reuse the just-completed voltage CV during report/replay iteration."""
    cv = pd.read_csv(OUT_BATTERY_FOLDS)
    profile = cv.groupby("capacity_ah", as_index=False).agg(
        lodo_mean_mae_v=("mae", "mean"), lodo_mean_rmse_v=("rmse", "mean"))
    profile = profile.sort_values("lodo_mean_mae_v")
    best = profile.iloc[0]
    q_selected = float(best.capacity_ah)
    q10 = 33.164
    q10row = profile[np.isclose(profile.capacity_ah, q10)]
    if len(q10row) and float(q10row.iloc[0].lodo_mean_mae_v) <= float(best.lodo_mean_mae_v) + 0.1:
        q_selected = q10
    profile["selected"] = np.isclose(profile.capacity_ah, q_selected)
    selected = cv[np.isclose(cv.capacity_ah, q_selected)].copy()
    fold_models = {int(r.held_out_day): {"capacity_ah": q_selected, "r_eff_ohm": float(r.r_eff_ohm)}
                   for r in selected.itertuples()}
    # Day 6 remains held out; using the mean clean-day fold coefficient gives its prior.
    fold_models[6] = {"capacity_ah": q_selected, "r_eff_ohm": float(selected.r_eff_ohm.mean())}
    summary = {
        "capacity_profile": profile.to_dict(orient="records"),
        "selected_capacity_ah": q_selected,
        "measured_10a_capacity_ah": 33.164,
        "measured_8a_capacity_ah": 34.597,
        "selected_full_clean_day_r_eff_ohm": float(selected.r_eff_ohm.mean()),
        "cv_mean_mae_v": float(selected.mae.mean()),
        "cv_mean_rmse_v": float(selected.rmse.mean()),
        "cv_mean_bias_v": float(selected.bias_measured_minus_model.mean()),
        "r_by_day_cv": {str(d): m["r_eff_ohm"] for d, m in fold_models.items()},
        "state_method": "Initialize each separate logged day from its robust first-3-minute 25S voltage/current anchor, then integrate measured battery current by elapsed UTC time. No overnight current is invented.",
        "charge_efficiency": 1.0,
        "charge_efficiency_note": "Unity coulombic charge efficiency is a transparent placeholder; available source has discharge curves but no charge test. Voltage validation is discharge-road-only.",
        "validation_grain": "1-minute medians; 5-second source rows are autocorrelated.",
        "soc_truth_available": False,
        "fit_source": "Reused reports/battery_voltage_model_cv.csv generated in this run; refit with the default command when source data or fitting method changes.",
    }
    return cv, summary, fold_models


def error_exclusion(d: pd.DataFrame) -> pd.Series:
    flags = d.event_flags.fillna("").astype(str)
    excluded = flags.str.contains(r"control_stop:|traffic_stop:|driver_change:|incident:|repair:|charge:|D6_ZP_ANOMALY|D6_RETIRE", regex=True)
    ts = pd.to_datetime(d.timestamp_local, errors="coerce", utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    for start, end in [("2025-08-28 10:31:00", "2025-08-28 10:37:00"),
                       ("2025-08-28 11:51:00", "2025-08-28 12:50:00")]:
        excluded |= d.day.eq(5) & ts.between(start, end)
    return excluded


def channel_quality(raw: pd.DataFrame) -> dict:
    d = raw.copy()
    for c in ["Solar[A]", "Battery[A]", "Main_Volt[V]", "Solar_Watt[W]", "Battery_Watt[W]", "Speed[kmh]"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    pv_calc = d["Solar[A]"] * d["Main_Volt[V]"]
    bat_calc = d["Battery[A]"] * d["Main_Volt[V]"]
    pve = (d["Solar_Watt[W]"] - pv_calc).abs()
    bate = (d["Battery_Watt[W]"] - bat_calc).abs()
    delta = d.groupby("day", sort=False).timestamp_utc.diff().dt.total_seconds()
    return {
        "rows": int(len(d)), "days": {str(k): int(v) for k, v in d.groupby("day").size().items()},
        "duplicate_day_timestamp_rows": int(d.duplicated(["day", "timestamp_utc"]).sum()),
        "sample_interval_seconds_counts": {str(k): int(v) for k, v in delta.dropna().value_counts().head(8).items()},
        "solar_power_equals_solar_current_times_main_voltage": {"median_abs_difference_w": float(pve.median()), "p99_abs_difference_w": float(pve.quantile(.99)), "max_abs_difference_w": float(pve.max())},
        "battery_power_equals_battery_current_times_main_voltage": {"median_abs_difference_w": float(bate.median()), "p99_abs_difference_w": float(bate.quantile(.99)), "max_abs_difference_w": float(bate.max())},
        "main_voltage_zero_rows": int(d["Main_Volt[V]"].eq(0).sum()),
        "battery_channel_interpretation": "Positive Battery_Watt/current is pack discharge; negative is charge. Solar_Watt + Battery_Watt is the integrated HV load proxy used in the team's energy reconciliation.",
        "calibration_note": "User reports that no calibration adjustment was made. This confirms no post-hoc correction, not traceable absolute sensor accuracy.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-battery-cv", action="store_true",
                        help="reuse the saved leave-one-day-out battery CV table; default recomputes the profile")
    args = parser.parse_args()
    # Source checks and current fitted-model inputs.
    base_pv = json.loads(PV_MAP.read_text(encoding="utf-8"))
    base_vehicle = json.loads(VEHICLE_MAP.read_text(encoding="utf-8"))
    hourly = pd.read_csv(PV_HOURLY)
    hourly["day"] = pd.to_numeric(hourly.day, errors="coerce").astype(int)
    pv_cv_table, pv_full, pv_summary = pv_cv(hourly)
    chosen_u0 = float(pv_summary["best_thermal_u0_w_m2c"])
    pv_cv_table.to_csv(OUT_PV_FOLDS, index=False, encoding="utf-8-sig")
    pv_models = pv_fold_models(hourly, pv_summary["selected_model"], chosen_u0)

    # Battery reference curves from user-supplied files.
    map_df, lab_r_curve, battery_source = make_lab_battery_maps()
    model_cols = ["day", "timestamp_local", "timestamp_utc", "event_phase", "event_flags", "Solar[A]", "Battery[A]",
                  "Main_Volt[V]", "Solar_Watt[W]", "Battery_Watt[W]", "Speed[kmh]", "distance_from_darwin_m",
                  "distance_anchor_confidence", "route_elevation_dem_m", "route_grade_smoothed_pct", "temperature_2m",
                  "surface_pressure", "wind_speed_10m", "wind_direction_10m", "shortwave_radiation_instant",
                  "shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation", "route_latitude",
                  "route_longitude"]
    raw = pd.read_csv(INPUT, usecols=lambda c: c in model_cols, low_memory=False)
    raw["timestamp_utc"] = pd.to_datetime(raw.timestamp_utc, utc=True)
    raw["timestamp_local"] = pd.to_datetime(raw.timestamp_local, utc=True).dt.tz_convert(TZ)
    raw["day"] = pd.to_numeric(raw.day, errors="coerce").astype(int)
    quality = channel_quality(raw)
    if args.reuse_battery_cv and OUT_BATTERY_FOLDS.exists():
        battery_cv, battery_summary, battery_fold_models = reuse_battery_cv()
    else:
        battery_cv, battery_summary, battery_fold_models = battery_lodo(raw)

    # Full row-wise replay, with day-blocked parameters for Days 1-5 and a full-clean-fit holdout for Day 6.
    road_cols = ["day", "timestamp_local", "timestamp_utc", "event_phase", "event_flags", "Solar[A]", "Battery[A]",
                 "Main_Volt[V]", "Solar_Watt[W]", "Battery_Watt[W]",
                 "Speed[kmh]", "distance_from_darwin_m", "distance_anchor_confidence", "route_elevation_dem_m",
                 "route_grade_smoothed_pct", "temperature_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m",
                 "shortwave_radiation_instant", "direct_normal_irradiance", "diffuse_radiation", "route_latitude", "route_longitude"]
    road = raw[road_cols].copy()
    road = features.make_features(road)
    road["measured_load_proxy_w"] = pd.to_numeric(road["Solar_Watt[W]"], errors="coerce") + pd.to_numeric(road["Battery_Watt[W]"], errors="coerce")
    seg = pd.read_csv(SEGMENTS)
    seg["day"] = pd.to_numeric(seg.day, errors="coerce").astype(int)
    paux = float(base_vehicle["fit_parameters"]["parameters"]["p_aux_w"])
    road_coefficients = {}
    for day in sorted(int(d) for d in road.day.unique()):
        if day <= 5:
            train = seg[seg.day.le(5) & seg.day.ne(day)].copy()
        else:
            train = seg[seg.day.le(5)].copy()
        train.attrs["p_aux_w"] = paux
        fold = vehicle.fit_mle(train)
        road_coefficients[day] = fold["parameters"]
        p = fold["parameters"]
        road.loc[road.day.eq(day), "predicted_vehicle_load_w"] = (
            paux + p["beta_roll"] * road.loc[road.day.eq(day), "roll_basis_w"]
            + p["beta_aero"] * road.loc[road.day.eq(day), "aero_basis_w"]
            + p["beta_grade"] * road.loc[road.day.eq(day), "grade_basis_w"]
        )
    local_hour = pd.to_datetime(road.timestamp_local, utc=True).dt.tz_convert(TZ).dt.floor("h").dt.strftime("%Y-%m-%dT%H:%M")
    road["hour_local"] = local_hour
    for day in sorted(int(d) for d in road.day.unique()):
        idx = road.day.eq(day)
        subset = road.loc[idx].copy()
        subset["pv_w"] = pd.to_numeric(subset["Solar_Watt[W]"], errors="coerce")
        # The weather inputs are hourly/15-minute model-grid values interpolated to 5 s.
        subset["temp_c"] = pd.to_numeric(raw.loc[idx, "temperature_2m"], errors="coerce").to_numpy()
        subset["ghi"] = pd.to_numeric(raw.loc[idx, "shortwave_radiation_instant"], errors="coerce").to_numpy()
        subset["direct_normal_irradiance"] = pd.to_numeric(raw.loc[idx, "direct_normal_irradiance"], errors="coerce").to_numpy()
        subset["diffuse_radiation"] = pd.to_numeric(raw.loc[idx, "diffuse_radiation"], errors="coerce").to_numpy()
        subset["wind_speed_10m"] = pd.to_numeric(raw.loc[idx, "wind_speed_10m"], errors="coerce").to_numpy()
        subset["route_latitude"] = pd.to_numeric(raw.loc[idx, "route_latitude"], errors="coerce").to_numpy()
        subset["route_longitude"] = pd.to_numeric(raw.loc[idx, "route_longitude"], errors="coerce").to_numpy()
        subset = pv.poa_tools.solar_geometry(subset)
        if pv_summary["selected_model"].endswith("faiman_temp"):
            pred = predict_pv_thermal(subset, pv_models[day])
        else:
            pred = pv.prediction(subset, pv_models[day]["parameters"], "faceted_body_poa")
        road.loc[idx, "predicted_pv_w"] = pred
    road["pv_error_measured_minus_model_w"] = pd.to_numeric(road["Solar_Watt[W]"], errors="coerce") - road.predicted_pv_w
    road["predicted_battery_power_required_w"] = road.predicted_vehicle_load_w - road.predicted_pv_w
    road["battery_power_error_measured_minus_model_w"] = pd.to_numeric(road["Battery_Watt[W]"], errors="coerce") - road.predicted_battery_power_required_w
    road["vehicle_load_error_measured_minus_model_w"] = road.measured_load_proxy_w - road.predicted_vehicle_load_w
    # The balance identity is also an internal check on the component decomposition.
    road["balance_decomposition_error_w"] = road.battery_power_error_measured_minus_model_w - (
        road.vehicle_load_error_measured_minus_model_w - road.pv_error_measured_minus_model_w)

    # Battery voltage/SOC replay by fold.
    battery_frames = []
    for day in sorted(int(d) for d in raw.day.unique()):
        bm = battery_fold_models[day]
        frame = battery_anchor_and_state(raw[raw.day.eq(day)], bm["capacity_ah"], bm["r_eff_ohm"])
        battery_frames.append(frame[["day", "timestamp_utc", "battery_soc_estimate_pct", "battery_voltage_predicted_v",
                                     "battery_voltage_reference_v", "battery_model_valid_voltage",
                                     "battery_charge_voltage_extrapolation", "battery_anchor_voltage_v", "battery_anchor_soc_pct"]])
    battery_trace = pd.concat(battery_frames, ignore_index=True)
    road = road.merge(battery_trace, on=["day", "timestamp_utc"], how="left", validate="one_to_one")
    road["replay_prediction_scope"] = np.where(road.day.eq(6), "day6 held out; trained on clean days 1-5", "day-blocked out-of-day replay")
    road["road_model_comparison_eligible"] = (road.event_phase.eq("road_day") & pd.to_numeric(road["Speed[kmh]"], errors="coerce").ge(45.0)
                                              & road.distance_anchor_confidence.isin(["high", "medium"])
                                              & ~error_exclusion(road))
    road["pv_model_comparison_eligible"] = (road.event_phase.eq("road_day") & pd.to_numeric(road["Speed[kmh]"], errors="coerce").ge(5.0)
                                            & pd.to_numeric(road["shortwave_radiation_instant"], errors="coerce").ge(50.0)
                                            & ~error_exclusion(road))
    keep = ["day", "timestamp_local", "timestamp_utc", "event_phase", "event_flags", "distance_from_darwin_m",
            "distance_anchor_confidence", "Speed[kmh]", "Solar_Watt[W]", "Battery_Watt[W]", "Main_Volt[V]",
            "Battery[A]", "temperature_2m", "shortwave_radiation_instant", "wind_speed_10m", "route_grade_smoothed_pct",
            "measured_load_proxy_w", "predicted_vehicle_load_w", "vehicle_load_error_measured_minus_model_w",
            "predicted_pv_w", "pv_error_measured_minus_model_w", "predicted_battery_power_required_w",
            "battery_power_error_measured_minus_model_w", "battery_soc_estimate_pct", "battery_voltage_predicted_v",
            "battery_voltage_reference_v", "battery_model_valid_voltage", "battery_charge_voltage_extrapolation",
            "battery_anchor_voltage_v", "battery_anchor_soc_pct", "road_model_comparison_eligible",
            "pv_model_comparison_eligible", "replay_prediction_scope", "balance_decomposition_error_w"]
    replay = road[keep].copy()
    replay.to_csv(OUT_REPLAY, index=False, encoding="utf-8-sig")

    # All-timepoint and race-day summaries; separate pointwise error from route-integrated energy error.
    daily = []
    segment_rows = []
    for day, g in road.groupby("day", sort=True):
        moving = g[g.road_model_comparison_eligible].copy()
        solar = g[g.pv_model_comparison_eligible].copy()
        batt = battery_validation_rows(g)
        row = {"day": int(day), "road_replay": score(moving.measured_load_proxy_w, moving.predicted_vehicle_load_w),
               "pv_5s_weather_forced": score(solar["Solar_Watt[W]"], solar.predicted_pv_w),
               "battery_voltage_1min_discharge": score(batt.measured_voltage_v, batt.predicted_voltage_v),
               "rows_all": int(len(g)), "road_rows_eligible": int(len(moving)), "pv_rows_eligible": int(len(solar)),
               "battery_1min_rows_eligible": int(len(batt)),
               "mean_measured_load_w": float(moving.measured_load_proxy_w.mean()) if len(moving) else None,
               "mean_measured_pv_w": float(solar["Solar_Watt[W]"].mean()) if len(solar) else None,
               "day6_held_out": bool(int(day) == 6)}
        daily.append(row)
        if len(moving):
            moving["segment_10km"] = np.floor(pd.to_numeric(moving.distance_from_darwin_m, errors="coerce") / 10000.0).astype("Int64")
            moving["dt_s"] = moving.timestamp_utc.diff().dt.total_seconds()
            # The first row in a segment has no previous interval here; each logged source sample is otherwise 5 s.
            moving["dt_s"] = moving.dt_s.where(moving.dt_s.between(2.0, 15.0), 0.0)
            for sid, s in moving.groupby("segment_10km", dropna=True):
                if len(s) < 5:
                    continue
                dt = s.dt_s.to_numpy(float)
                if dt.sum() <= 0:
                    continue
                measured_wh = float(np.sum(s.measured_load_proxy_w.to_numpy(float) * dt) / 3600.0)
                predicted_wh = float(np.sum(s.predicted_vehicle_load_w.to_numpy(float) * dt) / 3600.0)
                segment_rows.append({"day": int(day), "segment_10km": int(sid), "rows": int(len(s)),
                                     "distance_start_m": float(s.distance_from_darwin_m.min()),
                                     "distance_end_m": float(s.distance_from_darwin_m.max()),
                                     "measured_energy_wh": measured_wh, "predicted_energy_wh": predicted_wh,
                                     "error_measured_minus_model_wh": measured_wh - predicted_wh,
                                     "measured_mean_power_w": measured_wh * 3600 / dt.sum(),
                                     "predicted_mean_power_w": predicted_wh * 3600 / dt.sum(),
                                     "eligible_time_s": float(dt.sum())})
    daily_flat = []
    for r in daily:
        daily_flat.append({"day": r["day"], "rows_all": r["rows_all"], "road_rows_eligible": r["road_rows_eligible"],
                           "road_mae_w": r["road_replay"]["mae"], "road_rmse_w": r["road_replay"]["rmse"],
                           "road_bias_w": r["road_replay"]["bias_measured_minus_model"],
                           "pv_rows_eligible": r["pv_rows_eligible"], "pv_5s_mae_w": r["pv_5s_weather_forced"]["mae"],
                           "pv_5s_rmse_w": r["pv_5s_weather_forced"]["rmse"], "pv_5s_bias_w": r["pv_5s_weather_forced"]["bias_measured_minus_model"],
                           "battery_1min_rows_eligible": r["battery_1min_rows_eligible"],
                           "battery_voltage_mae_v": r["battery_voltage_1min_discharge"]["mae"],
                           "battery_voltage_rmse_v": r["battery_voltage_1min_discharge"]["rmse"],
                           "battery_voltage_bias_v": r["battery_voltage_1min_discharge"]["bias_measured_minus_model"],
                           "day6_held_out": r["day6_held_out"]})
    pd.DataFrame(daily_flat).to_csv(OUT_DAILY, index=False, encoding="utf-8-sig")
    segment_df = pd.DataFrame(segment_rows)
    segment_df.to_csv(OUT_SEGMENT_REPLAY, index=False, encoding="utf-8-sig")

    road_cv = pd.read_csv(ROOT / "reports/vehicle_power_cv_by_day.csv")
    inertial_path = ROOT / "models/vehicle_inertial_candidate_cv.json"
    inertial_summary = None
    if inertial_path.exists():
        inertial_result = json.loads(inertial_path.read_text(encoding="utf-8"))
        one_min = inertial_result["summary_clean_lodo"]["one_minute"]
        five_sec = inertial_result["summary_clean_lodo"]["five_second"]
        inertia_betas = [json.loads(r["coefficients_json"]).get("accel_positive_w")
                         for r in inertial_result["folds_minute"]
                         if r["model"] == "signed_inertia" and int(r["held_out_day"]) <= 5]
        inertial_summary = {
            "minute_baseline_mae_w": one_min["steady"]["mean_day_mae_w"],
            "minute_signed_inertia_mae_w": one_min["signed_inertia"]["mean_day_mae_w"],
            "five_second_baseline_mae_w": five_sec["steady"]["mean_day_mae_w"],
            "five_second_signed_inertia_mae_w": five_sec["signed_inertia"]["mean_day_mae_w"],
            "positive_inertia_coefficient_fold_values": inertia_betas,
            "positive_inertia_coefficient_range": [float(min(inertia_betas)), float(max(inertia_betas))],
            "decision": "tested but not adopted: the small minute-MAE gain does not resolve large residuals, and fitted positive-inertia coefficients fall below the electrical energy conversion expected from the 314 kg minimum-mass prior; coefficients remain confounded with noisy differentiated speed and road terms."
        }
    battery_r_candidate_path = ROOT / "models/battery_soc_resistance_candidate_cv.json"
    battery_r_candidate_summary = None
    if battery_r_candidate_path.exists():
        battery_r_candidate = json.loads(battery_r_candidate_path.read_text(encoding="utf-8"))
        csum = battery_r_candidate["candidate_summaries_clean_lodo"]
        rmap_scales = [json.loads(r["parameters_json"]).get("scale")
                       for r in battery_r_candidate["folds"]
                       if r["model"] == "lab_R_soc_scaled" and int(r["held_out_day"]) <= 5]
        d6_constant = next(r for r in battery_r_candidate["folds"]
                           if r["model"] == "constant_R" and int(r["held_out_day"]) == 6)
        d6_rmap = next(r for r in battery_r_candidate["folds"]
                       if r["model"] == "lab_R_soc_scaled" and int(r["held_out_day"]) == 6)
        battery_r_candidate_summary = {
            "clean_lodo_constant_r_mae_v": csum["constant_R"]["mean_day_mae_v"],
            "clean_lodo_soc_r_map_mae_v": csum["lab_R_soc_scaled"]["mean_day_mae_v"],
            "clean_lodo_soc_r_map_plus_offset_mae_v": csum["lab_R_soc_scaled_plus_train_offset"]["mean_day_mae_v"],
            "clean_fold_r_scale_values": rmap_scales,
            "clean_fold_r_scale_range": [float(min(rmap_scales)), float(max(rmap_scales))],
            "day6_constant_r_mae_v": d6_constant["mae_v"],
            "day6_soc_r_map_mae_v": d6_rmap["mae_v"],
            "decision": "Do not adopt the SOC-dependent map for the full-race model yet: it improves clean Days 1-5 but fails the anomalous Day 6 holdout. Keep it as the next candidate; the fitted voltage offset is also not adopted because it is not stable and is not a sensor calibration."
        }
    if battery_r_candidate_summary:
        battery_r_candidate_report_text = (
            f"An SOC-dependent candidate was then tested using that corrected lab curve with one fitted scale per training fold. "
            f"Clean-day (Days 1–5) MAE falls from **{battery_r_candidate_summary['clean_lodo_constant_r_mae_v']:.3f} V** "
            f"for constant R to **{battery_r_candidate_summary['clean_lodo_soc_r_map_mae_v']:.3f} V**; fold scales are "
            f"**{battery_r_candidate_summary['clean_fold_r_scale_range'][0]:.3f}–{battery_r_candidate_summary['clean_fold_r_scale_range'][1]:.3f}**. "
            f"However, on the separate Day 6 holdout its MAE is **{battery_r_candidate_summary['day6_soc_r_map_mae_v']:.3f} V**, "
            f"versus **{battery_r_candidate_summary['day6_constant_r_mae_v']:.3f} V** for constant R. "
            "The candidate is therefore **not used** in the full-race replay yet. Adding a training-derived voltage offset "
            "improves clean-day bias but worsens Day 6 and cannot be called sensor calibration."
        )
    else:
        battery_r_candidate_report_text = "An SOC-dependent resistance candidate has not yet been evaluated."
    selected_q = battery_summary["selected_capacity_ah"]
    battery_cv_selected = battery_cv[battery_cv.capacity_ah.eq(selected_q)]
    selected_pv_cv = pv_cv_table[pv_cv_table.model.eq(pv_summary["selected_model"])].copy()
    pv_cv_means = selected_pv_cv.groupby("u0", dropna=False).mae.mean().to_dict()
    best_pv_rows = pv_cv_table[pv_cv_table.u0.eq("baseline_no_temp")]
    if pv_summary["selected_model"].endswith("faiman_temp"):
        best_pv_rows = pv_cv_table[(pv_cv_table.model.eq(pv_summary["selected_model"])) & (pv_cv_table.u0.eq(chosen_u0))]
    model_result = {
        "scope": "BWSC2025 complete 5-second logs, 43,963 records; output is a retrospective kinematic replay with observed speed as an exogenous input, not closed-loop MPC.",
        "input_quality": quality,
        "pv": {"base_model": base_pv.get("selected_model"), "iteration_selected_model": pv_summary["selected_model"],
               "thermal_candidates": pv_summary["means"], "selection": pv_summary,
               "full_training_parameters": pv_full, "cv_rows": pv_cv_table.to_dict(orient="records"),
               "limitations": ["Only 23 moving hourly bins across Days 1-5 are used for fitting.",
                               "Open-Meteo weather is a spatial/time model-grid interpolation, not vehicle-mounted irradiance.",
                               "No module temperature sensor; Faiman-style cell temperature is a generic roof-mounted prior.",
                               "Panel normal/curvature is not surveyed at facet resolution; combined PV+MPPT gain is not intrinsic cell efficiency."]},
        "battery": {**battery_source, **battery_summary,
                    "soc_resistance_candidate_validation": battery_r_candidate_summary,
                    "cv_rows": battery_cv_selected.to_dict(orient="records"),
                    "lab_resistance_curve": lab_r_curve.to_dict(orient="records"),
                    "limitations": ["Voltage-SOC table is an empirical reference, not a certified open-circuit curve.",
                                    "Pack resistance is an effective on-car voltage-current coefficient; it includes wiring, polarization, temperature and aging.",
                                    "No true in-car SOC measurement, pack temperature measurement, pulse resistance test or charge curve.",
                                    "Day 3 startup voltage contains a 79.7 V sample followed by a rapid recovery to about 89.6 V; zero/malformed voltage samples are excluded from voltage scoring."]},
        "vehicle": {"cv_by_day_existing_10km": road_cv.to_dict(orient="records"),
                    "row_replay_coefficients": {str(k): v for k, v in road_coefficients.items()},
                    "short_horizon_inertial_candidate": inertial_summary,
                    "row_replay_scope": "Observed Speed is forced; no acceleration term is predicted. 10 km segment coefficients from other days are applied to local road-load terms for each source timestamp.",
                    "limitations": ["The 10 km fit does not identify rolling resistance, CdA, mass, grade bias and motor efficiency independently.",
                                    "Per-sample residual includes unmodeled acceleration, grade map error, wind/irradiance interpolation and telemetry quantization.",
                                    "Day 6 is held out because of the logged ZP/retirement anomaly."]},
        "by_day_replay": daily,
        "ten_km_oof_segments": segment_rows,
        "metric_definitions": {"load_proxy_w": "Solar_Watt + Battery_Watt; signed battery power is positive discharge and negative charge.",
                               "pv_prediction_scope": "Weather-forced, cross-day fold model; parked/tilted posture is outside fit eligibility.",
                               "battery_voltage_validation": "One-minute medians on moving road samples with positive battery current; SoC itself has no ground truth."},
        "calibration_statement": "No user calibration adjustment was made. Model-fit gains and R values correct model structure only and are not sensor recalibration.",
    }
    OUT_FIT.write_text(json.dumps(model_result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Diagnostic figure: time-series at the validation grain and daily model error distributions.
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)
    for day in sorted(road.day.unique()):
        s = road[(road.day.eq(day)) & road.road_model_comparison_eligible]
        if len(s):
            axes[0].plot(pd.to_datetime(s.timestamp_local, utc=True).dt.tz_convert(TZ),
                         s.measured_load_proxy_w, color=f"C{int(day)-1}", alpha=.30, lw=.7)
            axes[0].plot(pd.to_datetime(s.timestamp_local, utc=True).dt.tz_convert(TZ),
                         s.predicted_vehicle_load_w, color=f"C{int(day)-1}", alpha=.85, lw=.8, label=f"D{int(day)} model")
    axes[0].set_ylabel("HV load proxy [W]")
    axes[0].set_title("Observed-speed full-race replay; day-blocked fit")
    axes[0].legend(ncol=6, fontsize=8)
    axes[1].boxplot([road.loc[road.day.eq(d) & road.road_model_comparison_eligible,
                               "vehicle_load_error_measured_minus_model_w"].dropna().to_numpy()
                     for d in sorted(road.day.unique())], tick_labels=[f"D{d}" for d in sorted(road.day.unique())], showfliers=False)
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set_ylabel("Measured − model [W]")
    axes[1].set_title("Per-sample residual; 5 s autocorrelation and omitted transients remain")
    for day in sorted(raw.day.unique()):
        x = raw[raw.day.eq(day)].merge(battery_trace[battery_trace.day.eq(day)], on=["day", "timestamp_utc"], how="left")
        valid = x.battery_model_valid_voltage.fillna(False)
        axes[2].plot(pd.to_datetime(x.loc[valid, "timestamp_utc"], utc=True).dt.tz_convert(TZ),
                     pd.to_numeric(x.loc[valid, "Main_Volt[V]"], errors="coerce"), color=f"C{int(day)-1}", alpha=.55, lw=.7)
        axes[2].plot(pd.to_datetime(x.loc[valid, "timestamp_utc"], utc=True).dt.tz_convert(TZ),
                     x.loc[valid, "battery_voltage_predicted_v"], color=f"C{int(day)-1}", alpha=.9, lw=.7, ls="--")
    axes[2].set_ylabel("Pack voltage [V]")
    axes[2].set_title("Measured voltage and coulomb-count + fitted R replay (solid measured, dashed predicted)")
    fig.savefig(OUT_FIG, dpi=160)
    plt.close(fig)

    # Decision gate based on the size of a nominal 5 km/h road-load change.
    mean_speed = float(road.loc[road.road_model_comparison_eligible, "Speed[kmh]"].median())
    rho = float(road.loc[road.road_model_comparison_eligible, "air_density_kgm3"].median())
    v1, v2 = max(mean_speed - 2.5, 0.0) / 3.6, (mean_speed + 2.5) / 3.6
    aero_delta = 0.5 * rho * features.CD_A_NOMINAL_M2 * (v2**3 - v1**3)
    roll_delta = features.MASS_NOMINAL_KG * 9.80665 * features.CRR_NOMINAL * (v2 - v1)
    action_delta = float(aero_delta + roll_delta)
    road_clean = [r for r in daily if r["day"] <= 5]
    oof_10km = segment_df[segment_df.day.le(5)] if len(segment_df) else segment_df
    segment_mae_wh = float(np.mean(np.abs(oof_10km.error_measured_minus_model_wh))) if len(oof_10km) else None
    mean_segment_wh = float(oof_10km.measured_energy_wh.mean()) if len(oof_10km) else None
    gate = {
        "median_speed_kmh": mean_speed,
        "nominal_power_change_for_adjacent_5kmh_speeds_w": action_delta,
        "road_mpc_mae_gate_w": 0.25 * action_delta,
        "ten_km_oof_mae_wh": segment_mae_wh,
        "ten_km_mean_measured_energy_wh": mean_segment_wh,
        "ten_km_relative_mae_pct": 100.0 * segment_mae_wh / mean_segment_wh if segment_mae_wh and mean_segment_wh else None,
        "criterion": "Internal engineering gate: road-load OOF error must be below 25% of the predicted load change between adjacent ±2.5 km/h speed choices, so model error does not dominate the control choice. It is a decision-resolution criterion, not a BWSC regulation or published standard.",
        "road_lodo_mean_mae_w": float(road_cv.loc[road_cv.held_out_day.le(5), "mae_w"].mean()),
        "passed": bool(float(road_cv.loc[road_cv.held_out_day.le(5), "mae_w"].mean()) <= 0.25 * action_delta),
    }
    model_result["mpc_readiness_gate"] = gate
    OUT_FIT.write_text(json.dumps(model_result, indent=2, ensure_ascii=False), encoding="utf-8")

    if inertial_summary:
        inertia_gain_pct = 100.0 * (inertial_summary["minute_baseline_mae_w"]
                                    - inertial_summary["minute_signed_inertia_mae_w"])
        inertia_gain_pct /= inertial_summary["minute_baseline_mae_w"]
        inertial_report_text = (
            f"- **Acceleration candidate, tested and rejected**: adding positive/negative inertial-power terms "
            f"reduced 1-minute day-LODO MAE from {inertial_summary['minute_baseline_mae_w']:.1f} W to "
            f"{inertial_summary['minute_signed_inertia_mae_w']:.1f} W ({inertia_gain_pct:.1f}%), but MAE remained high. "
            f"The positive-inertia coefficient varied from {inertial_summary['positive_inertia_coefficient_range'][0]:.2f} "
            f"to {inertial_summary['positive_inertia_coefficient_range'][1]:.2f}, below the expected electrical conversion "
            f"factor for the 314 kg minimum-mass basis. At 5-second grain the descriptive day-LODO MAE changed from "
            f"{inertial_summary['five_second_baseline_mae_w']:.1f} W to "
            f"{inertial_summary['five_second_signed_inertia_mae_w']:.1f} W, so there is no persuasive pointwise gain. "
            "Keep the simpler 10 km model as the route-energy reference; do not use this acceleration candidate for control."
        )
    else:
        inertial_report_text = "- **Acceleration candidate**: not run; the full-race residual still includes acceleration omitted by the 10 km fit."

    report = f"""# BWSC2025 component-model iteration and full-race replay

## Scope and data checks

This replay covers **{quality['rows']:,} source records over Days 1–6**. Logged vehicle speed is supplied as an exogenous input at every 5-second timestamp. The result is a retrospective kinematic replay, not a closed-loop race simulation: it does not choose throttle, predict speed, or claim 5-second power accuracy.

- Duplicate `(day, timestamp_utc)` rows: {quality['duplicate_day_timestamp_rows']}.
- Dominant logged interval: {max(quality['sample_interval_seconds_counts'], key=quality['sample_interval_seconds_counts'].get)} seconds.
- `Solar_Watt` vs `Solar[A] × Main_Volt`: median absolute difference {quality['solar_power_equals_solar_current_times_main_voltage']['median_abs_difference_w']:.3f} W; 99th percentile {quality['solar_power_equals_solar_current_times_main_voltage']['p99_abs_difference_w']:.3f} W.
- `Battery_Watt` vs `Battery[A] × Main_Volt`: median absolute difference {quality['battery_power_equals_battery_current_times_main_voltage']['median_abs_difference_w']:.3f} W; 99th percentile {quality['battery_power_equals_battery_current_times_main_voltage']['p99_abs_difference_w']:.3f} W.
- Zero main-voltage samples: {quality['main_voltage_zero_rows']} (excluded from battery voltage scoring).
- The user reports no calibration adjustment. That means no post-hoc correction was applied; it does not establish traceable absolute sensor accuracy.

The component balance uses `measured load = Solar_Watt + Battery_Watt`, with positive battery power meaning discharge. PV and vehicle-load predictions are produced separately; the battery power required by the model is their difference.

## Component order and identifiability

1. **PV array + MPPT**: isolated first because `Solar_Watt` is a dedicated aggregate HV output and weather/sun geometry can be used as external drivers. The fit is an effective system map; array, incidence angle, temperature, shading, mismatch and MPPT are not separately observed.
2. **Battery**: voltage and current are logged directly, and the supplied 8 A/10 A discharge curves plus 25-series SoC-voltage reference constrain a pack-level model. Capacity/voltage and effective resistance can be assessed; true SoC, battery temperature, charge efficiency and cell-level imbalance cannot be verified from these channels.
3. **Motor/controller and road load**: keep as effective road-load coefficients for now. The race log has no torque, motor phase current, throttle/PWM command or wheel force. Motor efficiency is therefore confounded with tire, aerodynamics, route grade, wind, regen and telemetry.

The supplied vehicle setup is used as already specified by the user: 234 kg dry car, 80 kg regulatory occupant/ballast floor proxy, 0.08876 m² theoretical CdA, 0.0035 Crr, 558.4 mm tyre, three wheels/direct drive, bonded fixed-in-motion array, and the supplied 6 m²/layout basis. No vehicle-data confirmation is requested again.

## PV + MPPT map: added physical thermal candidate

The training set has **{pv_summary['eligible_bins']} moving hourly bins across Days 1–5**. PV power is matched at that weather resolution for fitting; the row replay then evaluates each logged time using the interpolated weather fields.

The existing faceted POA map has day-blocked mean MAE **{pv_summary['baseline_lodo_mean_mae_w']:.1f} W**. A generic Faiman-style cell-temperature estimate (`Tcell = Tamb + POA/(U0 + 6.84 × wind)`) was added with U0 profiled over 20, 25 and 30 W/m²°C. The best thermal candidate (U0={chosen_u0:.0f}) has day-blocked mean MAE **{pv_summary['best_thermal_lodo_mean_mae_w']:.1f} W**. The selection rule required at least a 2% improvement, stable gamma sign across held-out days, and no fold at its constraint bound. **Selected map: `{pv_summary['selected_model']}`.**

The fitted temperature coefficient is not a certified SunPower cell coefficient. The provided cell sheet gives STC electrical values but no installed module temperature coefficient. No module temperature sensor exists, and the roof-bonded geometry differs from generic open-rack thermal assumptions. The thermal candidate and fold coefficients are saved in `pv_temperature_model_cv.csv`; only a stable day-blocked improvement is retained.

## Battery map and voltage fit

The supplied 8 A and 10 A curves report **{battery_source['lab_capacity_8a_ah']:.3f} Ah** and **{battery_source['lab_capacity_10a_ah']:.3f} Ah**. Integrating voltage over discharged Ah and scaling the 2S6P test to 25S6P gives **{battery_source['lab_8a_integrated_energy_25s6p_wh']:.1f} Wh** at 8 A and **{battery_source['lab_10a_integrated_energy_25s6p_wh']:.1f} Wh** at 10 A. The 10 A result is only **{battery_source['lab_10a_energy_difference_pct']:+.2f}%** from the user-stated 3,011 Wh; its 33.164 Ah capacity is therefore the selected design/test prior. The voltage-only fit preferred 34 Ah by 0.045 V MAE, less than the logged Main_Volt 0.1 V step, so that small gain does not outweigh the direct discharge-energy evidence. The independent 5%-step 25-series voltage/SoC curve is exported without smoothing.

The lab differential-resistance extraction was also corrected in this iteration: the source SoC axis descends from 100% to 0%, but `np.interp` requires ascending input. After sorting, **{battery_source['positive_r_points']} of 21** 5%-step points yield a positive 8 A-to-10 A pack-resistance proxy (valid 5–95% SoC); the 0% and 100% endpoints are excluded because the two measured curves reverse order there. This is still a rate-differential proxy, not pulse-derived ohmic resistance or an in-car R(SOC,T) truth map.

{battery_r_candidate_report_text}

With capacity anchored to the direct test, day-blocked validation uses **{battery_summary['selected_capacity_ah']:.3f} Ah** and estimates a full clean-day effective pack resistance of **{battery_summary['selected_full_clean_day_r_eff_ohm']:.3f} Ω**. Leave-one-day-out moving-discharge voltage errors: MAE **{battery_summary['cv_mean_mae_v']:.2f} V**, RMSE **{battery_summary['cv_mean_rmse_v']:.2f} V**, mean bias **{battery_summary['cv_mean_bias_v']:+.2f} V** (1-minute medians). The 34 Ah unconstrained profile point remains in `component_replay_fit.json` to show the voltage-only fit's small preference; the capacity is no longer allowed to drift away from the 3,011 Wh evidence by less than one bus-voltage count. Fold errors are in `battery_voltage_model_cv.csv`.

Each day is initialized from its first robust 3-minute voltage/current anchor, then logged current is integrated over actual UTC elapsed seconds. No missing overnight charging is fabricated. This voltage fit does **not** validate absolute SoC: there is no in-car SoC ground truth. Charge efficiency is set to 1.0 as an explicit placeholder; voltage scoring excludes charge samples because the supplied lab work has no charge curve. A Day 3 start contains a 79.7 V one-sample transient followed by about 89.6 V; the robust anchor suppresses that startup point.

## Whole-race replay and residuals

`full_race_component_replay_5s.csv` contains a prediction and residual for every original record. Days 1–5 use road coefficients fit without that day; Day 6 is held out from all clean-day fitting. PV training is likewise day-blocked. The vehicle model remains the fitted 10 km MLE with logged speed forced; local residuals include omitted acceleration and route/weather error.

The 10 km cross-day replay contains **{len(oof_10km)} segments** on Days 1–5 with mean absolute energy error **{segment_mae_wh:.1f} Wh** ({100*segment_mae_wh/mean_segment_wh:.1f}% of mean measured segment energy) when available. Use `full_race_component_replay_10km.csv` for segment-level review; row-level residuals are in the full 5-second file.

### Daily metrics

| Day | Road-load OOF MAE (W) | PV 5 s weather-forced MAE (W) | Battery voltage 1 min MAE (V) | Notes |
|---:|---:|---:|---:|---|
"""
    for r in daily:
        rd, pp, bb = r["road_replay"], r["pv_5s_weather_forced"], r["battery_voltage_1min_discharge"]
        report += f"| {r['day']} | {rd['mae'] if rd['mae'] is not None else float('nan'):.1f} | {pp['mae'] if pp['mae'] is not None else float('nan'):.1f} | {bb['mae'] if bb['mae'] is not None else float('nan'):.2f} | {'held out; race ended early/anomaly' if r['day']==6 else 'day-blocked'} |\n"
    report += f"""
## MPC-use gate

At the median eligible cruise speed (**{gate['median_speed_kmh']:.1f} km/h**), the nominal physical road-load change between adjacent ±2.5 km/h choices is about **{action_delta:.1f} W** using the supplied Crr/CdA, 314 kg minimum-mass proxy and observed median air density. The internal decision-resolution gate is therefore an out-of-day model MAE below **{gate['road_mpc_mae_gate_w']:.1f} W** (25% of that action difference), so error does not dominate a speed choice. This is an engineering acceptance rule for this MPC decision, not a BWSC requirement or external standard.

Current 10 km leave-one-day-out road-load MAE on Days 1–5 is **{float(road_cv.loc[road_cv.held_out_day.le(5), 'mae_w'].mean()):.1f} W**. On the distinct 5-second row grain, Day 6 held out from clean-day fitting has MAE **{daily[5]['road_replay']['mae']:.1f} W** and bias **{daily[5]['road_replay']['bias_measured_minus_model']:+.1f} W**; these two grains are not directly comparable. **The model does not pass the decision-resolution gate yet.** The full-race playback supports diagnosis and route-level energy budgeting with caveats, but it is not an unguarded MPC speed command model.

## What the residuals imply

- **PV residuals**: hourly training has only 23 bins; sample-level weather values are spatial/time-interpolated weather-model estimates. A remaining mismatch cannot be attributed uniquely to cell temperature, incidence angle, passing cloud, partial shading, array mismatch or MPPT loss. Parked/tilted periods are outside the moving map.
- **Battery residuals**: the 1-minute voltage error tests the empirical SoC curve plus pack R; it cannot determine cell-level R, temperature response, capacity fade, charge efficiency or true SoC. No logging calibration means sensor scale remains unverified.
- **Vehicle residuals**: the 10 km fit's effective coefficients combine the stated physical priors with actual motor/controller operation, tire state, interpolated wind, route elevation, regeneration and logger error. Row-wise error additionally reflects acceleration omitted by the segment fit. Day 6's positive residual/bias is retained as a held-out failure signal, not absorbed into a refit.
{inertial_report_text}
- **Motor stage**: the manual separates current/PWM command mode from the Pow/Eco switch. Bench curves are labelled PWM_LO/PWM_HI/ECO_LO/ECO_HI and are single operating sweeps; they do not identify an in-car PWM+ECO efficiency surface. Use them as priors until a matching bench setting/command map or motor-side data is available.

## Current disposition

PV/MPPT is the first fitted component, but a true temperature-efficiency map is **not identified** by these logs. Battery voltage and capacity were advanced with lab priors and day-blocked validation; the resulting SoC remains an estimate. The combined model is **not ready for autonomous MPC speed selection** under the stated decision-resolution criterion. The errors are too large relative to the energy difference between nearby speed choices, and absolute sensor calibration/SoC is unverified.

To close the remaining high-impact gaps: collect a short synchronized array-plane irradiance and backsheet/module-temperature series; battery pack temperature and a rest/pulse sequence at known SoC; and motor/controller input/output or torque-speed data at the actual PWM-command + ECO-switch setting. These are the smallest new measurements that separate the current confounded terms.

## Source basis

- User-supplied SunPower E66Me3 cell sheet and 8 A/10 A battery discharge materials (source copies under `data/input/design/`).
- NREL PVWatts V5 uses plane-of-array irradiance, wind and ambient temperature in a transient cell-temperature model; its thermal assumptions are generic and are not a vehicle-panel validation: [PVWatts V5 Manual](https://pvwatts.nrel.gov/downloads/pvwattsv5.pdf).
- Faiman's module-temperature model is the basis for the thermal sensitivity candidate: [Faiman (2008), DOI 10.1002/pip.813](https://doi.org/10.1002/pip.813).
- Battery equivalent-circuit parameter identification and SOC methods rely on voltage/current plus appropriate pulse/rest characterization; this race log has voltage/current but no in-car true SoC or temperature: [Review of battery SOC estimation based on equivalent circuit models (2025)](https://doi.org/10.1016/j.est.2025.115346), [parameter-estimation methods benchmark (2023)](https://doi.org/10.1016/j.est.2023.108197).
- Road-load component separation normally needs controlled coastdown/road-load or independent tire/aero measurements; one blended race electrical channel cannot uniquely identify each physical term: [SAE J2263](https://doi.org/10.4271/J2263_200812).

Artifacts: `component_replay_fit.json`, `component_replay_by_day.csv`, `component_replay_diagnostics.png`, `full_race_component_replay_5s.csv`, `full_race_component_replay_10km.csv`, `battery_voltage_soc_map.csv`, and `battery_effective_resistance_map.csv`.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"quality": quality, "pv_selection": pv_summary, "battery": battery_summary,
                      "daily": daily, "mpc_gate": gate, "outputs": [str(OUT_REPLAY), str(OUT_SEGMENT_REPLAY),
                                                                         str(OUT_FIT), str(OUT_REPORT), str(OUT_FIG)]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
