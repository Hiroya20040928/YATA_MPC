#!/usr/bin/env python3
"""Fit and diagnose a first YATA PV power model against BWSC2025 ZP logs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr, t as student_t

ROOT = Path(__file__).resolve().parents[1]
TZ = "Australia/Darwin"
GHI_CHOICES = ["shortwave_radiation_instant", "shortwave_radiation"]
ROUTE_PATH = ROOT / "data/input/route/route_100m_dem.csv"


def fit_base(x: pd.DataFrame) -> dict:
    ghi = x.ghi.to_numpy(dtype=float)
    y = x.pv_w.to_numpy(dtype=float)
    def nll(z):
        pred = z[0] + z[1] * ghi
        return float(-np.sum(student_t.logpdf(y - pred, df=z[3], loc=0, scale=z[2])))
    fit = minimize(nll, x0=[0.0, 0.45, 40.0, 5.0], method="L-BFGS-B", bounds=[(-100, 150), (0, 5), (1, 500), (2.1, 30)])
    return {"intercept_w": float(fit.x[0]), "effective_gain_m2": float(fit.x[1]), "temperature_gamma_per_c": None,
            "likelihood_scale_w": float(fit.x[2]), "student_t_degrees_of_freedom": float(fit.x[3]),
            "log_likelihood": float(-fit.fun), "parameters": fit.x[:2], "model_kind": "base", "n": len(x)}


def fit_origin(x: pd.DataFrame) -> dict:
    """Fit a zero-offset conversion model, which is physically zero at zero irradiance."""
    ghi = x.ghi.to_numpy(dtype=float)
    y = x.pv_w.to_numpy(dtype=float)
    def nll(z):
        pred = z[0] * ghi
        return float(-np.sum(student_t.logpdf(y - pred, df=z[2], loc=0, scale=z[1])))
    fit = minimize(nll, x0=[0.9, 80.0, 5.0], method="L-BFGS-B", bounds=[(0, 5), (1, 500), (2.1, 30)])
    return {"intercept_w": 0.0, "effective_gain_m2": float(fit.x[0]), "temperature_gamma_per_c": None,
            "likelihood_scale_w": float(fit.x[1]), "student_t_degrees_of_freedom": float(fit.x[2]),
            "log_likelihood": float(-fit.fun), "parameters": [float(fit.x[0])], "model_kind": "origin", "n": len(x)}


def fit_temp(x: pd.DataFrame) -> dict:
    ghi = x.ghi.to_numpy(dtype=float)
    temp = x.temp_c.to_numpy(dtype=float)
    y = x.pv_w.to_numpy(dtype=float)
    def nll(z):
        pred = z[0] + z[1] * ghi * (1 + z[2] * (temp - 25))
        return float(-np.sum(student_t.logpdf(y - pred, df=z[4], loc=0, scale=z[3])))
    fit = minimize(nll, x0=[0.0, 0.45, -0.003, 40.0, 5.0], method="L-BFGS-B",
                   bounds=[(-100, 150), (0, 5), (-0.02, 0.015), (1, 500), (2.1, 30)])
    return {"intercept_w": float(fit.x[0]), "effective_gain_m2": float(fit.x[1]), "temperature_gamma_per_c": float(fit.x[2]),
            "likelihood_scale_w": float(fit.x[3]), "student_t_degrees_of_freedom": float(fit.x[4]),
            "log_likelihood": float(-fit.fun), "parameters": fit.x[:3], "model_kind": "temperature", "n": len(x)}


def fit_beam(x: pd.DataFrame) -> dict:
    ghi = x.ghi.to_numpy(dtype=float)
    dni = x.direct_normal_irradiance.to_numpy(dtype=float)
    y = x.pv_w.to_numpy(dtype=float)
    def nll(z):
        pred = z[0] + z[1] * ghi + z[2] * dni
        return float(-np.sum(student_t.logpdf(y - pred, df=z[4], loc=0, scale=z[3])))
    fit = minimize(nll, x0=[0.0, 0.4, 0.05, 40.0, 5.0], method="L-BFGS-B",
                   bounds=[(-100, 150), (0, 5), (0, 5), (1, 500), (2.1, 30)])
    return {"intercept_w": float(fit.x[0]), "effective_gain_m2": float(fit.x[1]), "direct_beam_gain_m2": float(fit.x[2]),
            "temperature_gamma_per_c": None, "likelihood_scale_w": float(fit.x[3]), "student_t_degrees_of_freedom": float(fit.x[4]),
            "log_likelihood": float(-fit.fun), "parameters": fit.x[:3], "model_kind": "beam", "n": len(x)}


def solar_geometry(x: pd.DataFrame) -> pd.DataFrame:
    """Attach approximate solar position and route bearing to hourly bins."""
    route = pd.read_csv(ROUTE_PATH, usecols=["distance_m", "lat", "lon"])
    route_s = route.distance_m.to_numpy(dtype=float)
    distance = x.distance_from_darwin_m.to_numpy(dtype=float)
    before = np.interp(np.maximum(distance - 250, route_s[0]), route_s, route.lat.to_numpy(dtype=float))
    before_lon = np.interp(np.maximum(distance - 250, route_s[0]), route_s, route.lon.to_numpy(dtype=float))
    after = np.interp(np.minimum(distance + 250, route_s[-1]), route_s, route.lat.to_numpy(dtype=float))
    after_lon = np.interp(np.minimum(distance + 250, route_s[-1]), route_s, route.lon.to_numpy(dtype=float))
    lat1, lat2 = np.radians(before), np.radians(after)
    dlon = np.radians(after_lon - before_lon)
    heading = np.degrees(np.arctan2(np.sin(dlon) * np.cos(lat2),
                                    np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon))) % 360

    # NOAA fractional-year approximation is adequate for comparing broad panel-angle hypotheses.
    local = pd.to_datetime(x.hour_local).dt.tz_localize(TZ)
    utc = local.dt.tz_convert("UTC")
    day_of_year = utc.dt.dayofyear.to_numpy(dtype=float)
    utc_hour = (utc.dt.hour + utc.dt.minute / 60 + utc.dt.second / 3600).to_numpy(dtype=float)
    gamma = 2 * np.pi / 365 * (day_of_year - 1 + (utc_hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    longitude = x.route_longitude.to_numpy(dtype=float)
    true_solar_minutes = (utc_hour * 60 + 4 * longitude + eqtime) % 1440
    hour_angle = np.radians(true_solar_minutes / 4 - 180)
    latitude = np.radians(x.route_latitude.to_numpy(dtype=float))
    sin_alt = np.sin(latitude) * np.sin(decl) + np.cos(latitude) * np.cos(decl) * np.cos(hour_angle)
    altitude = np.arcsin(np.clip(sin_alt, -1, 1))
    cos_alt = np.maximum(np.cos(altitude), 1e-9)
    sin_az = -np.cos(decl) * np.sin(hour_angle) / cos_alt
    cos_az = (np.sin(decl) - np.sin(latitude) * np.sin(altitude)) / (np.cos(latitude) * cos_alt)
    azimuth = np.degrees(np.arctan2(sin_az, cos_az)) % 360

    out = x.copy()
    out["route_heading_deg"] = heading
    out["solar_zenith_deg"] = 90 - np.degrees(altitude)
    out["solar_azimuth_deg"] = azimuth
    return out


def poa_irradiance(x: pd.DataFrame, tilt_deg: float, heading_offset_deg: float,
                   albedo: float = 0.2) -> np.ndarray:
    """Isotropic-sky POA proxy using Open-Meteo DNI/DHI and route heading."""
    tilt = np.radians(tilt_deg)
    zenith = np.radians(x.solar_zenith_deg.to_numpy(dtype=float))
    sun_az = np.radians(x.solar_azimuth_deg.to_numpy(dtype=float))
    panel_az = np.radians((x.route_heading_deg.to_numpy(dtype=float) + heading_offset_deg) % 360)
    dni = np.maximum(0, np.nan_to_num(x.direct_normal_irradiance.to_numpy(dtype=float), nan=0.0))
    dhi = np.maximum(0, np.nan_to_num(x.diffuse_radiation.to_numpy(dtype=float), nan=0.0))
    ghi = np.maximum(0, x.ghi.to_numpy(dtype=float))
    cos_aoi = np.cos(zenith) * np.cos(tilt) + np.sin(zenith) * np.sin(tilt) * np.cos(sun_az - panel_az)
    beam = dni * np.maximum(cos_aoi, 0)
    sky_diffuse = dhi * (1 + np.cos(tilt)) / 2
    ground_reflected = ghi * albedo * (1 - np.cos(tilt)) / 2
    return beam + sky_diffuse + ground_reflected


def fit_poa(x: pd.DataFrame) -> dict:
    """Fit an empirical effective gain and panel plane against a POA proxy."""
    ghi = x.ghi.to_numpy(dtype=float)
    y = x.pv_w.to_numpy(dtype=float)

    def nll(z):
        pred = z[0] + z[1] * poa_irradiance(x, z[2], z[3])
        return float(-np.sum(student_t.logpdf(y - pred, df=z[5], loc=0, scale=z[4])))

    best = None
    for tilt0 in (5.0, 20.0, 35.0):
        for offset0 in (-150.0, -90.0, -30.0, 30.0, 90.0, 150.0):
            fit = minimize(nll, x0=[0.0, 0.45, tilt0, offset0, 60.0, 5.0], method="L-BFGS-B",
                           bounds=[(-100, 150), (0, 5), (0, 60), (-180, 180), (1, 500), (2.1, 30)],
                           options={"maxiter": 800})
            if best is None or fit.fun < best.fun:
                best = fit
    return {"intercept_w": float(best.x[0]), "effective_gain_m2": float(best.x[1]),
            "panel_tilt_deg_inferred": float(best.x[2]), "panel_azimuth_offset_from_route_deg_inferred": float(best.x[3]),
            "temperature_gamma_per_c": None, "likelihood_scale_w": float(best.x[4]),
            "student_t_degrees_of_freedom": float(best.x[5]), "log_likelihood": float(-best.fun),
            "parameters": best.x[:4], "model_kind": "poa", "n": len(x),
            "optimizer_success": bool(best.success), "optimizer_message": str(best.message)}


def predict(x: pd.DataFrame, model: dict) -> np.ndarray:
    p = model["parameters"]
    ghi = x.ghi.to_numpy(dtype=float)
    if model["model_kind"] == "base":
        raw = p[0] + p[1] * ghi
    elif model["model_kind"] == "origin":
        raw = p[0] * ghi
    elif model["model_kind"] == "temperature":
        raw = p[0] + p[1] * ghi * (1 + p[2] * (x.temp_c.to_numpy(dtype=float) - 25))
    elif model["model_kind"] == "poa":
        raw = p[0] + p[1] * poa_irradiance(x, p[2], p[3])
    else:
        dni = np.nan_to_num(x.direct_normal_irradiance.to_numpy(dtype=float), nan=0.0)
        raw = p[0] + p[1] * ghi + p[2] * dni
    # Fits use irradiance >=50 W/m²; do not extrapolate the daylight line into twilight/night.
    return np.where(ghi >= 50, np.maximum(raw, 0), 0.0)


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    e = y - pred
    den = np.sum((y - np.mean(y)) ** 2)
    return {"n": int(len(y)), "mae_w": float(np.mean(np.abs(e))), "rmse_w": float(np.sqrt(np.mean(e**2))),
            "bias_measured_minus_model_w": float(np.mean(e)), "r2": float(1 - np.sum(e**2) / den) if den > 0 else None}


def lodo(data: pd.DataFrame, fitter) -> dict:
    rows = []
    for day in sorted(data.day.unique()):
        train = data[data.day != day]
        test = data[data.day == day]
        if len(test) < 5 or len(train) < 20:
            continue
        m = fitter(train)
        rows.append({"held_out_day": int(day), **scores(test.pv_w.to_numpy(), predict(test, m)),
                     "intercept_w": m["intercept_w"], "effective_gain_m2": m["effective_gain_m2"],
                     "temperature_gamma_per_c": m["temperature_gamma_per_c"], "likelihood_scale_w": m["likelihood_scale_w"],
                     "student_t_degrees_of_freedom": m["student_t_degrees_of_freedom"],
                     "panel_tilt_deg_inferred": m.get("panel_tilt_deg_inferred"),
                     "panel_azimuth_offset_from_route_deg_inferred": m.get("panel_azimuth_offset_from_route_deg_inferred")})
    if not rows:
        return {"per_day": [], "mean_mae_w": None, "mean_rmse_w": None}
    return {"per_day": rows, "mean_mae_w": float(np.mean([r["mae_w"] for r in rows])),
            "mean_rmse_w": float(np.mean([r["rmse_w"] for r in rows]))}


def main() -> None:
    path = ROOT / "data/processed/race_days_distance_geo_weather.csv"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    ghi_col = next((c for c in GHI_CHOICES if c in header), None)
    if ghi_col is None:
        raise ValueError(f"Neither GHI variable found in {path}: {GHI_CHOICES}")
    wanted = ["day", "timestamp_local", "event_flags", "Solar_Watt[W]", "Battery_Watt[W]", "Speed[kmh]", "distance_from_darwin_m",
              "route_latitude", "route_longitude", "temperature_2m", "wind_speed_10m", "cloud_cover", "direct_normal_irradiance",
              "direct_radiation", "diffuse_radiation", ghi_col]
    df = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
    df = df.rename(columns={ghi_col: "ghi", "Solar_Watt[W]": "pv_w", "Battery_Watt[W]": "battery_signed_w", "temperature_2m": "temp_c"})
    df["day"] = pd.to_numeric(df.day, errors="coerce").astype(int)
    for col in ("pv_w", "battery_signed_w", "Speed[kmh]", "distance_from_darwin_m", "route_latitude", "route_longitude", "ghi", "temp_c", "wind_speed_10m", "cloud_cover", "direct_normal_irradiance", "direct_radiation", "diffuse_radiation"):
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["affected_event"] = df.event_flags.fillna("").str.contains(r"incident:|repair:|hatch_repair:", regex=True)
    # Hourly aggregation matches the true temporal resolution of the weather source and avoids 5-second pseudoreplication.
    local_dt = pd.to_datetime(df.timestamp_local, utc=True).dt.tz_convert(TZ)
    df["hour_local"] = local_dt.dt.floor("h").dt.strftime("%Y-%m-%dT%H:%M")
    numeric = ["pv_w", "battery_signed_w", "Speed[kmh]", "distance_from_darwin_m", "route_latitude", "route_longitude", "ghi", "temp_c", "wind_speed_10m", "cloud_cover", "direct_normal_irradiance", "direct_radiation", "diffuse_radiation"]
    hourly = df.groupby(["day", "hour_local"], as_index=False)[numeric].mean()
    affected = df.groupby(["day", "hour_local"], as_index=False).affected_event.max()
    hourly = hourly.merge(affected, on=["day", "hour_local"], how="left")
    hourly = solar_geometry(hourly)
    full = hourly.dropna(subset=["pv_w", "ghi", "temp_c"]).copy()
    full["ghi"] = full.ghi.clip(lower=0)
    # Fit daylight rows with useful signal; retain event-affected daytime rows for diagnosis, exclude them from baseline fit.
    eligible = full[(full.ghi >= 50) & (full.pv_w >= 0) & (~full.affected_event)].copy()
    if len(eligible) < 20:
        raise ValueError(f"Too few eligible hourly PV points: {len(eligible)}")
    base = fit_base(eligible)
    origin = fit_origin(eligible)
    temp = fit_temp(eligible)
    beam_data = eligible.dropna(subset=["direct_normal_irradiance"])
    beam = fit_beam(beam_data) if len(beam_data) >= 20 else None
    poa_data = eligible.dropna(subset=["direct_normal_irradiance", "diffuse_radiation", "route_latitude", "route_longitude"])
    poa = fit_poa(poa_data) if len(poa_data) >= 20 else None
    comparison_data = beam_data if beam is not None else eligible
    base_lodo = lodo(comparison_data, fit_base)
    temp_lodo = lodo(comparison_data, fit_temp)
    beam_lodo = lodo(beam_data, fit_beam) if beam is not None else {"per_day": [], "mean_mae_w": None, "mean_rmse_w": None}
    origin_lodo = lodo(eligible, fit_origin)
    poa_lodo = lodo(poa_data, fit_poa) if poa is not None else {"per_day": [], "mean_mae_w": None, "mean_rmse_w": None}
    # Add complexity only when leave-one-day-out MAE improves by at least 2%.
    candidates = [("ghi_linear_effective_gain", base, base_lodo)]
    candidates.append(("zero_intercept_ghi_gain", origin, origin_lodo))
    if temp_lodo["mean_mae_w"] is not None:
        candidates.append(("ghi_plus_ambient_temperature_proxy", temp, temp_lodo))
    if beam is not None and beam_lodo["mean_mae_w"] is not None:
        candidates.append(("ghi_plus_direct_beam_empirical", beam, beam_lodo))
    if poa is not None and poa_lodo["mean_mae_w"] is not None:
        candidates.append(("route_heading_solar_angle_poa_proxy", poa, poa_lodo))
    chosen_name, chosen, chosen_lodo = candidates[0]
    for candidate_name, candidate, candidate_lodo in candidates[1:]:
        if candidate_lodo["mean_mae_w"] < chosen_lodo["mean_mae_w"] * 0.98:
            chosen_name, chosen, chosen_lodo = candidate_name, candidate, candidate_lodo
    full["predicted_pv_w"] = predict(full, chosen)
    full["residual_measured_minus_predicted_w"] = full.pv_w - full.predicted_pv_w
    full["fit_eligible"] = (full.ghi >= 50) & (full.pv_w >= 0) & (~full.affected_event)
    fit_metrics = scores(eligible.pv_w.to_numpy(), predict(eligible, chosen))
    # Temperature, cloud, wind, and direct-beam associations are diagnostic evidence only, not presumed causal effects.
    residual = eligible.pv_w.to_numpy() - predict(eligible, chosen)
    residual_correlations = {}
    for variable in ("ghi", "temp_c", "wind_speed_10m", "cloud_cover", "direct_normal_irradiance", "Speed[kmh]", "distance_from_darwin_m"):
        x = eligible[variable].to_numpy(dtype=float)
        good = np.isfinite(x) & np.isfinite(residual)
        if good.sum() >= 10:
            corr, pval = spearmanr(x[good], residual[good])
            residual_correlations[variable] = {"spearman_rho": float(corr), "p_value_descriptive_only": float(pval), "n": int(good.sum())}
    # Measured-to-model ratio by observed irradiance and ambient temperature, based on hourly aggregate data.
    map_data = eligible[eligible.ghi > 100].copy()
    map_data["effective_gain_m2"] = map_data.pv_w / map_data.ghi
    map_data["ghi_band"] = pd.cut(map_data.ghi, bins=[100, 250, 400, 550, 700, 850, 1000, 2000], right=False)
    map_data["ambient_temp_band_c"] = pd.cut(map_data.temp_c, bins=[-20, 10, 20, 30, 40, 60], right=False)
    gain_map = map_data.groupby(["ghi_band", "ambient_temp_band_c"], observed=True).agg(
        n_hourly_bins=("effective_gain_m2", "size"), median_effective_gain_m2=("effective_gain_m2", "median"),
        mean_effective_gain_m2=("effective_gain_m2", "mean"), median_pv_w=("pv_w", "median"), mean_ghi_w_m2=("ghi", "mean"), mean_ambient_temp_c=("temp_c", "mean")
    ).reset_index()
    gain_map.to_csv(ROOT / "models/pv_effective_gain_map.csv", index=False, encoding="utf-8-sig")
    # Residual summaries isolate known events and race days without treating them as new sensor measurements.
    day_rows = []
    for day, group in full.groupby("day"):
        row = {"day": int(day), **scores(group.pv_w.to_numpy(), group.predicted_pv_w.to_numpy()),
               "eligible_hours": int((group.fit_eligible).sum()), "event_affected_hours": int(group.affected_event.sum())}
        day_rows.append(row)
    outlog = full[["day", "hour_local", "distance_from_darwin_m", "ghi", "temp_c", "pv_w", "battery_signed_w", "predicted_pv_w", "residual_measured_minus_predicted_w", "affected_event", "fit_eligible", "wind_speed_10m", "cloud_cover", "direct_normal_irradiance"]]
    outlog.to_csv(ROOT / "data/processed/pv_model_hourly_diagnostic.csv", index=False, encoding="utf-8-sig")
    fit_json = {
        "target": "ZP Solar_Watt[W] (PV output channel)", "weather_driver": ghi_col,
        "weather_resolution_used_for_fit": "hourly Historical Forecast data, after spatial and time interpolation; ZP values aggregated to local hourly means",
        "baseline_model": "P_pv = b0 + b1*GHI; b1 has units W/(W/m2) = m2-equivalent effective gain; prediction is gated to zero below 50 W/m2 because the model is not trained there",
        "zero_intercept_candidate": "P_pv = b1*GHI, constrained through origin; prediction is gated to zero below 50 W/m2",
        "temperature_candidate": "P_pv = b0 + b1*GHI*(1 + gamma*(Tair-25C)); ambient temperature is only a proxy for module temperature",
        "direct_beam_candidate": "P_pv = b0 + bG*GHI + bD*DNI; bD is only an empirical direct-beam proxy, not a panel incidence-angle parameter",
        "poa_candidate": "P_pv = b0 + gain*POA(tilt, route heading + azimuth offset); isotropic diffuse-sky and 0.2 ground albedo are assumed; panel orientation is fitted, not measured",
        "selected_model": chosen_name, "parameters": {k: v for k, v in chosen.items() if k != "parameters"},
        "candidate_parameter_estimates": {
            "ghi_with_offset": {k: v for k, v in base.items() if k != "parameters"},
            "ghi_through_origin": {k: v for k, v in origin.items() if k != "parameters"},
            "ambient_temperature_proxy": {k: v for k, v in temp.items() if k != "parameters"},
            "direct_beam_empirical": {k: v for k, v in beam.items() if k != "parameters"} if beam is not None else None,
            "route_heading_poa_proxy": {k: v for k, v in poa.items() if k != "parameters"} if poa is not None else None,
        },
        "likelihood": "Student-t residual maximum-likelihood estimate; scale and degrees of freedom are estimated with the power parameters, and LODO scoring is grouped by race day.",
        "baseline_fit_metrics_in_sample": scores(eligible.pv_w.to_numpy(), predict(eligible, base)),
        "zero_intercept_fit_metrics_in_sample": scores(eligible.pv_w.to_numpy(), predict(eligible, origin)),
        "temperature_fit_metrics_in_sample": scores(eligible.pv_w.to_numpy(), predict(eligible, temp)),
        "direct_beam_fit_metrics_in_sample": scores(beam_data.pv_w.to_numpy(), predict(beam_data, beam)) if beam is not None else None,
        "poa_fit_metrics_in_sample": scores(poa_data.pv_w.to_numpy(), predict(poa_data, poa)) if poa is not None else None,
        "selected_fit_metrics_in_sample": fit_metrics, "baseline_leave_one_day_out": base_lodo, "temperature_leave_one_day_out": temp_lodo,
        "direct_beam_leave_one_day_out": beam_lodo, "poa_leave_one_day_out": poa_lodo,
        "zero_intercept_leave_one_day_out": origin_lodo,
        "selected_model_day_metrics": day_rows, "residual_correlations_spearman": residual_correlations,
        "event_affected_hours_excluded_from_baseline_fit": int(full.affected_event.sum()), "eligible_training_hour_bins": len(eligible),
        "physical_identifiability": "Effective gain b1 = module conversion efficiency × effective illuminated collector area × DC-side/MPPT factor; without installed active area and actual panel plane these factors cannot be separated and this is not a module-efficiency percentage.",
        "limitations": ["Open-Meteo hourly modeled GHI is not pyranometer truth; route interpolation is also modeled.",
                        "No module temperature, panel tilt/azimuth, active area, STC rating, spectral/angle response; a route-heading POA candidate can only infer an empirical orientation, not verify it or identify physical module efficiency.",
                        "Rows are temporally autocorrelated; fit/holdout is blocked by race day and weather aggregation avoids counting each 5-second sample as independent evidence.",
                        "Incident periods are omitted from baseline fitting and separately retained in hourly diagnostic output.",
                        "This fits PV output only. It does not yet identify the traction/accessory consumption model."]
    }
    (ROOT / "models/pv_model_fit.json").write_text(json.dumps(fit_json, indent=2, ensure_ascii=False), encoding="utf-8")
    # Diagnostic figure: model fit, residuals by day, and error versus irradiance.
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].scatter(eligible.ghi, eligible.pv_w, c=eligible.temp_c, cmap="coolwarm", s=19, alpha=0.75)
    maxghi = float(eligible.ghi.max())
    line_x = np.linspace(50, maxghi, 100)
    grid = pd.DataFrame({"ghi": line_x, "temp_c": float(eligible.temp_c.mean()),
                         "direct_normal_irradiance": float(eligible.direct_normal_irradiance.mean()),
                         "diffuse_radiation": float(eligible.diffuse_radiation.mean()),
                         "solar_zenith_deg": float(eligible.solar_zenith_deg.mean()),
                         "solar_azimuth_deg": float(eligible.solar_azimuth_deg.mean()),
                         "route_heading_deg": float(eligible.route_heading_deg.mean())})
    line_y = predict(grid, chosen)
    axes[0].plot(line_x, line_y, color="black", lw=2)
    axes[0].set(xlabel="Open-Meteo GHI [W/m²]", ylabel="ZP solar power [W]", title="Eligible hourly means")
    sc = axes[1].scatter(full.day, full.residual_measured_minus_predicted_w, c=full.ghi, cmap="viridis", s=22, alpha=0.8)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set(xlabel="Race day", ylabel="Measured − model [W]", title="Residual by day (color = GHI)")
    fig.colorbar(sc, ax=axes[1], label="GHI [W/m²]")
    axes[2].scatter(eligible.ghi, eligible.pv_w - predict(eligible, chosen), c=eligible.day, cmap="tab10", s=19, alpha=0.75)
    axes[2].axhline(0, color="black", lw=1)
    axes[2].set(xlabel="Open-Meteo GHI [W/m²]", ylabel="Residual [W]", title="Residual versus irradiance")
    fig.suptitle(f"YATA first-pass PV fit: {chosen_name}")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/pv_model_diagnostics.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    heldout_bias = {row["held_out_day"]: row["bias_measured_minus_model_w"] for row in base_lodo["per_day"]}
    heldout_bias_text = ", ".join(f"Day {day} {bias:+.0f} W" for day, bias in heldout_bias.items())
    poa_fold_offsets = [row["panel_azimuth_offset_from_route_deg_inferred"] for row in poa_lodo["per_day"]
                        if row.get("panel_azimuth_offset_from_route_deg_inferred") is not None]
    poa_offset_spread = max(poa_fold_offsets) - min(poa_fold_offsets) if poa_fold_offsets else float("nan")
    report = [
        "# YATA PV generation model: first fit", "",
        f"Selected model: `{chosen_name}`. It uses {len(eligible)} hourly bins from the six logged road days; day-held-out MAE is **{chosen_lodo['mean_mae_w']:.1f} W**.", "",
        f"The fitted effective gain is **{chosen['effective_gain_m2']:.4f} m²-equivalent** (intercept {chosen['intercept_w']:.1f} W). This is not panel efficiency: it combines cell conversion, illuminated area, orientation, wiring, and MPPT effects.", "",
        "## Data and fit choices", "",
        f"- ZP solar power is the target; GHI driver is `{ghi_col}` from Open-Meteo Historical Forecast. Model outputs were aggregated to local-hour bins to respect weather resolution.",
        "- Student-t residual maximum-likelihood fitting jointly estimates model parameters, residual scale, and degrees of freedom; leave-one-race-day-out scores check generalization across days.",
        "- MPPT crimp, hatch, tire, repair, and ZP anomaly incident windows were excluded from the healthy baseline and preserved for diagnosis.",
        f"- Healthy baseline in-sample MAE/RMSE: {fit_json['baseline_fit_metrics_in_sample']['mae_w']:.1f}/{fit_json['baseline_fit_metrics_in_sample']['rmse_w']:.1f} W. Temperature candidate: {fit_json['temperature_fit_metrics_in_sample']['mae_w']:.1f}/{fit_json['temperature_fit_metrics_in_sample']['rmse_w']:.1f} W.",
        f"- Leave-one-day-out mean MAE: GHI with offset {base_lodo['mean_mae_w']:.1f} W; through-origin GHI {origin_lodo['mean_mae_w']:.1f} W; temperature candidate {temp_lodo['mean_mae_w']:.1f} W; direct-beam candidate {beam_lodo['mean_mae_w']:.1f} W; route-heading POA candidate {poa_lodo['mean_mae_w']:.1f} W.",
        "- The fitted GHI intercept is at its 150 W upper bound; it is an empirical correction term, not a physical no-sun output. Predictions below the 50 W/m² training threshold are gated to zero. The fit should not be used outside its daylight training domain.", "",
        "## What the residuals say", "",
        f"The residual figure is saved to `reports/pv_model_diagnostics.png`. Day-held-out MAE: GHI with offset {base_lodo['mean_mae_w']:.1f} W; through-origin GHI {origin_lodo['mean_mae_w']:.1f} W; ambient-temperature {temp_lodo['mean_mae_w']:.1f} W; direct-beam {beam_lodo['mean_mae_w']:.1f} W; fitted-POA {poa_lodo['mean_mae_w']:.1f} W. A candidate replaces the selected model only when it improves held-out MAE by at least 2%.",
        f"Held-out signed residual (measured minus model) by race day: {heldout_bias_text}. Day 2 is absent from the held-out table because event exclusions leave fewer than five eligible hourly bins. The route-heading POA candidate inferred a {poa['panel_tilt_deg_inferred']:.1f}° tilt in its full fit; its fold azimuth-offset estimates span {poa_offset_spread:.0f}°, and its {poa_lodo['mean_mae_w']:.1f} W MAE is worse than the GHI baseline. This fit is not evidence that the actual panel has the inferred orientation.",
        "The binned lookup `models/pv_effective_gain_map.csv` reports measured PV/GHI ratios by irradiance and ambient-temperature band. Several cells have fewer than five hourly samples, and low-irradiance ratios can exceed 3 m²-equivalent because the fitted offset and model-weather mismatch dominate there. Treat this as a diagnostic lookup, not a physical efficiency map.",
        "Residuals vary by race day and remain large relative to the model output. Forecast-grid mismatch, missing panel-angle/temperature inputs, or configuration changes can explain this pattern, but the present data do not distinguish them. The POA candidate uses a simple isotropic diffuse model; NREL PVWatts V5 uses a more detailed Perez transposition and estimates cell temperature from POA irradiance, wind, and ambient temperature. Those are candidates for a later layer once actual panel geometry and temperature evidence are available. Temperature association is not evidence of a cell-temperature coefficient because no module sensor exists. The fit is only a first-stage empirical map and should not be called validated physical efficiency.",
        "", "## Required vehicle data to identify efficiency", "",
        "1. Installed active PV area and module make/model or measured STC Pmax/Vmp/Imp; distinguish total collector area from electrically active area.",
        "2. Panel plane while driving: tilt, azimuth relative to vehicle, curvature/segment geometry, and whether hatch panels were disconnected/removed during Day 5.",
        "3. Panel back-surface temperature sensor location/calibration, plus solar-channel electrical measurement points and calibration uncertainty.",
        "4. MPPT topology/limits, channel map, and exact Day 2 crimp disconnect/reconnect time from the ZP log if known.",
        "", "The next model step is to compare the inferred POA orientation against the actual panel geometry, then add a cell-temperature correction using a calibrated module sensor or an explicitly uncertain thermal model. Until panel geometry/area and temperature are known, the area-efficiency product is the identifiable parameter.",
        "", "## Sources", "",
        "- Open-Meteo Historical Forecast API documentation: https://open-meteo.com/en/docs/historical-forecast-api",
        "- Faiman (2008), module temperature model: https://doi.org/10.1002/pip.813",
        "- NREL PVWatts v5 model description (POA transposition and DC power temperature correction): https://www.nrel.gov/docs/fy14osti/62641.pdf",
    ]
    (ROOT / "reports/pv_fit_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"selected_model": chosen_name, "training_hour_bins": len(eligible), "fit_metrics": fit_metrics,
                      "baseline_lodo_mae_w": base_lodo["mean_mae_w"], "temperature_lodo_mae_w": temp_lodo["mean_mae_w"],
                      "direct_beam_lodo_mae_w": beam_lodo["mean_mae_w"],
                      "poa_lodo_mae_w": poa_lodo["mean_mae_w"], "origin_lodo_mae_w": origin_lodo["mean_mae_w"],
                      "effective_gain_m2": chosen["effective_gain_m2"], "report": str(ROOT / "reports/pv_fit_summary.md")}, indent=2))


if __name__ == "__main__":
    raise SystemExit("Superseded for model fitting. Run scripts/fit_pv_power_model_physical.py; this module remains a shared POA-geometry helper.")
