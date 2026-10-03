#!/usr/bin/env python3
"""Fit a bounded, first-stage PV power model against BWSC2025 ZP logs.

The model deliberately uses the logged HV solar channel and hourly-averaged
Open-Meteo irradiance. There is no module temperature sensor or verified panel
plane, so the fitted gain is an effective area-equivalent factor, not module
efficiency. Fits use Student-t MLE and leave-one-race-day-out validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import spearmanr, t as student_t
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fit_pv_model as poa_tools  # noqa: E402
INPUT = ROOT / "data/processed/race_days_distance_geo_weather.csv"
OUT_HOURLY = ROOT / "data/processed/pv_model_hourly_diagnostic.csv"
OUT_MODEL = ROOT / "models/pv_model_fit.json"
OUT_CV = ROOT / "reports/pv_model_cv_by_day.csv"
OUT_MAP = ROOT / "models/pv_effective_gain_map.csv"
OUT_FIG = ROOT / "reports/pv_model_diagnostics.png"
OUT_REPORT = ROOT / "reports/pv_fit_summary.md"
TZ = "Australia/Darwin"
DF = 4.0
# 386 E66Me3 cells is a provisional count from the near-6 m2 layout and cell
# area; 3.78 W/cell is the high end of the supplied cell datasheet.
CELL_COUNT_PROVISIONAL = 386
CELL_PMAX_MAX_W = 3.78
CELL_AREA_M2 = 0.015500692
P_STC_UPPER_W = CELL_COUNT_PROVISIONAL * CELL_PMAX_MAX_W
GAIN_UPPER = P_STC_UPPER_W / 1000.0


def score(y, yhat) -> dict:
    e = np.asarray(y, float) - np.asarray(yhat, float)
    den = np.sum((np.asarray(y)-np.mean(y))**2)
    return {"n_hour_bins": int(len(e)), "mae_w": float(np.mean(np.abs(e))),
            "rmse_w": float(np.sqrt(np.mean(e**2))),
            "bias_measured_minus_model_w": float(np.mean(e)),
            "r2": float(1-np.sum(e**2)/den) if den > 0 else None}


def prediction(x: pd.DataFrame, pars: dict, kind: str) -> np.ndarray:
    ghi = x.ghi.to_numpy(float).clip(0)
    base = pars["gain_m2"] * ghi
    if kind == "ambient_temperature_proxy":
        base *= np.maximum(0.0, 1.0 + pars["gamma_per_c"] * (x.temp_c.to_numpy(float) - 25.0))
    elif kind == "fixed_body_plane_poa":
        base = pars["gain_m2"] * poa_tools.poa_irradiance(
            x, pars["tilt_deg"], pars["heading_offset_deg"])
    elif kind == "faceted_body_poa":
        facets = facet_irradiance(x)
        gains = np.array([pars["facet_gains_m2"][name] for name in FACET_NAMES])
        base = facets @ gains
    return base


FACET_NAMES = ["roof", "front", "rear", "left_side", "right_side"]


def facet_irradiance(x: pd.DataFrame) -> np.ndarray:
    """Canonical flat roof plus vertical nose/tail/side surfaces, as a hypothesis."""
    planes = [(0., 0.), (90., 0.), (90., 180.), (90., 90.), (90., -90.)]
    return np.column_stack([poa_tools.poa_irradiance(x, tilt, offset) for tilt, offset in planes])


def fit(x: pd.DataFrame, kind: str) -> dict:
    ghi, y = x.ghi.to_numpy(float).clip(0), x.pv_w.to_numpy(float)
    temp = x.temp_c.to_numpy(float)
    if kind == "ghi_through_origin":
        def pred(z): return z[0] * ghi
        initial, bounds = [0.9, 100.0], [(0, GAIN_UPPER), (1, 600)]
        names = ["gain_m2", "scale_w"]
    elif kind == "ambient_temperature_proxy":
        def pred(z): return z[0] * ghi * np.maximum(0.0, 1.0 + z[1] * (temp - 25.0))
        initial, bounds = [0.9, -0.003, 100.0], [(0, GAIN_UPPER), (-0.02, 0.01), (1, 600)]
        names = ["gain_m2", "gamma_per_c", "scale_w"]
    elif kind == "fixed_body_plane_poa":
        # Panel body orientation is fixed while driving. This tests a fixed plane
        # relative to the route heading; it is not a substitute for measured normals.
        best = None
        def nll_poa(z):
            irradiance = poa_tools.poa_irradiance(x, z[1], z[2])
            scale = z[3]
            return float(-np.sum(student_t.logpdf((y-z[0]*irradiance)/scale,df=DF)-np.log(scale)))
        for tilt0 in (0., 10., 25., 40.):
            for offset0 in (-135., -45., 45., 135.):
                candidate=minimize(nll_poa,[0.9,tilt0,offset0,100.],method="L-BFGS-B",
                    bounds=[(0,GAIN_UPPER),(0,60),(-180,180),(1,600)],
                    options={"maxiter":350,"ftol":1e-9})
                if best is None or candidate.fun<best.fun: best=candidate
        return {"kind":kind,"parameters":{"gain_m2":float(best.x[0]),"tilt_deg":float(best.x[1]),
                "heading_offset_deg":float(best.x[2])},"scale_w":float(best.x[3]),"df":DF,
                "log_likelihood":float(-best.fun),"optimizer_success":bool(best.success),
                "optimizer_message":str(best.message),"n":len(x)}
    else:
        # Hypothesis prompted by large dawn/dusk underprediction: divide the
        # effective aperture among roof, nose, tail and left/right body facets.
        # Non-negative facet gains sum to the cell-STC area-equivalent ceiling.
        F = facet_irradiance(x)
        best = None
        starts = [[1.0,0.,0.,0.,0.,100.], [1.0,2.,0.,0.,0.,100.],
                  [1.0,0.,2.,0.,0.,100.], [1.0,0.,0.,2.,0.,100.],
                  [1.0,0.,0.,0.,2.,100.]]
        def unpack(z):
            fractions = softmax(np.r_[z[1:5], 0.0])
            return z[0] * fractions
        def nll_facets(z):
            gains = unpack(z)
            scale = z[5]
            return float(-np.sum(student_t.logpdf((y-F@gains)/scale,df=DF)-np.log(scale)))
        for start in starts:
            candidate=minimize(nll_facets,start,method="L-BFGS-B",
                bounds=[(0,GAIN_UPPER)]+[(-8,8)]*4+[(1,600)],
                options={"maxiter":800,"ftol":1e-10})
            if best is None or candidate.fun<best.fun: best=candidate
        gains=unpack(best.x)
        params={"gain_m2":float(gains.sum()),
                "facet_gains_m2":{name:float(value) for name,value in zip(FACET_NAMES,gains)}}
        return {"kind":kind,"parameters":params,"scale_w":float(best.x[5]),"df":DF,
                "log_likelihood":float(-best.fun),"optimizer_success":bool(best.success),
                "optimizer_message":str(best.message),"n":len(x)}
    def nll(z):
        scale = z[-1]
        return float(-np.sum(student_t.logpdf((y-pred(z))/scale, df=DF)-np.log(scale)))
    opt = minimize(nll, initial, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter":1000,"ftol":1e-11})
    p = {key: float(value) for key,value in zip(names,opt.x)}
    pars = {k:v for k,v in p.items() if k != "scale_w"}
    return {"kind":kind,"parameters":pars,"scale_w":p["scale_w"],"df":DF,
            "log_likelihood":float(-opt.fun),"optimizer_success":bool(opt.success),
            "optimizer_message":str(opt.message),"n":len(x)}


def lodo(x: pd.DataFrame, kind: str) -> list[dict]:
    rows=[]
    for day in sorted(x.day.unique()):
        tr,te=x[x.day.ne(day)],x[x.day.eq(day)]
        if len(te)<2 or len(tr)<10: continue
        f=fit(tr,kind)
        pred=prediction(te,f["parameters"],kind)
        rows.append({"held_out_day":int(day),**score(te.pv_w,pred),
                     "gain_m2":f["parameters"]["gain_m2"],
                     "gamma_per_c":f["parameters"].get("gamma_per_c"),
                     "tilt_deg":f["parameters"].get("tilt_deg"),
                     "heading_offset_deg":f["parameters"].get("heading_offset_deg"),
                     **{f"facet_{name}_gain_m2":value for name,value in f["parameters"].get("facet_gains_m2",{}).items()}})
    return rows


def main() -> None:
    wanted={"day","timestamp_local","event_flags","event_phase","Solar_Watt[W]","Speed[kmh]",
            "distance_from_darwin_m","route_latitude","route_longitude","temperature_2m",
            "shortwave_radiation_instant","shortwave_radiation","direct_normal_irradiance",
            "diffuse_radiation","wind_speed_10m","cloud_cover"}
    d=pd.read_csv(INPUT,usecols=lambda c:c in wanted,low_memory=False)
    d=d.rename(columns={"Solar_Watt[W]":"pv_w","temperature_2m":"temp_c",
                        "shortwave_radiation_instant":"ghi_instant"})
    for c in ["pv_w","temp_c","ghi_instant","shortwave_radiation","direct_normal_irradiance",
              "diffuse_radiation","Speed[kmh]","distance_from_darwin_m","route_latitude","route_longitude"]:
        if c in d: d[c]=pd.to_numeric(d[c],errors="coerce")
    d["in_motion_sample"]=(d.event_phase.eq("road_day")&d["Speed[kmh]"].ge(5.0))
    local=pd.to_datetime(d.timestamp_local,utc=True).dt.tz_convert(TZ)
    d["hour_local"]=local.dt.floor("h").dt.strftime("%Y-%m-%dT%H:%M")
    # Hourly power and modeled weather averages match the effective weather
    # resolution; the Australian 15-minute endpoint is interpolated from hourly.
    event_regex=r"incident:|repair:|hatch_repair:|D5_HATCH_FAILURE|D5_PUNCTURE|D6_ZP_ANOMALY|D6_RETIRE"
    d["event_affected"]=d.event_flags.fillna("").astype(str).str.contains(event_regex,regex=True)
    nums=[c for c in ["pv_w","temp_c","ghi_instant","shortwave_radiation","direct_normal_irradiance",
                      "diffuse_radiation","wind_speed_10m","cloud_cover","Speed[kmh]","distance_from_darwin_m",
                      "route_latitude","route_longitude"] if c in d]
    hourly=d.groupby(["day","hour_local"],as_index=False)[nums].mean()
    motion=d.groupby(["day","hour_local"],as_index=False).in_motion_sample.mean().rename(columns={"in_motion_sample":"motion_sample_fraction"})
    affected=d.groupby(["day","hour_local"],as_index=False).event_affected.max()
    hourly=hourly.merge(affected,on=["day","hour_local"],how="left").merge(motion,on=["day","hour_local"],how="left")
    # Instant weather irradiance from the Open-Meteo hourly dataset is more
    # appropriate for power; integrated GHI is retained for a sensitivity check.
    if "ghi_instant" in hourly and hourly.ghi_instant.notna().sum() >= len(hourly)*.5:
        hourly["ghi"]=hourly.ghi_instant.clip(lower=0)
        weather_name="shortwave_radiation_instant"
    else:
        hourly["ghi"]=hourly.shortwave_radiation.clip(lower=0)
        weather_name="shortwave_radiation"
    hourly["day"]=pd.to_numeric(hourly.day,errors="coerce").astype(int)
    # Use a fitted simple GHI map and only retain thermal complexity if held-out
    # error improves. The gamma is a temperature proxy, not a module coefficient.
    hourly=poa_tools.solar_geometry(hourly)
    eligible_all=hourly.dropna(subset=["pv_w","ghi","temp_c"])
    eligible_all=eligible_all[(eligible_all.ghi>=50)&(eligible_all.pv_w>=0)&(~eligible_all.event_affected)
                              &(eligible_all.motion_sample_fraction>=.8)].copy()
    heldout_d6=eligible_all[eligible_all.day.eq(6)].copy()
    eligible=eligible_all[eligible_all.day.ne(6)].copy()
    models={k:fit(eligible,k) for k in ["ghi_through_origin","ambient_temperature_proxy"]}
    poa_eligible=eligible.dropna(subset=["route_latitude","route_longitude","direct_normal_irradiance","diffuse_radiation"]).copy()
    if len(poa_eligible)>=20:
        models["fixed_body_plane_poa"]=fit(poa_eligible,"fixed_body_plane_poa")
        models["faceted_body_poa"]=fit(poa_eligible,"faceted_body_poa")
    folds={k:lodo(poa_eligible if k in {"fixed_body_plane_poa","faceted_body_poa"} else eligible,k) for k in models}
    means={k:float(np.mean([r["mae_w"] for r in rows])) if rows else float("inf") for k,rows in folds.items()}
    chosen="ghi_through_origin"
    # Temperature is only a proxy without a module sensor, and POA orientation
    # is only adopted when blocked-day angle estimates are also stable.
    if means.get("ambient_temperature_proxy",float("inf")) < .98*means[chosen]:
        chosen="ambient_temperature_proxy"
    poa_rows=folds.get("fixed_body_plane_poa",[])
    fold_tilts=[r["tilt_deg"] for r in poa_rows if r.get("tilt_deg") is not None]
    fold_az=[r["heading_offset_deg"] for r in poa_rows if r.get("heading_offset_deg") is not None]
    az_resultant=float(np.hypot(np.mean(np.cos(np.radians(fold_az))),np.mean(np.sin(np.radians(fold_az))))) if fold_az else None
    poa_angle_stable=bool(len(fold_tilts)>=5 and (max(fold_tilts)-min(fold_tilts)<=10.)
                      and az_resultant is not None and az_resultant>=.8)
    poa_candidate_qualifies=("fixed_body_plane_poa" in means and means["fixed_body_plane_poa"]<.98*means[chosen]
                             and poa_angle_stable)
    if poa_candidate_qualifies: chosen="fixed_body_plane_poa"
    facet_rows=folds.get("faceted_body_poa",[])
    facet_matrix=np.array([[r.get(f"facet_{name}_gain_m2",0.0) for name in FACET_NAMES] for r in facet_rows],dtype=float)
    facet_means=facet_matrix.mean(axis=0) if len(facet_matrix) else np.array([])
    active_facets=facet_means>.05 if len(facet_means) else np.array([],dtype=bool)
    facet_cv=(facet_matrix.std(axis=0,ddof=0)/np.maximum(facet_means,.01)) if len(facet_matrix) else np.array([])
    facet_stable=bool(len(facet_rows)>=5 and active_facets.any() and np.nanmax(facet_cv[active_facets])<=.75
                  and np.std(facet_matrix.sum(axis=1))/max(np.mean(facet_matrix.sum(axis=1)),.01)<=.25)
    facet_candidate_qualifies=bool("faceted_body_poa" in means and means["faceted_body_poa"]<.95*means[chosen]
                              and facet_stable)
    if facet_candidate_qualifies: chosen="faceted_body_poa"
    selected=models[chosen]
    hourly["model_kind"]=chosen
    hourly["predicted_pv_w"]=prediction(hourly.fillna({"ghi":0,"temp_c":25}),selected["parameters"],chosen)
    hourly.loc[hourly.ghi.fillna(0)<50,"predicted_pv_w"]=0.0
    hourly["residual_measured_minus_predicted_w"]=hourly.pv_w-hourly.predicted_pv_w
    hourly["fit_eligible"]=(hourly.ghi>=50)&(hourly.pv_w>=0)&(~hourly.event_affected)&hourly.day.ne(6)&hourly.motion_sample_fraction.ge(.8)
    hourly["heldout_day6"]=(hourly.ghi>=50)&(hourly.pv_w>=0)&(~hourly.event_affected)&hourly.day.eq(6)&hourly.motion_sample_fraction.ge(.8)
    hourly.to_csv(OUT_HOURLY,index=False,encoding="utf-8-sig")
    cv=[]
    for kind,rows in folds.items():
        for row in rows: cv.append({"model":kind,**row})
    pd.DataFrame(cv).to_csv(OUT_CV,index=False,encoding="utf-8-sig")
    fit_frame=poa_eligible if chosen in {"fixed_body_plane_poa","faceted_body_poa"} else eligible
    fit_pred=prediction(fit_frame,selected["parameters"],chosen)
    fit_metrics=score(fit_frame.pv_w,fit_pred)
    residual=fit_frame.pv_w.to_numpy(float)-fit_pred
    residual_correlations={}
    for variable in ["ghi","temp_c","wind_speed_10m","cloud_cover","Speed[kmh]","distance_from_darwin_m"]:
        if variable not in fit_frame: continue
        xv=pd.to_numeric(fit_frame[variable],errors="coerce").to_numpy(float)
        ok=np.isfinite(xv)&np.isfinite(residual)
        if ok.sum()>=8:
            rho,pval=spearmanr(xv[ok],residual[ok])
            residual_correlations[variable]={"spearman_rho":float(rho),"p_value_descriptive_only":float(pval),"n":int(ok.sum())}
    # Effective conversion map bins; report counts so sparse bins are obvious.
    ratio=eligible[eligible.ghi>100].copy()
    ratio["effective_gain_m2_observed"]=ratio.pv_w/ratio.ghi
    ratio["ghi_band_w_m2"]=pd.cut(ratio.ghi,[100,250,400,550,700,850,1000,1500,2500],right=False)
    ratio["ambient_temp_band_c"]=pd.cut(ratio.temp_c,[-20,10,20,30,40,60],right=False)
    gain_map=ratio.groupby(["ghi_band_w_m2","ambient_temp_band_c"],observed=True).agg(
        hour_bins=("pv_w","size"),median_observed_gain_m2=("effective_gain_m2_observed","median"),
        mean_pv_w=("pv_w","mean"),mean_ghi_w_m2=("ghi","mean"),mean_temp_c=("temp_c","mean")).reset_index()
    gain_map.to_csv(OUT_MAP,index=False,encoding="utf-8-sig")
    by_day=[]
    for day,g in eligible.groupby("day"):
        by_day.append({"day":int(day),**score(g.pv_w,prediction(g,selected["parameters"],chosen))})
    heldout_d6_metrics={**score(heldout_d6.pv_w,prediction(heldout_d6,selected["parameters"],chosen)),"day":6,"used_in_fit":False}
    ratio_gain=selected["parameters"]["gain_m2"]
    # Datasheet per-cell metrics and approximate area are kept separate.
    area_m2=CELL_COUNT_PROVISIONAL*CELL_AREA_M2
    json_out={
      "target":"Solar_Watt[W] ZP channel; direct HV-bus sum after MPPT outputs, per team statement",
      "weather_driver":weather_name,"weather_resolution":"local hourly means; upstream Open-Meteo route samples were spatial/time interpolations, not colocated pyranometer observations",
      "eligible_hour_bins":len(eligible),"eligible_days":[int(x) for x in sorted(eligible.day.unique())],
      "fit_posture":"moving only: event_phase=road_day and speed >=5 km/h; at least 80% of hourly source rows must meet this condition",
      "posture_note":"Moving panel posture is fixed relative to the body. The user confirms no panel tilt was used at BWSC2025 Control Stops; posture changes may occur at night or emergency stops. Parked/charging rows are excluded from the moving fit.",
      "day6_heldout_hour_bins":len(heldout_d6),"day6_heldout_metrics":heldout_d6_metrics,
      "day6_telemetry_anomaly_hour_bins_in_full_log":int(((hourly.day==6)&hourly.event_affected).sum()),
      "known_event_hours_excluded_from_fit":int(hourly.event_affected.sum()),
      "panel_design_basis":{"collector_area_user_approx_m2":6.0,"cell_count_provisional":CELL_COUNT_PROVISIONAL,
        "cell_area_datasheet_m2":CELL_AREA_M2,"calculated_cell_area_total_m2":area_m2,
        "cell_pmax_range_w":[3.72,3.78],"cell_efficiency_range_pct":[24.0,24.4],
        "gross_array_stc_range_w_before_encapsulation_mppt_mismatch":[CELL_COUNT_PROVISIONAL*3.72,P_STC_UPPER_W],
        "effective_gain_upper_m2_from_gross_stc_max":GAIN_UPPER,
        "count_note":"386 is a provisional area-based assumption, not the count from the layout image already supplied by the user. Replace it with the transcribed image count before using cell-count-based STC totals or area-normalized efficiency."},
      "model_candidates":models,"leave_one_race_day_out":folds,"lodo_mean_mae_w":means,
      "poa_candidate_fold_angle_diagnostic":{"stable_enough_for_selection":poa_angle_stable,
        "fold_tilt_range_deg":[float(min(fold_tilts)),float(max(fold_tilts))] if fold_tilts else None,
        "fold_heading_offset_circular_resultant":az_resultant,
        "selection_rule":"Only select fitted fixed plane when day-blocked MAE improves by >=2%, fold tilts span <=10 degrees, and heading-offset circular resultant >=0.8."},
      "faceted_body_candidate_diagnostic":{"stable_enough_for_selection":facet_stable,
        "candidate_lodo_mean_mae_w":means.get("faceted_body_poa"),
        "fold_mean_facet_gains_m2":{name:float(value) for name,value in zip(FACET_NAMES,facet_means)} if len(facet_means) else None,
        "fold_facet_gain_cv":{name:float(value) for name,value in zip(FACET_NAMES,facet_cv)} if len(facet_cv) else None,
        "selection_rule":"Only select the roof+four-vertical-facet hypothesis if day-blocked MAE improves by >=5%, every facet with mean gain >0.05 m2 has CV <=0.75 across folds, and total-gain CV <=0.25."},
      "selected_model":chosen,"selected_fit_metrics":fit_metrics,"selected_day_metrics":by_day,
      "selected_day6_heldout_metrics":heldout_d6_metrics,"residual_correlations_spearman_descriptive":residual_correlations,
      "effective_gain_m2":ratio_gain,"area_equivalent_operating_efficiency_pct":100*ratio_gain/area_m2,
      "thermal_interpretation":"No module temperature is measured. The selected fit uses ambient temperature only if it improves blocked-by-day error by 2%; any fitted gamma is an empirical proxy and must not be presented as a cell/module temperature coefficient.",
      "limitations":["The cell sheet supplies cell STC values, not a certified encapsulated array/module nameplate.",
       "Collector area and cell count are not yet verified from an original-resolution electrical layout.",
       "Panel orientation is fixed in motion but its normal/curvature relative to the vehicle body is not measured; Open-Meteo GHI therefore remains a proxy for plane-of-array irradiance.",
       "No module temperature sensor. Ambient temperature cannot identify the actual cell temperature or temperature coefficient.",
       "HV ZP solar power has MPPT aggregation, cable and conversion effects; observed effective gain includes orientation, shading, mismatch, temperature and DC conversion.",
       "Hourly means and day-blocked validation avoid 5-second pseudoreplication, but only a small number of daylight bins remain; coefficients are provisional.",
       "Day 2 MPPT crimp incident and Day 5 tire/hatch incident windows are excluded from the healthy PV fit; original diagnostic table preserves all hourly records."]}
    OUT_MODEL.write_text(json.dumps(json_out,indent=2,ensure_ascii=False),encoding="utf-8")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    good=fit_frame.copy()
    model_values=prediction(good,selected["parameters"],chosen)
    axes[0].scatter(model_values,good.pv_w,c=good.temp_c,cmap="coolwarm",s=24,alpha=.8)
    lo=float(min(np.nanmin(model_values),good.pv_w.min())); hi=float(max(np.nanmax(model_values),good.pv_w.max()))
    axes[0].plot([lo,hi],[lo,hi],"k--",lw=1)
    axes[0].set(xlabel="Modelled ZP Solar_Watt (W)",ylabel="Measured ZP Solar_Watt (W)",title=f"PV MLE: {chosen}; n={len(fit_frame)} hourly bins")
    for day,g in hourly[hourly.fit_eligible].groupby("day"):
        axes[1].scatter(np.full(len(g),day),g.residual_measured_minus_predicted_w,s=25,alpha=.8)
    axes[1].axhline(0,color="black",lw=1); axes[1].set(xlabel="Race day",ylabel="Measured − model (W)",title="Day-blocked residual diagnostic")
    fig.tight_layout(); fig.savefig(OUT_FIG,dpi=170); plt.close(fig)
    lodo_mae=means[chosen]
    gam=selected["parameters"].get("gamma_per_c")
    poa_note=(f" The fixed-plane POA fit estimated a {selected['parameters'].get('tilt_deg',float('nan')):.1f}° tilt and {selected['parameters'].get('heading_offset_deg',float('nan')):.1f}° offset from route heading; this is an empirical orientation proxy, not a verified body normal." if chosen=="fixed_body_plane_poa" else "")
    model_eq=("P = Σ facet_gainᵢ × POAᵢ (roof plus front/rear/left/right vertical facets)" if chosen=="faceted_body_poa" else "P = gain × isotropic POA(DNI, DHI, GHI; fitted fixed plane)" if chosen=="fixed_body_plane_poa" else "P = gain × GHI_proxy" if chosen=="ghi_through_origin" else "P = gain × GHI_proxy × [1 + gamma × (Tair − 25°C)]")
    text=f"""# BWSC2025 PV power model, initial physically bounded MLE\n\nThe fit uses **{len(eligible)} local-hour bins** from days {', '.join(map(str, sorted(eligible.day.unique())))}; ZP Solar_Watt is aggregated per hour and matched to hourly Open-Meteo irradiance. The reported 15-minute Australian weather series is interpolated from hourly model data, so it is not treated as an additional independent observation.\n\n## Result\n\n- Selected model: `{chosen}`.\n- Effective irradiance gain: **{ratio_gain:.3f} m²-equivalent**, corresponding to **{100*ratio_gain/area_m2:.1f}%** when divided by the provisional {area_m2:.3f} m² cell area. This is a system-level ratio to the modeled irradiance proxy, not cell efficiency.\n- Training MAE/RMSE: **{fit_metrics['mae_w']:.1f}/{fit_metrics['rmse_w']:.1f} W**; blocked leave-one-day-out mean MAE: **{lodo_mae:.1f} W**.\n- MLE Student-t residual scale: **{selected['scale_w']:.1f} W** (df={DF:g}).\n- Ambient-temperature proxy gamma: **{gam if gam is not None else 'not selected'}** per °C. It is not a module temperature coefficient.\n{poa_note}\nThe model is `{model_eq}`. There is no free positive power intercept, so predicted power is zero at zero irradiance. Gain is constrained below the upper gross cell STC area-equivalent **{GAIN_UPPER:.3f} m²**. The fitted {ratio_gain:.3f} m²-effective gain is consistent with combined losses/orientation but does not separate them.\n\n## Solar array design prior\n\nThe supplied SunPower E66Me3 cell data give 3.72–3.78 W per 125 mm cell at STC, 24.0–24.4% cell efficiency, with the stated active cell area around 0.01550 m². At a provisional 386 cells, cell-area sum is {area_m2:.3f} m² and gross cell Pmax is {386*3.72:.0f}–{P_STC_UPPER_W:.1f} W before encapsulation, wiring, shading, mismatch or MPPT loss. 386 is a provisional count inferred from the near-6 m² array and the BWSC 2027 rule's example, not a verified count from a high-resolution layout.\n\n## Residual diagnosis and model scope\n\nThe diagnostic table retains every hourly record; the healthy fit excludes known MPPT-crimp and Day 5 tire/hatch events, plus the Day 6 ZP-anomaly period. Day 5 hatch flapping/detachment and repair can change the active illuminated surface and aerodynamic condition, so a single steady PV gain should not absorb those periods. The PV power channel is the user's confirmed sum of MPPT outputs at the HV bus.\n\nOpen-Meteo GHI is a model-grid proxy, not an on-car irradiance reading. The roof array is fixed relative to the car in motion, but its plane normals, curvature/segments, and electrical sub-array geometry are not yet available at model resolution. No module temperature sensor exists. Thus we cannot yet fit a defensible panel incidence-angle or cell-temperature efficiency map. A temperature term is retained only if its day-held-out MAE improves by at least 2%; otherwise the simpler GHI map is used. The bounded model is the first step requested, not a validated PV efficiency map.\n\nOutputs: `pv_model_cv_by_day.csv`, `pv_model_hourly_diagnostic.csv`, `pv_effective_gain_map.csv`, and `pv_model_diagnostics.png`.\n"""
    text=text.replace("## Result", f"Only moving `road_day` hours are fitted (at least 80% of source rows at ≥5 km/h); parked/charging hours remain in the diagnostic CSV but stay outside the fit because their panel pose may change. Day 6 is a full held-out check ({heldout_d6_metrics['n_hour_bins']} moving bins); MAE {heldout_d6_metrics['mae_w']:.1f} W and bias {heldout_d6_metrics['bias_measured_minus_model_w']:+.1f} W.\n\n## Result",1)
    text=text.replace("Day 5 tire/hatch events, plus the Day 6 ZP-anomaly period", "Day 5 tire/hatch events; Day 6 is kept entirely outside fitting and used as a held-out ZP-anomaly check")
    if "fixed_body_plane_poa" in models:
        poa_note=(f"The fitted fixed-plane POA candidate had LODO mean MAE {means['fixed_body_plane_poa']:.1f} W "
                  f"versus {means['ghi_through_origin']:.1f} W for through-origin GHI, but the fold tilt range was "
                  f"{min(fold_tilts):.1f}–{max(fold_tilts):.1f}° and heading-offset circular concentration was "
                  f"{az_resultant:.2f}. It was {'selected' if chosen=='fixed_body_plane_poa' else 'not selected'} "
                  "because a fitted orientation must be stable across held-out days as well as improve error.")
        text=text.replace("## Residual diagnosis and model scope", "## POA sensitivity check\n\n"+poa_note+"\n\n## Residual diagnosis and model scope")
    if "faceted_body_poa" in models:
        facet_note=(f"The roof+four-vertical-facet hypothesis had LODO mean MAE {means['faceted_body_poa']:.1f} W "
                    f"and fold stability={facet_stable}; mean effective gains (m²-equivalent) were "
                    +", ".join(f"{name} {value:.3f}" for name,value in zip(FACET_NAMES,facet_means))
                    +f". It was {'selected' if chosen=='faceted_body_poa' else 'not selected'} under the pre-set 5% CV and parameter-stability gate. "
                    "These canonical facets are a testable approximation, not measured panel normals.")
        text=text.replace("## Validation caveat", "## Multi-facet body sensitivity\n\n"+facet_note+"\n\n## Validation caveat",1)
    poor_days=[r["held_out_day"] for r in folds[chosen] if r.get("r2") is not None and r["r2"]<0]
    day_bias="; ".join(f"Day {r['day']}: {r['bias_measured_minus_model_w']:+.0f} W" for r in by_day)
    speed_rho=residual_correlations.get("Speed[kmh]",{}).get("spearman_rho")
    extra=(f"Clean-day fitted residual bias by day: {day_bias}. "
           f"The held-out fit has negative R² on days {', '.join(map(str,poor_days)) if poor_days else 'none'}; "
           f"Day 6's held-out R² is {heldout_d6_metrics['r2']:.2f}. "
           +(f"Residual vs logged vehicle speed has descriptive Spearman rho {speed_rho:+.2f}; because speed co-varies with time, route and irradiance and hourly rows are autocorrelated, this is not a causal temperature/cooling finding. " if speed_rho is not None else "")
           +"The MAE reduction alone therefore does not validate this map for MPC control.")
    text=text.replace("## Residual diagnosis and model scope", "## Validation caveat\n\n"+extra+"\n\n## Residual diagnosis and model scope",1)
    if "faceted_body_poa" in models:
        text=text.replace("## Validation caveat", "## Multi-facet body sensitivity\n\n"+facet_note+"\n\n## Validation caveat",1)
    OUT_REPORT.write_text(text,encoding="utf-8")
    print(json.dumps({"eligible_hour_bins":len(eligible),"selected_model":chosen,"gain_m2":ratio_gain,
       "gain_area_equivalent_efficiency_pct":100*ratio_gain/area_m2,"fit_metrics":fit_metrics,"lodo_mae_w":lodo_mae,
       "candidate_lodo_mae_w":means,"outputs":[str(OUT_MODEL),str(OUT_CV),str(OUT_HOURLY),str(OUT_MAP),str(OUT_REPORT),str(OUT_FIG)]},indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
