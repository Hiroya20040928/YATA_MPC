# BWSC2025 vehicle energy model, initial MLE

The fitted target is the team-confirmed direct HV bus balance proxy, `Solar_Watt + Battery_Watt`. The sample is aggregated into **184 10 km segments**. The clean calibration uses Days 1–5 (162 segments); Day 6 (22 segments) is held out because its ZP anomaly was recorded.

## Fit and validation

- Stopped-bus median used for total auxiliary baseline: **67.6 W**. The separately provided 21 W weak-system draw is a component and is included within this baseline; it is not added again.
- Clean-day fit: MAE **44.9 W**, RMSE **57.5 W**, mean bias **+0.7 W**.
- Leave-one-clean-day-out mean MAE: **53.3 W**.
- Day 6 held-out MAE: **81.0 W**, bias **+74.6 W**.
- Fitted effective multipliers: rolling 0.918, aero 1.535, endpoint grade 0.741.

If the bench 1 kW ECO_LO efficiency were imposed, the multipliers would imply CdA 0.125 m², Crr 0.0030, and a grade-equivalent mass of only 214 kg. That last value is below the 234 kg dry vehicle mass and is physically impossible; it diagnoses route-grade/DEM, regeneration, telemetry, or model mismatch. These multipliers are **not true physical parameter estimates**. The grade coefficient differs materially from unity despite a 314 kg floor mass. Since mass, Crr, CdA, motor/controller efficiency, DEM grade, regen behavior, and bus metering are confounded in the available channels, the data do not support fitting each as an independent truth. Treat the coefficient as a correction to the joined route and energy model. The elevated Day 6 residual further means the model is an initial cruise predictor, not yet safe as an unguarded MPC command.

## Day 5 handling

The documented 10:31–10:37 tire change and 11:51–12:50 hatch-flap/detachment/repair window are excluded from the healthy-road-load fit. Day 5 records, events, and weather remain in the integrated source log. The tire change may alter rolling resistance; hatch loss/flapping can change drag and panel exposure, so neither interval should train a single steady-state model.

## Model evidence and limits

The road-load terms follow the standard longitudinal balance of rolling, aerodynamic, grade, and inertial forces. This early segment fit omits acceleration energy by restricting it to long, mostly steady 10 km bins. The vehicle uses the supplied theoretical CdA 0.08876 m² and Crr 0.0035 as priors, 234 kg dry vehicle plus the 80 kg regulatory occupant/ballast minimum as a 314 kg floor proxy, and the provided tire diameter. The ECO_LO bench curve is retained as a motor/controller prior, not equated to in-car PWM+ECO operation. The workbook reports 92.00% at 1 kW input; the installed serial and PWM+ECO controller-state equivalence are not confirmed. With the supplied 558.4 mm tyre and direct drive, bench motor rpm is converted to vehicle speed in `models/m2096_eco_lo_bench_map.csv`.

Resolution comparison (leave-one-clean-day-out) is in `vehicle_power_model_fit.json`; finer bins increase residual noise. The present 10 km segment model is appropriate only as an aggregate calibration/check, not yet as the MPC's short-horizon power map.

See `vehicle_power_cv_by_day.csv`, `vehicle_energy_segments_10km.csv`, and `vehicle_power_model_diagnostics.png` for row-level segment predictions and residuals.
