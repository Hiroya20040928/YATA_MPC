# BWSC2025 component-model iteration and full-race replay

## Scope and data checks

This replay covers **43,963 source records over Days 1–6**. Logged vehicle speed is supplied as an exogenous input at every 5-second timestamp. The result is a retrospective kinematic replay, not a closed-loop race simulation: it does not choose throttle, predict speed, or claim 5-second power accuracy.

- Duplicate `(day, timestamp_utc)` rows: 0.
- Dominant logged interval: 5.0 seconds.
- `Solar_Watt` vs `Solar[A] × Main_Volt`: median absolute difference 0.024 W; 99th percentile 0.050 W.
- `Battery_Watt` vs `Battery[A] × Main_Volt`: median absolute difference 0.024 W; 99th percentile 0.050 W.
- Zero main-voltage samples: 60 (excluded from battery voltage scoring).
- The user reports no calibration adjustment. That means no post-hoc correction was applied; it does not establish traceable absolute sensor accuracy.

The component balance uses `measured load = Solar_Watt + Battery_Watt`, with positive battery power meaning discharge. PV and vehicle-load predictions are produced separately; the battery power required by the model is their difference.

## Component order and identifiability

1. **PV array + MPPT**: isolated first because `Solar_Watt` is a dedicated aggregate HV output and weather/sun geometry can be used as external drivers. The fit is an effective system map; array, incidence angle, temperature, shading, mismatch and MPPT are not separately observed.
2. **Battery**: voltage and current are logged directly, and the supplied 8 A/10 A discharge curves plus 25-series SoC-voltage reference constrain a pack-level model. Capacity/voltage and effective resistance can be assessed; true SoC, battery temperature, charge efficiency and cell-level imbalance cannot be verified from these channels.
3. **Motor/controller and road load**: keep as effective road-load coefficients for now. The race log has no torque, motor phase current, throttle/PWM command or wheel force. Motor efficiency is therefore confounded with tire, aerodynamics, route grade, wind, regen and telemetry.

The supplied vehicle setup is used as already specified by the user: 234 kg dry car, 80 kg regulatory occupant/ballast floor proxy, 0.08876 m² theoretical CdA, 0.0035 Crr, 558.4 mm tyre, three wheels/direct drive, bonded fixed-in-motion array, and the supplied 6 m²/layout basis. No vehicle-data confirmation is requested again.

## PV + MPPT map: added physical thermal candidate

The training set has **23 moving hourly bins across Days 1–5**. PV power is matched at that weather resolution for fitting; the row replay then evaluates each logged time using the interpolated weather fields.

The existing faceted POA map has day-blocked mean MAE **50.5 W**. A generic Faiman-style cell-temperature estimate (`Tcell = Tamb + POA/(U0 + 6.84 × wind)`) was added with U0 profiled over 20, 25 and 30 W/m²°C. The best thermal candidate (U0=20) has day-blocked mean MAE **51.5 W**. The selection rule required at least a 2% improvement, stable gamma sign across held-out days, and no fold at its constraint bound. **Selected map: `faceted_body_poa`.**

The fitted temperature coefficient is not a certified SunPower cell coefficient. The provided cell sheet gives STC electrical values but no installed module temperature coefficient. No module temperature sensor exists, and the roof-bonded geometry differs from generic open-rack thermal assumptions. The thermal candidate and fold coefficients are saved in `pv_temperature_model_cv.csv`; only a stable day-blocked improvement is retained.

## Battery map and voltage fit

The supplied 8 A and 10 A curves report **34.597 Ah** and **33.164 Ah**. Integrating voltage over discharged Ah and scaling the 2S6P test to 25S6P gives **3168.7 Wh** at 8 A and **3013.0 Wh** at 10 A. The 10 A result is only **+0.07%** from the user-stated 3,011 Wh; its 33.164 Ah capacity is therefore the selected design/test prior. The voltage-only fit preferred 34 Ah by 0.045 V MAE, less than the logged Main_Volt 0.1 V step, so that small gain does not outweigh the direct discharge-energy evidence. The independent 5%-step 25-series voltage/SoC curve is exported without smoothing.

The lab differential-resistance extraction was also corrected in this iteration: the source SoC axis descends from 100% to 0%, but `np.interp` requires ascending input. After sorting, **19 of 21** 5%-step points yield a positive 8 A-to-10 A pack-resistance proxy (valid 5–95% SoC); the 0% and 100% endpoints are excluded because the two measured curves reverse order there. This is still a rate-differential proxy, not pulse-derived ohmic resistance or an in-car R(SOC,T) truth map.

An SOC-dependent candidate was then tested using that corrected lab curve with one fitted scale per training fold. Clean-day (Days 1–5) MAE falls from **1.029 V** for constant R to **0.672 V**; fold scales are **0.598–0.699**. However, on the separate Day 6 holdout its MAE is **1.874 V**, versus **0.712 V** for constant R. The candidate is therefore **not used** in the full-race replay yet. Adding a training-derived voltage offset improves clean-day bias but worsens Day 6 and cannot be called sensor calibration.

With capacity anchored to the direct test, day-blocked validation uses **33.164 Ah** and estimates a full clean-day effective pack resistance of **0.488 Ω**. Leave-one-day-out moving-discharge voltage errors: MAE **1.03 V**, RMSE **1.23 V**, mean bias **+0.90 V** (1-minute medians). The 34 Ah unconstrained profile point remains in `component_replay_fit.json` to show the voltage-only fit's small preference; the capacity is no longer allowed to drift away from the 3,011 Wh evidence by less than one bus-voltage count. Fold errors are in `battery_voltage_model_cv.csv`.

Each day is initialized from its first robust 3-minute voltage/current anchor, then logged current is integrated over actual UTC elapsed seconds. No missing overnight charging is fabricated. This voltage fit does **not** validate absolute SoC: there is no in-car SoC ground truth. Charge efficiency is set to 1.0 as an explicit placeholder; voltage scoring excludes charge samples because the supplied lab work has no charge curve. A Day 3 start contains a 79.7 V one-sample transient followed by about 89.6 V; the robust anchor suppresses that startup point.

## Whole-race replay and residuals

`full_race_component_replay_5s.csv` contains a prediction and residual for every original record. Days 1–5 use road coefficients fit without that day; Day 6 is held out from all clean-day fitting. PV training is likewise day-blocked. The vehicle model remains the fitted 10 km MLE with logged speed forced; local residuals include omitted acceleration and route/weather error.

The 10 km cross-day replay contains **139 segments** on Days 1–5 with mean absolute energy error **8.1 Wh** (7.3% of mean measured segment energy) when available. Use `full_race_component_replay_10km.csv` for segment-level review; row-level residuals are in the full 5-second file.

### Daily metrics

| Day | Road-load OOF MAE (W) | PV 5 s weather-forced MAE (W) | Battery voltage 1 min MAE (V) | Notes |
|---:|---:|---:|---:|---|
| 1 | 261.3 | 104.4 | 0.90 | day-blocked |
| 2 | 182.6 | 68.4 | 1.02 | day-blocked |
| 3 | 161.7 | 58.2 | 0.78 | day-blocked |
| 4 | 234.7 | 61.7 | 0.60 | day-blocked |
| 5 | 277.9 | 130.9 | 0.91 | day-blocked |
| 6 | 296.3 | 248.2 | 1.12 | held out; race ended early/anomaly |

## MPC-use gate

At the median eligible cruise speed (**70.0 km/h**), the nominal physical road-load change between adjacent ±2.5 km/h choices is about **95.2 W** using the supplied Crr/CdA, 314 kg minimum-mass proxy and observed median air density. The internal decision-resolution gate is therefore an out-of-day model MAE below **23.8 W** (25% of that action difference), so error does not dominate a speed choice. This is an engineering acceptance rule for this MPC decision, not a BWSC requirement or external standard.

Current 10 km leave-one-day-out road-load MAE on Days 1–5 is **53.3 W**. On the distinct 5-second row grain, Day 6 held out from clean-day fitting has MAE **296.3 W** and bias **+79.1 W**; these two grains are not directly comparable. **The model does not pass the decision-resolution gate yet.** The full-race playback supports diagnosis and route-level energy budgeting with caveats, but it is not an unguarded MPC speed command model.

## Additional road-model iteration (2026-10-01)

Four segment-level alternatives were evaluated using the same Student-t likelihood and Days 1–5 leave-one-day-out folds. The reference model's mean fold MAE is **53.35 W** (pooled held-out MAE **51.31 W**). Replacing endpoint-elevation grade work with integrated DEM-grade work reduced mean fold MAE to **52.71 W** (1.2%), but worsened the separate Day 6 MAE from **81.00 W** to **83.03 W**, so the original grade formulation remains selected. Adding a speed-proportional loss reduced mean fold MAE only to **52.95 W** (0.8%); its fitted coefficient reached zero because speed is already nearly collinear with the rolling term. Adding a quadratic speed term worsened mean fold MAE to **56.87 W**. None passes the **23.8 W** decision-resolution gate, and none is adopted.

The candidate fold scores and fitted terms are saved in `vehicle_model_candidate_cv.csv` and `vehicle_model_candidate_cv.json`. This iteration confirms that a simple extra speed-loss term does not explain the day-to-day residual. The remaining high-impact explanation is still an unmeasured day-dependent factor (actual all-up mass/tire state, route-grade error, motor/controller efficiency state, or bus-meter bias); these terms cannot be separated from the current logs alone.

The row-level residual was also separated into a speed-derived steady-cruise proxy (`|smoothed acceleration| <= 0.075 m/s²`) and all other eligible moving rows. Across **16,242** eligible records, MAE is **231.7 W** overall and **201.2 W** for the **14,863** steady-proxy records. The reduction is not enough for pointwise control accuracy, and the acceleration estimate itself comes from quantized speed rather than an accelerometer. A timestamp-unit audit found that Pandas 3 can expose parsed timestamps at microsecond resolution; the lag check has now been recomputed using `timedelta.total_seconds()`. The strongest residual–inertial-power correlation within ±30 seconds occurs at **0 s on Days 1 and 5, −5 s on Days 2, 3 and 6, and −10 s on Day 4**. This day-to-day variation supports an association with transient operation but does not identify one common logger offset or prove a causal mechanical coefficient. The corrected correlations are in `vehicle_transient_residual_diagnostics.json`.

On eligible Day 6 moving rows, **15** samples have `Battery_Watt < -1,000 W`; the minimum is **−3,484 W** at a logged speed of 65 km/h and current of −26.84 A (median among those samples). Abrupt speed drops occur in this region. These values could include regenerative braking and/or the documented ZP anomaly; the existing channels cannot distinguish them. See `vehicle_transient_residual_diagnostics.csv` and its JSON summary. Do not fit these transients into a steady-cruise efficiency map until the event and signal semantics are confirmed.

## What the residuals imply

- **PV residuals**: hourly training has only 23 bins; sample-level weather values are spatial/time-interpolated weather-model estimates. A remaining mismatch cannot be attributed uniquely to cell temperature, incidence angle, passing cloud, partial shading, array mismatch or MPPT loss. Parked/tilted periods are outside the moving map.
- **Battery residuals**: the 1-minute voltage error tests the empirical SoC curve plus pack R; it cannot determine cell-level R, temperature response, capacity fade, charge efficiency or true SoC. No logging calibration means sensor scale remains unverified.
- **Vehicle residuals**: the 10 km fit's effective coefficients combine the stated physical priors with actual motor/controller operation, tire state, interpolated wind, route elevation, regeneration and logger error. Row-wise error additionally reflects acceleration omitted by the segment fit. Day 6's positive residual/bias is retained as a held-out failure signal, not absorbed into a refit.
- **Acceleration candidate, tested and rejected**: adding positive/negative inertial-power terms reduced 1-minute day-LODO MAE from 137.8 W to 125.5 W (8.9%), but MAE remained high. The positive-inertia coefficient varied from 0.70 to 0.83, below the expected electrical conversion factor for the 314 kg minimum-mass basis. At 5-second grain the descriptive day-LODO MAE changed from 179.2 W to 177.7 W, so there is no persuasive pointwise gain. Keep the simpler 10 km model as the route-energy reference; do not use this acceleration candidate for control.
- **Motor stage**: the manual separates current/PWM command mode from the Pow/Eco switch. Bench curves are labelled PWM_LO/PWM_HI/ECO_LO/ECO_HI and are single operating sweeps; they do not identify an in-car PWM+ECO efficiency surface. The supplied ECO_LO sheet records DUTY=1.0 at 16 points, spanning **70.92–83.26 km/h**, **72–3,089 W** electrical input, and **0–40.7 N·m**. This only partially overlaps the 10 km model's 57.6–78.2 km/h training support and sweeps load at a fixed duty; it is not a 2D speed/torque map for the race's PWM+ECO state. The comparison is plotted in `motor_bench_vs_race.png` and tabulated in `motor_bench_road_comparison.csv`. Keep ECO_LO as a prior/reference only until its command and controller state are matched to the installed setting.

## Current disposition

PV/MPPT is the first fitted component, but a true temperature-efficiency map is **not identified** by these logs. Battery voltage and capacity were advanced with lab priors and day-blocked validation; the resulting SoC remains an estimate. The combined model is **not ready for autonomous MPC speed selection** under the stated decision-resolution criterion. The errors are too large relative to the energy difference between nearby speed choices, and absolute sensor calibration/SoC is unverified.

To close the remaining high-impact gaps: collect a short synchronized array-plane irradiance and backsheet/module-temperature series; battery pack temperature and a rest/pulse sequence at known SoC; and motor/controller input/output or torque-speed data at the actual PWM-command + ECO-switch setting. These are the smallest new measurements that separate the current confounded terms.

## Source basis

- User-supplied SunPower E66Me3 cell sheet and 8 A/10 A battery discharge materials (source copies under `data/input/design/`).
- NREL PVWatts V5 uses plane-of-array irradiance, wind and ambient temperature in a transient cell-temperature model; its thermal assumptions are generic and are not a vehicle-panel validation: [PVWatts V5 Manual](https://pvwatts.nrel.gov/downloads/pvwattsv5.pdf).
- Faiman's module-temperature model is the basis for the thermal sensitivity candidate: [Faiman (2008), DOI 10.1002/pip.813](https://doi.org/10.1002/pip.813).
- Battery equivalent-circuit parameter identification and SOC methods rely on voltage/current plus appropriate pulse/rest characterization; this race log has voltage/current but no in-car true SoC or temperature: [Review of battery SOC estimation based on equivalent circuit models (2025)](https://doi.org/10.1016/j.est.2025.115346), [parameter-estimation methods benchmark (2023)](https://doi.org/10.1016/j.est.2023.108197).
- Road-load component separation normally needs controlled coastdown/road-load or independent tire/aero measurements; SAE J2263 describes coastdown with onboard anemometry under reference conditions: [SAE J2263_202005](https://doi.org/10.4271/J2263_202005).

Artifacts: `component_replay_fit.json`, `component_replay_by_day.csv`, `component_replay_diagnostics.png`, `full_race_component_replay_5s.csv`, `full_race_component_replay_10km.csv`, `battery_voltage_soc_map.csv`, `battery_effective_resistance_map.csv`, `vehicle_model_candidate_cv.csv`, and `vehicle_transient_residual_diagnostics.csv`. Open `additional_data_request.md` for the prioritized list of missing information needed to continue fitting.

## Residual-learning iteration (2026-10-01)

The existing vehicle model was re-fitted by held-out day for this analysis. Each fold recalculates the stopped-load auxiliary median using only training days, then predicts the held-out day. The eligible moving rows were aggregated into 5-minute UTC blocks with at least 20 eligible 5-second records per block. Days 1–5 supplied 233 blocks; Day 6 supplied 44 anomaly-stress blocks and was never used to choose the correction.

Four causal candidates were compared: previous-block residual, rolling 3-block median, rolling 6-block median, and a small Ridge residual model using speed, relative air speed, route grade, ambient temperature, and density. Each correction is limited to ±100 W and resets after a gap longer than 10 minutes. The outer leave-one-day-out fold selected the rolling 6-block median on four of five days and the 3-block median on Day 4. The day-selected nested result reduces pooled 5-minute MAE from **70.2 W to 57.1 W (18.7%)**, with a warm-state paired improvement on **4/5 days**; Day 4 worsens. The Ridge candidate was not selected in any outer fold.

This is evidence for a bounded same-day bias correction after warm-up, not for changing or interpreting the physical road-load coefficients. The gain is evaluated at 5-minute mean power and does not demonstrate 5-second power accuracy or transfer to the different BWSC2027 route. The corrected 57.1 W remains above the provisional 23.8 W speed-decision gate. The candidate is stored for shadow evaluation only and is not connected to the existing race replay, Grafana, or vehicle commands.

Applying the selected correction to Day 6 as a stress check reduces its MAE from **108.6 W to 85.9 W**. Because this can absorb a persistent anomaly into the correction, anomaly and sensor-quality flags must freeze adaptation; the lower Day 6 error is not evidence of normal-operation accuracy.

Applying the same causal correction to the next block's 5-second rows reduced clean-day row MAE only from **217.9 W to 213.0 W (2.2%)**; this remains far above the 23.8 W action-resolution gate. The 18.7% gain applies to 5-minute mean power only, so the correction is not supported for 5-second throttle/speed control.

Clean-day 5-second absolute-error P90 moves only from **503.9 W to 496.3 W**; on Day 6 the 5-second P90 rises from **641.9 W to 647.1 W**, even as MAE falls. Thus this adaptation reduces persistent bias but does not remove the high-error tail, especially under anomalous conditions.

Detailed fold metrics and outputs: `reports/residual_learning_report.md`, `reports/residual_online_cv.csv`, `reports/residual_block_oof.csv`, `reports/residual_online_summary.json`, and `models/vehicle_residual_online.json`. Reproducible evaluation: `scripts/evaluate_vehicle_residual_learning.py`.

## Additional residual-learning iteration (2026-10-01)

The same-day correction family was extended to rolling medians over 3/6/9/12 blocks and EWMA updates. The nested day selector still most often chose the 6-block median; higher complexity did not displace it reliably. Cross-day correction was then evaluated using leakage-safe residual labels (the outer test day and each label day were excluded from the corresponding base fit). Ridge models based on speed, reconstructed relative wind, grade, temperature and density worsened held-out MAE. Fixed RBF Gaussian processes improved pooled MAE by 6%, but their predictive bands remained broad (245–364 W mean width for nominal 90% coverage).

A fixed GP plus the causal median of six prior same-day GP innovations achieved 56.65 W 5-minute MAE, 129.54 W P90 and a win over the physical baseline on all five clean day folds. This is only 0.42 W better than the simpler same-day median6 rule. The GP combination was selected after comparing several alternatives on those same folds, so the estimate is selection-biased; the 0.42 W increment does not justify the added model complexity. At 5-second row grain its MAE remains 212.24 W, and Day 6 P90 worsens from 641.91 to 649.28 W. The candidate is recorded as research-only and is not integrated into race replay, Grafana or MPC commands.

The remaining limit is data identifiability, not a lack of possible algorithms: Day1–5 cover largely disjoint route-distance bands, and the available race log has no independent panel temperature/POA sensor, known in-car battery SoC/temperature, motor torque/current feedback, or traceable absolute meter calibration. Flexible residual learners would use the same five folds for tuning and evaluation. Methodological precedent includes hybrid EV energy models and uncertainty-aware learning MPC, but does not establish YATA-specific gains: [Huang et al. (2024)](https://arxiv.org/abs/2405.10984), [Kabzan et al. (2019)](https://doi.org/10.3929/ethz-b-000351561), [Costa et al. (2023)](https://doi.org/10.1016/j.robot.2023.104469). Detailed results and cross-fitted outputs are in `reports/residual_crossday_report.md` and `reports/residual_gp_online_hybrid_report.md`.
