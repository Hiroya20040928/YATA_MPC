# BWSC2025 PV power model, initial physically bounded MLE

The fit uses **23 local-hour bins** from days 1, 2, 3, 4, 5; ZP Solar_Watt is aggregated per hour and matched to hourly Open-Meteo irradiance. The reported 15-minute Australian weather series is interpolated from hourly model data, so it is not treated as an additional independent observation.

Only moving `road_day` hours are fitted (at least 80% of source rows at ≥5 km/h); parked/charging hours remain in the diagnostic CSV but stay outside the fit because their panel pose may change. Day 6 is a full held-out check (2 moving bins); MAE 112.9 W and bias -112.9 W.

## Result

- Selected model: `faceted_body_poa`.
- Effective irradiance gain: **1.068 m²-equivalent**, corresponding to **17.8%** when divided by the provisional 5.983 m² cell area. This is a system-level ratio to the modeled irradiance proxy, not cell efficiency.
- Training MAE/RMSE: **45.9/65.0 W**; blocked leave-one-day-out mean MAE: **50.5 W**.
- MLE Student-t residual scale: **45.2 W** (df=4).
- Ambient-temperature proxy gamma: **not selected** per °C. It is not a module temperature coefficient.

The model is `P = Σ facet_gainᵢ × POAᵢ (roof plus front/rear/left/right vertical facets)`. There is no free positive power intercept, so predicted power is zero at zero irradiance. Gain is constrained below the upper gross cell STC area-equivalent **1.459 m²**. The fitted 1.068 m²-effective gain is consistent with combined losses/orientation but does not separate them.

## Solar array design prior

The supplied SunPower E66Me3 cell data give 3.72–3.78 W per 125 mm cell at STC, 24.0–24.4% cell efficiency, with the stated active cell area around 0.01550 m². At a provisional 386 cells, cell-area sum is 5.983 m² and gross cell Pmax is 1436–1459.1 W before encapsulation, wiring, shading, mismatch or MPPT loss. 386 is a provisional area-based assumption; the layout image is already supplied but its count has not yet been transcribed into the model ledger. Do not treat count-based totals or area-normalized efficiency as final until that is done.

## POA sensitivity check

The fitted fixed-plane POA candidate had LODO mean MAE 53.5 W versus 104.8 W for through-origin GHI, but the fold tilt range was 3.0–9.6° and heading-offset circular concentration was 0.98. It was not selected because a fitted orientation must be stable across held-out days as well as improve error.

## Multi-facet body sensitivity

The roof+four-vertical-facet hypothesis had LODO mean MAE 50.5 W and fold stability=True; mean effective gains (m²-equivalent) were roof 1.011, front 0.000, rear 0.047, left_side 0.000, right_side 0.011. It was selected under the pre-set 5% CV and parameter-stability gate. These canonical facets are a testable approximation, not measured panel normals.

## Validation caveat

Clean-day fitted residual bias by day: Day 1: -93 W; Day 2: -7 W; Day 3: +5 W; Day 4: +9 W; Day 5: -10 W. The held-out fit has negative R² on days none; Day 6's held-out R² is -12.33. Residual vs logged vehicle speed has descriptive Spearman rho +0.11; because speed co-varies with time, route and irradiance and hourly rows are autocorrelated, this is not a causal temperature/cooling finding. The MAE reduction alone therefore does not validate this map for MPC control.

## Residual diagnosis and model scope

The diagnostic table retains every hourly record; the healthy fit excludes known MPPT-crimp and Day 5 tire/hatch events; Day 6 is kept entirely outside fitting and used as a held-out ZP-anomaly check. Day 5 hatch flapping/detachment and repair can change the active illuminated surface and aerodynamic condition, so a single steady PV gain should not absorb those periods. The PV power channel is the user's confirmed sum of MPPT outputs at the HV bus.

Open-Meteo GHI is a model-grid proxy, not an on-car irradiance reading. The roof array is fixed relative to the car in motion, but its plane normals, curvature/segments, and electrical sub-array geometry are not yet available at model resolution. No module temperature sensor exists. Thus we cannot yet fit a defensible panel incidence-angle or cell-temperature efficiency map. A temperature term is retained only if its day-held-out MAE improves by at least 2%; otherwise the simpler GHI map is used. The bounded model is the first step requested, not a validated PV efficiency map.

Outputs: `pv_model_cv_by_day.csv`, `pv_model_hourly_diagnostic.csv`, `pv_effective_gain_map.csv`, and `pv_model_diagnostics.png`.
