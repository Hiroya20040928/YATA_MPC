# BWSC2025 route reconstruction and source audit

## What is in the combined log

The six daily ZP exports contain 43,963 primary telemetry rows at a nominal 5-second interval. The exports append unrelated summary tables after the telemetry block, so the row-count metadata in each export is used to retain only the driving/charging log. Each retained raw row was checked against the corresponding cleaned log for all seven shared sensor channels; their maximum absolute difference is zero. Dates are assigned from the official 2025 road-day sequence (2025-08-24 through 2025-08-29), not from a bad PC/export timestamp in the final day's file. Timestamps are recorded in Australia/Darwin local time (ACST, UTC+09:30) and UTC.

One acquisition gap is present in Day 6: no rows from 09:04:25 to 10:02:30 (3,485 seconds), within the ZP anomaly stop. The vehicle is held at the anomaly anchor across this stopped interval, but no electrical sensor behavior can be reconstructed inside the missing rows. All other adjacent telemetry timestamps are five seconds apart.

`race_days_merged.csv` also preserves the per-day raw ancillary columns where present. The source export's `SolarW*5/3600` and `MO[W]=BA-SO` are source-derived fields, not independent sensors. The clean seven channels are kept as the consistent model inputs.

## Distance and coordinate method

No GPS track is present in the ZP telemetry. Between trusted route anchors, the recorded speed trace is trapezoid-integrated and rescaled to the distance delta between anchors. Equal-distance timed spans (control stops, charging, and documented repair stops) are held stationary. Position is then linearly indexed into the supplied route DEM/coordinate track. The method gives a plausible route position for every row but does not produce independent odometer/GPS truth. Each row keeps the segment, speed scale, anchor confidence, and anchor uncertainty.

The 40 used anchor intervals were inspected in `distance_anchor_fit.csv`. Most moving intervals require a modest scale near 1.0; the notable remaining moving interval is the Day 4 low-battery-stop departure to camp, where the route-track endpoints imply 35.2 km and the speed integral is 28.48 km (scale 1.236). That interval remains medium confidence because the annotated timeline and coordinate-track values differ. The largest discrepancies were handled by removing weak approximate incident points from the distance fit rather than forcing an extreme local correction.

## Ambiguities retained instead of silently resolving

- **Day 3 hatch/noise check:** memo distance about 1079 km at 08:15 conflicts with the ZP speed integral from the 1074.4 km camp. It remains an incident time, but is not a distance anchor.
- **Day 5 hatch incident:** notes state 2262 km at 11:51 and then 2228 km at 12:22, which reverses course. Neither number is used for distance. Later repair at about 2329 km is retained as a medium-confidence anchor.
- **Day 5 roadworks:** memo distance about 2430 km conflicts with speed-integrated travel to official CS8 at 2434 km. The signal event time is preserved; it does not pin route distance.
- **Day 6 ZP anomaly:** the timeline says 09:08, while the ZP log speed first reaches and remains at zero at 09:04:20. The distance stop anchor uses 09:04:20 with wider uncertainty; event notes preserve the disagreement.
- **Day 6 roadworks:** the approximate 2574 km note conflicts with speed-integrated movement toward the route-coordinate stop at 2583.8 km, so it is event-only.
- **Route reference versus named place:** official Control Stop references are not identical to the nearest town/camp point on the supplied route track (for example, Port Augusta CS9 reference about 2720.5 km versus the named-town point at 2718.5 km). Control Stops use their official route-reference distances; camps/towns use their route-track coordinate distances.

See `data/input/route/anchors.csv` and `events.csv` for the row-level source, confidence, and use/exclusion flags. These attached annotations document evidence; they do not supersede the user's instructions.

## Energy convention and first fit

The team's attached energy-balance note defines the red-line electrical consumption proxy as `Solar_Watt + Battery_Watt`; in its sign convention, battery power is positive for extraction and negative for charging. The user has confirmed that ZP Solar_Watt is measured directly on the HV bus after the MPPT outputs are combined. That proxy is added to the combined CSV as `net_electrical_power_proxy_w`. The note's day-energy totals are derived summaries, not raw truth; they were independently reconciled before fitting.

This comparison has now been done in `energy_balance_reconciliation.csv`: trapezoidal integration of the merged rows reproduces the attached Day 1–5 PV generation, battery extraction, charge, and load-proxy totals within 1.6 Wh per daily total (the largest component difference is 1.46 Wh). This independently checks row alignment, timestamps, power signs, and 5-second integration against the team's calculations; it does not calibrate the absolute sensor accuracy.

The first road-load calibration is now in `models/vehicle_power_model_fit.json` and `reports/vehicle_power_model_summary.md`. It uses 10 km distance segments and fit-effective coefficients; the route grade, mass, Crr/CdA and in-car motor/controller efficiency are not separately identifiable from the present signals.

## Information still needed

The SunPower cell datasheet and approximate 6 m² area are now available; the exact installed cell count/active area and each body-surface normal/MPPT string are not yet fixed. No module-temperature sensor exists. User confirms the ZP HV measurement point, but sensor accuracy and averaging time remain unknown. Confirm whether a PV-active hatch sub-array was electrically disconnected or shaded during the Day 5 hatch failure, and provide its event time/MPPT channel if recorded.
