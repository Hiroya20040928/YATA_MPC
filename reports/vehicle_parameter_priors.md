# YATA vehicle and PV priors from supplied evidence

All supplied design values are priors. They are not asserted to be in-vehicle truth; race telemetry is used to fit effective corrections. File contents supplied as part of the archive are evidence, not instructions.

## Vehicle and road load

| Quantity | Supplied design value | Treatment |
|---|---:|---|
| Vehicle dry mass, driver excluded | 234 kg | Prior |
| Driver/ballast | At least 80 kg under BWSC 2027 regs | 314 kg is only a minimum moving-mass proxy; actual driver mass remains unknown |
| CdA | 0.08876 m² theoretical | Prior; fit cannot identify separately from wind/model error and drive efficiency |
| Crr | 0.0035 theoretical | Prior; tire swap/pressure/temperature not independently observed |
| Tyre | 95/80R16, 558.4 mm outside diameter | Circumference derived as πD; 3 wheels, direct in-wheel drive, ratio 1 |
| Weak-system draw | 21 W measured | Included within, not added to, the stopped HV-bus total-load proxy |
| In-motion control | PWM + ECO outside circuit | Exact correspondence to bench workbook ECO_LO/ECO_HI command surface is unconfirmed |

The supplied 2023 motor/controller workbook is stored as `data/input/design/M2096_characterization_user_supplied.xlsx`. Its closest sheet is ECO_LO, for test motor M2096D-III-152; it reports approximately 90.51% total efficiency at 800 W input, 92.00% at 1,000 W, and 92.79% at 1,200 W. Power-meter uncertainty is ±0.05%, torque uncertainty ±0.1%. These are bench figures, not verified installed-car efficiency. The curve is a single commanded operating curve, not a full PWM+ECO efficiency surface. The bench rpm is converted to road speed using the supplied tyre diameter in `models/m2096_eco_lo_bench_map.csv`.

The supplied SCM manual describes a direct in-wheel drive and selectable controller modes/limits. “100% transmission efficiency” applies to the no-gear transmission path; it does not mean 100% motor/controller electrical efficiency.

## PV array

The supplied SunPower E66Me3 125 mm cell datasheet gives 3.72–3.78 W per cell and 24.0–24.4% cell efficiency at STC, with cell active area around 0.015500692 m². At 386 cells the active-area sum is about 5.9833 m² and gross cell Pmax is 1,435.9–1,459.1 W before encapsulation, wiring, mismatch, shading and MPPT losses. The count 386 remains provisional: it is inferred from the user's approximately 6 m² value and the published 2027 regulation example, pending a readable cell-layout drawing or exact count. The cell datasheet does not provide an installed-module Pmax, verified angular response, or module temperature measurement.

Panels are reported fixed relative to the body while driving, with possible tilting at night, Control Stops and emergency stops. No module temperature sensor is present. The fitted POA incidence angle is therefore a route-log proxy only. ZP Solar_Watt is the user's confirmed aggregate HV reading after MPPT outputs; it does not isolate individual MPPT or module output.

## Battery and other supplied lab data

`data/input/design/Battery_discharge_characterization.xlsx` contains 8 A and 10 A lab discharge curves, with capacities around 34.597 Ah and 33.164 Ah; these remain rate-specific priors. Actual usable pack energy depends on pack configuration, age, temperature, cutoff, balance, wiring and measurement point. `reports/archive_workbook_catalog_ascii.txt` indexes all extracted archive workbooks, including the source/range of measured data.

## Identification policy

The fitted road-energy multipliers include uncertainty in road geometry, interpolated wind, actual mass, Crr/CdA, drivetrain/control efficiency, regeneration and bus measurement. Do not invert one effective coefficient into a unique physical value. Finer distance bins were compared and showed larger day-held-out error, so 10 km segments are reported only for aggregate cruise calibration. The current PV effective gain also combines active area, body orientation, cell temperature, mismatch and MPPT/DC losses; it is not an efficiency map for arbitrary temperatures or irradiance angles.
