# BWSC2025 speed–power prediction at race-representative conditions

![Speed–power curve](speed_power_prediction.png)

The solid curve is the current 10 km segment MLE gross HV load model evaluated at representative BWSC2025 conditions. The 2025 telemetry points use `Solar_Watt + Battery_Watt`; the purple dotted curve subtracts median modeled daylight PV. This is a steady-cruise model (zero acceleration), not a closed-loop simulation.

## Representative conditions

| Condition | Value | Basis |
|---|---:|---|
| Road rows | 16,240 | clean moving rows, Days 1–6 |
| Ambient temperature | 27.814 | °C median |
| Surface pressure | 990.858 | hPa median |
| Air density | 1.148 | kg/m³ median |
| Wind speed | 4.027 | km/h median |
| Wind direction | 184.314 | degrees FROM median |
| Along-route headwind component | -0.214 | m/s median; negative is tailwind |
| Route grade | 0.067 | % median |
| Auxiliary load | 67.600 | W; includes 21 W weak systems |
| PV power | 552.051 | W median of modeled moving-daylight output |
| Nominal pack energy | 3011.000 | Wh; user basis, 3011 Wh |

## Predictions

| Speed (km/h) | Model domain | Gross vehicle power (W) | Net battery power at median PV (W) | Raw sample n | Raw median (W) | Standardized median (W) | Idealized 3011 Wh depletion time (h) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 50 | extrapolation | 429.3 | -122.7 | 39 | 824.7 | 956.2 | surplus at median PV |
| 60 | supported | 610.7 | 58.7 | 272 | 729.1 | 673.8 | 51.33 |
| 70 | supported | 851.9 | 299.8 | 2352 | 864.0 | 886.6 | 10.04 |
| 80 | extrapolation | 1162.9 | 610.8 | 105 | 777.5 | 1096.0 | 4.93 |
| 85 | extrapolation | 1347.7 | 795.7 | 0 | — | — | 3.78 |

The depletion-time column is only a gross nominal-energy ratio (3011 Wh divided by positive net power). It assumes the entire theoretical pack energy is usable and ignores reserve SoC, voltage/current limits, temperature, discharge efficiency, and changing solar/weather. Negative battery power means modeled PV exceeds modeled vehicle demand under the chosen median conditions; it is not a charging-limit prediction.

The actual telemetry IQR includes acceleration, braking/regeneration, route grade and measurement/model error; it is observed spread, not a confidence interval. Condition-standardized telemetry subtracts the full-fit model at each row's logged conditions and adds its prediction at median conditions, so it is a model adjustment rather than independent validation. The 10 km model was trained on segment mean speeds from 57.6 to 78.2 km/h; solid portions are within that support and dashed portions are extrapolations. The 50, 80 and 85 km/h table rows are outside training support; 85 km/h has no clean plotted telemetry samples. The adjacent ±2.5 km/h decision-resolution gate remains failed (53.3 W 10 km LODO MAE vs. 23.8 W internal gate). See `BWSC2025_component_model_iteration.md` for validation and limits.
