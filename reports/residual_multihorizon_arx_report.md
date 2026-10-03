# Horizon-specific causal ARX residual evaluation

At each origin, the predictor is trained only on earlier completed 5-second rows. It propagates a residual state with a lagged residual and a signed inertial input derived from the logged speed trajectory. Horizon-specific parameters are selected from past origins whose target residual is already observed. The logged future speed is a conditional input, not a speed forecast.

| Horizon | Day1–5 n | Baseline MAE (W) | ARX MAE (W) | Improvement | Baseline P90 (W) | ARX P90 (W) | Day6 MAE baseline→ARX (W) | Day6 P90 baseline→ARX (W) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 s | 11698 | 206.2 | 103.2 | +50.0% | 471.5 | 233.4 | 278.6→124.2 | 620.0→275.4 |
| 30 s | 11643 | 205.6 | 199.2 | +3.1% | 469.8 | 461.1 | 277.1→264.7 | 613.4→622.8 |
| 60 s | 11577 | 205.4 | 206.3 | -0.4% | 467.5 | 478.7 | 276.0→260.5 | 613.4→606.0 |
| 300 s | 11049 | 205.9 | 213.8 | -3.8% | 473.6 | 492.1 | 277.4→314.6 | 627.0→750.5 |

Future speed inputs come from the recorded telemetry and are treated as known. That makes this a conditional residual-power test; it does not include error in the future speed plan, weather forecast, or route condition. The inertial basis uses a 314 kg design proxy and does not identify loaded mass or motor efficiency.

Outputs: `reports\residual_multihorizon_arx_forecasts.csv`, `reports\residual_multihorizon_arx_summary.json`.
