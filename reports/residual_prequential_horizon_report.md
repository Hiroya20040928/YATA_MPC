# Prequential residual horizon decay

The residual state at each forecast origin uses only already completed samples. It is propagated without any intervening telemetry update. The physical baseline at the target uses the logged speed and route inputs, so this measures residual-correction decay conditional on the observed driving profile.

| Horizon | Day1–5 forecasts | Baseline MAE (W) | Filter MAE (W) | Improvement | Baseline P90 (W) | Filter P90 (W) | Day6 MAE baseline→filter (W) | Day6 P90 baseline→filter (W) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 s | 12631 | 214.7 | 109.1 | +49.2% | 491.4 | 245.2 | 292.7→142.4 | 649.9→319.4 |
| 30 s | 12510 | 212.4 | 231.7 | -9.1% | 484.4 | 550.4 | 287.6→315.1 | 638.9→748.5 |
| 60 s | 12395 | 211.8 | 238.3 | -12.5% | 483.5 | 565.4 | 284.2→335.9 | 632.3→788.3 |
| 300 s | 11640 | 206.6 | 228.0 | -10.4% | 472.0 | 534.7 | 280.3→318.4 | 630.2→754.3 |

The 5-second result is not a multi-minute accuracy guarantee. This horizon test holds the residual estimate fixed between measurements and therefore shows how quickly the usable online correction decays.

Outputs: `reports\residual_prequential_horizon_metrics.csv`, `reports\residual_prequential_horizon_summary.json`.
