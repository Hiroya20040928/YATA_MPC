# Horizon-specific online residual models

For each horizon, filter and cap selection uses only earlier forecast origins whose matching target power has already been observed. The chosen model is then propagated to future target rows without intervening residual measurements. Future physical baseline power uses logged future speed/route inputs, so this is conditional on the recorded trajectory.

| Horizon | Day1–5 n | Baseline MAE (W) | Filter MAE (W) | Improvement | Baseline P90 (W) | Filter P90 (W) | Day6 MAE base→filter (W) | Day6 P90 base→filter (W) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 s | 12631 | 214.7 | 108.4 | +49.5% | 491.4 | 243.2 | 292.7→139.1 | 649.9→305.7 |
| 30 s | 12510 | 212.4 | 209.0 | +1.6% | 484.4 | 486.3 | 287.6→285.4 | 638.9→673.4 |
| 60 s | 12395 | 211.8 | 210.8 | +0.5% | 483.5 | 488.2 | 284.2→281.5 | 632.3→652.8 |
| 300 s | 11640 | 206.6 | 210.3 | -1.8% | 472.0 | 487.2 | 280.3→279.1 | 630.2→646.0 |

These tests report residual prediction only. Replaying measured future speeds does not verify future-weather or driver-command prediction, and block autocorrelation means the forecast count is not the number of independent experiments.

Outputs: `reports\residual_multihorizon_prequential_5s.csv`, `reports\residual_multihorizon_prequential_5s_summary.json`.
