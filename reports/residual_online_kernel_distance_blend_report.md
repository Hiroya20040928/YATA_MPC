# Route-distance model blend evaluation

The distance-enabled and location-free prequential kernel models are blended with a weight selected from the prior 12 completed blocks. The current target is excluded from weight selection.

| Model | Day1–5 MAE (W) | P90 (W) | improvement vs physics |
|---|---:|---:|---:|
| adaptive_median | 55.6 | 133.7 | +20.9% |
| distance_median | 56.4 | 130.2 | +19.6% |
| adaptive_distance_blend | 55.6 | 130.2 | +20.7% |

Selected distance weights: `{'distance_weight=0': 72, 'distance_weight=1': 21, 'distance_weight=0.75': 9, 'distance_weight=0.25': 5}`. Prior inner MAE 45.6 W; next-block MAE 51.2 W over 107 blocks.
Day6 P90: physics 175.0 W, adaptive median 152.6 W, distance blend 152.6 W.
5-second row MAE: physics 217.9 W, blended correction 212.2 W.

The route feature is evaluated as a predictive input. Its fitted effect does not identify a physical cause such as road surface or wind exposure.

Outputs: `reports\residual_online_kernel_distance_blend_oof.csv`, `reports\residual_online_kernel_distance_blend_summary.json`, `data\processed\full_race_residual_online_kernel_distance_blend_5s.csv`.
