# Cross-day residual-model evaluation

Residual corrections trained across days were tested in outer leave-one-day-out folds. Within each fold, every residual label for a training day was generated from a physics model excluding both that day and the outer test day. Day 6 was not used.

## Results

| Method | pooled MAE after (W) | pooled change (%) | macro daily MAE (W) | day wins / 5 | P90 after (W) | 90% interval coverage* |
|---|---:|---:|---:|---:|---:|---:|
| crossday_bias | 69.6 | +0.8 | 69.7 | 3/5 | 142.2 | — |
| ridge_a1 | 73.1 | -4.1 | 72.8 | 1/5 | 139.9 | — |
| ridge_a10 | 71.7 | -2.1 | 72.1 | 1/5 | 144.9 | — |
| gpr_rbf_noise60 | 65.6 | +6.6 | 64.9 | 3/5 | 128.5 | 88.4% |
| gpr_rbf_noise100 | 65.9 | +6.1 | 65.5 | 4/5 | 137.4 | 96.1% |
| route_10km_diagnostic | 70.2 | +0.0 | 69.4 | 0/5 | 142.0 | — |

The fixed physical baseline is included as the zero-correction reference in the CSV. Ridge fits use only speed, reconstructed relative air speed, route grade, ambient temperature and air density, with equal total weight per training day. The fixed RBF Gaussian-process candidates use those same five inputs and day-scaled observation noise; their nominal 90% intervals are reported as diagnostics, not guaranteed coverage. The distance-bin method is a 2025-course memorization diagnostic and cannot transfer to a different route.

Day-specific distance ranges in the eligible data are: Day 1: 3–533 km; Day 2: 539–720 km; Day 3: 1077–1491 km; Day 4: 1605–1689 km; Day 5: 2089–2560 km. The ranges do not overlap in these blocks, explaining why a 10 km location lookup has no repeated-day evidence and cannot be trained from this race log.

*For the Gaussian-process candidates, coverage is measured against the held-out residuals across five outer folds. The estimated intervals omit day-level distribution shift and cannot be treated as calibrated MPC safety bounds.

No cross-day model is adopted. Training-fold residual labels come from three-day physical fits while the outer predictor uses four-day fits, so this deliberately leakage-safe design is conservative but not perfectly identical in base-model training size. Five independent day folds are too few to tune flexible models reliably.

Outputs: `residual_crossday_oof.csv`, `residual_crossday_by_day.csv`, `residual_crossday_summary.json`.
