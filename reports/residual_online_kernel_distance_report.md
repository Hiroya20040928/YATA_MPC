# BWSC2025 online kernel residual learner

## Validation design

Each 5-minute block is predicted using completed residual blocks only. Before predicting, an RBF kernel-ridge learner selects length scale, regularization, 12/24/36/48-block lookback and squared-error versus Huber loss (δ=30/60/100 W) from expanding one-step forecasts inside the prior 12 blocks (minimum 8 earlier training blocks), then refits using the selected history. Two causal residual-state corrections are compared: a past-only median (window 3/6/9 blocks) and a robust local-level Kalman filter whose process/measurement variance ratio (0.001/0.01/0.05/0.1/0.5) and total correction cap (30/50/75/100/150 W) are selected from prior one-step errors. Kalman updates clip past innovations at three robust standard deviations. The current target is excluded from fitting and parameter choice. Histories reset after gaps over 10 minutes.

## Day 1–5 results

| Model | pooled MAE (W) | improvement vs physics | P90 (W) | day wins | warm-state wins |
|---|---:|---:|---:|---:|---:|
| prequential_kernel | 61.6 | +12.2% | 136.9 | 4/5 | 4/5 |
| prequential_kernel_plus_adaptive_bias | 56.4 | +19.6% | 130.2 | 5/5 | 5/5 |
| prequential_kernel_plus_kalman_bias | 62.1 | +11.5% | 135.3 | 5/5 | 5/5 |
| nested_selected_online_rule | 57.1 | +18.7% | 130.7 | 4/5 | 5/5 |

## Hyperparameter and bias diagnostics

Selected RBF/Huber configurations over clean-day prequential updates: `{'ell=2,alpha=0.1,lookback=48,huber=none': 25, 'ell=1,alpha=0.1,lookback=12,huber=none': 16, 'ell=2,alpha=0.1,lookback=12,huber=none': 10, 'ell=1,alpha=1,lookback=12,huber=30W': 8, 'ell=2,alpha=0.1,lookback=48,huber=30W': 8, 'ell=2,alpha=1,lookback=12,huber=30W': 7, 'ell=0.5,alpha=100,lookback=12,huber=30W': 6, 'ell=1,alpha=0.1,lookback=48,huber=none': 5, 'ell=0.5,alpha=100,lookback=48,huber=30W': 5, 'ell=0.5,alpha=0.1,lookback=12,huber=none': 4, 'ell=2,alpha=1,lookback=12,huber=none': 3, 'ell=2,alpha=1,lookback=48,huber=60W': 3, 'ell=0.5,alpha=1,lookback=12,huber=none': 2, 'ell=1,alpha=1,lookback=48,huber=30W': 1, 'ell=0.5,alpha=0.1,lookback=48,huber=none': 1, 'ell=2,alpha=1,lookback=48,huber=none': 1, 'ell=2,alpha=0.1,lookback=12,huber=30W': 1, 'ell=2,alpha=0.1,lookback=36,huber=none': 1}`.
Selected innovation window/cap configurations: `{'median=9,cap=30W': 27, 'median=9,cap=100W': 17, 'median=9,cap=75W': 13, 'median=3,cap=30W': 11, 'median=6,cap=100W': 9, 'median=3,cap=100W': 7, 'median=9,cap=50W': 6, 'median=3,cap=75W': 4, 'median=6,cap=50W': 4, 'median=9,cap=150W': 3, 'median=3,cap=50W': 2, 'median=6,cap=30W': 2, 'median=6,cap=75W': 1, 'median=3,cap=150W': 1}`.
Selected Kalman process/measurement ratio and cap configurations: `{'q_over_r=0.5,cap=30W': 16, 'q_over_r=0.001,cap=100W': 15, 'q_over_r=0.01,cap=100W': 13, 'q_over_r=0.001,cap=75W': 12, 'q_over_r=0.5,cap=50W': 9, 'q_over_r=0.01,cap=30W': 8, 'q_over_r=0.001,cap=30W': 7, 'q_over_r=0.05,cap=30W': 4, 'q_over_r=0.1,cap=30W': 3, 'q_over_r=0.01,cap=75W': 3, 'q_over_r=0.1,cap=75W': 3, 'q_over_r=0.5,cap=75W': 3, 'q_over_r=0.05,cap=50W': 2, 'q_over_r=0.001,cap=50W': 2, 'q_over_r=0.01,cap=50W': 1, 'q_over_r=0.001,cap=150W': 1, 'q_over_r=0.01,cap=150W': 1, 'q_over_r=0.1,cap=100W': 1, 'q_over_r=0.05,cap=100W': 1, 'q_over_r=0.5,cap=100W': 1, 'q_over_r=0.05,cap=150W': 1}`.

The time-ordered holdout directly tests whether model selection on prior residuals generalizes to later blocks. It avoids selecting one global configuration on the same outer-fold outcomes. Repeated blocks remain autocorrelated, so daily paired results and P90 are retained; the five race days are not treated as hundreds of independent days.

For the median-corrected model, prior inner-validation MAE averaged 40.9 W and the next untouched-block MAE averaged 53.5 W over 98 blocks. For the Kalman model the corresponding values were 41.6 W and 54.4 W. The inner-to-next gap makes tuning optimism visible; adoption is based on the outer next-block scores.

Day 6 is reported only as anomaly stress and never selects the model. The inputs are retrospective race-weather values rather than archived forecasts; forecast error is not represented. This method is not connected to the model replay, Grafana, or commands.

Applying the median correction to the original 5-second rows gives clean-day MAE 217.9→212.4 W; the Kalman version gives 214.5 W. On Day6, P90 is 641.9 W for baseline, 651.5 W for median, and 653.8 W for Kalman. Full row output: `data\processed\full_race_residual_online_kernel_distance_5s.csv`.

## Current decision

A candidate is promoted to shadow evaluation only if it meets the project gate of ≥10% pooled clean-day MAE improvement, at least four of five day wins, at least four of five warm-state wins, and no pooled-clean or Day 6 stress-set P90 worsening. This is a provisional engineering rule. Passing it does not establish BWSC2027 transfer or satisfy the separate 23.8 W speed-decision error gate.

Outputs: `residual_online_kernel_oof.csv`, `residual_online_kernel_by_day.csv`, `residual_online_kernel_summary.json`, this report, and the full five-second corrected log.
