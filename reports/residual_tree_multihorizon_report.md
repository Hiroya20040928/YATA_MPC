# Tree residual rollout by prediction horizon

At each origin, the model uses residual measurements through the origin. After that, it receives no residual updates. Predicted residuals are recursively fed into the next step. Future speed, route and weather inputs are taken from the logged path and treated as exact; this is a conditional test, not a complete MPC forecast.

| Horizon | Day1–5 n | Baseline MAE | Tree rollout MAE | Baseline P90 | Tree P90 | Day6 baseline→tree MAE | Day6 baseline→tree P90 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 s | 32 | 215.1 W | 111.5 W | 451.4 W | 221.2 W | 151.8→65.0 W | 238.6→120.0 W |
| 30 s | 32 | 217.9 W | 150.5 W | 577.4 W | 340.5 W | 198.3→185.0 W | 300.2→364.5 W |
| 60 s | 32 | 202.4 W | 198.2 W | 602.1 W | 496.9 W | 213.3→313.2 W | 377.4→571.7 W |
| 300 s | 28 | 305.3 W | 293.7 W | 482.7 W | 429.2 W | 525.3→421.9 W | 946.3→838.0 W |

Origins are sampled every 25 minutes after warm-up. The forecast windows overlap, so the rows are not independent samples; day-level differences and tail behavior matter more than nominal row counts.

Detailed forecasts: `reports\residual_tree_multihorizon_forecasts.csv`.
