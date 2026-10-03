# Causal ARX residual candidate for 5-second forecasts

The prediction combines the previous measured power residual with current causal signed inertial power. It fits a robust regularized linear model on prior rows only. Speed smoothing, lookback and correction cap are selected every 60 completed samples using a 60-sample inner one-step holdout.

Day1–5 MAE: physics 219.8 W; residual-only AR filter 122.4 W; causal ARX 121.6 W. ARX P90 509.3→282.7 W; day wins 5/5, warm-state wins 5/5.
Day6 stress MAE 296.3→155.0 W; P90 647.6→355.4 W.
Inner MAE 99.5 W, next-sample MAE 108.1 W over 12631 updates.
Selected configurations: `{'accel_alpha=0.2,window=300,cap=500W': 1320, 'accel_alpha=0.2,window=300,cap=1000W': 1296, 'accel_alpha=0.5,window=300,cap=1000W': 1057, 'accel_alpha=0.2,window=300,cap=300W': 869, 'accel_alpha=0.5,window=120,cap=1000W': 720, 'accel_alpha=0.5,window=300,cap=500W': 684, 'accel_alpha=0.5,window=300,cap=300W': 660, 'accel_alpha=0.2,window=120,cap=1000W': 540, 'accel_alpha=0.35,window=300,cap=1000W': 540, 'accel_alpha=0.2,window=120,cap=500W': 540, 'accel_alpha=0.35,window=300,cap=300W': 480, 'accel_alpha=0.5,window=120,cap=500W': 360, 'accel_alpha=0.35,window=300,cap=500W': 360, 'accel_alpha=0.2,window=120,cap=300W': 360, 'accel_alpha=0.35,window=120,cap=1000W': 360, 'accel_alpha=0.2,window=60,cap=1000W': 300, 'accel_alpha=0.5,window=60,cap=1000W': 240, 'accel_alpha=0.2,window=60,cap=300W': 240, 'accel_alpha=0.2,window=60,cap=500W': 240, 'accel_alpha=0.35,window=120,cap=300W': 240, 'accel_alpha=0.5,window=120,cap=300W': 240, 'accel_alpha=0.5,window=60,cap=500W': 205, 'accel_alpha=0.35,window=60,cap=500W': 120, 'accel_alpha=0.2,window=300,cap=100W': 120, 'accel_alpha=0.5,window=60,cap=300W': 120, 'accel_alpha=0.35,window=60,cap=1000W': 120, 'accel_alpha=0.35,window=120,cap=500W': 60, 'accel_alpha=0.2,window=60,cap=100W': 60, 'accel_alpha=0.35,window=300,cap=100W': 60, 'accel_alpha=0.5,window=120,cap=100W': 60, 'accel_alpha=0.2,window=120,cap=100W': 60}`.

This is one-step prediction only. The 314 kg feature scaling is a design-mass proxy, not a measurement of driver-loaded mass.

Outputs: `data\processed\full_race_residual_arx_5s.csv`, `reports\residual_arx_5s_by_day.csv`, `reports\residual_arx_5s_summary.json`.
