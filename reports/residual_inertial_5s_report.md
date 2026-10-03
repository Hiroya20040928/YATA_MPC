# Causal acceleration residual candidate

This candidate uses a backward-looking exponentially smoothed speed derivative to form signed inertial power bases. A robust bounded linear fit is trained on completed rows only; smoothing time constant, training window and correction cap are selected every five minutes from past one-step errors. This avoids the centered smoothing's future leakage in the earlier descriptive inertial analysis.

Day1–5 baseline MAE 219.8 W → candidate 199.7 W (+9.2%), P90 509.3→466.5 W; full-day wins 4/5 and warm-state wins 4/5.
Day6 stress MAE 296.3→273.2 W; P90 647.6→622.6 W.
Prior inner MAE averaged 184.2 W and next one-step MAE 192.9 W over 12631 samples.
Selected settings: `{'EWMA alpha=0.5,window=120,cap=100W': 1812, 'EWMA alpha=0.2,window=300,cap=100W': 1440, 'EWMA alpha=0.5,window=300,cap=100W': 1380, 'EWMA alpha=0.2,window=120,cap=100W': 1320, 'EWMA alpha=0.5,window=120,cap=300W': 1181, 'EWMA alpha=0.5,window=120,cap=500W': 904, 'EWMA alpha=0.5,window=300,cap=500W': 660, 'EWMA alpha=0.5,window=300,cap=300W': 660, 'EWMA alpha=0.35,window=120,cap=100W': 624, 'EWMA alpha=0.5,window=300,cap=1000W': 540, 'EWMA alpha=0.5,window=120,cap=1000W': 408, 'EWMA alpha=0.35,window=300,cap=100W': 385, 'EWMA alpha=0.35,window=120,cap=300W': 360, 'EWMA alpha=0.2,window=120,cap=300W': 357, 'EWMA alpha=0.2,window=300,cap=300W': 240, 'EWMA alpha=0.35,window=300,cap=300W': 180, 'EWMA alpha=0.35,window=120,cap=1000W': 120, 'EWMA alpha=0.35,window=120,cap=500W': 60}`.

The 314 kg basis is a prior proxy rather than confirmed loaded mass. Its coefficient is predictive, not an identified physical mass or motor efficiency. The 5-second one-step result does not establish multi-step MPC accuracy.

Outputs: `data\processed\full_race_residual_inertial_5s.csv`, `reports\residual_inertial_5s_by_day.csv`, `reports\residual_inertial_5s_summary.json`.
