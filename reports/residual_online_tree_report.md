# Online nonlinear residual candidate (5-second one-step)

The candidate compares shallow histogram gradient boosting and extremely randomized trees. At each 300-row update, model family and history window are selected using a 60-row inner holdout made entirely from completed data. The selected model is refit every 12 rows using only residuals that had already been observed. It resets after a logging gap over ten minutes.

Day1–5 MAE: baseline 219.8 W; tree candidate 138.6 W. P90 509.3→339.3 W.
Warm-state MAE: baseline 211.4 W; tree candidate 105.0 W. Day6 stress MAE 296.3→190.7 W.
Mean inner MAE 119.5 W; following untouched outer-block MAE 110.8 W.

Feature inputs include logged speed, causal inertial power using the 314 kg proxy, route grade, distance, wind and ambient temperature. These are predictive features only; they do not identify motor efficiency, true mass or sensor bias. Future MPC inputs and forecast errors are not simulated.

Outputs: `data\processed\full_race_residual_online_tree_5s.csv`, `reports\residual_online_tree_by_day.csv`, `reports\residual_online_tree_summary.json`.
