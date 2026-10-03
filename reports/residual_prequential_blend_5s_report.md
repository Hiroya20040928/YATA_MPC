# Past-only blend of online 5-second residual predictors

At each five-minute update, the blend weight is selected from the preceding 60 completed samples. Both component predictors were generated causally on the same race log. The current target is appended only after its prediction is fixed.

Day1–5 MAE: physics 219.8 W; AR/persistence 122.4 W; causal inertial 200.6 W; selected blend 123.4 W. Blend P90 509.3→284.3 W; Day6 P90 647.6→381.1 W.
Prior blend-selection MAE averaged 112.1 W; next-sample MAE averaged 110.0 W. Selected AR weights: `{'ar_weight=1': 10747, 'ar_weight=0.75': 1704, 'ar_weight=0.5': 180}`.

This evaluates only the next 5-second sample. The same controller cannot carry this correction into longer horizons without a separate forecast model.

Outputs: `data\processed\full_race_residual_prequential_blend_5s.csv`, `reports\residual_prequential_blend_5s_by_day.csv`, `reports\residual_prequential_blend_5s_summary.json`.
