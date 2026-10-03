# BWSC2025 past-only 5-second residual prediction

At each row, the predictor uses only earlier 5-second telemetry. It chooses among rolling median, EWMA, persistence and AR(1) filters every 60 completed samples, scoring candidate choices on the preceding 60 one-step forecasts. The current measured power enters the state only after its prediction is recorded. State resets after a gap over ten minutes.

## Results

| Data | Baseline MAE (W) | Residual filter MAE (W) | Improvement | P90 baseline → filter (W) |
|---|---:|---:|---:|---:|
| Day1–5 all rows (n=13711) | 219.8 | 122.4 | +44.3% | 509.3 → 281.0 |
| Day6 stress (n=2531) | 296.3 | 160.3 | — | 647.6 → 386.3 |

Day wins: 5/5; warm-state wins: 5/5. Prior inner MAE averaged 99.8 W; next untouched-sample MAE averaged 109.0 W over 12631 scored updates.
Selected settings: `{'ar1(300),cap=1000W': 1752, 'persistence(1),cap=1000W': 1320, 'persistence(1),cap=500W': 1080, 'persistence(0.75),cap=1000W': 924, 'persistence(0.75),cap=500W': 900, 'ar1(300),cap=500W': 865, 'ar1(120),cap=1000W': 777, 'ar1(300),cap=300W': 660, 'ar1(60),cap=500W': 540, 'persistence(1),cap=300W': 480, 'ar1(120),cap=500W': 480, 'ar1(300),cap=200W': 480, 'ar1(120),cap=300W': 389, 'persistence(1),cap=200W': 360, 'ar1(60),cap=1000W': 304, 'ar1(60),cap=300W': 240, 'ar1(120),cap=200W': 240, 'ar1(300),cap=100W': 240, 'ar1(120),cap=100W': 180, 'ar1(60),cap=200W': 180, 'persistence(0.75),cap=300W': 120, 'persistence(0.75),cap=200W': 60, 'ewma(0.75),cap=100W': 60}`.

The 5-second residual autocorrelation is useful for receding one-step correction but falls toward zero over about one minute. This result cannot be interpreted as a long-horizon MPC accuracy claim. The project shadow gate also requires no pooled clean-day or Day6 P90 worsening; passing it does not establish BWSC2027 transfer or meet the 23.8 W speed-decision threshold.

Outputs: `data\processed\full_race_residual_prequential_5s.csv`, `reports\residual_prequential_5s_by_day.csv`, `reports\residual_prequential_5s_summary.json`.
