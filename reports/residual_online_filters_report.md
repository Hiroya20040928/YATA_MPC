# BWSC2025 past-only residual filter comparison

## Method

At each 5-minute block, the filter family and correction cap are selected from one-step forecasts on earlier completed blocks only. Candidates are rolling medians, exponentially weighted averages, and a Theil–Sen robust local trend. The current observation is appended only after its prediction is recorded. This follows the state-estimation and robust online-filtering method families in Kalman (1960) and Lanius & Gather (2010); the specific candidate grid and clipping limits are engineering choices, not values prescribed by those papers.

## Day 1–5 next-block results

| Model | MAE (W) | improvement vs physics | P90 (W) | day wins | warm-state wins | Day6 P90 (W) |
|---|---:|---:|---:|---:|---:|---:|
| prequential_kernel | 61.3 | +12.6% | 134.2 | 4/5 | 4/5 | 163.0 |
| adaptive_median | 55.6 | +20.9% | 133.7 | 5/5 | 5/5 | 152.6 |
| adaptive_kalman | 61.5 | +12.4% | 134.1 | 5/5 | 5/5 | 167.5 |
| adaptive_filter_selection | 61.3 | +12.7% | 135.7 | 5/5 | 5/5 | 167.5 |

Selected configurations on warm clean-day updates: `{'median(9),cap=100W': 17, 'median(9),cap=75W': 15, 'median(6),cap=100W': 9, 'theil_sen(24),cap=30W': 9, 'median(9),cap=30W': 8, 'theil_sen(24),cap=50W': 8, 'median(9),cap=50W': 6, 'ewma(0.35),cap=75W': 4, 'theil_sen(6),cap=30W': 4, 'theil_sen(12),cap=50W': 3, 'ewma(0.5),cap=30W': 3, 'median(9),cap=150W': 3, 'theil_sen(24),cap=150W': 3, 'theil_sen(6),cap=100W': 3, 'theil_sen(12),cap=100W': 3, 'theil_sen(24),cap=100W': 2, 'theil_sen(24),cap=75W': 2, 'theil_sen(12),cap=150W': 2, 'ewma(0.2),cap=50W': 1, 'median(6),cap=30W': 1, 'median(3),cap=100W': 1}`.
The filter selector's prior inner MAE averaged 39.4 W; the next untouched block MAE averaged 52.8 W over 98 blocks.

At the original 5-second grain, clean-day MAE was 217.9 W baseline and 214.4 W after filtering; Day6 P90 was 641.9→653.6 W.

A shadow candidate must improve pooled clean MAE by at least 10%, win at least four of five days and warm-state days, and avoid worsening pooled-clean and Day6 block P90. This project gate is an engineering criterion, not an external standard. Passing it still does not meet the separate 23.8 W speed-decision threshold or establish BWSC2027 performance.

References: [Kalman (1960)](https://doi.org/10.1115/1.3662552); [Lanius & Gather (2010)](https://doi.org/10.1016/j.csda.2009.10.009).

Outputs: `reports\residual_online_filters_oof.csv`, `reports\residual_online_filters_by_day.csv`, `reports\residual_online_filters_summary.json`, `data\processed\full_race_residual_online_filters_5s.csv`.
