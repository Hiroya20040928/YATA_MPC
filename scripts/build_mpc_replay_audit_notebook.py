#!/usr/bin/env python3
"""Build a portable audit notebook for the BWSC2025 two-layer MPC replay."""
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/mpc_2025_shadow_replay_audit.ipynb"

cells = [
    nbformat.v4.new_markdown_cell("""# BWSC2025 二層MPCシャドーリプレイ監査

## tl;dr

このノートブックは、2025年ZPログを使った上位/下位MPCの時系列監査を再現します。完走時間の改善や実車の閉ループ追従を測ったものではありません。ログにPWM/スロットル指令がなく、係数もDay1–5全体から事後適合しているため、予報ビンテージの因果性とモデル学習の因果性は別々に評価します。

実行結果の要約は下のセルから読み込みます。元データ、計画CSV、日別集計、最適化statusを確認します。"""),
    nbformat.v4.new_markdown_cell("""## Context & Methods

### Key Assumptions

- 上位MPCは30分ごと、60–78 km/hを1 km/h刻み、日末17時・残量15%を暫定条件にして全ルート残区間を計画します。終端残量感度は0/10/15/20%です。
- 下位MPCは5秒ログから30秒一定加速度の速度追従を提案します。速度変化履歴は運転者/車両の代理値であり、アクチュエータ同定値ではありません。
- 予報はOpen-Meteo ECMWF IFS Single Runs。各判断で初期化から6時間以上経過したランだけを使います。座標は50 km間隔に補間されます。
- 車両/PVの係数は2025 Day1–5ログを使って事後適合済みです。これは判断時点に係数が既知だったという意味ではありません。
- トラブルイベントはログ上の発生時点で出力を停止します。速度変更でイベントの発生時刻や場所が変わる過程はモデル化していません。

入力と再現コードは`data/processed/mpc_2025_as_live_replay.csv`、`scripts/replay_hierarchical_mpc_2025.py`です。Open-MeteoのSingle Runsは過去のモデル初期化を個別に指定できます。最適化statusは[scipy.optimize.milp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)の終了状態とMIP gapに従います。"""),
    nbformat.v4.new_markdown_cell("""## Data

### 1. Load artifacts

The notebook reads existing workspace artifacts only; it does not fetch or rewrite the weather archive."""),
    nbformat.v4.new_code_cell("""from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / "reports/mpc_2025_replay_summary.json").exists():
    ROOT = ROOT.parent

replay = pd.read_csv(ROOT / "data/processed/mpc_2025_as_live_replay.csv", low_memory=False)
days = pd.read_csv(ROOT / "reports/mpc_2025_day_comparison.csv")
reserve = pd.read_csv(ROOT / "reports/mpc_2025_reserve_sensitivity.csv")
effective = pd.read_csv(ROOT / "reports/mpc_2025_effective_intervals.csv")
effective_days = pd.read_csv(ROOT / "reports/mpc_2025_effective_interval_day_summary.csv")
effective_summary = json.loads((ROOT / "reports/mpc_2025_effective_interval_summary.json").read_text(encoding="utf-8"))
summary = json.loads((ROOT / "reports/mpc_2025_replay_summary.json").read_text(encoding="utf-8"))
replay["timestamp_utc"] = pd.to_datetime(replay["timestamp_utc"], utc=True)
replay["forecast_run_init_utc"] = pd.to_datetime(replay["forecast_run_init_utc"], utc=True)
replay["forecast_age_at_decision_h"] = pd.to_numeric(replay["forecast_age_at_decision_h"], errors="coerce")

decision = replay[replay["upper_replan_now"].fillna(False)].copy()
status_counts = decision["upper_solver_status"].value_counts()
print(f"Rows: {len(replay):,}; days: {replay['day'].nunique()}; upper solves: {len(decision)}")
display(days.round(2))"""),
    nbformat.v4.new_markdown_cell("""### 2. Check timing, events, and optimization state

The checks below distinguish archived forecast availability from retrospective model fitting. They also check event suppression, solver feasibility for certified rows, and the telemetry energy reconciliation."""),
    nbformat.v4.new_code_cell("""event_pattern = r"incident:|control_stop:|traffic_stop:|driver_change:|repair:|charge:|hatch_repair:"
event_now = replay["event_flags"].fillna("").str.contains(event_pattern, regex=True)
has_command = replay["upper_target_speed_kmh"].notna() | replay["lower_accel_request_mps2"].notna()
optimal = decision[decision["upper_solver_status"].eq("optimal_finite_grid_milp")]
energy_slack_wh = (pd.to_numeric(optimal["upper_energy_budget_wh"], errors="coerce")
                   - pd.to_numeric(optimal["upper_predicted_remaining_net_energy_wh"], errors="coerce"))
forecast_age = decision["forecast_age_at_decision_h"]

checks = {
    "all_2025_log_rows_present": len(replay) == 43963,
    "six_race_days_present": replay["day"].nunique() == 6,
    "forecast_publication_delay_respected": bool((forecast_age >= 6.0).all()),
    "no_command_during_flagged_event_rows": int((event_now & has_command).sum()) == 0,
    "optimal_rows_within_energy_budget_0.01Wh": bool((energy_slack_wh >= -0.01).all()),
    "decision_count_matches_day_summary": len(decision) == int(days["upper_replans_30min"].sum()),
    "effective_intervals_match_saved_summary": len(effective) == effective_summary["human_beating_counts"]["intervals"],
}
display(pd.Series(checks, name="passed"))
assert all(checks.values()), checks

print("Solver result counts:")
display(status_counts.rename_axis("status").to_frame("decisions"))
print("Forecast age (hours):")
display(forecast_age.describe().to_frame("hours"))
print("Certified-optimum minimum energy slack [Wh]:", round(float(energy_slack_wh.min()), 4))"""),
    nbformat.v4.new_code_cell("""# Reconcile the day-level net battery energy with the independent log summary.
recon = pd.read_csv(ROOT / "reports/energy_balance_reconciliation.csv").set_index("day")
day_energy = days.set_index("day")["observed_battery_energy_change_wh"]
source_energy = recon["logged_battery_signed_wh"]
energy_check = pd.concat([day_energy.rename("replay_wh"), source_energy.rename("reconciled_source_wh")], axis=1)
energy_check["difference_wh"] = energy_check["replay_wh"] - energy_check["reconciled_source_wh"]
display(energy_check.round(3))
assert energy_check.loc[energy_check.index <= 5, "difference_wh"].abs().max() < 0.01
print("Days 1-5 reconcile within 0.01 Wh. Day 6 is shown but excluded from that assertion:")
print("its 3,485 s acquisition gap makes trapezoidal bridging across the gap non-observed energy.")

# Day 6 contains an acquisition gap. It is excluded from battery integration.
dt = replay.groupby("day")["timestamp_utc"].diff().dt.total_seconds()
gaps = replay.loc[dt > 30, ["day", "timestamp_utc"]].copy()
gaps["gap_seconds"] = dt[dt > 30].to_numpy()
display(gaps)"""),
    nbformat.v4.new_markdown_cell("""## Results

### 3. Compare logged speeds with advisory targets

This is a speed-target comparison, not a counterfactual finish-time estimate. Targets are not applied to the logged vehicle trajectory."""),
    nbformat.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
x = np.arange(len(days))
axes[0].plot(x, days["logged_speed_median_kmh"], marker="o", label="Human logged median")
axes[0].plot(x, days["shadow_target_median_kmh"], marker="s", label="MPC target median")
axes[0].set_xticks(x, [f"Day {d}" for d in days["day"]])
axes[0].set_ylabel("Speed [km/h]")
axes[0].set_title("Logged speed vs advisory target")
axes[0].grid(alpha=.25)
axes[0].legend()

status_counts.sort_values().plot(kind="barh", ax=axes[1], color="#35618f")
axes[1].set_xlabel("Upper MPC solves")
axes[1].set_title("MILP termination / fallback status")
axes[1].grid(axis="x", alpha=.25)
fig.suptitle("BWSC2025 shadow replay; recommendations were not applied")
fig.tight_layout()
plt.show()"""),
    nbformat.v4.new_code_cell("""# The reserve sensitivity table compares alternate constraints at each day's first decision.
display(reserve[["day", "reserve_pct", "target_speed_kmh", "status",
                 "predicted_remaining_time_h", "predicted_remaining_net_energy_wh"]].round(2))"""),
    nbformat.v4.new_markdown_cell("""### 4. Compare time over matching effective intervals

Intervals are restricted to clean continuous records, 1 km or longer, where the upper MILP proved its finite-grid optimum and its battery-energy constraint was feasible. The lower-layer acceleration request is re-simulated from the logged entry speed. This estimates a same-distance counterfactual; it is not an observed vehicle run."""),
    nbformat.v4.new_code_cell("""display(effective_days.round(2))
print(f"Matched distance: {effective['distance_km'].sum():.1f} km")
print(f"Human elapsed: {effective['human_elapsed_min'].sum()/60:.2f} h")
print(f"Lower-MPC nominal simulated time: {effective['mpc_lower_layer_nominal_min'].sum()/60:.2f} h")
print(f"Nominal time saved: {effective['saving_nominal_min'].sum()/60:+.2f} h")
print("Intervals faster in all three acceleration-limit cases:",
      int((effective[["saving_half_accel_min", "saving_nominal_min", "saving_1p5x_accel_min"]].gt(0).all(axis=1)).sum()),
      "/", len(effective))
display(effective[["day", "start_distance_km", "end_distance_km", "distance_km",
                   "human_elapsed_min", "mpc_target_kmh", "mpc_lower_layer_nominal_min",
                   "saving_nominal_min", "saving_half_accel_min", "saving_1p5x_accel_min",
                   "minimum_energy_slack_wh"]].round(2))"""),
    nbformat.v4.new_markdown_cell("""## Takeaways

- `optimal_finite_grid_milp` proves the optimum only for that row's finite 60–78 km/h problem and the fitted model. A time-limited incumbent is advisory only after it passes the same hard reserve and deadline checks. Infeasible/no-incumbent rows are diagnostic only and contain no speed recommendation.
- Archived forecast runs satisfy the 6-hour availability rule. Vehicle/PV calibration still uses later Day1–5 logs, so the replay is not a fully causal historical deployment test.
- The human speed trace remains the actual plant trajectory. There is no logged PWM/throttle input, measured tracking response, or independent future-weather truth, so this audit cannot establish a human-beating race or real-vehicle MPC performance.
- On strict matched effective intervals, positive modeled time savings support local model-based superiority under the stated assumptions. They do not establish measured superiority or whole-race superiority; prior residual model families are scored alongside each interval.
- The output is suitable for continued shadow review. It is not calibrated for vehicle actuation.

For the full controller design and daily results, see [`reports/hierarchical_mpc_2025_replay.md`](../reports/hierarchical_mpc_2025_replay.md). For the matching-interval time criterion and model-usage audit, see [`reports/mpc_2025_effective_interval_assessment.md`](../reports/mpc_2025_effective_interval_assessment.md)."""),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, OUT)
print(OUT)
