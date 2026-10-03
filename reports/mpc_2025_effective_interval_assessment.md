# 有効区間での人間対二層MPCの所要時間比較

## 判定定義

同じ始点・終点を、イベント/Control Stop/充電/修理を含まない連続5秒ログ区間で比較しました。区間距離1 km以上、実走速度60–78.2 km/h、上位MILPが有限速度格子内でoptimal、予測エネルギー余裕非負を必須にしました。各区間内で上位目標速度を固定し、下位の30秒加速度MPCを5秒ごとに再求解して、実走時の入口速度から同じ距離を進む予測時間を求めました。

正の`human elapsed - MPC simulated`は、その選別区間においてモデル上のMPC時間が短いことを意味します。この判定はユーザー指定の最低限の時間指標に対応します。

## 結果

- 対象: 62区間、1241.2 km。人間 17.93 h、下位MPC付き予測 17.60 h。名目差は +0.33 h。
- 名目加速度上限でMPCが短時間: 39/62区間。上限を0.5倍/1.5倍にした3条件すべてで短時間だった区間: 37/62。
- 日別集計:

| day | intervals | distance_km | human_h | mpc_lower_nominal_h | nominal_saving_min | nominal_faster | all_accel_cases_faster |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 19 | 76.47 | 1.10 | 1.07 | 1.90 | 12 | 11 |
| 2 | 3 | 64.26 | 0.92 | 0.89 | 1.84 | 3 | 3 |
| 3 | 17 | 469.80 | 6.70 | 6.40 | 17.89 | 13 | 13 |
| 4 | 12 | 278.53 | 4.12 | 4.60 | -28.52 | 1 | 0 |
| 5 | 11 | 352.12 | 5.09 | 4.65 | 26.87 | 10 | 10 |

## すべての既存モデルの扱い

| model | use_in_replay | assessment |
| --- | --- | --- |
| vehicle 10 km Student-t MLE | upper energy model | LODO MAE 53.3 W; effective joined road-load coefficients; includes drivetrain behavior implicitly |
| PV faceted body POA | upper energy model | day-blocked MAE 50.5 W; selected over temperature candidate |
| PV Faiman temperature candidate | candidate comparison | day-blocked MAE 51.5 W vs POA 50.5 W; module temperature is unavailable, so not selected |
| battery capacity + voltage SOC + constant effective R | initial state and coulomb energy | 3011 Wh; 33.164 Ah; R=0.488 ohm; robust 3 min anchor |
| battery SOC-dependent R candidate | sensitivity audit | clean LODO MAE 0.658 V but Day6 worsens; not selected for primary state |
| vehicle inertial candidate | lower-layer sensitivity audit | adopt only if one-minute leave-one-day-out mean MAE improves and signed-inertia coefficients are stable in sign/magnitude; otherwise retain baseline and flag derivative as unidentifiable/noisy |
| M2096 ECO_LO bench map | interval support audit | speed support 70.92-83.26 km/h; installed PWM+ECO equivalence unconfirmed; not multiplied on top of fitted HV load |
| causal previous-6-block median residual | upper first-five-minute energy correction | nested same-day 5-minute MAE 70.2->57.1 W; applied only for 5 min, capped ±100 W |
| causal 5 s ARX residual | interval power-error diagnostic | conditional one-step correction; not extrapolated over upper horizon |
| causal 5 s online tree residual | interval power-error diagnostic | conditional one-step correction; not extrapolated over upper horizon |
| prequential residual filter | interval power-error diagnostic | past-only correction; not extrapolated over upper horizon |
| HGB full-data max fit | interval power-error diagnostic | max-fit committed; in-sample predictions are labeled, not used as validation |
| meta full-data max fit | interval power-error diagnostic | max-fit committed; in-sample predictions are labeled, not used as validation |
| residual candidate HGB_full_feature_sweep | common held-out model comparison | clean_days_1_5: n=2744, MAE=107.1 W, P90=221.5 W; not a multi-hour MPC forecast |
| residual candidate HGB_full_feature_sweep | common held-out model comparison | day6_stress: n=507, MAE=158.8 W, P90=353.9 W; not a multi-hour MPC forecast |
| residual candidate physical_baseline | common held-out model comparison | clean_days_1_5: n=2744, MAE=198.3 W, P90=440.9 W; not a multi-hour MPC forecast |
| residual candidate physical_baseline | common held-out model comparison | day6_stress: n=507, MAE=359.6 W, P90=767.2 W; not a multi-hour MPC forecast |
| residual candidate causal_ARX | common held-out model comparison | clean_days_1_5: n=2744, MAE=106.6 W, P90=226.4 W; not a multi-hour MPC forecast |
| residual candidate causal_ARX | common held-out model comparison | day6_stress: n=507, MAE=188.6 W, P90=437.0 W; not a multi-hour MPC forecast |
| residual candidate online_tree | common held-out model comparison | clean_days_1_5: n=2744, MAE=107.3 W, P90=228.7 W; not a multi-hour MPC forecast |
| residual candidate online_tree | common held-out model comparison | day6_stress: n=507, MAE=191.2 W, P90=440.1 W; not a multi-hour MPC forecast |
| residual candidate adaptive_filter | common held-out model comparison | clean_days_1_5: n=2744, MAE=107.5 W, P90=234.0 W; not a multi-hour MPC forecast |
| residual candidate adaptive_filter | common held-out model comparison | day6_stress: n=507, MAE=199.9 W, P90=476.8 W; not a multi-hour MPC forecast |
| residual candidate KRR_plus_adaptive_bias | common held-out model comparison | clean_days_1_5: n=2718, MAE=191.9 W, P90=425.8 W; not a multi-hour MPC forecast |
| residual candidate KRR_plus_adaptive_bias | common held-out model comparison | day6_stress: n=496, MAE=363.0 W, P90=785.3 W; not a multi-hour MPC forecast |
| residual candidate GP_plus_online_filter | common held-out model comparison | clean_days_1_5: n=2718, MAE=192.9 W, P90=417.7 W; not a multi-hour MPC forecast |
| residual candidate GP_plus_online_filter | common held-out model comparison | day6_stress: n=496, MAE=362.1 W, P90=778.9 W; not a multi-hour MPC forecast |
| residual candidate filter_family | common held-out model comparison | clean_days_1_5: n=2718, MAE=192.3 W, P90=431.4 W; not a multi-hour MPC forecast |
| residual candidate filter_family | common held-out model comparison | day6_stress: n=496, MAE=366.6 W, P90=769.6 W; not a multi-hour MPC forecast |
| residual candidate TCN | common held-out model comparison | clean_days_1_5: n=2744, MAE=151.2 W, P90=367.0 W; not a multi-hour MPC forecast |
| residual candidate TCN | common held-out model comparison | day6_stress: n=507, MAE=315.4 W, P90=670.0 W; not a multi-hour MPC forecast |

同じHV負荷を予測する複数モデルは足し合わせず、各々の検証指標として並行評価しました。最大適合HGB/メタ残差器は全データ学習内予測、ARX/木/フィルタは一段先条件付き予測です。上位MPCの複数時間エネルギー予測に一段先残差を再帰適用すると検証済み範囲を越えるため、主計画は車両10 kmモデル、PVモデル、電池状態推定を使用しています。モーターECO_LOベンチ値は設置中のPWM+ECOとの同一性が未確認なので、車両HVモデルへ二重に掛けず適用範囲を併記しています。

## 解釈上の制限

- これは実車で短縮した時間ではなく、ログを固定した反実仮想のモデル比較です。区間入口速度から下位MPCの加速度要求を積分しましたが、実車PWM/トルク応答がログにないため、加速度上限は人間運転履歴の代理値です。
- 上位目標、予報と再計画は人間ログの時刻列上で再生しています。速度変更に応じた実際の到達時刻の天気、故障発生位置、Control Stop到着時刻は再計算していません。
- 日別の部分区間を合算した時間差から、完走時間の短縮や人間より優れた大会結果は結論できません。全区間、停止、残量、イベントの反実仮想が整合する全日シミュレーションが必要です。
- 詳細な区間値は`mpc_2025_effective_intervals.csv`、モデル別電力誤差は各区間の`mae_*`列、モデル台帳は`mpc_2025_model_usage_audit.csv`に保存しています。
