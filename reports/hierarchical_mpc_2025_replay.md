# 上位・下位2層MPC：BWSC2025 as-of シャドーリプレイ

## 結果の扱い

2025年の走行ログを時刻順に再生し、各判断時点で6時間以上前に初期化された気象予報ランを上位計画に渡した**シャドー評価**です。車両/PV係数は既存のDay1–5ログで事後適合した値で、2025年当日に既知だった係数ではありません。気象予報と状態更新は時系列に沿っていますが、パラメータ学習まで厳密なas-liveではありません。実大会へ時刻を戻して実車運用した事実ではなく、実スロットル／PWM指令もログにないため、人間運転を上回った・実車で追従したという検証ではありません。

## 二層の入出力・目的・校正

| 層 | 入力 | 目的・制約 | 出力 | 校正状態 |
|---|---|---|---|---|
| 上位（30分ごと） | 現在距離・時刻、過去ログからの電池残量推定、次の既知Control Stopとその日の目的地、標高、当時利用可能なECMWF IFS予報 | 60–78 km/h候補を使い、17:00締切と15%暫定残量を満たす範囲で残り移動時間を最小化。ステージのネット電池Whは到達可能な予報時間内の最大値を制約に使用 | 次ステージの巡航目標速度と日末までの有限速度計画 | 混合整数線形計画。最適性は各行のsolver status / MIP gapに保存。有限候補集合で証明できるのはstatus=optimalの行だけ |
| 下位（5秒ごと） | 実ログ車速、上位目標速度、過去30分の速度変化履歴 | 30秒間一定加速度とする1変数MPC: `Σ(vₖ−vref)² + a² + 4(a−aprev)²`。60–78.2 km/h範囲と過去速度履歴からの加速度上下限 | 加速度要求、30秒先のモデル車速、機械的慣性電力増分`mva` | 凸二次目的の解析解。速度変化履歴は人間運転の代理値で、車両アクチュエータを校正していない。PWM／スロットル指令は出力しない |

**事後適合モデル:** 車両HV負荷は`67.6 + 0.918·rolling + 1.535·aero + 0.741·grade` W。これは10 km区間のStudent-t MLEであり、部品の真Crr/CdA/効率ではありません。PVは既存`faceted_body_poa`実効gain 1.068 m²（Day1–5の23時間ビン）を使用。日別ホールドアウトMAEは車両53.3 W（Day6は81.0 W）、PV50.5 Wです。電池は3011 Wh設計容量、選定33.164 Ah、実効パック抵抗0.488 Ω、最初の3分から選ぶ電圧・電流ロバストアンカーと以後のBattery_Watt積分で推定します。これは成分再生モデルに揃えた状態推定で、実SoC、セル温度、PWM→トルク/電力対応の校正値はありません。

## 予報ビンテージと時系列漏れ

- 使用したのはOpen-Meteo Single Runs APIのECMWF IFS単一ラン。各時刻で初期化から6時間以上たった最新ランのみを選び、気温・風・気圧・瞬時日射を取得しました。ルート上50 km間隔の地点を予報空間補間し、72時間先まで使います。
- 予報ラン初期時刻と利用可能時刻に関する未来漏れは判断94回で0件でした。一方、車両/PV係数は全Day1–5ログから後付け適合しており、初期日の係数学習に未来データが入る後知恵バイアスは残ります。予報履歴の因果性と係数学習の因果性を区別してください。
- ユーザーの指示に従い、nested day-held-outで選んだ過去6ブロック中央値の因果残差補正（±100 W）を、次の5分だけ最初のルートステージ電力量に適用しました。30秒以降で物理基準を改善しない5秒残差モデルや、全データ学習内予測は上位ホライズンに再帰適用していません。天候は計画の到達時間幅内で最大消費側を使います。既存の車両Day-LODO MAE 53.3 W、Day 6 81.0 W、PV日別MAE 50.5 Wの残差は残るため、計画達成は実車完走保証ではありません。
- 判断行の予報経過時間は中央値 9.0 h、範囲 6.0–11.9 h、6時間未満の違反 0 件です。リプレイCSVの`forecast_run_init_utc`を各出力の監査に使えます。
- 将来の実測気象を上位計画へ入れず、現在の予報ランの中から候補到達時刻全体を評価しました。トラブル停止は発生した時点で下位出力を止めます。将来の故障・道路工事を事前知識にはしません。Control Stopと17時締切はスケジュール制約として扱います。
- 初期SoCは各日の最初の3分から、電圧70–115 Vかつ|電流|≤5 Aを満たす中央サンプルを選び、実効抵抗0.488 Ωで負荷電圧を補正して電圧-SOC表に照合します。以後はBattery_Wattを積分しました。SOC依存抵抗マップはDay6ホールドアウトで定抵抗より悪化したため主推定には採用せず、別候補として保持します。Day 6の3,485秒欠測区間は積算から除外しています。OCVの温度・休止依存が未校正なので、このSoCは概算です。

## 日別シャドー出力と実績

| 日 | ログ行 | 上位解き直し | 実走距離 km | 人間ログ速度中央値 km/h | MPC目標中央値 km/h | 予備制約が実行不能な判断 | 実測電池エネルギー変化 Wh |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7,160 | 19 | 535.4 | 72.0 | 72.5 | 11 | 902 |
| 2 | 8,117 | 7 | 539.0 | 71.0 | 71.5 | 1 | 531 |
| 3 | 7,868 | 19 | 528.3 | 70.0 | 74.0 | 0 | -102 |
| 4 | 8,090 | 17 | 483.8 | 67.0 | 60.0 | 5 | -592 |
| 5 | 7,642 | 19 | 476.5 | 70.0 | 77.0 | 0 | 177 |
| 6 | 5,086 | 13 | 269.0 | 68.0 | nan | 13 | -433 |

実測列は人間運転時の結果、MPC列は当時の予報に基づく提案です。MPC速度を車両へ適用した場合の時間・電池消費は測定されていません。人間ログ速度中央値より目標速度が低い日もあり、全日で人間を上回る根拠は得られていません。モデルの10 km日別ホールドアウト電力MAEは平均53.3 W、Day 6は81.0 Wで、隣接速度の選択差を保証する精度ではありません。94回中、51回はMILP status=optimal、13回は時間制限付き実行可能解（最大相対ギャップ 3.9e-05）、30回は元のハード制約を満たす解が得られず、速度助言を出しません（infeasible_hard_constraints: 30）。その判断では、最小エネルギー診断でも締切超過が確認された回数は22、最短時間診断でも残量不足が確認された回数は25です。残量スラック付きPhase-I診断が解けた回数は22、その不足量中央値は311 Whです。Phase-I解は不足Whの診断に限り、指令候補には使いません。

日別の実測電池エネルギー変化は、連続ログ時刻差が30秒以内の区間のみ積分しています。Day1–5は独立の電力量整理表と0.01 Wh以内で一致します。Day6は3,485秒のログ欠落があり、その前後サンプルを台形積分でつなぐと未観測区間を作ってしまうため、リプレイでは除外しています。

## 現時点の結論

上位層の最適性は各行の`upper_solver_status`と`upper_mip_gap`で判定します。`optimal_finite_grid_milp`行だけが指定した有限問題の全体最適です。時間制限時も元のハード制約を通過したincumbentだけが助言対象です。制約不能、タイムアウトでincumbentなし、またはモデル/予報入力不正の判断は目標速度を空欄にし、CSVに`upper_no_advisory_reason`と、最小エネルギー/最短時間プロファイルの残量不足・締切超過診断を記録します。これらは「実車がその時点で走行不能だった」という意味ではなく、入力推定と指定した15%予備残量・17:00締切・速度候補集合の下で、モデルが両制約を満たす計画を証明できなかったという意味です。速度を連続値にした場合、実際の気象・故障、未同定のバッテリー・モーター・アクチュエータを含む実車全体の大域最適や人間を上回る結果は保証できません。実走行ログから実車閉ループ追従を検証できないため、出力はGrafana候補のシャドー提案です。

事故、パンク、MPPT故障、退役などは現在行のイベントフラグが立った時点で助言を停止します。記録された実際の発生時刻・位置は人間運転軌跡上のものです。速度変更後にも発生時刻・位置が不変だと物理シミュレーションしたわけではなく、実際の未来気象を測ったデータもありません。したがって「同じ事故・天候の下でMPCが実際に人間を上回る」反実仮想走行は今回のログから検証できていません。

残量の合意が未確定なので15%を暫定既定値としました。0/10/15/20%感度表を作成しています。計画不能となった区間は最小速度などの代替速度を助言せず、`upper_solver_status`と診断欄に原因を記録します。Phase-Iは残量不足量を推定しますが、締切制約は維持します。これは完走可能性の証明ではなく、競技運用責任者が状態推定・残量目標・締切/停止条件を再評価するための診断です。

初回判断での目標残量感度（0/10/15/20%）は次の通りです。これは残量値だけを変えた同一状態の再求解であり、別々の走行結果ではありません。

| 日 | 終端残量 | 次区間目標 km/h | 解の状態 | 予測残り時間 h | 予測ネット電力量 Wh |
|---:|---:|---:|---|---:|---:|
| 1 | 0% | 78 | feasible_time_limited_milp_not_proven_optimal | 8.30 | 2899 |
| 1 | 10% | 78 | feasible_time_limited_milp_not_proven_optimal | 8.47 | 2597 |
| 1 | 15% | 77 | optimal_finite_grid_milp | 8.57 | 2447 |
| 1 | 20% | nan | infeasible_hard_constraints | nan | nan |
| 2 | 0% | 78 | optimal_finite_grid_milp | 8.59 | 2405 |
| 2 | 10% | 77 | feasible_time_limited_milp_not_proven_optimal | 8.77 | 2104 |
| 2 | 15% | nan | infeasible_hard_constraints | nan | nan |
| 2 | 20% | nan | infeasible_hard_constraints | nan | nan |
| 3 | 0% | 78 | feasible_time_limited_milp_not_proven_optimal | 8.30 | 1801 |
| 3 | 10% | 76 | optimal_finite_grid_milp | 8.50 | 1500 |
| 3 | 15% | 76 | feasible_time_limited_milp_not_proven_optimal | 8.60 | 1350 |
| 3 | 20% | 75 | feasible_time_limited_milp_not_proven_optimal | 8.71 | 1199 |
| 4 | 0% | 67 | optimal_finite_grid_milp | 8.46 | 1432 |
| 4 | 10% | nan | infeasible_hard_constraints | nan | nan |
| 4 | 15% | nan | infeasible_hard_constraints | nan | nan |
| 4 | 20% | nan | infeasible_hard_constraints | nan | nan |
| 5 | 0% | 77 | optimal_finite_grid_milp | 7.24 | 2251 |
| 5 | 10% | 74 | optimal_finite_grid_milp | 7.40 | 1950 |
| 5 | 15% | 73 | optimal_finite_grid_milp | 7.48 | 1800 |
| 5 | 20% | 72 | optimal_finite_grid_milp | 7.57 | 1649 |
| 6 | 0% | nan | infeasible_hard_constraints | nan | nan |
| 6 | 10% | nan | infeasible_hard_constraints | nan | nan |
| 6 | 15% | nan | infeasible_hard_constraints | nan | nan |
| 6 | 20% | nan | infeasible_hard_constraints | nan | nan |

## 再現コマンドと出力

`python scripts/replay_hierarchical_mpc_2025.py`
`python scripts/evaluate_mpc_effective_intervals.py`

- 行ごとの提案と予報ラン監査: `data/processed/mpc_2025_as_live_replay.csv`
- 日別集計: `reports/mpc_2025_day_comparison.csv`
- 初回終端残量感度: `reports/mpc_2025_reserve_sensitivity.csv`
- 数値メタデータ: `reports/mpc_2025_replay_summary.json`
- 図: `reports/hierarchical_mpc_2025_replay.png`
- イベントなし・上位最適解・残量余裕ありの同一距離区間比較: [`mpc_2025_effective_interval_assessment.md`](mpc_2025_effective_interval_assessment.md)。正の時間差は、下位加速度モデルを含む予測上の区間短縮を示します。実車性能の実測ではありません。
- 区間別時間・全残差モデルの電力誤差: `reports/mpc_2025_effective_intervals.csv`
- 各モデルの適用先・検証状態: `reports/mpc_2025_model_usage_audit.csv`

## 根拠と限界

Open-MeteoはECMWF IFS HRESの単一初期化ランを過去に遡って指定でき、同じ予報の履歴再現にはHistorical Forecastの連続接続データよりSingle Runsが適します。予報ランが公開されるまでに数時間の遅れがあるため、ここでは保守的に6時間を置きました。詳細は[Single Runs API](https://open-meteo.com/en/docs/single-runs-api)と[Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api)。上位計画は各距離ステージで速度候補を選ぶ線形制約付き混合整数問題として[scipy.optimize.milp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)で解き、status=0を最適解、status=1を時間制限として扱います。

車両出力・PV出力の係数と電池容量の値は既存の[車両モデル要約](vehicle_power_model_summary.md)、[PVモデル要約](pv_fit_summary.md)、[モデル複雑度判断](model_complexity_decisions.md)を使用しています。BWSC2025規則にある60 km/h最低速度、Control Stop、17:00道路走行締切は[規則読了メモ](BWSC2025_regulatory_context.md)に保存済みです。MPCの再計画は制約付き最適化の標準構成に沿いますが、MPC文献はYATAの未測定アクチュエータ・誤差を同定するものではありません。
