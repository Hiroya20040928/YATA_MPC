# YATA階層型MPCの設計判断と現行プログラム一式

**基準日:** 2026-10-03  
**対象:** BWSC2025ログによるシャドー検証、BWSC2027運用へ向けた制御器設計

## 要約

YATAに適した二層構成は、**上位をレース時間・エネルギーの計画器、下位を短周期の速度追従器**とし、上位が下位の実際の追従能力と誤差幅を制約として受け取る方式である。ドライバーが最終判断をする運用では、下位層の出力は当面、速度・加減速の助言とし、PWM指令には接続しない。

現行実装には、上位の有限速度格子MILPと下位の一変数加速度モデル、BWSC2025全6日ログのシャドー再生、Grafana表示までがある。ただし、下位モデルは実車アクチュエータの校正を受けていない。したがって「設定した有限最適化問題の最適解」は一部で証明できても、「実車で最高性能」「人間を上回る」「全パラメータを含む大域最適」は現時点で主張できない。

BWSCを対象にした2026年の査読論文は、ルール由来の時空間制約を含む長距離ソーラー車計画を最適制御として解き、実レースのテレメトリとも比較している。著者らは、シミュレーションとテレメトリの主な不一致を開ループ計画とし、実レース中の予報・状態に応じた縮小ホライズン再計画と、日射・速度制限の不確かさ導入を改善方向に挙げている。[Montalto, Murgovski & Jarebrant (2026)](https://doi.org/10.1016/j.conengprac.2026.106964), [公開全文](https://research.chalmers.se/publication/551615/file/551615_Fulltext.pdf)

## 推奨する階層構成

```mermaid
flowchart LR
  A[ルート・規則・予報・停止予定] --> U[上位: レース計画MPC]
  S[距離・時刻・電池状態推定] --> U
  U -->|速度軌道・エネルギー予算・許容帯| L[下位: 速度追従MPC]
  T[実測速度・電圧・電力・イベント] --> L
  L -->|追従誤差・到達可能範囲| U
  L --> H[人間向け助言 / Grafana]
  H --> D[ドライバーとチームの最終判断]
```

### 層ごとの入出力、目的、校正状況

| 層 | 入力 | 目標・目的 | 出力 | 周期・ホライズン | 現状と校正 |
|---|---|---|---|---|---|
| 上位: レース計画 | 位置・距離、時刻、推定電池エネルギーと不確かさ、公式ルート/勾配、Control Stopと日末締切、利用可能時点を守った天気予報、イベント/道路情報 | まず規則・走行可能性・終端残量を満たし、その中で完走時間を最小化する。日射、負荷、予報の不確かさを安全余裕として扱う | 距離に対する速度軌道または目標帯、区間別エネルギー予算、推定残量の下限、求解状態と信頼度 | 現行は30分ごと、区間候補は60–78 km/hを1 km/h刻み、日末までの計画 | 有限格子MILP。17:00締切と仮の15%残量を使用。BWSC2025の予報ラン因果性は監査済みだが、係数は大会後に合わせたため完全な当時再現ではない。15%は安全認証値ではない。 |
| 下位: 速度追従 | 実測速度、上位の速度軌道/許容帯、路面勾配、HV電圧・電力、車両/ドライバー応答の直近履歴、イベント | 速度目標へ滑らかに近づきながら、加速度・加加速度・速度・電池/駆動系制約を満たす | 人間運用では速度/加減速の助言。自動化する場合のみ、校正済みアクチュエータへの指令と追従誤差 | 現行リプレイは5秒更新、30秒先、一つの一定加速度入力 | 現行はログ上の加速度履歴から作った代理制約で、PWM/ECO→トルク/加速度の対応は未校正。実車閉ループではなく、PWM指令も生成しない。 |
| 層間契約 | 下位層が実現できる追従誤差・電力偏差と、それらの適用条件 | 上位は「下位が追従できる」範囲だけを計画し、下位は計画の許容帯と終端到達条件を守る | `|v_actual-v_ref|` の許容帯、加速度/電力上限、終端速度・距離の到達範囲、契約違反フラグ | 再計画時と各下位周期で更新 | まだ未実装。高性能化と安全な分離に必要な最優先の層間機能。 |

### 層間契約を設ける理由

上位が一定速巡航を計画しても、実車がその加速度・電力・速度を所定時間内に出せなければ、上位の電力量・到着時間予測は崩れる。階層MPCの研究では、上位の簡略モデルが下位制御器の精度や誤差範囲を取り込み、下位が守る追従契約・到達集合を制約に使うことで、層間の実現可能性を維持する構成が研究されている。[Koeln, Raghuraman & Hencey (2020)](https://doi.org/10.1016/j.automatica.2020.108817), [Kögel et al. (2025)](https://onlinelibrary.wiley.com/doi/10.1002/rnc.6808)。また、上位制御器と不確かな下位アクチュエータの間に誤差上限の契約を設ける考え方も報告されている。[Baethge et al. (2018)](https://www.merl.com/publications/TR2018-123)

YATAで契約を実測校正するまでは、下位層の到達能力を決め打ちせず、ドライバー助言として表示する。将来の自動指令化では、試験で得た速度追従誤差・電力偏差の範囲を上位の制約余裕へ反映する。

## 現行実装とBWSC2025リプレイ結果

### 実装済み

- `scripts/replay_hierarchical_mpc_2025.py` は、30分周期の上位MILPと、5秒周期・30秒ホライズンの下位加速度最適化を実装する。
- 上位候補は60–78 km/hの19点。目的は残り時間の短縮で、予報時間、ルートステージ、電池エネルギー、日末17:00締切と仮15%残量を扱う。
- Open-Meteo ECMWF IFS単一ランは初期化から6時間以上経過したものに限定。予報ランの利用可能時刻違反は0/94判断。車両/PV係数はDay1–5ログから事後適合しているので、係数選択に関する未来情報は残る。
- 車両負荷モデルのDay1–5日別ホールドアウトMAEは53.3 W、Day6は81.0 W。PVモデルの日別MAEは50.5 W。これらは現在のモデルと記録粒度に対する誤差で、隣接速度候補の優劣や将来大会の性能を保証しない。
- 電池状態は3011 Whの設計容量を上限に、33.164 Ahと0.488 Ωの選定値、各日の開始時の電圧・電流アンカー、以後のBattery_Watt積分で推定する。実測BMS SoCやセル温度ではない。
- 上位の修正後リプレイは94判断中64回で元の残量/締切制約を満たす解があり、30回は目標速度なし。2秒MIP上限によってoptimalと時間制限付き実行可能の内訳は近接実行で50–51回対13–14回と変動する。optimalの証明は離散候補の数理問題内に限る。実行不能行には速度を返さない。
- 速度変更の影響まで評価できた同一始終点区間は62区間・1241.2 km。下位加速度モデルで推定した所要時間は人間の17.93 hに対し17.60 h、差は名目約20分。39/62区間で短縮、加速度上限0.5/1/1.5倍の全条件で短縮した区間は37/62。Day4単独では推定約28.5分遅く、日別の符号も揃わない。

### 結果の読み方

区間比較は実車にMPCを適用した結果ではない。人間走行ログから作った加速度上限を車両応答の代わりに使い、上位速度目標を人間ログの時刻列に重ねている。速度変更後の実到着時刻に合わせて天気、停止、故障の発生時刻・位置を再計算していない。したがって、約20分は**選択した区間と仮定内のシミュレーション差**であって、全大会の短縮実績や人間への勝利ではない。

## 1 Hzテレメトリ受け口と実運用向け設定

今回、既存の5秒/30分歴史リプレイとは分けて、1秒テレメトリを受け取り、最新距離・時刻の予報と既存フィットモデルから電力予測を更新し、上位速度軌道をもとに下位助言を計算するローカルAPIを追加した。

| 設定 | 採択値 | 根拠と適用範囲 |
|---|---:|---|
| テレメトリ受信 | 1 Hz | ユーザーが将来受信可能と指定。時刻・連番が新しい1秒サンプルのみ助言に利用。2.5秒超古い状態、GPS精度>50m、センサ品質不良、HV電力不整合、予報の時空間欠損ではnot-ready。 |
| 下位MPC更新 | 1 s | 108候補のnested day-blocked比較では5 sがわずかに最良。1 sは最新状態を毎秒最適化する要件を守るため採用。選択した他パラメータ固定では、5 sの20.396分に対して1 sの20.125分で差0.271分（17.93 hの0.03%未満）。5秒ログから1秒実性能を検証した値ではない。 |
| 下位予測時間 | 20 s | 5日を外側ホールドアウトにした内側選択で一貫して選ばれた値。係数はこのスカラー加速度追従モデル内の助言パラメータ。実際のアクチュエータ時定数は未同定。 |
| 下位速度助言刻み | 1 km/h | 2025ログの速度量子化が約1 km/hであり、下位層の1/2 km/h候補はnested day-blocked比較で1 km/h設定を選択した。 |
| 下位加減速包絡 | データ側q95とM2096物理上限の小さい方 | 314 kg、558.4 mmタイヤ、理論Crr/CdAを使いECO_LO DUTY=1のトルク曲線を平坦路の上限に換算。そこへ他日学習のq95速度ランプ分位点を課す。減速はログ分位点のみ。速度範囲外のベンチ値は端点外挿フラグ付き。 |
| 上位MPC更新 | 300 s＋イベント駆動 | 最初の各日60分を固定ログ状態から再求解し、5/10/15/30/60分を比較。5分はp95求解2.73 s・32回の目標変更・24/68件no-advisory。短周期がレース時間を縮める因果効果は未検証。 |
| 上位MILP求解上限 | 5 s | 2秒で実行可能・最適性未証明の18判断を5秒で再求解し、16件optimal・2件time-limited feasible・0件no-advisory。ターゲット変更0件、予測時間改善中央値0分。関数全体p95 5.76 s、最大5.79 s。ライブ上位solverへ接続済み。 |
| 上位速度格子 | 1 km/h | 10 km距離stage・60–78 km/h候補を5秒solver上限で比較。0.5 km/h格子は最適1/6・時間制限2/6・no-advisory 3/6、1 km/h格子は最適3/6・no-advisory 3/6。実行可能な日での予測残り時間差は最大約0.38秒/日。0.5格子のp95求解5.52秒に対し1 km/h格子は3.49秒。1 km/hを暫定維持。 |

下位設定の比較は62区間・1241.18 km・人間17.93 hを対象とし、更新周期、20/30/45秒先読み、ログ加速度分位点0.75/0.90/0.95、速度刻み1/2 km/h、3組の制御重みから108候補を評価した。外側日を完全に選択から外すnested 5-fold day-blocked検証で、内側は5 foldすべてq95・20秒・1 km/h・重み(10,16)を選んだ。選択候補の外側時間差はDay1 +1.91分、Day2 +1.75分、Day3 +17.66分、Day4 −27.54分、Day5 +26.62分で、1秒実運用設定の交差適合合計は+20.13分、4/5日で短縮・Day4で延長だった。損失日が残るため、これを全大会の短縮保証や人間に対する勝利とは解釈できない。

### 推定アクチュエータ応答の扱い

ユーザー許可に基づき、設計質量下限314 kg、タイヤ半径0.2792 m、理論CdA/Crrと製品ECO_LOトルク曲線から正味加速度の物理上限を逆算し、5秒速度差分から得た人間＋車両の加減速分位点で抑えた。q95全日値は+0.1667/−0.1667 m/s²。ベンチ物理上限は約70.9 km/hで0.36 m/s²、78.1 km/hで0.054 m/s²であり、高速域ではモーター曲線が制約になる。

これはPWM/ECO指令に対する真のアクチュエータ応答ではない。製品試験は固定DUTYのECO_LO曲線で、搭載PWM+ECO設定との同一性、ログ上の指令、実モータトルク、制御遅れは記録されていない。速度差分は1 km/h単位・5秒間隔で量子化されているため、時定数/加加速度限界を推定可能と偽らず、実効到達包絡の助言上限として使う。

### ランタイムAPIの責務と境界

- `POST /v1/telemetry` は毎秒の車速、GPS/ルート距離、HV電圧・電流・符号付きバッテリー電力、MPPT合算PV電力、イベントを受け付ける。既存の25S電圧-SoC表、0.488 Ω、3011 WhとBattery_Watt積分から不確かさ付き電池エネルギーも逐次更新する。BMS SoCとは呼ばない。
- `POST /v1/forecast/route` はOpen-Meteo等のroute-distance×valid-time予報格子を受け付ける。GETごとに現在の距離・時刻へ空間/時間補間し、風向は円周補間する。外部天気APIを毎秒呼び出す設計ではない。
- `GET /v1/controller/input` は最新1秒状態と現在位置・時刻に対応する予報、既存の2025フィット車両負荷/PV予測、バッテリー電力残差、推定エネルギー/不確かさを返す。上位層はエネルギー点推定でなく保守的な`energy_lower_2sigma_wh`を使う契約にした。
- `POST /v1/race-schedule`で当日端点・締切・停止所要時間を受け、`GET /v1/controller/advice`が最新の状態/電池/route-weatherを使い、300秒周期/イベント条件で上位MILPをバックグラウンド再求解する。`GET`は新しいsequenceごとに1秒/20秒下位MPCを一度実行する。同じsequenceの再GETはキャッシュする。
- `127.0.0.1:8125`で、短距離の合成ルート・予報・電池入力から上位最適解→下位助言までの統合スモークを確認した。実レースの2027ルート/規則連携やネットワーク運用を検証した意味ではない。PWMやモータ指令は出さず、Grafanaの再生API(8123)とは別サービスである。
- `POST /v1/upper-plan`は同じ現行schedule_idに結び付けた外部計画の任意入力である。default routeはBWSC2025検証ルートのため、2027正式ルートCSV・route_id・各日の停止/締切を供給する必要がある。APIが上位MILPを自動再求解するようになったが、2027の公式スケジュール自体は未確定/未投入である。
- 上位層の速度格子・距離ステージ、更新周期、求解時間上限の比較と人間比較の制約は[`upper_mpc_tuning_and_human_comparison.md`](upper_mpc_tuning_and_human_comparison.md)に追加した。72件の格子感度、159件の周期プローブ、18件の5秒求解時間比較をCSV/JSONへ保存している。

関連仕様は [`telemetry_input_requirements_v1.md`](telemetry_input_requirements_v1.md)、下位チューニング全結果は [`mpc_tuning_and_actuator_inference.md`](mpc_tuning_and_actuator_inference.md)、108候補表は `mpc_lower_parameter_sweep_2025.csv`、日別外側foldは `mpc_lower_nested_cv_2025.csv`。

上位の有限候補最適化と、現実の車両全体の最適性は区別する。MILPのoptimalは離散格子の数理問題内の最適性を示すが、連続速度空間、誤差を含む車両、実際のドライバー/アクチュエータ、未観測イベントまで含む大域最適の証明ではない。下位の解析解も、仮定した一変数・一定加速度モデル内の解である。

## 性能向上の優先順位

1. **予測と計画の評価を因果化する。** 各日を丸ごと外す検証に加えて、判断時刻より後のログを使わない予報ビンテージと係数学習を揃える。報告する指標は日別/区間別の完走時間、残量違反、実行不能率、エネルギー誤差、予測区間被覆率。学習内誤差だけで速度を選ばない。
2. **上位と下位の到達可能性契約を強める。** 下位から速度/加速度/電力の到達可能範囲を返し、上位は追従誤差とエネルギー予測誤差を差し引いた余裕を計画する。トラブル、欠測、契約逸脱、実行不能、残量推定の信頼度不足では「推奨なし/人の判断」を明確にする。現行の離散MILP+下位加速度助言はハード制約違反候補を出さないが、再帰的実行可能性や実車安全を保証する終端集合/終端方策は未実装。
3. **下位の実車応答を校正する。** ECO/PWM設定を含め、速度目標またはPWM変更、車速、HV電圧/電流、時刻、路面勾配を対応させる。停車状態の安全な試験と実走行の範囲内で追従遅れ、加速度、加加速度、消費電力を推定し、日/区間単位の未使用データで検証する。既存M2096ベンチ曲線は、実装設定との同等性が未確認のため代替の閉ループ校正にはならない。
4. **予報・PV・電池の不確かさを上位へ入れる。** 当面は検証ログから得た外側誤差帯で保守制約を置く。将来、複数予報/シナリオが揃えば、日射・風・温度とバッテリー初期状態の複数シナリオに対して終端残量・締切を満たすシナリオMPCを比較する。誤差分布が検証される前に、確率制約の信頼水準を作らない。
5. **解法高度化は物理/予測精度と分けて判断する。** 現在のMILPは有限候補の基準解として保持する。より細かい連続速度最適化や、BWSCの時空間停車制約を含む多重射撃法は比較候補だが、モデル誤差が速度候補差より大きい現在、探索解像度だけを上げても選択の信頼性は上がらない。2026年BWSC論文は、ローリング再計画と天気・速度制限の不確かさを次段階として示している。[公開全文](https://research.chalmers.se/publication/551615/file/551615_Fulltext.pdf)
6. **シャドー運用後に人間判断支援へ進む。** 2025の同一ログ再生では、実測、人間速度、MPC提案、予測幅、解状態、イベント、推定残量を同時に記録する。2027規則・大会情報は更新版を適用し、周回/距離・停止制約を含む全行再生で候補比較した後、まず提案表示のみで運用する。自動速度指令は追従契約、異常時停止、運転者手順が確認できるまで有効化しない。

この分割は既往研究と整合する。太陽車レースの先行研究は、速度プロファイル、電力/電池制約、ドライバーが実行できる加速助言を一緒に扱う。[Atmaca (2015)](https://journals.tubitak.gov.tr/elektrik/vol23/iss4/16/)。接続型PHEVの階層エコドライブ研究は、上位の速度計画と下位の車両エネルギー管理を分け、計算負荷を抑える方式を検討しているが、YATAのモーター/電池定数や効果量を提供するものではない。[Xue et al. (2023)](https://doi.org/10.1016/j.energy.2023.129058)。

## 現行プログラムと成果物

### 実行可能な主処理

| 用途 | 主ファイル | 出力/役割 |
|---|---|---|
| 任意地点の予報取得 | [`openmeteo_fetcher.py`](../openmeteo_fetcher.py) | Open-Meteo応答を保存。広範な現在/時間/日次/15分変数を要求する単体ツール。Grafana/MPCへの直接出力はまだしない。 |
| BWSCログ統合・距離・天気 | `scripts/prepare_race_log.py`, `scripts/fetch_weather.py`, `scripts/reconcile_energy_balance.py` | `race_days_merged.csv`, 距離/座標/イベント付きログ、Open-Meteo天気付き全行CSV、エネルギー収支照合。提供済み43,963行。 |
| PV・車両・電池モデル | `scripts/fit_pv_power_model_physical.py`, `scripts/fit_vehicle_energy_segments.py`, `scripts/refine_component_models.py` | PV/車両/電池推定、日別検証、成分別5秒リプレイ。 |
| 二層MPCリプレイ | [`scripts/replay_hierarchical_mpc_2025.py`](../scripts/replay_hierarchical_mpc_2025.py) | 2025時系列シャドー提案 `data/processed/mpc_2025_as_live_replay.csv`、JSON集計、比較図、Markdownレポート。 |
| MPC有効区間比較 | `scripts/evaluate_mpc_effective_intervals.py` | イベントを除いた同一区間のモデル時間比較CSV/JSON/Markdown。 |
| MPCパラメータnested検証 | `scripts/tune_mpc_parameters_2025.py` | 108設定の日単位nested検証、`mpc_operating_config.json`、M2096×ログ応答包絡を生成。 |
| 上位MPC周期プローブ | `scripts/benchmark_upper_mpc_replan_period_2025.py` | 周期5/10/15/30/60分でログ初期1時間の状態を再求解（159 solves）。速度案変化とwall timeを比較。因果的なレース時間比較ではない。 |
| 上位MPC格子/ステージ感度 | `scripts/benchmark_upper_mpc_grid_sensitivity_2025.py`, `scripts/benchmark_upper_mpc_speed_grid_5s_2025.py` | 6日初回状態×12設定（72 solves）の速度格子/距離stage比較に加え、5秒上限で0.5/1 km/h格子を比較。全日閉ループ性能とは区別。 |
| 上位MILP求解時間感度 | `scripts/benchmark_upper_mpc_solver_budget_2025.py` | 2秒で未証明だった16状態を5秒で再求解し、有限格子内の最適性証明率と速度案変化を出力。 |
| ライブ入力/MPC下位助言 | `scripts/mpc_live_input_api.py`, `src/yata_mpc/telemetry.py`, `src/yata_mpc/controller.py` | 毎秒テレメトリ/route weather/上位計画を検証し、最新位置・時刻の予測と下位助言をJSON APIで返す。 |
| Grafana生成・API | `scripts/generate_grafana_dashboard.py`, `scripts/grafana_local_api.py` | 履歴リプレイの読み取り専用APIと23パネルのローカルダッシュボード。 |
| Grafana起動/停止 | `scripts/start_local_grafana.ps1`, `scripts/stop_local_grafana.ps1` | Grafana/APIを起動・停止。2026-10-03の確認時は両方ともloopbackで稼働。 |

### ディレクトリ構成と規模

| 場所 | 内容 |
|---|---|
| `openmeteo_fetcher.py` | 単地点の将来予報取得ツール。 |
| `scripts/` | Pythonスクリプト群とPowerShell 2本。入力監査、前処理、モデル同定、残差学習/比較、MPC、Grafana、ライブ入力、可視化ユーティリティ。 |
| `models/` | モデル/候補/校正台帳22件。CSV/JSONに加えてHGB等のjoblib、TCNのptを含む。最大適合モデルはMPCへ自動適用されず、診断/残差成果物として保持。 |
| `src/yata_mpc/`, `schemas/` | 1 Hz telemetry/forecast/upper-plan contract、位置時刻で補間する受け口、電池残量推定、下位MPCと上位再計画ゲート。 |
| `data/` | 685ファイル。元ログ、公式ルート/アンカー、天気API応答、日別/全行加工ログ、モデル出力、リプレイ。 |
| `reports/` | 154ファイル。データ品質、規則、PV/車両/電池、残差候補、MPC、Grafanaのレポート/CSV/JSON/画像。 |
| `references/` | 283ファイル。受領資料とアーカイブ展開資料。分析上の入力資料であり、プログラム実行命令ではない。 |
| `grafana/` | JSONダッシュボードとdatasource/dashboard provisioning設定4ファイル。 |
| `notebooks/` | `mpc_2025_shadow_replay_audit.ipynb` 1冊。シャドー再生の監査・閲覧用。 |
| `requirements.txt` | pandas, numpy, scipy, matplotlib, openpyxl, scikit-learn, xgboost, catboost。 |

### `scripts/` の全Pythonファイル

<details>
<summary>ファイル名一覧（Python 59本、PowerShell 2本）</summary>

**入力監査・ログ/ルート/天気前処理**

`audit_attached_archive.py`, `catalog_archive_workbooks.py`, `extract_archive_documents.py`, `extract_new_pdf_text.py`, `inspect_new_sources.py`, `render_scanned_pdfs.py`, `summarize_archive_data.py`, `prepare_race_log.py`, `fetch_weather.py`, `reconcile_energy_balance.py`, `summarize_field_irradiance.py`

**物理モデル・成分推定・診断**

`compare_motor_bench_to_race.py`, `evaluate_battery_soc_resistance_candidate.py`, `evaluate_inertial_vehicle_candidate.py`, `evaluate_vehicle_model_candidates.py`, `fit_pv_model.py`, `fit_pv_power_model_physical.py`, `fit_vehicle_energy_segments.py`, `fit_vehicle_power_model.py`, `refine_component_models.py`, `plot_battery_soc_resistance.py`, `plot_speed_power_prediction.py`, `diagnose_vehicle_transient_residuals.py`

**残差モデルの学習・比較・監査**

`audit_horizon_residual_blends.py`, `audit_residual_learning_outputs.py`, `audit_residual_oracle_identity.py`, `compare_residual_models_common_test.py`, `evaluate_5s_residual_horizon.py`, `evaluate_crossday_residual_models.py`, `evaluate_multihorizon_arx.py`, `evaluate_multihorizon_prequential_5s.py`, `evaluate_online_kernel_distance_blend.py`, `evaluate_online_kernel_distance.py`, `evaluate_online_kernel_residual.py`, `evaluate_online_residual_filters.py`, `evaluate_online_tree_residual.py`, `evaluate_prequential_5s_arx.py`, `evaluate_prequential_5s_blend.py`, `evaluate_prequential_5s_residual_model.py`, `evaluate_prequential_inertial_residual.py`, `evaluate_residual_daywise_selector.py`, `evaluate_residual_gp_online_hybrid.py`, `evaluate_residual_meta_stacking.py`, `evaluate_residual_method_sweep.py`, `evaluate_residual_stacking.py`, `evaluate_residual_tcn.py`, `evaluate_tree_multihorizon.py`, `evaluate_vehicle_residual_learning.py`

**MPC・Grafana・監査ノート**

`build_mpc_replay_audit_notebook.py`, `evaluate_mpc_effective_intervals.py`, `tune_mpc_parameters_2025.py`, `benchmark_upper_mpc_grid_sensitivity_2025.py`, `benchmark_upper_mpc_speed_grid_5s_2025.py`, `benchmark_upper_mpc_replan_period_2025.py`, `benchmark_upper_mpc_solver_budget_2025.py`, `replay_hierarchical_mpc_2025.py`, `mpc_live_input_api.py`, `generate_grafana_dashboard.py`, `grafana_local_api.py`, `start_local_grafana.ps1`, `stop_local_grafana.ps1`

</details>

### 再実行コマンド

依存パッケージを導入した環境でプロジェクト直下から実行する。

```powershell
python -m pip install -r requirements.txt
python scripts/replay_hierarchical_mpc_2025.py
python scripts/evaluate_mpc_effective_intervals.py
powershell -ExecutionPolicy Bypass -File .\scripts\start_local_grafana.ps1
```

Grafanaの既定URLは [http://127.0.0.1:3000/d/yata-bwsc2025-local/yata-mpc-bwsc2025-shadow-operations](http://127.0.0.1:3000/d/yata-bwsc2025-local/yata-mpc-bwsc2025-shadow-operations)、リプレイAPIのhealthは `http://127.0.0.1:8123/api/health`。APIとGrafanaはloopback限定で、現行ダッシュボードはBWSC2025ログの再生であり実車ライブ入力ではない。OpenStreetMapの背景タイルだけはインターネットを利用する。起動時に `start_local_grafana.ps1` がAPI側の再生を開始する。

## 参考文献と適用範囲

1. Montalto, Murgovski & Jarebrant, “Optimal energy management under spatio-temporal constraints: An application to solar-powered vehicles,” *Control Engineering Practice*, 173 (2026), 106964. [DOI](https://doi.org/10.1016/j.conengprac.2026.106964), [公開全文](https://research.chalmers.se/publication/551615/file/551615_Fulltext.pdf). BWSCのルール制約、予報更新、開ループとの差、日射/速度制限不確かさの根拠。
2. Atmaca, “Energy management of solar car in circuit race,” *Turkish Journal of Electrical Engineering and Computer Sciences*, 23(4), 2015, 1142–1158. [論文ページ](https://journals.tubitak.gov.tr/elektrik/vol23/iss4/16/), [DOI](https://doi.org/10.3906/elk-1212-37). エネルギー制約つき速度プロファイルとドライバー向け操作量の先例。
3. Xue et al., “Predictive hierarchical eco-driving control involving speed planning and energy management for connected plug-in hybrid electric vehicles,” *Energy*, 283 (2023), 129058. [DOI](https://doi.org/10.1016/j.energy.2023.129058). 上位速度計画と下位エネルギー管理を分ける先例。プラグインハイブリッドの結果はYATAに直接転用しない。
4. Koeln, Raghuraman & Hencey, “Vertical hierarchical MPC for constrained linear systems,” *Automatica*, 113 (2020), 108817. [DOI](https://doi.org/10.1016/j.automatica.2020.108817). 長短周期層間で到達集合を連携し、制約を扱う理論的根拠。
5. Kögel et al., “Safe hierarchical model predictive control and planning for autonomous systems,” *International Journal of Robust and Nonlinear Control* (2025). [論文](https://onlinelibrary.wiley.com/doi/10.1002/rnc.6808). 下位の追従誤差契約を上位の計画制約へ返す構成。
6. Baethge et al., “Contract-based Predictive Control for Modularity in Hierarchical Systems,” IFAC NMPC (2018), 499–504. [MERL論文紹介](https://www.merl.com/publications/TR2018-123), [DOI](https://doi.org/10.1016/j.ifacol.2018.11.040). 未知/不確かなアクチュエータを上位計画が無視しないための誤差契約。
7. Picasso, De Vito, Scattolini & Colaneri, “An MPC approach to the design of two-layer hierarchical control systems,” *Automatica*, 46(5) (2010), 823–831. [DOI](https://doi.org/10.1016/j.automatica.2010.02.013). 下位追従が完全と仮定しない二層制御設計の先例。

論文は制御構造を選ぶ根拠であり、YATA車両の未同定定数、2025反実仮想、将来大会の最適速度・短縮時間を保証するものではない。
