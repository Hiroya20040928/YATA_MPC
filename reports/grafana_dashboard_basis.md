# ローカル Grafana 運用ダッシュボード: 構成根拠とデータ範囲

## 利用目的

BWSC2025 の記録ログを時刻順に再生し、実測値と二層MPCのシャドー提案を人が同時に確認できる Grafana 画面を作成した。表示を一画面の冒頭に集め、速度・目標速度・推定残量・バッテリー/PV電力を先に見せ、地図と制御器状態、警報、診断時系列、イベント記録へ順に進む構成とした。

この画面でいう「更新中」はローカル再生APIの更新であり、2025年大会当時の実時間ライブ配信ではない。2025年の大会ログ43,963行を5秒ごとに再生し、ログの累積距離を公式ルート再構成線へ投影する。速度やPV電力等は元ログ値、MPC目標やSOCは既存モデルの推定/提案値で、ラベルと説明でも区別する。車両アクチュエータには接続しない。

## 配置と更新周期の根拠

- Grafana Geomap の公式仕様は、緯度経度を持つデータをマーカーに表示でき、auto-refresh と合わせて移動体のリアルタイム位置追跡に使えるとしている。BWSCルートを固定 GeoJSON、車両位置を更新マーカー、公式ノート由来イベントを点として重ねた。
- Grafana Labs のライブ飛行機追跡例は、最後の位置だけを描き、5秒更新をデータ取得間隔に合わせる構成を示している。本画面は元ログの5秒間隔に揃えて5秒更新とした。再生周波数を後で実車の受信周期へ合わせる場合も、APIとダッシュボードの周期を揃える。
- 先頭のKPIカードは意思決定の一次情報として車速、MPC目標速度、推定残量、蓄電量、Battery_Watt、ZPのPV合算出力を並べ、その下に電圧/電流と気象を置く。続けて現在の上下位MPC出力・求解状態・予報runの時刻、警報判定、地図、時系列、記録イベントの順に配置した。
- 5秒時系列は直近1時間の最大720行を返す。InfinityデータソースでHTTP JSON APIを読み、時刻/数値列をGrafanaの表・時系列・Geomapへ渡す。Grafanaはファイルベースのデータソースとダッシュボードをprovisioningできるので、ローカルでも再起動後に設定を再現できる。
- ルートの固定線は100 m DEMルートを500 m間隔に間引いて描き、ダッシュボードの自動更新のたびに大きな座標配列を再取得しない。位置マーカーはAPIの現在行から毎回更新する。

## 指標の定義と出典

| 表示 | ソース/定義 | 解釈上の限界 |
|---|---|---|
| 実測速度、HV電圧、バッテリー電流、Battery_Watt、PV合算出力 | data/processed/mpc_2025_as_live_replay.csv の元ログ列 | 校正済み真値や符号方向が確認されたとは扱わない。Battery_Watt は符号付きの記録値をそのまま表示。 |
| MPC目標速度、MIP gap、上下位求解状態、加速度要求、予測残り時間/電力量 | BWSC2025 二層MPCのas-liveシャドー再生CSV | 実車指令ではない。時間制限解はhard constraint再検査後のみ助言対象。実行不能時は目標速度を出さず、診断欄に制約不足量を表示する。有限速度格子等の最適性範囲は reports/hierarchical_mpc_2025_replay.md を参照。 |
| 蓄電量 / 推定残量 | stored_energy_est_wh ÷ 理論設計容量3011 Wh | 電圧アンカーとBattery_Watt積分に基づくモデル推定。BMSのSOCやセル温度の実測ではない。15%は当面のモデル予約値。 |
| 外気温、瞬時短波放射、風速 | data/processed/full_race_component_replay_5s.csv のログ日時・距離に整合したOpen-Meteo履歴値 | 車載気象センサ値や将来予報の測定値ではない。PV面POA日射、モジュール温度も直接測定していない。 |
| 車両座標 | distance_from_darwin_m を data/input/route/route_100m_dem.csv の緯度経度へ距離補間 | 実車GPSトラックではなく、アンカー距離上のルート位置推定。 |
| ルートイベント | data/input/route/events.csv および100 m DEMルート | 実際の人間運転時の停止/トラブル記録であり、MPC速度へ変えた場合に同じ時刻・距離で発生すると予測したものではない。 |
| 警報判定 | 現在行のイベントフラグ、MPC状態、推定残量、予報経過時間からローカルAPIで分類 | 運用の注意喚起用。安全認証済み閾値ではなく、MPC出力を実行する許可ではない。 |

## 操作と制約

- 自動更新は5秒。Grafana上部の時間範囲で直近ログ再生時系列を確認できる。
- 地図の背景タイルはOpenStreetMapのオンライン配信を使う。APIと再生ログはローカルに残るが、ネットワークが無い場合は背景地図タイルが表示されない可能性がある。
- 既定 replay は1倍速、開始位置は0始まりindex 159。起動スクリプトへ --rate と --start-index を渡すと再生速度と開始点を変えられる。
- 実大会のライブ運用へ移すには、ローカルAPIの入力を実車テレメトリと現時点で利用できる気象予報/位置に差し替え、表示更新周期、欠測/通信断、センサ値の校正と符号、アラーム閾値、運転者手順を別途確認する。

## 参照

1. Grafana Labs, [Geomap visualization documentation](https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/visualizations/geomap/): 経緯度データ、更新位置の表示、マーカー/GeoJSON/ルートレイヤー。
2. Grafana Labs, [Track planes in real time with Grafana](https://grafana.com/blog/tracking-planes-with-grafana-in-real-time-how-to-visualize-the-aircraft-overhead-with-your-own-dashboard/): 最新位置だけの表示と5秒自動更新を使うライブ追跡例。
3. Grafana Labs, [Infinity data source: JSON](https://grafana.com/docs/plugins/yesoreyeram-infinity-datasource/latest/data-formats/json/), [Query editor](https://grafana.com/docs/plugins/yesoreyeram-infinity-datasource/latest/query/): HTTP JSON API、列型、表/時系列データの取り込み。
4. Grafana Labs, [Provision Grafana](https://grafana.com/docs/grafana/latest/administration/provisioning/): datasource と dashboard をローカルファイルで構成する方式。
5. Grafana Labs, [Install Grafana on Windows](https://grafana.com/docs/grafana/latest/setup-grafana/installation/windows/): Windows standalone ZIP実行方式。

## ローカル起動確認 (2026-10-02)

- Grafana 13.2.3 と Infinity datasource 4.1.0 を起動し、両ポートが 127.0.0.1 にだけ bind されていることを確認した。
- Grafana health API は database=ok を返し、provisioned dashboard UID yata-bwsc2025-local は23 panelを返した。
- 22本の Grafana datasource query を HTTP API 経由で実行し、すべて status 200 を確認した。統計カードは各1行、地図マーカーは1行、制御判断表は9行、イベント表は37行、時系列は再生開始後の191行が返った。時系列APIは最大720行まで直近データを返す。
- ルートGeoJSONは28 feature (Darwin–AdelaideのLineString 1本と、距離が置ける記録イベント27点) を返し、Grafana origin に対するCORS応答を確認した。
- この実行環境には操作可能なブラウザセッションが接続されていないため、最終的な画面上の読みやすさ・地図タイル描画の目視確認は未完了。ローカル画面URLは http://127.0.0.1:3000/d/yata-bwsc2025-local/yata-mpc-bwsc2025-shadow-operations。
