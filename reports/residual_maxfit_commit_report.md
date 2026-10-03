# BWSC2025 残差学習の最大適合結果

更新日: 2026-10-02  
対象: 提供済みBWSC2025 Day1–6ログ、全16,242行  
目的: 今あるログに許される範囲で残差補正を強く適合し、時系列を保った未使用区間で改善を点検する。

## 結論

ご指定に従い、過学習リスクを許容して全16,242行へ最終再フィットした。全データ上の最小MAEを優先する主モデルは `HistGradientBoosting / hgb_l15_i200` で、全ログへの学習内適合度は残差MAE **85.23 W**、RMSE **187.43 W**、P90絶対誤差 **177.76 W**。学習と同じ行を評価しているので、この数字は将来予測精度ではない。

全ログでの物理モデル残差MAEは231.73 Wから85.23 Wへ下がった（学習内で63.2%減）。

時系列検証を残した比較では、CatBoostメタモデルがDay1–5の未使用末尾区間で最良となり、MAE **99.70 W**、P90 **209.25 W**を記録した。HGB単独は **107.12 / 221.50 W**、検証で決めた凸ブレンドは **103.03 / 214.23 W**、物理モデル単独は **198.29 / 440.87 W**。ただしDay6ストレス区間ではCatBoostメタモデルのMAE **166.63 W**がHGBの **158.76 W**より悪い。よってHGB全データfitを主たる最大適合モデルとして保存し、CatBoostメタモデルは通常日の外側検証が良い比較候補として別保存した。

実測値のSolar_Watt + Battery_Wattをそのまま使うと残差は数値誤差の範囲でゼロになる。しかしそれは目的変数そのものを入力に戻す測定オラクルで、未来予測器ではない。実測を使わない予測モデルで残差ゼロには到達していない。

## 評価値

| データ区間・方式 | MAE (W) | RMSE (W) | P90絶対誤差 (W) | 読み方 |
|---|---:|---:|---:|---|
| Day1–5 最終20% 未使用・物理モデル | 198.29 | 314.57 | 440.87 | 2,744行、基準 |
| 同・HGB残差補正 | 107.12 | 200.52 | 221.50 | 物理モデル比でMAE約46%減 |
| 同・検証選択凸ブレンド | 103.03 | 199.20 | 214.23 | HGB/ARX/Treeを検証区間で混合 |
| 同・CatBoostメタモデル | **99.70** | **196.71** | **209.25** | 出力と速度・距離・風・過去残差などを再学習 |
| Day6最終20% 未使用・物理モデル | 359.55 | 499.62 | 767.25 | Day6は選択に使わないストレス評価 |
| 同・HGB残差補正 | **158.76** | **298.96** | **353.92** | Day6ではメタモデルより低誤差 |
| 同・CatBoostメタモデル | 166.63 | 308.83 | 366.43 | 基準から改善するがHGBを下回る |
| 全16,242行・HGB全データfit | **85.23** | 187.43 | 177.76 | in-sample、最大適合の主モデル |
| 全16,242行・CatBoostメタ全データfit | 85.45 | **187.36** | **176.08** | in-sample、HGBよりMAEは0.23 W高い |

CatBoostメタモデルはDay1, 2, 5でHGBを改善し、Day3, 4では悪化した。Day6も悪化したため、全条件へ一律に優越するとは見なさない。全日全行のHGB対メタモデルのMAE差は0.23 Wで、測定・モデル不確かさに比べてごく小さい。

## 探索・選択・過学習確認

比較した方式は、物理ベース、Ridge、HistGradientBoosting、ExtraTrees、RandomForest、XGBoost、CatBoost、MLP、TCN、ARX、過去残差median/EWMA/Kalman/Theil–Sen、逐次木、オンラインGP/KRR、日別選択、凸制約ブレンド、メタ学習である。17構成の表形式モデル探索に加え、出力同士と走行コンテキストから残差を再学習するメタモデルを追加した。

時間順を維持し、各日の先頭60%で候補を学習、続く20%で選択、末尾20%は外側評価用に留保した。メタモデルはさらに中間20%を時系列順に70/30へ分け、前半で学習、後半でCatBoost等の設定を選び、末尾20%を一度だけ評価した。Day6は設定選択に使わず別ストレス評価にした。この分割は同じレース区間内での次時点予測を調べる設計で、別年・別ルートへの独立検証ではない。

Day1–5外側評価で、試した補正はいずれも物理モデルから大きく改善した。CatBoostメタはHGB単独よりMAEを7.42 W下げたが、日別には一貫せず、Day6でも悪化した。中央値バイアス補正は外側MAEを改善しなかった。ニューラルTCNは同じ外側区間でMAE151.21 Wに留まり、HGB系より劣った。これ以上、自由度を増やすだけでは外側性能改善が確認できず、現データで採用できる追加補正はここまでと判断した。

時系列ブロック検証を使ったのは、未来の行を学習に混ぜないためである。時間予測のrolling-origin/prequential評価とリーク回避の議論は[Cerqueira, Torgo & Mozetič (2022)](https://link.springer.com/article/10.1007/s10618-022-00894-5)にまとめられている。候補には勾配ブースティングの代表的手法である[XGBoost (Chen & Guestrin, 2016)](https://doi.org/10.1145/2939672.2939785)と、予測shift軽減を狙うordered boostingを提案した[CatBoost (Prokhorenkova et al., 2018)](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html)を含めた。系列畳み込みの[TCN評価 (Bai, Kolter & Koltun, 2018)](https://arxiv.org/abs/1803.01271)も候補として実測比較した。これらの論文は手法選定の根拠であり、YATAでの結果は本ログの評価値である。

## MPCへ使うときの解釈

上の外側評価は5秒先の一段条件付き推定で、当該時点の速度、ルート、アーカイブ気象を既知としている。MPCが提案する未来速度やOpen-Meteo予報の誤差は含まない。先読み評価ではARX補正が5秒先では有効でも、30秒先で改善が小さく、60秒・300秒先では物理モデルに届かない結果が既に出ている（[継続ホライズン評価](residual_learning_progress_and_mpc_readiness.md)）。

物理モデルにデータ駆動残差を足す方式や、誤差を使う学習MPCには先行研究がある（[EVハイブリッド消費モデル](https://arxiv.org/abs/2405.10984)、[LMPC for racing](https://doi.org/10.3929/ethz-b-000351561)、[オンライン学習MPC](https://doi.org/10.1016/j.robot.2023.104469)）。ただし、そこからYATAの速度最適化精度を保証することはできない。

したがって、今回の残差モデルは2025ログの再構成・シャドーMPC評価へコミットする。これをもって最適巡航速度の精度が確定したとは扱わない。特に速度選択に必要な電力差は残差誤差に比べ小さいため、速度指令へ進む前に、複数ホライズン・予報入力・候補速度順位の安定性を同じ分割で別途検証する必要がある。

## 成果物

- 全データ主モデル・行別残差: [`full_race_residual_maxfit_5s.csv`](../data/processed/full_race_residual_maxfit_5s.csv)
- 主モデル: [`vehicle_residual_maxfit.joblib`](../models/vehicle_residual_maxfit.joblib)
- 主モデルの特徴量・条件: [`vehicle_residual_maxfit_manifest.json`](../models/vehicle_residual_maxfit_manifest.json)
- 外側テスト全手法比較: [`residual_common_test_comparison.json`](residual_common_test_comparison.json)
- メタモデルの外側予測・評価: [`residual_meta_stacking_test.csv`](residual_meta_stacking_test.csv), [`residual_meta_stacking_summary.json`](residual_meta_stacking_summary.json)
- 全データfitメタモデルの行別比較: [`full_race_residual_risk_committed_5s.csv`](../data/processed/full_race_residual_risk_committed_5s.csv)
- メタモデルartifact: [`vehicle_residual_meta_maxfit.joblib`](../models/vehicle_residual_meta_maxfit.joblib)
- 再実行スクリプト: [`evaluate_residual_method_sweep.py`](../scripts/evaluate_residual_method_sweep.py), [`evaluate_residual_meta_stacking.py`](../scripts/evaluate_residual_meta_stacking.py)
- 速度–消費電力の現行代表条件カーブ: [`speed_power_prediction.png`](speed_power_prediction.png)、数値表は[`speed_power_prediction_notes.md`](speed_power_prediction_notes.md)
