# BWSC2025 残差モデル全候補スイープと全データ再フィット

## 結果

Day1–5の各日の60%で学習、次の20%で全候補を選択し、最後の20%を一度だけ評価した。選択したモデルは **hist_gradient_boosting / hgb_l15_i200**。
未使用テスト区間の平均日別MAEは物理ベース 207.2 W、残差補正後 106.8 W。
Days1–5で選んだままのDay6ストレス評価は: [{"day": 6, "family": "hist_gradient_boosting", "configuration": "hgb_l15_i200", "scope": "day6_stress_test", "n": 507, "mae_w": 158.7611588745568, "rmse_w": 298.9626898114092, "p90_abs_error_w": 353.91837647340753, "bias_w": -21.395487092193072, "baseline_mae_w": 359.55104491557694, "baseline_p90_w": 767.24957292252}]

全17構成を検証し、最終選択モデルを16,242行すべてに再フィットした学習内残差はMAE 85.23 W、P90 177.76 W。これは同じ行を使って学習・評価した値なので、性能証拠ではなく学習内適合度である。

## 評価の切り分け

- 予測モデル特徴量は速度、走行距離、ルート勾配・位置、Open-Meteo由来の複数高度の風・気温、雲量、短波・直達・散乱・傾斜面日射、日射角の風成分、5秒間隔の過去残差ラグ／移動統計量である。
- 現在時刻のSolar_Watt・Battery_Wattは応答値を構成する実測列なので、予測モデルの特徴量から除外した。
- 別枠の測定オラクルはSolar_Watt + Battery_Wattをそのまま用い、定義上の測定値を完全再構成する。残差ゼロだが、将来予測器として無効である。
- 最終学習内スコアは全日全行のフィット結果である。未使用の時系列テスト指標は上記の最後20%だけである。Day6はモデル選択に使っていない。
- 速度と天候はログ上の値を既知として条件づけた。MPCで生成する未来速度・天気予報の誤差は含まない。

## 出力

- 検証スコア全候補: `reports\residual_method_sweep_validation.csv`
- 最終20%の行別予測: `reports/residual_method_sweep_oof_predictions.csv`
- 全行再フィットと測定オラクル列: `data\processed\full_race_residual_maxfit_5s.csv`
- シリアライズ済み学習器: `models\vehicle_residual_maxfit.joblib`
- 特徴量・モデル・検証条件: `models\vehicle_residual_maxfit_manifest.json`
