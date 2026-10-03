# BWSC2025残差モデル：Gaussian Process＋同日オンライン補正

## 結果

外側はDay1–5の日単位ホールドアウトです。GP学習用の各残差ラベルは、評価日とラベル日を除いた物理モデルから生成しました。GPの後に同日で完了した過去6ブロック（30分）のinnovation中央値を加え、履歴が10分以上途切れたらリセットします。Day6は選択・調整に使わず、異常ストレス確認だけに用いました。

| モデル | 5分MAE (W) | MAE改善 | P90絶対誤差 (W) | 日別勝ち | 補足 |
|---|---:|---:|---:|---:|---|
| 物理基準 | 70.20 | — | 142.04 | — | Day1–5、233ブロック |
| GPのみ、noise=60 W | 65.59 | 6.6% | 128.50 | 3/5 | 90%区間被覆88.4%、平均幅245 W |
| GPのみ、noise=100 W | 65.91 | 6.1% | 137.37 | 4/5 | 90%区間被覆96.1%、平均幅364 W |
| GP noise=100＋median6 | 56.65 | 19.3% | 129.54 | 5/5 | 保守設定を含む探索上の最良候補 |
| 同日median6のみ | 57.07 | 18.7% | 130.72 | 4/5 | より単純な既存シャドー候補 |

組み合わせ候補はDay1–5で5日すべて基準MAEを下回りました。一方、より複雑にした効果は同日median6だけに対して**0.42 W**です。複数候補を同じ5日で比較したため、19.3%は選択後の楽観値です。独立データで同程度の差が再現されたとは扱いません。

| Grain | 評価 | 物理基準MAE | GP＋online MAE | P90前 → 後 |
|---|---|---:|---:|---:|
| 5分平均 | Day1–5 | 70.20 W | 56.65 W | 142.04 → 129.54 W |
| 5秒ログ | Day1–5、13,627行 | 217.86 W | 212.24 W | 503.95 → 496.68 W |
| 5秒ログ | Day6異常、2,497行 | 292.41 W | 278.60 W | 641.91 → 649.28 W |

Day6では平均誤差が下がってもP90が悪化しました。適応器が異常なbiasを学習・隠蔽する可能性は残ります。Day6の改善は通常性能の証拠に数えていません。

## 採否

速度候補を区別する暫定ゲートは23.8 Wです。これは70 km/h付近の隣接候補差の25%として置いたプロジェクト内の工学基準で、外部規格ではありません。GP＋onlineの5分MAEはゲートの約2.4倍、5秒MAEは約8.9倍です。また、単独GPの誤差帯は十分に狭くなく、noise=100 W設定でもDay1–5の実測被覆が96.1%になる平均幅は364 Wでした。

よって、GP＋online候補はJSON台帳とCSVに保存しましたが、既存の同日median6シャドー候補を置き換えず、Grafana表示・全レース再生・MPC速度選択・車両指令にも接続しません。現ログでの性能保証や「最高精度」の断定はできません。手法をさらに複雑化するより、独立した新しい走行条件で事前固定検証を取ることが必要です。

## 方法の根拠と適用範囲

物理モデルに残差学習を足す方法はEVエネルギー予測でも提案され、GAM、Random Forest、boosting等を比較した研究があります。学習ベースMPCでは、Gaussian Processで残差不確かさを扱い、安全制約を保つ研究や、過去のモデル予測誤差からオンライン更新するレース車両の研究があります。[Huang et al. (2024)](https://arxiv.org/abs/2405.10984)、[Kabzan et al. (2019)](https://doi.org/10.3929/ethz-b-000351561)、[Costa et al. (2023)](https://doi.org/10.1016/j.robot.2023.104469)。これらは異なる車両・課題・データ量で得た方法論の根拠であり、公表された改善率はYATAへ移転できません。

## 生成物

- `scripts/evaluate_crossday_residual_models.py`, `scripts/evaluate_residual_gp_online_hybrid.py`
- `reports/residual_gp_online_hybrid_summary.json`, `reports/residual_gp_online_hybrid.csv`
- `reports/residual_gp_online_hybrid_block_oof.csv`
- `data/processed/full_race_residual_gp_online_hybrid_5s.csv`
- `models/vehicle_residual_hybrid_candidate.json`（研究候補、production_ready=false）
