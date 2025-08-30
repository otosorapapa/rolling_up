
# 年計ダッシュボード（Streamlit版）

## セットアップ & 実行
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## 使い方
1. 左メニュー「データ取込」で Excel/CSV をアップロード
2. 商品名列・（あれば）商品コード列を指定 → **変換＆取込**
3. 「ダッシュボード」で KPI とトレンドを確認（終端月を選択）
4. 「ランキング」でゼロ除外バーを確認、「SKU詳細」で複数SKU比較、「相関分析」で指標間の関係を確認
5. 「設定」で閾値・ウィンドウを修正して**再計算**

## 仕様（MVP）
- 12カ月移動累計（年計）、YoY、Δ、直近Nの傾き（OLS）
- 欠測ポリシー：`zero_fill`（0埋め）/ `mark_missing`（欠測含む窓は非計上）
- アラート（YoY / Δ / 傾き）
- 相関分析（ヒートマップ・散布図マトリクス）
- ランキングで年計ゼロを除外するバー表示
- CSV/XLSX/PDF（KPI+Top10）エクスポート
- ビュー保存（閾値/ウィンドウ/単位）
