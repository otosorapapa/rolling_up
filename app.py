
import io
import json
from datetime import datetime
from typing import Optional, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from services import (
    parse_uploaded_table,
    fill_missing_months,
    compute_year_rolling,
    compute_slopes,
    abc_classification,
    compute_hhi,
    build_alerts,
    aggregate_overview,
    get_comparables,
    build_indexed_series,
)

APP_TITLE = "売上年計（12カ月移動累計）ダッシュボード"
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

# ---------------- Session State ----------------
if "data_monthly" not in st.session_state:
    st.session_state.data_monthly = None  # long-form DF
if "data_year" not in st.session_state:
    st.session_state.data_year = None
if "settings" not in st.session_state:
    st.session_state.settings = {
        "window": 12,
        "last_n": 12,
        "missing_policy": "zero_fill",
        "yoy_threshold": -0.10,
        "delta_threshold": -300000.0,
        "slope_threshold": -1.0,
        "currency_unit": "円",
    }
if "notes" not in st.session_state:
    st.session_state.notes = {}  # product_code -> str
if "tags" not in st.session_state:
    st.session_state.tags = {}  # product_code -> List[str]
if "saved_views" not in st.session_state:
    st.session_state.saved_views = {}  # name -> dict
if "compare_params" not in st.session_state:
    st.session_state.compare_params = {}
if "compare_results" not in st.session_state:
    st.session_state.compare_results = None


# ---------------- Helpers ----------------
def require_data():
    if st.session_state.data_year is None or st.session_state.data_monthly is None:
        st.info("データが未取り込みです。左メニュー「データ取込」からアップロードしてください。")
        st.stop()


def month_options(df: pd.DataFrame) -> List[str]:
    return sorted(df["month"].dropna().unique().tolist())


def end_month_selector(df: pd.DataFrame, key="end_month"):
    mopts = month_options(df)
    default = mopts[-1] if mopts else None
    return st.selectbox("終端月（年計の計算対象）", mopts, index=(len(mopts)-1) if mopts else 0, key=key)


def download_excel(df: pd.DataFrame, filename: str) -> bytes:
    import xlsxwriter  # noqa
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    return output.getvalue()


def download_pdf_overview(kpi: dict, top_df: pd.DataFrame, filename: str) -> bytes:
    # Minimal PDF using reportlab (text only)
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    y = h - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "年計KPIサマリー")
    y -= 24
    c.setFont("Helvetica", 11)
    for k, v in kpi.items():
        c.drawString(40, y, f"{k}: {v}")
        y -= 14
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "TOP10（年計）")
    y -= 18
    c.setFont("Helvetica", 10)
    cols = ["product_code", "product_name", "year_sum"]
    for _, row in top_df[cols].head(10).iterrows():
        c.drawString(40, y, f"{row['product_code']}  {row['product_name']}  {int(row['year_sum']):,}")
        y -= 12
        if y < 60:
            c.showPage()
            y = h - 50
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# ---------------- Sidebar ----------------
st.sidebar.title(APP_TITLE)
page = st.sidebar.radio("メニュー", ["ダッシュボード", "ランキング", "比較ビュー", "SKU詳細", "データ取込", "アラート", "設定", "保存ビュー"])

# ---------------- Pages ----------------

# 1) データ取込
if page == "データ取込":
    st.header("データ取込 / マッピング / 品質チェック")

    st.markdown("**Excel(.xlsx) / CSV をアップロードしてください。** "
                "列に `YYYY-MM`（または日付系）形式の月度が含まれている必要があります。")

    col_u1, col_u2 = st.columns([2,1])
    with col_u1:
        file = st.file_uploader("ファイル選択", type=["xlsx","csv"])
    with col_u2:
        st.session_state.settings["missing_policy"] = st.selectbox("欠測月ポリシー",
            options=["zero_fill","mark_missing"],
            format_func=lambda x: "ゼロ補完(推奨)" if x=="zero_fill" else "欠測含む窓は非計上",
            index=0)

    if file is not None:
        try:
            if file.name.lower().endswith(".csv"):
                df_raw = pd.read_csv(file)
            else:
                df_raw = pd.read_excel(file, engine="openpyxl")
        except Exception as e:
            st.error(f"読込エラー: {e}")
            st.stop()

        st.caption("アップロードプレビュー（先頭100行）")
        st.dataframe(df_raw.head(100), use_container_width=True)

        cols = df_raw.columns.tolist()
        product_name_col = st.selectbox("商品名列の選択", options=cols, index=0)
        product_code_col = st.selectbox("商品コード列の選択（任意）", options=["<なし>"] + cols, index=0)
        code_col = None if product_code_col == "<なし>" else product_code_col

        if st.button("変換＆取込", type="primary"):
            try:
                long_df = parse_uploaded_table(df_raw, product_name_col=product_name_col, product_code_col=code_col)
                long_df = fill_missing_months(long_df, policy=st.session_state.settings["missing_policy"])
                # Compute year rolling & slopes
                year_df = compute_year_rolling(long_df, window=st.session_state.settings["window"],
                                               policy=st.session_state.settings["missing_policy"])
                year_df = compute_slopes(year_df, last_n=st.session_state.settings["last_n"])

                st.session_state.data_monthly = long_df
                st.session_state.data_year = year_df
                st.success("取込完了。ダッシュボードへ移動して可視化を確認してください。")

                st.subheader("品質チェック（欠測月/非数値/重複）")
                # 欠測月
                miss_rate = (long_df["is_missing"].sum(), len(long_df))
                st.write(f"- 欠測セル数: {miss_rate[0]:,} / {miss_rate[1]:,}")
                # 月レンジ
                st.write(f"- データ期間: {long_df['month'].min()} 〜 {long_df['month'].max()}")
                # SKU数
                st.write(f"- SKU数: {long_df['product_code'].nunique():,}")
                st.write(f"- レコード数: {len(long_df):,}")

                st.download_button("年計テーブルをCSVでダウンロード", data=st.session_state.data_year.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="year_rolling.csv", mime="text/csv")
            except Exception as e:
                st.exception(e)

# 2) ダッシュボード
elif page == "ダッシュボード":
    require_data()
    st.header("ダッシュボード")

    end_m = end_month_selector(st.session_state.data_year, key="end_month_dash")

    # KPI
    kpi = aggregate_overview(st.session_state.data_year, end_m)
    hhi = compute_hhi(st.session_state.data_year, end_m)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("年計総額", f"{int(kpi['total_year_sum']):,} 円" if kpi["total_year_sum"] is not None else "-")
    c2.metric("年計YoY", f"{kpi['yoy']*100:.1f} %" if kpi["yoy"] is not None else "—")
    c3.metric("前月差(Δ)", f"{int(kpi['delta']):,} 円" if kpi["delta"] is not None else "—")
    c4.metric("HHI(集中度)", f"{hhi:.3f}")

    # 総合年計トレンド（全SKU合計）
    totals = st.session_state.data_year.groupby("month", as_index=False)["year_sum"].sum()
    fig = px.line(totals, x="month", y="year_sum", title="総合 年計トレンド", markers=True)
    fig.update_layout(height=350, margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ランキング（年計）
    snap = st.session_state.data_year[st.session_state.data_year["month"] == end_m].dropna(subset=["year_sum"]).copy()
    snap = snap.sort_values("year_sum", ascending=False)
    st.subheader(f"ランキング（{end_m} 時点 年計）")
    st.dataframe(snap[["product_code","product_name","year_sum","yoy","delta"]].head(20), use_container_width=True)
    st.download_button("この表をCSVでダウンロード", data=snap.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"ranking_{end_m}.csv", mime="text/csv")

    # PDF出力（KPI + TOP10）
    pdf_bytes = download_pdf_overview({"total_year_sum": int(kpi["total_year_sum"]),
                                       "yoy": round(kpi["yoy"],4) if kpi["yoy"] is not None else None,
                                       "delta": int(kpi["delta"]) if kpi["delta"] is not None else None},
                                      snap, filename=f"overview_{end_m}.pdf")
    st.download_button("会議用PDF（KPI+Top10）を出力", data=pdf_bytes, file_name=f"overview_{end_m}.pdf", mime="application/pdf")

# 3) ランキング
elif page == "ランキング":
    require_data()
    st.header("ランキング / ワースト")
    end_m = end_month_selector(st.session_state.data_year, key="end_month_rank")
    metric = st.selectbox("指標", options=["year_sum","yoy","delta","slope_beta"], index=0)
    order = st.radio("並び順", options=["desc","asc"], horizontal=True)

    snap = st.session_state.data_year[st.session_state.data_year["month"] == end_m].copy()
    snap = snap.dropna(subset=[metric])
    snap = snap.sort_values(metric, ascending=(order=="asc"))
    st.dataframe(snap[["product_code","product_name","year_sum","yoy","delta","slope_beta"]].head(100), use_container_width=True)

    st.download_button("CSVダウンロード", data=snap.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"ranking_{metric}_{end_m}.csv", mime="text/csv")
    st.download_button("Excelダウンロード", data=download_excel(snap, f"ranking_{metric}_{end_m}.xlsx"),
                       file_name=f"ranking_{metric}_{end_m}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# 4) 比較ビュー
elif page == "比較ビュー":
    require_data()
    st.header("比較ビュー")
    params = st.session_state.compare_params
    year_df = st.session_state.data_year
    prods = year_df[["product_code","product_name"]].drop_duplicates().sort_values("product_code")
    prod_opts = prods["product_code"] + " | " + prods["product_name"]
    default_idx = 0
    if params.get("target_code"):
        try:
            default_idx = prod_opts[prod_opts.str.startswith(params["target_code"])].index[0]
        except Exception:
            default_idx = 0
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        end_m = end_month_selector(year_df, key="compare_end_month")
    with c2:
        target_label = st.selectbox("基準SKU", options=prod_opts, index=default_idx)
        target_code = target_label.split(" | ")[0]
    with c3:
        mode = st.radio("抽出モード", ["abs","pct","rank"], index=["abs","pct","rank"].index(params.get("mode","pct")), horizontal=True)
    with c4:
        if mode == "abs":
            range_val = st.number_input("±金額", value=float(params.get("range",100000.0)), step=10000.0, format="%.0f")
        elif mode == "pct":
            range_val = st.number_input("±%", value=float(params.get("range",0.10)), step=0.01, format="%.2f")
        else:
            range_val = st.number_input("±順位", value=int(params.get("range",5)), step=1)
    with st.expander("フィルタ / オプション"):
        fil_abc = st.multiselect("ABCクラス", ["A","B","C"], default=params.get("filters", {}).get("abc", []))
        fil_tags = st.text_input("タグ（カンマ区切り）", value=",".join(params.get("filters", {}).get("tags", [])))
        fil_yoy = st.number_input("YoY <= ", value=params.get("filters", {}).get("yoy_le", 0.0), step=0.01, format="%.2f")
        fil_delta = st.number_input("Δ <= ", value=params.get("filters", {}).get("delta_le", 0.0), step=1000.0, format="%.0f")
        fil_slope = st.number_input("傾き <= ", value=params.get("filters", {}).get("slope_le", 0.0), step=0.1, format="%.2f")
        index_opt = st.checkbox("インデックス化", value=params.get("index", False))
        log_opt = st.checkbox("ログ軸", value=params.get("log", False))
        max_items = st.slider("最大表示件数", 5, 100, value=int(params.get("max_items",20)))
        sort_metric = st.selectbox("並び順", ["year_sum","yoy","delta","slope_beta"], index=["year_sum","yoy","delta","slope_beta"].index(params.get("sort_metric","year_sum")))
    if st.button("抽出実行", type="primary"):
        filters = {}
        if fil_abc:
            filters["abc"] = fil_abc
        tag_list = [t.strip() for t in fil_tags.split(",") if t.strip()]
        if tag_list:
            filters["tags"] = tag_list
        if fil_yoy != 0.0:
            filters["yoy_le"] = fil_yoy
        if fil_delta != 0.0:
            filters["delta_le"] = fil_delta
        if fil_slope != 0.0:
            filters["slope_le"] = fil_slope
        params = {
            "end_month": end_m,
            "target_code": target_code,
            "mode": mode,
            "range": range_val,
            "filters": filters,
            "index": index_opt,
            "log": log_opt,
            "max_items": max_items,
            "sort_metric": sort_metric,
            "pinned": params.get("pinned", []),
            "excluded": params.get("excluded", []),
        }
        st.session_state.compare_params = params
        if mode == "rank":
            comps = get_comparables(year_df, end_m, target_code, mode=mode, rank_k=int(range_val), filters=filters, tags_map=st.session_state.tags)
        else:
            comps = get_comparables(year_df, end_m, target_code, mode=mode, low=-range_val, high=range_val, filters=filters, tags_map=st.session_state.tags)
        pinned = params.get("pinned", [])
        if pinned:
            pins = get_comparables(year_df, end_m, target_code, mode="rank", rank_k=len(prods), tags_map=st.session_state.tags)
            pins = pins[pins["product_code"].isin(pinned)]
            comps = pd.concat([comps, pins], ignore_index=True)
        excluded = params.get("excluded", [])
        if excluded:
            comps = comps[~comps["product_code"].isin(excluded)]
        comps = comps.drop_duplicates("product_code")
        comps = comps.sort_values(sort_metric, ascending=False).head(max_items)
        if target_code not in comps["product_code"].tolist():
            base = get_comparables(year_df, end_m, target_code, mode="rank", rank_k=0, tags_map=st.session_state.tags)
            comps = pd.concat([base, comps], ignore_index=True).drop_duplicates("product_code")
        st.session_state.compare_results = comps
    comps = st.session_state.compare_results
    if isinstance(comps, pd.DataFrame) and not comps.empty:
        st.subheader("年計ライン比較")
        codes = comps["product_code"].tolist()
        data = st.session_state.data_year[st.session_state.data_year["product_code"].isin(codes)].copy()
        data = data.merge(comps[["product_code","product_name","abc_class","tags"]], on="product_code", how="left")
        if params.get("index"):
            idx = build_indexed_series(st.session_state.data_year, codes)
            data = data.merge(idx, on=["product_code","month"], how="left")
            ycol = "index_value"
        else:
            ycol = "year_sum"
        fig = px.line(data, x="month", y=ycol, color="product_code",
                      hover_data=["product_name","year_sum","yoy","delta","slope_beta","abc_class","tags"])
        for t in fig.data:
            if t.name == params.get("target_code"):
                t.update(line={"width":4})
            else:
                t.update(line={"width":1, "opacity":0.3})
        if params.get("log"):
            fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True, height=500)
        pin_sel = st.multiselect("固定SKU", options=comps["product_code"], default=params.get("pinned", []))
        ex_sel = st.multiselect("除外SKU", options=comps["product_code"], default=params.get("excluded", []))
        if st.button("Pin/Exclude更新"):
            params['pinned'] = pin_sel
            params['excluded'] = ex_sel
            st.session_state.compare_params = params
        exp_cols = ["product_code","product_name","year_sum","yoy","delta","slope_beta","abc_class","tags"]
        st.download_button("CSVエクスポート", data=comps[exp_cols].to_csv(index=False).encode("utf-8-sig"), file_name=f"comparables_{params.get('target_code','')}_{params.get('end_month','')}.csv", mime="text/csv")
        st.subheader("スパークライン")
        cols = st.columns(4)
        for i, row in comps.iterrows():
            g = data[data['product_code'] == row['product_code']].sort_values('month')
            fig_s = px.line(g, x='month', y='year_sum', height=120)
            fig_s.update_layout(margin=dict(l=10,r=10,t=20,b=20), xaxis={'visible':False}, yaxis={'visible':False})
            with cols[i % 4]:
                st.markdown(f"**{row['product_code']} | {row['product_name']}**")
                st.plotly_chart(fig_s, use_container_width=True, height=120)
                st.caption(f"{int(row['year_sum']):,} / YoY {row['yoy']*100:.1f}% / Δ {int(row['delta']):,}")

# 5) SKU詳細
elif page == "SKU詳細":
    require_data()
    st.header("SKU 詳細")
    end_m = end_month_selector(st.session_state.data_year, key="end_month_detail")
    prods = st.session_state.data_year[["product_code","product_name"]].drop_duplicates().sort_values("product_code")
    prod_label = st.selectbox("SKU選択", options=prods["product_code"] + " | " + prods["product_name"])
    code = prod_label.split(" | ")[0]

    # 時系列
    g_m = st.session_state.data_monthly[st.session_state.data_monthly["product_code"] == code].sort_values("month")
    g_y = st.session_state.data_year[st.session_state.data_year["product_code"] == code].sort_values("month")

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.line(g_m, x="month", y="sales_amount_jpy", title="単月 売上推移", markers=True)
        st.plotly_chart(fig1, use_container_width=True, height=350)
    with c2:
        fig2 = px.line(g_y, x="month", y="year_sum", title="年計 推移", markers=True)
        st.plotly_chart(fig2, use_container_width=True, height=350)

    # 指標
    row = g_y[g_y["month"] == end_m]
    if not row.empty:
        rr = row.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("年計", f"{int(rr['year_sum']) if not pd.isna(rr['year_sum']) else '—'}")
        c2.metric("YoY", f"{rr['yoy']*100:.1f} %" if not pd.isna(rr["yoy"]) else "—")
        c3.metric("Δ", f"{int(rr['delta'])}" if not pd.isna(rr["delta"]) else "—")

    # メモ / タグ
    st.subheader("メモ / タグ")
    note = st.text_area("メモ（保存で保持）", value=st.session_state.notes.get(code, ""), height=100)
    tags_str = st.text_input("タグ（カンマ区切り）", value=",".join(st.session_state.tags.get(code, [])))
    c1, c2 = st.columns([1,1])
    if c1.button("保存"):
        st.session_state.notes[code] = note
        st.session_state.tags[code] = [t.strip() for t in tags_str.split(",") if t.strip()]
        st.success("保存しました")
    if c2.button("CSVでエクスポート"):
        meta = pd.DataFrame([{"product_code": code, "note": st.session_state.notes.get(code, ""),
                              "tags": ",".join(st.session_state.tags.get(code, []))}])
        st.download_button("ダウンロード", data=meta.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"notes_{code}.csv", mime="text/csv")

# 5) アラート
elif page == "アラート":
    require_data()
    st.header("アラート")
    end_m = end_month_selector(st.session_state.data_year, key="end_month_alert")
    s = st.session_state.settings
    alerts = build_alerts(st.session_state.data_year, end_month=end_m,
                          yoy_threshold=s["yoy_threshold"],
                          delta_threshold=s["delta_threshold"],
                          slope_threshold=s["slope_threshold"])
    if alerts.empty:
        st.success("閾値に該当するアラートはありません。")
    else:
        st.dataframe(alerts, use_container_width=True)
        st.download_button("CSVダウンロード", data=alerts.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"alerts_{end_m}.csv", mime="text/csv")

# 6) 設定
elif page == "設定":
    st.header("設定")
    s = st.session_state.settings
    c1, c2, c3 = st.columns(3)
    with c1:
        s["window"] = st.number_input("年計ウィンドウ（月）", min_value=3, max_value=24, value=int(s["window"]), step=1)
        s["last_n"] = st.number_input("傾き算出の対象点数", min_value=3, max_value=36, value=int(s["last_n"]), step=1)
    with c2:
        s["yoy_threshold"] = st.number_input("YoY 閾値（<=）", value=float(s["yoy_threshold"]), step=0.01, format="%.2f")
        s["delta_threshold"] = st.number_input("Δ 閾値（<= 円）", value=float(s["delta_threshold"]), step=10000.0, format="%.0f")
    with c3:
        s["slope_threshold"] = st.number_input("傾き 閾値（<=）", value=float(s["slope_threshold"]), step=0.1, format="%.2f")
        s["currency_unit"] = st.selectbox("通貨単位表記", options=["円","千円","百万円"], index=["円","千円","百万円"].index(s["currency_unit"]))

    st.caption("※ 設定変更後は再計算が必要です。")
    if st.button("年計の再計算を実行", type="primary"):
        if st.session_state.data_monthly is None:
            st.warning("先にデータを取り込んでください。")
        else:
            long_df = st.session_state.data_monthly
            year_df = compute_year_rolling(long_df, window=s["window"], policy=s["missing_policy"])
            year_df = compute_slopes(year_df, last_n=s["last_n"])
            st.session_state.data_year = year_df
            st.success("再計算が完了しました。")

# 7) 保存ビュー
elif page == "保存ビュー":
    st.header("保存ビュー / ブックマーク")
    s = st.session_state.settings
    cparams = st.session_state.compare_params
    st.write("現在の設定・選択（閾値、ウィンドウ、単位など）を名前を付けて保存します。")

    name = st.text_input("ビュー名")
    if st.button("保存"):
        if not name:
            st.warning("ビュー名を入力してください。")
        else:
            st.session_state.saved_views[name] = {"settings": dict(s), "compare": dict(cparams)}
            st.success(f"ビュー「{name}」を保存しました。")

    st.subheader("保存済みビュー")
    if not st.session_state.saved_views:
        st.info("保存済みビューはありません。")
    else:
        for k, v in st.session_state.saved_views.items():
            st.write(f"**{k}**: {json.dumps(v, ensure_ascii=False)}")
            if st.button(f"適用: {k}"):
                st.session_state.settings.update(v.get("settings", {}))
                st.session_state.compare_params = v.get("compare", {})
                cp = st.session_state.compare_params
                if cp and st.session_state.data_year is not None:
                    filt = cp.get("filters", {})
                    if cp.get("mode") == "rank":
                        st.session_state.compare_results = get_comparables(
                            st.session_state.data_year, cp.get("end_month"), cp.get("target_code"),
                            mode=cp.get("mode"), rank_k=int(cp.get("range", 5)), filters=filt, tags_map=st.session_state.tags)
                    else:
                        st.session_state.compare_results = get_comparables(
                            st.session_state.data_year, cp.get("end_month"), cp.get("target_code"),
                            mode=cp.get("mode"), low=-float(cp.get("range", 0)), high=float(cp.get("range", 0)),
                            filters=filt, tags_map=st.session_state.tags)
                st.success(f"ビュー「{k}」を適用しました。")
