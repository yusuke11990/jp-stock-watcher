"""全銘柄のスコア一覧を閲覧するStreamlitダッシュボード。

起動: streamlit run scripts/webapp/scores_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection  # noqa: E402

CATEGORY_COLS = {
    "score_safety": "安全性",
    "score_growth": "成長性",
    "score_profitability": "収益性",
    "score_efficiency": "効率性",
    "score_valuation": "割安性",
    "score_shareholder_return": "還元性",
}

st.set_page_config(page_title="日本株スコア一覧", layout="wide")
st.title("日本株 銘柄スコア一覧")


@st.cache_data(ttl=300)
def load_data() -> tuple[pd.DataFrame, str | None]:
    conn = get_connection()
    snapshot_date = conn.execute("SELECT MAX(snapshot_date) FROM scores").fetchone()[0]
    if snapshot_date is None:
        conn.close()
        return pd.DataFrame(), None

    query = """
    SELECT
        s.ticker, t.name, t.sector, t.market,
        s.score_safety, s.score_growth, s.score_profitability,
        s.score_efficiency, s.score_valuation, s.score_shareholder_return,
        s.total_score, s.grade, s.sector_rank, s.sector_size,
        f.per, f.pbr, f.roe, f.dividend_yield, f.market_cap,
        (SELECT close FROM price_daily p WHERE p.ticker = s.ticker ORDER BY p.date DESC LIMIT 1) AS price
    FROM scores s
    JOIN tickers t ON s.ticker = t.ticker
    LEFT JOIN fundamentals_weekly f ON f.ticker = s.ticker AND f.snapshot_date = s.snapshot_date
    WHERE s.snapshot_date = ?
    """
    df = pd.read_sql_query(query, conn, params=(snapshot_date,))
    conn.close()
    return df, snapshot_date


df, snapshot_date = load_data()

if df.empty:
    st.warning("scoresテーブルにデータがありません。先にcompute_scores.pyを実行してください。")
    st.stop()

st.caption(f"データ基準日(snapshot_date): {snapshot_date}　対象銘柄数: {len(df)}")

with st.sidebar:
    st.header("絞り込み")
    search = st.text_input("銘柄名・コードで検索")

    sectors = ["すべて"] + sorted(df["sector"].dropna().unique().tolist())
    sector_sel = st.selectbox("業種", sectors)

    markets = sorted(df["market"].dropna().unique().tolist())
    market_sel = st.multiselect("市場", markets, default=markets)

    grades = ["S", "A", "B", "C", "D", "E"]
    grade_sel = st.multiselect("グレード", grades, default=grades)

    score_min = st.slider("総合スコア(以上)", 0, 100, 0)

filtered = df.copy()
if search:
    filtered = filtered[
        filtered["name"].str.contains(search, na=False) | filtered["ticker"].str.contains(search, na=False)
    ]
if sector_sel != "すべて":
    filtered = filtered[filtered["sector"] == sector_sel]
filtered = filtered[filtered["market"].isin(market_sel)]
filtered = filtered[filtered["grade"].isin(grade_sel)]
filtered = filtered[filtered["total_score"].fillna(-1) >= score_min]
filtered = filtered.sort_values("total_score", ascending=False)

st.subheader(f"スクリーニング結果 ({len(filtered)}件)")

display_df = filtered.rename(columns={**CATEGORY_COLS, "total_score": "総合スコア", "grade": "グレード",
                                       "ticker": "コード", "name": "銘柄名", "sector": "業種", "market": "市場",
                                       "per": "PER", "pbr": "PBR", "roe": "ROE", "dividend_yield": "配当利回り",
                                       "price": "株価", "sector_rank": "業種内順位", "sector_size": "業種銘柄数"})
display_cols = ["コード", "銘柄名", "業種", "市場", "株価", "PER", "PBR", "ROE",
                "安全性", "成長性", "収益性", "効率性", "割安性", "還元性",
                "総合スコア", "グレード", "業種内順位", "業種銘柄数"]
st.dataframe(
    display_df[display_cols].round(1),
    use_container_width=True,
    hide_index=True,
    height=500,
)

st.divider()
st.subheader("個別銘柄の詳細")

ticker_options = filtered["ticker"] + " " + filtered["name"].fillna("")
selected_label = st.selectbox("銘柄を選択", ticker_options.tolist())
if selected_label:
    selected_ticker = selected_label.split(" ")[0]
    row = df[df["ticker"] == selected_ticker].iloc[0]

    cols = st.columns(6)
    for col, (key, label) in zip(cols, CATEGORY_COLS.items()):
        col.metric(label, f"{row[key]:.0f}" if pd.notna(row[key]) else "-")

    labels = list(CATEGORY_COLS.values())
    values = [row[k] if pd.notna(row[k]) else 0 for k in CATEGORY_COLS.keys()]

    sector_peers = df[df["sector"] == row["sector"]]
    sector_medians = [sector_peers[k].median() for k in CATEGORY_COLS.keys()]

    # Plotlyはブラウザ側のフォントで描画するため、サーバーに日本語フォントが
    # 無い環境(Streamlit Community Cloud等)でも文字化けしない
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + values[:1], theta=labels + labels[:1],
        fill="toself", name=f"{row['ticker']} {row['name']}",
    ))
    fig.add_trace(go.Scatterpolar(
        r=sector_medians + sector_medians[:1], theta=labels + labels[:1],
        name=f"{row['sector']} 中央値", line=dict(dash="dash", color="gray"),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
