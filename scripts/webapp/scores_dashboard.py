"""全銘柄のスコア一覧を閲覧するStreamlitダッシュボード。

起動: streamlit run scripts/webapp/scores_dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection  # noqa: E402

TOPIX_TICKER = "1306.T"
CATEGORY_COLS = {
    "score_safety": "安全性",
    "score_growth": "成長性",
    "score_profitability": "収益性",
    "score_efficiency": "効率性",
    "score_valuation": "割安性",
    "score_shareholder_return": "還元性",
}
PERIOD_DAYS = {"1ヶ月": 30, "3ヶ月": 90, "6ヶ月": 180, "1年": 365, "3年": 365 * 3}
COMPACT_HEIGHT = 230

st.set_page_config(page_title="日本株スコア一覧", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 1rem;}
    h1 {font-size: 1.4rem !important;}
    h2, h3 {font-size: 1.05rem !important; margin-top: 0.3rem !important; margin-bottom: 0.2rem !important;}
    div[data-testid="stMetric"] {padding: 2px 0;}
    div[data-testid="stMetricValue"] {font-size: 1.1rem;}
    div[data-testid="stMetricLabel"] {font-size: 0.75rem;}
    .at-a-glance-card {
        border-radius: 6px; padding: 6px 10px; text-align: center; height: 100%;
    }
    .at-a-glance-card .label {font-size: 0.72rem; opacity: 0.75;}
    .at-a-glance-card .value {font-size: 1.15rem; font-weight: 700;}
    .at-a-glance-card .sub {font-size: 0.68rem; opacity: 0.7;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("日本株 銘柄スコア一覧")


def _compact(fig: go.Figure, height: int = COMPACT_HEIGHT) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=35, r=35, t=28, b=25),
        font=dict(size=11),
        legend=dict(font=dict(size=9), orientation="h", yanchor="bottom", y=1.0),
    )
    return fig


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
        f.per, f.pbr, f.roe, f.dividend_yield, f.market_cap, f.equity_ratio, f.revenue_growth_3y_cagr,
        (SELECT close FROM price_daily p WHERE p.ticker = s.ticker ORDER BY p.date DESC LIMIT 1) AS price
    FROM scores s
    JOIN tickers t ON s.ticker = t.ticker
    LEFT JOIN fundamentals_weekly f ON f.ticker = s.ticker AND f.snapshot_date = s.snapshot_date
    WHERE s.snapshot_date = ?
    """
    df = pd.read_sql_query(query, conn, params=(snapshot_date,))
    conn.close()
    return df, snapshot_date


@st.cache_data(ttl=300)
def load_yearly(ticker: str) -> pd.DataFrame:
    conn = get_connection()
    query = """
    SELECT fiscal_year_end, revenue, operating_income, ordinary_income, net_income,
           operating_margin, net_margin, eps, dividend_per_share, payout_ratio,
           total_assets, total_liabilities, equity, equity_ratio,
           operating_cf, investing_cf, financing_cf, free_cf, cash_and_equivalents
    FROM fundamentals_yearly
    WHERE ticker = ?
    ORDER BY fiscal_year_end ASC
    """
    df = pd.read_sql_query(query, conn, params=(ticker,))
    conn.close()
    df["year"] = pd.to_datetime(df["fiscal_year_end"]).dt.year
    return df


@st.cache_data(ttl=300)
def load_price_history(ticker: str, days: int) -> pd.DataFrame:
    conn = get_connection()
    query = """
    SELECT date, close FROM price_daily
    WHERE ticker = ? AND date >= date('now', ?)
    ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(ticker, f"-{days} days"), parse_dates=["date"])
    conn.close()
    return df


@st.cache_data(ttl=300)
def load_decisions(ticker: str, days: int) -> pd.DataFrame:
    conn = get_connection()
    query = """
    SELECT decision_date, action, price_at_decision, reason
    FROM decisions
    WHERE ticker = ? AND decision_source = 'rule' AND action IN ('buy', 'sell')
      AND decision_date >= date('now', ?)
    ORDER BY decision_date ASC
    """
    df = pd.read_sql_query(query, conn, params=(ticker, f"-{days} days"), parse_dates=["decision_date"])
    conn.close()
    return df


def cagr(series: pd.Series) -> float | None:
    valid = series.dropna()
    if len(valid) < 2 or valid.iloc[0] <= 0:
        return None
    years = len(valid) - 1
    return (valid.iloc[-1] / valid.iloc[0]) ** (1 / years) - 1


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
    height=350,
)

st.divider()
st.subheader("個別銘柄の詳細")

ticker_options = filtered["ticker"] + " " + filtered["name"].fillna("")
selected_label = st.selectbox("銘柄を選択", ticker_options.tolist())

if selected_label:
    selected_ticker = selected_label.split(" ")[0]
    row = df[df["ticker"] == selected_ticker].iloc[0]

    # --- ひと目カード(バフェット・コード風の要約カード) ---
    glance_items = [
        ("時価総額(億円)", f"{row['market_cap'] / 1e8:,.0f}" if pd.notna(row["market_cap"]) else "-", "#FFF6E0"),
        ("PER(倍)", f"{row['per']:.1f}" if pd.notna(row["per"]) else "-", "#E8F5E9"),
        ("PBR(倍)", f"{row['pbr']:.2f}" if pd.notna(row["pbr"]) else "-", "#E8F5E9"),
        ("ROE(%)", f"{row['roe'] * 100:.1f}" if pd.notna(row["roe"]) else "-", "#FCE9E9"),
        ("自己資本比率(%)", f"{row['equity_ratio']:.1f}" if pd.notna(row["equity_ratio"]) else "-", "#E6F0FA"),
        ("配当利回り(%)", f"{row['dividend_yield']:.2f}" if pd.notna(row["dividend_yield"]) else "-", "#F1F1F1"),
    ]
    cols = st.columns(len(glance_items))
    for col, (label, value, color) in zip(cols, glance_items):
        col.markdown(
            f"""<div class="at-a-glance-card" style="background:{color};">
                <div class="label">{label}</div><div class="value">{value}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.caption(
        f"{row['ticker']} {row['name']}　業種: {row['sector']}　"
        f"総合スコア {row['total_score']:.1f}({row['grade']})　業種内 {int(row['sector_rank'])}/{int(row['sector_size'])}位"
    )

    # --- レーダーチャート + 株価/相対株価 ---
    period_label = st.radio("株価チャートの期間", list(PERIOD_DAYS.keys()), index=3, horizontal=True)
    days = PERIOD_DAYS[period_label]

    col_radar, col_price, col_rel = st.columns(3)

    with col_radar:
        labels = list(CATEGORY_COLS.values())
        values = [row[k] if pd.notna(row[k]) else 0 for k in CATEGORY_COLS.keys()]
        sector_peers = df[df["sector"] == row["sector"]]
        sector_medians = [sector_peers[k].median() for k in CATEGORY_COLS.keys()]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values + values[:1], theta=labels + labels[:1], fill="toself", name="本銘柄"))
        fig.add_trace(go.Scatterpolar(
            r=sector_medians + sector_medians[:1], theta=labels + labels[:1],
            name="業種中央値", line=dict(dash="dash", color="gray"),
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
        st.caption("6軸スコア")
        st.plotly_chart(_compact(fig), use_container_width=True)

    price_hist = load_price_history(selected_ticker, days)
    decisions = load_decisions(selected_ticker, days)

    with col_price:
        st.caption("株価 + AI判断日")
        if price_hist.empty:
            st.info("価格データがありません")
        else:
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=price_hist["date"], y=price_hist["close"], name="株価", line=dict(width=1.5)))
            for action, color, symbol in [("buy", "green", "triangle-up"), ("sell", "red", "triangle-down")]:
                sub = decisions[decisions["action"] == action]
                if not sub.empty:
                    fig_price.add_trace(go.Scatter(
                        x=sub["decision_date"], y=sub["price_at_decision"], mode="markers", name=action,
                        marker=dict(symbol=symbol, size=10, color=color),
                    ))
            st.plotly_chart(_compact(fig_price), use_container_width=True)

    with col_rel:
        st.caption("対TOPIX相対株価")
        topix_hist = load_price_history(TOPIX_TICKER, days)
        if price_hist.empty or topix_hist.empty:
            st.info("比較用データがありません")
        else:
            merged = pd.merge(price_hist, topix_hist, on="date", suffixes=("_stock", "_topix")).dropna()
            if merged.empty:
                st.info("比較用データがありません")
            else:
                rel = merged["close_stock"] / merged["close_stock"].iloc[0] * 100 - merged["close_topix"] / merged["close_topix"].iloc[0] * 100
                fig_rel = go.Figure()
                fig_rel.add_trace(go.Scatter(x=merged["date"], y=rel, name="対TOPIX超過(%pt)", line=dict(color="purple", width=1.5)))
                fig_rel.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"))
                st.plotly_chart(_compact(fig_rel), use_container_width=True)

    if not decisions.empty:
        with st.expander(f"この期間のAI判断ログ({len(decisions)}件)"):
            st.dataframe(decisions.rename(columns={
                "decision_date": "判断日", "action": "判断", "price_at_decision": "株価", "reason": "理由",
            }), use_container_width=True, hide_index=True)

    # --- 業績・配当・財務・CF推移 ---
    yearly = load_yearly(selected_ticker)
    if yearly.empty:
        st.info("この銘柄の複数年推移データはまだありません(次回のfetch_fundamentals.py/fetch_edinet_history.py実行で蓄積されます)。")
    else:
        revenue_cagr = cagr(yearly["revenue"])
        cagr_text = f"売上高{len(yearly) - 1}年CAGR: {revenue_cagr * 100:.1f}%" if revenue_cagr is not None else ""

        row1a, row1b = st.columns(2)
        with row1a:
            st.caption(f"業績推移　{cagr_text}")
            fig_perf = go.Figure()
            fig_perf.add_bar(x=yearly["year"], y=yearly["revenue"], name="売上高")
            fig_perf.add_trace(go.Scatter(x=yearly["year"], y=yearly["operating_margin"] * 100, name="営業利益率(%)", yaxis="y2"))
            fig_perf.add_trace(go.Scatter(x=yearly["year"], y=yearly["net_margin"] * 100, name="純利益率(%)", yaxis="y2"))
            fig_perf.update_layout(yaxis=dict(title="売上高"), yaxis2=dict(title="利益率(%)", overlaying="y", side="right"))
            st.plotly_chart(_compact(fig_perf), use_container_width=True)

        with row1b:
            st.caption("配当推移(1株配当・EPS)")
            fig_div = go.Figure()
            fig_div.add_bar(x=yearly["year"], y=yearly["dividend_per_share"], name="1株配当")
            fig_div.add_trace(go.Scatter(x=yearly["year"], y=yearly["eps"], name="EPS", yaxis="y2"))
            fig_div.update_layout(yaxis=dict(title="1株配当"), yaxis2=dict(title="EPS", overlaying="y", side="right"))
            st.plotly_chart(_compact(fig_div), use_container_width=True)

        row2a, row2b = st.columns(2)
        with row2a:
            st.caption("財務推移(純資産・負債・自己資本比率)")
            fig_bs = go.Figure()
            fig_bs.add_bar(x=yearly["year"], y=yearly["equity"], name="純資産")
            fig_bs.add_bar(x=yearly["year"], y=yearly["total_liabilities"], name="負債")
            fig_bs.add_trace(go.Scatter(x=yearly["year"], y=yearly["equity_ratio"], name="自己資本比率(%)", yaxis="y2"))
            fig_bs.update_layout(barmode="group", yaxis=dict(title="金額"), yaxis2=dict(title="自己資本比率(%)", overlaying="y", side="right"))
            st.plotly_chart(_compact(fig_bs), use_container_width=True)

        with row2b:
            st.caption("キャッシュフロー推移")
            fig_cf = go.Figure()
            fig_cf.add_bar(x=yearly["year"], y=yearly["operating_cf"], name="営業CF")
            fig_cf.add_bar(x=yearly["year"], y=yearly["investing_cf"], name="投資CF")
            fig_cf.add_bar(x=yearly["year"], y=yearly["financing_cf"], name="財務CF")
            fig_cf.add_trace(go.Scatter(x=yearly["year"], y=yearly["free_cf"], name="フリーCF", mode="lines+markers"))
            fig_cf.update_layout(barmode="relative")
            st.plotly_chart(_compact(fig_cf), use_container_width=True)
