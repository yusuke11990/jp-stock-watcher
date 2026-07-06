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
from scoring.compute_scores import load_config, score_to_grade  # noqa: E402

_SCORING_CONFIG = load_config()
_GRADE_THRESHOLDS = _SCORING_CONFIG["grade_thresholds"]

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
        font=dict(size=11, family="Hiragino Sans, Yu Gothic, Meiryo, sans-serif", color="#333"),
        legend=dict(font=dict(size=9), orientation="h", yanchor="bottom", y=1.0),
        plot_bgcolor="white",
        hovermode="x unified",
    )
    return fig


def _yearly_chart_layout(fig: go.Figure) -> go.Figure:
    """バフェット・コード風: 年度ラベルを斜めに、グリッド線を薄く、マーカー付きの折れ線"""
    fig.update_xaxes(tickangle=-40, showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    for trace in fig.data:
        if trace.type == "scatter" and trace.mode is None:
            trace.mode = "lines+markers"
            trace.marker = dict(size=5)
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
def load_today_decisions() -> tuple[pd.DataFrame, str | None]:
    """最新decision_dateの買い/売り判断一覧(v1/v2両方)。"""
    conn = get_connection()
    decision_date = conn.execute("SELECT MAX(decision_date) FROM decisions").fetchone()[0]
    if decision_date is None:
        conn.close()
        return pd.DataFrame(), None

    query = """
    SELECT
        d.ticker, t.name, t.sector, d.action, d.rule_version, d.grade, d.total_score,
        d.confidence, d.reason, d.price_at_decision,
        d.stop_loss_price, d.take_profit_price, d.risk_reward_ratio
    FROM decisions d
    JOIN tickers t ON t.ticker = d.ticker
    WHERE d.decision_date = ? AND d.decision_source = 'rule' AND d.action IN ('buy', 'sell')
    ORDER BY d.rule_version, d.action, d.confidence DESC
    """
    df = pd.read_sql_query(query, conn, params=(decision_date,))
    conn.close()
    return df, decision_date


@st.cache_data(ttl=300)
def load_yearly(ticker: str) -> pd.DataFrame:
    conn = get_connection()
    query = """
    SELECT fiscal_year_end, revenue, operating_income, ordinary_income, net_income,
           operating_margin, net_margin, eps, dividend_per_share, payout_ratio,
           total_assets, total_liabilities, equity, equity_ratio,
           operating_cf, investing_cf, financing_cf, free_cf, cash_and_equivalents, buyback_amount
    FROM fundamentals_yearly
    WHERE ticker = ?
    ORDER BY fiscal_year_end ASC
    """
    df = pd.read_sql_query(query, conn, params=(ticker,))
    conn.close()
    fy = pd.to_datetime(df["fiscal_year_end"])
    df["year"] = fy.dt.year
    # バフェット・コード等でお馴染みの「YY.M」形式の決算期表記(例: 2026-03-31 -> "26.3")
    df["fy_label"] = fy.dt.strftime("%y.") + fy.dt.month.astype(str)
    if "net_income" in df.columns and "equity" in df.columns:
        df["roe"] = df["net_income"] / df["equity"]
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


def cagr_over_n_years(series: pd.Series, n: int) -> float | None:
    """直近n年分(n+1時点)でのCAGR。データがn+1年分に満たない場合はNone"""
    valid = series.dropna()
    if len(valid) < n + 1:
        return None
    return cagr(valid.iloc[-(n + 1):])


def _labels(series: pd.Series, fmt: str = "{:,.0f}") -> list[str]:
    return [fmt.format(v) if pd.notna(v) else "" for v in series]


def dividend_streak_stats(dividend_series: pd.Series) -> dict:
    """配当の年次推移から、増配/減配回数・連続増配・非減配年数・累進配当かを算出する"""
    valid = dividend_series.dropna()
    if len(valid) < 2:
        return {"increase": 0, "decrease": 0, "no_decrease_years": 0, "consecutive_increase": 0, "progressive": None}

    diffs = valid.diff().dropna()
    increase = int((diffs > 0).sum())
    decrease = int((diffs < 0).sum())

    consecutive_increase = 0
    for d in reversed(diffs.tolist()):
        if d > 0:
            consecutive_increase += 1
        else:
            break

    no_decrease_years = 0
    for d in reversed(diffs.tolist()):
        if d >= 0:
            no_decrease_years += 1
        else:
            break

    return {
        "increase": increase, "decrease": decrease,
        "no_decrease_years": no_decrease_years, "consecutive_increase": consecutive_increase,
        "progressive": decrease == 0,
    }


df, snapshot_date = load_data()

if df.empty:
    st.warning("scoresテーブルにデータがありません。先にcompute_scores.pyを実行してください。")
    st.stop()

st.caption(f"データ基準日(snapshot_date): {snapshot_date}　対象銘柄数: {len(df)}")

# --- 買い/売り銘柄一覧(このダッシュボードで最も重要なセクション) ---
today_decisions, decision_date = load_today_decisions()

st.subheader("本日の買い・売り判断")
if today_decisions.empty:
    st.info("判断データがありません。先にdecide_rule.py/decide_composite.pyを実行してください。")
else:
    versions = sorted(today_decisions["rule_version"].dropna().unique().tolist())
    default_version = "v2.0" if "v2.0" in versions else versions[0]
    version_sel = st.radio(
        "判断エンジン", versions, index=versions.index(default_version),
        horizontal=True, help="v1.0=グレード×二値シグナル(実績重視)、v2.0=連続値テクニカル合成(検証中)",
    )
    st.caption(f"判断基準日(decision_date): {decision_date}")

    version_df = today_decisions[today_decisions["rule_version"] == version_sel].rename(columns={
        "ticker": "コード", "name": "銘柄名", "sector": "業種", "action": "判断", "grade": "グレード",
        "total_score": "総合スコア", "confidence": "確信度", "reason": "理由", "price_at_decision": "判断時株価",
        "stop_loss_price": "損切りライン", "take_profit_price": "利確ライン", "risk_reward_ratio": "リスクリワード比",
    })

    col_buy, col_sell = st.columns(2)
    with col_buy:
        buy_df = version_df[version_df["判断"] == "buy"].drop(columns=["判断", "rule_version"])
        st.markdown(f"**買い候補 ({len(buy_df)}件)**")
        st.dataframe(buy_df.round(2), use_container_width=True, hide_index=True, height=300)
    with col_sell:
        sell_df = version_df[version_df["判断"] == "sell"].drop(columns=["判断", "rule_version"])
        st.markdown(f"**売り候補 ({len(sell_df)}件)**")
        st.dataframe(sell_df.round(2), use_container_width=True, hide_index=True, height=300)

st.divider()

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

filters_active = bool(
    search or sector_sel != "すべて" or set(market_sel) != set(markets)
    or set(grade_sel) != set(grades) or score_min > 0
)

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

if filters_active:
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

    yearly = load_yearly(selected_ticker)
    fiscal_month = pd.to_datetime(yearly["fiscal_year_end"]).max().month if not yearly.empty else None
    mix_ratio = (row["per"] * row["pbr"]) if pd.notna(row["per"]) and pd.notna(row["pbr"]) else None
    div_stats = dividend_streak_stats(yearly["dividend_per_share"]) if not yearly.empty else dividend_streak_stats(pd.Series(dtype=float))

    # --- ひと目カード(バフェット・コード風の要約カード) ---
    glance_items = [
        ("時価総額(億円)", f"{row['market_cap'] / 1e8:,.0f}" if pd.notna(row["market_cap"]) else "-", "#FFF6E0"),
        ("PER(倍)", f"{row['per']:.1f}" if pd.notna(row["per"]) else "-", "#E8F5E9"),
        ("PBR(倍)", f"{row['pbr']:.2f}" if pd.notna(row["pbr"]) else "-", "#E8F5E9"),
        ("ROE(%)", f"{row['roe'] * 100:.1f}" if pd.notna(row["roe"]) else "-", "#FCE9E9"),
        ("自己資本比率(%)", f"{row['equity_ratio']:.1f}" if pd.notna(row["equity_ratio"]) else "-", "#E6F0FA"),
        ("配当利回り(%)", f"{row['dividend_yield']:.2f}" if pd.notna(row["dividend_yield"]) else "-", "#F1F1F1"),
        ("MIX係数(PER×PBR)", f"{mix_ratio:.1f}" if mix_ratio is not None else "-", "#EDE7F6"),
        ("決算月", f"{fiscal_month}月" if fiscal_month else "-", "#F1F1F1"),
    ]
    cols = st.columns(len(glance_items))
    for col, (label, value, color) in zip(cols, glance_items):
        col.markdown(
            f"""<div class="at-a-glance-card" style="background:{color};">
                <div class="label">{label}</div><div class="value">{value}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    # --- 配当性向まわりのカード(累進配当・増配/減配回数・増配率など) ---
    div_growth_1y = cagr_over_n_years(yearly["dividend_per_share"], 1) if not yearly.empty else None
    div_growth_3y = cagr_over_n_years(yearly["dividend_per_share"], 3) if not yearly.empty else None
    div_growth_5y = cagr_over_n_years(yearly["dividend_per_share"], 5) if not yearly.empty else None
    div_growth_10y = cagr_over_n_years(yearly["dividend_per_share"], 10) if not yearly.empty else None

    div_cards = [
        ("累進配当", "○" if div_stats["progressive"] else ("-" if div_stats["progressive"] is None else "×"), "#E0F2F1"),
        ("連続増配", f"{div_stats['consecutive_increase']}年", "#E0F2F1"),
        ("増配回数", f"{div_stats['increase']}回", "#E0F2F1"),
        ("減配回数", f"{div_stats['decrease']}回", "#FCE9E9"),
        ("非減配年数", f"{div_stats['no_decrease_years']}年", "#E0F2F1"),
        ("増配率(1年)", f"{div_growth_1y * 100:.1f}%" if div_growth_1y is not None else "-", "#E3F2FD"),
        ("増配率(3年)", f"{div_growth_3y * 100:.1f}%" if div_growth_3y is not None else "-", "#E3F2FD"),
        ("増配率(5年)", f"{div_growth_5y * 100:.1f}%" if div_growth_5y is not None else "-", "#E3F2FD"),
        ("増配率(10年)", f"{div_growth_10y * 100:.1f}%" if div_growth_10y is not None else "-", "#E3F2FD"),
    ]
    cols2 = st.columns(len(div_cards))
    for col, (label, value, color) in zip(cols2, div_cards):
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

    # --- レーダーチャート(本銘柄/業種中央) + 株価/相対株価 ---
    period_label = st.radio("株価チャートの期間", list(PERIOD_DAYS.keys()), index=3, horizontal=True)
    days = PERIOD_DAYS[period_label]

    col_radar1, col_radar2, col_price, col_rel = st.columns(4)

    labels = list(CATEGORY_COLS.values())
    values = [row[k] if pd.notna(row[k]) else 0 for k in CATEGORY_COLS.keys()]
    axis_grades = [score_to_grade(row[k], _GRADE_THRESHOLDS) or "-" for k in CATEGORY_COLS.keys()]
    sector_peers = df[df["sector"] == row["sector"]]
    sector_medians = [sector_peers[k].median() for k in CATEGORY_COLS.keys()]
    sector_median_grades = [score_to_grade(v, _GRADE_THRESHOLDS) or "-" for v in sector_medians]

    with col_radar1:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + values[:1], theta=labels + labels[:1], fill="toself", name="本銘柄",
            text=[f"{v:.0f}({g})" for v, g in zip(values, axis_grades)] + [f"{values[0]:.0f}({axis_grades[0]})"],
            mode="lines+markers+text", textposition="top center",
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False)
        st.caption(f"スコア(総合評価 {row['grade']})")
        st.plotly_chart(_compact(fig), use_container_width=True)

    with col_radar2:
        fig_sector = go.Figure()
        fig_sector.add_trace(go.Scatterpolar(
            r=sector_medians + sector_medians[:1], theta=labels + labels[:1], fill="toself",
            name="業種中央値", line=dict(color="orange"),
            text=[f"{v:.0f}({g})" for v, g in zip(sector_medians, sector_median_grades)] + [f"{sector_medians[0]:.0f}"],
            mode="lines+markers+text", textposition="top center",
        ))
        fig_sector.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False)
        st.caption(f"業種中央スコア({row['sector']})")
        st.plotly_chart(_compact(fig_sector), use_container_width=True)

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
    if yearly.empty:
        st.info("この銘柄の複数年推移データはまだありません(次回のfetch_fundamentals.py/fetch_edinet_history.py実行で蓄積されます)。")
    else:
        revenue_cagr = cagr(yearly["revenue"])
        cagr_text = f"売上高{len(yearly) - 1}年CAGR: {revenue_cagr * 100:.1f}%" if revenue_cagr is not None else ""

        x = yearly["fy_label"]

        TXT = dict(textposition="outside", textfont=dict(size=8))
        TXT_LINE = dict(textposition="top center", textfont=dict(size=8))

        row1a, row1b = st.columns(2)
        with row1a:
            st.caption(f"業績推移　{cagr_text}")
            fig_perf = go.Figure()
            fig_perf.add_bar(x=x, y=yearly["revenue"], name="売上高", marker_color="#BBBBBB",
                              text=_labels(yearly["revenue"]), **TXT)
            fig_perf.add_trace(go.Scatter(x=x, y=yearly["operating_margin"] * 100, name="営業利益率(%)", yaxis="y2",
                                           line=dict(color="#D62728"), text=_labels(yearly["operating_margin"] * 100, "{:.1f}%"), **TXT_LINE))
            fig_perf.add_trace(go.Scatter(x=x, y=yearly["net_margin"] * 100, name="純利益率(%)", yaxis="y2",
                                           line=dict(color="#FF7F0E"), text=_labels(yearly["net_margin"] * 100, "{:.1f}%"), **TXT_LINE))
            fig_perf.add_trace(go.Scatter(x=x, y=yearly["roe"] * 100, name="ROE(%)", yaxis="y2",
                                           line=dict(color="#2CA02C"), text=_labels(yearly["roe"] * 100, "{:.1f}%"), **TXT_LINE))
            fig_perf.update_layout(yaxis=dict(title="売上高"), yaxis2=dict(title="利益率(%)", overlaying="y", side="right"))
            st.plotly_chart(_yearly_chart_layout(_compact(fig_perf)), use_container_width=True)

        with row1b:
            st.caption("配当推移(1株配当・EPS)")
            fig_div = go.Figure()
            fig_div.add_bar(x=x, y=yearly["dividend_per_share"], name="1株配当", marker_color="#BBBBBB",
                             text=_labels(yearly["dividend_per_share"], "{:.1f}"), **TXT)
            fig_div.add_trace(go.Scatter(x=x, y=yearly["eps"], name="EPS", yaxis="y2",
                                          line=dict(color="#D62728"), text=_labels(yearly["eps"], "{:.1f}"), **TXT_LINE))
            fig_div.update_layout(yaxis=dict(title="1株配当"), yaxis2=dict(title="EPS", overlaying="y", side="right"))
            st.plotly_chart(_yearly_chart_layout(_compact(fig_div)), use_container_width=True)

        row2a, row2b = st.columns(2)
        with row2a:
            st.caption("財務推移(純資産・負債・自己資本比率)")
            fig_bs = go.Figure()
            fig_bs.add_bar(x=x, y=yearly["equity"], name="純資産", marker_color="#4C72B0",
                            text=_labels(yearly["equity"]), **TXT)
            fig_bs.add_bar(x=x, y=yearly["total_liabilities"], name="負債", marker_color="#DD8452",
                            text=_labels(yearly["total_liabilities"]), **TXT)
            fig_bs.add_trace(go.Scatter(x=x, y=yearly["equity_ratio"], name="自己資本比率(%)", yaxis="y2",
                                         line=dict(color="#2CA02C"), text=_labels(yearly["equity_ratio"], "{:.1f}%"), **TXT_LINE))
            fig_bs.update_layout(barmode="group", yaxis=dict(title="金額"), yaxis2=dict(title="自己資本比率(%)", overlaying="y", side="right"))
            st.plotly_chart(_yearly_chart_layout(_compact(fig_bs)), use_container_width=True)

        with row2b:
            st.caption("キャッシュフロー推移")
            fig_cf = go.Figure()
            fig_cf.add_bar(x=x, y=yearly["operating_cf"], name="営業CF", marker_color="#4C72B0")
            fig_cf.add_bar(x=x, y=yearly["investing_cf"], name="投資CF", marker_color="#DD8452")
            fig_cf.add_bar(x=x, y=yearly["financing_cf"], name="財務CF", marker_color="#BBBBBB")
            fig_cf.add_trace(go.Scatter(x=x, y=yearly["free_cf"], name="フリーCF", line=dict(color="#2CA02C")))
            fig_cf.update_layout(barmode="relative")
            st.plotly_chart(_yearly_chart_layout(_compact(fig_cf)), use_container_width=True)

        if "buyback_amount" in yearly.columns and yearly["buyback_amount"].notna().any():
            st.caption("自己株式の取得")
            fig_buyback = go.Figure()
            fig_buyback.add_bar(x=x, y=yearly["buyback_amount"], name="自己株式取得額", marker_color="#4C72B0",
                                 text=_labels(yearly["buyback_amount"]), **TXT)
            st.plotly_chart(_yearly_chart_layout(_compact(fig_buyback)), use_container_width=True)
