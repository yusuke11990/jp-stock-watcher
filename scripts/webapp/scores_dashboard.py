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
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection  # noqa: E402
from scoring.compute_scores import load_config, score_to_grade  # noqa: E402

_SCORING_CONFIG = load_config()
_GRADE_THRESHOLDS = _SCORING_CONFIG["grade_thresholds"]

CATEGORY_COLS = {
    "score_safety": "安全性",
    "score_growth": "成長性",
    "score_profitability": "収益性",
    "score_efficiency": "効率性",
    "score_valuation": "割安性",
    "score_shareholder_return": "還元性",
}
PERIOD_DAYS = {"1ヶ月": 30, "3ヶ月": 90, "6ヶ月": 180, "1年": 365, "3年": 365 * 3, "5年": 365 * 5, "10年": 365 * 10}
PERIOD_YF = {"1ヶ月": "1mo", "3ヶ月": "3mo", "6ヶ月": "6mo", "1年": "1y", "3年": "3y", "5年": "5y", "10年": "10y"}
# price_dailyはローリング取得の都合上、実質1年分程度しか保持していない。
# それを超える期間はDBに無いため、選択銘柄のみその場でyfinanceから取得する(全銘柄分をDBに溜めると
# 10年分で500MB超になりGitHubコミット運用が破綻するため、意図的にオンデマンド取得にしている)。
DB_HISTORY_DAYS = 400
COMPACT_HEIGHT = 230
YEARLY_HEIGHT = 420
# fundamentals_yearlyは円単位で保存されているため、しけなぎ/バフェットコード風の「百万円」表示に変換する
YEN_TO_MILLION_COLS = {
    "revenue", "operating_income", "ordinary_income", "net_income",
    "total_assets", "total_liabilities", "equity",
    "operating_cf", "investing_cf", "financing_cf", "free_cf", "cash_and_equivalents", "buyback_amount",
}

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
        border-radius: 6px; padding: 6px 6px; text-align: center;
        height: 62px; display: flex; flex-direction: column; justify-content: center; align-items: center;
    }
    .at-a-glance-card .label {font-size: 0.68rem; opacity: 0.75; line-height: 1.15;}
    .at-a-glance-card .value {font-size: 1.1rem; font-weight: 700; line-height: 1.2;}
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
    """しけなぎ/バフェット・コード風: 全年度を等間隔表示、太めのバー、大きめの文字"""
    fig.update_xaxes(tickangle=-40, showgrid=False, type="category", tickmode="linear", dtick=1)
    fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    fig.update_layout(
        height=YEARLY_HEIGHT,
        margin=dict(l=15, r=15, t=32, b=40),
        font=dict(size=13, family="Hiragino Sans, Yu Gothic, Meiryo, sans-serif", color="#333"),
        legend=dict(font=dict(size=11), orientation="h", yanchor="bottom", y=1.0),
        plot_bgcolor="white",
        hovermode="x unified",
        bargap=0.15,
        bargroupgap=0.08,
    )
    for trace in fig.data:
        if trace.type == "scatter" and trace.mode is None:
            trace.mode = "lines+markers"
            trace.marker = dict(size=6)
    return fig


def _to_million(series: pd.Series) -> pd.Series:
    return series / 1e6


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
        f.per, f.pbr, f.roe, f.dividend_yield, f.market_cap, f.equity_ratio, f.revenue_growth_3y_cagr, f.payout_ratio,
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


@st.cache_data(ttl=3600)
def load_price_history_yf(ticker: str, yf_period: str) -> pd.DataFrame:
    """DBに無い長期(3年超)の株価は選択銘柄のみその場でyfinanceから取得する。"""
    try:
        raw = yf.download(ticker, period=yf_period, progress=False, auto_adjust=False)
    except Exception:
        return pd.DataFrame(columns=["date", "close"])
    if raw.empty:
        return pd.DataFrame(columns=["date", "close"])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Close"]].reset_index().rename(columns={"Date": "date", "Close": "close"})
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

st.session_state.setdefault("jump_to_ticker", None)
st.session_state.setdefault("jump_pending", False)


def _render_clickable_decisions_table(table: pd.DataFrame, key: str) -> None:
    """コード列を「.T」無しで表示し、行クリックでその銘柄の詳細にジャンプできる表を描画する。"""
    table = table.reset_index(drop=True)
    raw_tickers = table["コード"].tolist()
    display_table = table.copy()
    display_table["コード"] = display_table["コード"].str.replace(".T", "", regex=False)
    event = st.dataframe(
        display_table.round(2), use_container_width=True, hide_index=True, height=300,
        on_select="rerun", selection_mode="single-row", key=key,
    )
    if event and event.selection and event.selection.rows:
        clicked_ticker = raw_tickers[event.selection.rows[0]]
        if clicked_ticker != st.session_state["jump_to_ticker"]:
            st.session_state["jump_to_ticker"] = clicked_ticker
            st.session_state["jump_pending"] = True
            st.rerun()


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

    st.caption("行をクリックするとその銘柄の詳細に移動します")
    col_buy, col_sell = st.columns(2)
    with col_buy:
        buy_df = version_df[version_df["判断"] == "buy"].drop(columns=["判断", "rule_version"])
        st.markdown(f"**買い候補 ({len(buy_df)}件)**")
        _render_clickable_decisions_table(buy_df, key=f"buy_table_{version_sel}")
    with col_sell:
        sell_df = version_df[version_df["判断"] == "sell"].drop(columns=["判断", "rule_version"])
        st.markdown(f"**売り候補 ({len(sell_df)}件)**")
        _render_clickable_decisions_table(sell_df, key=f"sell_table_{version_sel}")

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
    display_df["コード"] = display_df["コード"].str.replace(".T", "", regex=False)
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

ticker_options = (filtered["ticker"].str.replace(".T", "", regex=False) + " " + filtered["name"].fillna("")).tolist()

# 買い/売り一覧のクリック、またはサイドバー絞り込みで選択中の銘柄が候補から消えても
# 選択状態を保てるよう、現在値がoptionsに無ければ先頭に補って残す
current_val = st.session_state.get("ticker_selectbox")
if current_val and current_val not in ticker_options:
    ticker_options = [current_val] + ticker_options

if st.session_state["jump_pending"]:
    jump_ticker = st.session_state["jump_to_ticker"]
    match = df[df["ticker"] == jump_ticker]
    if not match.empty:
        jump_label = f"{jump_ticker.replace('.T', '')} {match.iloc[0]['name']}"
        if jump_label not in ticker_options:
            ticker_options = [jump_label] + ticker_options
        st.session_state["ticker_selectbox"] = jump_label
    st.session_state["jump_pending"] = False

selected_label = st.selectbox("銘柄を選択", ticker_options, key="ticker_selectbox")

if selected_label:
    selected_ticker = selected_label.split(" ")[0] + ".T"
    row = df[df["ticker"] == selected_ticker].iloc[0]

    yearly = load_yearly(selected_ticker)
    mix_ratio = (row["per"] * row["pbr"]) if pd.notna(row["per"]) and pd.notna(row["pbr"]) else None
    div_stats = dividend_streak_stats(yearly["dividend_per_share"]) if not yearly.empty else dividend_streak_stats(pd.Series(dtype=float))

    def _render_glance_row(items: list[tuple[str, str, str]]) -> None:
        cols = st.columns(len(items))
        for col, (label, value, color) in zip(cols, items):
            col.markdown(
                f"""<div class="at-a-glance-card" style="background:{color};">
                    <div class="label">{label}</div><div class="value">{value}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    # --- ひと目カード(バフェット・コード風の要約カード) ---
    glance_items = [
        ("株価", f"{row['price']:,.0f}" if pd.notna(row["price"]) else "-", "#F1F1F1"),
        ("時価総額(億円)", f"{row['market_cap'] / 1e8:,.0f}" if pd.notna(row["market_cap"]) else "-", "#FFF6E0"),
        ("PER(倍)", f"{row['per']:.1f}" if pd.notna(row["per"]) else "-", "#E8F5E9"),
        ("PBR(倍)", f"{row['pbr']:.2f}" if pd.notna(row["pbr"]) else "-", "#E8F5E9"),
        ("ROE(%)", f"{row['roe'] * 100:.1f}" if pd.notna(row["roe"]) else "-", "#FCE9E9"),
        ("MIX係数(PER×PBR)", f"{mix_ratio:.1f}" if mix_ratio is not None else "-", "#EDE7F6"),
        ("自己資本比率(%)", f"{row['equity_ratio']:.1f}" if pd.notna(row["equity_ratio"]) else "-", "#E6F0FA"),
        ("配当利回り(%)", f"{row['dividend_yield']:.2f}" if pd.notna(row["dividend_yield"]) else "-", "#F1F1F1"),
        ("配当性向(%)", f"{row['payout_ratio'] * 100:.1f}" if pd.notna(row["payout_ratio"]) else "-", "#F1F1F1"),
        ("総合評価", f"{row['total_score']:.1f}({row['grade']})" if pd.notna(row["total_score"]) else "-", "#FFF6E0"),
    ]
    _render_glance_row(glance_items)

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
        ("非減配年", f"{div_stats['no_decrease_years']}年", "#E0F2F1"),
        ("増配率(1年)", f"{div_growth_1y * 100:.1f}%" if div_growth_1y is not None else "-", "#E3F2FD"),
        ("増配率(3年)", f"{div_growth_3y * 100:.1f}%" if div_growth_3y is not None else "-", "#E3F2FD"),
        ("増配率(5年)", f"{div_growth_5y * 100:.1f}%" if div_growth_5y is not None else "-", "#E3F2FD"),
        ("増配率(10年)", f"{div_growth_10y * 100:.1f}%" if div_growth_10y is not None else "-", "#E3F2FD"),
    ]
    _render_glance_row(div_cards)

    st.caption(
        f"{row['ticker'].replace('.T', '')} {row['name']}　業種: {row['sector']}　"
        f"総合スコア {row['total_score']:.1f}({row['grade']})　業種内 {int(row['sector_rank'])}/{int(row['sector_size'])}位"
    )

    # --- レーダーチャート用データ(fundamentals_yearlyの有無に関わらず表示できるよう先に用意) ---
    labels = list(CATEGORY_COLS.values())
    values = [row[k] if pd.notna(row[k]) else 0 for k in CATEGORY_COLS.keys()]
    axis_grades = [score_to_grade(row[k], _GRADE_THRESHOLDS) or "-" for k in CATEGORY_COLS.keys()]
    sector_peers = df[df["sector"] == row["sector"]]
    sector_medians = [sector_peers[k].median() for k in CATEGORY_COLS.keys()]
    sector_median_grades = [score_to_grade(v, _GRADE_THRESHOLDS) or "-" for v in sector_medians]

    # --- 1行目: 株価 + AI判断日 ---
    period_label = st.radio("株価チャートの期間", list(PERIOD_DAYS.keys()), index=3, horizontal=True)
    days = PERIOD_DAYS[period_label]
    price_source_note = None
    if days > DB_HISTORY_DAYS:
        price_hist = load_price_history_yf(selected_ticker, PERIOD_YF[period_label])
        if price_hist.empty:
            # クラウド環境からyfinanceへのアクセスが失敗する場合があるため、DB保有分にフォールバック
            price_hist = load_price_history(selected_ticker, days)
            if not price_hist.empty:
                price_source_note = "長期データの取得に失敗したため、DBに保存されている範囲のみ表示しています"
    else:
        price_hist = load_price_history(selected_ticker, days)
    decisions = load_decisions(selected_ticker, days)

    st.caption("株価 + AI判断日")
    if price_source_note:
        st.caption(f"⚠ {price_source_note}")
    if price_hist.empty:
        st.info("価格データがありません")
    else:
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=price_hist["date"], y=price_hist["close"], name="株価",
            line=dict(width=2, color="#1f6fd6"),
            fill="tozeroy", fillcolor="rgba(31, 111, 214, 0.12)",
        ))
        for action, color, symbol in [("buy", "green", "triangle-up"), ("sell", "red", "triangle-down")]:
            sub = decisions[decisions["action"] == action]
            if not sub.empty:
                fig_price.add_trace(go.Scatter(
                    x=sub["decision_date"], y=sub["price_at_decision"], mode="markers", name=action,
                    marker=dict(symbol=symbol, size=12, color=color),
                ))
        last_row = price_hist.iloc[-1]
        fig_price.add_annotation(
            x=last_row["date"], y=last_row["close"], text=f"  {last_row['close']:,.0f}  ",
            showarrow=False, xanchor="left", yanchor="middle", xshift=4,
            font=dict(color="white", size=12), bgcolor="#1f6fd6", borderpad=3,
        )
        fig_price = _compact(fig_price, height=340)
        fig_price.update_layout(margin=dict(l=35, r=70, t=28, b=25))
        fig_price.update_yaxes(side="right", tickformat=",")
        if period_label == "1ヶ月":
            # 短期間は月表記だと同じラベルが並んでしまうため、1週間おきに日付を刻む
            fig_price.update_xaxes(tickformat="%-m/%-d", dtick=7 * 24 * 60 * 60 * 1000)
        elif period_label in ("3ヶ月", "6ヶ月"):
            fig_price.update_xaxes(tickformat="%Y/%-m", dtick="M1")
        else:
            fig_price.update_xaxes(tickformat="%Y/%-m")
        fig_price.update_xaxes(hoverformat="%Y/%-m/%-d")
        st.plotly_chart(fig_price, use_container_width=True)

    if not decisions.empty:
        with st.expander(f"この期間のAI判断ログ({len(decisions)}件)"):
            st.dataframe(decisions.rename(columns={
                "decision_date": "判断日", "action": "判断", "price_at_decision": "株価", "reason": "理由",
            }), use_container_width=True, hide_index=True)

    yearly_available = not yearly.empty
    if not yearly_available:
        st.info("この銘柄の複数年推移データはまだありません(次回のfetch_fundamentals.py/fetch_edinet_history.py実行で蓄積されます)。")
    else:
        revenue_cagr = cagr(yearly["revenue"])
        cagr_text = f"売上高{len(yearly) - 1}年CAGR: {revenue_cagr * 100:.1f}%" if revenue_cagr is not None else ""
        x = yearly["fy_label"]
        TXT = dict(textposition="outside", textfont=dict(size=10))

        def _txt_line(color: str) -> dict:
            """折れ線の数値ラベルを線の色と揃えて表示する(色以外は凡例で判別できるようにする)"""
            return dict(textposition="top center", textfont=dict(size=10, color=color))

        # --- 2行目: 業績推移, 配当推移 ---
        row1a, row1b = st.columns(2)
        with row1a:
            st.caption(f"業績推移(百万円)　{cagr_text}")
            revenue_m = _to_million(yearly["revenue"])
            fig_perf = go.Figure()
            fig_perf.add_bar(x=x, y=revenue_m, name="売上高", marker_color="#BBBBBB",
                              text=_labels(revenue_m), **TXT)
            fig_perf.add_trace(go.Scatter(x=x, y=yearly["operating_margin"] * 100, name="営業利益率(%)", yaxis="y2",
                                           line=dict(color="#D62728", width=2.5), text=_labels(yearly["operating_margin"] * 100, "{:.2f}%"), **_txt_line("#D62728")))
            fig_perf.add_trace(go.Scatter(x=x, y=yearly["net_margin"] * 100, name="純利益率(%)", yaxis="y2",
                                           line=dict(color="#FF7F0E", width=2.5), text=_labels(yearly["net_margin"] * 100, "{:.2f}%"), **_txt_line("#FF7F0E")))
            fig_perf.add_trace(go.Scatter(x=x, y=yearly["roe"] * 100, name="ROE(%)", yaxis="y2",
                                           line=dict(color="#2CA02C", width=2.5), text=_labels(yearly["roe"] * 100, "{:.2f}%"), **_txt_line("#2CA02C")))
            fig_perf.update_layout(yaxis=dict(), yaxis2=dict(overlaying="y", side="right", showgrid=False))
            st.plotly_chart(_yearly_chart_layout(fig_perf), use_container_width=True)

        with row1b:
            st.caption("配当推移(1株配当・EPS)")
            fig_div = go.Figure()
            fig_div.add_bar(x=x, y=yearly["dividend_per_share"], name="1株配当", marker_color="#BBBBBB",
                             text=_labels(yearly["dividend_per_share"], "{:.1f}"), **TXT)
            fig_div.add_trace(go.Scatter(x=x, y=yearly["eps"], name="EPS", yaxis="y2",
                                          line=dict(color="#D62728", width=2.5), text=_labels(yearly["eps"], "{:.1f}"), **_txt_line("#D62728")))
            fig_div.update_layout(yaxis=dict(), yaxis2=dict(overlaying="y", side="right", showgrid=False))
            st.plotly_chart(_yearly_chart_layout(fig_div), use_container_width=True)

    # --- 3行目: 財務推移, スコア, 業種中央スコア ---
    row2a, row2b, row2c = st.columns([2, 1, 1])
    with row2a:
        if not yearly_available:
            st.caption("財務推移")
            st.info("複数年推移データがありません")
        else:
            st.caption("財務推移(百万円・自己資本比率)")
            equity_m = _to_million(yearly["equity"])
            liabilities_m = _to_million(yearly["total_liabilities"])
            fig_bs = go.Figure()
            fig_bs.add_bar(x=x, y=equity_m, name="純資産", marker_color="#4C72B0",
                            text=_labels(equity_m), **TXT)
            fig_bs.add_bar(x=x, y=liabilities_m, name="負債", marker_color="#DD8452",
                            text=_labels(liabilities_m), **TXT)
            fig_bs.add_trace(go.Scatter(x=x, y=yearly["equity_ratio"], name="自己資本比率(%)", yaxis="y2",
                                         line=dict(color="#2CA02C", width=2.5), text=_labels(yearly["equity_ratio"], "{:.1f}%"), **_txt_line("#2CA02C")))
            fig_bs.update_layout(barmode="group", yaxis=dict(), yaxis2=dict(overlaying="y", side="right", showgrid=False))
            st.plotly_chart(_yearly_chart_layout(fig_bs), use_container_width=True)

    with row2b:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + values[:1], theta=labels + labels[:1], fill="toself", name="本銘柄",
            text=[f"{v:.0f}({g})" for v, g in zip(values, axis_grades)] + [f"{values[0]:.0f}({axis_grades[0]})"],
            mode="lines+markers+text", textposition="top center",
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False)
        st.caption(f"スコア(総合評価 {row['grade']})")
        st.plotly_chart(_compact(fig, height=YEARLY_HEIGHT), use_container_width=True)

    with row2c:
        fig_sector = go.Figure()
        fig_sector.add_trace(go.Scatterpolar(
            r=sector_medians + sector_medians[:1], theta=labels + labels[:1], fill="toself",
            name="業種中央値", line=dict(color="orange"),
            text=[f"{v:.0f}({g})" for v, g in zip(sector_medians, sector_median_grades)] + [f"{sector_medians[0]:.0f}"],
            mode="lines+markers+text", textposition="top center",
        ))
        fig_sector.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False)
        st.caption(f"業種中央スコア({row['sector']})")
        st.plotly_chart(_compact(fig_sector, height=YEARLY_HEIGHT), use_container_width=True)

    # --- 4行目: キャッシュフロー推移, 自己株式の取得 ---
    if yearly_available:
        row3a, row3b = st.columns(2)
        with row3a:
            st.caption("キャッシュフロー推移(百万円)")
            fig_cf = go.Figure()
            fig_cf.add_bar(x=x, y=_to_million(yearly["operating_cf"]), name="営業CF", marker_color="#4C72B0")
            fig_cf.add_bar(x=x, y=_to_million(yearly["investing_cf"]), name="投資CF", marker_color="#DD8452")
            fig_cf.add_bar(x=x, y=_to_million(yearly["financing_cf"]), name="財務CF", marker_color="#BBBBBB")
            fig_cf.add_trace(go.Scatter(x=x, y=_to_million(yearly["free_cf"]), name="フリーCF", line=dict(color="#2CA02C", width=2.5)))
            fig_cf.update_layout(barmode="relative", yaxis=dict())
            st.plotly_chart(_yearly_chart_layout(fig_cf), use_container_width=True)

        with row3b:
            if "buyback_amount" in yearly.columns and yearly["buyback_amount"].notna().any():
                st.caption("自己株式の取得(百万円)")
                buyback_m = _to_million(yearly["buyback_amount"])
                fig_buyback = go.Figure()
                fig_buyback.add_bar(x=x, y=buyback_m, name="自己株式取得額", marker_color="#4C72B0",
                                     text=_labels(buyback_m), **TXT)
                fig_buyback.update_layout(yaxis=dict())
                st.plotly_chart(_yearly_chart_layout(fig_buyback), use_container_width=True)
            else:
                st.caption("自己株式の取得")
                st.info("自己株式取得のデータがありません")
