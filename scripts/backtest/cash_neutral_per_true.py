"""清原達郎氏(『わが投資術』)の原文どおりの「キャッシュニュートラルPER」を検証する。

原文の定義(p.52): キャッシュニュートラルPER = PER × (1 - ネットキャッシュ比率)
ネットキャッシュ比率 = (流動資産 + 投資有価証券×0.7 - 負債) / 時価総額 (p.51)

net_cash_ratio_kiyohara.pyではネットキャッシュ比率のみで三分位分割したが、
原文はPERとの複合指標を主軸に提示しているため、こちらも検証した。

結果: net_cash_ratio単体の三分位(低53.05%/高70.38%、24ヶ月)よりも、
この複合指標の五分位の方が明確に効果が大きい(Q1最割安+78.47%・中央値+56.04%
に対しQ5最割高+47.99%・中央値+34.35%)。2022〜2024年の全年で頑健
(Q1/Q5: 71.99/54.35、76.52/43.89、87.85/52.01)。この結果を受けて
scripts/decisions/decide_quality_timing_v4.py(v4)の条件を、単純な
ネットキャッシュ比率上位1/3からキャッシュニュートラルPER下位40%
(五分位のQ1+Q2相当)に変更した。

既知の制約: current_assets・investment_securitiesの取得はEDINETの貸借対照表
本体タグ(jppfs_cor:CurrentAssets/InvestmentSecurities)に依存しており、
銀行業(0%)・保険業(7.1%)はほぼ完全にデータが取得できない(貸借対照表が
流動/固定資産の区分を使わないため)。全active銘柄のうち約70%(2,477/3,549)
がカバー対象。

実行: python scripts/backtest/cash_neutral_per_true.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection, get_backtest_connection
from common.technical import calc_indicators
from scoring.compute_scores import load_config, score_to_grade
from backtest.quality_score_v2_backtest import CATEGORY_METRICS_V2, compute_score_per_cohort
from backtest.decision_engine_extended_backtest import build_fundamental_panel_bt, attach_value_momentum_bt, load_bt_price_history
from backtest.decision_engine_comparison_backtest import build_cohort_windows, find_applicable_cohort

conn = get_connection()
bt_conn = get_backtest_connection()
scoring_config = load_config()

panel = build_fundamental_panel_bt(conn, bt_conn)
panel = attach_value_momentum_bt(bt_conn, panel)

prod_weights_raw = scoring_config['category_weights']
weight_map = {
    "safety": prod_weights_raw.get("safety", 0), "growth": prod_weights_raw.get("growth", 0),
    "profitability": prod_weights_raw.get("profitability", 0), "efficiency": prod_weights_raw.get("efficiency", 0),
    "shareholder_return": prod_weights_raw.get("shareholder_return", 0), "value": prod_weights_raw.get("valuation", 0),
    "momentum": prod_weights_raw.get("momentum", 0), "earnings_quality": 0.0,
}
total_w = sum(weight_map.values())
weight_map = {k: v / total_w for k, v in weight_map.items()}

panel = compute_score_per_cohort(panel, CATEGORY_METRICS_V2, weight_map, 'total_score_prodw')
panel['grade_prodw'] = panel['total_score_prodw'].apply(lambda v: score_to_grade(v, scoring_config['grade_thresholds']))
panel = panel.dropna(subset=['total_score_prodw', 'signal_date', 'grade_prodw', 'score_value'])

bs_df = pd.read_sql_query(
    """SELECT ticker, fiscal_year_end, current_assets, investment_securities,
              total_liabilities, net_income AS ni_yearly
       FROM fundamentals_yearly""",
    conn, parse_dates=["fiscal_year_end"]
)
panel = panel.merge(bs_df, on=["ticker", "fiscal_year_end"], how="left")

panel["market_cap_approx"] = panel["per"] * panel["ni_yearly"]
valid = (
    panel["market_cap_approx"].notna() & (panel["market_cap_approx"] > 0)
    & panel["current_assets"].notna() & panel["investment_securities"].notna()
    & panel["total_liabilities"].notna() & panel["per"].notna() & (panel["per"] > 0)
)
panel["net_cash_kiyohara"] = None
panel.loc[valid, "net_cash_kiyohara"] = (
    panel.loc[valid, "current_assets"] + 0.7 * panel.loc[valid, "investment_securities"] - panel.loc[valid, "total_liabilities"]
) / panel.loc[valid, "market_cap_approx"]

# キャッシュニュートラルPER = PER × (1 - ネットキャッシュ比率)
# ネットキャッシュ比率が1以上(会社がタダで買えるほど割安)だとマイナスになるが、
# それ自体が「非常識に割安」というシグナルなので、そのまま計算する(原文どおり)
panel["cash_neutral_per_true"] = None
panel.loc[valid, "cash_neutral_per_true"] = panel.loc[valid, "per"] * (1 - panel.loc[valid, "net_cash_kiyohara"])

print(f"清原式ネットキャッシュ比率/キャッシュニュートラルPER 計算可能件数: {valid.sum()} / 全体{len(panel)}件")

panel = build_cohort_windows(panel)
target_df = panel[panel['grade_prodw'].isin(['B', 'C'])]
valuation_median = target_df['score_value'].median()

HORIZONS_TRADING_DAYS = {"252営業日(12ヶ月)": 252, "504営業日(24ヶ月)": 504}

records = []
tickers = panel['ticker'].unique()
start = time.monotonic()
for n, ticker in enumerate(tickers, start=1):
    rows = panel[panel['ticker'] == ticker]
    price_df = load_bt_price_history(bt_conn, ticker)
    if len(price_df) < 126:
        continue
    ind = calc_indicators(price_df.copy())
    close = ind['Close']
    dates = price_df.index

    rolling_high = close.rolling(252, min_periods=126).max()
    near_high = close >= rolling_high * 0.95
    near_high_trigger = near_high & ~near_high.shift(1).fillna(False)

    idx = [i for i, v in enumerate(near_high_trigger) if v]
    for i in idx:
        day = dates[i]
        cohort = find_applicable_cohort(rows, day)
        if cohort is None or cohort['grade_prodw'] not in ('B', 'C'):
            continue
        if cohort['score_value'] < valuation_median:
            continue
        entry_price = close.iloc[i]
        if pd.isna(entry_price) or entry_price <= 0:
            continue
        rec = {
            "ticker": ticker, "date": day,
            "net_cash_kiyohara": cohort['net_cash_kiyohara'],
            "cash_neutral_per_true": cohort['cash_neutral_per_true'],
        }
        for h_label, h_days in HORIZONS_TRADING_DAYS.items():
            exit_pos = i + h_days
            if exit_pos >= len(dates):
                continue
            exit_price = close.iloc[exit_pos]
            if pd.notna(exit_price) and exit_price > 0:
                rec[h_label] = (exit_price / entry_price - 1) * 100
        records.append(rec)
    if n % 1000 == 0:
        print(f'[{n}/{len(tickers)}] {time.monotonic()-start:.0f}s')

df = pd.DataFrame(records)
df["cash_neutral_per_true"] = pd.to_numeric(df["cash_neutral_per_true"], errors="coerce")
df_valid = df.dropna(subset=["cash_neutral_per_true"]).copy()
print(f"\nサンプル総数(B/C×52週高値×割安性上位50%): {len(df)}, うち算出可={len(df_valid)}, 経過{time.monotonic()-start:.0f}秒\n")

df_valid["quintile"] = pd.qcut(df_valid["cash_neutral_per_true"], 5, labels=["Q1(最割安)", "Q2", "Q3", "Q4", "Q5(最割高)"], duplicates="drop")
print("=== キャッシュニュートラルPER(原文どおりの複合指標) 五分位別フォワードリターン ===")
for h in HORIZONS_TRADING_DAYS:
    sub = df_valid.dropna(subset=[h])
    print(f"\n[{h}]")
    print(sub.groupby("quintile")[h].agg(["size", "mean", "median"]).round(2))

print("\n=== 年別頑健性(24ヶ月、Q1 vs Q5) ===")
df_valid['year'] = pd.to_datetime(df_valid['date']).dt.year
h24 = "504営業日(24ヶ月)"
for yr, g in df_valid.dropna(subset=[h24]).groupby('year'):
    q1 = g[g['quintile'] == 'Q1(最割安)'][h24]
    q5 = g[g['quintile'] == 'Q5(最割高)'][h24]
    if len(q1) < 20 or len(q5) < 20:
        continue
    print(f"{yr}: Q1(最割安) n={len(q1)} 平均={q1.mean():.2f}%  |  Q5(最割高) n={len(q5)} 平均={q5.mean():.2f}%")

conn.close()
bt_conn.close()
