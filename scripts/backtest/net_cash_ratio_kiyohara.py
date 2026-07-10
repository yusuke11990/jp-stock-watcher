"""清原達郎氏(『わが投資術』)の正式なネットキャッシュ比率の定義:
  ネットキャッシュ比率 = (流動資産 + 0.7×投資有価証券 - 負債) / 時価総額
を、EDINETからバックフィルしたcurrent_assets・investment_securitiesを使って
実際に計算し、既存コホート(グレードB/C×52週高値接近×割安性上位50%)内での
フォワードリターン予測力を検証する。

結果: 三分位の「高」(現金潤沢)が「低」を24ヶ月保有で大きく上回った
(低53.05%・中央値33.78% vs 高70.38%・中央値51.59%)。2022〜2024年の
全年で頑健(低/高: 49.47/63.66、47.71/69.91、64.41/76.00)。この結果を受けて
scripts/decisions/decide_quality_timing_v4.py(v4)にネットキャッシュ比率上位1/3の
条件を追加した。current_assets・investment_securitiesはEDINETの貸借対照表本体
タグ(jppfs_cor:CurrentAssets/InvestmentSecurities)から取得しており、当期末・
前期末の2期分しか開示されないため、scripts/fetch_edinet_history.pyの書類取得
間隔をDEDUP_MIN_GAP_DAYS(約2年)に短縮し、scripts/fetch_edinet_index.pyの
デフォルトスキャン世代も0〜10年分に拡張して全期間をカバーできるようにした。

実行: python scripts/backtest/net_cash_ratio_kiyohara.py
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
    & panel["total_liabilities"].notna()
)
panel["net_cash_kiyohara"] = None
panel.loc[valid, "net_cash_kiyohara"] = (
    panel.loc[valid, "current_assets"] + 0.7 * panel.loc[valid, "investment_securities"] - panel.loc[valid, "total_liabilities"]
) / panel.loc[valid, "market_cap_approx"]

print(f"清原式ネットキャッシュ比率 計算可能件数: {valid.sum()} / 全体{len(panel)}件")

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
df["net_cash_kiyohara"] = pd.to_numeric(df["net_cash_kiyohara"], errors="coerce")
df_valid = df.dropna(subset=["net_cash_kiyohara"]).copy()
print(f"\nサンプル総数(B/C×52週高値×割安性上位50%): {len(df)}, うち清原式ネットキャッシュ比率算出可={len(df_valid)}, 経過{time.monotonic()-start:.0f}秒\n")

df_valid["tertile"] = pd.qcut(df_valid["net_cash_kiyohara"], 3, labels=["低", "中", "高"], duplicates="drop")
print("=== 清原式ネットキャッシュ比率 三分位別フォワードリターン ===")
for h in HORIZONS_TRADING_DAYS:
    sub = df_valid.dropna(subset=[h])
    print(f"\n[{h}]")
    print(sub.groupby("tertile")[h].agg(["size", "mean", "median"]).round(2))

print("\n=== 年別頑健性(24ヶ月、低 vs 高) ===")
df_valid['year'] = pd.to_datetime(df_valid['date']).dt.year
h24 = "504営業日(24ヶ月)"
for yr, g in df_valid.dropna(subset=[h24]).groupby('year'):
    low = g[g['tertile'] == '低'][h24]
    high = g[g['tertile'] == '高'][h24]
    if len(low) < 20 or len(high) < 20:
        continue
    print(f"{yr}: 低 n={len(low)} 平均={low.mean():.2f}%  |  高 n={len(high)} 平均={high.mean():.2f}%")

conn.close()
bt_conn.close()
