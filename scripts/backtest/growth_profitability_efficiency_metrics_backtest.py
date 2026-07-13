"""成長性(growth)・収益性(profitability)・効率性(efficiency)・安全性(safety)の
各サブ指標について、単独でフォワードリターンを予測できているか(IC)を検証する。
shareholder_return_metrics_backtest.pyの続き。

制約: fundamentals_yearlyには current_ratio(流動比率)・interest_bearing_debt
(有利子負債)・EBITDA・interest_expense(支払利息)に相当する生データが無いため、
safetyの5指標のうち equity_ratio 以外(current_ratio・interest_bearing_debt_to_market_cap・
net_debt_to_ebitda・interest_coverage_ratio)は歴史的バックテストで再現できない。
これはデータ収集の粒度の制約であり、本番のfundamentals_weeklyでは直近分のみ
計算できている(過去に遡って検証はできない)。

実行: python scripts/backtest/growth_profitability_efficiency_metrics_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection, get_backtest_connection  # noqa: E402
from backtest.quality_score_multi_period_backtest import attach_sector, MIN_GAP_DAYS, DISCLOSURE_LAG_DAYS, FORWARD_DAYS  # noqa: E402
from backtest.quality_score_v2_backtest import (  # noqa: E402
    load_real_disclosure_dates, attach_signal_date, attach_price_features, FORWARD_WINDOWS_DAYS,
)

RAW_COLUMNS = [
    "ticker", "fiscal_year_end", "revenue", "operating_income", "ordinary_income", "net_income",
    "operating_margin", "net_margin", "eps", "total_assets", "equity", "equity_ratio", "operating_cf",
]


def load_yearly_panel_extended(conn) -> pd.DataFrame:
    query = f"SELECT {', '.join(RAW_COLUMNS)} FROM fundamentals_yearly ORDER BY ticker, fiscal_year_end"
    return pd.read_sql_query(query, conn, parse_dates=["fiscal_year_end"])


def safe_div(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return None
    return a / b


def safe_growth(cur, prev):
    if pd.isna(cur) or pd.isna(prev) or prev <= 0:
        return None
    return cur / prev - 1


def safe_cagr(cur, past, years):
    if pd.isna(cur) or pd.isna(past) or past <= 0 or cur <= 0:
        return None
    return (cur / past) ** (1.0 / years) - 1


def compute_period_metrics(window: pd.DataFrame) -> dict | None:
    """windowは最新期が末尾。最新期時点の各サブ指標を計算する。"""
    if len(window) < 2:
        return None
    latest = window.iloc[-1]
    prev = window.iloc[-2]
    three_back = window.iloc[-4] if len(window) >= 4 else None

    m = {
        "fiscal_year_end": latest["fiscal_year_end"],
        # --- growth ---
        "revenue_growth_1y": safe_growth(latest["revenue"], prev["revenue"]),
        "operating_income_growth_1y": safe_growth(latest["operating_income"], prev["operating_income"]),
        "ordinary_income_growth_1y": safe_growth(latest["ordinary_income"], prev["ordinary_income"]),
        "eps_growth_1y": safe_growth(latest["eps"], prev["eps"]),
        # --- profitability ---
        "roe": safe_div(latest["net_income"], latest["equity"]),
        "operating_margin": latest["operating_margin"],
        "net_margin": latest["net_margin"],
        "ordinary_income_margin": safe_div(latest["ordinary_income"], latest["revenue"]),
        "operating_cf_margin": safe_div(latest["operating_cf"], latest["revenue"]),
        # --- efficiency ---
        "roa": safe_div(latest["net_income"], latest["total_assets"]),
        "asset_turnover": safe_div(latest["revenue"], latest["total_assets"]),
        # --- safety(equity_ratioのみ再現可能) ---
        "equity_ratio": latest["equity_ratio"],
    }
    if three_back is not None:
        m["revenue_growth_3y_cagr"] = safe_cagr(latest["revenue"], three_back["revenue"], 3)
        m["eps_growth_3y_cagr"] = safe_cagr(latest["eps"], three_back["eps"], 3)
    else:
        m["revenue_growth_3y_cagr"] = None
        m["eps_growth_3y_cagr"] = None
    return m


def build_period_panel(yearly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, df_ticker in yearly.groupby("ticker"):
        df_ticker = df_ticker.sort_values("fiscal_year_end").reset_index(drop=True)
        for i in range(1, len(df_ticker)):
            window = df_ticker.iloc[: i + 1]
            if (window.iloc[-1]["fiscal_year_end"] - window.iloc[-2]["fiscal_year_end"]).days < MIN_GAP_DAYS:
                continue
            m = compute_period_metrics(window)
            if m is None:
                continue
            m["ticker"] = ticker
            rows.append(m)
    df = pd.DataFrame(rows)
    df["signal_date"] = df["fiscal_year_end"] + pd.Timedelta(days=DISCLOSURE_LAG_DAYS)
    df["target_date"] = df["signal_date"] + pd.Timedelta(days=FORWARD_DAYS)
    df["cohort_year"] = df["fiscal_year_end"].dt.year
    return df


METRICS_BY_CATEGORY = {
    "growth": ["revenue_growth_1y", "revenue_growth_3y_cagr", "operating_income_growth_1y",
               "ordinary_income_growth_1y", "eps_growth_1y", "eps_growth_3y_cagr"],
    "profitability": ["roe", "operating_margin", "net_margin", "ordinary_income_margin", "operating_cf_margin"],
    "efficiency": ["roa", "asset_turnover"],
    "safety(equity_ratioのみ再現可能)": ["equity_ratio"],
}


def report_ic(df: pd.DataFrame, col: str, return_col: str, label: str) -> None:
    valid = df.dropna(subset=[col, return_col])
    if len(valid) < 50:
        print(f"    [{label}] サンプル不足(n={len(valid)})")
        return
    ic = spearmanr(valid[col], valid[return_col]).correlation
    valid = valid.copy()
    valid["quintile"] = pd.qcut(valid[col].rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    q_mean = valid.groupby("quintile", observed=True)[return_col].mean()
    spread = q_mean["Q5"] - q_mean["Q1"]
    print(f"    [{label}] n={len(valid)}  IC={ic:+.4f}  Q5-Q1スプレッド={spread:+.4f}  (Q1={q_mean['Q1']:+.4f} / Q5={q_mean['Q5']:+.4f})")


def main():
    conn = get_connection()
    bt_conn = get_backtest_connection()

    yearly = load_yearly_panel_extended(conn)
    panel = build_period_panel(yearly)
    panel = attach_sector(conn, panel)

    disclosure = load_real_disclosure_dates(conn)
    panel = attach_signal_date(panel, disclosure)

    print(f"パネル総観測数: {len(panel)}件")
    panel = attach_price_features(bt_conn, panel)
    n_with_price = panel["price_entry"].notna().sum()
    print(f"signal_date時点の株価が取れた観測数: {n_with_price}件\n")

    for category, metrics in METRICS_BY_CATEGORY.items():
        print(f"\n########## {category} ##########")
        for h in FORWARD_WINDOWS_DAYS:
            return_col = f"forward_return_{h}"
            print(f"\n  --- {h}後 ---")
            for col in metrics:
                report_ic(panel, col, return_col, col)

    print("\n\n########## コホート年度別IC(12ヶ月後、頑健性の確認) ##########")
    h = "forward_return_12ヶ月"
    for category, metrics in METRICS_BY_CATEGORY.items():
        print(f"\n=== {category} ===")
        for col in metrics:
            print(f"  --- {col} ---")
            for cohort, sub in panel.dropna(subset=[col, h]).groupby("cohort_year"):
                if len(sub) < 50:
                    continue
                ic = spearmanr(sub[col], sub[h]).correlation
                print(f"    {cohort}年度: IC={ic:+.4f} (n={len(sub)})")

    conn.close()
    bt_conn.close()


if __name__ == "__main__":
    main()
