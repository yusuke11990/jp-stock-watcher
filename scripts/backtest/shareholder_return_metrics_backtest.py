"""還元性(shareholder_return)カテゴリの6サブ指標それぞれについて、単独でフォワード
リターンを予測できているか(IC)を検証する。

scoring_config.yamlのshareholder_return: dividend_yield(0.25)・payout_ratio(0.15、
special ロジック)・total_shareholder_return_yield(0.20)・dividend_history_count(0.10)・
doe(0.15)・dividend_growth_1y(0.15)は、カテゴリ全体としてのIC(+0.022、
scoring_config.yamlのコメント参照)は検証済みだが、個別サブ指標ごとのICは
一度も検証していなかった。growth/safety軸が個別検証で「効果が薄い」と判明し
縮小した前例に倣い、還元性も同様に検証する。

fundamentals_weeklyは直近4スナップショットしか無く多年度検証に使えないため、
fundamentals_yearly(決算年次)からquality_score_multi_period_backtest.pyと
同じ手法で複数期間パネルを再構築し、以下のように各指標を近似計算する:
- dividend_yield = dividend_per_share / signal_date時点の株価
- payout_ratio = fundamentals_yearlyに直接ある値をそのまま使用
- total_shareholder_return_yield ≈ (payout_ratio×net_income + buyback_amount) / 時価総額近似
  (時価総額近似 = PER × net_income、cash_neutral_per_true.py等と同じ手法)
- dividend_history_count = その決算期までの連続増配ではなく「連続配当(0円でない)年数」
  (本番のfetch_fundamentals.pyでの定義を踏襲、直近から遡って配当が途切れるまでの年数)
- doe = payout_ratio × ROE (doe_proxyと同一の恒等式、本番のdoe定義に一致)
- dividend_growth_1y = quality_score_backtest.pyのcompute_ticker_metricsで既存計算

実行: python scripts/backtest/shareholder_return_metrics_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection, get_backtest_connection  # noqa: E402
from backtest.quality_score_backtest import compute_ticker_metrics  # noqa: E402
from backtest.quality_score_multi_period_backtest import attach_sector, MIN_GAP_DAYS, DISCLOSURE_LAG_DAYS, FORWARD_DAYS  # noqa: E402
from backtest.quality_score_v2_backtest import (  # noqa: E402
    load_real_disclosure_dates, attach_signal_date, attach_price_features, add_valuation_metrics,
    FORWARD_WINDOWS_DAYS,
)

RAW_COLUMNS = [
    "ticker", "fiscal_year_end", "revenue", "net_income", "equity", "total_assets",
    "equity_ratio", "net_margin", "dividend_per_share", "payout_ratio", "eps", "operating_cf",
    "buyback_amount",
]


def load_yearly_panel_extended(conn) -> pd.DataFrame:
    query = f"SELECT {', '.join(RAW_COLUMNS)} FROM fundamentals_yearly ORDER BY ticker, fiscal_year_end"
    return pd.read_sql_query(query, conn, parse_dates=["fiscal_year_end"])


def compute_dividend_history_count(window: pd.DataFrame) -> int:
    """windowは最新期が末尾。最新期から遡り、配当(dividend_per_share>0)が
    途切れるまでの連続年数を数える。"""
    count = 0
    for i in range(len(window) - 1, -1, -1):
        dps = window.iloc[i]["dividend_per_share"]
        if pd.notna(dps) and dps > 0:
            count += 1
        else:
            break
    return count


def build_period_panel_extended(yearly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, df_ticker in yearly.groupby("ticker"):
        df_ticker = df_ticker.sort_values("fiscal_year_end").reset_index(drop=True)
        for i in range(1, len(df_ticker)):
            window = df_ticker.iloc[: i + 1]
            if (window.iloc[-1]["fiscal_year_end"] - window.iloc[-2]["fiscal_year_end"]).days < MIN_GAP_DAYS:
                continue
            m = compute_ticker_metrics(window)
            if m is None:
                continue
            latest = window.iloc[-1]
            m["ticker"] = ticker
            m["payout_ratio"] = latest["payout_ratio"]
            m["buyback_amount"] = latest["buyback_amount"]
            m["net_income"] = latest["net_income"]
            m["dividend_per_share"] = latest["dividend_per_share"]
            m["dividend_history_count"] = compute_dividend_history_count(window)
            rows.append(m)
    df = pd.DataFrame(rows)
    df["signal_date"] = df["fiscal_year_end"] + pd.Timedelta(days=DISCLOSURE_LAG_DAYS)
    df["target_date"] = df["signal_date"] + pd.Timedelta(days=FORWARD_DAYS)
    df["cohort_year"] = df["fiscal_year_end"].dt.year
    return df


def report_ic(df: pd.DataFrame, col: str, return_col: str, label: str) -> None:
    valid = df.dropna(subset=[col, return_col])
    if len(valid) < 50:
        print(f"  [{label}] サンプル不足(n={len(valid)})")
        return
    ic = spearmanr(valid[col], valid[return_col]).correlation
    valid = valid.copy()
    valid["quintile"] = pd.qcut(valid[col].rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    q_mean = valid.groupby("quintile", observed=True)[return_col].mean()
    spread = q_mean["Q5"] - q_mean["Q1"]
    print(f"  [{label}] n={len(valid)}  IC={ic:+.4f}  Q5-Q1スプレッド={spread:+.4f}  (Q1={q_mean['Q1']:+.4f} / Q5={q_mean['Q5']:+.4f})")


def main():
    conn = get_connection()
    bt_conn = get_backtest_connection()

    yearly = load_yearly_panel_extended(conn)
    panel = build_period_panel_extended(yearly)
    panel = attach_sector(conn, panel)

    disclosure = load_real_disclosure_dates(conn)
    panel = attach_signal_date(panel, disclosure)

    print(f"パネル総観測数: {len(panel)}件")
    panel = attach_price_features(bt_conn, panel)
    panel = add_valuation_metrics(panel)  # per, pbr を追加(時価総額近似に使う)

    n_with_price = panel["price_entry"].notna().sum()
    print(f"signal_date時点の株価が取れた観測数: {n_with_price}件\n")

    # --- 各サブ指標の再構築 ---
    panel["dividend_yield"] = panel["dividend_per_share"] / panel["price_entry"]
    panel.loc[(panel["dividend_yield"] < 0) | (panel["dividend_yield"] > 0.3), "dividend_yield"] = None

    panel["market_cap_approx"] = panel["per"] * panel["net_income"]
    panel.loc[panel["market_cap_approx"] <= 0, "market_cap_approx"] = None

    total_dividend = panel["payout_ratio"] * panel["net_income"]
    buyback = panel["buyback_amount"].fillna(0)
    panel["total_shareholder_return_yield"] = (total_dividend + buyback) / panel["market_cap_approx"]
    panel.loc[(panel["total_shareholder_return_yield"] < 0) | (panel["total_shareholder_return_yield"] > 0.3), "total_shareholder_return_yield"] = None

    panel["doe"] = panel["payout_ratio"] * panel["roe"]
    panel.loc[(panel["doe"] < 0) | (panel["doe"] > 0.3), "doe"] = None

    panel.loc[(panel["payout_ratio"] < 0) | (panel["payout_ratio"] > 3), "payout_ratio"] = None

    METRICS = {
        "dividend_yield": True,
        "payout_ratio": True,  # 本番はspecialロジックだが、まずは素朴に higher_is_better=true として検証
        "total_shareholder_return_yield": True,
        "dividend_history_count": True,
        "doe": True,
        "dividend_growth_1y": True,
    }

    print("=== 還元性サブ指標ごとの単独IC(higher_is_betterと仮定した場合) ===")
    for h in FORWARD_WINDOWS_DAYS:
        return_col = f"forward_return_{h}"
        print(f"\n--- {h}後 ---")
        for col in METRICS:
            report_ic(panel, col, return_col, col)

    print("\n=== payout_ratioの実際のスコアリングロジック(0-60%:百分位、60-100%:70点、100%超:30点)との比較 ===")
    from scoring.compute_scores import percentile_rank, score_payout_ratio  # noqa: E402
    valid = panel.dropna(subset=["payout_ratio"]).copy()
    normal_pct = percentile_rank(valid["payout_ratio"], higher_is_better=True)
    valid["payout_ratio_score"] = score_payout_ratio(valid["payout_ratio"], normal_pct)
    for h in FORWARD_WINDOWS_DAYS:
        return_col = f"forward_return_{h}"
        report_ic(valid, "payout_ratio_score", return_col, f"payout_ratio特殊スコア / {h}後")

    print("\n=== コホート年度別IC(12ヶ月後、頑健性の確認) ===")
    h = "forward_return_12ヶ月"
    for col in METRICS:
        print(f"\n  --- {col} ---")
        for cohort, sub in panel.dropna(subset=[col, h]).groupby("cohort_year"):
            if len(sub) < 50:
                continue
            ic = spearmanr(sub[col], sub[h]).correlation
            print(f"    {cohort}年度: IC={ic:+.4f} (n={len(sub)})")

    conn.close()
    bt_conn.close()


if __name__ == "__main__":
    main()
