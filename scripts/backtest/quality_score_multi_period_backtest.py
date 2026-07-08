"""quality_score_backtest.pyの単一期間版を、複数決算期にまたがるパネル検証に拡張したもの。

各銘柄・各決算期(2期目以降)ごとに、その時点までに開示済みだったはずのデータだけで
質的スコアを再現計算し、開示からおよそ1年後までの株価騰落率と突き合わせる。
これを2017〜2024年度あたりの各コホートについて行い、プールしたIC/五分位分析に加え、
年度別のICも出すことで、単一期間では分からない頑健性(相場局面をまたいだ安定性)を見る。

前提: scripts/backtest/fetch_price_history_backtest.pyでdata/backtest.dbのprice_historyに
全銘柄の長期株価(yfinance最大10年分)を取得済みであること。

実行: python scripts/backtest/quality_score_multi_period_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection, get_backtest_connection  # noqa: E402
from backtest.quality_score_backtest import (  # noqa: E402
    CATEGORY_METRICS, CATEGORY_WEIGHTS, compute_ticker_metrics, percentile_rank,
    MIN_SECTOR_SAMPLE_SIZE,
)

DISCLOSURE_LAG_DAYS = 100
FORWARD_DAYS = 365
MIN_GAP_DAYS = 300  # 同一世代の重複決算(中間決算修正など)を間引く際の目安


def load_yearly_panel(conn) -> pd.DataFrame:
    query = """
    SELECT ticker, fiscal_year_end, revenue, net_income, equity, total_assets,
           equity_ratio, net_margin, dividend_per_share, payout_ratio, eps, operating_cf
    FROM fundamentals_yearly
    ORDER BY ticker, fiscal_year_end
    """
    return pd.read_sql_query(query, conn, parse_dates=["fiscal_year_end"])


def build_period_panel(yearly: pd.DataFrame) -> pd.DataFrame:
    """各銘柄・各決算期(2期目以降)ごとに、その期までのデータだけで指標を計算する"""
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
            m["ticker"] = ticker
            rows.append(m)
    df = pd.DataFrame(rows)
    df["signal_date"] = df["fiscal_year_end"] + pd.Timedelta(days=DISCLOSURE_LAG_DAYS)
    df["target_date"] = df["signal_date"] + pd.Timedelta(days=FORWARD_DAYS)
    df["cohort_year"] = df["fiscal_year_end"].dt.year
    return df


def attach_sector(conn, df: pd.DataFrame) -> pd.DataFrame:
    tickers = pd.read_sql_query("SELECT ticker, sector FROM tickers", conn)
    return df.merge(tickers, on="ticker", how="inner")


def compute_quality_score_per_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """コホート(決算期)ごとに独立して業種内相対パーセンタイルを計算する"""
    result_parts = []
    for cohort, df_cohort in df.groupby("cohort_year"):
        df_cohort = df_cohort.copy()
        sector_counts = df_cohort["sector"].value_counts().to_dict()
        df_cohort["comparison_group"] = df_cohort["sector"].apply(
            lambda s: s if sector_counts.get(s, 0) >= MIN_SECTOR_SAMPLE_SIZE else "__MARKET_WIDE__"
        )
        category_scores = {cat: pd.Series(index=df_cohort.index, dtype=float) for cat in CATEGORY_METRICS}
        category_conf = {cat: pd.Series(index=df_cohort.index, dtype=float) for cat in CATEGORY_METRICS}

        for group, df_group in df_cohort.groupby("comparison_group"):
            for category, metrics in CATEGORY_METRICS.items():
                weighted_sum = pd.Series(0.0, index=df_group.index)
                total_weight = pd.Series(0.0, index=df_group.index)
                weight_each = 1.0 / len(metrics)
                for col, higher_is_better in metrics:
                    pct = percentile_rank(df_group[col], higher_is_better)
                    valid = pct.notna()
                    weighted_sum[valid] += pct[valid] * weight_each
                    total_weight[valid] += weight_each
                scores = (weighted_sum / total_weight).where(total_weight > 0)
                category_scores[category].loc[df_group.index] = scores
                category_conf[category].loc[df_group.index] = total_weight

        for category in CATEGORY_METRICS:
            df_cohort[f"score_{category}"] = category_scores[category]
            df_cohort[f"confidence_{category}"] = category_conf[category]

        weighted_sum = pd.Series(0.0, index=df_cohort.index)
        total_weight = pd.Series(0.0, index=df_cohort.index)
        for category, w in CATEGORY_WEIGHTS.items():
            score = df_cohort[f"score_{category}"]
            conf = df_cohort[f"confidence_{category}"]
            valid = score.notna() & conf.notna()
            eff_w = w * conf
            weighted_sum[valid] += (score * eff_w)[valid]
            total_weight[valid] += eff_w[valid]
        df_cohort["quality_score"] = (weighted_sum / total_weight).where(total_weight > 0)
        result_parts.append(df_cohort)
    return pd.concat(result_parts, ignore_index=True)


def attach_forward_return(conn, df: pd.DataFrame) -> pd.DataFrame:
    prices = pd.read_sql_query("SELECT ticker, date, close FROM price_history", conn, parse_dates=["date"])
    prices = prices.sort_values("date")

    entry = df[["ticker", "signal_date"]].sort_values("signal_date")
    entry_price = pd.merge_asof(
        entry, prices, left_on="signal_date", right_on="date", by="ticker", direction="forward",
        tolerance=pd.Timedelta(days=14),
    ).rename(columns={"close": "price_entry"})[["ticker", "signal_date", "price_entry"]]

    exitp = df[["ticker", "target_date"]].sort_values("target_date")
    exit_price = pd.merge_asof(
        exitp, prices, left_on="target_date", right_on="date", by="ticker", direction="forward",
        tolerance=pd.Timedelta(days=14),
    ).rename(columns={"close": "price_exit"})[["ticker", "target_date", "price_exit"]]

    df = df.merge(entry_price, on=["ticker", "signal_date"], how="left")
    df = df.merge(exit_price, on=["ticker", "target_date"], how="left")
    df["forward_return"] = df["price_exit"] / df["price_entry"] - 1
    return df


def quintile_analysis(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["quality_score", "forward_return"]).copy()
    valid["quintile"] = pd.qcut(valid["quality_score"].rank(method="first"), 5, labels=["Q1(低)", "Q2", "Q3", "Q4", "Q5(高)"])
    return valid.groupby("quintile", observed=True)["forward_return"].agg(["mean", "median", "count"])


def main():
    conn = get_connection()
    bt_conn = get_backtest_connection()

    yearly = load_yearly_panel(conn)
    panel = build_period_panel(yearly)
    panel = attach_sector(conn, panel)
    print(f"パネル総観測数(銘柄×決算期): {len(panel)}件")
    print("コホート(決算年度)別の銘柄数:")
    print(panel["cohort_year"].value_counts().sort_index())

    panel = compute_quality_score_per_cohort(panel)
    panel = attach_forward_return(bt_conn, panel)

    valid = panel.dropna(subset=["quality_score", "forward_return"])
    print(f"\nスコア・フォワードリターン両方揃った観測数: {len(valid)}件\n")

    pooled_ic = spearmanr(valid["quality_score"], valid["forward_return"]).correlation
    print(f"=== プールIC(全コホート合算、Spearman) === \nIC = {pooled_ic:.4f}\n")

    print("=== 五分位分析(全コホート合算) ===")
    print(quintile_analysis(valid).round(4))

    print("\n=== コホート(決算年度)別IC(頑健性の確認) ===")
    for cohort, sub in valid.groupby("cohort_year"):
        if len(sub) < 50:
            continue
        ic = spearmanr(sub["quality_score"], sub["forward_return"]).correlation
        print(f"  {cohort}年度: IC={ic:+.4f} (n={len(sub)})")

    conn.close()
    bt_conn.close()


if __name__ == "__main__":
    main()
