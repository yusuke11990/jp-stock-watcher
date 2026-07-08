"""株価に依存しない「質的スコア」(安全性・成長性・収益性・効率性・還元性の5軸、
割安性を除く)を最新の決算年次データ(fundamentals_yearly)から算出し、直近1年間の
株価騰落率と突き合わせてクロスセクションで検証する。

制約:
- scores/fundamentals_weeklyはまだ2日分しか無く歴史的スコアが存在しないため、
  本来の複数年バックテストはできない。
- fundamentals_yearlyはPER/PBR/時価総額/配当利回りのような株価が要る指標を含まない
  ため、割安性(valuation)・配当利回りを含む還元性の一部は評価対象から除外する。
- price_dailyは直近1年分(約243営業日)しか無いため、フォワードリターンはこの
  1年間の単一区間のみで評価する(複数期間にまたがる本格バックテストではない)。

実行: python scripts/backtest/quality_score_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection  # noqa: E402

MIN_SECTOR_SAMPLE_SIZE = 5

# 質的スコア(株価に依存しない5軸)のカテゴリ重み。valuationを除いた分を他4軸+
# growthに再配分するのではなく、単純に5軸均等(各20%)にする
CATEGORY_WEIGHTS = {
    "safety": 0.20,
    "growth": 0.20,
    "profitability": 0.20,
    "efficiency": 0.20,
    "shareholder_return": 0.20,
}


def load_yearly_panel(conn) -> pd.DataFrame:
    query = """
    SELECT ticker, fiscal_year_end, revenue, net_income, equity, total_assets,
           equity_ratio, net_margin, dividend_per_share, payout_ratio, eps, operating_cf
    FROM fundamentals_yearly
    ORDER BY ticker, fiscal_year_end
    """
    return pd.read_sql_query(query, conn, parse_dates=["fiscal_year_end"])


def compute_ticker_metrics(df_ticker: pd.DataFrame) -> dict | None:
    """1銘柄の複数年パネルから、最新期のスナップショット指標を計算する"""
    if len(df_ticker) < 2:
        return None
    latest = df_ticker.iloc[-1]
    prev = df_ticker.iloc[-2]

    revenue_growth_1y = (
        (latest["revenue"] / prev["revenue"] - 1)
        if pd.notna(latest["revenue"]) and pd.notna(prev["revenue"]) and prev["revenue"] > 0
        else None
    )
    dividend_growth_1y = (
        (latest["dividend_per_share"] / prev["dividend_per_share"] - 1)
        if pd.notna(latest["dividend_per_share"]) and pd.notna(prev["dividend_per_share"]) and prev["dividend_per_share"] > 0
        else None
    )
    roe = (
        latest["net_income"] / latest["equity"]
        if pd.notna(latest["net_income"]) and pd.notna(latest["equity"]) and latest["equity"] > 0
        else None
    )
    roa = (
        latest["net_income"] / latest["total_assets"]
        if pd.notna(latest["net_income"]) and pd.notna(latest["total_assets"]) and latest["total_assets"] > 0
        else None
    )
    asset_turnover = (
        latest["revenue"] / latest["total_assets"]
        if pd.notna(latest["revenue"]) and pd.notna(latest["total_assets"]) and latest["total_assets"] > 0
        else None
    )
    # DOE(純資産配当率) = 配当性向 × ROE の恒等式で近似する(株主資本簿価/株数が無いため)
    doe_proxy = (
        latest["payout_ratio"] * roe
        if pd.notna(latest["payout_ratio"]) and roe is not None
        else None
    )
    # 会計発生高(Sloan Accrual): (純利益-営業CF)/総資産。高いほど利益がキャッシュフローに
    # 裏付けられておらず「利益の質」が低い(=将来リターンが低い)とされるアノマリー
    accruals = (
        (latest["net_income"] - latest["operating_cf"]) / latest["total_assets"]
        if pd.notna(latest["net_income"]) and pd.notna(latest["operating_cf"]) and pd.notna(latest["total_assets"]) and latest["total_assets"] > 0
        else None
    )

    return {
        "fiscal_year_end": latest["fiscal_year_end"],
        "equity_ratio": latest["equity_ratio"],
        "revenue_growth_1y": revenue_growth_1y,
        "roe": roe,
        "net_margin": latest["net_margin"],
        "roa": roa,
        "asset_turnover": asset_turnover,
        "dividend_growth_1y": dividend_growth_1y,
        "doe_proxy": doe_proxy,
        "eps": latest.get("eps"),
        "accruals": accruals,
    }


DISCLOSURE_LAG_DAYS = 100  # 決算期末から有価証券報告書開示までの目安(約3ヶ月+安全マージン)


def build_metrics_table(conn, price_start: pd.Timestamp) -> pd.DataFrame:
    panel = load_yearly_panel(conn)
    # フォワードリターン計測期間の開始日より前に「開示済みだったはず」の決算だけを使う
    # (直近決算が開示される前にその情報でスコアを付けている=先読みバイアスを避ける)
    cutoff = price_start - pd.Timedelta(days=DISCLOSURE_LAG_DAYS)
    panel = panel[panel["fiscal_year_end"] <= cutoff]

    rows = []
    for ticker, df_ticker in panel.groupby("ticker"):
        m = compute_ticker_metrics(df_ticker)
        if m is not None:
            m["ticker"] = ticker
            rows.append(m)
    df = pd.DataFrame(rows)

    tickers = pd.read_sql_query("SELECT ticker, sector FROM tickers WHERE is_active = 1", conn)
    df = df.merge(tickers, on="ticker", how="inner")
    return df


def percentile_rank(series: pd.Series, higher_is_better: bool) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(index=series.index, dtype=float)
    ranks = valid.rank(pct=True) * 100
    if not higher_is_better:
        ranks = 100 - ranks
    return ranks.reindex(series.index)


CATEGORY_METRICS = {
    "safety": [("equity_ratio", True)],
    "growth": [("revenue_growth_1y", True)],
    "profitability": [("roe", True), ("net_margin", True)],
    "efficiency": [("roa", True), ("asset_turnover", True)],
    "shareholder_return": [("dividend_growth_1y", True), ("doe_proxy", True)],
}


def compute_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    sector_counts = df["sector"].value_counts().to_dict()
    df["comparison_group"] = df["sector"].apply(
        lambda s: s if sector_counts.get(s, 0) >= MIN_SECTOR_SAMPLE_SIZE else "__MARKET_WIDE__"
    )

    category_scores = {cat: pd.Series(index=df.index, dtype=float) for cat in CATEGORY_METRICS}
    category_conf = {cat: pd.Series(index=df.index, dtype=float) for cat in CATEGORY_METRICS}

    for group, df_group in df.groupby("comparison_group"):
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
        df[f"score_{category}"] = category_scores[category]
        df[f"confidence_{category}"] = category_conf[category]

    weighted_sum = pd.Series(0.0, index=df.index)
    total_weight = pd.Series(0.0, index=df.index)
    for category, w in CATEGORY_WEIGHTS.items():
        score = df[f"score_{category}"]
        conf = df[f"confidence_{category}"]
        valid = score.notna() & conf.notna()
        eff_w = w * conf
        weighted_sum[valid] += (score * eff_w)[valid]
        total_weight[valid] += eff_w[valid]
    df["quality_score"] = (weighted_sum / total_weight).where(total_weight > 0)
    return df


def load_forward_return(conn) -> pd.DataFrame:
    query = """
    SELECT ticker, MIN(date) AS first_date, MAX(date) AS last_date FROM price_daily GROUP BY ticker
    """
    dates = pd.read_sql_query(query, conn)
    prices = pd.read_sql_query("SELECT ticker, date, close FROM price_daily", conn, parse_dates=["date"])

    first_price = prices.merge(dates[["ticker", "first_date"]], on="ticker") \
        .query("date == first_date")[["ticker", "close"]].rename(columns={"close": "price_start"})
    last_price = prices.merge(dates[["ticker", "last_date"]], on="ticker") \
        .query("date == last_date")[["ticker", "close"]].rename(columns={"close": "price_end"})

    merged = first_price.merge(last_price, on="ticker")
    merged["forward_return"] = merged["price_end"] / merged["price_start"] - 1
    return merged[["ticker", "forward_return"]]


def quintile_analysis(df: pd.DataFrame, score_col: str, return_col: str) -> pd.DataFrame:
    valid = df.dropna(subset=[score_col, return_col]).copy()
    valid["quintile"] = pd.qcut(valid[score_col].rank(method="first"), 5, labels=["Q1(低)", "Q2", "Q3", "Q4", "Q5(高)"])
    summary = valid.groupby("quintile", observed=True)[return_col].agg(["mean", "median", "count"])
    return summary


def main():
    conn = get_connection()
    price_start_str = conn.execute("SELECT MIN(date) FROM price_daily").fetchone()[0]
    price_start = pd.Timestamp(price_start_str)
    cutoff = price_start - pd.Timedelta(days=DISCLOSURE_LAG_DAYS)
    print(f"株価データ開始日: {price_start.date()} / 決算データの先読み防止カットオフ(開示ラグ{DISCLOSURE_LAG_DAYS}日): 期末日が{cutoff.date()}以前の決算のみ使用\n")

    df = build_metrics_table(conn, price_start)
    print(f"質的スコア算出対象: {len(df)}銘柄(カットオフ以前に開示済みの決算が2期分ある銘柄)")

    df = compute_quality_score(df)
    scored = df.dropna(subset=["quality_score"])
    print(f"quality_score算出できた銘柄: {len(scored)}件")

    fwd = load_forward_return(conn)
    merged = scored.merge(fwd, on="ticker", how="inner")
    merged = merged.dropna(subset=["forward_return"])
    print(f"株価データも突き合わせられた銘柄: {len(merged)}件\n")

    ic = spearmanr(merged["quality_score"], merged["forward_return"]).correlation
    print(f"=== quality_score と直近1年フォワードリターンの情報係数(Spearman IC) ===")
    print(f"IC = {ic:.4f}\n")

    print("=== 五分位分析(スコア低→高で1年リターンがどう変わるか) ===")
    print(quintile_analysis(merged, "quality_score", "forward_return").round(4))

    print("\n=== カテゴリ別スコアの単独IC(どの軸が効いているか) ===")
    for category in CATEGORY_METRICS:
        col = f"score_{category}"
        sub = merged.dropna(subset=[col, "forward_return"])
        if len(sub) < 30:
            continue
        cat_ic = spearmanr(sub[col], sub["forward_return"]).correlation
        print(f"  {category:20s}: IC={cat_ic:+.4f} (n={len(sub)})")

    conn.close()


if __name__ == "__main__":
    main()
