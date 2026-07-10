"""「良い銘柄を見つけて1〜2年程度持つ」方針の判断エンジン(v4)。

decide_quality_timing.py(v3: グレードB/C×52週高値接近×割安性上位50%)は
scripts/backtest/near_52w_high_robustness_backtest.py等で頑健性を確認済みの
ベースラインだが、清原達郎氏(『わが投資術』)・へム氏(『「増配」株投資』)が
共に「最強の武器」として挙げる**ネットキャッシュ比率**をさらに重ねられるか検証した
(scripts/backtest/net_cash_ratio_kiyohara.py)。

清原氏の定義: ネットキャッシュ比率 = (流動資産 + 0.7×投資有価証券 - 負債) / 時価総額
(投資有価証券は売却時の税負担を見込んで70%評価)

v3のコホート(グレードB/C×52週高値接近×割安性上位50%)内をこの指標でさらに
三分位に分けたところ、24ヶ月保有で低位+53.05%(中央値+33.78%)に対し
上位+70.38%(中央値+51.59%)と、これまで検証した中でも最大級の差が出た。
2022〜2024年の全年で頑健(低 vs 高: 49.47/63.66、47.71/69.91、64.41/76.00)。

本エンジンはv3の条件に「ネットキャッシュ比率がその日の対象銘柄群の上位1/3」を
追加する。current_assets・investment_securitiesはEDINETの貸借対照表本体タグ
(jppfs_cor:CurrentAssets/InvestmentSecurities)から取得しており、
「経営指標等の推移」表と異なり当期末・前期末の2期分しか開示されないため、
全銘柄をカバーしきれない(欠損は約3割)。データが無い銘柄は判定不能として
buyにしない(v3のscore_valuation欠損時と同じ、安全側=fail-closedの扱い)。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection  # noqa: E402
from common.technical import calc_indicators, check_near_52_week_high, get_technical_state  # noqa: E402

RULE_VERSION = "v4.0"
TARGET_GRADES = {"B", "C"}
MIN_HISTORY_ROWS = 126  # 52週高値判定に必要な最低限の遡り日数(check_near_52_week_highのmin_periodsと合わせる)
INVESTMENT_SECURITIES_HAIRCUT = 0.7  # 投資有価証券は売却時の税負担を見込んで70%評価(清原式)
NET_CASH_TOP_FRACTION = 1.0 / 3.0  # バックテストで検証した上位1/3(三分位の「高」)を採用


@dataclass
class Decision:
    action: str
    reason: str
    confidence: float


def load_price_history(conn, ticker: str, as_of_date: str) -> pd.DataFrame:
    query = "SELECT date, open, high, low, close, volume FROM price_daily WHERE ticker = ? AND date <= ? ORDER BY date ASC"
    df = pd.read_sql_query(query, conn, params=(ticker, as_of_date), parse_dates=["date"])
    if df.empty:
        return df
    return df.set_index("date").rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )


def load_latest_scores(conn, snapshot_date: str) -> pd.DataFrame:
    query = """
    SELECT s.ticker, s.total_score, s.grade, s.score_valuation, f.market_cap
    FROM scores s
    LEFT JOIN fundamentals_weekly f ON f.ticker = s.ticker AND f.snapshot_date = s.snapshot_date
    WHERE s.snapshot_date = ?
    """
    return pd.read_sql_query(query, conn, params=(snapshot_date,))


def load_latest_balance_sheet(conn) -> pd.DataFrame:
    """各銘柄の最新決算期(fiscal_year_end)の貸借対照表項目を1行ずつ取得する"""
    query = """
    SELECT f.ticker, f.current_assets, f.investment_securities, f.total_liabilities
    FROM fundamentals_yearly f
    INNER JOIN (
        SELECT ticker, MAX(fiscal_year_end) AS max_fye FROM fundamentals_yearly GROUP BY ticker
    ) latest ON f.ticker = latest.ticker AND f.fiscal_year_end = latest.max_fye
    """
    return pd.read_sql_query(query, conn)


def compute_net_cash_ratio(current_assets, investment_securities, total_liabilities, market_cap):
    if any(v is None or pd.isna(v) for v in (current_assets, investment_securities, total_liabilities, market_cap)):
        return None
    if market_cap <= 0:
        return None
    return (current_assets + INVESTMENT_SECURITIES_HAIRCUT * investment_securities - total_liabilities) / market_cap


def decide_quality_timing_v4(
    grade: str, total_score: float, score_valuation, valuation_median: float,
    near_52w_high: bool, net_cash_ratio, net_cash_cutoff: float,
) -> Decision:
    is_cheap_enough = score_valuation is not None and pd.notna(score_valuation) and score_valuation >= valuation_median
    is_cash_rich = net_cash_ratio is not None and pd.notna(net_cash_ratio) and net_cash_ratio >= net_cash_cutoff
    if grade in TARGET_GRADES and near_52w_high and is_cheap_enough and is_cash_rich:
        reason = (
            f"グレード{grade}(総合スコア{total_score:.1f}、割安性{score_valuation:.0f})で52週高値圏へ接近、"
            f"かつネットキャッシュ比率{net_cash_ratio * 100:.1f}%が上位1/3圏内"
            "(1〜2年保有のバックテストで頑健な優位性を確認済み)"
        )
        return Decision("buy", reason, 0.65)
    if grade in TARGET_GRADES and near_52w_high and is_cheap_enough and net_cash_ratio is None:
        return Decision("hold", f"グレード{grade}、条件はB/C×52週高値×割安性を満たすがネットキャッシュ比率データなし", 0.3)
    return Decision("hold", f"グレード{grade}、条件を満たさず", 0.3)


def upsert_decision(conn, ticker: str, decision_date: str, decision: Decision, total_score, grade, flags: dict, price: float) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO decisions
                (ticker, decision_date, action, decision_source, rule_version,
                 total_score, grade, technical_signals, reason, price_at_decision, confidence)
            VALUES (?, ?, ?, 'rule', ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, decision_date, decision_source, rule_version) DO UPDATE SET
                action=excluded.action, total_score=excluded.total_score, grade=excluded.grade,
                technical_signals=excluded.technical_signals, reason=excluded.reason,
                price_at_decision=excluded.price_at_decision, confidence=excluded.confidence
            """,
            (
                ticker, decision_date, decision.action, RULE_VERSION,
                total_score, grade, json.dumps(flags, ensure_ascii=False),
                decision.reason, price, decision.confidence,
            ),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    conn = get_connection()
    snapshot_date = args.snapshot_date or conn.execute("SELECT MAX(snapshot_date) FROM scores").fetchone()[0]
    if snapshot_date is None:
        print("scoresにデータがありません。先にcompute_scores.pyを実行してください")
        return

    scores_df = load_latest_scores(conn, snapshot_date)
    if args.limit:
        scores_df = scores_df.head(args.limit)

    bs_df = load_latest_balance_sheet(conn)
    scores_df = scores_df.merge(bs_df, on="ticker", how="left")
    scores_df["net_cash_ratio"] = scores_df.apply(
        lambda r: compute_net_cash_ratio(
            r["current_assets"], r["investment_securities"], r["total_liabilities"], r["market_cap"]
        ),
        axis=1,
    )

    # 割安性の中央値・ネットキャッシュ比率のカットオフは、本日のグレードB/C対象銘柄群の
    # 中で相対的に決める(固定しきい値ではなく、その日の相場・銘柄構成に応じて動的に決まる)
    target_df = scores_df[scores_df["grade"].isin(TARGET_GRADES)]
    valuation_median = target_df["score_valuation"].median()
    net_cash_valid = target_df["net_cash_ratio"].dropna()
    net_cash_cutoff = net_cash_valid.quantile(1 - NET_CASH_TOP_FRACTION) if not net_cash_valid.empty else float("inf")

    decision_date = snapshot_date
    action_counts = {"buy": 0, "hold": 0}
    skipped = 0

    for _, row in scores_df.iterrows():
        ticker = row["ticker"]
        hist = load_price_history(conn, ticker, snapshot_date)
        if len(hist) < MIN_HISTORY_ROWS:
            skipped += 1
            continue

        hist = calc_indicators(hist)
        near_52w_high = check_near_52_week_high(hist)
        flags = {"near_52_week_high": near_52w_high, "net_cash_ratio": row["net_cash_ratio"]}
        technical = get_technical_state(hist)

        decision = decide_quality_timing_v4(
            row["grade"], row["total_score"], row["score_valuation"], valuation_median,
            near_52w_high, row["net_cash_ratio"], net_cash_cutoff,
        )
        upsert_decision(conn, ticker, decision_date, decision, row["total_score"], row["grade"], flags, technical.get("current_price"))
        action_counts[decision.action] += 1

    conn.close()
    print(f"snapshot_date={snapshot_date}, decision_date={decision_date}, rule_version={RULE_VERSION}")
    print(f"割安性中央値(グレードB/C対象): {valuation_median:.1f}")
    print(f"ネットキャッシュ比率カットオフ(上位1/3、グレードB/C対象): {net_cash_cutoff * 100:.1f}% (データ有り{len(net_cash_valid)}/{len(target_df)}件)")
    print(f"action分布: {action_counts}, 履歴不足でスキップ: {skipped}")


if __name__ == "__main__":
    main()
