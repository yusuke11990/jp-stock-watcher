"""「良い銘柄を見つけて1〜2年程度持つ」方針の判断エンジン(v3)。

decide_rule.py(v1)・decide_composite.py(v2)は、バックテストを複数年に拡張した結果、
どちらも技術的タイミング層(GC/MACD/合成テクニカルスコア)に頑健な優位性が
見られなかった(scripts/backtest/decision_engine_extended_backtest.py参照)。

scripts/backtest/technical_signal_event_study.pyで保有期間を21営業日〜24ヶ月まで
広げて個別検証したところ、RSI反発・BB反発(短期の押し目狙い)は24ヶ月保有では
ベースライン以下(むしろ逆行)になる一方、GC(ゴールデンクロス)は12〜24ヶ月保有で
最も良い成績だった。ここまではGCを採用していたが、その後さらに2つの検証を重ねた:

1. scripts/backtest/grade_breakdown_check.py的な検証(GC発生時のグレード別リターン):
   グレードSは「クオリティ・トラップ」でむしろ最も成績が悪く(12ヶ月中央値+5.22%)、
   グレードB/Cの方が明確に良い(B:+11.77%、C:+10.47%)と判明。
2. scripts/backtest/fundamental_plus_timing_backtest.py + alt_signals_check的な検証:
   グレードB/C×割安性上位50%の中で、GCと代替シグナル(株価のMA75上抜け、MA75の
   傾き反転、52週高値接近、GC強化版)を比較した結果、**52週高値の95%圏内への接近**が
   24ヶ月保有で圧倒的に優位(+52.81% vs GCの+44.47%、2022-2024年の全年で頑健)と
   判明した。行動ファイナンスで知られる「52週高値モメンタム」(George & Hwang, 2004)
   と整合する結果。

本エンジンはこれらの検証結果を反映し、以下のシンプルなルールを採用する:
  グレードがB/C かつ 52週高値の95%圏内に本日初めて接近 かつ
  割安性スコアがその日の対象銘柄群の中央値以上 -> buy
  それ以外 -> hold

GC・MACD・出来高急増・RSI/BB反発は個別検証で今回の想定保有期間(1〜2年)には
及ばなかったため使わない。グレードSも「クオリティ・トラップ」のため意図的に除外する。
「良い銘柄を長く持つ」というコンセプトのため、積極的な売り推奨(sell)は設けない
(保有継続中の警戒情報が欲しければv1のsellロジックを別途参照する)。
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

RULE_VERSION = "v3.0"
TARGET_GRADES = {"B", "C"}
MIN_HISTORY_ROWS = 126  # 52週高値判定に必要な最低限の遡り日数(check_near_52_week_highのmin_periodsと合わせる)


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
    query = "SELECT ticker, total_score, grade, score_valuation FROM scores WHERE snapshot_date = ?"
    return pd.read_sql_query(query, conn, params=(snapshot_date,))


def decide_quality_timing(grade: str, total_score: float, score_valuation, valuation_median: float, near_52w_high: bool) -> Decision:
    is_cheap_enough = score_valuation is not None and pd.notna(score_valuation) and score_valuation >= valuation_median
    if grade in TARGET_GRADES and near_52w_high and is_cheap_enough:
        reason = (
            f"グレード{grade}(総合スコア{total_score:.1f}、割安性{score_valuation:.0f})で"
            "52週高値圏へ接近(1〜2年保有のバックテストで頑健な優位性を確認済み)"
        )
        return Decision("buy", reason, 0.6)
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

    # 割安性の中央値は、本日のグレードB/C対象銘柄群の中で相対的に決める
    # (固定しきい値ではなく、その日の相場・銘柄構成に応じて動的に決まる)
    target_df = scores_df[scores_df["grade"].isin(TARGET_GRADES)]
    valuation_median = target_df["score_valuation"].median()

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
        flags = {"near_52_week_high": near_52w_high}
        technical = get_technical_state(hist)

        decision = decide_quality_timing(
            row["grade"], row["total_score"], row["score_valuation"], valuation_median, near_52w_high
        )
        upsert_decision(conn, ticker, decision_date, decision, row["total_score"], row["grade"], flags, technical.get("current_price"))
        action_counts[decision.action] += 1

    conn.close()
    print(f"snapshot_date={snapshot_date}, decision_date={decision_date}, rule_version={RULE_VERSION}")
    print(f"割安性中央値(グレードB/C対象): {valuation_median:.1f}")
    print(f"action分布: {action_counts}, 履歴不足でスキップ: {skipped}")


if __name__ == "__main__":
    main()
