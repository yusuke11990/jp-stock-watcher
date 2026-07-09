"""「良い銘柄を見つけて1〜2年程度持つ」方針の判断エンジン(v3)。

decide_rule.py(v1)・decide_composite.py(v2)は、バックテストを複数年に拡張した結果、
どちらも技術的タイミング層(GC/MACD/合成テクニカルスコア)に頑健な優位性が
見られなかった(scripts/backtest/decision_engine_extended_backtest.py参照)。

scripts/backtest/technical_signal_event_study.pyで保有期間を21営業日〜24ヶ月まで
広げて個別検証したところ、重要な発見があった: RSI反発・BB反発(短期の押し目狙い)は
21営業日保有では強いが、12〜24ヶ月保有ではベースライン並みかむしろ下回る
(24ヶ月保有のRSIはベースライン-7.79pt)。逆にGC(ゴールデンクロス、トレンド転換)は
短期(21営業日)では弱いが、12ヶ月保有(+2.55pt)・24ヶ月保有(+1.09pt)では
最も良い成績だった。scripts/backtest/fundamental_plus_timing_backtest.pyで
「総合スコア上位(グレードS/A/B)」と組み合わせて検証しても同じ傾向で、GCは
2022〜2025年の全年で12ヶ月+16〜32%、24ヶ月+26〜38%と頑健な結果を示した
(RSI/BBは24ヶ月でベースライン以下に沈む)。

本エンジンは1〜2年保有を前提にこの検証結果を反映し、以下のシンプルな
二値ルールのみを採用する:
  グレードがS/A/B かつ GC(ゴールデンクロス)が点灯 -> buy
  それ以外 -> hold

MACD・出来高急増は個別検証で優位性が確認できず、RSI/BB反発は今回の想定保有期間
(1〜2年)には不向きと判明したため使わない。「良い銘柄を長く持つ」というコンセプトの
ため、積極的な売り推奨(sell)は設けない(保有継続中の警戒情報が欲しければv1の
sellロジックを別途参照する)。
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
from common.technical import calc_indicators, check_signals, get_technical_state  # noqa: E402

RULE_VERSION = "v3.0"
HIGH_GRADES = {"S", "A", "B"}

SIGNAL_LABELS = {"GC": "ゴールデンクロス"}


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
    query = "SELECT ticker, total_score, grade FROM scores WHERE snapshot_date = ?"
    return pd.read_sql_query(query, conn, params=(snapshot_date,))


def timing_flags(signals: list[dict]) -> dict:
    types = {s["type"] for s in signals}
    return {"golden_cross": "GC" in types}


def decide_quality_timing(grade: str, total_score: float, flags: dict) -> Decision:
    if grade in HIGH_GRADES and flags.get("golden_cross"):
        confidence = 0.6 + (0.1 if grade == "S" else 0.0)
        reason = (
            f"グレード{grade}(総合スコア{total_score:.1f})でゴールデンクロスを検知"
            "(1〜2年保有のバックテストで頑健な優位性を確認済み)"
        )
        return Decision("buy", reason, min(confidence, 1.0))
    return Decision("hold", f"グレード{grade}、タイミングシグナルなし", 0.3)


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

    decision_date = snapshot_date
    action_counts = {"buy": 0, "hold": 0}
    skipped = 0

    for _, row in scores_df.iterrows():
        ticker = row["ticker"]
        hist = load_price_history(conn, ticker, snapshot_date)
        if len(hist) < 75:
            skipped += 1
            continue

        hist = calc_indicators(hist)
        signals = check_signals(hist)
        flags = timing_flags(signals)
        technical = get_technical_state(hist)

        decision = decide_quality_timing(row["grade"], row["total_score"], flags)
        upsert_decision(conn, ticker, decision_date, decision, row["total_score"], row["grade"], flags, technical.get("current_price"))
        action_counts[decision.action] += 1

    conn.close()
    print(f"snapshot_date={snapshot_date}, decision_date={decision_date}, rule_version={RULE_VERSION}")
    print(f"action分布: {action_counts}, 履歴不足でスキップ: {skipped}")


if __name__ == "__main__":
    main()
