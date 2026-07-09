"""「良い銘柄を見つけて長めに持つ」方針の判断エンジン(v3)。

decide_rule.py(v1)・decide_composite.py(v2)は、バックテストを複数年に拡張した結果、
どちらも技術的タイミング層(GC/MACD/合成テクニカルスコア)に頑健な優位性が
見られなかった(scripts/backtest/decision_engine_extended_backtest.py参照)。

一方、scripts/backtest/technical_signal_event_study.pyで個別検証したところ、
RSI反発・BB反発の2つだけは5年間・複数保有期間で一貫してベースライン超過の
プラスの予測力があると判明。さらにscripts/backtest/fundamental_plus_timing_backtest.py
で「総合スコア上位(グレードS/A/B)」と組み合わせたところ、2021〜2026年の
全ての年でRSI/BB単体を上回り(特に不調な年ほど改善幅が大きい)、頑健性を確認済み。

本エンジンはこの検証結果に忠実に、以下のシンプルな二値ルールのみを採用する:
  グレードがS/A/B かつ (RSI反発 または BB反発) が点灯 -> buy
  それ以外 -> hold

GC・MACD・出来高急増は個別検証で短中期にマイナスの寄与だったため使わない。
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
from common.technical import calc_indicators, check_signals, get_technical_state  # noqa: E402

RULE_VERSION = "v3.0"
HIGH_GRADES = {"S", "A", "B"}

SIGNAL_LABELS = {"RSI": "RSI反発", "BB": "ボリンジャー-2σ反発"}


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
    return {"rsi_oversold_bounce": "RSI" in types, "bb_lower_bounce": "BB" in types}


def decide_quality_timing(grade: str, total_score: float, flags: dict) -> Decision:
    if grade in HIGH_GRADES and (flags.get("rsi_oversold_bounce") or flags.get("bb_lower_bounce")):
        fired = [SIGNAL_LABELS[k] for k, label in (("RSI", "rsi_oversold_bounce"), ("BB", "bb_lower_bounce")) if flags.get(label)]
        confidence = 0.5 + 0.1 * len(fired) + (0.1 if grade == "S" else 0.0)
        reason = f"グレード{grade}(総合スコア{total_score:.1f})で{'・'.join(fired)}を検知(バックテストで頑健な優位性を確認済み)"
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
