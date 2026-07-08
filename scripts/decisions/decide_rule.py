"""スコア＋テクニカルシグナルによるルールベース売買判断。

「グレードでファンダ選別 → テクニカルシグナルでタイミング判定」の二段階。
優先順位は上から評価し、最初に一致した条件を採用する:
1. grade in {S,A} かつ bullish_signals>=1 -> buy
2. grade == B かつ bullish_signals>=2 -> buy
3. grade in {D,E} かつ 出来高急増 -> sell
4. grade in {D,E} かつ MA25・MA75両方を下回る -> sell
5. valuation_score<15 かつ BB反発なし -> hold(割高警戒)
6. デフォルト -> hold
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

RULE_VERSION = "v1.0"

SIGNAL_LABELS = {
    "GC": "ゴールデンクロス",
    "RSI": "RSI反発",
    "MACD": "MACDクロス",
    "BB": "ボリンジャー-2σ反発",
    "VOL": "出来高急増",
}


@dataclass
class Decision:
    action: str
    reason: str
    confidence: float


def load_price_history(conn, ticker: str, as_of_date: str) -> pd.DataFrame:
    # as_of_date以前のデータだけに絞らないと、過去のsnapshot_dateを指定して再実行(バックフィル)
    # した際に、実際には決定時点でまだ存在しなかった未来の株価まで見て判断してしまう
    query = "SELECT date, open, high, low, close, volume FROM price_daily WHERE ticker = ? AND date <= ? ORDER BY date ASC"
    df = pd.read_sql_query(query, conn, params=(ticker, as_of_date), parse_dates=["date"])
    if df.empty:
        return df
    df = df.set_index("date").rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    return df


def load_latest_scores(conn, snapshot_date: str) -> pd.DataFrame:
    query = "SELECT ticker, total_score, grade, score_valuation FROM scores WHERE snapshot_date = ?"
    return pd.read_sql_query(query, conn, params=(snapshot_date,))


def signal_flags(signals: list[dict]) -> dict:
    types = {s["type"] for s in signals}
    return {
        "golden_cross": "GC" in types,
        "rsi_oversold_bounce": "RSI" in types,
        "macd_cross": "MACD" in types,
        "bb_lower_bounce": "BB" in types,
        "volume_surge": "VOL" in types,
    }


def signal_names(flags: dict) -> str:
    type_map = {
        "golden_cross": "GC",
        "rsi_oversold_bounce": "RSI",
        "macd_cross": "MACD",
        "bb_lower_bounce": "BB",
    }
    return "、".join(SIGNAL_LABELS[type_map[k]] for k, v in flags.items() if v and k in type_map)


def decide_rule_based(grade: str, total_score: float, valuation_score: float | None, flags: dict, technical: dict) -> Decision:
    bullish_signals = sum([
        flags.get("golden_cross", False),
        flags.get("rsi_oversold_bounce", False),
        flags.get("macd_cross", False),
        flags.get("bb_lower_bounce", False),
    ])

    if grade in ("S", "A") and bullish_signals >= 1:
        confidence = min(1.0, 0.5 + 0.15 * bullish_signals + (0.1 if grade == "S" else 0))
        reason = f"グレード{grade}(総合スコア{total_score:.1f})に加え、{bullish_signals}件の買いシグナル({signal_names(flags)})が点灯"
        return Decision("buy", reason, confidence)

    if grade == "B" and bullish_signals >= 2:
        confidence = 0.4 + 0.1 * bullish_signals
        reason = f"グレード{grade}だが{bullish_signals}件のシグナルが同時点灯({signal_names(flags)})"
        return Decision("buy", reason, confidence)

    if grade in ("D", "E") and flags.get("volume_surge", False):
        return Decision("sell", f"グレード{grade}で出来高急増を検知", 0.6)

    if grade in ("D", "E") and technical.get("price_below_ma25", False) and technical.get("price_below_ma75", False):
        return Decision("sell", f"グレード{grade}かつ価格が両移動平均線を下回る", 0.55)

    if valuation_score is not None and valuation_score < 15 and not flags.get("bb_lower_bounce", False):
        return Decision("hold", f"割安性スコアが低く({valuation_score:.0f})過熱感あり、新規買いは様子見", 0.3)

    return Decision("hold", f"グレード{grade}、明確な売買シグナルなし", 0.3)


def upsert_decision(conn, ticker: str, decision_date: str, decision: Decision, total_score, grade, flags: dict, price: float) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO decisions
                (ticker, decision_date, action, decision_source, rule_version,
                 total_score, grade, technical_signals, reason, price_at_decision, confidence)
            VALUES (?, ?, ?, 'rule', ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, decision_date, decision_source, rule_version) DO UPDATE SET
                action=excluded.action, rule_version=excluded.rule_version,
                total_score=excluded.total_score, grade=excluded.grade,
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

    # decision_dateはsnapshot_dateと一致させる(通常運用では同日だが、過去日付を
    # 指定してバックフィルした場合に「今日判断した」と誤って記録されるのを防ぐ)
    decision_date = snapshot_date
    action_counts = {"buy": 0, "sell": 0, "hold": 0}
    skipped = 0

    for _, row in scores_df.iterrows():
        ticker = row["ticker"]
        hist = load_price_history(conn, ticker, snapshot_date)
        if len(hist) < 75:
            skipped += 1
            continue

        hist = calc_indicators(hist)
        signals = check_signals(hist)
        flags = signal_flags(signals)
        technical = get_technical_state(hist)

        decision = decide_rule_based(row["grade"], row["total_score"], row["score_valuation"], flags, technical)
        upsert_decision(conn, ticker, decision_date, decision, row["total_score"], row["grade"], flags, technical.get("current_price"))
        action_counts[decision.action] += 1

    conn.close()
    print(f"snapshot_date={snapshot_date}, decision_date={decision_date}")
    print(f"action分布: {action_counts}, 履歴不足でスキップ: {skipped}")


if __name__ == "__main__":
    main()
