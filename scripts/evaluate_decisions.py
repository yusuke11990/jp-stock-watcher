"""判断(decisions)から営業日21日(約1ヶ月)経過した銘柄を、実際の株価で評価する。

price_dailyに実在する取引日数でカウントすることで休場日・祝日は自動的に除外される。
60営業日経過しても評価できない(上場廃止等でprice_dailyが途絶)場合はunevaluableとして打ち切る。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.db import get_connection  # noqa: E402

EVAL_HORIZON_TRADING_DAYS = 21
UNEVALUABLE_HORIZON_TRADING_DAYS = 60
NEUTRAL_THRESHOLD_PCT = 2.0


def find_pending_decisions(conn) -> list[dict]:
    query = """
    SELECT d.decision_id, d.ticker, d.decision_date, d.action, d.price_at_decision
    FROM decisions d
    LEFT JOIN decision_outcomes o ON d.decision_id = o.decision_id
    WHERE o.decision_id IS NULL
    """
    cols = ["decision_id", "ticker", "decision_date", "action", "price_at_decision"]
    return [dict(zip(cols, row)) for row in conn.execute(query)]


def count_market_trading_days_elapsed(conn, since_date: str) -> int:
    """市場全体の取引日カレンダーで経過営業日数を数える。

    特定銘柄自身のprice_dailyの行数で数えると、上場廃止等でその銘柄の
    データが二度と増えなくなった場合に経過日数が永遠に増えず、
    unevaluable判定に到達できなくなるため、市場全体の営業日を基準にする。
    """
    query = "SELECT COUNT(DISTINCT date) FROM price_daily WHERE date > ?"
    return conn.execute(query, (since_date,)).fetchone()[0]


def get_nth_trading_day(conn, ticker: str, since_date: str, n: int):
    query = """
    SELECT date, close FROM price_daily
    WHERE ticker = ? AND date > ?
    ORDER BY date ASC
    LIMIT 1 OFFSET ?
    """
    return conn.execute(query, (ticker, since_date, n - 1)).fetchone()


def evaluate_decision(conn, decision: dict) -> dict | None:
    eval_row = get_nth_trading_day(conn, decision["ticker"], decision["decision_date"], EVAL_HORIZON_TRADING_DAYS)
    if eval_row is None:
        return None

    eval_date, price_at_eval = eval_row
    return_pct = (price_at_eval - decision["price_at_decision"]) / decision["price_at_decision"] * 100

    action = decision["action"]
    if action == "buy":
        if return_pct > NEUTRAL_THRESHOLD_PCT:
            is_correct, label = True, "correct"
        elif return_pct < -NEUTRAL_THRESHOLD_PCT:
            is_correct, label = False, "incorrect"
        else:
            is_correct, label = False, "neutral"
    elif action == "sell":
        if return_pct < -NEUTRAL_THRESHOLD_PCT:
            is_correct, label = True, "correct"
        elif return_pct > NEUTRAL_THRESHOLD_PCT:
            is_correct, label = False, "incorrect"
        else:
            is_correct, label = False, "neutral"
    else:  # hold
        if abs(return_pct) <= NEUTRAL_THRESHOLD_PCT:
            is_correct, label = True, "correct"
        else:
            is_correct, label = False, "incorrect"

    return {
        "decision_id": decision["decision_id"],
        "eval_date": eval_date,
        "price_at_eval": price_at_eval,
        "return_pct": return_pct,
        "is_correct": int(is_correct),
        "outcome_label": label,
    }


def mark_unevaluable(conn, decision: dict) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO decision_outcomes
                (decision_id, eval_date, price_at_eval, return_pct, is_correct, outcome_label)
            VALUES (?, ?, 0, 0, NULL, 'unevaluable')
            """,
            (decision["decision_id"], decision["decision_date"]),
        )


def upsert_outcome(conn, outcome: dict) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO decision_outcomes
                (decision_id, eval_date, price_at_eval, return_pct, is_correct, outcome_label)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                outcome["decision_id"], outcome["eval_date"], outcome["price_at_eval"],
                outcome["return_pct"], outcome["is_correct"], outcome["outcome_label"],
            ),
        )


def main():
    conn = get_connection()
    pending = find_pending_decisions(conn)

    evaluated, unevaluable, still_pending = 0, 0, 0
    for decision in pending:
        elapsed = count_market_trading_days_elapsed(conn, decision["decision_date"])
        if elapsed < EVAL_HORIZON_TRADING_DAYS:
            still_pending += 1
            continue

        outcome = evaluate_decision(conn, decision)
        if outcome is not None:
            upsert_outcome(conn, outcome)
            evaluated += 1
        elif elapsed >= UNEVALUABLE_HORIZON_TRADING_DAYS:
            mark_unevaluable(conn, decision)
            unevaluable += 1
        else:
            still_pending += 1

    conn.close()
    print(f"評価済み: {evaluated}, 評価不能: {unevaluable}, 未到達(継続待ち): {still_pending}")


if __name__ == "__main__":
    main()
