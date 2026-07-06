"""ルールベース判断の的中率サマリー(グレード別・アクション別・シグナル別)。

自動チューニングは行わず、config/scoring_config.yamlやdecide_rule.pyの
閾値を人間が調整するための参考情報として提示する(Phase 1)。
将来のLLM層(Phase 2)では、この集計結果をプロンプトのコンテキストとして注入する想定。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection  # noqa: E402


def load_evaluated_decisions(conn, lookback_days: int) -> pd.DataFrame:
    query = """
    SELECT d.decision_id, d.ticker, d.action, d.grade, d.technical_signals, d.confidence,
           o.is_correct, o.return_pct, o.outcome_label
    FROM decisions d
    JOIN decision_outcomes o ON d.decision_id = o.decision_id
    WHERE d.decision_date >= date('now', ?) AND o.outcome_label != 'unevaluable'
    """
    return pd.read_sql_query(query, conn, params=(f"-{lookback_days} days",))


def summarize_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = df.groupby(group_col).agg(
        n=("decision_id", "count"),
        win_rate=("is_correct", "mean"),
        avg_return_pct=("return_pct", "mean"),
    )
    grouped["win_rate"] = (grouped["win_rate"] * 100).round(1)
    grouped["avg_return_pct"] = grouped["avg_return_pct"].round(2)
    return grouped.sort_values("n", ascending=False)


def summarize_by_signal_combo(df: pd.DataFrame) -> pd.DataFrame:
    def signal_combo(json_str: str) -> str:
        flags = json.loads(json_str)
        active = sorted(k for k, v in flags.items() if v)
        return "+".join(active) if active else "(シグナルなし)"

    df = df.copy()
    df["signal_combo"] = df["technical_signals"].apply(signal_combo)
    return summarize_by(df, "signal_combo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=90)
    args = parser.parse_args()

    conn = get_connection()
    df = load_evaluated_decisions(conn, args.lookback_days)
    conn.close()

    if df.empty:
        print(f"直近{args.lookback_days}日で評価済みのdecisionsがありません")
        return

    print(f"=== ルールベース判断のパフォーマンス(直近{args.lookback_days}日、評価済みN={len(df)}件) ===\n")

    print("--- action別 ---")
    print(summarize_by(df, "action"))

    print("\n--- grade別 ---")
    print(summarize_by(df, "grade"))

    print("\n--- シグナル組み合わせ別 ---")
    print(summarize_by_signal_combo(df))


if __name__ == "__main__":
    main()
