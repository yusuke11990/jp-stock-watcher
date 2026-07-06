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
           d.rule_version, d.technical_composite_score, d.regime_market,
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
    """v1(二値シグナルのJSON)専用。v2はtechnical_signalsの中身が連続値スコアのため対象外"""
    def signal_combo(json_str: str) -> str | None:
        flags = json.loads(json_str)
        if not all(isinstance(v, bool) for v in flags.values()):
            return None  # v2形式(連続値)は非対応
        active = sorted(k for k, v in flags.items() if v)
        return "+".join(active) if active else "(シグナルなし)"

    df = df.copy()
    df["signal_combo"] = df["technical_signals"].apply(signal_combo)
    df = df.dropna(subset=["signal_combo"])
    if df.empty:
        return df
    return summarize_by(df, "signal_combo")


def summarize_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    """v2専用。レジーム(trend/range/transition)別の的中率"""
    sub = df[df["regime_market"].notna()]
    if sub.empty:
        return sub
    return summarize_by(sub, "regime_market")


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

    print("--- rule_version別(v1旧ルール vs v2合成エンジンの比較) ---")
    print(summarize_by(df, "rule_version"))

    for version in sorted(df["rule_version"].dropna().unique()):
        sub = df[df["rule_version"] == version]
        print(f"\n### rule_version={version} (N={len(sub)}) ###")

        print("--- action別 ---")
        print(summarize_by(sub, "action"))

        print("\n--- grade別 ---")
        print(summarize_by(sub, "grade"))

        if version == "v1.0":
            print("\n--- シグナル組み合わせ別 ---")
            combo = summarize_by_signal_combo(sub)
            print(combo if not combo.empty else "(該当データなし)")
        else:
            print("\n--- レジーム別 ---")
            regime = summarize_by_regime(sub)
            print(regime if not regime.empty else "(該当データなし)")


if __name__ == "__main__":
    main()
