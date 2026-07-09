"""「総合スコア上位(ファンダメンタルズ)」×「タイミングシグナル」を組み合わせた場合に、
単体より優位性が上がるか・落ちるかを検証する。1-2年保有を想定する場合、短期の押し目
シグナル(RSI/BB反発)と、トレンド転換シグナル(GC)のどちらが適しているかも比較する。

technical_signal_event_study.pyの複数保有期間検証で、RSI反発は21日保有では強いが
24ヶ月保有ではベースライン以下(むしろ逆行)になる一方、GC(ゴールデンクロス)は
短期は弱いが12ヶ月保有では最も良い成績になると分かった。本スクリプトはこれを
ファンダメンタルズと組み合わせた場合でも同じ傾向になるか、かつ年ごとに頑健か検証する。

1. ベースライン: 銘柄を問わず、20営業日おきにサンプルした日のフォワードリターン
2. タイミングシグナル単体: ファンダスコアを問わず、シグナルが出た日
3. 高スコア単体: シグナルの有無を問わず、コホートのgradeがS/A/Bの期間の全営業日
4. 組み合わせ: 高グレード期間中に、かつシグナルが出た日

実行: python scripts/backtest/fundamental_plus_timing_backtest.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection, get_backtest_connection  # noqa: E402
from common.technical import calc_indicators  # noqa: E402
from scoring.compute_scores import load_config, score_to_grade  # noqa: E402
from backtest.quality_score_v2_backtest import CATEGORY_METRICS_V2, CATEGORY_WEIGHTS_V2, compute_score_per_cohort  # noqa: E402
from backtest.decision_engine_extended_backtest import (  # noqa: E402
    build_fundamental_panel_bt, attach_value_momentum_bt, load_bt_price_history,
)
from backtest.decision_engine_comparison_backtest import build_cohort_windows, find_applicable_cohort  # noqa: E402
from backtest.technical_signal_event_study import detect_signals, HORIZONS

HIGH_GRADES = {"S", "A", "B"}
SIGNAL_GROUPS = {
    "RSI/BB(短期押し目)": lambda s: s["RSI"].fillna(False) | s["BB"].fillna(False),
    "GC(トレンド転換)": lambda s: s["GC"].fillna(False),
}
FOCUS_HORIZONS = ["21営業日", "252営業日(12ヶ月)", "504営業日(24ヶ月)"]


def main():
    start = time.monotonic()
    conn = get_connection()
    bt_conn = get_backtest_connection()
    scoring_config = load_config()

    panel = build_fundamental_panel_bt(conn, bt_conn)
    panel = attach_value_momentum_bt(bt_conn, panel)
    panel = compute_score_per_cohort(panel, CATEGORY_METRICS_V2, CATEGORY_WEIGHTS_V2, "total_score_7axis")
    panel["total_score"] = panel["total_score_7axis"]
    panel["grade"] = panel["total_score"].apply(lambda v: score_to_grade(v, scoring_config["grade_thresholds"]))
    panel = panel.dropna(subset=["total_score", "signal_date", "grade"])
    panel = build_cohort_windows(panel)
    print(f"コホート数: {len(panel)}件、対象銘柄数: {panel['ticker'].nunique()}件")

    # records[group_name] = list of {"date":..., "return":..., "is_high_grade":...} per horizon
    records: dict[str, dict[str, list[dict]]] = {g: {h: [] for h in HORIZONS} for g in SIGNAL_GROUPS}
    baseline: dict[str, list[float]] = {h: [] for h in HORIZONS}

    tickers = panel["ticker"].unique()
    for n, ticker in enumerate(tickers, start=1):
        cohorts = panel[panel["ticker"] == ticker]
        price_df = load_bt_price_history(bt_conn, ticker)
        if len(price_df) < 100:
            continue

        ind = calc_indicators(price_df.copy())
        signals = detect_signals(ind)
        close = ind["Close"]
        dates = price_df.index

        fwd_returns = {h_label: (close.shift(-h) / close - 1) * 100 for h_label, h in HORIZONS.items()}

        for i in range(0, len(dates), 20):
            for h_label in HORIZONS:
                v = fwd_returns[h_label].iloc[i]
                if pd.notna(v):
                    baseline[h_label].append(v)

        for group_name, sig_fn in SIGNAL_GROUPS.items():
            sig_bool = sig_fn(signals)
            sig_idx = [i for i, d in enumerate(dates) if sig_bool.iloc[i]]
            for i in sig_idx:
                day = dates[i]
                cohort = find_applicable_cohort(cohorts, day)
                is_high = cohort is not None and cohort["grade"] in HIGH_GRADES
                for h_label in HORIZONS:
                    v = fwd_returns[h_label].iloc[i]
                    if pd.isna(v):
                        continue
                    records[group_name][h_label].append({"date": day, "return": v, "is_high_grade": is_high})

        if n % 500 == 0:
            print(f"[{n}/{len(tickers)}] 経過{time.monotonic()-start:.0f}秒")

    print(f"\n全{len(tickers)}銘柄完了(経過{time.monotonic()-start:.0f}秒)\n")

    print("=== 保有期間別の比較(平均リターン%、高グレードのみ vs 全グレード) ===")
    for group_name in SIGNAL_GROUPS:
        print(f"\n--- {group_name} ---")
        print(f"{'保有期間':25s} {'ベースライン':>12s} {'シグナル単体(全グレード)':>18s} {'組み合わせ(高グレード)':>18s}")
        for h_label in HORIZONS:
            b = pd.Series(baseline[h_label])
            all_recs = pd.DataFrame(records[group_name][h_label])
            if all_recs.empty:
                print(f"{h_label:25s} サンプルなし")
                continue
            s = all_recs["return"]
            c = all_recs.loc[all_recs["is_high_grade"], "return"]
            print(
                f"{h_label:25s} {b.mean():>11.2f}% {s.mean():>11.2f}%(n={len(s)}) "
                f"{c.mean():>11.2f}%(n={len(c)})"
            )

    print("\n\n=== 年別の頑健性チェック(組み合わせ、高グレードのみ) ===")
    for group_name in SIGNAL_GROUPS:
        print(f"\n--- {group_name} ---")
        for h_label in FOCUS_HORIZONS:
            all_recs = pd.DataFrame(records[group_name][h_label])
            if all_recs.empty:
                continue
            combo = all_recs[all_recs["is_high_grade"]].copy()
            combo["year"] = pd.to_datetime(combo["date"]).dt.year
            print(f"\n[{h_label}]")
            print(combo.groupby("year")["return"].agg(["size", "mean"]).round(2))

    conn.close()
    bt_conn.close()


if __name__ == "__main__":
    main()
