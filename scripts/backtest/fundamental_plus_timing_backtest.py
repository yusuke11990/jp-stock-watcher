"""「総合スコア上位(ファンダメンタルズ)」×「RSI反発/BB反発(タイミング)」を組み合わせた
場合に、単体より優位性が上がるか・落ちるかを検証する。

technical_signal_event_study.pyでRSI反発・BB反発は単体で(銘柄のファンダ的な質を
問わず)ベースライン超過のプラスの予測力があると分かった。ただしそれは「どんな銘柄でも
RSI/BB反発は効く」という検証であり、「質の良い銘柄に限定した場合」の効果は別に
確認する必要がある。本スクリプトは以下を比較する:

1. ベースライン: 銘柄を問わず、20営業日おきにサンプルした日のフォワードリターン
2. RSI/BB単体: ファンダスコアを問わず、シグナルが出た日
3. 高スコア単体: シグナルの有無を問わず、コホートのgradeがS/A/Bの期間の全営業日
4. 組み合わせ: 高グレード期間中に、かつRSI/BB反発が出た日

fundamental panelはdecision_engine_extended_backtest.pyと同じ開示日ベースの
点在時点コホートスコアを使う(scoring_config.yaml本番の全指標ではなく、
既に検証済みの7軸percentileスコアの代理値)。

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

    baseline: dict[str, list[float]] = {h: [] for h in HORIZONS}
    signal_only: dict[str, list[float]] = {h: [] for h in HORIZONS}
    high_grade_only: dict[str, list[float]] = {h: [] for h in HORIZONS}
    combined: dict[str, list[float]] = {h: [] for h in HORIZONS}

    tickers = panel["ticker"].unique()
    for n, ticker in enumerate(tickers, start=1):
        cohorts = panel[panel["ticker"] == ticker]
        price_df = load_bt_price_history(bt_conn, ticker)
        if len(price_df) < 100:
            continue

        ind = calc_indicators(price_df.copy())
        signals = detect_signals(ind)
        rsi_or_bb = signals["RSI"].fillna(False) | signals["BB"].fillna(False)
        close = ind["Close"]

        fwd_returns = {h_label: (close.shift(-h) / close - 1) * 100 for h_label, h in HORIZONS.items()}

        # 各営業日について、その日時点で有効なコホートのgradeを引く(コホート数は
        # 銘柄あたり数件程度なので、シグナル発生日・サンプル対象日だけループすれば軽い)
        dates = price_df.index

        # ベースライン(20営業日おきにサンプル)
        for i in range(0, len(dates), 20):
            day = dates[i]
            for h_label in HORIZONS:
                v = fwd_returns[h_label].iloc[i]
                if pd.notna(v):
                    baseline[h_label].append(v)

        # RSI/BB反発が出た日
        sig_idx = [i for i, d in enumerate(dates) if rsi_or_bb.iloc[i]]
        for i in sig_idx:
            day = dates[i]
            cohort = find_applicable_cohort(cohorts, day)
            for h_label in HORIZONS:
                v = fwd_returns[h_label].iloc[i]
                if pd.isna(v):
                    continue
                signal_only[h_label].append(v)
                if cohort is not None and cohort["grade"] in HIGH_GRADES:
                    combined[h_label].append(v)

        # 高グレード期間中の全営業日(タイミングを問わない、20営業日おきサンプル)
        for i in range(0, len(dates), 20):
            day = dates[i]
            cohort = find_applicable_cohort(cohorts, day)
            if cohort is not None and cohort["grade"] in HIGH_GRADES:
                for h_label in HORIZONS:
                    v = fwd_returns[h_label].iloc[i]
                    if pd.notna(v):
                        high_grade_only[h_label].append(v)

        if n % 500 == 0:
            print(f"[{n}/{len(tickers)}] 経過{time.monotonic()-start:.0f}秒")

    print(f"\n全{len(tickers)}銘柄完了(経過{time.monotonic()-start:.0f}秒)\n")

    print("=== 保有期間別の比較(平均リターン%) ===")
    print(f"{'保有期間':10s} {'ベースライン':>12s} {'RSI/BB単体':>12s} {'高グレード単体':>14s} {'組み合わせ':>12s}")
    for h_label in HORIZONS:
        b = pd.Series(baseline[h_label])
        s = pd.Series(signal_only[h_label])
        g = pd.Series(high_grade_only[h_label])
        c = pd.Series(combined[h_label])
        print(
            f"{h_label:10s} {b.mean():>11.2f}% {s.mean():>11.2f}%(n={len(s)}) "
            f"{g.mean():>13.2f}%(n={len(g)}) {c.mean():>11.2f}%(n={len(c)})"
        )

    print("\n=== 勝率(>+2%)比較 ===")
    for h_label in HORIZONS:
        s = pd.Series(signal_only[h_label])
        g = pd.Series(high_grade_only[h_label])
        c = pd.Series(combined[h_label])
        print(
            f"{h_label:10s} RSI/BB単体={((s>2).mean() if len(s) else float('nan')):.1%}  "
            f"高グレード単体={((g>2).mean() if len(g) else float('nan')):.1%}  "
            f"組み合わせ={((c>2).mean() if len(c) else float('nan')):.1%}"
        )

    conn.close()
    bt_conn.close()


if __name__ == "__main__":
    main()
