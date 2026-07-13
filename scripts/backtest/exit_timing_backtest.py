"""decide_quality_timing.py(v3)・decide_quality_timing_v4.py(v4)は「良い銘柄を買って
1〜2年持つ」方針で、買いシグナルの精度はここまで繰り返し検証・改善してきたが、
「いつ売るか」は一度も検証していなかった。今は固定期間(12ヶ月・24ヶ月)保有の
バイ・アンド・ホールドを前提にバックテストしているが、実際には途中で利確/損切りする
選択肢がある。それが実際のリターンを改善するのか検証する。

対象イベントはgrade_valuation_near52w_backtest.pyと同じ(グレードS/A/B/C×52週高値の
95%圏内への接近×割安性上位50%)。各イベントについて、単純な固定期間保有(ベースライン)
と、トレーリングストップ(直近高値からX%下落したら決済)を比較する。トレーリングストップは
「上昇を伸ばしつつ下落を限定する」ための最も基本的な出口ルールで、固定の利確ライン
(ATRベースの3.5倍等)は1〜2年保有では早すぎる利確になりがちなため採用しない。

実行: python scripts/backtest/exit_timing_backtest.py
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

HORIZON_DAYS = 504  # 24ヶ月
TARGET_GRADES = ("S", "A", "B", "C")
TRAIL_PCTS = (0.10, 0.15, 0.20, 0.25, 0.30)


def build_events(panel: pd.DataFrame, bt_conn) -> list[dict]:
    events = []
    tickers = panel["ticker"].unique()
    start = time.monotonic()
    for n, ticker in enumerate(tickers, start=1):
        rows = panel[panel["ticker"] == ticker]
        price_df = load_bt_price_history(bt_conn, ticker)
        if len(price_df) < 126:
            continue
        ind = calc_indicators(price_df.copy())
        close = ind["Close"]
        dates = price_df.index
        rolling_high = close.rolling(252, min_periods=126).max()
        near_high = close >= rolling_high * 0.95
        near_high_trigger = near_high & ~near_high.shift(1).fillna(False).infer_objects(copy=False)
        idx = [i for i, v in enumerate(near_high_trigger) if v]
        for i in idx:
            day = dates[i]
            cohort = find_applicable_cohort(rows, day)
            if cohort is None or cohort["grade"] not in TARGET_GRADES:
                continue
            entry_price = close.iloc[i]
            if pd.isna(entry_price) or entry_price <= 0:
                continue
            if i + HORIZON_DAYS >= len(dates):
                continue
            path = close.iloc[i : i + HORIZON_DAYS + 1].reset_index(drop=True)
            if path.isna().any():
                continue
            events.append({
                "ticker": ticker, "date": day, "grade": cohort["grade"], "score_value": cohort["score_value"],
                "entry_price": entry_price, "path": path,
            })
        if n % 1000 == 0:
            print(f"[{n}/{len(tickers)}] {time.monotonic()-start:.0f}s")
    return events


def simulate_trailing_stop(path: pd.Series, trail_pct: float) -> tuple[float, int]:
    """pathはentry日を含むclose系列(0番目がentry)。戻り値は(実現リターン, 保有日数)。"""
    entry_price = path.iloc[0]
    running_max = entry_price
    for t in range(1, len(path)):
        price = path.iloc[t]
        running_max = max(running_max, price)
        stop_price = running_max * (1 - trail_pct)
        if price <= stop_price:
            return (price / entry_price - 1) * 100, t
    return (path.iloc[-1] / entry_price - 1) * 100, len(path) - 1


def main():
    conn = get_connection()
    bt_conn = get_backtest_connection()
    scoring_config = load_config()

    panel = build_fundamental_panel_bt(conn, bt_conn)
    panel = attach_value_momentum_bt(bt_conn, panel)
    panel = compute_score_per_cohort(panel, CATEGORY_METRICS_V2, CATEGORY_WEIGHTS_V2, "total_score_7axis")
    panel["total_score"] = panel["total_score_7axis"]
    panel["grade"] = panel["total_score"].apply(lambda v: score_to_grade(v, scoring_config["grade_thresholds"]))
    panel = panel.dropna(subset=["total_score", "signal_date", "grade", "score_value"])
    panel = build_cohort_windows(panel)

    events = build_events(panel, bt_conn)
    print(f"\n24ヶ月分の価格パスが確保できたイベント数: {len(events)}")

    # v3と同じ「割安性上位50%(グループ内基準)」フィルタ
    score_values = [e["score_value"] for e in events]
    median_val = pd.Series(score_values).median()
    events = [e for e in events if e["score_value"] >= median_val]
    print(f"割安性上位50%フィルタ後: {len(events)}件\n")

    baseline_returns = pd.Series([(e["path"].iloc[-1] / e["entry_price"] - 1) * 100 for e in events])
    print(f"=== ベースライン(24ヶ月固定バイ・アンド・ホールド) ===")
    print(f"  n={len(baseline_returns)}  平均={baseline_returns.mean():+.2f}%  中央値={baseline_returns.median():+.2f}%")

    print(f"\n=== トレーリングストップ(直近高値からX%下落で決済、それ以外は24ヶ月保有) ===")
    for trail_pct in TRAIL_PCTS:
        results = [simulate_trailing_stop(e["path"], trail_pct) for e in events]
        returns = pd.Series([r[0] for r in results])
        holding_days = pd.Series([r[1] for r in results])
        triggered = (holding_days < HORIZON_DAYS).mean() * 100
        print(f"  trail={trail_pct:.0%}: 平均={returns.mean():+.2f}%  中央値={returns.median():+.2f}%  "
              f"平均保有日数={holding_days.mean():.0f}日  途中決済率={triggered:.1f}%")

    print("\n=== 年別ロバスト性(trail=20%、ベースライン vs トレーリングストップ) ===")
    events_df = pd.DataFrame({
        "date": [e["date"] for e in events],
        "baseline": baseline_returns.values,
    })
    results20 = [simulate_trailing_stop(e["path"], 0.20) for e in events]
    events_df["trail20"] = [r[0] for r in results20]
    events_df["year"] = pd.to_datetime(events_df["date"]).dt.year
    for yr, g in events_df.groupby("year"):
        if len(g) < 20:
            continue
        print(f"  {yr}: n={len(g)}  ベースライン平均={g['baseline'].mean():+.2f}%  trail20%平均={g['trail20'].mean():+.2f}%")

    conn.close()
    bt_conn.close()


if __name__ == "__main__":
    main()
