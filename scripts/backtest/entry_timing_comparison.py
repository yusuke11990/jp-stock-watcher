"""16:20にbuyと判断された場合、実際にいつ買うのがよいかを検証する。

decide_rule.py/decide_composite.pyの判断は当日(T日)の引け後データで生成されるため、
「T日の引けで即座に買う」ことは物理的に不可能(15:00引け後の16:20に判断が出るため)。
現実的な選択肢は:
  (a) 翌営業日(T+1)の寄り付きで成り行き買い
  (b) 翌営業日(T+1)の午前の値動きを見て、引け際(15時頃、ほぼT+1の終値)に買う

decision_engine_comparison_backtest.pyと同じ日次シミュレーションで買いシグナルを
再現し、エントリー価格だけを3パターン(T日終値=参考のみ/T+1日始値/T+1日終値)に
差し替えて、同じ決定日T基準で21営業日後の株価までのリターンを比較する。
エグジット地点をT基準で揃えることで、エントリータイミングの違いだけを比較できる。

実行: python scripts/backtest/entry_timing_comparison.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection  # noqa: E402
from common.technical import calc_indicators, check_signals, get_technical_state  # noqa: E402
from common.technical_v2 import compute_technical_v2  # noqa: E402
from scoring.compute_scores import load_config, score_to_grade  # noqa: E402
from decisions.decide_rule import decide_rule_based, signal_flags  # noqa: E402
from decisions.decide_composite import decide_composite, load_configs as load_v2_configs  # noqa: E402
from backtest.quality_score_v2_backtest import CATEGORY_METRICS_V2, CATEGORY_WEIGHTS_V2, compute_score_per_cohort  # noqa: E402
from backtest.decision_engine_comparison_backtest import (  # noqa: E402
    EVAL_HORIZON_TRADING_DAYS, NEUTRAL_THRESHOLD_PCT, MIN_TECHNICAL_HISTORY_ROWS,
    build_fundamental_panel, attach_value_momentum, build_cohort_windows,
    find_applicable_cohort, load_full_price_daily,
    cached_market_regime, cached_sector_regime, classify_outcome,
)


def simulate_ticker_entry_timing(conn, ticker: str, cohorts: pd.DataFrame, technical_config: dict, decision_config: dict) -> list:
    """買いシグナルが出た日について、3つのエントリー地点でのリターンを同時に記録する。
    (T日終値は参考値。実際に取引可能なのはT+1日の始値・終値の2択)
    """
    price_df = load_full_price_daily(conn, ticker)
    records = []
    if len(price_df) < MIN_TECHNICAL_HISTORY_ROWS + EVAL_HORIZON_TRADING_DAYS + 1:
        return records

    dates = price_df.index
    prev_action_v1, prev_action_v2 = "hold", "hold"

    # T+1が存在し、かつT+21(エグジット)も存在する範囲に限定
    for i in range(MIN_TECHNICAL_HISTORY_ROWS - 1, len(dates) - EVAL_HORIZON_TRADING_DAYS - 1):
        day = dates[i]
        cohort = find_applicable_cohort(cohorts, day)
        if cohort is None:
            continue

        window_df = price_df.iloc[: i + 1]
        day_str = day.strftime("%Y-%m-%d")

        hist = calc_indicators(window_df.copy())
        flags = signal_flags(check_signals(hist))
        technical = get_technical_state(hist)
        decision_v1 = decide_rule_based(cohort["grade"], cohort["total_score"], cohort["score_valuation"], flags, technical)

        technical_scores = compute_technical_v2(window_df.copy())
        market_regime = cached_market_regime(conn, day_str)
        sector_regime = cached_sector_regime(conn, cohort["sector"], day_str)
        decision_v2 = decide_composite(cohort["total_score"], technical_scores, market_regime, sector_regime, technical_config, decision_config)

        exit_close = float(price_df["Close"].iloc[i + EVAL_HORIZON_TRADING_DAYS])
        next_open = float(price_df["Open"].iloc[i + 1]) if pd.notna(price_df["Open"].iloc[i + 1]) else None
        next_close = float(price_df["Close"].iloc[i + 1])
        decision_close = float(price_df["Close"].iloc[i])

        for version, decision, prev_action in (("v1", decision_v1, prev_action_v1), ("v2", decision_v2, prev_action_v2)):
            if decision.action == "buy" and decision.action != prev_action:
                row = {"version": version, "ticker": ticker, "date": day_str}
                row["ret_decision_close"] = (exit_close - decision_close) / decision_close * 100
                row["ret_next_open"] = (exit_close - next_open) / next_open * 100 if next_open else None
                row["ret_next_close"] = (exit_close - next_close) / next_close * 100
                records.append(row)

        prev_action_v1, prev_action_v2 = decision_v1.action, decision_v2.action

    return records


def report(label: str, series: pd.Series) -> None:
    valid = series.dropna()
    if valid.empty:
        print(f"  {label}: サンプルなし")
        return
    win_rate = (valid > NEUTRAL_THRESHOLD_PCT).mean()
    loss_rate = (valid < -NEUTRAL_THRESHOLD_PCT).mean()
    print(
        f"  {label}: n={len(valid)}  平均={valid.mean():+.2f}%  中央値={valid.median():+.2f}%  "
        f"勝率(>+{NEUTRAL_THRESHOLD_PCT:.0f}%)={win_rate:.1%}  負け率(<-{NEUTRAL_THRESHOLD_PCT:.0f}%)={loss_rate:.1%}"
    )


def main():
    start = time.monotonic()
    conn = get_connection()
    technical_config, decision_config = load_v2_configs()
    scoring_config = load_config()

    panel = build_fundamental_panel(conn)
    panel = attach_value_momentum(conn, panel)
    panel = compute_score_per_cohort(panel, CATEGORY_METRICS_V2, CATEGORY_WEIGHTS_V2, "total_score_7axis")
    panel["total_score"] = panel["total_score_7axis"]
    panel["score_valuation"] = panel["score_value"]
    panel["grade"] = panel["total_score"].apply(lambda v: score_to_grade(v, scoring_config["grade_thresholds"]))
    panel = panel.dropna(subset=["total_score", "signal_date", "grade"])
    panel = build_cohort_windows(panel)

    tickers = panel["ticker"].unique()
    all_records = []
    for n, ticker in enumerate(tickers, start=1):
        cohorts = panel[panel["ticker"] == ticker]
        all_records.extend(simulate_ticker_entry_timing(conn, ticker, cohorts, technical_config, decision_config))
        if n % 500 == 0:
            print(f"[{n}/{len(tickers)}] 経過{time.monotonic()-start:.0f}秒  買いシグナル累計={len(all_records)}")

    conn.close()
    df = pd.DataFrame(all_records)
    print(f"\n全{len(tickers)}銘柄完了(経過{time.monotonic()-start:.0f}秒)、買いシグナル総数={len(df)}\n")

    for version in ("v1", "v2"):
        sub = df[df["version"] == version]
        print(f"=== {version} buyシグナル(n={len(sub)}) ===")
        report("T日終値エントリー(参考、実際は購入不可)", sub["ret_decision_close"])
        report("T+1日 寄り付き(Open)で成り行き買い", sub["ret_next_open"])
        report("T+1日 引け(Close、15時頃)で買い", sub["ret_next_close"])
        print()


if __name__ == "__main__":
    main()
