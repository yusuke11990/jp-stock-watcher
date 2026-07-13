"""グレードS/Aが「クオリティ・トラップ」だという当初の結論(grade_breakdown_by_gc_backtest.py)
を、本番シグナル(52週高値接近、GCではない)で検証し直す。

背景: 当初の結論はGC発生時のグレード別リターンに基づいていたが、パネル全体で
グレードSの観測数はわずか10件(GCイベントは16件)しかなく、統計的に無意味な
サンプルサイズだった。さらにユーザーから「市場に評価され尽くしているなら
PER/PBRは高く出て割安性スコアは低くなるはずで、割安性スコアが高いグレードSが
劣後するのは矛盾するのでは」という指摘があり、再検証した。

結果: 実際にはグレードが高いほど割安性スコア(業種内パーセンタイル)の分布も
高め(S平均72.6 > A平均58.4 > B平均54.8 > C平均49.3)であり、「質が高く、かつ
統計的にも割安」という組み合わせ自体がグレードSほど出やすい。本番シグナル
(52週高値の95%圏内への接近、decide_quality_timing.pyと同じ)×割安性上位50%の
条件で再検証すると、グレードが高いほどリターンも高い(12ヶ月平均:
S+61.0%[n=27] > A+33.6%[n=1213] > B+25.9%[n=8798] > C+23.2%[n=11037])。
24ヶ月保有でも同順。この結果を受けてdecide_quality_timing.py(v3)・
decide_quality_timing_v4.py(v4)のTARGET_GRADESを{B,C}から{S,A,B,C}に拡張した。

実行: python scripts/backtest/grade_valuation_near52w_backtest.py
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

HORIZONS_TRADING_DAYS = {"252営業日(12ヶ月)": 252, "504営業日(24ヶ月)": 504}
TARGET_GRADES_ALL = ("S", "A", "B", "C")


def build_events(panel: pd.DataFrame, bt_conn) -> pd.DataFrame:
    records = []
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
            if cohort is None or cohort["grade"] not in TARGET_GRADES_ALL:
                continue
            entry_price = close.iloc[i]
            if pd.isna(entry_price) or entry_price <= 0:
                continue
            rec = {"ticker": ticker, "date": day, "grade": cohort["grade"], "score_value": cohort["score_value"]}
            for h_label, h_days in HORIZONS_TRADING_DAYS.items():
                exit_pos = i + h_days
                if exit_pos >= len(dates):
                    continue
                exit_price = close.iloc[exit_pos]
                if pd.notna(exit_price) and exit_price > 0:
                    rec[h_label] = (exit_price / entry_price - 1) * 100
            records.append(rec)
        if n % 1000 == 0:
            print(f"[{n}/{len(tickers)}] {time.monotonic()-start:.0f}s")
    return pd.DataFrame(records)


def report(label: str, subset: pd.DataFrame, h: str) -> None:
    s = subset.dropna(subset=[h])
    if len(s) < 5:
        print(f"  {label}: n={len(s)} (サンプル不足)")
        return
    print(f"  {label}: n={len(s)}  平均={s[h].mean():+.2f}%  中央値={s[h].median():+.2f}%")


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

    df = build_events(panel, bt_conn)
    print(f"\n52週高値接近イベント総数(grade S/A/B/C): {len(df)}")
    print("grade別サンプル数:", df["grade"].value_counts().to_dict())

    print("\n=== グレード別・割安性上位50%(グループ内基準)条件下のフォワードリターン ===")
    for h in HORIZONS_TRADING_DAYS:
        print(f"\n--- {h} ---")
        for g in TARGET_GRADES_ALL:
            sub = df[df["grade"] == g]
            if sub.empty:
                continue
            med = sub["score_value"].median()
            report(f"grade={g} 全体", sub, h)
            report(f"grade={g} + 割安性上位50%", sub[sub["score_value"] >= med], h)

        bc = df[df["grade"].isin(["B", "C"])]
        sabc = df[df["grade"].isin(list(TARGET_GRADES_ALL))]
        if not bc.empty and not sabc.empty:
            bc_med = bc["score_value"].median()
            sabc_med = sabc["score_value"].median()
            report("現行v3(B/C) + 割安性上位50%", bc[bc["score_value"] >= bc_med], h)
            report("拡張案(S/A/B/C) + 割安性上位50%", sabc[sabc["score_value"] >= sabc_med], h)

    conn.close()
    bt_conn.close()


if __name__ == "__main__":
    main()
