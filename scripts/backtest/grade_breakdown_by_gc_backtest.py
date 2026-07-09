"""GC(ゴールデンクロス)発生時のフォワードリターンを、グレード(S/A/B/C/D/E)別に分解する。

発見: グレードSはGC発生時のフォワードリターンが最も低い(12ヶ月中央値+5.22%)。
B/Cグレードの方が明確に良い(B:+11.77%、C:+10.47%)。これは「クオリティ・トラップ」
(既に市場に評価され尽くした銘柄は、GCが出ても再評価の伸びしろが少ない)と解釈できる。
この結果を受けてdecide_quality_timing.py(v3)はグレードB/Cを対象にしている
(alt_timing_signals_backtest.py・fundamental_plus_timing_backtest.pyで
さらにシグナル自体もGCから52週高値接近に更新された)。

実行: python scripts/backtest/grade_breakdown_by_gc_backtest.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection, get_backtest_connection
from common.technical import calc_indicators
from scoring.compute_scores import load_config, score_to_grade
from backtest.quality_score_v2_backtest import CATEGORY_METRICS_V2, CATEGORY_WEIGHTS_V2, compute_score_per_cohort
from backtest.decision_engine_extended_backtest import build_fundamental_panel_bt, attach_value_momentum_bt, load_bt_price_history
from backtest.decision_engine_comparison_backtest import build_cohort_windows, find_applicable_cohort
from backtest.technical_signal_event_study import detect_signals, HORIZONS

conn = get_connection()
bt_conn = get_backtest_connection()
scoring_config = load_config()

panel = build_fundamental_panel_bt(conn, bt_conn)
panel = attach_value_momentum_bt(bt_conn, panel)
panel = compute_score_per_cohort(panel, CATEGORY_METRICS_V2, CATEGORY_WEIGHTS_V2, 'total_score_7axis')
panel['total_score'] = panel['total_score_7axis']
panel['grade'] = panel['total_score'].apply(lambda v: score_to_grade(v, scoring_config['grade_thresholds']))
panel = panel.dropna(subset=['total_score', 'signal_date', 'grade'])
panel = build_cohort_windows(panel)
print('grade distribution in panel:', panel['grade'].value_counts().to_dict())

FOCUS = ['252営業日(12ヶ月)', '504営業日(24ヶ月)']
records = []
tickers = panel['ticker'].unique()
start = time.monotonic()
for n, ticker in enumerate(tickers, start=1):
    cohorts = panel[panel['ticker'] == ticker]
    price_df = load_bt_price_history(bt_conn, ticker)
    if len(price_df) < 100:
        continue
    ind = calc_indicators(price_df.copy())
    signals = detect_signals(ind)
    gc = signals['GC'].fillna(False)
    close = ind['Close']
    dates = price_df.index
    fwd = {h: (close.shift(-HORIZONS[h]) / close - 1) * 100 for h in FOCUS}
    idx = [i for i, d in enumerate(dates) if gc.iloc[i]]
    for i in idx:
        day = dates[i]
        cohort = find_applicable_cohort(cohorts, day)
        if cohort is None:
            continue
        rec = {"ticker": ticker, "date": day, "grade": cohort["grade"]}
        for h in FOCUS:
            v = fwd[h].iloc[i]
            if pd.notna(v):
                rec[h] = v
        records.append(rec)
    if n % 1000 == 0:
        print(f'[{n}/{len(tickers)}] {time.monotonic()-start:.0f}s')

df = pd.DataFrame(records)
print(f"\ntotal GC events with grade: {len(df)}, time={time.monotonic()-start:.0f}s\n")
print("=== グレード別・GC発生時のフォワードリターン ===")
for h in FOCUS:
    print(f"\n[{h}]")
    print(df.groupby('grade')[h].agg(['size','mean','median']).round(2).reindex(['S','A','B','C','D','E']).dropna(how='all'))

conn.close()
bt_conn.close()
