"""GC(ゴールデンクロス)に代わる長期保有向けタイミングシグナルを比較検証する。

grade_breakdown_by_gc_backtest.pyでグレードB/Cが優位と判明した後、B/C×割安性上位50%
という条件下で、GCと4つの代替シグナル(株価のMA75上抜け、MA75の傾き反転、
52週高値の95%圏内への接近、GC+MA75上昇の強化版)を12ヶ月・24ヶ月保有で比較した。

結果: 52週高値接近(NEAR_52W_HIGH)が24ヶ月保有で圧倒的に優位
(+52.81% vs GCの+44.47%、サンプル数もGCの3倍以上)。行動ファイナンスで知られる
「52週高値モメンタム」(George & Hwang, 2004)と整合する。この結果を受けて
common/technical.pyにcheck_near_52_week_high()を追加し、decide_quality_timing.py(v3)の
タイミングシグナルをGCからこれに置き換えた。年別頑健性は別途
near_52w_high_robustness_backtest.pyで確認済み(2022-2024年の全年で優位)。

実行: python scripts/backtest/alt_timing_signals_backtest.py
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
from backtest.technical_signal_event_study import HORIZONS

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

FOCUS = ['252営業日(12ヶ月)', '504営業日(24ヶ月)']

def detect_alt_signals(df):
    prev = df.shift(1)
    sig = pd.DataFrame(index=df.index)
    # 現行GC: MA25がMA75を上抜けた瞬間
    sig['GC'] = (prev['MA25'] <= prev['MA75']) & (df['MA25'] > df['MA75'])
    # 株価自体がMA75を上抜けた瞬間(価格とトレンドの関係)
    sig['PRICE_ABOVE_MA75'] = (prev['Close'] <= prev['MA75']) & (df['Close'] > df['MA75'])
    # MA75自体の傾き(20日前と比べて上昇)がプラスに転じた瞬間(=長期トレンドが上向きになった)
    ma75_slope = df['MA75'] - df['MA75'].shift(20)
    ma75_slope_prev = prev['MA75'] - prev['MA75'].shift(20)
    sig['MA75_RISING'] = (ma75_slope_prev <= 0) & (ma75_slope > 0)
    # 52週高値の95%圏内に入った瞬間(モメンタム/強さの古典的指標)
    rolling_high = df['Close'].rolling(252, min_periods=100).max()
    near_high = df['Close'] >= rolling_high * 0.95
    sig['NEAR_52W_HIGH'] = near_high & ~near_high.shift(1).fillna(False)
    # GC + MA75自体も上向き(トレンドの成熟度を要求する強化版GC)
    sig['GC_STRONG'] = sig['GC'] & (ma75_slope > 0)
    return sig

records = {sig: {h: [] for h in FOCUS} for sig in ['GC','PRICE_ABOVE_MA75','MA75_RISING','NEAR_52W_HIGH','GC_STRONG']}
tickers = panel['ticker'].unique()
start = time.monotonic()
for n, ticker in enumerate(tickers, start=1):
    cohorts = panel[panel['ticker'] == ticker]
    price_df = load_bt_price_history(bt_conn, ticker)
    if len(price_df) < 100:
        continue
    ind = calc_indicators(price_df.copy())
    sig = detect_alt_signals(ind)
    close = ind['Close']
    dates = price_df.index
    fwd = {h: (close.shift(-HORIZONS[h]) / close - 1) * 100 for h in FOCUS}

    # 銘柄ごとの割安性中央値判定用に、コホートのscore_valueを都度引く必要があるので、
    # シグナル発生日だけループ(件数は少ないので軽い)
    for sig_name in records:
        idx = [i for i, d in enumerate(dates) if sig[sig_name].fillna(False).iloc[i]]
        for i in idx:
            day = dates[i]
            cohort = find_applicable_cohort(cohorts, day)
            if cohort is None or cohort['grade'] not in ('B', 'C'):
                continue
            for h in FOCUS:
                v = fwd[h].iloc[i]
                if pd.notna(v):
                    records[sig_name][h].append({"return": v, "score_value": cohort["score_value"], "year": day.year})
    if n % 1000 == 0:
        print(f'[{n}/{len(tickers)}] {time.monotonic()-start:.0f}s')

print(f"\n完了(経過{time.monotonic()-start:.0f}秒)\n")
print("=== シグナル別比較(グレードB/C限定、割安性上位50%で絞り込み) ===")
for sig_name in records:
    print(f"\n--- {sig_name} ---")
    for h in FOCUS:
        df_h = pd.DataFrame(records[sig_name][h])
        if df_h.empty:
            print(f"  {h}: サンプルなし")
            continue
        median_val = df_h['score_value'].median()
        high_val = df_h[df_h['score_value'] >= median_val]
        print(f"  {h}: 全体n={len(df_h)} 平均={df_h['return'].mean():+.2f}%  |  割安性上位50%: n={len(high_val)} 平均={high_val['return'].mean():+.2f}% 中央値={high_val['return'].median():+.2f}%")

conn.close()
bt_conn.close()
