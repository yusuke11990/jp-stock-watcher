import sys, time
sys.path.insert(0, 'scripts')
import pandas as pd
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
panel['score_value'] = panel['score_value']  # valuation軸の単独percentile(既に計算済み)
panel = panel.dropna(subset=['total_score', 'signal_date', 'grade'])
panel = build_cohort_windows(panel)

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
        rec = {"ticker": ticker, "date": day, "grade": cohort["grade"], "score_value": cohort["score_value"]}
        for h in FOCUS:
            v = fwd[h].iloc[i]
            if pd.notna(v):
                rec[h] = v
        records.append(rec)
    if n % 1000 == 0:
        print(f'[{n}/{len(tickers)}] {time.monotonic()-start:.0f}s')

df = pd.DataFrame(records)
df['year'] = pd.to_datetime(df['date']).dt.year
print(f"\ntotal GC events: {len(df)}, time={time.monotonic()-start:.0f}s\n")

def report(label, subset, h):
    s = subset.dropna(subset=[h])
    print(f"  {label}: n={len(s)}  平均={s[h].mean():+.2f}%  中央値={s[h].median():+.2f}%")

print("=== 全体比較(12ヶ月・24ヶ月) ===")
for h in FOCUS:
    print(f"\n[{h}]")
    report("S/A/B + GC (現行v3)", df[df['grade'].isin(['S','A','B'])], h)
    report("B/C + GC", df[df['grade'].isin(['B','C'])], h)
    report("C単体 + GC", df[df['grade']=='C'], h)
    # B/C+GCのうち、さらにscore_value上位50%に絞る
    bc = df[df['grade'].isin(['B','C'])]
    median_val = bc['score_value'].median()
    report("B/C + GC + 割安性上位50%", bc[bc['score_value'] >= median_val], h)
    report("B/C + GC + 割安性下位50%", bc[bc['score_value'] < median_val], h)

print("\n=== 年別頑健性チェック(B/C + GC vs S/A/B + GC、12ヶ月) ===")
h = FOCUS[0]
for label, subset in [("S/A/B+GC", df[df['grade'].isin(['S','A','B'])]), ("B/C+GC", df[df['grade'].isin(['B','C'])])]:
    print(f"\n--- {label} ---")
    g = subset.dropna(subset=[h]).groupby('year')[h].agg(['size','mean']).round(2)
    print(g)

conn.close()
bt_conn.close()
