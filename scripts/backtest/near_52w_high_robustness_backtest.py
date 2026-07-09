import sys, time
sys.path.insert(0, 'scripts')
import pandas as pd
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

h = '504営業日(24ヶ月)'
H = HORIZONS[h]
records = []
tickers = panel['ticker'].unique()
start = time.monotonic()
for n, ticker in enumerate(tickers, start=1):
    cohorts = panel[panel['ticker'] == ticker]
    price_df = load_bt_price_history(bt_conn, ticker)
    if len(price_df) < 100:
        continue
    ind = calc_indicators(price_df.copy())
    close = ind['Close']
    rolling_high = close.rolling(252, min_periods=100).max()
    near_high = close >= rolling_high * 0.95
    sig = near_high & ~near_high.shift(1).fillna(False).infer_objects(copy=False)
    dates = price_df.index
    fwd = (close.shift(-H) / close - 1) * 100
    idx = [i for i, d in enumerate(dates) if sig.iloc[i]]
    for i in idx:
        day = dates[i]
        cohort = find_applicable_cohort(cohorts, day)
        if cohort is None or cohort['grade'] not in ('B', 'C'):
            continue
        v = fwd.iloc[i]
        if pd.notna(v):
            records.append({"return": v, "score_value": cohort["score_value"], "year": day.year})
    if n % 1000 == 0:
        print(f'[{n}/{len(tickers)}] {time.monotonic()-start:.0f}s')

df = pd.DataFrame(records)
median_val = df['score_value'].median()
df['high_value'] = df['score_value'] >= median_val
print(f"\n完了(経過{time.monotonic()-start:.0f}秒), n={len(df)}\n")
print("=== 年別頑健性: NEAR_52W_HIGH + B/C + 割安性上位50%(24ヶ月) ===")
sub = df[df['high_value']]
print(sub.groupby('year')['return'].agg(['size','mean']).round(2))

conn.close()
bt_conn.close()
