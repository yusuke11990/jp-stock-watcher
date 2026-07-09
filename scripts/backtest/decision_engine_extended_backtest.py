"""decision_engine_comparison_backtest.pyを、本番price_dailyの1年分ではなく
data/backtest.dbの複数年OHLCV(fetch_price_history_ohlcv_backtest.pyで取得)を使って
拡張したもの。目的は2つ:

1. 検証期間を1年から複数年に伸ばし、単一の相場局面に偏らないv1/v2の勝率・期待値を見る。
2. v2の各トレードにconfidence(確信度)・final_score(合成スコア)を記録し、
   「確信度が高いトレードだけに絞ったら勝率は上がるか」「buyのしきい値を引き上げたら
   勝率と件数のトレードオフはどうなるか」を、シミュレーションをやり直さずに
   後段の集計だけで検証できるようにする。

計算量が非常に大きくなる(5年分 × 全銘柄)ため、営業日を間引いて(既定3営業日に1回)
判定する。間引きは「毎日ツールを見るわけではないが、数日に一度は必ずチェックする」
運用に相当し、完全な毎日判定よりは粗いが、複数年の相場局面をカバーする方を優先した。

実行: python scripts/backtest/decision_engine_extended_backtest.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection, get_backtest_connection  # noqa: E402
from common.technical import calc_indicators, check_signals, get_technical_state  # noqa: E402
from common.technical_v2 import compute_technical_v2, calc_all_indicators, trend_alignment_score  # noqa: E402
from scoring.compute_scores import load_config, score_to_grade  # noqa: E402
from decisions.decide_rule import decide_rule_based, signal_flags  # noqa: E402
from decisions.decide_composite import decide_composite, load_configs as load_v2_configs  # noqa: E402
from backtest.quality_score_backtest import MIN_SECTOR_SAMPLE_SIZE  # noqa: E402
from backtest.quality_score_multi_period_backtest import (  # noqa: E402
    load_yearly_panel, build_period_panel, attach_sector,
)
from backtest.quality_score_v2_backtest import (  # noqa: E402
    CATEGORY_METRICS_V2, CATEGORY_WEIGHTS_V2,
    load_real_disclosure_dates, attach_signal_date, compute_score_per_cohort,
)
from backtest.decision_engine_comparison_backtest import (  # noqa: E402
    EVAL_HORIZON_TRADING_DAYS, NEUTRAL_THRESHOLD_PCT, MIN_TECHNICAL_HISTORY_ROWS,
    build_cohort_windows, find_applicable_cohort, classify_outcome,
)

DAY_SAMPLE_INTERVAL = 5  # 計算量削減のため、この日数おきにしか判定しない(1=毎日)
WINDOW_CAP = 400  # 指標計算に使う直近日数の上限(MA75等はこれで十分、全履歴を使うと日が進むほど遅くなる)
TOPIX_TICKER = "1306.T"

def load_bt_price_history(bt_conn, ticker: str) -> pd.DataFrame:
    query = "SELECT date, open, high, low, close, volume FROM price_history WHERE ticker = ? ORDER BY date ASC"
    df = pd.read_sql_query(query, bt_conn, params=(ticker,), parse_dates=["date"])
    if df.empty:
        return df
    return df.set_index("date").rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )


def precompute_topix_regime_lookup(topix_df: pd.DataFrame) -> dict[str, dict]:
    """TOPIXのMA5/25/75は純粋に過去方向のみを見る指標(先読みなし)なので、
    日ごとにウィンドウを切り詰めて再計算する必要はなく、全期間を一度だけ計算して
    日付ごとに引くだけでよい。1銘柄分の指標計算なのでこれ自体は軽い。
    """
    ind = calc_all_indicators(topix_df.copy())
    lookup: dict[str, dict] = {}
    for date, row in ind.iterrows():
        trend_score = trend_alignment_score(row.get("MA5"), row.get("MA25"), row.get("MA75"))
        if pd.isna(trend_score):
            trend_score = 0.0
        lookup[date.strftime("%Y-%m-%d")] = {
            "topix_trend_score": trend_score,
            "market_regime_multiplier": 1.0 + 0.3 * trend_score,
        }
    return lookup


def precompute_sector_regime_lookup(conn, bt_conn, lookback_days: int = 5) -> dict[tuple[str, str], dict]:
    """(セクター, 日付)ごとの地合いスコアを、銘柄横断でベクトル化して一括計算する。

    銘柄ごと・日ごとに個別SQLクエリを投げるとテーブル全体スキャンが日数×セクター数分
    発生し致命的に遅い(実測15秒/銘柄、全銘柄で10時間超)。事前に全価格・全セクターの
    5日騰落率を一括計算し、辞書引きだけで済むようにする。
    """
    tickers_df = pd.read_sql_query("SELECT ticker, sector FROM tickers WHERE is_active = 1", conn)
    sector_counts = tickers_df["sector"].value_counts()
    valid_sectors = set(sector_counts[sector_counts >= MIN_SECTOR_SAMPLE_SIZE].index)

    prices = pd.read_sql_query("SELECT ticker, date, close FROM price_history", bt_conn, parse_dates=["date"])
    prices = prices.merge(tickers_df, on="ticker", how="inner")
    prices = prices[prices["sector"].isin(valid_sectors)].sort_values(["ticker", "date"])
    prices["return_5d"] = prices.groupby("ticker")["close"].transform(lambda s: s / s.shift(lookback_days) - 1)

    daily = prices.dropna(subset=["return_5d"]).groupby(["sector", "date"])["return_5d"].median().reset_index()
    daily["sector_regime_score"] = daily["return_5d"].apply(lambda v: max(-1.0, min(1.0, v * 10)))

    lookup: dict[tuple[str, str], dict] = {}
    sample_sizes = tickers_df[tickers_df["sector"].isin(valid_sectors)]["sector"].value_counts().to_dict()
    for row in daily.itertuples():
        key = (row.sector, row.date.strftime("%Y-%m-%d"))
        lookup[key] = {
            "sector_return_median": float(row.return_5d),
            "sector_regime_score": float(row.sector_regime_score),
            "sample_size": sample_sizes.get(row.sector, 0),
        }
    return lookup, sample_sizes


def bt_market_regime_lookup(topix_lookup: dict, day_str: str) -> dict:
    return topix_lookup.get(day_str, {"topix_trend_score": 0.0, "market_regime_multiplier": 1.0})


def bt_sector_regime_lookup(sector_lookup: dict, sample_sizes: dict, sector: str, day_str: str) -> dict:
    result = sector_lookup.get((sector, day_str))
    if result is not None:
        return result
    return {"sector_return_median": None, "sector_regime_score": 0.0, "sample_size": sample_sizes.get(sector, 0)}


def attach_value_momentum_bt(bt_conn, panel: pd.DataFrame) -> pd.DataFrame:
    """backtest.dbの長期close履歴からPER/PBR・12-1モメンタムを計算する
    (attach_value_momentumのprice_daily版と同じロジック、データソースだけ長期の方に変更)
    """
    prices = pd.read_sql_query("SELECT ticker, date, close FROM price_history", bt_conn, parse_dates=["date"])
    prices = prices.sort_values("date")

    def asof_price(dates_df: pd.DataFrame, date_col: str, direction: str, tolerance_days: int = 10) -> pd.Series:
        sub = dates_df[["ticker", date_col]].sort_values(date_col)
        merged = pd.merge_asof(
            sub, prices, left_on=date_col, right_on="date", by="ticker", direction=direction,
            tolerance=pd.Timedelta(days=tolerance_days),
        )
        return merged.set_index(sub.index)["close"]

    panel = panel.copy()
    panel["price_entry"] = asof_price(panel, "signal_date", "backward")

    skip_date = (panel["signal_date"] - pd.Timedelta(days=30)).rename("skip_date")
    tmp_skip = pd.concat([panel["ticker"], skip_date], axis=1)
    price_skip = asof_price(tmp_skip, "skip_date", "backward", tolerance_days=21)

    lookback_date = (panel["signal_date"] - pd.Timedelta(days=365)).rename("lookback_date")
    tmp_lb = pd.concat([panel["ticker"], lookback_date], axis=1)
    price_12m_ago = asof_price(tmp_lb, "lookback_date", "backward", tolerance_days=21)

    panel["momentum_12_1"] = price_skip / price_12m_ago - 1

    bvps = panel["eps"] / panel["roe"]
    panel["per"] = panel["price_entry"] / panel["eps"]
    panel["pbr"] = panel["price_entry"] / bvps
    for col in ("per", "pbr"):
        panel.loc[(panel[col] <= 0) | (panel[col] > 500), col] = None
    return panel


def build_fundamental_panel_bt(conn, bt_conn) -> pd.DataFrame:
    yearly = load_yearly_panel(conn)
    panel = build_period_panel(yearly)
    panel = attach_sector(conn, panel)
    disclosure = load_real_disclosure_dates(conn)
    panel = attach_signal_date(panel, disclosure)

    base_metrics = {k: v for k, v in CATEGORY_METRICS_V2.items() if k not in ("value", "momentum", "earnings_quality")}
    base_weights = {k: 1.0 / len(base_metrics) for k in base_metrics}
    panel = compute_score_per_cohort(panel, base_metrics, base_weights, "total_score_5axis")
    return panel


def simulate_ticker_bt(ticker: str, price_df: pd.DataFrame, cohorts: pd.DataFrame,
                       topix_lookup: dict, sector_lookup: dict, sector_sample_sizes: dict,
                       technical_config: dict, decision_config: dict) -> tuple[list, list]:
    trades_v1, trades_v2 = [], []
    if len(price_df) < MIN_TECHNICAL_HISTORY_ROWS + EVAL_HORIZON_TRADING_DAYS:
        return trades_v1, trades_v2

    dates = price_df.index
    prev_action_v1, prev_action_v2 = "hold", "hold"

    idx_range = range(MIN_TECHNICAL_HISTORY_ROWS - 1, len(dates) - EVAL_HORIZON_TRADING_DAYS, DAY_SAMPLE_INTERVAL)
    for i in idx_range:
        day = dates[i]
        cohort = find_applicable_cohort(cohorts, day)
        if cohort is None:
            continue

        # MA75/ADX/RSI/MACD/BB/divergence等はいずれも高々数十〜100日程度の遡りしか
        # 使わないため、全履歴(iまでの累積、日が進むほど増え続けO(n^2)化する)ではなく
        # 直近WINDOW_CAP日だけの固定長ウィンドウで計算しても結果は変わらない
        # (OBVは累積指標だが、divergenceは相対的な傾きしか見ないため基準点のズレは影響しない)
        window_df = price_df.iloc[max(0, i + 1 - WINDOW_CAP) : i + 1]
        day_str = day.strftime("%Y-%m-%d")

        hist = calc_indicators(window_df.copy())
        flags = signal_flags(check_signals(hist))
        technical = get_technical_state(hist)
        decision_v1 = decide_rule_based(cohort["grade"], cohort["total_score"], cohort["score_valuation"], flags, technical)
        entry_price_v1 = technical.get("current_price")

        technical_scores = compute_technical_v2(window_df.copy())
        market_regime = bt_market_regime_lookup(topix_lookup, day_str)
        sector_regime = bt_sector_regime_lookup(sector_lookup, sector_sample_sizes, cohort["sector"], day_str)
        decision_v2 = decide_composite(cohort["total_score"], technical_scores, market_regime, sector_regime, technical_config, decision_config)
        entry_price_v2 = technical_scores.get("close")

        exit_price = float(price_df["Close"].iloc[i + EVAL_HORIZON_TRADING_DAYS])

        if decision_v1.action in ("buy", "sell") and decision_v1.action != prev_action_v1 and entry_price_v1:
            ret = (exit_price - entry_price_v1) / entry_price_v1 * 100
            is_correct, label = classify_outcome(decision_v1.action, ret)
            trades_v1.append({"ticker": ticker, "date": day_str, "action": decision_v1.action, "return_pct": ret, "is_correct": is_correct, "label": label})
        prev_action_v1 = decision_v1.action

        if decision_v2.action in ("buy", "sell") and decision_v2.action != prev_action_v2 and entry_price_v2:
            ret = (exit_price - entry_price_v2) / entry_price_v2 * 100
            is_correct, label = classify_outcome(decision_v2.action, ret)
            trades_v2.append({
                "ticker": ticker, "date": day_str, "action": decision_v2.action, "return_pct": ret,
                "is_correct": is_correct, "label": label,
                "confidence": decision_v2.confidence, "final_score": decision_v2.final_score,
                "grade": cohort["grade"], "total_score": cohort["total_score"],
            })
        prev_action_v2 = decision_v2.action

    return trades_v1, trades_v2


def report(label: str, trades: list) -> None:
    df = pd.DataFrame(trades)
    print(f"=== {label} ===")
    if df.empty:
        print("  トレードなし")
        return
    print(f"  トレード数: {len(df)}件  銘柄数: {df['ticker'].nunique()}件")
    for action in ("buy", "sell"):
        sub = df[df["action"] == action]
        if sub.empty:
            print(f"  {action}: 0件")
            continue
        print(f"  {action}: n={len(sub)}  勝率={sub['is_correct'].mean():.1%}  平均={sub['return_pct'].mean():+.2f}%  中央値={sub['return_pct'].median():+.2f}%")
    directional = df.apply(lambda r: r["return_pct"] if r["action"] == "buy" else -r["return_pct"], axis=1)
    print(f"  方向調整済み平均リターン: {directional.mean():+.2f}%  (n={len(df)})")
    print()


def main():
    start = time.monotonic()
    conn = get_connection()
    bt_conn = get_backtest_connection()
    technical_config, decision_config = load_v2_configs()
    scoring_config = load_config()

    bounds = bt_conn.execute("SELECT MIN(date), MAX(date) FROM price_history WHERE open IS NOT NULL").fetchone()
    print(f"backtest.db OHLCV利用可能期間: {bounds[0]} 〜 {bounds[1]}")
    print(f"営業日サンプリング間隔: {DAY_SAMPLE_INTERVAL}日に1回")

    panel = build_fundamental_panel_bt(conn, bt_conn)
    panel = attach_value_momentum_bt(bt_conn, panel)
    panel = compute_score_per_cohort(panel, CATEGORY_METRICS_V2, CATEGORY_WEIGHTS_V2, "total_score_7axis")
    panel["total_score"] = panel["total_score_7axis"]
    panel["score_valuation"] = panel["score_value"]
    panel["grade"] = panel["total_score"].apply(lambda v: score_to_grade(v, scoring_config["grade_thresholds"]))
    panel = panel.dropna(subset=["total_score", "signal_date", "grade"])
    panel = build_cohort_windows(panel)
    print(f"ファンダ・開示日が揃ったコホート数: {len(panel)}件、対象銘柄数: {panel['ticker'].nunique()}件")

    topix_df = load_bt_price_history(bt_conn, TOPIX_TICKER)

    print("TOPIX地合いスコアを事前計算中...")
    topix_lookup = precompute_topix_regime_lookup(topix_df)
    print(f"  {len(topix_lookup)}日分")

    print("業種地合いスコアを事前計算中(全銘柄×全日をベクトル化)...")
    t0 = time.monotonic()
    sector_lookup, sector_sample_sizes = precompute_sector_regime_lookup(conn, bt_conn)
    print(f"  {len(sector_lookup)}(業種×日)件、{time.monotonic()-t0:.0f}秒")

    all_trades_v1, all_trades_v2 = [], []
    tickers = panel["ticker"].unique()
    for n, ticker in enumerate(tickers, start=1):
        cohorts = panel[panel["ticker"] == ticker]
        price_df = load_bt_price_history(bt_conn, ticker)
        t1, t2 = simulate_ticker_bt(
            ticker, price_df, cohorts, topix_lookup, sector_lookup, sector_sample_sizes,
            technical_config, decision_config,
        )
        all_trades_v1.extend(t1)
        all_trades_v2.extend(t2)
        if n % 200 == 0:
            elapsed = time.monotonic() - start
            print(f"[{n}/{len(tickers)}] 経過{elapsed:.0f}秒  v1累計={len(all_trades_v1)}  v2累計={len(all_trades_v2)}")

    print(f"\n全{len(tickers)}銘柄完了(経過{time.monotonic()-start:.0f}秒)\n")
    report("v1(decide_rule.py、複数年・間引き判定)", all_trades_v1)
    report("v2(decide_composite.py、複数年・間引き判定)", all_trades_v2)

    # v2トレードをCSVに保存し、confidence/final_scoreによる絞り込み効果を後段で分析できるようにする
    v2_df = pd.DataFrame(all_trades_v2)
    out_path = Path(__file__).resolve().parent / "v2_extended_trades.csv"
    v2_df.to_csv(out_path, index=False)
    print(f"v2の全トレードを{out_path}に保存しました({len(v2_df)}件)")

    conn.close()
    bt_conn.close()


if __name__ == "__main__":
    main()
