"""v1(decide_rule.py)とv2(decide_composite.py)の"判断ロジックそのもの"を、
本番と同じ「毎日判定する」運用に合わせて再現し、勝率・期待値を比較するバックテスト。

最初のバージョン(コホートの開示日1点だけを判定日にする設計)では、GC/RSI/MACD/BB等の
イベント型シグナルがその1日にちょうど発生する確率が低く、buyが極端に少なくなる
(v1=8件, v2=24件)という問題があった。本版はファンダスコアの有効期間
(ある開示日から次の開示日まで)の間、price_dailyの全営業日で毎日判定を再現し、
実運用(毎日ツールを見て、シグナルが出たら入る)によりリアルに近づける。

同じシグナルが何日も連続してbuy/sellのままだと同じトレードを何度も数えてしまうため、
「直前の判断がbuy/sellではなかった日」に限ってのみ新規トレードとして記録する
(=シグナルが継続している間は追加エントリーしない)。

制約:
- ファンダメンタルズのスコアはquality_score_v2_backtest.pyと同じ7軸(コホート・実開示日
  ベースの点在時点percentile)を使う。scoring_config.yaml本番の全25指標を過去分すべて
  再現するのは別データ(過去時点の時価総額等)が必要になり別の大工事になるため、
  既に検証済みのこの7軸を「総合スコア」の代理として使う。
- テクニカルシグナルは本番と同じOHLCV(price_daily)が必要。price_dailyは直近1年分しか
  無いため、実際に日次判定を再現できるのはこの1年に収まる範囲に限られる。

実行: python scripts/backtest/decision_engine_comparison_backtest.py
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
from common.regime import market_regime_score, sector_regime_score  # noqa: E402
from scoring.compute_scores import load_config, score_to_grade  # noqa: E402
from decisions.decide_rule import decide_rule_based, signal_flags  # noqa: E402
from decisions.decide_composite import decide_composite, load_configs as load_v2_configs  # noqa: E402
from backtest.quality_score_multi_period_backtest import (  # noqa: E402
    load_yearly_panel, build_period_panel, attach_sector,
)
from backtest.quality_score_v2_backtest import (  # noqa: E402
    CATEGORY_METRICS_V2, CATEGORY_WEIGHTS_V2,
    load_real_disclosure_dates, attach_signal_date, compute_score_per_cohort,
)

EVAL_HORIZON_TRADING_DAYS = 21  # evaluate_decisions.pyと同じ評価期間
NEUTRAL_THRESHOLD_PCT = 2.0  # evaluate_decisions.pyと同じ中立帯
MIN_TECHNICAL_HISTORY_ROWS = 75  # decide_rule.py/decide_composite.pyと同じ閾値

_market_regime_cache: dict[str, dict] = {}
_sector_regime_cache: dict[tuple[str, str], dict] = {}


def cached_market_regime(conn, date_str: str) -> dict:
    if date_str not in _market_regime_cache:
        _market_regime_cache[date_str] = market_regime_score(conn, date_str)
    return _market_regime_cache[date_str]


def cached_sector_regime(conn, sector: str, date_str: str) -> dict:
    key = (sector, date_str)
    if key not in _sector_regime_cache:
        _sector_regime_cache[key] = sector_regime_score(conn, sector, date_str)
    return _sector_regime_cache[key]


def classify_outcome(action: str, return_pct: float) -> tuple[bool, str]:
    """evaluate_decisions.pyのevaluate_decision()と同一のロジック"""
    if action == "buy":
        if return_pct > NEUTRAL_THRESHOLD_PCT:
            return True, "correct"
        elif return_pct < -NEUTRAL_THRESHOLD_PCT:
            return False, "incorrect"
        return False, "neutral"
    elif action == "sell":
        if return_pct < -NEUTRAL_THRESHOLD_PCT:
            return True, "correct"
        elif return_pct > NEUTRAL_THRESHOLD_PCT:
            return False, "incorrect"
        return False, "neutral"
    else:  # hold
        if abs(return_pct) <= NEUTRAL_THRESHOLD_PCT:
            return True, "correct"
        return False, "incorrect"


def build_fundamental_panel(conn) -> pd.DataFrame:
    """quality_score_v2_backtest.pyと同じ7軸(点在時点percentile)でtotal_score/grade/
    score_valuationの代理値を作る。
    """
    yearly = load_yearly_panel(conn)
    panel = build_period_panel(yearly)
    panel = attach_sector(conn, panel)
    disclosure = load_real_disclosure_dates(conn)
    panel = attach_signal_date(panel, disclosure)

    base_metrics = {k: v for k, v in CATEGORY_METRICS_V2.items() if k not in ("value", "momentum", "earnings_quality")}
    base_weights = {k: 1.0 / len(base_metrics) for k in base_metrics}
    panel = compute_score_per_cohort(panel, base_metrics, base_weights, "total_score_5axis")
    return panel


def attach_value_momentum(conn, panel: pd.DataFrame) -> pd.DataFrame:
    """price_dailyから決定日時点の終値を取り、PER/PBR・12-1モメンタムを付与する"""
    prices = pd.read_sql_query("SELECT ticker, date, close FROM price_daily", conn, parse_dates=["date"])
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


def build_cohort_windows(panel: pd.DataFrame) -> pd.DataFrame:
    """各銘柄のコホートに、次のコホート開示日までの有効期間を付与する。
    最後のコホートはvalid_until=NaT(=価格データの範囲内ではずっと有効)とする。
    """
    panel = panel.sort_values(["ticker", "signal_date"]).copy()
    panel["valid_until"] = panel.groupby("ticker")["signal_date"].shift(-1)
    return panel


def find_applicable_cohort(cohorts: pd.DataFrame, day: pd.Timestamp):
    applicable = cohorts[cohorts["signal_date"] <= day]
    if applicable.empty:
        return None
    row = applicable.iloc[-1]
    if pd.notna(row["valid_until"]) and day >= row["valid_until"]:
        return None
    return row


def load_full_price_daily(conn, ticker: str) -> pd.DataFrame:
    query = "SELECT date, open, high, low, close, volume FROM price_daily WHERE ticker = ? ORDER BY date ASC"
    df = pd.read_sql_query(query, conn, params=(ticker,), parse_dates=["date"])
    if df.empty:
        return df
    return df.set_index("date").rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )


def simulate_ticker(conn, ticker: str, cohorts: pd.DataFrame, technical_config: dict, decision_config: dict) -> tuple[list, list]:
    price_df = load_full_price_daily(conn, ticker)
    trades_v1, trades_v2 = [], []
    if len(price_df) < MIN_TECHNICAL_HISTORY_ROWS + EVAL_HORIZON_TRADING_DAYS:
        return trades_v1, trades_v2

    dates = price_df.index
    prev_action_v1, prev_action_v2 = "hold", "hold"

    for i in range(MIN_TECHNICAL_HISTORY_ROWS - 1, len(dates) - EVAL_HORIZON_TRADING_DAYS):
        day = dates[i]
        cohort = find_applicable_cohort(cohorts, day)
        if cohort is None:
            continue

        window_df = price_df.iloc[: i + 1]
        day_str = day.strftime("%Y-%m-%d")

        # --- v1 ---
        hist = calc_indicators(window_df.copy())
        signals = check_signals(hist)
        flags = signal_flags(signals)
        technical = get_technical_state(hist)
        decision_v1 = decide_rule_based(cohort["grade"], cohort["total_score"], cohort["score_valuation"], flags, technical)
        entry_price_v1 = technical.get("current_price")

        # --- v2 ---
        technical_scores = compute_technical_v2(window_df.copy())
        market_regime = cached_market_regime(conn, day_str)
        sector_regime = cached_sector_regime(conn, cohort["sector"], day_str)
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
            trades_v2.append({"ticker": ticker, "date": day_str, "action": decision_v2.action, "return_pct": ret, "is_correct": is_correct, "label": label})
        prev_action_v2 = decision_v2.action

    return trades_v1, trades_v2


def report(label: str, trades: list) -> None:
    df = pd.DataFrame(trades)
    print(f"=== {label} ===")
    if df.empty:
        print("  トレードなし")
        return
    print(f"  トレード数(buy+sell、シグナル継続中の重複は除く): {len(df)}件")
    for action in ("buy", "sell"):
        sub = df[df["action"] == action]
        if sub.empty:
            print(f"  {action}: 0件")
            continue
        win_rate = sub["is_correct"].mean()
        mean_return = sub["return_pct"].mean()
        print(f"  {action}: n={len(sub)}  勝率={win_rate:.1%}  平均リターン={mean_return:+.2f}%  中央値={sub['return_pct'].median():+.2f}%")
    directional = df.apply(lambda r: r["return_pct"] if r["action"] == "buy" else -r["return_pct"], axis=1)
    print(f"  方向調整済み平均リターン(期待値の目安、buy+sell全体): {directional.mean():+.2f}%  (n={len(df)})")
    print(f"  銘柄数(ユニーク): {df['ticker'].nunique()}件")
    print()


def main():
    start = time.monotonic()
    conn = get_connection()
    technical_config, decision_config = load_v2_configs()
    scoring_config = load_config()

    price_bounds = conn.execute("SELECT MIN(date), MAX(date) FROM price_daily").fetchone()
    print(f"price_daily利用可能期間: {price_bounds[0]} 〜 {price_bounds[1]}")

    panel = build_fundamental_panel(conn)
    panel = attach_value_momentum(conn, panel)
    panel = compute_score_per_cohort(panel, CATEGORY_METRICS_V2, CATEGORY_WEIGHTS_V2, "total_score_7axis")
    panel["total_score"] = panel["total_score_7axis"]
    panel["score_valuation"] = panel["score_value"]
    panel["grade"] = panel["total_score"].apply(lambda v: score_to_grade(v, scoring_config["grade_thresholds"]))

    panel = panel.dropna(subset=["total_score", "signal_date", "grade"])
    panel = build_cohort_windows(panel)
    print(f"ファンダ・開示日が揃ったコホート数: {len(panel)}件、対象銘柄数: {panel['ticker'].nunique()}件")

    all_trades_v1, all_trades_v2 = [], []
    tickers = panel["ticker"].unique()
    for n, ticker in enumerate(tickers, start=1):
        cohorts = panel[panel["ticker"] == ticker]
        trades_v1, trades_v2 = simulate_ticker(conn, ticker, cohorts, technical_config, decision_config)
        all_trades_v1.extend(trades_v1)
        all_trades_v2.extend(trades_v2)
        if n % 200 == 0:
            elapsed = time.monotonic() - start
            print(f"[{n}/{len(tickers)}] 経過{elapsed:.0f}秒  v1トレード累計={len(all_trades_v1)}  v2トレード累計={len(all_trades_v2)}")

    print(f"\n全{len(tickers)}銘柄の日次シミュレーション完了(経過{time.monotonic()-start:.0f}秒)\n")
    report("v1(decide_rule.py、毎日判定)", all_trades_v1)
    report("v2(decide_composite.py、毎日判定)", all_trades_v2)

    conn.close()


if __name__ == "__main__":
    main()
