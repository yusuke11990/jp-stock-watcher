"""全銘柄×全営業日のシグナルスコアを計算し、signal_historyテーブルに保存する。

バックテスト専用のテーブルであり、本番のdecisionsとは完全に分離している。
1年・243営業日というデータ制約を踏まえ、時系列の長さではなく銘柄横断の
厚み(3,548銘柄×約240日)を活かしたイベントスタディ分析(event_study.py)の
入力データとして使う。

市場・セクターレジームは銘柄ごとにDB問い合わせすると3,548銘柄×約170日で
非常に遅くなるため、全銘柄の価格を一括ロードしてpandasでベクトル化して
計算する(市場全体で1回、セクターは(sector, date)単位で1回)。
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection, get_backtest_connection  # noqa: E402
from common.technical_v2 import (  # noqa: E402
    calc_all_indicators,
    trend_family_score,
    mean_reversion_family_score,
    volume_family_score,
    volatility_regime,
    bb_width_percentile,
    trend_alignment_score,
)

JST = timezone(timedelta(hours=9))
TOPIX_TICKER = "1306.T"
MIN_SECTOR_SAMPLE_SIZE = 5
WARMUP_DAYS = 80  # MA75+バッファ分、これより前の日はスコア計算しない


def load_all_prices(conn) -> pd.DataFrame:
    query = """
    SELECT ticker, date, open, high, low, close, volume FROM price_daily ORDER BY ticker, date ASC
    """
    df = pd.read_sql_query(query, conn, parse_dates=["date"])
    return df


def load_sector_map(conn) -> dict:
    return dict(conn.execute("SELECT ticker, sector FROM tickers WHERE is_active = 1"))


def compute_topix_trend_by_date(all_prices: pd.DataFrame) -> pd.Series:
    topix = all_prices[all_prices["ticker"] == TOPIX_TICKER].set_index("date").rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    if topix.empty:
        return pd.Series(dtype=float)
    topix = calc_all_indicators(topix)
    scores = topix.apply(lambda r: trend_alignment_score(r.get("MA5"), r.get("MA25"), r.get("MA75")), axis=1)
    return scores


def compute_sector_regime_by_date(all_prices: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    """(sector, date)ごとの5日騰落率中央値スコアを一括計算する"""
    df = all_prices[["ticker", "date", "close"]].copy()
    df["sector"] = df["ticker"].map(sector_map)
    df = df.dropna(subset=["sector"])
    df = df.sort_values(["ticker", "date"])
    df["ret_5d"] = df.groupby("ticker")["close"].pct_change(5)

    sector_counts = df.groupby("sector")["ticker"].nunique()
    valid_sectors = sector_counts[sector_counts >= MIN_SECTOR_SAMPLE_SIZE].index
    df = df[df["sector"].isin(valid_sectors)]

    grouped = df.groupby(["sector", "date"])["ret_5d"].median().reset_index()
    grouped["sector_regime_score"] = grouped["ret_5d"].apply(
        lambda v: max(-1.0, min(1.0, v * 10)) if pd.notna(v) else 0.0
    )
    return grouped.set_index(["sector", "date"])["sector_regime_score"]


def compute_ticker_history(ticker_df: pd.DataFrame) -> pd.DataFrame:
    """1銘柄分の価格履歴(日付昇順)から、各日のtrend/mean_reversion/volumeスコアを計算する"""
    if len(ticker_df) < WARMUP_DAYS:
        return pd.DataFrame()

    df = ticker_df.set_index("date").rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    df = calc_all_indicators(df)

    rows = []
    for i in range(WARMUP_DAYS, len(df)):
        sub = df.iloc[: i + 1]
        row = sub.iloc[-1]
        regime = volatility_regime(row.get("ADX"), bb_width_percentile(sub["BB_width"]))
        rows.append({
            "date": sub.index[-1],
            "trend_score": trend_family_score(row, sub),
            "mean_reversion_score": mean_reversion_family_score(row, sub),
            "volume_score": volume_family_score(row, sub),
            "regime_volatility": regime["regime"],
            "close": row["Close"],
        })
    return pd.DataFrame(rows)


def add_forward_returns(hist_df: pd.DataFrame) -> pd.DataFrame:
    hist_df = hist_df.sort_values("date").reset_index(drop=True)
    close = hist_df["close"].values
    n = len(close)
    for horizon, col in [(5, "forward_return_5d"), (10, "forward_return_10d"), (21, "forward_return_21d")]:
        fwd = np.full(n, np.nan)
        for i in range(n - horizon):
            if close[i] > 0:
                fwd[i] = (close[i + horizon] - close[i]) / close[i]
        hist_df[col] = fwd
    return hist_df


def upsert_signal_history(conn, ticker: str, df: pd.DataFrame) -> int:
    now_iso = datetime.now(JST).isoformat()
    written = 0
    with conn:
        for _, row in df.iterrows():
            conn.execute(
                """
                INSERT INTO signal_history
                    (ticker, date, trend_score, mean_reversion_score, volume_score, regime_volatility,
                     market_regime_score, sector_regime_score, composite_technical_score,
                     forward_return_5d, forward_return_10d, forward_return_21d, computed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ticker, date) DO UPDATE SET
                    trend_score=excluded.trend_score, mean_reversion_score=excluded.mean_reversion_score,
                    volume_score=excluded.volume_score, regime_volatility=excluded.regime_volatility,
                    market_regime_score=excluded.market_regime_score, sector_regime_score=excluded.sector_regime_score,
                    composite_technical_score=excluded.composite_technical_score,
                    forward_return_5d=excluded.forward_return_5d, forward_return_10d=excluded.forward_return_10d,
                    forward_return_21d=excluded.forward_return_21d, computed_at=excluded.computed_at
                """,
                (
                    ticker, row["date"].strftime("%Y-%m-%d"),
                    row["trend_score"], row["mean_reversion_score"], row["volume_score"], row["regime_volatility"],
                    row.get("market_regime_score"), row.get("sector_regime_score"), row.get("composite_technical_score"),
                    row.get("forward_return_5d"), row.get("forward_return_10d"), row.get("forward_return_21d"), now_iso,
                ),
            )
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="処理する銘柄数の上限(検証用)")
    parser.add_argument("--time-budget-sec", type=int, default=20 * 60)
    args = parser.parse_args()

    conn = get_connection()
    backtest_conn = get_backtest_connection()
    print("全銘柄の価格データを読み込み中...")
    all_prices = load_all_prices(conn)
    sector_map = load_sector_map(conn)

    print("TOPIX市場レジームを計算中...")
    topix_trend = compute_topix_trend_by_date(all_prices)

    print("セクターレジームを計算中...")
    sector_regime = compute_sector_regime_by_date(all_prices, sector_map)

    tickers = sorted(all_prices["ticker"].unique())
    tickers = [t for t in tickers if t != TOPIX_TICKER]
    if args.limit:
        tickers = tickers[: args.limit]

    start_time = time.monotonic()
    total_rows = 0
    for i, ticker in enumerate(tickers, start=1):
        if time.monotonic() - start_time > args.time_budget_sec:
            print(f"時間予算に到達。{i - 1}/{len(tickers)}銘柄処理して終了")
            break

        ticker_df = all_prices[all_prices["ticker"] == ticker]
        hist = compute_ticker_history(ticker_df)
        if hist.empty:
            continue

        hist["market_regime_score"] = hist["date"].map(topix_trend).fillna(0.0)
        sector = sector_map.get(ticker)
        if sector:
            hist["sector_regime_score"] = hist["date"].apply(
                lambda d: sector_regime.get((sector, d), 0.0)
            )
        else:
            hist["sector_regime_score"] = 0.0

        # 暫定の合成スコア(チューニング前の単純平均。weight_optimizer.py実行後にconfigの重みで再計算する)
        hist["composite_technical_score"] = (
            hist["trend_score"] * 0.35 + hist["mean_reversion_score"] * 0.25
            + hist["volume_score"] * 0.20 + hist["market_regime_score"] * 0.15
            + hist["sector_regime_score"] * 0.05
        )

        hist = add_forward_returns(hist)
        total_rows += upsert_signal_history(backtest_conn, ticker, hist)

        if i % 100 == 0:
            print(f"[{i}/{len(tickers)}] 累計{total_rows}行")

    conn.close()
    backtest_conn.close()
    print(f"完了: {total_rows}行をsignal_historyに保存")


if __name__ == "__main__":
    main()
