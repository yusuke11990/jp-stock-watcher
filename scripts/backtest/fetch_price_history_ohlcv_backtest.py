"""v1/v2判断エンジンの日次シミュレーションバックテストを、本番price_dailyの1年分だけでなく
複数年(既定5年)に拡張するため、全銘柄の完全なOHLCV(始値・高値・安値・終値・出来高)を
data/backtest.db(.gitignore対象、本番stock.dbとは分離)に取得する。

既存のfetch_price_history_backtest.pyはcloseのみを取得していたが、技術シグナルv2
(ADX/ATR/出来高系ファミリー)の完全な再現にはOHLCVすべてが必要なため、
本スクリプトで別途OHLCV版をprice_historyテーブルに追加(COALESCE保護つき)する。

本番のfetch_price_daily.pyと違い、これは1回限りのバックテスト用データ準備であり、
定期実行やGitHubへのコミットは想定しない。

実行: python scripts/backtest/fetch_price_history_ohlcv_backtest.py --period 5y
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection, get_backtest_connection  # noqa: E402

CHUNK_SIZE = 150
CHUNK_SLEEP_SEC = 3


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def load_active_tickers(limit: int | None = None) -> list[str]:
    conn = get_connection()
    tickers = [r[0] for r in conn.execute("SELECT ticker FROM tickers WHERE is_active = 1 ORDER BY ticker")]
    conn.close()
    return tickers[:limit] if limit else tickers


def upsert_ohlcv(conn, ticker: str, df: pd.DataFrame) -> int:
    rows = 0
    with conn:
        for idx, row in df.iterrows():
            close = row.get("Close")
            if pd.isna(close):
                continue
            conn.execute(
                """
                INSERT INTO price_history (ticker, date, close, open, high, low, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, date) DO UPDATE SET
                    close=excluded.close,
                    open=COALESCE(excluded.open, price_history.open),
                    high=COALESCE(excluded.high, price_history.high),
                    low=COALESCE(excluded.low, price_history.low),
                    volume=COALESCE(excluded.volume, price_history.volume)
                """,
                (
                    ticker, idx.strftime("%Y-%m-%d"), float(close),
                    float(row["Open"]) if pd.notna(row.get("Open")) else None,
                    float(row["High"]) if pd.notna(row.get("High")) else None,
                    float(row["Low"]) if pd.notna(row.get("Low")) else None,
                    float(row["Volume"]) if pd.notna(row.get("Volume")) else None,
                ),
            )
            rows += 1
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="5y")
    parser.add_argument("--limit", type=int, default=None, help="検証用に処理する銘柄数を制限")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    tickers = load_active_tickers(limit=args.limit)
    conn = get_backtest_connection()

    total_success, total_failed = 0, 0
    chunks = list(chunked(tickers, args.chunk_size))
    start = time.monotonic()

    for i, chunk in enumerate(chunks, start=1):
        try:
            # auto_adjust=Trueで株式分割・配当調整済みのOHLCを取得する。Falseの生値だと、
            # 5年という長期間では実際に株式分割を経験する銘柄があり、分割日をまたぐ
            # リターン計算が実際には起きていない暴騰・暴落として混入してしまう
            # (本番のfetch_price_daily.pyはFalseのままでよい。直近5営業日の洗い替えのみで、
            # その短期間に分割が起きることは通常なく実務上問題にならないため)
            batch_df = yf.download(
                tickers=chunk, period=args.period, group_by="ticker", threads=True,
                progress=False, auto_adjust=True,
            )
        except Exception as e:
            print(f"[{i}/{len(chunks)}] chunk failed: {e}")
            total_failed += len(chunk)
            continue

        multi = isinstance(batch_df.columns, pd.MultiIndex)
        for t in chunk:
            try:
                if multi:
                    if t not in batch_df.columns.get_level_values(0):
                        raise ValueError("no data returned")
                    df = batch_df[t].dropna(how="all")
                else:
                    df = batch_df.dropna(how="all")
                if df.empty:
                    raise ValueError("no data returned")
                upsert_ohlcv(conn, t, df)
                total_success += 1
            except Exception:
                total_failed += 1

        if i % 5 == 0 or i == len(chunks):
            print(f"[{i}/{len(chunks)}] 経過{time.monotonic()-start:.0f}秒  success={total_success} failed={total_failed}")
        time.sleep(CHUNK_SLEEP_SEC)

    conn.close()
    print(f"完了: success={total_success}, failed={total_failed}")


if __name__ == "__main__":
    main()
