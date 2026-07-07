"""複数期間のファンダ質的スコアバックテスト用に、全銘柄の長期株価履歴(yfinance最大10年分)を
data/backtest.db(.gitignore対象、本番stock.dbとは分離)に取得する。

本番のfetch_price_daily.pyと違い、これは1回限りのバックテスト用データ準備であり、
定期実行やGitHubへのコミットは想定しない。
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


def upsert_prices(conn, ticker: str, df: pd.DataFrame) -> int:
    rows = 0
    with conn:
        for idx, row in df.iterrows():
            close = row.get("Close")
            if pd.isna(close):
                continue
            conn.execute(
                """
                INSERT INTO price_history (ticker, date, close) VALUES (?, ?, ?)
                ON CONFLICT(ticker, date) DO UPDATE SET close=excluded.close
                """,
                (ticker, idx.strftime("%Y-%m-%d"), float(close)),
            )
            rows += 1
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="10y")
    parser.add_argument("--limit", type=int, default=None, help="検証用に処理する銘柄数を制限")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    tickers = load_active_tickers(limit=args.limit)
    conn = get_backtest_connection()

    total_success, total_failed = 0, 0
    chunks = list(chunked(tickers, args.chunk_size))

    for i, chunk in enumerate(chunks, start=1):
        print(f"[{i}/{len(chunks)}] fetching {len(chunk)} tickers...")
        try:
            batch_df = yf.download(
                tickers=chunk, period=args.period, group_by="ticker", threads=True,
                progress=False, auto_adjust=False,
            )
        except Exception as e:
            print(f"  chunk failed: {e}")
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
                upsert_prices(conn, t, df)
                total_success += 1
            except Exception:
                total_failed += 1

        if i % 5 == 0 or i == len(chunks):
            print(f"  進捗: success={total_success} failed={total_failed}")
        time.sleep(CHUNK_SLEEP_SEC)

    conn.close()
    print(f"完了: success={total_success}, failed={total_failed}")


if __name__ == "__main__":
    main()
