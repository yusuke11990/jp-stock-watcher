"""全銘柄の価格データ(日次)をチャンク単位でバッチ取得し、price_dailyへ洗い替え保存する。

1銘柄ずつ.history()を呼ぶと3,900銘柄で数時間かかるため、
yf.download()のバッチ機能でチャンク(既定150銘柄)単位にまとめて取得する。
period="5d"で直近5営業日を毎回洗い替えすることで、
祝日や一時的な取得失敗による欠損を自然に埋める。
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.db import get_connection  # noqa: E402
from common.yf_client import TooManyFailuresError, ConsecutiveFailureGuard  # noqa: E402

CHUNK_SIZE = 150
CHUNK_SLEEP_SEC = 3
JST = timezone(timedelta(hours=9))


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=5, min=5, max=20), reraise=True)
def download_chunk(tickers: list[str], period: str = "5d") -> pd.DataFrame:
    return yf.download(
        tickers=tickers,
        period=period,
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=False,
    )


def load_active_tickers(limit: int | None = None) -> list[str]:
    conn = get_connection()
    query = "SELECT ticker FROM tickers WHERE is_active = 1 ORDER BY ticker"
    tickers = [r[0] for r in conn.execute(query)]
    conn.close()
    return tickers[:limit] if limit else tickers


def extract_ticker_df(batch_df: pd.DataFrame, ticker: str, multi: bool) -> pd.DataFrame:
    if multi:
        if ticker not in batch_df.columns.get_level_values(0):
            return pd.DataFrame()
        return batch_df[ticker].dropna(how="all")
    return batch_df.dropna(how="all")


def upsert_prices(conn, ticker: str, df: pd.DataFrame) -> int:
    rows = 0
    with conn:
        for idx, row in df.iterrows():
            if pd.isna(row.get("Close")):
                continue
            conn.execute(
                """
                INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, date) DO UPDATE SET
                    open=excluded.open, high=excluded.high, low=excluded.low,
                    close=excluded.close, volume=excluded.volume
                """,
                (
                    ticker,
                    idx.strftime("%Y-%m-%d"),
                    float(row["Open"]) if pd.notna(row.get("Open")) else None,
                    float(row["High"]) if pd.notna(row.get("High")) else None,
                    float(row["Low"]) if pd.notna(row.get("Low")) else None,
                    float(row["Close"]),
                    int(row["Volume"]) if pd.notna(row.get("Volume")) else None,
                ),
            )
            rows += 1
    return rows


def log_result(conn, run_date: str, ticker: str, status: str, error_message: str = "") -> None:
    with conn:
        conn.execute(
            "INSERT INTO fetch_log (run_date, job_type, ticker, status, error_message) VALUES (?, ?, ?, ?, ?)",
            (run_date, "price", ticker, status, error_message),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="検証用に処理する銘柄数を制限")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--period", default="5d", help="初回バックフィル時は6mo/1y等を指定")
    args = parser.parse_args()

    tickers = load_active_tickers(limit=args.limit)
    conn = get_connection()
    run_date = datetime.now(JST).strftime("%Y-%m-%d")
    guard = ConsecutiveFailureGuard()

    total_success, total_failed = 0, 0
    chunks = list(chunked(tickers, args.chunk_size))

    for i, chunk in enumerate(chunks, start=1):
        print(f"[{i}/{len(chunks)}] fetching {len(chunk)} tickers...")
        try:
            batch_df = download_chunk(chunk, period=args.period)
        except Exception as e:
            print(f"  chunk failed: {e}")
            for t in chunk:
                log_result(conn, run_date, t, "failed", str(e))
                total_failed += 1
            continue

        multi = isinstance(batch_df.columns, pd.MultiIndex)
        for t in chunk:
            try:
                df = extract_ticker_df(batch_df, t, multi)
                if df.empty:
                    raise ValueError("no data returned")
                upsert_prices(conn, t, df)
                log_result(conn, run_date, t, "success")
                guard.record_success()
                total_success += 1
            except Exception as e:
                log_result(conn, run_date, t, "failed", str(e))
                total_failed += 1
                try:
                    guard.record_failure()
                except TooManyFailuresError as blocked:
                    print(f"停止: {blocked}")
                    conn.close()
                    sys.exit(1)

        time.sleep(CHUNK_SLEEP_SEC)

    conn.close()
    print(f"完了: success={total_success}, failed={total_failed}")


if __name__ == "__main__":
    main()
