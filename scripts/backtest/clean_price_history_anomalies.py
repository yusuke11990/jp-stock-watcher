"""backtest.dbのprice_historyに混入したデータ異常を検出・除去する。

fetch_price_history_ohlcv_backtest.py(auto_adjust=True)で株式分割・配当調整は
できるが、それとは別にyfinance側の一時的なデータ障害(特定日だけ現実離れした価格が
入る、あるいは特定期間ずっとゼロ・負値になる)が混入することがある。実際に発見した例:

- 8303.T(SBI新生銀行、2023年に非公開化・上場廃止): 2022-03-30〜2023-03-29の
  245営業日分がゼロ以下の値、さらに一部の日は5.5兆円等の桁違いの値になっていた。
  上場廃止前後でデータ品質が悪化したとみられ、当該期間だけの部分修正では済まないため
  銘柄ごと除外する。
- 9204.T, 5537.T, 7946.T: auto_adjust=Trueでも解消されない、特定の1日だけの
  桁違いなジャンプ(株式分割にしては説明のつかないパターン)があったため除外する。

検出方法: 121営業日ローリング中央値に対する比率が10倍を超える/10分の1を下回る日を
異常とみなす。ローリング窓を使うのは、複数日にまたがる障害(8303.Tのケース)を
単純な前日比だけでは検知できないため。

実行: python scripts/backtest/clean_price_history_anomalies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_backtest_connection  # noqa: E402

ROLLING_WINDOW = 121
RATIO_THRESHOLD = 10.0


def find_anomalous_tickers(conn) -> set[str]:
    df = pd.read_sql_query(
        "SELECT ticker, date, close FROM price_history WHERE close IS NOT NULL ORDER BY ticker, date",
        conn, parse_dates=["date"],
    )
    if df.empty:
        return set()

    zero_or_negative = set(df.loc[df["close"] <= 0, "ticker"].unique())

    df["rolling_median"] = df.groupby("ticker")["close"].transform(
        lambda s: s.rolling(ROLLING_WINDOW, center=True, min_periods=30).median()
    )
    df["ratio"] = df["close"] / df["rolling_median"]
    jump_tickers = set(
        df.loc[(df["ratio"] > RATIO_THRESHOLD) | (df["ratio"] < 1 / RATIO_THRESHOLD), "ticker"].dropna().unique()
    )
    return zero_or_negative | jump_tickers


def main():
    conn = get_backtest_connection()
    bad_tickers = find_anomalous_tickers(conn)
    print(f"異常データを検出した銘柄: {sorted(bad_tickers)}")

    total_deleted = 0
    with conn:
        for ticker in bad_tickers:
            cur = conn.execute("DELETE FROM price_history WHERE ticker = ?", (ticker,))
            total_deleted += cur.rowcount
            print(f"  {ticker}: {cur.rowcount}行削除")

    print(f"合計{total_deleted}行削除(銘柄数: {len(bad_tickers)})")
    conn.close()


if __name__ == "__main__":
    main()
