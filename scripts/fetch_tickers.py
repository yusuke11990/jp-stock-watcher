"""JPX公開の東証上場銘柄一覧(data_j.xls)を取得し、tickersテーブルへupsertする。

参照元URLはJPXサイトのリニューアルで変わることがあるため、
取得失敗時はJPXの一覧ページを確認して定数JPX_DATA_URLを更新すること。
https://www.jpx.co.jp/markets/statistics-equities/misc/01.html
"""

import sys
from datetime import datetime, timezone, timedelta
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.db import get_connection  # noqa: E402

JPX_DATA_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
JST = timezone(timedelta(hours=9))

# 普通株のみ対象にする場合はここに含める市場区分(除外したい場合はNoneのまま全件対象)
EXCLUDE_MARKETS = {
    "ETF・ETN",
    "REIT・ベンチャーファンド・カントリーファンド・インフラファンド",
    "PRO Market",
    "出資証券",
}


def download_jpx_list() -> pd.DataFrame:
    resp = requests.get(JPX_DATA_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_excel(BytesIO(resp.content))
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "コード": "code",
            "銘柄名": "name",
            "市場・商品区分": "market",
            "33業種区分": "sector",
        }
    )
    df["code"] = df["code"].astype(str).str.strip()
    df = df[df["code"].str.fullmatch(r"\d{4}")]
    if EXCLUDE_MARKETS:
        df = df[~df["market"].isin(EXCLUDE_MARKETS)]
    df["ticker"] = df["code"] + ".T"
    return df[["ticker", "code", "name", "market", "sector"]]


def upsert(df: pd.DataFrame) -> None:
    conn = get_connection()
    now = datetime.now(JST).isoformat()
    seen = set(df["ticker"])

    with conn:
        for _, row in df.iterrows():
            conn.execute(
                """
                INSERT INTO tickers (ticker, code, name, market, sector, is_active, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    code=excluded.code,
                    name=excluded.name,
                    market=excluded.market,
                    sector=excluded.sector,
                    is_active=1,
                    updated_at=excluded.updated_at
                """,
                (row["ticker"], row["code"], row["name"], row["market"], row["sector"], now),
            )

        # 一覧から消えた銘柄(上場廃止等)は論理削除
        existing = [r[0] for r in conn.execute("SELECT ticker FROM tickers WHERE is_active = 1")]
        delisted = [t for t in existing if t not in seen]
        for t in delisted:
            conn.execute("UPDATE tickers SET is_active = 0, updated_at = ? WHERE ticker = ?", (now, t))

    conn.close()
    print(f"upserted {len(df)} tickers, delisted {len(delisted)} tickers")


def main():
    raw = download_jpx_list()
    normalized = normalize(raw)
    upsert(normalized)


if __name__ == "__main__":
    main()
