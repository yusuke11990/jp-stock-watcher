"""EDINETの提出書類一覧を日付ごとに1回だけスキャンし、全銘柄分をedinet_documentsにインデックス化する。

1銘柄ずつAPI検索すると3,548銘柄×複数世代分で膨大な呼び出し数になるため、
日付単位で書類一覧(その日提出された全社分)を1回取得してsecCode付きで
保存しておき、fetch_edinet_history.pyはこのローカルインデックスを
参照するだけで済むようにする。

対象期間は「直近1年」「約5年前の1年」「約10年前の1年」のように
世代(generation)を指定してスキャンする。決算期は3月末以外の企業も
あるため、各世代につき1年分(365日)をスキャンして取りこぼしを防ぐ。
"""

from __future__ import annotations

import argparse
import os
import re
import time
from datetime import datetime, timedelta, timezone

import requests

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.db import get_connection  # noqa: E402

JST = timezone(timedelta(hours=9))
EDINET_BASE = "https://api.edinet-fsa.go.jp/api/v2"
SECURITIES_REPORT_DOC_TYPE = "120"


def _api_key() -> str:
    key = os.environ.get("EDINET_API_KEY")
    if not key:
        raise RuntimeError("環境変数EDINET_API_KEYが未設定です")
    return key


def _redact_api_key(text: str) -> str:
    """requestsの例外メッセージにSubscription-Key付きURLが含まれることがあり、
    そのままprintするとGitHub Actionsのログにキーが残ってしまうため必ずこれを通す。
    """
    key = os.environ.get("EDINET_API_KEY")
    if key:
        text = text.replace(key, "***")
    return re.sub(r"Subscription-Key=[^&\s]+", "Subscription-Key=***", text)


def already_scanned_dates(conn) -> set:
    return {r[0] for r in conn.execute("SELECT scan_date FROM edinet_scanned_dates")}


def scan_date(conn, key: str, d) -> int:
    resp = requests.get(
        f"{EDINET_BASE}/documents.json",
        params={"date": d.isoformat(), "type": 2, "Subscription-Key": key},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    count = 0
    now_iso = datetime.now(JST).isoformat()
    with conn:
        for item in data.get("results", []):
            if item.get("docTypeCode") != SECURITIES_REPORT_DOC_TYPE:
                continue
            if not item.get("secCode"):
                continue
            conn.execute(
                """
                INSERT INTO edinet_documents (doc_id, sec_code, doc_type_code, period_end, submit_date_time, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    sec_code=excluded.sec_code, period_end=excluded.period_end,
                    submit_date_time=excluded.submit_date_time, fetched_at=excluded.fetched_at
                """,
                (item["docID"], item["secCode"], item["docTypeCode"], item.get("periodEnd"), item.get("submitDateTime"), now_iso),
            )
            count += 1
        conn.execute(
            "INSERT OR REPLACE INTO edinet_scanned_dates (scan_date, doc_count, scanned_at) VALUES (?, ?, ?)",
            (d.isoformat(), count, now_iso),
        )
    return count


def generation_windows(generations: list[int]) -> list[tuple]:
    """generationsは「何年前」のリスト(例: [0, 5, 10])。各世代につき前後200日の窓を返す"""
    today = datetime.now(JST).date()
    windows = []
    for years_back in generations:
        center = today.replace(year=today.year - years_back)
        start = center - timedelta(days=200)
        end = center + timedelta(days=20)
        windows.append((start, end))
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, nargs="+", default=[0, 5, 10],
                         help="何年前を中心にスキャンするか(既定: 直近・5年前・10年前)")
    parser.add_argument("--time-budget-sec", type=int, default=25 * 60)
    args = parser.parse_args()

    key = _api_key()
    conn = get_connection()
    scanned = already_scanned_dates(conn)

    all_dates = []
    for start, end in generation_windows(args.generations):
        d = start
        while d <= end:
            if d.isoformat() not in scanned and d <= datetime.now(JST).date():
                all_dates.append(d)
            d += timedelta(days=1)

    print(f"未スキャンの対象日数: {len(all_dates)}")
    start_time = time.monotonic()
    processed, total_docs = 0, 0

    for d in all_dates:
        if time.monotonic() - start_time > args.time_budget_sec:
            print(f"時間予算に到達。{processed}/{len(all_dates)}日処理して終了")
            break
        try:
            count = scan_date(conn, key, d)
            total_docs += count
            processed += 1
        except Exception as e:
            print(f"  {d}: 失敗 {_redact_api_key(str(e))}")
        time.sleep(0.2)
        if processed % 50 == 0:
            print(f"[{processed}/{len(all_dates)}] 累計{total_docs}件の書類をインデックス化")

    conn.close()
    print(f"完了: {processed}日スキャン、{total_docs}件の書類をインデックス化")


if __name__ == "__main__":
    main()
