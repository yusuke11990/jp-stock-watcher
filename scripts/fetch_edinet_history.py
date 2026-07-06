"""EDINETから有価証券報告書を取得し、yfinanceの4〜5年より長期の業績推移を蓄積する。

1件の有価証券報告書には「主要な経営指標等の推移」として直近5年分
(当期/前期/前々期/三期前/四期前)の連結決算サマリーが含まれている。
これを約5年おきの過去の提出書類まで遡って複数件取得することで、
fundamentals_yearlyに15年超の推移を積み上げる。

APIキーは環境変数EDINET_API_KEYから読む(GitHub Secretsに格納想定)。
連結値は「コンテキストID」に_NonConsolidatedMember等の接尾辞が付かない
行として判定できる(IFRS/J-GAAP問わず共通のパターン)。
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.db import get_connection  # noqa: E402

JST = timezone(timedelta(hours=9))
EDINET_BASE = "https://api.edinet-fsa.go.jp/api/v2"
SECURITIES_REPORT_DOC_TYPE = "120"  # 有価証券報告書

# 相対年度ラベル -> 基準期(periodEnd)からの年オフセット
RELATIVE_YEAR_OFFSET = {
    "当期": 0, "当期末": 0,
    "前期": -1, "前期末": -1,
    "前々期": -2, "前々期末": -2,
    "三期前": -3, "三期前時点": -3,
    "四期前": -4, "四期前時点": -4,
}

# メトリクスごとのタグ候補(先に見つかったものを採用)。IFRS/J-GAAP/銀行等の差異を吸収する
METRIC_TAG_CANDIDATES = {
    "revenue": [
        "jpcrp_cor:NetSalesSummaryOfBusinessResults",
        "jpcrp_cor:NetSalesIFRSSummaryOfBusinessResults",
        "jpcrp_cor:OperatingRevenue1SummaryOfBusinessResults",
        "jpcrp_cor:OrdinaryIncomeSummaryOfBusinessResults",
        "jpcrp_cor:OperatingRevenuesIFRSSummaryOfBusinessResults",
    ],
    "ordinary_income": [
        "jpcrp_cor:OrdinaryIncomeLossSummaryOfBusinessResults",
        "jpcrp_cor:ProfitLossBeforeTaxIFRSSummaryOfBusinessResults",
    ],
    "net_income": [
        # 連結の「親会社株主に帰属する当期純利益」を優先。NetIncomeLossSummaryOfBusinessResultsは
        # J-GAAP企業では個別(非連結)専用のタグになっているため後回しにする
        "jpcrp_cor:ProfitLossAttributableToOwnersOfParentSummaryOfBusinessResults",
        "jpcrp_cor:ProfitLossAttributableToOwnersOfParentIFRSSummaryOfBusinessResults",
        "jpcrp_cor:NetIncomeLossSummaryOfBusinessResults",
    ],
    "total_assets": [
        "jpcrp_cor:TotalAssetsSummaryOfBusinessResults",
        "jpcrp_cor:TotalAssetsIFRSSummaryOfBusinessResults",
    ],
    "equity": [
        "jpcrp_cor:NetAssetsSummaryOfBusinessResults",
        "jpcrp_cor:EquityAttributableToOwnersOfParentIFRSSummaryOfBusinessResults",
    ],
    "eps": [
        "jpcrp_cor:BasicEarningsLossPerShareSummaryOfBusinessResults",
        "jpcrp_cor:BasicEarningsLossPerShareIFRSSummaryOfBusinessResults",
    ],
    "dividend_per_share": [
        "jpcrp_cor:DividendPaidPerShareSummaryOfBusinessResults",
    ],
    "payout_ratio_pct": [
        "jpcrp_cor:PayoutRatioSummaryOfBusinessResults",
    ],
    "operating_cf": [
        "jpcrp_cor:CashFlowsFromUsedInOperatingActivitiesIFRSSummaryOfBusinessResults",
        "jpcrp_cor:NetCashProvidedByUsedInOperatingActivitiesSummaryOfBusinessResults",
    ],
    "investing_cf": [
        "jpcrp_cor:CashFlowsFromUsedInInvestingActivitiesIFRSSummaryOfBusinessResults",
        "jpcrp_cor:NetCashProvidedByUsedInInvestingActivitiesSummaryOfBusinessResults",
    ],
    "financing_cf": [
        "jpcrp_cor:CashFlowsFromUsedInFinancingActivitiesIFRSSummaryOfBusinessResults",
        "jpcrp_cor:NetCashProvidedByUsedInFinancingActivitiesSummaryOfBusinessResults",
    ],
    "cash_and_equivalents": [
        "jpcrp_cor:CashAndCashEquivalentsIFRSSummaryOfBusinessResults",
        "jpcrp_cor:CashAndCashEquivalentsSummaryOfBusinessResults",
    ],
}


def _api_key() -> str:
    key = os.environ.get("EDINET_API_KEY")
    if not key:
        raise RuntimeError("環境変数EDINET_API_KEYが未設定です")
    return key


def find_securities_report(sec_code: str, around_date: "datetime.date", window_days: int = 120):
    """around_dateから過去window_days日を遡り、secCodeに一致する有価証券報告書を探す"""
    key = _api_key()
    for offset in range(window_days):
        d = around_date - timedelta(days=offset)
        resp = requests.get(
            f"{EDINET_BASE}/documents.json",
            params={"date": d.isoformat(), "type": 2, "Subscription-Key": key},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("results", []):
            if item.get("secCode") == sec_code and item.get("docTypeCode") == SECURITIES_REPORT_DOC_TYPE:
                return item
        time.sleep(0.3)
    return None


def download_csv_tables(doc_id: str) -> list[pd.DataFrame]:
    key = _api_key()
    resp = requests.get(
        f"{EDINET_BASE}/documents/{doc_id}",
        params={"type": 5, "Subscription-Key": key},
        timeout=30,
    )
    resp.raise_for_status()
    tables = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if name.startswith("XBRL_TO_CSV/jpcrp") and name.endswith(".csv"):
                with zf.open(name) as f:
                    tables.append(pd.read_csv(f, encoding="utf-16", sep="\t"))
    return tables


def _is_consolidated_context(context_id: str) -> bool:
    return "NonConsolidatedMember" not in context_id and "Member" not in context_id


# 大企業はIFRS決算指標を"SummaryOfBusinessResults"ではなく企業固有の拡張タグ
# "...KeyFinancialData"で開示することがある(例: トヨタのOperatingRevenuesIFRSKeyFinancialData)。
# 名前空間の接頭辞(会社ごとのEDINETコードを含む)が異なるため、要素IDのローカル名(:以降)を
# 正規表現で照合するフォールバックを売上高にのみ用意する。
REVENUE_FALLBACK_PATTERN = re.compile(r"^(NetSales|OperatingRevenues?)(IFRS)?(SummaryOfBusinessResults|KeyFinancialData)$")


def _find_revenue_fallback(df: pd.DataFrame) -> pd.DataFrame:
    local_names = df["要素ID"].str.split(":").str[-1]
    mask = local_names.str.match(REVENUE_FALLBACK_PATTERN, na=False) & df["コンテキストID"].apply(_is_consolidated_context)
    return df[mask]


def _apply_rows(result: dict, metric: str, sub: pd.DataFrame) -> bool:
    found = False
    for _, row in sub.iterrows():
        rel_year = row["相対年度"]
        if rel_year not in RELATIVE_YEAR_OFFSET:
            continue
        value = pd.to_numeric(row["値"], errors="coerce")
        if pd.notna(value):
            # sqlite3はnumpy.float64をバインドできずBLOB化してしまうためPython floatに変換する
            result.setdefault(rel_year, {})[metric] = float(value)
            found = True
    return found


def extract_yearly_metrics(df: pd.DataFrame) -> dict:
    """相対年度ラベル -> {metric: value} の辞書を返す"""
    result: dict = {}
    for metric, candidates in METRIC_TAG_CANDIDATES.items():
        found = False
        for tag in candidates:
            sub = df[(df["要素ID"] == tag) & df["コンテキストID"].apply(_is_consolidated_context)]
            if sub.empty:
                continue
            found = _apply_rows(result, metric, sub)
            if found:
                break  # このメトリクスは見つかったので次の候補は試さない

        if not found and metric == "revenue":
            # 一部のIFRS大企業は企業固有の拡張タグ(...KeyFinancialData)で売上高を開示するため
            _apply_rows(result, metric, _find_revenue_fallback(df))

    return result


def upsert_yearly(conn, ticker: str, period_end: "datetime.date", yearly_metrics: dict) -> int:
    now_iso = datetime.now(JST).isoformat()
    written = 0
    with conn:
        for rel_year, metrics in yearly_metrics.items():
            offset = RELATIVE_YEAR_OFFSET[rel_year]
            fiscal_year_end = period_end.replace(year=period_end.year + offset)
            revenue = metrics.get("revenue")
            net_income = metrics.get("net_income")
            ordinary_income = metrics.get("ordinary_income")
            equity = metrics.get("equity")
            total_assets = metrics.get("total_assets")
            payout_ratio = metrics.get("payout_ratio_pct")
            conn.execute(
                """
                INSERT INTO fundamentals_yearly
                    (ticker, fiscal_year_end, revenue, ordinary_income, net_income,
                     operating_margin, net_margin, eps, dividend_per_share, payout_ratio,
                     total_assets, equity, equity_ratio,
                     operating_cf, investing_cf, financing_cf, cash_and_equivalents, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ticker, fiscal_year_end) DO UPDATE SET
                    revenue=COALESCE(excluded.revenue, fundamentals_yearly.revenue),
                    ordinary_income=COALESCE(excluded.ordinary_income, fundamentals_yearly.ordinary_income),
                    net_income=COALESCE(excluded.net_income, fundamentals_yearly.net_income),
                    operating_margin=COALESCE(excluded.operating_margin, fundamentals_yearly.operating_margin),
                    net_margin=COALESCE(excluded.net_margin, fundamentals_yearly.net_margin),
                    eps=COALESCE(excluded.eps, fundamentals_yearly.eps),
                    dividend_per_share=COALESCE(excluded.dividend_per_share, fundamentals_yearly.dividend_per_share),
                    payout_ratio=COALESCE(excluded.payout_ratio, fundamentals_yearly.payout_ratio),
                    total_assets=COALESCE(excluded.total_assets, fundamentals_yearly.total_assets),
                    equity=COALESCE(excluded.equity, fundamentals_yearly.equity),
                    equity_ratio=COALESCE(excluded.equity_ratio, fundamentals_yearly.equity_ratio),
                    operating_cf=COALESCE(excluded.operating_cf, fundamentals_yearly.operating_cf),
                    investing_cf=COALESCE(excluded.investing_cf, fundamentals_yearly.investing_cf),
                    financing_cf=COALESCE(excluded.financing_cf, fundamentals_yearly.financing_cf),
                    cash_and_equivalents=COALESCE(excluded.cash_and_equivalents, fundamentals_yearly.cash_and_equivalents),
                    updated_at=excluded.updated_at
                """,
                (
                    ticker, fiscal_year_end.isoformat(), revenue, ordinary_income, net_income,
                    (ordinary_income / revenue) if ordinary_income and revenue else None,
                    (net_income / revenue) if net_income and revenue else None,
                    metrics.get("eps"), metrics.get("dividend_per_share"),
                    (payout_ratio / 100) if payout_ratio else None,
                    total_assets, equity,
                    (equity / total_assets * 100) if equity and total_assets else None,
                    metrics.get("operating_cf"), metrics.get("investing_cf"), metrics.get("financing_cf"),
                    metrics.get("cash_and_equivalents"), now_iso,
                ),
            )
            written += 1
    return written


def fetch_ticker_history(conn, ticker: str, sec_code: str, lookback_filings: int = 3) -> int:
    """約5年おきに過去のlookback_filings件の有価証券報告書を遡って取得する"""
    total_written = 0
    search_date = datetime.now(JST).date()
    seen_doc_ids = set()

    for _ in range(lookback_filings):
        doc = find_securities_report(sec_code, search_date)
        if doc is None or doc["docID"] in seen_doc_ids:
            break
        seen_doc_ids.add(doc["docID"])

        tables = download_csv_tables(doc["docID"])
        period_end = datetime.strptime(doc["periodEnd"], "%Y-%m-%d").date()

        for df in tables:
            yearly_metrics = extract_yearly_metrics(df)
            if yearly_metrics:
                total_written += upsert_yearly(conn, ticker, period_end, yearly_metrics)

        # 次はこの提出書類が拾った最古年度(四期前)より前の提出書類を探す。
        # 提出日は決算期末の約3ヶ月後(法定提出期限)なので、その分を加算した日付を起点に遡る
        next_period_end = period_end.replace(year=period_end.year - 4)
        search_date = next_period_end + timedelta(days=110)
        time.sleep(1)

    return total_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", required=True, help="例: 7203.T 1301.T 8306.T")
    parser.add_argument("--lookback-filings", type=int, default=3)
    args = parser.parse_args()

    conn = get_connection()
    for ticker in args.tickers:
        code = ticker.split(".")[0]
        sec_code = f"{code}0"
        print(f"{ticker}: 検索中...")
        try:
            written = fetch_ticker_history(conn, ticker, sec_code, args.lookback_filings)
            print(f"  {written}件の年次データを保存")
        except Exception as e:
            print(f"  失敗: {e}")
    conn.close()


if __name__ == "__main__":
    main()
