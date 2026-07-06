"""EDINETから有価証券報告書を取得し、yfinanceの4〜5年より長期の業績推移を蓄積する。

1件の有価証券報告書には「主要な経営指標等の推移」として直近5年分
(当期/前期/前々期/三期前/四期前)の連結決算サマリーが含まれている。
書類の検索はfetch_edinet_index.pyが事前に構築したローカルインデックス
(edinet_documentsテーブル)を参照するだけで済み、API呼び出しは
CSVダウンロードのみになる(1銘柄あたりの検索コストがほぼゼロ)。

有価証券報告書は年1回しか提出されないため、全銘柄への初回バックフィルが
終わったあとは月次〜年次程度の低頻度実行で十分(fetch_fundamentals.pyのような
日次ローリングは不要)。

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
        "jpcrp_cor:RevenueIFRSSummaryOfBusinessResults",
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


def find_indexed_reports(conn, sec_code: str, max_filings: int) -> list[dict]:
    """fetch_edinet_index.pyが構築したローカルインデックスから、期末日が新しい順に
    有価証券報告書を取得する。期末日が近すぎる(同一世代の重複)ものは間引く。
    """
    rows = conn.execute(
        """
        SELECT doc_id, period_end FROM edinet_documents
        WHERE sec_code = ? AND doc_type_code = ? AND period_end IS NOT NULL
        ORDER BY period_end DESC
        """,
        (sec_code, SECURITIES_REPORT_DOC_TYPE),
    ).fetchall()

    selected = []
    last_period_end = None
    for doc_id, period_end_str in rows:
        period_end = datetime.strptime(period_end_str, "%Y-%m-%d").date()
        if last_period_end is not None and (last_period_end - period_end).days < 365 * 3:
            continue  # 同じ~5年世代内の重複書類はスキップ
        selected.append({"docID": doc_id, "periodEnd": period_end_str})
        last_period_end = period_end
        if len(selected) >= max_filings:
            break
    return selected


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


# 1株配当・配当性向は「提出会社(=親会社)の株式1株あたり」の指標であり、そもそも連結概念が
# 存在しないため、EDINETは常に非連結(NonConsolidatedMember)コンテキストでのみ開示する。
# 他の指標と同じ_is_consolidated_contextでフィルタすると常に除外されてしまうため、
# この2指標だけは非連結コンテキストも許可する。
NON_CONSOLIDATED_ONLY_METRICS = {"dividend_per_share", "payout_ratio_pct"}


def _matches_context(metric: str, context_id: str) -> bool:
    if metric in NON_CONSOLIDATED_ONLY_METRICS:
        return "NonConsolidatedMember" in context_id or _is_consolidated_context(context_id)
    return _is_consolidated_context(context_id)


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
    """まだ埋まっていない相対年度だけ値を書き込む(優先度の高い候補タグの値を後発候補で上書きしない)"""
    found = False
    for _, row in sub.iterrows():
        rel_year = row["相対年度"]
        if rel_year not in RELATIVE_YEAR_OFFSET:
            continue
        if metric in result.get(rel_year, {}):
            continue
        value = pd.to_numeric(row["値"], errors="coerce")
        if pd.notna(value):
            # sqlite3はnumpy.float64をバインドできずBLOB化してしまうためPython floatに変換する
            result.setdefault(rel_year, {})[metric] = float(value)
            found = True
    return found


def extract_yearly_metrics(df: pd.DataFrame) -> dict:
    """相対年度ラベル -> {metric: value} の辞書を返す。

    IFRS移行期の企業は、同一書類内で古い年度がJ-GAAPタグ・直近年度がIFRSタグという
    形でタグが混在することがある。以前は候補タグを順に試して最初に1件でも見つかった
    時点で打ち切っていたため、旧タグで一部年度だけヒットすると新タグ側の残り年度を
    見に行かず欠損していた。全5相対年度が埋まるまで(または候補を使い切るまで)候補を
    試し続け、複数タグの結果をマージする。
    """
    result: dict = {}
    all_years = set(RELATIVE_YEAR_OFFSET.keys())
    for metric, candidates in METRIC_TAG_CANDIDATES.items():
        for tag in candidates:
            filled_years = {y for y, m in result.items() if metric in m}
            if filled_years >= all_years:
                break
            sub = df[(df["要素ID"] == tag) & df["コンテキストID"].apply(lambda c: _matches_context(metric, c))]
            if sub.empty:
                continue
            _apply_rows(result, metric, sub)

        filled_years = {y for y, m in result.items() if metric in m}
        if metric == "revenue" and filled_years < all_years:
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
            # EDINETのPayoutRatioSummaryOfBusinessResultsは0.515のような小数(既に比率)で
            # 開示されており、パーセント表記(51.5)ではないため/100してはいけない
            payout_ratio = metrics.get("payout_ratio_pct")
            # EDINETの5年サマリー表には負債額そのものは無いが、総資産・純資産は取れるため差分で算出する
            total_liabilities = (total_assets - equity) if total_assets is not None and equity is not None else None
            conn.execute(
                """
                INSERT INTO fundamentals_yearly
                    (ticker, fiscal_year_end, revenue, ordinary_income, net_income,
                     operating_margin, net_margin, eps, dividend_per_share, payout_ratio,
                     total_assets, total_liabilities, equity, equity_ratio,
                     operating_cf, investing_cf, financing_cf, cash_and_equivalents, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                    total_liabilities=COALESCE(excluded.total_liabilities, fundamentals_yearly.total_liabilities),
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
                    metrics.get("eps"), metrics.get("dividend_per_share"), payout_ratio,
                    total_assets, total_liabilities, equity,
                    (equity / total_assets * 100) if equity and total_assets else None,
                    metrics.get("operating_cf"), metrics.get("investing_cf"), metrics.get("financing_cf"),
                    metrics.get("cash_and_equivalents"), now_iso,
                ),
            )
            written += 1
    return written


def fetch_ticker_history(conn, ticker: str, sec_code: str, lookback_filings: int = 3) -> int:
    """ローカルインデックスから対象銘柄の有価証券報告書を探し、CSVをダウンロードして取り込む"""
    docs = find_indexed_reports(conn, sec_code, lookback_filings)
    total_written = 0

    for doc in docs:
        tables = download_csv_tables(doc["docID"])
        period_end = datetime.strptime(doc["periodEnd"], "%Y-%m-%d").date()

        for df in tables:
            yearly_metrics = extract_yearly_metrics(df)
            if yearly_metrics:
                total_written += upsert_yearly(conn, ticker, period_end, yearly_metrics)
        time.sleep(1)

    return total_written


def log_result(conn, run_date: str, ticker: str, status: str, error_message: str = "") -> None:
    with conn:
        conn.execute(
            "INSERT INTO fetch_log (run_date, job_type, ticker, status, error_message) VALUES (?, ?, ?, ?, ?)",
            (run_date, "edinet", ticker, status, error_message),
        )


def load_rolling_targets(conn, limit: int | None) -> list[str]:
    """edinetジョブでまだ処理していない(またはより古くに処理した)銘柄から順に選ぶ"""
    query = """
    SELECT t.ticker
    FROM tickers t
    LEFT JOIN (
        SELECT ticker, MAX(run_date) AS last_run FROM fetch_log WHERE job_type = 'edinet' GROUP BY ticker
    ) f ON t.ticker = f.ticker
    WHERE t.is_active = 1
    ORDER BY f.last_run IS NOT NULL, f.last_run ASC
    """
    tickers = [r[0] for r in conn.execute(query)]
    return tickers[:limit] if limit else tickers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None, help="例: 7203.T 1301.T 8306.T (省略時は全銘柄をローリング処理)")
    parser.add_argument("--limit", type=int, default=None, help="ローリング処理する銘柄数の上限(検証用)")
    parser.add_argument("--lookback-filings", type=int, default=3)
    parser.add_argument("--time-budget-sec", type=int, default=20 * 60)
    args = parser.parse_args()

    conn = get_connection()
    run_date = datetime.now(JST).strftime("%Y-%m-%d")

    targets = args.tickers if args.tickers else load_rolling_targets(conn, args.limit)
    start_time = time.monotonic()
    success, failed = 0, 0

    for i, ticker in enumerate(targets, start=1):
        if time.monotonic() - start_time > args.time_budget_sec:
            print(f"時間予算に到達。{i - 1}/{len(targets)}件処理して終了")
            break
        code = ticker.split(".")[0]
        sec_code = f"{code}0"
        try:
            written = fetch_ticker_history(conn, ticker, sec_code, args.lookback_filings)
            log_result(conn, run_date, ticker, "success")
            success += 1
        except Exception as e:
            written = 0
            log_result(conn, run_date, ticker, "failed", str(e))
            failed += 1
        if i % 20 == 0:
            print(f"[{i}/{len(targets)}] success={success} failed={failed}")

    conn.close()
    print(f"完了: success={success}, failed={failed}")


if __name__ == "__main__":
    main()
