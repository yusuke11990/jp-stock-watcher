"""ファンダメンタルズのローリング更新。

.info呼び出しは1銘柄あたり1〜2秒かかるため全銘柄を毎日回さず、
last_updatedが古い銘柄から時間予算(既定40分)いっぱいまで処理する
「ローリング更新」方式にする。銘柄の増減にも自然に追従できる。

配当利回り(dividend_yield)はyfinanceの仕様上%表記(例: 2.89 = 2.89%)で
返ってくる点に注意。他の比率(PER/PBR=倍率、ROE=%)と単位が混在する。
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.db import get_connection  # noqa: E402
from common.yf_client import TooManyFailuresError, ConsecutiveFailureGuard  # noqa: E402

JST = timezone(timedelta(hours=9))
DEFAULT_TIME_BUDGET_SEC = 40 * 60

# balance_sheet/financials/cashflowは銘柄・年度により行ラベルが揺れるため候補で探索する
BALANCE_SHEET_LABELS = {
    "total_assets": ["Total Assets"],
    "equity": ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"],
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "total_liabilities": ["Total Liabilities Net Minority Interest", "Total Liab"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
}
FINANCIALS_LABELS = {
    "revenue": ["Total Revenue"],
    "operating_income": ["Operating Income"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "interest_expense": ["Interest Expense"],
    # 経常利益(日本基準の"税引前当期純利益"に近い概念)はPretax Incomeで代替する
    "ordinary_income": ["Pretax Income"],
}
CASHFLOW_LABELS = {
    "buyback": ["Repurchase Of Capital Stock"],
    "operating_cf": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "investing_cf": ["Investing Cash Flow", "Cash Flow From Continuing Investing Activities"],
    "financing_cf": ["Financing Cash Flow", "Cash Flow From Continuing Financing Activities"],
    "free_cf": ["Free Cash Flow"],
    "dividends_paid": ["Cash Dividends Paid", "Common Stock Dividend Paid"],
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=5, min=5, max=20), reraise=True)
def fetch_all(ticker_symbol: str) -> dict:
    t = yf.Ticker(ticker_symbol)
    return {
        "info": t.info or {},
        "balance_sheet": t.balance_sheet,
        "financials": t.financials,
        "cashflow": t.cashflow,
        "dividends": t.dividends,
    }


def _get_row(df, candidates: list[str]):
    if df is None or df.empty:
        return None
    for label in candidates:
        if label in df.index:
            return df.loc[label]
    return None


def _first_two(series):
    """最新期・前期の値を(latest, prev)で返す。無ければNone。"""
    if series is None or len(series) < 2:
        return None, None
    return series.iloc[0], series.iloc[1]


def _cagr(latest, oldest, years):
    if latest is None or oldest is None or oldest == 0 or years <= 0:
        return None
    if latest / oldest < 0:
        return None
    return (latest / oldest) ** (1 / years) - 1


def compute_extra_fields(data: dict) -> dict:
    info = data["info"]
    bs = data["balance_sheet"]
    fin = data["financials"]
    cf = data["cashflow"]

    result: dict = {}

    total_assets_row = _get_row(bs, BALANCE_SHEET_LABELS["total_assets"])
    equity_row = _get_row(bs, BALANCE_SHEET_LABELS["equity"])
    current_assets_row = _get_row(bs, BALANCE_SHEET_LABELS["current_assets"])
    current_liabilities_row = _get_row(bs, BALANCE_SHEET_LABELS["current_liabilities"])

    revenue_row = _get_row(fin, FINANCIALS_LABELS["revenue"])
    op_income_row = _get_row(fin, FINANCIALS_LABELS["operating_income"])
    net_income_row = _get_row(fin, FINANCIALS_LABELS["net_income"])
    interest_expense_row = _get_row(fin, FINANCIALS_LABELS["interest_expense"])
    ordinary_income_row = _get_row(fin, FINANCIALS_LABELS["ordinary_income"])

    buyback_row = _get_row(cf, CASHFLOW_LABELS["buyback"])
    operating_cf_row = _get_row(cf, CASHFLOW_LABELS["operating_cf"])

    total_assets = total_assets_row.iloc[0] if total_assets_row is not None and len(total_assets_row) else None
    equity = equity_row.iloc[0] if equity_row is not None and len(equity_row) else None
    net_income_latest = net_income_row.iloc[0] if net_income_row is not None and len(net_income_row) else None

    result["total_assets"] = total_assets
    result["net_income"] = net_income_latest
    result["roa"] = (net_income_latest / total_assets) if net_income_latest and total_assets else None

    revenue_latest = revenue_row.iloc[0] if revenue_row is not None and len(revenue_row) else None
    result["asset_turnover"] = (revenue_latest / total_assets) if revenue_latest and total_assets else None

    if current_assets_row is not None and current_liabilities_row is not None:
        ca = current_assets_row.iloc[0] if len(current_assets_row) else None
        cl = current_liabilities_row.iloc[0] if len(current_liabilities_row) else None
        result["current_ratio"] = (ca / cl) if ca and cl else None
    else:
        result["current_ratio"] = None

    # 成長率(直近期 vs 前期)
    rev_latest, rev_prev = _first_two(revenue_row)
    result["revenue_growth_1y"] = ((rev_latest - rev_prev) / abs(rev_prev)) if rev_prev else None

    op_latest, op_prev = _first_two(op_income_row)
    result["operating_income_growth_1y"] = ((op_latest - op_prev) / abs(op_prev)) if op_prev else None

    eps = info.get("trailingEps")
    shares = info.get("sharesOutstanding")
    eps_prev = None
    if net_income_row is not None and shares and len(net_income_row) >= 2:
        eps_prev = net_income_row.iloc[1] / shares
    result["eps_growth_1y"] = ((eps - eps_prev) / abs(eps_prev)) if eps and eps_prev else None

    # 3年CAGR(取得できた年数分。yfinanceは通常4〜5年分)
    years_available = len(revenue_row) if revenue_row is not None else 0
    result["growth_years_available"] = years_available
    if revenue_row is not None and years_available >= 2:
        n = min(years_available, 4) - 1  # 3年分=4期点、無ければ取得可能な期間で代替
        result["revenue_growth_3y_cagr"] = _cagr(revenue_row.iloc[0], revenue_row.iloc[n], n)
    else:
        result["revenue_growth_3y_cagr"] = None

    if net_income_row is not None and shares and years_available >= 2:
        n = min(years_available, 4) - 1
        eps_oldest = net_income_row.iloc[n] / shares
        result["eps_growth_3y_cagr"] = _cagr(eps, eps_oldest, n) if eps else None
    else:
        result["eps_growth_3y_cagr"] = None

    # 還元性
    market_cap = info.get("marketCap")
    result["market_cap"] = market_cap
    buyback_amount = abs(buyback_row.iloc[0]) if buyback_row is not None and len(buyback_row) else None
    result["buyback_amount"] = buyback_amount

    dividend_yield_pct = info.get("dividendYield")  # %表記(例: 2.89)
    div_total = (dividend_yield_pct / 100 * market_cap) if dividend_yield_pct and market_cap else None
    if market_cap and (div_total is not None or buyback_amount is not None):
        result["total_shareholder_return_yield"] = ((div_total or 0) + (buyback_amount or 0)) / market_cap
    else:
        result["total_shareholder_return_yield"] = None

    # 安全性の補強
    interest_bearing_debt = info.get("totalDebt")
    if interest_bearing_debt and market_cap:
        result["net_debt_to_ebitda"] = None  # EBITDA未算出のためNone(financialsにEBITDA行が無い銘柄が多い)
    else:
        result["net_debt_to_ebitda"] = None

    if op_income_row is not None and interest_expense_row is not None and len(op_income_row) and len(interest_expense_row):
        op0 = op_income_row.iloc[0]
        ie0 = interest_expense_row.iloc[0]
        result["interest_coverage_ratio"] = (op0 / abs(ie0)) if ie0 else None
    else:
        result["interest_coverage_ratio"] = None

    # 割安性: PSR(株価売上高比率)・PCFR(株価営業CF倍率)。PERが効かない赤字/成長企業も評価できる
    operating_cf_latest = operating_cf_row.iloc[0] if operating_cf_row is not None and len(operating_cf_row) else None
    result["operating_cashflow"] = operating_cf_latest
    result["psr"] = (market_cap / revenue_latest) if market_cap and revenue_latest else None
    result["pcfr"] = (market_cap / operating_cf_latest) if market_cap and operating_cf_latest and operating_cf_latest > 0 else None
    result["operating_cf_margin"] = (operating_cf_latest / revenue_latest) if operating_cf_latest and revenue_latest else None

    # 還元性: DOE(自己資本配当率) = 配当金総額 / 自己資本
    div_total_for_doe = div_total
    result["doe"] = (div_total_for_doe / equity) if div_total_for_doe and equity else None

    # 還元性: 配当成長率(直近の年間配当総額の前年比)
    divs = data.get("dividends")
    result["dividend_growth_1y"] = _dividend_growth_1y(divs)

    # 収益性・成長性: 経常利益(Pretax Incomeで代替)
    ordinary_income_latest, ordinary_income_prev = _first_two(ordinary_income_row)
    result["ordinary_income"] = ordinary_income_latest
    result["ordinary_income_margin"] = (ordinary_income_latest / revenue_latest) if ordinary_income_latest and revenue_latest else None
    result["ordinary_income_growth_1y"] = (
        (ordinary_income_latest - ordinary_income_prev) / abs(ordinary_income_prev) if ordinary_income_prev else None
    )

    return result


def _dividend_growth_1y(divs) -> float | None:
    """直近12ヶ月の配当総額と、その前の12ヶ月の配当総額を比較した成長率"""
    if divs is None or len(divs) < 2:
        return None
    import pandas as pd

    now = divs.index.max()
    last_12m = divs[divs.index > now - pd.DateOffset(months=12)].sum()
    prev_12m = divs[(divs.index <= now - pd.DateOffset(months=12)) & (divs.index > now - pd.DateOffset(months=24))].sum()
    if prev_12m == 0:
        return None
    return (last_12m - prev_12m) / prev_12m


def upsert_yearly_fundamentals(conn, ticker: str, data: dict) -> int:
    """複数年分の業績・財務・配当・CF推移をfundamentals_yearlyへ保存する(ダッシュボードの推移グラフ用)"""
    bs, fin, cf = data["balance_sheet"], data["financials"], data["cashflow"]
    if fin is None or fin.empty:
        return 0

    revenue_row = _get_row(fin, FINANCIALS_LABELS["revenue"])
    op_income_row = _get_row(fin, FINANCIALS_LABELS["operating_income"])
    net_income_row = _get_row(fin, FINANCIALS_LABELS["net_income"])
    ordinary_income_row = _get_row(fin, FINANCIALS_LABELS["ordinary_income"])
    total_assets_row = _get_row(bs, BALANCE_SHEET_LABELS["total_assets"])
    total_liabilities_row = _get_row(bs, BALANCE_SHEET_LABELS["total_liabilities"])
    equity_row = _get_row(bs, BALANCE_SHEET_LABELS["equity"])
    cash_row = _get_row(bs, BALANCE_SHEET_LABELS["cash"])
    operating_cf_row = _get_row(cf, CASHFLOW_LABELS["operating_cf"])
    investing_cf_row = _get_row(cf, CASHFLOW_LABELS["investing_cf"])
    financing_cf_row = _get_row(cf, CASHFLOW_LABELS["financing_cf"])
    free_cf_row = _get_row(cf, CASHFLOW_LABELS["free_cf"])
    dividends_paid_row = _get_row(cf, CASHFLOW_LABELS["dividends_paid"])

    shares = data["info"].get("sharesOutstanding")
    now_iso = datetime.now(JST).isoformat()
    rows_written = 0

    with conn:
        for col in fin.columns:
            fiscal_year_end = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)

            def val(row):
                return row.get(col) if row is not None and col in row.index else None

            revenue = val(revenue_row)
            net_income = val(net_income_row)
            op_income = val(op_income_row)
            equity = val(equity_row)
            total_assets = val(total_assets_row)
            # 発行済株式数は直近の値で近似(過去の実株数はyfinanceから取得できないため)
            eps_year = (net_income / shares) if net_income and shares else None
            dps_year = abs(val(dividends_paid_row) / shares) if val(dividends_paid_row) and shares else None
            payout_ratio_year = (dps_year / eps_year) if dps_year and eps_year and eps_year > 0 else None

            conn.execute(
                """
                INSERT INTO fundamentals_yearly
                    (ticker, fiscal_year_end, revenue, operating_income, ordinary_income, net_income,
                     operating_margin, net_margin, eps, dividend_per_share, payout_ratio,
                     total_assets, total_liabilities, equity, equity_ratio,
                     operating_cf, investing_cf, financing_cf, free_cf, cash_and_equivalents, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ticker, fiscal_year_end) DO UPDATE SET
                    revenue=excluded.revenue, operating_income=excluded.operating_income,
                    ordinary_income=excluded.ordinary_income, net_income=excluded.net_income,
                    operating_margin=excluded.operating_margin, net_margin=excluded.net_margin,
                    eps=excluded.eps, dividend_per_share=excluded.dividend_per_share,
                    payout_ratio=excluded.payout_ratio, total_assets=excluded.total_assets,
                    total_liabilities=excluded.total_liabilities, equity=excluded.equity,
                    equity_ratio=excluded.equity_ratio, operating_cf=excluded.operating_cf,
                    investing_cf=excluded.investing_cf, financing_cf=excluded.financing_cf,
                    free_cf=excluded.free_cf, cash_and_equivalents=excluded.cash_and_equivalents,
                    updated_at=excluded.updated_at
                """,
                (
                    ticker, fiscal_year_end, revenue, op_income, val(ordinary_income_row), net_income,
                    (op_income / revenue) if op_income and revenue else None,
                    (net_income / revenue) if net_income and revenue else None,
                    eps_year, dps_year, payout_ratio_year,
                    total_assets, val(total_liabilities_row), equity,
                    (equity / total_assets * 100) if equity and total_assets else None,
                    val(operating_cf_row), val(investing_cf_row), val(financing_cf_row),
                    val(free_cf_row), val(cash_row), now_iso,
                ),
            )
            rows_written += 1
    return rows_written


def upsert_fundamentals(conn, ticker: str, snapshot_date: str, info: dict, extra: dict, bs, fin, divs) -> None:
    equity_row = _get_row(bs, BALANCE_SHEET_LABELS["equity"])
    equity = equity_row.iloc[0] if equity_row is not None and len(equity_row) else None
    total_assets = extra.get("total_assets")
    equity_ratio = (equity / total_assets * 100) if equity and total_assets else None

    row = {
        "ticker": ticker,
        "snapshot_date": snapshot_date,
        "per": info.get("trailingPE"),
        "pbr": info.get("priceToBook"),
        "roe": info.get("returnOnEquity"),
        "dividend_yield": info.get("dividendYield"),
        "payout_ratio": info.get("payoutRatio"),
        "interest_bearing_debt": info.get("totalDebt"),
        "avg_volume": info.get("averageVolume"),
        "eps": info.get("trailingEps"),
        "revenue": info.get("totalRevenue"),
        "operating_margin": info.get("operatingMargins"),
        "net_margin": info.get("profitMargins"),
        "equity_ratio": equity_ratio,
        "earnings_years": len(fin.columns) if fin is not None and not fin.empty else 0,
        "dividend_history_count": len(divs) if divs is not None else 0,
        **extra,
    }
    cols = list(row.keys())
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in ("ticker", "snapshot_date"))
    with conn:
        conn.execute(
            f"""
            INSERT INTO fundamentals_weekly ({", ".join(cols)}) VALUES ({placeholders})
            ON CONFLICT(ticker, snapshot_date) DO UPDATE SET {updates}
            """,
            [row[c] for c in cols],
        )


def log_result(conn, run_date: str, ticker: str, status: str, error_message: str = "") -> None:
    with conn:
        conn.execute(
            "INSERT INTO fetch_log (run_date, job_type, ticker, status, error_message) VALUES (?, ?, ?, ?, ?)",
            (run_date, "fundamentals", ticker, status, error_message),
        )


def load_rolling_targets(conn, limit: int | None) -> list[str]:
    # last_updatedが古い順(未取得はNULLとして最優先)にfundamentals_weeklyの更新対象を選ぶ
    query = """
    SELECT t.ticker
    FROM tickers t
    LEFT JOIN (
        SELECT ticker, MAX(snapshot_date) AS last_snapshot FROM fundamentals_weekly GROUP BY ticker
    ) f ON t.ticker = f.ticker
    WHERE t.is_active = 1
    ORDER BY f.last_snapshot IS NOT NULL, f.last_snapshot ASC
    """
    tickers = [r[0] for r in conn.execute(query)]
    return tickers[:limit] if limit else tickers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="処理する銘柄数の上限(検証用)")
    parser.add_argument("--time-budget-sec", type=int, default=DEFAULT_TIME_BUDGET_SEC)
    args = parser.parse_args()

    conn = get_connection()
    run_date = datetime.now(JST).strftime("%Y-%m-%d")
    guard = ConsecutiveFailureGuard()

    targets = load_rolling_targets(conn, args.limit)
    start = time.monotonic()
    success, failed = 0, 0

    for i, ticker in enumerate(targets, start=1):
        if time.monotonic() - start > args.time_budget_sec:
            print(f"時間予算({args.time_budget_sec}秒)に到達。{i - 1}/{len(targets)}件処理して終了")
            break
        try:
            data = fetch_all(ticker)
            extra = compute_extra_fields(data)
            upsert_fundamentals(conn, ticker, run_date, data["info"], extra, data["balance_sheet"], data["financials"], data["dividends"])
            upsert_yearly_fundamentals(conn, ticker, data)
            log_result(conn, run_date, ticker, "success")
            guard.record_success()
            success += 1
        except Exception as e:
            log_result(conn, run_date, ticker, "failed", str(e))
            failed += 1
            try:
                guard.record_failure()
            except TooManyFailuresError as blocked:
                print(f"停止: {blocked}")
                break
        if i % 20 == 0:
            print(f"[{i}/{len(targets)}] success={success} failed={failed}")

    conn.close()
    print(f"完了: success={success}, failed={failed}")


if __name__ == "__main__":
    main()
