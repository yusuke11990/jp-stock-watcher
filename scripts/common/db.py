"""SQLite接続とスキーマ初期化"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "stock.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickers (
    ticker TEXT PRIMARY KEY,
    code TEXT NOT NULL,
    name TEXT,
    market TEXT,
    sector TEXT,
    is_active INTEGER DEFAULT 1,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS price_daily (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS fundamentals_weekly (
    ticker TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    per REAL,
    pbr REAL,
    roe REAL,
    dividend_yield REAL,
    payout_ratio REAL,
    interest_bearing_debt REAL,
    avg_volume REAL,
    eps REAL,
    revenue REAL,
    operating_margin REAL,
    net_margin REAL,
    equity_ratio REAL,
    earnings_years INTEGER,
    dividend_history_count INTEGER,
    PRIMARY KEY (ticker, snapshot_date)
);

CREATE TABLE IF NOT EXISTS fetch_log (
    run_date TEXT,
    job_type TEXT,
    ticker TEXT,
    status TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS alert_history (
    ticker TEXT,
    alert_type TEXT,
    triggered_date TEXT,
    detail TEXT,
    PRIMARY KEY (ticker, alert_type, triggered_date)
);

CREATE TABLE IF NOT EXISTS scores (
    ticker TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    sector TEXT NOT NULL,
    score_safety REAL,
    score_growth REAL,
    score_profitability REAL,
    score_efficiency REAL,
    score_valuation REAL,
    score_shareholder_return REAL,
    confidence_safety REAL,
    confidence_growth REAL,
    confidence_profitability REAL,
    confidence_efficiency REAL,
    confidence_valuation REAL,
    confidence_shareholder_return REAL,
    total_score REAL,
    grade TEXT,
    sector_median_score REAL,
    sector_rank INTEGER,
    sector_size INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (ticker, snapshot_date)
);
CREATE INDEX IF NOT EXISTS idx_scores_sector_snapshot ON scores(sector, snapshot_date);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    decision_date TEXT NOT NULL,
    action TEXT NOT NULL,
    decision_source TEXT NOT NULL,
    rule_version TEXT,
    total_score REAL,
    grade TEXT,
    technical_signals TEXT,
    reason TEXT,
    price_at_decision REAL,
    confidence REAL,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(ticker, decision_date, decision_source)
);
CREATE INDEX IF NOT EXISTS idx_decisions_ticker ON decisions(ticker, decision_date);

CREATE TABLE IF NOT EXISTS decision_outcomes (
    decision_id INTEGER NOT NULL PRIMARY KEY,
    eval_date TEXT NOT NULL,
    price_at_eval REAL NOT NULL,
    return_pct REAL NOT NULL,
    is_correct INTEGER,
    outcome_label TEXT,
    benchmark_return_pct REAL,
    excess_return_pct REAL,
    evaluated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (decision_id) REFERENCES decisions(decision_id)
);

CREATE TABLE IF NOT EXISTS fundamentals_yearly (
    ticker TEXT NOT NULL,
    fiscal_year_end TEXT NOT NULL,
    revenue REAL,
    operating_income REAL,
    ordinary_income REAL,
    net_income REAL,
    operating_margin REAL,
    net_margin REAL,
    eps REAL,
    dividend_per_share REAL,
    payout_ratio REAL,
    total_assets REAL,
    total_liabilities REAL,
    equity REAL,
    equity_ratio REAL,
    operating_cf REAL,
    investing_cf REAL,
    financing_cf REAL,
    free_cf REAL,
    cash_and_equivalents REAL,
    buyback_amount REAL,
    updated_at TEXT,
    PRIMARY KEY (ticker, fiscal_year_end)
);
CREATE INDEX IF NOT EXISTS idx_fundamentals_yearly_ticker ON fundamentals_yearly(ticker, fiscal_year_end);

CREATE TABLE IF NOT EXISTS edinet_documents (
    doc_id TEXT PRIMARY KEY,
    sec_code TEXT,
    doc_type_code TEXT,
    period_end TEXT,
    submit_date_time TEXT,
    fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_edinet_documents_sec_code ON edinet_documents(sec_code, period_end);

CREATE TABLE IF NOT EXISTS edinet_scanned_dates (
    scan_date TEXT PRIMARY KEY,
    doc_count INTEGER,
    scanned_at TEXT
);
"""

# fundamentals_weeklyへ追加するカラム(6軸スコアリング用)。既存DBには起動時にマイグレーションする。
FUNDAMENTALS_EXTRA_COLUMNS = {
    "roa": "REAL",
    "asset_turnover": "REAL",
    "total_assets": "REAL",
    "net_income": "REAL",
    "revenue_growth_1y": "REAL",
    "revenue_growth_3y_cagr": "REAL",
    "operating_income_growth_1y": "REAL",
    "eps_growth_1y": "REAL",
    "eps_growth_3y_cagr": "REAL",
    "growth_years_available": "INTEGER",
    "buyback_amount": "REAL",
    "total_shareholder_return_yield": "REAL",
    "market_cap": "REAL",
    "current_ratio": "REAL",
    "net_debt_to_ebitda": "REAL",
    "interest_coverage_ratio": "REAL",
    "psr": "REAL",
    "pcfr": "REAL",
    "operating_cashflow": "REAL",
    "doe": "REAL",
    "dividend_growth_1y": "REAL",
    "ordinary_income": "REAL",
    "ordinary_income_margin": "REAL",
    "ordinary_income_growth_1y": "REAL",
    "operating_cf_margin": "REAL",
}


def _migrate_fundamentals_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(fundamentals_weekly)")}
    for col, coltype in FUNDAMENTALS_EXTRA_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE fundamentals_weekly ADD COLUMN {col} {coltype}")

    existing_yearly = {row[1] for row in conn.execute("PRAGMA table_info(fundamentals_yearly)")}
    if "buyback_amount" not in existing_yearly:
        conn.execute("ALTER TABLE fundamentals_yearly ADD COLUMN buyback_amount REAL")


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    _migrate_fundamentals_columns(conn)
    conn.commit()
    return conn


if __name__ == "__main__":
    conn = get_connection()
    conn.close()
    print(f"DB initialized at {DB_PATH}")
