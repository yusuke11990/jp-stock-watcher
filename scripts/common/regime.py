"""市場(TOPIX)・セクター全体のレジーム判定。DBアクセスを伴うためtechnical_v2.pyから分離する。"""

from __future__ import annotations

import pandas as pd

from common.technical_v2 import calc_all_indicators, trend_alignment_score

TOPIX_TICKER = "1306.T"
MIN_SECTOR_SAMPLE_SIZE = 5


def load_price_df(conn, ticker: str, as_of_date: str | None = None, min_rows: int = 80) -> pd.DataFrame:
    # as_of_dateを指定しないと、過去のsnapshot_dateでバックフィルした際に決定時点では
    # まだ存在しなかった未来の株価まで見てレジーム判定してしまう
    if as_of_date is not None:
        query = """
        SELECT date, open, high, low, close, volume FROM price_daily
        WHERE ticker = ? AND date <= ? ORDER BY date ASC
        """
        df = pd.read_sql_query(query, conn, params=(ticker, as_of_date), parse_dates=["date"])
    else:
        query = """
        SELECT date, open, high, low, close, volume FROM price_daily
        WHERE ticker = ? ORDER BY date ASC
        """
        df = pd.read_sql_query(query, conn, params=(ticker,), parse_dates=["date"])
    if len(df) < min_rows:
        return pd.DataFrame()
    df = df.set_index("date").rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    return df


def market_regime_score(conn, as_of_date: str | None = None) -> dict:
    """TOPIX ETF(1306.T)のトレンドスコアを個別銘柄シグナルの倍率として使う"""
    df = load_price_df(conn, TOPIX_TICKER, as_of_date)
    if df.empty:
        return {"topix_trend_score": 0.0, "market_regime_multiplier": 1.0}

    df = calc_all_indicators(df)
    row = df.iloc[-1]
    trend_score = trend_alignment_score(row.get("MA5"), row.get("MA25"), row.get("MA75"))
    return {
        "topix_trend_score": trend_score,
        "market_regime_multiplier": 1.0 + 0.3 * trend_score,  # 0.7~1.3
    }


def sector_regime_score(conn, sector: str, decision_date: str, lookback_days: int = 5) -> dict:
    """業種内銘柄のlookback_days騰落率中央値でセクター全体の強弱を判定する"""
    query = """
    SELECT t.ticker
    FROM tickers t
    WHERE t.sector = ? AND t.is_active = 1
    """
    tickers = [r[0] for r in conn.execute(query, (sector,))]
    if len(tickers) < MIN_SECTOR_SAMPLE_SIZE:
        return {"sector_return_median": None, "sector_regime_score": 0.0, "sample_size": len(tickers)}

    placeholders = ",".join("?" for _ in tickers)
    price_query = f"""
    SELECT ticker, date, close FROM price_daily
    WHERE ticker IN ({placeholders}) AND date <= ?
    ORDER BY ticker, date ASC
    """
    df = pd.read_sql_query(price_query, conn, params=(*tickers, decision_date))
    if df.empty:
        return {"sector_return_median": None, "sector_regime_score": 0.0, "sample_size": len(tickers)}

    returns = []
    for _, g in df.groupby("ticker"):
        g = g.tail(lookback_days + 1)
        if len(g) >= 2 and g["close"].iloc[0] > 0:
            returns.append((g["close"].iloc[-1] - g["close"].iloc[0]) / g["close"].iloc[0])

    if not returns:
        return {"sector_return_median": None, "sector_regime_score": 0.0, "sample_size": len(tickers)}

    median_return = float(pd.Series(returns).median())
    score = max(-1.0, min(1.0, median_return * 10))  # ±10%騰落で±1に達する簡易スケーリング
    return {"sector_return_median": median_return, "sector_regime_score": score, "sample_size": len(tickers)}
