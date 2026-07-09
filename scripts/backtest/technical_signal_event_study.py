"""GC/RSI反発/MACDクロス/BB反発/出来高急増を、v1(decide_rule.py)のように合算せず
1シグナルずつ独立に、間引きなしの毎日粒度・複数保有期間でイベントスタディする。

decision_engine_extended_backtest.pyでのv1検証は、v2に合わせて5営業日おきにしか
判定していなかったため、GC等の単発イベントシグナルの大半を取りこぼしていた
(クロスの起きた日がサンプリング日と一致しないと検知できない)。また4シグナルを
「何個点灯したか」で合算していたため、個別の予測力を検証できていなかった。

本スクリプトはcalc_indicators()の出力(MA25/MA75/RSI/MACD/BB/出来高移動平均)を
1銘柄につき1回ベクトル化して計算し、shift比較でクロス判定を全期間分一括抽出する
(1日ずつループしない)ため、5年分・全銘柄でも高速に毎日粒度で検証できる。

各シグナル発生日について、5・10・21・63営業日後のリターンを計算し、
その銘柄・その日を含む「その他すべての日」のリターン分布(ベースライン)と比較する。

実行: python scripts/backtest/technical_signal_event_study.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_backtest_connection  # noqa: E402
from common.technical import calc_indicators  # noqa: E402

HORIZONS = {
    "5営業日": 5, "10営業日": 10, "21営業日": 21, "63営業日": 63,
    "126営業日(6ヶ月)": 126, "252営業日(12ヶ月)": 252, "504営業日(24ヶ月)": 504,
}
NEUTRAL_THRESHOLD_PCT = 2.0
MIN_ROWS = 100


def load_all_tickers(bt_conn) -> list[str]:
    return [r[0] for r in bt_conn.execute("SELECT DISTINCT ticker FROM price_history ORDER BY ticker")]


def load_ticker_df(bt_conn, ticker: str) -> pd.DataFrame:
    query = "SELECT date, open, high, low, close, volume FROM price_history WHERE ticker = ? ORDER BY date ASC"
    df = pd.read_sql_query(query, bt_conn, params=(ticker,), parse_dates=["date"])
    if df.empty:
        return df
    return df.set_index("date").rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )


def detect_signals(df: pd.DataFrame) -> pd.DataFrame:
    """calc_indicators適用済みのdfから、5シグナルの発生日(bool)を一括抽出する"""
    prev = df.shift(1)
    signals = pd.DataFrame(index=df.index)
    signals["GC"] = (prev["MA25"] <= prev["MA75"]) & (df["MA25"] > df["MA75"])
    signals["RSI"] = (prev["RSI"] < 30) & (df["RSI"] >= 30)
    signals["MACD"] = (prev["MACD"] <= prev["MACD_signal"]) & (df["MACD"] > df["MACD_signal"])
    signals["BB"] = (prev["Close"] <= prev["BB_lower"]) & (df["Close"] > df["BB_lower"])
    signals["VOL"] = df["Volume"] > df["Vol_MA20"] * 2
    return signals


def main():
    start = time.monotonic()
    bt_conn = get_backtest_connection()
    tickers = load_all_tickers(bt_conn)
    print(f"対象銘柄数: {len(tickers)}")

    signal_events: dict[tuple[str, str], list[float]] = {
        (sig, h): [] for sig in ("GC", "RSI", "MACD", "BB", "VOL") for h in HORIZONS
    }
    baseline_returns: dict[str, list[float]] = {h: [] for h in HORIZONS}
    corr_frames = []

    for n, ticker in enumerate(tickers, start=1):
        df = load_ticker_df(bt_conn, ticker)
        if len(df) < MIN_ROWS:
            continue
        df = calc_indicators(df.copy())
        signals = detect_signals(df)
        close = df["Close"]

        fwd_returns = {}
        for h_label, h in HORIZONS.items():
            fwd_returns[h_label] = (close.shift(-h) / close - 1) * 100

        for sig in ("GC", "RSI", "MACD", "BB", "VOL"):
            idx = signals.index[signals[sig].fillna(False)]
            for h_label in HORIZONS:
                vals = fwd_returns[h_label].loc[idx].dropna()
                signal_events[(sig, h_label)].extend(vals.tolist())

        # ベースライン(全営業日、シグナル有無を問わない)。サンプルが巨大になりすぎないよう
        # 銘柄ごとに20営業日おきに間引いて集める(母集団としては十分)
        for h_label in HORIZONS:
            baseline_returns[h_label].extend(fwd_returns[h_label].iloc[::20].dropna().tolist())

        corr_frames.append(signals[["GC", "RSI", "MACD", "BB", "VOL"]].astype(float))

        if n % 500 == 0:
            print(f"[{n}/{len(tickers)}] 経過{time.monotonic()-start:.0f}秒")

    print(f"\n全{len(tickers)}銘柄の指標計算・シグナル抽出完了(経過{time.monotonic()-start:.0f}秒)\n")

    print("=== シグナル別・保有期間別のフォワードリターン(個別、間引きなし) ===")
    for sig in ("GC", "RSI", "MACD", "BB", "VOL"):
        print(f"\n--- {sig} ---")
        for h_label in HORIZONS:
            vals = pd.Series(signal_events[(sig, h_label)])
            base = pd.Series(baseline_returns[h_label])
            if len(vals) < 30:
                print(f"  {h_label}: サンプル不足(n={len(vals)})")
                continue
            win_rate = (vals > NEUTRAL_THRESHOLD_PCT).mean()
            base_mean = base.mean()
            print(
                f"  {h_label}: n={len(vals)}  平均={vals.mean():+.2f}%(ベースライン{base_mean:+.2f}%比 "
                f"{vals.mean()-base_mean:+.2f}pt)  中央値={vals.median():+.2f}%  勝率(>+2%)={win_rate:.1%}"
            )

    print("\n=== シグナル間の相関(高すぎると独立した根拠になっていない) ===")
    corr_all = pd.concat(corr_frames, ignore_index=True)
    print(corr_all.corr().round(3))

    bt_conn.close()


if __name__ == "__main__":
    main()
