"""テクニカルシグナル判定。

`アプリ開発/株式投資ツール/stock_tool.py`のcalc_indicators/check_signalsを
Streamlit依存なしの純粋な計算関数として移植したもの。
ロジックは移植元と同一(MA25/MA75, RSI(14), MACD, ボリンジャーバンド(20,2σ), 出来高移動平均(20))。
"""

from __future__ import annotations

import pandas as pd
import ta


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    volume = df["Volume"]

    df["MA25"] = ta.trend.sma_indicator(close, window=25)
    df["MA75"] = ta.trend.sma_indicator(close, window=75)

    df["RSI"] = ta.momentum.rsi(close, window=14)

    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(close, window=20)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_mid"] = bb.bollinger_mavg()

    df["Vol_MA20"] = volume.rolling(20).mean()

    return df


def check_signals(df: pd.DataFrame) -> list[dict]:
    if df.empty or len(df) < 75:
        return []

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []

    if prev["MA25"] <= prev["MA75"] and latest["MA25"] > latest["MA75"]:
        signals.append({"type": "GC", "label": "ゴールデンクロス", "strength": "強", "emoji": "🟡"})

    if prev["RSI"] < 30 and latest["RSI"] >= 30:
        signals.append({"type": "RSI", "label": "RSI売られすぎ反発", "strength": "中", "emoji": "🔵"})

    if prev["MACD"] <= prev["MACD_signal"] and latest["MACD"] > latest["MACD_signal"]:
        signals.append({"type": "MACD", "label": "MACDクロス", "strength": "中", "emoji": "🟢"})

    if prev["Close"] <= prev["BB_lower"] and latest["Close"] > latest["BB_lower"]:
        signals.append({"type": "BB", "label": "ボリンジャー-2σ反発", "strength": "中", "emoji": "🟠"})

    if latest["Volume"] > latest["Vol_MA20"] * 2:
        signals.append({"type": "VOL", "label": "出来高急増", "strength": "参考", "emoji": "⚡"})

    return signals


NEAR_52W_HIGH_WINDOW = 252  # 約1年分の営業日
NEAR_52W_HIGH_THRESHOLD = 0.95  # 52週高値の何%まで接近したら「近い」とみなすか


def check_near_52_week_high(df: pd.DataFrame) -> bool:
    """株価が52週高値の95%圏内に、今日初めて入った(=前日は入っていなかった)かを判定する。

    backtest/alt_signals_check.pyでの検証結果: グレードB/C×割安性上位50%と組み合わせた場合、
    GC(ゴールデンクロス)より24ヶ月保有で明確に優位(+8pt超、2022-2024年の全年で頑健)。
    行動ファイナンスで知られる「52週高値モメンタム」(George & Hwang, 2004)と整合する。
    """
    if df.empty or len(df) < NEAR_52W_HIGH_WINDOW // 2:
        return False
    rolling_high = df["Close"].rolling(NEAR_52W_HIGH_WINDOW, min_periods=NEAR_52W_HIGH_WINDOW // 2).max()
    near_high = df["Close"] >= rolling_high * NEAR_52W_HIGH_THRESHOLD
    if len(near_high) < 2 or pd.isna(near_high.iloc[-1]):
        return False
    return bool(near_high.iloc[-1] and not near_high.iloc[-2])


def get_technical_state(df: pd.DataFrame) -> dict:
    """decide_rule.pyでMA25/MA75割れ判定に使う補助情報"""
    if df.empty or len(df) < 75:
        return {}
    latest = df.iloc[-1]
    return {
        "current_price": float(latest["Close"]),
        "price_below_ma25": bool(pd.notna(latest["MA25"]) and latest["Close"] < latest["MA25"]),
        "price_below_ma75": bool(pd.notna(latest["MA75"]) and latest["Close"] < latest["MA75"]),
    }
