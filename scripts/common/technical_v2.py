"""テクニカル判断エンジンv2: 相関の低い複数シグナルファミリーを連続値(-1.0〜+1.0)で返す。

既存のtechnical.py(v1: GC/RSI/MACD/BB/出来高の二値シグナル)は変更せず残す。
v2は「強さ」を連続値で表現し、順張り(trend)と逆張り(mean_reversion)を
レジーム判定で使い分けることで、v1にあった以下の問題を解消する狙い:
  - GC・MACDクロスが遅行・順張りで相関が高く独立した根拠になっていない
  - RSI反発が単純閾値のみでダイバージェンスを見ていない
  - BB反発(逆張り)とGC/MACD(順張り)を同列の「シグナル数」として合算している
  - 出来高急増が価格方向と紐付いていない

このモジュールは価格データ(DataFrame)のみを入力とする純粋な計算関数群で、
DBアクセスを伴う市場・セクターレジームはcommon/regime.pyに分離する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta
from scipy.signal import argrelextrema


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def calc_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCVから、v2で使う指標一式を計算して列を追加した新しいDataFrameを返す"""
    df = df.copy()
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    df["MA5"] = ta.trend.sma_indicator(close, window=5)
    df["MA25"] = ta.trend.sma_indicator(close, window=25)
    df["MA75"] = ta.trend.sma_indicator(close, window=75)

    df["RSI"] = ta.momentum.rsi(close, window=14)

    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close, window=20)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    bb_range = (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
    df["percent_b"] = (close - df["BB_lower"]) / bb_range
    df["BB_width"] = bb_range / bb.bollinger_mavg()

    adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
    df["ADX"] = adx_ind.adx()
    df["PLUS_DI"] = adx_ind.adx_pos()
    df["MINUS_DI"] = adx_ind.adx_neg()

    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["Vol_MA20"] = volume.rolling(20).mean()
    df["ATR"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    return df


# --- trend family ---

def trend_alignment_score(ma_short: float, ma_mid: float, ma_long: float) -> float:
    """MA5/25/75の並び順で順張りの地合いを判定する。GC単体より情報量が多い"""
    if pd.isna(ma_short) or pd.isna(ma_mid) or pd.isna(ma_long):
        return 0.0
    if ma_short > ma_mid > ma_long:
        return 1.0
    if ma_long > ma_mid > ma_short:
        return -1.0
    if ma_short > ma_mid and ma_mid <= ma_long:
        return 0.3  # 短期先行の上昇転換初期
    if ma_short <= ma_mid and ma_mid > ma_long:
        return -0.3  # 短期先行の下降転換初期
    if ma_short > ma_long:
        return 0.15
    if ma_short < ma_long:
        return -0.15
    return 0.0


def ma_slope_score(ma_series: pd.Series, window: int = 5, scale: float = 0.05) -> float:
    """移動平均の傾き(直近window日の変化率)をtanhでスケーリングし連続値化する"""
    valid = ma_series.dropna()
    if len(valid) < window + 1:
        return 0.0
    change = (valid.iloc[-1] - valid.iloc[-window - 1]) / valid.iloc[-window - 1]
    return float(np.tanh(change / scale))


def adx_directional_score(adx: float, plus_di: float, minus_di: float, cap: float = 40.0) -> float:
    """ADXでトレンド強度、+DI/-DIで方向を判定する"""
    if pd.isna(adx) or pd.isna(plus_di) or pd.isna(minus_di):
        return 0.0
    direction = 1.0 if plus_di > minus_di else -1.0
    strength = min(adx / cap, 1.0)
    return direction * strength


def macd_histogram_accel_score(hist_series: pd.Series, scale: float = 0.5) -> float:
    """MACDヒストグラムの2階差分(加速度)。縮小→拡大への転換を捉える"""
    valid = hist_series.dropna()
    if len(valid) < 3:
        return 0.0
    accel = (valid.iloc[-1] - valid.iloc[-2]) - (valid.iloc[-2] - valid.iloc[-3])
    return float(np.tanh(accel / scale)) if scale else 0.0


def trend_family_score(row: pd.Series, df: pd.DataFrame, weights: dict | None = None) -> float:
    weights = weights or {"alignment": 0.35, "slope": 0.25, "adx": 0.25, "macd_accel": 0.15}
    alignment = trend_alignment_score(row.get("MA5"), row.get("MA25"), row.get("MA75"))
    slope = ma_slope_score(df["MA25"])
    adx = adx_directional_score(row.get("ADX"), row.get("PLUS_DI"), row.get("MINUS_DI"))
    macd_accel = macd_histogram_accel_score(df["MACD_hist"])
    return _clip(
        alignment * weights["alignment"] + slope * weights["slope"]
        + adx * weights["adx"] + macd_accel * weights["macd_accel"]
    )


# --- mean_reversion family ---

def rsi_divergence_score(close: pd.Series, rsi: pd.Series, lookback: int = 20, order: int = 3) -> float:
    """直近lookback日で価格安値切り下げ+RSI切り上げ(強気ダイバージェンス)、逆は弱気を検出する"""
    c = close.tail(lookback).reset_index(drop=True)
    r = rsi.tail(lookback).reset_index(drop=True)
    if len(c) < lookback or c.isna().any() or r.isna().any():
        return 0.0

    lows_idx = argrelextrema(c.values, np.less_equal, order=order)[0]
    highs_idx = argrelextrema(c.values, np.greater_equal, order=order)[0]

    score = 0.0
    if len(lows_idx) >= 2:
        p1, p2 = lows_idx[-2], lows_idx[-1]
        if c.iloc[p2] < c.iloc[p1] and r.iloc[p2] > r.iloc[p1]:
            score += min((r.iloc[p2] - r.iloc[p1]) / 30.0, 1.0)  # 強気ダイバージェンス
    if len(highs_idx) >= 2:
        p1, p2 = highs_idx[-2], highs_idx[-1]
        if c.iloc[p2] > c.iloc[p1] and r.iloc[p2] < r.iloc[p1]:
            score -= min((r.iloc[p1] - r.iloc[p2]) / 30.0, 1.0)  # 弱気ダイバージェンス
    return _clip(score)


def percent_b_score(percent_b: float) -> float:
    """%B=0.5(バンド中心)を0、下抜けほど+(買い場)、上抜けほど-(過熱)に連続値化"""
    if pd.isna(percent_b):
        return 0.0
    return _clip((0.5 - percent_b) * 2)


def rsi_level_velocity_score(rsi_series: pd.Series, window: int = 5) -> float:
    """RSI水準が低いほど、かつ反転速度が速いほどスコアが高い"""
    valid = rsi_series.dropna()
    if len(valid) < window + 1:
        return 0.0
    level = valid.iloc[-1]
    velocity = valid.iloc[-1] - valid.iloc[-window - 1]
    level_component = _clip((50 - level) / 50)  # RSI0で+1, RSI100で-1
    velocity_component = _clip(velocity / 20)
    return _clip(level_component * 0.6 + velocity_component * 0.4)


def mean_reversion_family_score(row: pd.Series, df: pd.DataFrame, weights: dict | None = None) -> float:
    weights = weights or {"divergence": 0.40, "percent_b": 0.35, "rsi_velocity": 0.25}
    divergence = rsi_divergence_score(df["Close"], df["RSI"])
    pb = percent_b_score(row.get("percent_b"))
    velocity = rsi_level_velocity_score(df["RSI"])
    return _clip(
        divergence * weights["divergence"] + pb * weights["percent_b"] + velocity * weights["rsi_velocity"]
    )


# --- volume family ---

def directional_volume_score(volume: float, vol_ma20: float, price_change_pct: float, surge_cap: float = 3.0) -> float:
    """出来高急増を、当日の値上がり/値下がり方向と紐付けて評価する(v1の欠陥修正)"""
    if pd.isna(volume) or pd.isna(vol_ma20) or vol_ma20 == 0 or pd.isna(price_change_pct):
        return 0.0
    surge_ratio = volume / vol_ma20
    surge_component = _clip((surge_ratio - 1.0) / (surge_cap - 1.0), 0.0, 1.0)
    direction = 1.0 if price_change_pct > 0 else (-1.0 if price_change_pct < 0 else 0.0)
    return direction * surge_component


def obv_divergence_score(obv_series: pd.Series, close_series: pd.Series, window: int = 10) -> float:
    """OBVの傾きが価格の傾きと逆の時だけ非ゼロを返す(trend系との相関を意図的に排除)"""
    obv_valid, close_valid = obv_series.dropna(), close_series.dropna()
    if len(obv_valid) < window + 1 or len(close_valid) < window + 1:
        return 0.0
    obv_slope = (obv_valid.iloc[-1] - obv_valid.iloc[-window - 1])
    obv_scale = obv_valid.tail(window + 1).abs().max() or 1.0
    obv_slope_norm = obv_slope / obv_scale

    price_slope = (close_valid.iloc[-1] - close_valid.iloc[-window - 1]) / close_valid.iloc[-window - 1]

    if obv_slope_norm > 0 and price_slope <= 0:
        return _clip(obv_slope_norm * 3)
    if obv_slope_norm < 0 and price_slope >= 0:
        return _clip(obv_slope_norm * 3)
    return 0.0  # 同方向はtrend系と重複するため0(独立性を保つ)


def volume_family_score(row: pd.Series, df: pd.DataFrame, weights: dict | None = None) -> float:
    weights = weights or {"directional_surge": 0.45, "obv_divergence": 0.55}
    close = df["Close"]
    if len(close) < 2 or pd.isna(close.iloc[-2]) or close.iloc[-2] == 0:
        price_change_pct = 0.0
    else:
        price_change_pct = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
    surge = directional_volume_score(row.get("Volume"), row.get("Vol_MA20"), price_change_pct)
    obv_div = obv_divergence_score(df["OBV"], df["Close"])
    return _clip(surge * weights["directional_surge"] + obv_div * weights["obv_divergence"])


# --- volatility regime (フィルター、シグナルスコアではない) ---

def volatility_regime(adx: float, bb_width_percentile: float | None,
                       trend_threshold: float = 25.0, range_threshold: float = 20.0) -> dict:
    if pd.isna(adx):
        regime, mr_mult, trend_mult = "unknown", 1.0, 1.0
    elif adx > trend_threshold:
        regime, mr_mult, trend_mult = "trend", 0.3, 1.3
    elif adx < range_threshold:
        regime, mr_mult, trend_mult = "range", 1.3, 0.3
    else:
        regime, mr_mult, trend_mult = "transition", 1.0, 1.0

    squeeze = bool(bb_width_percentile is not None and bb_width_percentile < 0.2)
    return {
        "regime": regime,
        "bb_squeeze": squeeze,
        "mean_reversion_weight_multiplier": mr_mult,
        "trend_weight_multiplier": trend_mult,
    }


def bb_width_percentile(bb_width_series: pd.Series, window: int = 60) -> float | None:
    valid = bb_width_series.dropna()
    if len(valid) < window:
        return None
    recent = valid.tail(window)
    return float((recent < recent.iloc[-1]).mean())


def compute_technical_v2(df: pd.DataFrame) -> dict:
    """OHLCVのDataFrame(Open/High/Low/Close/Volume列、直近日が末尾)を受け取り、
    各ファミリーのスコアとレジーム情報をまとめて返す。データ不足時は0.0/Noneで安全に返す。
    """
    if df.empty or len(df) < 30:
        return {
            "trend_score": 0.0, "mean_reversion_score": 0.0, "volume_score": 0.0,
            "regime": {"regime": "unknown", "bb_squeeze": False,
                       "mean_reversion_weight_multiplier": 1.0, "trend_weight_multiplier": 1.0},
            "atr": None, "close": None,
        }

    df = calc_all_indicators(df)
    row = df.iloc[-1]

    regime = volatility_regime(row.get("ADX"), bb_width_percentile(df["BB_width"]))

    trend = trend_family_score(row, df)
    mean_rev = mean_reversion_family_score(row, df)
    volume = volume_family_score(row, df)

    return {
        "trend_score": trend,
        "mean_reversion_score": mean_rev,
        "volume_score": volume,
        "regime": regime,
        "atr": float(row["ATR"]) if pd.notna(row.get("ATR")) else None,
        "close": float(row["Close"]) if pd.notna(row.get("Close")) else None,
    }
