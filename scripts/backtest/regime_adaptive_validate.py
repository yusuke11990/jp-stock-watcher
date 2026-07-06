"""静的な1つの重みではなく、ADXレジーム(トレンド/レンジ)に応じてtrend/mean_reversionの
重みを日々動的に切り替える「レジーム適応型」合成が、cross_validate.pyで見つかった
時系列不安定性(前半→後半でICの符号が反転する問題)を緩和できるかを検証する。

適応型の重み = ベース重み × レジーム倍率(technical_v2.volatility_regimeと同じ倍率)
  regime="trend"  : trend_mult=1.3, mean_reversion_mult=0.3
  regime="range"  : trend_mult=0.3, mean_reversion_mult=1.3
  regime="transition"/"unknown": どちらも1.0

静的合成との比較を、時系列2分割(train/test)の両方で行う。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_backtest_connection  # noqa: E402

BASE_WEIGHTS = {
    "trend_score": 0.35, "mean_reversion_score": 0.25, "volume_score": 0.20,
    "market_regime_score": 0.15, "sector_regime_score": 0.05,
}
REGIME_MULTIPLIER = {
    "trend": {"trend_score": 1.3, "mean_reversion_score": 0.3},
    "range": {"trend_score": 0.3, "mean_reversion_score": 1.3},
    "transition": {"trend_score": 1.0, "mean_reversion_score": 1.0},
    "unknown": {"trend_score": 1.0, "mean_reversion_score": 1.0},
}


def load_data(conn, horizon: str) -> pd.DataFrame:
    query = f"""
    SELECT ticker, date, trend_score, mean_reversion_score, volume_score,
           market_regime_score, sector_regime_score, regime_volatility, {horizon} as y
    FROM signal_history WHERE {horizon} IS NOT NULL
    """
    return pd.read_sql_query(query, conn, parse_dates=["date"]).dropna(
        subset=["trend_score", "mean_reversion_score", "volume_score",
                "market_regime_score", "sector_regime_score", "y"]
    )


def static_composite(df: pd.DataFrame) -> pd.Series:
    return sum(df[col] * w for col, w in BASE_WEIGHTS.items())


def adaptive_composite(df: pd.DataFrame) -> pd.Series:
    trend_mult = df["regime_volatility"].map(lambda r: REGIME_MULTIPLIER.get(r, REGIME_MULTIPLIER["unknown"])["trend_score"])
    mr_mult = df["regime_volatility"].map(lambda r: REGIME_MULTIPLIER.get(r, REGIME_MULTIPLIER["unknown"])["mean_reversion_score"])
    composite = (
        df["trend_score"] * BASE_WEIGHTS["trend_score"] * trend_mult
        + df["mean_reversion_score"] * BASE_WEIGHTS["mean_reversion_score"] * mr_mult
        + df["volume_score"] * BASE_WEIGHTS["volume_score"]
        + df["market_regime_score"] * BASE_WEIGHTS["market_regime_score"]
        + df["sector_regime_score"] * BASE_WEIGHTS["sector_regime_score"]
    )
    return composite


def ic(composite: pd.Series, y: pd.Series) -> float:
    result = spearmanr(composite, y).correlation
    return 0.0 if pd.isna(result) else result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", default="forward_return_21d",
                         choices=["forward_return_5d", "forward_return_10d", "forward_return_21d"])
    parser.add_argument("--split-date", default=None)
    args = parser.parse_args()

    conn = get_backtest_connection()
    df = load_data(conn, args.horizon)
    conn.close()

    if len(df) < 200:
        print("データが不足しています")
        return

    split_date = args.split_date or str(df["date"].median().date())
    train = df[df["date"] < split_date]
    test = df[df["date"] >= split_date]
    print(f"horizon={args.horizon}, split_date={split_date}, train={len(train)}件, test={len(test)}件\n")

    print("=== 静的合成(固定重み) ===")
    static_train_ic = ic(static_composite(train), train["y"])
    static_test_ic = ic(static_composite(test), test["y"])
    print(f"train IC={static_train_ic:.4f} / test IC={static_test_ic:.4f}")

    print("\n=== レジーム適応型合成(ADXで日々trend/mean_reversionの重みを切替) ===")
    adaptive_train_ic = ic(adaptive_composite(train), train["y"])
    adaptive_test_ic = ic(adaptive_composite(test), test["y"])
    print(f"train IC={adaptive_train_ic:.4f} / test IC={adaptive_test_ic:.4f}")

    print("\n=== 比較 ===")
    print(f"test期間IC: 静的={static_test_ic:+.4f} / 適応型={adaptive_test_ic:+.4f}")
    if adaptive_test_ic > static_test_ic and adaptive_test_ic > 0:
        print("✓ レジーム適応型の方がtest期間で優位。この設計は不安定性の緩和に寄与している可能性がある。")
    elif adaptive_test_ic > 0 and static_test_ic <= 0:
        print("✓ 静的合成はtest期間でIC<=0だが、適応型は正のICを維持。レジーム適応が有効な可能性が高い。")
    else:
        print("△ レジーム適応型でも根本的な時系列不安定性は解消されていない。")

    print("\n--- 各時点のregime_volatility分布(train/test) ---")
    print("train:", train["regime_volatility"].value_counts(normalize=True).round(3).to_dict())
    print("test :", test["regime_volatility"].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
