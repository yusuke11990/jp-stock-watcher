"""重み最適化結果の頑健性を、時系列分割と銘柄k-foldの2方向で検証する。

1年分のデータしかないため、通常の「訓練5年・検証1年」のような分割はできない。
代わりに以下の2つの検証を行い、両方をクリアしたパラメータのみ採用する:
  1. 時系列2分割: 前半6ヶ月で最適化した重みを、後半6ヶ月でIC検証する
  2. 銘柄k-fold: 銘柄をランダムにK分割し、各foldでのICのばらつきを見る
     (特定の銘柄群にだけ効くパラメータへの過学習を検出する)
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
from backtest.weight_optimizer import FEATURES, optimize_weights, negative_ic  # noqa: E402


def load_signal_history_with_date(conn, horizon: str) -> pd.DataFrame:
    query = f"""
    SELECT ticker, date, {", ".join(FEATURES)}, {horizon} as y
    FROM signal_history WHERE {horizon} IS NOT NULL
    """
    return pd.read_sql_query(query, conn, parse_dates=["date"]).dropna()


def walk_forward_validate(df: pd.DataFrame, split_date: str) -> dict:
    train = df[df["date"] < split_date]
    test = df[df["date"] >= split_date]
    if len(train) < 100 or len(test) < 100:
        return {"error": f"train={len(train)}, test={len(test)}件は検証に不十分です"}

    result = optimize_weights(train)
    weights_arr = np.array([result["weights"][f] for f in FEATURES])

    test_ic = -negative_ic(weights_arr, test[FEATURES].values, test["y"].values)
    return {
        "train_n": len(train), "test_n": len(test),
        "train_ic": result["ic"], "test_ic": test_ic,
        "weights": result["weights"],
        "degradation": result["ic"] - test_ic,
    }


def ticker_kfold_validate(df: pd.DataFrame, weights: dict, k: int = 5, seed: int = 42) -> list[float]:
    tickers = df["ticker"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(tickers)
    folds = np.array_split(tickers, k)

    weights_arr = np.array([weights[f] for f in FEATURES])
    ics = []
    for fold in folds:
        subset = df[df["ticker"].isin(fold)]
        if len(subset) < 50:
            continue
        ic = -negative_ic(weights_arr, subset[FEATURES].values, subset["y"].values)
        ics.append(ic)
    return ics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", default="forward_return_21d",
                         choices=["forward_return_5d", "forward_return_10d", "forward_return_21d"])
    parser.add_argument("--split-date", default=None, help="省略時はデータ期間の中央日")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    conn = get_backtest_connection()
    df = load_signal_history_with_date(conn, args.horizon)
    conn.close()

    if len(df) < 200:
        print("データが不足しています")
        return

    split_date = args.split_date or str(df["date"].median().date())
    print(f"=== 時系列2分割検証 (split_date={split_date}) ===")
    wf = walk_forward_validate(df, split_date)
    if "error" in wf:
        print(wf["error"])
        return
    print(f"訓練期間: {wf['train_n']}件, 検証期間: {wf['test_n']}件")
    print(f"訓練IC: {wf['train_ic']:.4f} / 検証IC: {wf['test_ic']:.4f} (劣化幅: {wf['degradation']:+.4f})")
    print("訓練期間で最適化した重み:")
    for k_, v in sorted(wf["weights"].items(), key=lambda kv: -kv[1]):
        print(f"  {k_:25s}: {v:.3f}")

    if wf["test_ic"] <= 0:
        print("\n⚠ 検証期間のICが0以下。この重みは前半期間に過学習している可能性が高く、採用は推奨しません。")
    elif wf["degradation"] > wf["train_ic"] * 0.5:
        print("\n⚠ 訓練→検証でICが50%以上劣化。過学習の懸念があります。")
    else:
        print("\n✓ 検証期間でも正のICを維持。時系列方向の頑健性は一定程度確認できました。")

    print(f"\n=== 銘柄{args.k}-fold交差検証 ===")
    ics = ticker_kfold_validate(df, wf["weights"], k=args.k)
    print(f"fold別IC: {[round(v, 4) for v in ics]}")
    print(f"平均: {np.mean(ics):.4f}, 標準偏差: {np.std(ics):.4f}")
    if np.std(ics) > abs(np.mean(ics)):
        print("⚠ fold間のばらつきが平均ICより大きく、特定銘柄群への過学習の可能性があります。")
    else:
        print("✓ fold間のばらつきは平均IC以下で、銘柄横断の安定性は一定程度確認できました。")


if __name__ == "__main__":
    main()
