"""シグナルファミリー間の重みを、情報係数(IC)最大化で決定する。

線形結合 composite = Σ(weight_i × family_score_i) の重みを、
将来リターンとのSpearman順位相関(IC)が最大になるようscipy.optimizeで求める。
制約: 重みの合計=1、各重み>=0(空売り前提の負の重みは今回は考慮しない)。

1年分のデータでの最適化は過学習しやすいため、この結果は
cross_validate.pyでの頑健性チェックとセットで解釈すること。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_backtest_connection  # noqa: E402

FEATURES = ["trend_score", "mean_reversion_score", "volume_score", "market_regime_score", "sector_regime_score"]


def load_signal_history(conn, horizon: str) -> pd.DataFrame:
    query = f"SELECT {', '.join(FEATURES)}, {horizon} as y FROM signal_history WHERE {horizon} IS NOT NULL"
    return pd.read_sql_query(query, conn).dropna()


def negative_ic(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    composite = X @ weights
    ic = spearmanr(composite, y).correlation
    if np.isnan(ic):
        return 0.0
    return -ic


def optimize_weights(df: pd.DataFrame) -> dict:
    X = df[FEATURES].values
    y = df["y"].values
    n = len(FEATURES)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n
    x0 = np.ones(n) / n

    result = minimize(negative_ic, x0, args=(X, y), bounds=bounds, constraints=constraints, method="SLSQP")
    weights = dict(zip(FEATURES, result.x))
    achieved_ic = -result.fun
    return {"weights": weights, "ic": achieved_ic, "success": result.success}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", default="forward_return_21d",
                         choices=["forward_return_5d", "forward_return_10d", "forward_return_21d"])
    args = parser.parse_args()

    conn = get_backtest_connection()
    df = load_signal_history(conn, args.horizon)
    conn.close()

    if len(df) < 100:
        print("データが不足しています。先にbuild_signal_history.pyを実行してください")
        return

    print(f"対象レコード数: {len(df)}件 (horizon={args.horizon})\n")

    # 比較用: 現行の暫定重み(build_signal_history.pyのcomposite_technical_score算出時の重み)
    baseline_weights = {"trend_score": 0.35, "mean_reversion_score": 0.25, "volume_score": 0.20,
                         "market_regime_score": 0.15, "sector_regime_score": 0.05}
    X = df[FEATURES].values
    baseline_ic = -negative_ic(np.array([baseline_weights[f] for f in FEATURES]), X, df["y"].values)
    print(f"暫定重み(build_signal_history.py初期値)でのIC: {baseline_ic:.4f}")
    print(f"  重み: {baseline_weights}\n")

    result = optimize_weights(df)
    print(f"最適化後のIC: {result['ic']:.4f} (収束: {result['success']})")
    print("最適化後の重み:")
    for k, v in sorted(result["weights"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:25s}: {v:.3f}")

    print(f"\nIC改善幅: {result['ic'] - baseline_ic:+.4f}")
    print("\n※この重みはconfig/technical_config.yamlへ手動反映する前に、")
    print("  cross_validate.pyで時系列分割・銘柄k-foldの頑健性を必ず確認してください。")


if __name__ == "__main__":
    main()
