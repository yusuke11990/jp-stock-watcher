"""シグナルファミリー間の相関行列とquintile別リターン分析。

「相関の低い複数の独立シグナル」という設計目標を検証するゲート。
trend/mean_reversion等が高相関(|ρ|>0.5等)ならファミリー設計を見直す必要がある。
また各ファミリーが実際にその後のリターンと関係があるか(情報量があるか)を、
5分位(quintile)に分けた平均リターン・勝率・情報係数(IC)で確認する。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_backtest_connection  # noqa: E402

FAMILY_COLS = ["trend_score", "mean_reversion_score", "volume_score", "market_regime_score", "sector_regime_score"]


def load_signal_history(conn) -> pd.DataFrame:
    query = f"""
    SELECT ticker, date, {", ".join(FAMILY_COLS)},
           composite_technical_score, forward_return_5d, forward_return_10d, forward_return_21d
    FROM signal_history
    """
    return pd.read_sql_query(query, conn, parse_dates=["date"])


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[FAMILY_COLS].corr(method="spearman")


def quintile_analysis(df: pd.DataFrame, family_col: str, horizon: str) -> pd.DataFrame | None:
    sub = df[[family_col, horizon]].dropna()
    if len(sub) < 50:
        return None
    sub = sub.copy()
    n_unique = sub[family_col].nunique()

    if n_unique < 5:
        # market_regime_scoreのように離散値しか取らない列は、値そのものでグループ化する
        sub["quintile"] = sub[family_col].astype(str)
    else:
        # volume_scoreのように同値(特に0)が大量に重複する列は、qcutが素の値に対して
        # 分位境界を作れずエラーになることがあるため、順位(rank)に対してqcutする
        # (同値はraw値の並び順で機械的に分散されるが、5分位を安定して作れる)
        ranks = sub[family_col].rank(method="first")
        sub["quintile"] = pd.qcut(ranks, 5, labels=["Q1(弱)", "Q2", "Q3", "Q4", "Q5(強)"])

    result = sub.groupby("quintile", observed=True)[horizon].agg(
        mean_return="mean", median_return="median",
        win_rate=lambda x: (x > 0).mean(), n="count",
    )
    return result.sort_index()


def information_coefficient(df: pd.DataFrame, family_col: str, horizon: str) -> float | None:
    sub = df[[family_col, horizon]].dropna()
    if len(sub) < 50:
        return None
    return sub[family_col].corr(sub[horizon], method="spearman")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", default="forward_return_21d",
                         choices=["forward_return_5d", "forward_return_10d", "forward_return_21d"])
    args = parser.parse_args()

    conn = get_backtest_connection()
    df = load_signal_history(conn)
    conn.close()

    if df.empty:
        print("signal_historyにデータがありません。先にbuild_signal_history.pyを実行してください")
        return

    print(f"対象レコード数: {len(df)}件\n")

    print("=== ファミリー間相関行列(Spearman) ===")
    corr = correlation_matrix(df)
    print(corr.round(3))
    print()
    high_corr_pairs = []
    for i, c1 in enumerate(FAMILY_COLS):
        for c2 in FAMILY_COLS[i + 1:]:
            rho = corr.loc[c1, c2]
            if pd.notna(rho) and abs(rho) > 0.5:
                high_corr_pairs.append((c1, c2, rho))
    if high_corr_pairs:
        print("⚠ 相関が高い(|ρ|>0.5)ペア(設計見直しの検討対象):")
        for c1, c2, rho in high_corr_pairs:
            print(f"  {c1} - {c2}: ρ={rho:.3f}")
    else:
        print("✓ ファミリー間の相関は0.5以下(独立性は保たれている)")
    print()

    print(f"=== ファミリー別 情報係数(IC, horizon={args.horizon}) ===")
    for col in FAMILY_COLS + ["composite_technical_score"]:
        ic = information_coefficient(df, col, args.horizon)
        print(f"  {col:30s}: IC={ic:.4f}" if ic is not None else f"  {col:30s}: データ不足")
    print()

    print(f"=== quintile別リターン分析(horizon={args.horizon}) ===")
    for col in FAMILY_COLS + ["composite_technical_score"]:
        result = quintile_analysis(df, col, args.horizon)
        if result is None:
            print(f"--- {col}: データ不足 ---\n")
            continue
        print(f"--- {col} ---")
        print(result.round(4))
        spread = result["mean_return"].iloc[-1] - result["mean_return"].iloc[0]
        print(f"  Q5-Q1スプレッド: {spread:.4f}\n")


if __name__ == "__main__":
    main()
