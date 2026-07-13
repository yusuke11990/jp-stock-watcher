"""全カテゴリ・全サブ指標の重みを同時に動かして、総合スコアのIC(12ヶ月後リターンとの
Spearman相関)を最大化する組み合わせをscipy.optimizeで探索する。

これまでの手動調整(shareholder_return_metrics_backtest.py等)は1カテゴリずつ
試していたが、efficiencyの検証で「単独では妥当に見える重み変更が、他指標との
組み合わせでは悪化する」ことが判明した。手動で総当たりするのは非現実的なため、
数値最適化で系統的に探索する。

方法:
- 各サブ指標は既に業種内パーセンタイル化済み(0-100)なので、カテゴリ内の重み付き
  平均・カテゴリ間の重み付き平均という総合スコアの構造はそのまま保つ
- カテゴリ内の重みはシンプレックス制約(合計1、各0以上)。カテゴリ間の重み
  (category_weights)は本番の値で固定し、サブ指標配分のみ最適化する
  (カテゴリ間配分は別の手法・観点で既に決めているため、混同を避ける)
- 目的関数はSpearman ICではなくPearson相関(既にパーセンタイル化済みの値の
  線形結合なので、Spearmanの良い近似になる。SLSQPが使う勾配計算のため
  微分可能な代理指標が必要)
- 過学習を避けるため、2021〜2023年度で最適化(train)し、2024〜2025年度
  (test、最適化に一切使わない)でSpearman ICを検証する

結果(不採用): 最適化はカテゴリごとに極端な配分を選びがちだった(profitability:
roe=1.00で他4指標が0.00、growth: revenue_growth_1yに0.73集中、等)。以前の検証で
net_margin/operating_margin/ordinary_income_marginがほぼ同水準に健全と分かって
いたのに、trainデータのノイズに釣られて1指標に全振りする過学習が発生。結果、
最適化後の重みはtrainデータ自体でもIC悪化(0.2224→0.2112)、testデータでも悪化
(0.2243→0.2083)。カテゴリ単体の予測力(Pearson相関)を最大化しても、それが
category_weightsで他カテゴリと混ざった総合スコア全体の精度向上には繋がらない
ことを再確認した(efficiencyの手動検証で見た現象と同じ構造)。結論として、
shareholder_return_metrics_backtest.py等での手動・個別検証(1カテゴリずつ、
複数指標を残しつつ緩やかに再配分)の方が、この規模のデータでは機械的な最適化
より頑健。scoring_config.yamlは変更せず、記録として本スクリプトのみ残す。

実行: python scripts/backtest/joint_weight_optimization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection, get_backtest_connection  # noqa: E402
from scoring.compute_scores import percentile_rank, score_payout_ratio  # noqa: E402
from backtest.quality_score_backtest import MIN_SECTOR_SAMPLE_SIZE  # noqa: E402
from backtest.quality_score_multi_period_backtest import attach_sector  # noqa: E402
from backtest.quality_score_v2_backtest import (  # noqa: E402
    load_real_disclosure_dates, attach_signal_date, attach_price_features, add_valuation_metrics,
)
from backtest.growth_profitability_efficiency_metrics_backtest import (  # noqa: E402
    load_yearly_panel_extended as load_gpe_yearly, build_period_panel as build_gpe_panel,
)
from backtest.shareholder_return_metrics_backtest import (  # noqa: E402
    load_yearly_panel_extended as load_sr_yearly, build_period_panel_extended as build_sr_panel,
)

CATEGORY_WEIGHTS = {
    "valuation": 0.30, "profitability": 0.18, "efficiency": 0.15, "momentum": 0.15,
    "shareholder_return": 0.10, "safety": 0.07, "growth": 0.05,
}

# {category: [(metric, higher_is_better_or_"payout_special"), ...]}
CATEGORY_METRICS = {
    "safety": [("equity_ratio", True)],
    "growth": [
        ("revenue_growth_1y", True), ("revenue_growth_3y_cagr", True), ("operating_income_growth_1y", True),
        ("ordinary_income_growth_1y", True), ("eps_growth_1y", True), ("eps_growth_3y_cagr", True),
    ],
    "profitability": [
        ("roe", True), ("operating_margin", True), ("net_margin", True),
        ("ordinary_income_margin", True), ("operating_cf_margin", True),
    ],
    "efficiency": [("roa", True), ("asset_turnover", True)],
    "valuation": [("per", False), ("pbr", False), ("psr", False), ("pcfr", False)],
    "shareholder_return": [
        ("dividend_yield", True), ("payout_ratio", "payout_special"), ("total_shareholder_return_yield", True),
        ("dividend_history_count", True), ("doe", True), ("dividend_growth_1y", True),
    ],
    "momentum": [("momentum_12_1", False)],
}

# 現行(本番scoring_config.yaml)の重み。最適化の初期値・比較対象に使う
CURRENT_WEIGHTS = {
    "safety": {"equity_ratio": 1.0},
    "growth": {
        "revenue_growth_1y": 0.25, "revenue_growth_3y_cagr": 0.05, "operating_income_growth_1y": 0.20,
        "ordinary_income_growth_1y": 0.25, "eps_growth_1y": 0.20, "eps_growth_3y_cagr": 0.05,
    },
    "profitability": {
        "roe": 0.25, "operating_margin": 0.25, "net_margin": 0.20,
        "ordinary_income_margin": 0.15, "operating_cf_margin": 0.15,
    },
    "efficiency": {"roa": 0.50, "asset_turnover": 0.50},
    "valuation": {"per": 0.35, "pbr": 0.30, "psr": 0.20, "pcfr": 0.15},
    "shareholder_return": {
        "dividend_yield": 0.35, "payout_ratio": 0.05, "total_shareholder_return_yield": 0.25,
        "dividend_history_count": 0.20, "doe": 0.05, "dividend_growth_1y": 0.10,
    },
    "momentum": {"momentum_12_1": 1.0},
}


def sector_percentile_per_cohort(df: pd.DataFrame, col: str, higher_is_better) -> pd.Series:
    result = pd.Series(index=df.index, dtype=float)
    for cohort, df_cohort in df.groupby("cohort_year"):
        sector_counts = df_cohort["sector"].value_counts().to_dict()
        comparison_group = df_cohort["sector"].apply(
            lambda s: s if sector_counts.get(s, 0) >= MIN_SECTOR_SAMPLE_SIZE else "__MARKET_WIDE__"
        )
        for group, idx in comparison_group.groupby(comparison_group).groups.items():
            ref = df_cohort if group == "__MARKET_WIDE__" else df_cohort.loc[idx]
            if higher_is_better == "payout_special":
                normal_pct = percentile_rank(ref[col], higher_is_better=True)
                pct = score_payout_ratio(ref[col], normal_pct)
            else:
                pct = percentile_rank(ref[col], higher_is_better=higher_is_better)
            result.loc[idx] = pct.reindex(idx)
    return result


def build_panel(conn, bt_conn) -> pd.DataFrame:
    yearly_gpe = load_gpe_yearly(conn)
    panel_gpe = build_gpe_panel(yearly_gpe)
    yearly_sr = load_sr_yearly(conn)
    panel_sr = build_sr_panel(yearly_sr)
    sr_only_cols = ["ticker", "fiscal_year_end", "payout_ratio", "buyback_amount", "net_income",
                     "dividend_history_count", "dividend_per_share", "dividend_growth_1y", "eps"]
    panel = panel_gpe.merge(panel_sr[sr_only_cols], on=["ticker", "fiscal_year_end"], how="inner")
    panel = attach_sector(conn, panel)
    disclosure = load_real_disclosure_dates(conn)
    panel = attach_signal_date(panel, disclosure)
    panel = attach_price_features(bt_conn, panel)
    panel = add_valuation_metrics(panel)

    panel["dividend_yield"] = panel["dividend_per_share"] / panel["price_entry"]
    panel.loc[(panel["dividend_yield"] < 0) | (panel["dividend_yield"] > 0.3), "dividend_yield"] = None
    panel["market_cap_approx"] = panel["per"] * panel["net_income"]
    panel.loc[panel["market_cap_approx"] <= 0, "market_cap_approx"] = None
    total_div = panel["payout_ratio"] * panel["net_income"]
    buyback = panel["buyback_amount"].fillna(0)
    panel["total_shareholder_return_yield"] = (total_div + buyback) / panel["market_cap_approx"]
    panel.loc[(panel["total_shareholder_return_yield"] < 0) | (panel["total_shareholder_return_yield"] > 0.3), "total_shareholder_return_yield"] = None
    panel["doe"] = panel["payout_ratio"] * panel["roe"]
    panel.loc[(panel["doe"] < 0) | (panel["doe"] > 0.3), "doe"] = None
    panel.loc[(panel["payout_ratio"] < 0) | (panel["payout_ratio"] > 3), "payout_ratio"] = None

    bs = pd.read_sql_query("SELECT ticker, fiscal_year_end, revenue, operating_cf FROM fundamentals_yearly", conn, parse_dates=["fiscal_year_end"])
    panel = panel.drop(columns=["revenue", "operating_cf"], errors="ignore").merge(bs, on=["ticker", "fiscal_year_end"], how="left")
    panel["psr"] = panel["market_cap_approx"] / panel["revenue"]
    panel["pcfr"] = panel["market_cap_approx"] / panel["operating_cf"]
    panel.loc[(panel["psr"] <= 0) | (panel["psr"] > 50), "psr"] = None
    panel.loc[(panel["pcfr"] <= 0) | (panel["pcfr"] > 200), "pcfr"] = None
    return panel


def precompute_percentiles(panel: pd.DataFrame) -> None:
    for category, metrics in CATEGORY_METRICS.items():
        for metric, hib in metrics:
            panel[f"{metric}__pct"] = sector_percentile_per_cohort(panel, metric, hib)


def compute_total_score(panel: pd.DataFrame, weights: dict) -> pd.Series:
    category_scores = {}
    for category, metrics in CATEGORY_METRICS.items():
        weighted_sum = pd.Series(0.0, index=panel.index)
        total_weight = pd.Series(0.0, index=panel.index)
        for metric, _ in metrics:
            w = weights[category][metric]
            col = panel[f"{metric}__pct"]
            valid = col.notna()
            weighted_sum[valid] += col[valid] * w
            total_weight[valid] += w
        category_scores[category] = (weighted_sum / total_weight).where(total_weight > 0)

    total_weighted_sum = pd.Series(0.0, index=panel.index)
    total_weight_sum = pd.Series(0.0, index=panel.index)
    for category, cat_w in CATEGORY_WEIGHTS.items():
        score = category_scores[category]
        valid = score.notna()
        total_weighted_sum[valid] += score[valid] * cat_w
        total_weight_sum[valid] += cat_w
    return (total_weighted_sum / total_weight_sum).where(total_weight_sum > 0)


def optimize_category(panel: pd.DataFrame, category: str, metrics: list, return_col: str) -> dict:
    """1カテゴリ内のサブ指標重みを、そのカテゴリ単体スコアとリターンのPearson相関を
    最大化するように最適化する(他カテゴリは固定して独立最適化する近似)。"""
    metric_names = [m for m, _ in metrics]
    n = len(metric_names)
    if n == 1:
        return {metric_names[0]: 1.0}

    cols = [panel[f"{m}__pct"] for m in metric_names]
    mat = pd.concat(cols, axis=1).values  # (N, n)
    y = panel[return_col].values
    valid_row = ~np.isnan(mat).all(axis=1) & ~np.isnan(y)
    mat = mat[valid_row]
    y = y[valid_row]
    col_mean = np.nanmean(mat, axis=0)
    inds = np.where(np.isnan(mat))
    mat[inds] = np.take(col_mean, inds[1])

    def neg_corr(x):
        w = np.abs(x)
        w = w / w.sum()
        score = mat @ w
        if np.std(score) == 0 or np.std(y) == 0:
            return 0.0
        return -pearsonr(score, y)[0]

    x0 = np.ones(n) / n
    res = minimize(neg_corr, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    w = np.abs(res.x)
    w = w / w.sum()
    return dict(zip(metric_names, w))


def report(panel: pd.DataFrame, col: str, h: str, label: str) -> None:
    valid = panel.dropna(subset=[col, h])
    if len(valid) < 50:
        print(f"  [{label}] サンプル不足(n={len(valid)})")
        return
    ic = spearmanr(valid[col], valid[h]).correlation
    valid = valid.copy()
    valid["q"] = pd.qcut(valid[col].rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    qm = valid.groupby("q", observed=True)[h].mean()
    spread = qm["Q5"] - qm["Q1"]
    print(f"  [{label}] n={len(valid)}  IC={ic:+.4f}  Q5-Q1スプレッド={spread:+.4f}")


def main():
    conn = get_connection()
    bt_conn = get_backtest_connection()

    panel = build_panel(conn, bt_conn)
    print(f"パネル総観測数: {len(panel)}件")
    precompute_percentiles(panel)

    h = "forward_return_12ヶ月"
    train = panel[panel["cohort_year"].isin([2021, 2022, 2023])].copy()
    test = panel[panel["cohort_year"].isin([2024, 2025])].copy()
    print(f"train(2021-2023): {len(train)}件 / test(2024-2025、最適化に未使用): {len(test)}件\n")

    print("=== カテゴリごとにtrainデータでサブ指標重みを最適化 ===")
    optimized_weights = {}
    for category, metrics in CATEGORY_METRICS.items():
        w = optimize_category(train, category, metrics, h)
        optimized_weights[category] = w
        w_str = ", ".join(f"{m}={v:.2f}" for m, v in sorted(w.items(), key=lambda kv: -kv[1]))
        print(f"  {category}: {w_str}")

    panel["total_score_current"] = compute_total_score(panel, CURRENT_WEIGHTS)
    panel["total_score_optimized"] = compute_total_score(panel, optimized_weights)

    print("\n=== train/testでの総合スコアIC比較 ===")
    for name, sub in (("train(2021-2023)", panel[panel["cohort_year"].isin([2021, 2022, 2023])]),
                       ("test(2024-2025、未使用)", panel[panel["cohort_year"].isin([2024, 2025])])):
        print(f"\n[{name}]")
        report(sub, "total_score_current", h, "現行の重み")
        report(sub, "total_score_optimized", h, "最適化後の重み")

    conn.close()
    bt_conn.close()


if __name__ == "__main__":
    main()
