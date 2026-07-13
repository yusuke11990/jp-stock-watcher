"""sector_neutral_metrics_backtest.py・shareholder_return_metrics_backtest.py・
growth_profitability_efficiency_metrics_backtest.pyの検証結果を踏まえた
scoring_config.yamlの重み変更案が、総合スコアレベルで実際にIC・五分位スプレッドを
改善するか検証する。

変更案(4カテゴリまとめて検証)と、その後の切り分け結果:
- efficiency: roa 0.50→0.85, asset_turnover 0.50→0.15
  (asset_turnoverは業種内パーセンタイルでIC+0.008とほぼノイズ、roaは+0.055と健全…
  だったが、単独で試すと総合スコアIC が明確に悪化した。12ヶ月0.220→0.210、
  24ヶ月0.250→0.238。単独ICがゼロでも、roaと組み合わせた際の補完情報(業種特性・
  事業モデルの違いを拾う分散効果)を持っている可能性が高く、**この変更は不採用**)
- profitability: roe 0.25→0.22, operating_margin 0.25→0.25, net_margin 0.20→0.23,
  ordinary_income_margin 0.15→0.20, operating_cf_margin 0.15→0.10
  (単独で試すと総合スコアICはほぼ無変化=誤差レベル。リスクに見合わないため**不採用**)
- shareholder_return: dividend_yield 0.25→0.35, total_shareholder_return_yield 0.20→0.25,
  dividend_history_count 0.10→0.20, dividend_growth_1y 0.15→0.10, payout_ratio 0.15→0.05,
  doe 0.15→0.05
  (上位3指標がIC+0.12〜0.21と強く、doe/payout_ratioは+0.01程度で弱いだけ。単独で
  試すと総合スコアICも改善。**採用**)
- growth: revenue_growth_1y 0.20→0.25, operating_income_growth_1y 0.15→0.20,
  ordinary_income_growth_1y 0.15→0.25, eps_growth_1y 0.15→0.20,
  revenue_growth_3y_cagr 0.25→0.05, eps_growth_3y_cagr 0.10→0.05
  (3年CAGR系は業種調整後も弱い逆行-0.02〜-0.03が残る。単独で試すと総合スコアICも
  小さく改善。**採用**)
- shareholder_return+growthを同時に変更した最終案は、12ヶ月IC 0.2200→0.2227、
  24ヶ月IC 0.2497→0.2545に改善、2021-2024年度で頑健(2025年度のみ僅かに悪化)。
  config/scoring_config.yamlに反映済み。
- safety・valuation・momentumは変更なし(safetyはequity_ratio以外のデータが
  過去に遡って取得できないため判断材料が無い、valuation/momentumは既に健全)

このスクリプトのOLD_WEIGHTS/NEW_WEIGHTSは「4カテゴリまとめて変更」した最初の
検証時点のものをそのまま残してある(悪化した組み合わせも含む)。最終的に採用したのは
shareholder_return+growthの2カテゴリのみの変更で、その結果は上記コメント参照。

実行: python scripts/backtest/total_score_reweight_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection, get_backtest_connection  # noqa: E402
from scoring.compute_scores import percentile_rank, score_payout_ratio, load_config  # noqa: E402
from backtest.quality_score_backtest import MIN_SECTOR_SAMPLE_SIZE  # noqa: E402
from backtest.quality_score_multi_period_backtest import attach_sector  # noqa: E402
from backtest.quality_score_v2_backtest import (  # noqa: E402
    load_real_disclosure_dates, attach_signal_date, attach_price_features, add_valuation_metrics,
    FORWARD_WINDOWS_DAYS,
)
from backtest.growth_profitability_efficiency_metrics_backtest import (  # noqa: E402
    load_yearly_panel_extended as load_gpe_yearly, build_period_panel as build_gpe_panel,
)
from backtest.shareholder_return_metrics_backtest import (  # noqa: E402
    load_yearly_panel_extended as load_sr_yearly, build_period_panel_extended as build_sr_panel,
)

# {category: {metric: (weight, higher_is_better_or_"payout_special")}}
OLD_WEIGHTS = {
    "safety": {"equity_ratio": (1.0, True)},
    "growth": {
        "revenue_growth_1y": (0.20, True), "revenue_growth_3y_cagr": (0.25, True),
        "operating_income_growth_1y": (0.15, True), "ordinary_income_growth_1y": (0.15, True),
        "eps_growth_1y": (0.15, True), "eps_growth_3y_cagr": (0.10, True),
    },
    "profitability": {
        "roe": (0.25, True), "operating_margin": (0.25, True), "net_margin": (0.20, True),
        "ordinary_income_margin": (0.15, True), "operating_cf_margin": (0.15, True),
    },
    "efficiency": {"roa": (0.50, True), "asset_turnover": (0.50, True)},
    "valuation": {"per": (0.35, False), "pbr": (0.30, False), "psr": (0.20, False), "pcfr": (0.15, False)},
    "shareholder_return": {
        "dividend_yield": (0.25, True), "payout_ratio": (0.15, "payout_special"),
        "total_shareholder_return_yield": (0.20, True), "dividend_history_count": (0.10, True),
        "doe": (0.15, True), "dividend_growth_1y": (0.15, True),
    },
    "momentum": {"momentum_12_1": (1.0, False)},
}

NEW_WEIGHTS = {
    "safety": {"equity_ratio": (1.0, True)},
    "growth": {
        "revenue_growth_1y": (0.25, True), "revenue_growth_3y_cagr": (0.05, True),
        "operating_income_growth_1y": (0.20, True), "ordinary_income_growth_1y": (0.25, True),
        "eps_growth_1y": (0.20, True), "eps_growth_3y_cagr": (0.05, True),
    },
    "profitability": {
        "roe": (0.22, True), "operating_margin": (0.25, True), "net_margin": (0.23, True),
        "ordinary_income_margin": (0.20, True), "operating_cf_margin": (0.10, True),
    },
    "efficiency": {"roa": (0.85, True), "asset_turnover": (0.15, True)},
    "valuation": {"per": (0.35, False), "pbr": (0.30, False), "psr": (0.20, False), "pcfr": (0.15, False)},
    "shareholder_return": {
        "dividend_yield": (0.35, True), "payout_ratio": (0.05, "payout_special"),
        "total_shareholder_return_yield": (0.25, True), "dividend_history_count": (0.20, True),
        "doe": (0.05, True), "dividend_growth_1y": (0.10, True),
    },
    "momentum": {"momentum_12_1": (1.0, False)},
}

CATEGORY_WEIGHTS = {
    "valuation": 0.30, "profitability": 0.18, "efficiency": 0.15, "momentum": 0.15,
    "shareholder_return": 0.10, "safety": 0.07, "growth": 0.05,
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


def compute_total_score(df: pd.DataFrame, weights: dict, suffix: str) -> pd.Series:
    category_scores = {}
    for category, metrics in weights.items():
        weighted_sum = pd.Series(0.0, index=df.index)
        total_weight = pd.Series(0.0, index=df.index)
        for metric, (w, hib) in metrics.items():
            pct_col = f"{metric}_pct_{suffix}"
            if pct_col not in df.columns:
                df[pct_col] = sector_percentile_per_cohort(df, metric, hib)
            valid = df[pct_col].notna()
            weighted_sum[valid] += df.loc[valid, pct_col] * w
            total_weight[valid] += w
        category_scores[category] = (weighted_sum / total_weight).where(total_weight > 0)

    total_weighted_sum = pd.Series(0.0, index=df.index)
    total_weight_sum = pd.Series(0.0, index=df.index)
    for category, cat_w in CATEGORY_WEIGHTS.items():
        score = category_scores[category]
        valid = score.notna()
        total_weighted_sum[valid] += score[valid] * cat_w
        total_weight_sum[valid] += cat_w
    return (total_weighted_sum / total_weight_sum).where(total_weight_sum > 0)


def report(df: pd.DataFrame, col: str, h: str, label: str) -> None:
    valid = df.dropna(subset=[col, h])
    if len(valid) < 50:
        print(f"  [{label}] サンプル不足(n={len(valid)})")
        return
    ic = spearmanr(valid[col], valid[h]).correlation
    valid = valid.copy()
    valid["q"] = pd.qcut(valid[col].rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    qm = valid.groupby("q", observed=True)[h].mean()
    spread = qm["Q5"] - qm["Q1"]
    print(f"  [{label}] n={len(valid)}  IC={ic:+.4f}  Q5-Q1スプレッド={spread:+.4f}  (Q1={qm['Q1']:+.4f} / Q5={qm['Q5']:+.4f})")


def main():
    conn = get_connection()
    bt_conn = get_backtest_connection()

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
    print(f"パネル総観測数: {len(panel)}件")
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

    n_with_price = panel["price_entry"].notna().sum()
    print(f"signal_date時点の株価が取れた観測数: {n_with_price}件\n")

    panel["total_score_old"] = compute_total_score(panel, OLD_WEIGHTS, "old")
    panel["total_score_new"] = compute_total_score(panel, NEW_WEIGHTS, "new")

    print("=== 総合スコアIC比較(現行 vs 変更案) ===")
    for h_label in FORWARD_WINDOWS_DAYS:
        h = f"forward_return_{h_label}"
        print(f"\n--- {h_label}後 ---")
        report(panel, "total_score_old", h, "現行の重み")
        report(panel, "total_score_new", h, "変更案の重み")

    print("\n=== 年度別ロバスト性(12ヶ月後) ===")
    h = "forward_return_12ヶ月"
    for cohort, sub in panel.dropna(subset=["total_score_old", "total_score_new", h]).groupby("cohort_year"):
        if len(sub) < 50:
            continue
        ic_old = spearmanr(sub["total_score_old"], sub[h]).correlation
        ic_new = spearmanr(sub["total_score_new"], sub[h]).correlation
        print(f"  {cohort}年度 (n={len(sub)}): 現行IC={ic_old:+.4f}  変更案IC={ic_new:+.4f}")

    conn.close()
    bt_conn.close()


if __name__ == "__main__":
    main()
