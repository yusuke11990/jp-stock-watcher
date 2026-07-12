"""割安性(valuation)スコアを「業種内パーセンタイルのみ」で出す現行方式と、
「業種内パーセンタイル + 市場全体パーセンタイルのブレンド」で出す方式を比較する。

背景: 日本郵船(9101.T)のPBR0.71倍が業種内順位では45点程度にしかならない事象を
ユーザーに説明した際、「業種内相対だけでいいのか」という疑問が出た。海運業のように
業種全体が絶対的に割安/割高な局面では、業種内順位だけだとその情報が消える。

検証方法: quality_score_v2_backtest.pyと同じコホートパネル(決算期ごと、PER/PBRを
signal_date時点の株価から逆算)を使い、value(割安性)カテゴリのスコアを
1) 現行方式: 業種内(またはmin_sector_sample_size未満なら市場全体)パーセンタイルのみ
2) ブレンド方式: 0.7×業種内パーセンタイル + 0.3×市場全体パーセンタイル
の2通りで計算し、12ヶ月・24ヶ月フォワードリターンとのSpearman IC、五分位スプレッドを比較する。

実行: python scripts/backtest/valuation_score_blend_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection, get_backtest_connection  # noqa: E402
from backtest.quality_score_backtest import percentile_rank, MIN_SECTOR_SAMPLE_SIZE  # noqa: E402
from backtest.quality_score_multi_period_backtest import (  # noqa: E402
    build_period_panel, load_yearly_panel, attach_sector,
)
from backtest.quality_score_v2_backtest import (  # noqa: E402
    load_real_disclosure_dates, attach_signal_date, attach_price_features, add_valuation_metrics,
    FORWARD_WINDOWS_DAYS, report,
)

VALUE_METRICS = [("per", False), ("pbr", False)]
BLEND_SECTOR_WEIGHT = 0.7
BLEND_MARKET_WEIGHT = 0.3


def compute_value_score_sector_only(df_cohort: pd.DataFrame) -> pd.Series:
    """現行方式: comparison_group(業種、または市場全体フォールバック)内パーセンタイルのみ"""
    sector_counts = df_cohort["sector"].value_counts().to_dict()
    comparison_group = df_cohort["sector"].apply(
        lambda s: s if sector_counts.get(s, 0) >= MIN_SECTOR_SAMPLE_SIZE else "__MARKET_WIDE__"
    )
    score = pd.Series(index=df_cohort.index, dtype=float)
    for group, idx in comparison_group.groupby(comparison_group).groups.items():
        sub = df_cohort.loc[idx]
        weighted_sum = pd.Series(0.0, index=sub.index)
        total_weight = pd.Series(0.0, index=sub.index)
        weight_each = 1.0 / len(VALUE_METRICS)
        for col, higher_is_better in VALUE_METRICS:
            pct = percentile_rank(sub[col], higher_is_better)
            valid = pct.notna()
            weighted_sum[valid] += pct[valid] * weight_each
            total_weight[valid] += weight_each
        score.loc[sub.index] = (weighted_sum / total_weight).where(total_weight > 0)
    return score


def compute_value_score_blended(df_cohort: pd.DataFrame) -> pd.Series:
    """ブレンド方式: 業種内パーセンタイル×0.7 + 市場全体パーセンタイル×0.3"""
    sector_only = compute_value_score_sector_only(df_cohort)

    market_weighted_sum = pd.Series(0.0, index=df_cohort.index)
    market_total_weight = pd.Series(0.0, index=df_cohort.index)
    weight_each = 1.0 / len(VALUE_METRICS)
    for col, higher_is_better in VALUE_METRICS:
        pct = percentile_rank(df_cohort[col], higher_is_better)
        valid = pct.notna()
        market_weighted_sum[valid] += pct[valid] * weight_each
        market_total_weight[valid] += weight_each
    market_wide = (market_weighted_sum / market_total_weight).where(market_total_weight > 0)

    both_valid = sector_only.notna() & market_wide.notna()
    blended = pd.Series(index=df_cohort.index, dtype=float)
    blended[both_valid] = (
        BLEND_SECTOR_WEIGHT * sector_only[both_valid] + BLEND_MARKET_WEIGHT * market_wide[both_valid]
    )
    # 市場全体が計算できてsector_onlyだけ欠ける事は無いが念のためフォールバック
    only_sector = sector_only.notna() & ~both_valid
    blended[only_sector] = sector_only[only_sector]
    return blended


def compute_per_cohort(panel: pd.DataFrame, fn, col_name: str) -> pd.DataFrame:
    parts = []
    for cohort, df_cohort in panel.groupby("cohort_year"):
        df_cohort = df_cohort.copy()
        df_cohort[col_name] = fn(df_cohort)
        parts.append(df_cohort)
    return pd.concat(parts, ignore_index=True)


def main():
    conn = get_connection()
    bt_conn = get_backtest_connection()

    yearly = load_yearly_panel(conn)
    panel = build_period_panel(yearly)
    panel = attach_sector(conn, panel)

    disclosure = load_real_disclosure_dates(conn)
    panel = attach_signal_date(panel, disclosure)

    print(f"パネル総観測数: {len(panel)}件")
    panel = attach_price_features(bt_conn, panel)
    panel = add_valuation_metrics(panel)

    n_with_price = panel["price_entry"].notna().sum()
    print(f"signal_date時点の株価が取れた観測数: {n_with_price}件\n")

    panel = compute_per_cohort(panel, compute_value_score_sector_only, "value_score_sector_only")
    panel = compute_per_cohort(panel, compute_value_score_blended, "value_score_blended")

    print("=== フォワード期間別の比較(現行=業種内のみ vs ブレンド=業種内70%+市場全体30%) ===")
    for label in FORWARD_WINDOWS_DAYS:
        return_col = f"forward_return_{label}"
        report(panel, "value_score_sector_only", return_col, f"現行(業種内のみ)      / {label}後")
        report(panel, "value_score_blended", return_col, f"ブレンド(業種内70%+市場30%) / {label}後")
        print()

    print("=== コホート年度別IC(12ヶ月後、頑健性の確認) ===")
    h = "forward_return_12ヶ月"
    for cohort, sub in panel.dropna(subset=[h]).groupby("cohort_year"):
        row = []
        for col, tag in (("value_score_sector_only", "現行"), ("value_score_blended", "ブレンド")):
            s = sub.dropna(subset=[col, h])
            if len(s) < 30:
                continue
            ic = spearmanr(s[col], s[h]).correlation
            row.append(f"{tag}IC={ic:+.4f}(n={len(s)})")
        if row:
            print(f"  {cohort}年度: " + "  |  ".join(row))

    print("\n=== 現行スコアと市場全体パーセンタイルの乖離が大きい銘柄(直近コホート、上位10件) ===")
    latest_cohort = panel["cohort_year"].max()
    latest = panel[panel["cohort_year"] == latest_cohort].copy()
    latest["diff"] = latest["value_score_blended"] - latest["value_score_sector_only"]
    cols = ["ticker", "sector", "per", "pbr", "value_score_sector_only", "value_score_blended", "diff"]
    print(latest.dropna(subset=["diff"]).reindex(latest["diff"].abs().sort_values(ascending=False).index)[cols].head(10).to_string(index=False))

    conn.close()
    bt_conn.close()


if __name__ == "__main__":
    main()
