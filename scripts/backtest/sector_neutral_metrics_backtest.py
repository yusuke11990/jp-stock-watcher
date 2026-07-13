"""growth_profitability_efficiency_metrics_backtest.py・shareholder_return_metrics_backtest.py
で見つかった「逆行している」指標(equity_ratio・asset_turnover・revenue_growth_3y_cagr・
eps_growth_3y_cagr等)が、本番と同じ土俵(業種内パーセンタイル化した後)でも逆行するのか、
それとも業種構成の偏りによる見せかけだったのかを検証する。

前回の検証は全銘柄を業種問わずプールしてSpearman IC(生の指標値 vs フォワードリターン)
を計算していたが、本番のcompute_scores.pyは指標を業種内(小規模業種は市場全体に
フォールバック)でパーセンタイル化してから使う。業種構成の偏り(例: 自己資本比率が
低い業種がたまたまこの数年好調だった等)が交絡している可能性があるため、同じ
パーセンタイル化ロジックを再現した上で同じ指標を再検証する。

実行: python scripts/backtest/sector_neutral_metrics_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection, get_backtest_connection  # noqa: E402
from backtest.quality_score_backtest import percentile_rank, MIN_SECTOR_SAMPLE_SIZE  # noqa: E402
from backtest.quality_score_multi_period_backtest import attach_sector, MIN_GAP_DAYS, DISCLOSURE_LAG_DAYS, FORWARD_DAYS  # noqa: E402
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

# 検証対象。前回「逆行」と出たものを優先しつつ、還元性の弱い指標も含めて一通り確認する
METRICS = [
    "equity_ratio", "asset_turnover", "roa",
    "revenue_growth_3y_cagr", "eps_growth_3y_cagr", "revenue_growth_1y", "eps_growth_1y",
    "roe", "operating_margin", "net_margin", "ordinary_income_margin", "operating_cf_margin",
    "payout_ratio", "doe",
]


def sector_percentile_per_cohort(df: pd.DataFrame, col: str) -> pd.Series:
    """本番のcompute_category_score/get_comparison_groupと同じロジックで、
    コホート年度ごと・業種内(小規模業種は市場全体)でパーセンタイル化する。"""
    result = pd.Series(index=df.index, dtype=float)
    for cohort, df_cohort in df.groupby("cohort_year"):
        sector_counts = df_cohort["sector"].value_counts().to_dict()
        comparison_group = df_cohort["sector"].apply(
            lambda s: s if sector_counts.get(s, 0) >= MIN_SECTOR_SAMPLE_SIZE else "__MARKET_WIDE__"
        )
        for group, idx in comparison_group.groupby(comparison_group).groups.items():
            if group == "__MARKET_WIDE__":
                # 寄せ集め同士でなく、そのコホート全体を母集団にする(本番と同じ)
                pct = percentile_rank(df_cohort[col], higher_is_better=True).reindex(idx)
            else:
                pct = percentile_rank(df_cohort.loc[idx, col], higher_is_better=True)
            result.loc[idx] = pct
    return result


def report(df: pd.DataFrame, col: str, h: str) -> None:
    valid = df.dropna(subset=[col, h])
    if len(valid) < 50:
        print(f"    [{col}] サンプル不足(n={len(valid)})")
        return
    ic = spearmanr(valid[col], valid[h]).correlation
    print(f"    [{col}] n={len(valid)}  業種内パーセンタイルIC={ic:+.4f}")


def main():
    conn = get_connection()
    bt_conn = get_backtest_connection()

    # --- growth/profitability/efficiency/safety系パネル ---
    yearly_gpe = load_gpe_yearly(conn)
    panel_gpe = build_gpe_panel(yearly_gpe)
    panel_gpe = attach_sector(conn, panel_gpe)
    disclosure = load_real_disclosure_dates(conn)
    panel_gpe = attach_signal_date(panel_gpe, disclosure)
    panel_gpe = attach_price_features(bt_conn, panel_gpe)

    # --- shareholder_return系パネル(別途構築、payout_ratio/doeを含む) ---
    yearly_sr = load_sr_yearly(conn)
    panel_sr = build_sr_panel(yearly_sr)
    panel_sr = attach_sector(conn, panel_sr)
    panel_sr = attach_signal_date(panel_sr, disclosure)
    panel_sr = attach_price_features(bt_conn, panel_sr)
    panel_sr = add_valuation_metrics(panel_sr)  # per, pbr (net_income*per で時価総額近似に使うわけではないが一貫性のため)
    panel_sr["doe"] = panel_sr["payout_ratio"] * panel_sr["roe"]
    panel_sr.loc[(panel_sr["doe"] < 0) | (panel_sr["doe"] > 0.3), "doe"] = None
    panel_sr.loc[(panel_sr["payout_ratio"] < 0) | (panel_sr["payout_ratio"] > 3), "payout_ratio"] = None

    print(f"gpeパネル: {len(panel_gpe)}件, srパネル: {len(panel_sr)}件\n")

    print("=== 業種内パーセンタイル化した後のIC(higher_is_better=Trueで統一、符号がそのまま方向) ===")
    for h_label in FORWARD_WINDOWS_DAYS:
        h = f"forward_return_{h_label}"
        print(f"\n--- {h_label}後 ---")
        for col in METRICS:
            if col in panel_gpe.columns:
                pct_col = f"{col}_pct"
                panel_gpe[pct_col] = sector_percentile_per_cohort(panel_gpe, col)
                report(panel_gpe, pct_col, h)
            elif col in panel_sr.columns:
                pct_col = f"{col}_pct"
                panel_sr[pct_col] = sector_percentile_per_cohort(panel_sr, col)
                report(panel_sr, pct_col, h)

    print("\n=== 年度別ロバスト性(12ヶ月後、業種内パーセンタイル) ===")
    h = "forward_return_12ヶ月"
    for col in ("equity_ratio", "asset_turnover", "revenue_growth_3y_cagr", "eps_growth_3y_cagr"):
        pct_col = f"{col}_pct"
        print(f"  --- {col} ---")
        for cohort, sub in panel_gpe.dropna(subset=[pct_col, h]).groupby("cohort_year"):
            if len(sub) < 50:
                continue
            ic = spearmanr(sub[pct_col], sub[h]).correlation
            print(f"    {cohort}年度: IC={ic:+.4f} (n={len(sub)})")

    conn.close()
    bt_conn.close()


if __name__ == "__main__":
    main()
