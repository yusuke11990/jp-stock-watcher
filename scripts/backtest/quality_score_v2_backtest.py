"""quality_score_multi_period_backtest.pyの拡張版。以下を追加する:

1. 開示ラグの実測値: edinet_documents.submit_date_time(有価証券報告書の実際の提出日時)を
   使い、無い場合のみ「決算期末+100日」の仮定にフォールバックする。
2. バリュー(割安性)軸: PER=株価/EPS、PBR=株価/BVPSをsignal_date時点の株価から逆算する。
   BVPS(1株純資産)は別データが無いため、EPS/ROE(= (純利益/株数)/(純利益/純資産) = 純資産/株数)
   という恒等式で近似する。
3. モメンタム軸: signal_dateから遡った6ヶ月・12ヶ月の株価騰落率。
4. 複数フォワード期間(6ヶ月・12ヶ月・24ヶ月)での比較。
5. 五分位だけでなく、Q5-Q1のロング・ショートスプレッドも算出する。

実行: python scripts/backtest/quality_score_v2_backtest.py
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

FALLBACK_DISCLOSURE_LAG_DAYS = 100
FORWARD_WINDOWS_DAYS = {"6ヶ月": 182, "12ヶ月": 365, "24ヶ月": 730}
MOMENTUM_WINDOWS_DAYS = {"momentum_6m": 182, "momentum_12m": 365}

# クオリティ5軸(既存) + バリュー・モメンタムを加えた重み。
# 一軸の影響が大きくなりすぎないよう、6軸均等(各1/6)を基本にする
CATEGORY_METRICS_V2 = {
    "safety": [("equity_ratio", True)],
    "growth": [("revenue_growth_1y", True)],
    "profitability": [("roe", True), ("net_margin", True)],
    "efficiency": [("roa", True), ("asset_turnover", True)],
    "shareholder_return": [("dividend_growth_1y", True), ("doe_proxy", True)],
    "value": [("per", False), ("pbr", False)],
    # この市場・期間では継続(モメンタム)ではなく逆張り(リバーサル)方向にICが出るため、
    # higher_is_better=Falseにして直近の下落銘柄を高スコアにする
    "momentum": [("momentum_12_1", False)],
}
CATEGORY_WEIGHTS_V2 = {cat: 1.0 / len(CATEGORY_METRICS_V2) for cat in CATEGORY_METRICS_V2}


def load_real_disclosure_dates(conn) -> pd.DataFrame:
    """edinet_documentsから有価証券報告書の実際の提出日時を(ticker, period_end)単位で取得する"""
    df = pd.read_sql_query(
        "SELECT sec_code, period_end, submit_date_time FROM edinet_documents WHERE doc_type_code = '120'",
        conn, parse_dates=["period_end", "submit_date_time"],
    )
    # sec_code(5桁、末尾0付与)からticker(4桁+.T)へ変換
    df["ticker"] = df["sec_code"].str.slice(0, 4) + ".T"
    df = df.rename(columns={"period_end": "fiscal_year_end"})
    # 同一期に複数提出(訂正報告書等)がある場合は最初の提出日時を使う
    df = df.sort_values("submit_date_time").drop_duplicates(subset=["ticker", "fiscal_year_end"], keep="first")
    return df[["ticker", "fiscal_year_end", "submit_date_time"]]


def attach_signal_date(panel: pd.DataFrame, disclosure: pd.DataFrame) -> pd.DataFrame:
    panel = panel.merge(disclosure, on=["ticker", "fiscal_year_end"], how="left")
    fallback = panel["fiscal_year_end"] + pd.Timedelta(days=FALLBACK_DISCLOSURE_LAG_DAYS)
    panel["signal_date"] = panel["submit_date_time"].fillna(fallback)
    n_real = panel["submit_date_time"].notna().sum()
    print(f"開示日: 実測値を使えた件数={n_real} / 仮定({FALLBACK_DISCLOSURE_LAG_DAYS}日)にフォールバックした件数={len(panel) - n_real}")
    return panel.drop(columns=["submit_date_time"])


def attach_price_features(conn, panel: pd.DataFrame) -> pd.DataFrame:
    prices = pd.read_sql_query("SELECT ticker, date, close FROM price_history", conn, parse_dates=["date"])
    prices = prices.sort_values("date")

    def asof_price(dates_df: pd.DataFrame, date_col: str, direction: str, tolerance_days: int = 14) -> pd.Series:
        sub = dates_df[["ticker", date_col]].sort_values(date_col)
        merged = pd.merge_asof(
            sub, prices, left_on=date_col, right_on="date", by="ticker", direction=direction,
            tolerance=pd.Timedelta(days=tolerance_days),
        )
        return merged.set_index(sub.index)["close"]

    panel = panel.copy()
    panel["price_entry"] = asof_price(panel, "signal_date", "forward")

    # 素朴な「signal_date時点までの騰落率」(直近1ヶ月の短期リバーサルが混入しうる)
    for label, days in MOMENTUM_WINDOWS_DAYS.items():
        lookback_date = (panel["signal_date"] - pd.Timedelta(days=days)).rename("lookback_date")
        tmp = pd.concat([panel["ticker"], lookback_date], axis=1)
        past_price = asof_price(tmp, "lookback_date", "backward", tolerance_days=21)
        panel[label] = panel["price_entry"] / past_price - 1

    # 学術的に定石の「12-1ヶ月モメンタム」: 直近1ヶ月を除外し、
    # signal_date-1ヶ月 時点 と signal_date-12ヶ月 時点 の間の騰落率を見る
    skip_date = (panel["signal_date"] - pd.Timedelta(days=30)).rename("skip_date")
    tmp_skip = pd.concat([panel["ticker"], skip_date], axis=1)
    price_skip = asof_price(tmp_skip, "skip_date", "backward", tolerance_days=21)

    lookback_12m_date = (panel["signal_date"] - pd.Timedelta(days=365)).rename("lookback_12m_date")
    tmp_12m = pd.concat([panel["ticker"], lookback_12m_date], axis=1)
    price_12m_ago = asof_price(tmp_12m, "lookback_12m_date", "backward", tolerance_days=21)

    panel["momentum_12_1"] = price_skip / price_12m_ago - 1

    for label, days in FORWARD_WINDOWS_DAYS.items():
        target_date = (panel["signal_date"] + pd.Timedelta(days=days)).rename("target_date")
        tmp = pd.concat([panel["ticker"], target_date], axis=1)
        exit_price = asof_price(tmp, "target_date", "forward", tolerance_days=21)
        panel[f"forward_return_{label}"] = exit_price / panel["price_entry"] - 1

    return panel


def add_valuation_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    bvps = panel["eps"] / panel["roe"]
    panel["per"] = panel["price_entry"] / panel["eps"]
    panel["pbr"] = panel["price_entry"] / bvps
    for col in ("per", "pbr"):
        panel.loc[(panel[col] <= 0) | (panel[col] > 500), col] = None  # 赤字銘柄等の異常値を除外
    return panel


def compute_score_per_cohort(df: pd.DataFrame, category_metrics: dict, category_weights: dict, score_col: str) -> pd.DataFrame:
    result_parts = []
    for cohort, df_cohort in df.groupby("cohort_year"):
        df_cohort = df_cohort.copy()
        sector_counts = df_cohort["sector"].value_counts().to_dict()
        df_cohort["comparison_group"] = df_cohort["sector"].apply(
            lambda s: s if sector_counts.get(s, 0) >= MIN_SECTOR_SAMPLE_SIZE else "__MARKET_WIDE__"
        )
        category_scores = {cat: pd.Series(index=df_cohort.index, dtype=float) for cat in category_metrics}
        category_conf = {cat: pd.Series(index=df_cohort.index, dtype=float) for cat in category_metrics}

        for group, df_group in df_cohort.groupby("comparison_group"):
            for category, metrics in category_metrics.items():
                weighted_sum = pd.Series(0.0, index=df_group.index)
                total_weight = pd.Series(0.0, index=df_group.index)
                weight_each = 1.0 / len(metrics)
                for col, higher_is_better in metrics:
                    pct = percentile_rank(df_group[col], higher_is_better)
                    valid = pct.notna()
                    weighted_sum[valid] += pct[valid] * weight_each
                    total_weight[valid] += weight_each
                scores = (weighted_sum / total_weight).where(total_weight > 0)
                category_scores[category].loc[df_group.index] = scores
                category_conf[category].loc[df_group.index] = total_weight

        for category in category_metrics:
            df_cohort[f"score_{category}"] = category_scores[category]
            df_cohort[f"confidence_{category}"] = category_conf[category]

        weighted_sum = pd.Series(0.0, index=df_cohort.index)
        total_weight = pd.Series(0.0, index=df_cohort.index)
        for category, w in category_weights.items():
            score = df_cohort[f"score_{category}"]
            conf = df_cohort[f"confidence_{category}"]
            valid = score.notna() & conf.notna()
            eff_w = w * conf
            weighted_sum[valid] += (score * eff_w)[valid]
            total_weight[valid] += eff_w[valid]
        df_cohort[score_col] = (weighted_sum / total_weight).where(total_weight > 0)
        result_parts.append(df_cohort)
    return pd.concat(result_parts, ignore_index=True)


def report(df: pd.DataFrame, score_col: str, return_col: str, label: str) -> None:
    valid = df.dropna(subset=[score_col, return_col])
    if len(valid) < 50:
        print(f"[{label}] サンプル不足({len(valid)}件)")
        return
    ic = spearmanr(valid[score_col], valid[return_col]).correlation
    valid = valid.copy()
    valid["quintile"] = pd.qcut(valid[score_col].rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    q_mean = valid.groupby("quintile", observed=True)[return_col].mean()
    spread = q_mean["Q5"] - q_mean["Q1"]
    print(f"[{label}] n={len(valid)}  IC={ic:+.4f}  Q5-Q1スプレッド={spread:+.4f}  (Q1={q_mean['Q1']:+.4f} / Q5={q_mean['Q5']:+.4f})")


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

    # --- ベースライン(クオリティ5軸のみ、12ヶ月フォワード) ---
    baseline_metrics = {k: v for k, v in CATEGORY_METRICS_V2.items() if k not in ("value", "momentum")}
    baseline_weights = {k: 1.0 / len(baseline_metrics) for k in baseline_metrics}
    panel = compute_score_per_cohort(panel, baseline_metrics, baseline_weights, "quality_score_baseline")

    # --- クオリティ+バリュー+モメンタム(6軸) ---
    panel = compute_score_per_cohort(panel, CATEGORY_METRICS_V2, CATEGORY_WEIGHTS_V2, "quality_score_v2")

    print("=== フォワード期間別の比較(全コホート合算) ===")
    for label in FORWARD_WINDOWS_DAYS:
        return_col = f"forward_return_{label}"
        report(panel, "quality_score_baseline", return_col, f"ベースライン(5軸) / {label}後")
        report(panel, "quality_score_v2", return_col, f"v2(バリュー+モメンタム込み7軸) / {label}後")
        print()

    print("=== 12ヶ月後リターンでの単軸IC(バリュー・モメンタムがどれだけ効くか) ===")
    valid = panel.dropna(subset=["forward_return_12ヶ月"])
    for category in CATEGORY_METRICS_V2:
        col = f"score_{category}"
        if col not in panel.columns:
            continue
        sub = valid.dropna(subset=[col])
        if len(sub) < 50:
            continue
        ic = spearmanr(sub[col], sub["forward_return_12ヶ月"]).correlation
        print(f"  {category:20s}: IC={ic:+.4f} (n={len(sub)})")

    print("\n=== モメンタムの計算方法比較(直近1ヶ月を含む/除外した12-1ヶ月) ===")
    for col in ("momentum_12m", "momentum_6m", "momentum_12_1"):
        sub = valid.dropna(subset=[col])
        ic = spearmanr(sub[col], sub["forward_return_12ヶ月"]).correlation
        print(f"  {col:20s}: IC={ic:+.4f} (n={len(sub)})")

    print("\n=== v2スコアのコホート別IC(12ヶ月後、頑健性の確認) ===")
    for cohort, sub in panel.dropna(subset=["quality_score_v2", "forward_return_12ヶ月"]).groupby("cohort_year"):
        if len(sub) < 50:
            continue
        ic = spearmanr(sub["quality_score_v2"], sub["forward_return_12ヶ月"]).correlation
        print(f"  {cohort}年度: IC={ic:+.4f} (n={len(sub)})")

    conn.close()
    bt_conn.close()


if __name__ == "__main__":
    main()
