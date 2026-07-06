"""業種内相対評価による6軸スコアリングエンジン。

各指標を同一sector内でパーセンタイル化(0-100)し、カテゴリごとに
信頼度加重で集約する。サンプル数が少ない業種(min_sector_sample_size未満)は
市場全体(__MARKET_WIDE__)にフォールバックする。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "scoring_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_latest_snapshot_date(conn) -> str | None:
    row = conn.execute("SELECT MAX(snapshot_date) FROM fundamentals_weekly").fetchone()
    return row[0] if row else None


def load_data(conn, snapshot_date: str) -> pd.DataFrame:
    """スコア計算日(snapshot_date、resultのラベル用)は指定日を使うが、
    元データは「各銘柄が実際に取得できた最新のfundamentals_weekly行」を使う。
    ローリング更新は日によって異なる銘柄群を取得するため、単一の日付で絞ると
    その日だけ取得された数銘柄しかスコア化できなくなってしまうため。
    """
    query = """
    SELECT f.*, t.sector
    FROM fundamentals_weekly f
    JOIN (
        SELECT ticker, MAX(snapshot_date) AS latest_date
        FROM fundamentals_weekly
        WHERE snapshot_date <= ?
        GROUP BY ticker
    ) latest ON f.ticker = latest.ticker AND f.snapshot_date = latest.latest_date
    JOIN tickers t ON f.ticker = t.ticker
    WHERE t.is_active = 1
    """
    df = pd.read_sql_query(query, conn, params=(snapshot_date,))
    df["interest_bearing_debt_to_market_cap"] = df.apply(
        lambda r: (r["interest_bearing_debt"] / r["market_cap"])
        if pd.notna(r["interest_bearing_debt"]) and pd.notna(r["market_cap"]) and r["market_cap"] != 0
        else None,
        axis=1,
    )
    return df


def percentile_rank(series: pd.Series, higher_is_better: bool) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(index=series.index, dtype=float)
    ranks = valid.rank(pct=True) * 100
    if not higher_is_better:
        ranks = 100 - ranks
    return ranks.reindex(series.index)


def score_payout_ratio(series: pd.Series, group_percentile: pd.Series) -> pd.Series:
    result = pd.Series(index=series.index, dtype=float)
    for idx, val in series.items():
        if pd.isna(val) or val <= 0:
            result[idx] = 0
        elif val <= 0.60:
            result[idx] = group_percentile.get(idx)
        elif val <= 1.00:
            result[idx] = 70
        else:
            result[idx] = 30
    return result


def compute_category_score(df_group: pd.DataFrame, metric_defs: list[dict]) -> tuple[pd.Series, pd.Series]:
    percentile_cols = {}
    for m in metric_defs:
        col = m["col"]
        if m.get("special") == "payout_ratio_logic":
            normal_pct = percentile_rank(df_group[col], higher_is_better=True)
            percentile_cols[col] = score_payout_ratio(df_group[col], normal_pct)
        else:
            percentile_cols[col] = percentile_rank(df_group[col], m["higher_is_better"])

    scores = pd.Series(index=df_group.index, dtype=float)
    confidences = pd.Series(index=df_group.index, dtype=float)
    total_defined_weight = sum(m["weight"] for m in metric_defs)

    for idx in df_group.index:
        weighted_sum, total_weight = 0.0, 0.0
        for m in metric_defs:
            val = percentile_cols[m["col"]].get(idx)
            if pd.notna(val):
                weighted_sum += val * m["weight"]
                total_weight += m["weight"]
        if total_weight > 0:
            scores[idx] = weighted_sum / total_weight
            confidences[idx] = total_weight / total_defined_weight
        else:
            scores[idx] = None
            confidences[idx] = 0.0

    return scores, confidences


def get_comparison_group(sector: str, sector_counts: dict, min_size: int) -> str:
    return sector if sector_counts.get(sector, 0) >= min_size else "__MARKET_WIDE__"


def score_to_grade(total_score, thresholds: dict) -> str | None:
    if total_score is None or pd.isna(total_score):
        return None
    for grade in ("S", "A", "B", "C", "D"):
        if total_score >= thresholds[grade]:
            return grade
    return "E"


def compute_total_score(row: dict, category_weights: dict) -> float | None:
    weighted_sum, total_weight = 0.0, 0.0
    for cat, w in category_weights.items():
        score = row.get(f"score_{cat}")
        conf = row.get(f"confidence_{cat}")
        if score is not None and pd.notna(score) and conf is not None and pd.notna(conf):
            effective_weight = w * conf
            weighted_sum += score * effective_weight
            total_weight += effective_weight
    return weighted_sum / total_weight if total_weight > 0 else None


def compute_scores_for_snapshot(conn, snapshot_date: str, config: dict) -> pd.DataFrame:
    df = load_data(conn, snapshot_date)
    if df.empty:
        return df

    min_size = config["min_sector_sample_size"]
    sector_counts = df["sector"].value_counts().to_dict()
    df["comparison_group"] = df["sector"].apply(lambda s: get_comparison_group(s, sector_counts, min_size))

    category_metrics = config["category_metrics"]
    category_weights = config["category_weights"]
    grade_thresholds = config["grade_thresholds"]

    category_score_series = {cat: pd.Series(index=df.index, dtype=float) for cat in category_metrics}
    category_conf_series = {cat: pd.Series(index=df.index, dtype=float) for cat in category_metrics}

    for group, df_group in df.groupby("comparison_group"):
        for category, metric_defs in category_metrics.items():
            scores, confidences = compute_category_score(df_group, metric_defs)
            category_score_series[category].loc[df_group.index] = scores
            category_conf_series[category].loc[df_group.index] = confidences

    result_rows = []
    for idx in df.index:
        row = {
            "ticker": df.loc[idx, "ticker"],
            "snapshot_date": snapshot_date,
            "sector": df.loc[idx, "sector"],
        }
        for category in category_metrics:
            row[f"score_{category}"] = category_score_series[category].get(idx)
            row[f"confidence_{category}"] = category_conf_series[category].get(idx)
        row["total_score"] = compute_total_score(row, category_weights)
        row["grade"] = score_to_grade(row["total_score"], grade_thresholds)
        result_rows.append(row)

    result_df = pd.DataFrame(result_rows)
    result_df["sector_median_score"] = result_df.groupby("sector")["total_score"].transform("median")
    result_df["sector_rank"] = result_df.groupby("sector")["total_score"].rank(ascending=False, method="min")
    result_df["sector_size"] = result_df.groupby("sector")["ticker"].transform("count")
    return result_df


def upsert_scores(conn, df: pd.DataFrame) -> None:
    if df.empty:
        return
    cols = [c for c in df.columns]
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in ("ticker", "snapshot_date"))
    with conn:
        for _, row in df.iterrows():
            values = [None if pd.isna(v) else v for v in row[cols]]
            conn.execute(
                f"""
                INSERT INTO scores ({", ".join(cols)}) VALUES ({placeholders})
                ON CONFLICT(ticker, snapshot_date) DO UPDATE SET {updates}
                """,
                values,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", default=None, help="対象snapshot_date(省略時は最新)")
    args = parser.parse_args()

    conn = get_connection()
    config = load_config()
    snapshot_date = args.snapshot_date or load_latest_snapshot_date(conn)
    if snapshot_date is None:
        print("fundamentals_weeklyにデータがありません")
        return

    result_df = compute_scores_for_snapshot(conn, snapshot_date, config)
    upsert_scores(conn, result_df)
    conn.close()

    print(f"snapshot_date={snapshot_date}: {len(result_df)}件のスコアを計算")
    print(result_df["grade"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
