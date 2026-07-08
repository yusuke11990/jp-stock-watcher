"""price_dailyの鮮度チェック(fetch_price_daily.pyとdaily_decision.ymlの両方から使う共通ロジック)。

yfinanceをGitHub Actionsのような共有IPのクラウド環境から呼ぶと、直近数日分だけ
Close(終値)がNaNで返ってきて静かにスキップされることがあり、その場合でも
「5日分のうちどれかは非空」なのでticker単位のfetch_logは全件successのまま記録されて
しまう(実際に本番で発生した)。全体のMAX(date)だけ見ても、ごく一部の銘柄だけ新しければ
「新鮮」に見えて隠れてしまうため、銘柄ごとの遅れを個別に集計し、一定割合以上が
遅延していればstale(鮮度異常)と判定する。

価格が古いまま放置されると、compute_scores.py/decide_rule.py/decide_composite.pyが
古い株価を元にスコア・売買判断を計算してしまうため、daily_decision.yml側でも
このチェックを流用する。
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

JST = timezone(timedelta(hours=9))


def check_freshness(
    conn, max_stale_business_days: int = 1, stale_ticker_ratio_threshold: float = 0.1
) -> tuple[bool, str]:
    """銘柄ごとの最新取得日が本日から何営業日遅れているかを見て、鮮度を判定する。"""
    today = datetime.now(JST).date()
    df = pd.read_sql_query("SELECT ticker, MAX(date) AS max_date FROM price_daily GROUP BY ticker", conn)
    if df.empty:
        return True, "price_dailyにデータがありません"

    df["max_date"] = pd.to_datetime(df["max_date"]).dt.date
    df["business_days_behind"] = df["max_date"].apply(
        lambda d: int(np.busday_count(d, today))
    )
    stale_count = int((df["business_days_behind"] > max_stale_business_days).sum())
    stale_ratio = stale_count / len(df)
    is_stale = stale_ratio > stale_ticker_ratio_threshold
    msg = (
        f"価格データ鮮度チェック: 全{len(df)}銘柄中{stale_count}件({stale_ratio:.0%})が"
        f"{max_stale_business_days}営業日を超えて遅延しています(最新の取得日: {df['max_date'].max()})"
    )
    return is_stale, msg
