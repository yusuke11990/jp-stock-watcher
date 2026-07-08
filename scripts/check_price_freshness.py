"""daily_decision.yml用: compute_scores.py/decide_rule.py/decide_composite.pyが
古い株価のままスコア・売買判断を計算していないかを確認する。

fetch_price_daily.pyの取得ステップ自体は成功していても(鮮度異常時はそちらのジョブが
既に失敗表示になる)、そのジョブと本ジョブは別ワークフロー・別実行なので、
判断エンジン側でも独立して確認しておかないと「価格ジョブは失敗表示だが判断ジョブは
成功表示のまま、古い株価に基づくbuy/sellがDiscordに通知される」という抜け道になる。

実行: python scripts/check_price_freshness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.db import get_connection  # noqa: E402
from common.freshness import check_freshness  # noqa: E402


def main():
    conn = get_connection()
    is_stale, msg = check_freshness(conn)
    conn.close()
    print(msg)
    if is_stale:
        print("⚠ 価格データが古いまま計算されている可能性があります。"
              "compute_scores.py/decide_rule.py/decide_composite.pyの結果は"
              "古い株価に基づいている可能性が高いです。")
        sys.exit(1)


if __name__ == "__main__":
    main()
