"""当日のbuy/sell判断をDiscord Webhookへ通知する。

複数銘柄を1メッセージのEmbed配列にまとめてAPI呼び出し回数を抑制する。
Webhook URLは環境変数DISCORD_WEBHOOK_URL(GitHub Secretsに格納)から読む。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.db import get_connection  # noqa: E402

MAX_EMBEDS_PER_MESSAGE = 10  # Discordの1メッセージあたりembed上限は10

ACTION_COLOR = {"buy": 0x2ECC71, "sell": 0xE74C3C}  # 緑/赤。holdは通知しない
ACTION_LABEL = {"buy": "買い", "sell": "売り"}


def load_today_actionable_decisions(conn, decision_date: str):
    # v2(decide_composite.py)は複数年バックテストで技術タイミング層の頑健性が
    # 確認できなかったため通知対象に含めない。v3(decide_quality_timing.py)は
    # グレード×RSI/BB反発の組み合わせが2021〜2026年の全年で頑健と確認済みのため
    # v1と合わせて通知する。v1/v2/v3の実績比較はrule_performance.pyで確認する。
    query = """
    SELECT d.ticker, t.name, d.action, d.grade, d.total_score, d.reason, d.price_at_decision, d.confidence,
           d.rule_version
    FROM decisions d
    JOIN tickers t ON t.ticker = d.ticker
    WHERE d.decision_date = ? AND d.decision_source = 'rule' AND d.rule_version IN ('v1.0', 'v3.0')
      AND d.action IN ('buy', 'sell')
    ORDER BY d.rule_version, d.confidence DESC
    """
    cols = ["ticker", "name", "action", "grade", "total_score", "reason", "price_at_decision", "confidence", "rule_version"]
    return [dict(zip(cols, row)) for row in conn.execute(query, (decision_date,))]


def build_embed(decision: dict) -> dict:
    name = decision.get("name") or decision["ticker"]
    return {
        "title": f"{ACTION_LABEL[decision['action']]}：{name}({decision['ticker']})",
        "description": decision["reason"],
        "color": ACTION_COLOR[decision["action"]],
        "fields": [
            {"name": "グレード", "value": decision["grade"] or "-", "inline": True},
            {"name": "総合スコア", "value": f"{decision['total_score']:.1f}" if decision["total_score"] is not None else "-", "inline": True},
            {"name": "株価", "value": f"{decision['price_at_decision']:,.0f}" if decision["price_at_decision"] else "-", "inline": True},
            {"name": "確信度", "value": f"{decision['confidence']:.0%}", "inline": True},
            {"name": "エンジン", "value": decision["rule_version"], "inline": True},
        ],
    }


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def send_to_discord(webhook_url: str, decisions: list[dict], dry_run: bool = False) -> None:
    if not decisions:
        payload = {"content": "本日は買い/売りシグナルの発生した銘柄がありませんでした。"}
        payloads = [payload]
    else:
        payloads = []
        for chunk in chunked(decisions, MAX_EMBEDS_PER_MESSAGE):
            payloads.append({"embeds": [build_embed(d) for d in chunk]})

    for i, payload in enumerate(payloads):
        if dry_run:
            print(payload)
            continue
        if i > 0:
            # Discordのwebhookレート制限(概ね5リクエスト/2秒)に引っかからないよう、
            # 通知件数が多い日にメッセージを連続送信する際は間隔を空ける
            time.sleep(1)
        for attempt in range(3):
            resp = requests.post(webhook_url, json=payload, timeout=15)
            if resp.status_code == 429:
                # 429時はDiscordが返すRetry-After(秒)だけ待ってから再試行する。
                # これをせずraise_for_status()を素通しすると、通知失敗が
                # ワークフロー全体をexit code 1で止めてしまう(実際に本番で発生した)
                retry_after = float(resp.headers.get("Retry-After", 2))
                print(f"  レート制限(429)。{retry_after:.1f}秒待って再試行します({attempt + 1}/3)")
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            break
        else:
            resp.raise_for_status()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-date", default=None)
    parser.add_argument("--dry-run", action="store_true", help="送信せずペイロードを表示するのみ")
    args = parser.parse_args()

    conn = get_connection()
    decision_date = args.decision_date or conn.execute("SELECT MAX(decision_date) FROM decisions").fetchone()[0]
    if decision_date is None:
        print("decisionsにデータがありません")
        return

    decisions = load_today_actionable_decisions(conn, decision_date)
    conn.close()

    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url and not args.dry_run:
        print("環境変数DISCORD_WEBHOOK_URLが未設定です。--dry-runで確認するか設定してください")
        sys.exit(1)

    send_to_discord(webhook_url or "", decisions, dry_run=args.dry_run)
    print(f"decision_date={decision_date}: {len(decisions)}件のbuy/sellを通知{'(dry-run)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
