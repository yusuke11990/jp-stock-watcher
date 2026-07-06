"""当日のbuy/sell判断をDiscord Webhookへ通知する。

複数銘柄を1メッセージのEmbed配列にまとめてAPI呼び出し回数を抑制する。
Webhook URLは環境変数DISCORD_WEBHOOK_URL(GitHub Secretsに格納)から読む。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.db import get_connection  # noqa: E402

MAX_EMBEDS_PER_MESSAGE = 10  # Discordの1メッセージあたりembed上限は10

ACTION_COLOR = {"buy": 0x2ECC71, "sell": 0xE74C3C}  # 緑/赤。holdは通知しない
ACTION_LABEL = {"buy": "買い", "sell": "売り"}


def load_today_actionable_decisions(conn, decision_date: str):
    query = """
    SELECT ticker, action, grade, total_score, reason, price_at_decision, confidence
    FROM decisions
    WHERE decision_date = ? AND decision_source = 'rule' AND action IN ('buy', 'sell')
    ORDER BY confidence DESC
    """
    cols = ["ticker", "action", "grade", "total_score", "reason", "price_at_decision", "confidence"]
    return [dict(zip(cols, row)) for row in conn.execute(query, (decision_date,))]


def build_embed(decision: dict) -> dict:
    return {
        "title": f"{ACTION_LABEL[decision['action']]}: {decision['ticker']} (grade {decision['grade']})",
        "description": decision["reason"],
        "color": ACTION_COLOR[decision["action"]],
        "fields": [
            {"name": "総合スコア", "value": f"{decision['total_score']:.1f}" if decision["total_score"] is not None else "-", "inline": True},
            {"name": "株価", "value": f"¥{decision['price_at_decision']:,.0f}" if decision["price_at_decision"] else "-", "inline": True},
            {"name": "確信度", "value": f"{decision['confidence']:.0%}", "inline": True},
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

    for payload in payloads:
        if dry_run:
            print(payload)
            continue
        resp = requests.post(webhook_url, json=payload, timeout=15)
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
