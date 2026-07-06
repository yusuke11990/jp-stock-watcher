"""ファンダ6軸スコア×テクニカルv2(相関の低い複数シグナルファミリー)の合成判断エンジン。

decide_rule.py(v1.0、ルールベースの二値シグナル判定)は変更せず残し、
本エンジンはrule_version="v2.0"として同日・同銘柄でも別レコードとして
decisionsテーブルに共存させる(UNIQUE制約はrule_version込みに移行済み)。

重みはconfig/technical_config.yaml・config/decision_config.yamlで管理する。
backtest/配下での検証で、1年分データへの重み最適化は時系列的に不安定
(前半→後半でIC符号反転)と分かったため、意図的に保守的な重みを採用している。
今後はevaluate_decisions.py/rule_performance.pyの実績を見て手動調整する。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import get_connection  # noqa: E402
from common.technical_v2 import compute_technical_v2  # noqa: E402
from common.regime import market_regime_score, sector_regime_score  # noqa: E402

RULE_VERSION = "v2.0"
JST = timezone(timedelta(hours=9))
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_configs() -> tuple[dict, dict]:
    with open(CONFIG_DIR / "technical_config.yaml", encoding="utf-8") as f:
        technical_config = yaml.safe_load(f)
    with open(CONFIG_DIR / "decision_config.yaml", encoding="utf-8") as f:
        decision_config = yaml.safe_load(f)
    return technical_config, decision_config


@dataclass
class Decision:
    action: str
    reason: str
    confidence: float
    final_score: float
    technical_composite: float


def load_price_history(conn, ticker: str) -> pd.DataFrame:
    query = "SELECT date, open, high, low, close, volume FROM price_daily WHERE ticker = ? ORDER BY date ASC"
    df = pd.read_sql_query(query, conn, params=(ticker,), parse_dates=["date"])
    if df.empty:
        return df
    df = df.set_index("date").rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    return df


def normalize_fundamental(total_score: float | None) -> float:
    """0-100の総合スコアを-1~+1に変換する"""
    if total_score is None or pd.isna(total_score):
        return 0.0
    return (total_score - 50) / 50


def decide_composite(
    total_score: float | None,
    technical_scores: dict,
    market_regime: dict,
    sector_regime: dict,
    technical_config: dict,
    decision_config: dict,
) -> Decision:
    weights = technical_config["family_weights"]
    technical_composite = (
        technical_scores["trend_score"] * weights["trend_score"]
        + technical_scores["mean_reversion_score"] * weights["mean_reversion_score"]
        + technical_scores["volume_score"] * weights["volume_score"]
        + market_regime["topix_trend_score"] * weights["market_regime_score"]
        + sector_regime["sector_regime_score"] * weights["sector_regime_score"]
    )

    comp = decision_config["composite"]
    fundamental_norm = normalize_fundamental(total_score)
    final_score = comp["fundamental_weight"] * fundamental_norm + comp["technical_weight"] * technical_composite

    thresholds = decision_config["action_thresholds"]
    if final_score >= thresholds["buy"]:
        action = "buy"
    elif final_score <= thresholds["sell"]:
        action = "sell"
    else:
        action = "hold"

    scale = decision_config["confidence"]["scale"]
    confidence = min(1.0, abs(final_score) / scale)

    reason = (
        f"合成スコア{final_score:+.2f}(ファンダ{fundamental_norm:+.2f}×{comp['fundamental_weight']:.2f} + "
        f"テクニカル{technical_composite:+.2f}×{comp['technical_weight']:.2f})。"
        f"レジーム={technical_scores['regime']['regime']}、業種地合い={sector_regime['sector_regime_score']:+.2f}"
    )
    return Decision(action, reason, confidence, final_score, technical_composite)


def compute_risk_levels(close: float | None, atr: float | None, action: str, decision_config: dict) -> dict:
    risk = decision_config["risk_overlay"]
    if close is None or atr is None or action == "hold":
        return {"stop_loss_price": None, "take_profit_price": None, "risk_reward_ratio": None}

    if action == "buy":
        stop_loss = close - atr * risk["atr_stop_multiplier"]
        take_profit = close + atr * risk["atr_take_profit_multiplier"]
    else:  # sell(保有中への警戒ラインとして同様に算出)
        stop_loss = close + atr * risk["atr_stop_multiplier"]
        take_profit = close - atr * risk["atr_take_profit_multiplier"]

    risk_amount = abs(close - stop_loss)
    reward_amount = abs(take_profit - close)
    rr_ratio = (reward_amount / risk_amount) if risk_amount > 0 else None
    return {
        "stop_loss_price": round(stop_loss, 1),
        "take_profit_price": round(take_profit, 1),
        "risk_reward_ratio": round(rr_ratio, 2) if rr_ratio is not None else None,
    }


def upsert_decision(conn, ticker: str, decision_date: str, decision: Decision, total_score, grade,
                     technical_scores: dict, market_regime: dict, sector_regime: dict, risk: dict, price: float | None) -> None:
    signal_scores_v2 = json.dumps({
        "trend_score": technical_scores["trend_score"],
        "mean_reversion_score": technical_scores["mean_reversion_score"],
        "volume_score": technical_scores["volume_score"],
        "market_regime_score": market_regime["topix_trend_score"],
        "sector_regime_score": sector_regime["sector_regime_score"],
        "regime": technical_scores["regime"]["regime"],
    }, ensure_ascii=False)

    with conn:
        conn.execute(
            """
            INSERT INTO decisions
                (ticker, decision_date, action, decision_source, rule_version,
                 total_score, grade, technical_signals, reason, price_at_decision, confidence,
                 stop_loss_price, take_profit_price, risk_reward_ratio,
                 technical_composite_score, regime_market, signal_scores_v2)
            VALUES (?, ?, ?, 'rule', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, decision_date, decision_source, rule_version) DO UPDATE SET
                action=excluded.action, total_score=excluded.total_score, grade=excluded.grade,
                technical_signals=excluded.technical_signals, reason=excluded.reason,
                price_at_decision=excluded.price_at_decision, confidence=excluded.confidence,
                stop_loss_price=excluded.stop_loss_price, take_profit_price=excluded.take_profit_price,
                risk_reward_ratio=excluded.risk_reward_ratio,
                technical_composite_score=excluded.technical_composite_score,
                regime_market=excluded.regime_market, signal_scores_v2=excluded.signal_scores_v2
            """,
            (
                ticker, decision_date, decision.action, RULE_VERSION,
                total_score, grade, signal_scores_v2, decision.reason, price, decision.confidence,
                risk["stop_loss_price"], risk["take_profit_price"], risk["risk_reward_ratio"],
                decision.technical_composite, technical_scores["regime"]["regime"], signal_scores_v2,
            ),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    technical_config, decision_config = load_configs()
    conn = get_connection()
    snapshot_date = args.snapshot_date or conn.execute("SELECT MAX(snapshot_date) FROM scores").fetchone()[0]
    if snapshot_date is None:
        print("scoresにデータがありません。先にcompute_scores.pyを実行してください")
        return

    scores_df = pd.read_sql_query(
        "SELECT ticker, sector, total_score, grade FROM scores WHERE snapshot_date = ?",
        conn, params=(snapshot_date,),
    )
    if args.limit:
        scores_df = scores_df.head(args.limit)

    decision_date = datetime.now(JST).strftime("%Y-%m-%d")
    market_regime = market_regime_score(conn)
    action_counts = {"buy": 0, "sell": 0, "hold": 0}
    skipped = 0

    for _, row in scores_df.iterrows():
        ticker = row["ticker"]
        price_df = load_price_history(conn, ticker)
        technical_scores = compute_technical_v2(price_df)
        if technical_scores["close"] is None:
            skipped += 1
            continue

        sector_regime = sector_regime_score(conn, row["sector"], decision_date)
        decision = decide_composite(
            row["total_score"], technical_scores, market_regime, sector_regime,
            technical_config, decision_config,
        )
        risk = compute_risk_levels(technical_scores["close"], technical_scores["atr"], decision.action, decision_config)
        upsert_decision(conn, ticker, decision_date, decision, row["total_score"], row["grade"],
                         technical_scores, market_regime, sector_regime, risk, technical_scores["close"])
        action_counts[decision.action] += 1

    conn.close()
    print(f"snapshot_date={snapshot_date}, decision_date={decision_date}, rule_version={RULE_VERSION}")
    print(f"action分布: {action_counts}, データ不足でスキップ: {skipped}")


if __name__ == "__main__":
    main()
