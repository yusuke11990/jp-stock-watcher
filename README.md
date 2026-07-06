# jp-stock-watcher

日本株全上場銘柄(約3,548銘柄、ETF/REIT/PRO Market除外)を対象に、yfinanceでデータを取得し、
業種内相対評価による6軸スコアリング(安全性・成長性・収益性・効率性・割安性・還元性)とテクニカルシグナルを
組み合わせたルールベースの売買判断を毎日自動実行するツール。判断結果は1ヶ月後に自動評価され、
成績サマリーがルール改善の参考情報として蓄積されていく。

## 構成

```
scripts/
  common/
    db.py           # SQLite接続・スキーマ初期化(scores/decisions/decision_outcomes等)
    yf_client.py     # yfinance呼び出しの共通リトライ・ブロック検知
    technical.py      # テクニカルシグナル判定(GC/RSI/MACD/BB/出来高)
  fetch_tickers.py     # JPX上場銘柄一覧の取得(月次)
  fetch_price_daily.py # 価格の日次バッチ取得
  fetch_fundamentals.py # ファンダメンタルズのローリング取得(週次相当)
  scoring/compute_scores.py   # 業種内相対評価による6軸スコアリング
  decisions/decide_rule.py    # ルールベース売買判断
  evaluate_decisions.py       # 判断から1ヶ月後の自動評価
  reports/rule_performance.py # 的中率サマリー
  notify_discord.py           # Discord Webhook通知
config/scoring_config.yaml    # スコアリングの重み・閾値(コード変更なしで調整可能)
data/stock.db                 # SQLite本体(リポジトリにコミットして永続化)
.github/workflows/            # GitHub Actions(スケジュール実行)
```

## セットアップ(GitHub Actionsで自動運用する場合)

このツールはローカルで一通り動作確認済みですが、日次自動実行にはGitHub側の設定が必要です。

1. **GitHubリポジトリを作成**(private推奨。保有銘柄・判断ログ等の個人情報を含むため)
   ```
   cd "jp-stock-watcher"
   git remote add origin <あなたのリポジトリURL>
   git push -u origin main
   ```
2. **Discord Webhookを発行**: Discordサーバーの通知先チャンネルの「連携サービス」→「ウェブフックを作成」でURLを取得
3. **GitHub Secretsに登録**: リポジトリの Settings → Secrets and variables → Actions → New repository secret
   - 名前: `DISCORD_WEBHOOK_URL`
   - 値: 手順2で取得したURL
4. 上記2つが終われば、`.github/workflows/`配下の5つのワークフローが以下のスケジュールで自動実行される:
   - `fetch_tickers.yml`: 毎月1日、銘柄一覧更新
   - `fetch_price_daily.yml`: 平日引け後、価格の日次バッチ取得
   - `fetch_fundamentals.yml`: 平日、ファンダメンタルズのローリング更新(時間予算30分)
   - `daily_decision.yml`: 平日、スコアリング→ルールベース判断→Discord通知
   - `evaluate_decisions.yml`: 平日、1ヶ月前の判断を評価し成績サマリーをログ出力

初回のみ、価格の長期履歴(テクニカル指標の計算にMA75等で75日以上必要)をバックフィルする:
```
python scripts/fetch_price_daily.py --period 1y
```

## ローカルでの手動実行

```
pip install -r requirements.txt
python scripts/common/db.py            # DB初期化
python scripts/fetch_tickers.py        # 銘柄一覧取得
python scripts/fetch_price_daily.py --period 1y   # 初回は長期分でバックフィル
python scripts/fetch_fundamentals.py   # ファンダメンタルズ取得(--limitで件数制限可)
python scripts/scoring/compute_scores.py
python scripts/decisions/decide_rule.py
python scripts/notify_discord.py --dry-run   # Webhook未設定でも構造確認可能
python scripts/evaluate_decisions.py
python scripts/reports/rule_performance.py
```

## 設計上の注意点

- 配当利回り(`dividend_yield`)はyfinanceの仕様上%表記(2.89 = 2.89%)で返る
- スコアは同一業種(東証33業種)内でのパーセンタイル順位を基準にした相対評価。業種内5銘柄未満は市場全体にフォールバック
- 指標が欠損している銘柄は信頼度(`confidence_*`)が下がり、総合スコアへの影響が自動的に割り引かれる(銀行業の流動比率欠損など)
- 売買判断(`decide_rule.py`)は`decision_source='rule'`で記録。将来LLM(Claude API)による判断を追加する場合は`decision_source='llm'`として同じテーブルに並行記録できる設計
- ルールの自動チューニングは行わない。`rule_performance.py`の出力を見て`config/scoring_config.yaml`や`decide_rule.py`の閾値を手動調整する運用を想定
