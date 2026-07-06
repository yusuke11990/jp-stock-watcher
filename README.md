# jp-stock-watcher

日本株全上場銘柄(約3,548銘柄、ETF/REIT/PRO Market除外)を対象に、yfinanceでデータを取得し、
業種内相対評価による6軸スコアリング(安全性・成長性・収益性・効率性・割安性・還元性)とテクニカルシグナルを
組み合わせたルールベースの売買判断を毎日自動実行するツール。判断結果は1ヶ月後に自動評価され、
成績サマリーがルール改善の参考情報として蓄積されていく。

## 構成

```
scripts/
  common/
    db.py           # SQLite接続・スキーマ初期化(scores/decisions/decision_outcomes/signal_history等)
    yf_client.py     # yfinance呼び出しの共通リトライ・ブロック検知
    technical.py      # テクニカルシグナル判定v1(GC/RSI/MACD/BB/出来高、二値)
    technical_v2.py    # テクニカル判断エンジンv2(trend/mean_reversion/volumeの連続値スコア)
    regime.py           # 市場(TOPIX)・セクターのレジーム判定(v2用)
  fetch_tickers.py     # JPX上場銘柄一覧の取得(月次)
  fetch_price_daily.py # 価格の日次バッチ取得
  fetch_fundamentals.py # ファンダメンタルズのローリング取得(週次相当)
  fetch_edinet_history.py # EDINETから有価証券報告書を遡り、yfinanceの4〜5年を超える長期の業績推移を取得
  scoring/compute_scores.py   # 業種内相対評価による6軸スコアリング
  decisions/
    decide_rule.py      # ルールベース売買判断(v1.0、グレード×二値シグナル)
    decide_composite.py # 合成判断エンジン(v2.0、ファンダ×連続値テクニカルスコア、ATRリスクオーバーレイ)
  backtest/              # v2のシグナル検証・重みチューニング用(本番運用とは別系統)
    build_signal_history.py  # 全銘柄×全営業日のシグナルスコアをsignal_historyに構築
    event_study.py            # ファミリー間相関・quintile分析
    weight_optimizer.py        # IC最大化による重み最適化
    cross_validate.py           # 時系列2分割・銘柄k-foldでの頑健性検証
    regime_adaptive_validate.py # ADXレジーム適応型 vs 静的重みの比較検証
  evaluate_decisions.py       # 判断から1ヶ月後の自動評価
  reports/rule_performance.py # 的中率サマリー(rule_version別にv1/v2を比較)
  notify_discord.py           # Discord Webhook通知(現在はv1のみ通知、v2は検証中)
config/
  scoring_config.yaml    # ファンダ6軸スコアリングの重み・閾値
  technical_config.yaml   # v2テクニカルのファミリー重み・レジーム閾値
  decision_config.yaml     # ファンダ/テクニカル合成重み・action閾値・リスクオーバーレイ設定
data/stock.db                 # SQLite本体(リポジトリにコミットして永続化)
data/backtest.db               # バックテスト専用(signal_history)。.gitignore対象、build_signal_history.pyで再生成
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
   - (長期業績推移を使う場合)名前: `EDINET_API_KEY` / 値: https://api.edinet-fsa.go.jp/ で発行したAPIキー
4. 上記2つが終われば、`.github/workflows/`配下のワークフローが以下のスケジュールで自動実行される:
   - `fetch_tickers.yml`: 毎月1日、銘柄一覧更新
   - `fetch_price_daily.yml`: 平日引け後、価格の日次バッチ取得
   - `fetch_fundamentals.yml`: 平日、ファンダメンタルズのローリング更新(時間予算30分)
   - `daily_decision.yml`: 平日、スコアリング→ルールベース判断→Discord通知
   - `evaluate_decisions.yml`: 平日、1ヶ月前の判断を評価し成績サマリーをログ出力
   - `edinet_index.yml`: 毎日、EDINET書類一覧のインデックス化(全銘柄分が揃うまで数週間かかる想定)
   - `edinet_history.yml`: 毎週日曜、インデックスを使って長期業績推移をローリング取得(`EDINET_API_KEY`必須)

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

# 長期業績推移(EDINET、任意)
EDINET_API_KEY=xxxx python scripts/fetch_edinet_history.py --tickers 7203.T 1301.T --lookback-filings 3
```

## 設計上の注意点

- 配当利回り(`dividend_yield`)はyfinanceの仕様上%表記(2.89 = 2.89%)で返る
- スコアは同一業種(東証33業種)内でのパーセンタイル順位を基準にした相対評価。業種内5銘柄未満は市場全体にフォールバック
- 指標が欠損している銘柄は信頼度(`confidence_*`)が下がり、総合スコアへの影響が自動的に割り引かれる(銀行業の流動比率欠損など)
- 売買判断(`decide_rule.py`)は`decision_source='rule'`で記録。将来LLM(Claude API)による判断を追加する場合は`decision_source='llm'`として同じテーブルに並行記録できる設計
- ルールの自動チューニングは行わない。`rule_performance.py`の出力を見て`config/scoring_config.yaml`や`decide_rule.py`の閾値を手動調整する運用を想定
- `fetch_edinet_history.py`は1件の有価証券報告書に含まれる直近5年分の連結決算サマリーを、約5年おきの過去の書類まで遡って`fundamentals_yearly`に蓄積する(実測: 極洋で13年分、トヨタで8年分)。連結値は「コンテキストID」に`_NonConsolidatedMember`等の接尾辞が付かない行として判定。IFRS採用の大企業は企業固有の拡張タグ(`...KeyFinancialData`)で売上高を開示することがあり、正規表現フォールバックで対応している。会計基準が大きく変わった年代(米国基準→IFRS移行前など)の書類は自動では拾えないことがある

## テクニカル判断エンジンv2について

v1(`decide_rule.py`)のGC/RSI/MACD/BB/出来高という二値シグナルは、相関の高いシグナルを頭数だけ数える設計だったため、`decide_composite.py`でゼロから再設計した。trend(MA配列・傾き・ADX)/mean_reversion(RSIダイバージェンス・%B)/volume(方向付き出来高・OBVダイバージェンス)/market_regime(TOPIX)/sector_regimeの5ファミリーを連続値(-1〜+1)で算出し、ファンダ6軸スコアと加重合成、ATRベースのストップロス・利確ラインも付与する。

`scripts/backtest/`で全銘柄1年分の実データを使ってこのv2エンジンを検証した結果、重要な発見があった:
- `trend_score`と`mean_reversion_score`の相関がρ=-0.79と高く、設計上完全な独立にはできていない
- `weight_optimizer.py`で1年分データにIC最大化で重みを最適化しても、`cross_validate.py`の時系列2分割検証で**全ファミリーのICが前半→後半で符号反転**した(市場全体の地合いが年途中で転換したため)
- `regime_adaptive_validate.py`でADXによる日次のtrend/mean_reversion重み切り替えを試しても、この不安定性は解消できなかった

このため、1年分データへの後付け最適化を信用して重みを固定するのではなく、`config/technical_config.yaml`/`config/decision_config.yaml`では意図的に保守的な重み(ファンダ0.65:テクニカル0.35、テクニカル内は均等に近い配分)を採用している。v2は`rule_version='v2.0'`として`decisions`テーブルにv1と並行記録され(Discord通知は現状v1のみ)、`rule_performance.py`でv1/v2の実績比較ができる。今後は後付けのバックテストではなく、実際の判断→1ヶ月後評価という前向きの実績蓄積で重みを調整していく方針。
