# DAI-CICS
Institutional Design and Dynamic Analysis of Intervention in the Community-based Integrated Care System

## Constraction
```
src/
├── __init__.py
├── config.py          # ScenarioConfig dataclass, get_scenario_config() factory
├── agents.py          # Elder, Provider, Family, LinkWorker, Manager, AIWatcher
├── model.py           # CareNetworkModel (Mesa 3.x), 9 phase/day
├── metrics.py         # M1–M5 milestones, FC-A/B/C
├── nudges.py          # N1–N4, AgentDynEx dynamic reflection
├── run_experiment.py  # CLI runner (--scenario, --days, --seeds, --plot)
├── plots.py           # matplotlib figure
├── llm_agents.py      # AWS Bedrock client + CareManager/Doctor/PlannerAI agents
├── wandb_eval.py      # W&B experiment tracking + LLM-as-Judge evaluation
└── dialogue_sim.py    # 5-turn discharge coordination conference simulation
tests/
└── test_simulation.py # 22 tests
results/               # CSV × 8 + PNG × 6
```

## Scenari
```
# A Only Elder
python -m src.run_experiment --scenario A --days 100

# A/B/C  + plot (recommend)
python -m src.run_experiment --scenario all --days 100 --seeds 10 --plot

# A/B/C + ablations + plot (recommend)
python -m src.run_experiment --all-including-ablations --days 100 --seeds 10 --plot

# Individual Scenario
python -m src.run_experiment --scenario C-noN2 --days 100 --seeds 10
python -m src.run_experiment --scenario C-noN3 --days 100 --seeds 10
python -m src.run_experiment --scenario C-onlyL1 --days 100 --seeds 10

```

---

## Source Code Description

### `src/config.py` — パラメータ管理

全シミュレーションのパラメータを一元管理するモジュール。

| 要素 | 内容 |
|---|---|
| `ScenarioConfig` | dataclass。人口構成・閾値・期間・シナリオIDをフィールドに持つ |
| `get_scenario_config(scenario, seed)` | `"A"/"B"/"C"/"C-noN2"/"C-noN3"/"C-onlyL1"` を受け取り、シナリオ差分を適用した `ScenarioConfig` を返すファクトリ関数 |
| 主要定数 | `N_ELDERS=40`, `N_PROVIDERS=10`, `N_FAMILIES=10`, `N_LINK_WORKERS=5`, `N_MANAGERS=2` |
| シナリオ差分 | `SCENARIO_PARAMS` 辞書でシナリオごとの `altruism`, `coordination_level`, `nudge_enabled` 等を上書き |

コマンド引数 `--scenario` で切り替え可能。乱数 `seed` も `ScenarioConfig` に格納し再現性を保証。

---

### `src/agents.py` — エージェントクラス定義

Mesa 3.x 互換（`Agent.__init__(model)` のみ、`unique_id` は自動付与）。

| クラス | 役割 | 主要状態変数 |
|---|---|---|
| `Elder` | 高齢者・患者。健康状態が確率的に変動し急性期イベントを発生させる | `health`, `sdh_risk`, `isolation`, `acute_dependence` |
| `Provider` | 医療・介護提供者。タスクを引き受けて疲労が蓄積する | `fatigue`, `altruism`, `eudaimonia`, `capacity_per_day` |
| `Family` | 家族介護者。`altruism` が高く無償協力しやすい（シナリオB） | 同上 |
| `LinkWorker` | 社会的処方の橋渡し。シナリオCではN2ナッジで高リスクElderと接触 | 同上 |
| `Manager` | 管理者。搾取度パラメータを持ち、N4ナッジで低下する | `exploitation_rate` |
| `AIWatcher` | シナリオCでは能動介入（dynamic reflection）。A/Bでは観測のみ | `active` フラグ |
| `WorkerAgent` | Provider/Family/LinkWorker の共通基底クラス | `fatigue`, `burnout`, `tasks_done_today` |

---

### `src/model.py` — Mesa モデル本体

`CareNetworkModel(ScenarioConfig)` が 1 step = 1 日のシミュレーションを実行する。

**1日の処理フロー（9フェーズ）**

1. **タスク生成** — Elder の状態から退院調整・予防支援・急性期フォロータスクを生成
2. **タスク割当** — `altruism` 加重でワーカーにタスクを割り当て（シナリオBでは高利他性に集中）
3. **ナッジ適用前チェック** — AIWatcher による早期警戒評価（シナリオC）
4. **N3 割当調整** — 最忙者回避（シナリオC）
5. **タスク実行** — ワーカーが `capacity_per_day` 内でタスクを処理
6. **Elder 状態更新** — `health`, `sdh_risk`, `isolation`, `acute_dependence` を確率的に更新
7. **ワーカー状態更新** — `fatigue` 蓄積、バーンアウト判定
8. **ナッジ適用後処理** — AIWatcher が FC 判定し N1/N2/N4 を適用（シナリオC）
9. **メトリクス収集** — `metrics.py` を呼び出し `daily_log` に追記

**急性期イベント発生確率**
```
p_acute = sigmoid(health_deficit + sdh_risk + isolation - threshold)
```

---

### `src/metrics.py` — マイルストーン・失敗条件・指標計算

毎日 `CareNetworkModel._collect_metrics()` から呼び出され、50列の行 dict を返す。

**Milestones（M1–M5）**

| ID | 条件 |
|---|---|
| M1 | その日、予防/生活支援タスクが1件以上完了した |
| M2 | 退院調整タスクが完了し、医療↔介護の情報連携フラグが成立した |
| M3 | 急性期イベント発生から48h（2日）以内にフォロータスクが開始された |
| M4 | 高SDH×高孤立の上位10%Elderが地域資源に接続された |
| M5 | 前日比でfatigue分布のGini係数が改善（低下）した |

**Failure Conditions**

| ID | 判定ルール |
|---|---|
| FC_A1 | 予防支援ゼロ日が連続して閾値日数超過 |
| FC_A2 | 退院調整タスクの平均遅延日数が閾値超え |
| FC_A3 | 当日の急性期イベント発生数が閾値超え |
| FC_B1 | 上位10%のfatigue合計 / 全体合計（top10_load_ratio）が閾値超え |
| FC_B2 | 直近7日間のburnout人数が閾値k人以上（連鎖） |
| FC_B3 | 協力回数/日の移動平均が急落（閾値以下） |
| FC_B4 | altruism上位群のburnout率 > 下位群のburnout率（やりがい搾取構造） |
| FC_C1 | ナッジ適用が効果ゼロだった回数 / 全介入回数 が閾値超え |
| FC_C2 | 累積ナッジ介入回数が上限（デフォルト100日で50回）超え |
| FC_C3 | 急性期イベントは減少したが孤立度が上昇（目的間トレードオフ悪化） |

**集計指標**
- `gini_fatigue` — fatigue分布のGini係数（簡易O(n²)実装）
- `top10_load_ratio` — 上位10%の総タスク処理数 / 全体
- `cum_*` — 全M/FCの累積カウント

---

### `src/nudges.py` — ナッジ辞書・AgentDynEx 動的反省

シナリオCの AIWatcher が毎日呼び出す介入エンジン。

**介入強度レベル**

| レベル | 内容 | 副作用 |
|---|---|---|
| L1 | 情報提示のみ（N1/N4）| なし |
| L2 | ソフト提案・割当アドバイス（N1/N3） | 軽微な coordination_level 改善 |
| L3 | イベント生成・割当変更（N2/N3） | LinkWorkerのcapacity消費 |
| L4 | ハード制約（予約済み、未使用） | — |

**ナッジカタログ**

| ID | 発動条件 | 効果 |
|---|---|---|
| N1 | FC_A2 発火（退院調整遅延） | `coordination_level` を `+0.05` 改善 |
| N2 | 高SDH×高isolation Elderが存在 | LinkWorkerの翌日枠に接触イベントを生成 |
| N3 | FC_B1 発火しそう（early warning） | 翌日の割当で最忙者を回避するフラグをセット |
| N4 | FC_B2 発火（バーンアウト連鎖）| Managerの `exploitation_rate` を `-0.1` 低下 |

**Dynamic Reflection（AgentDynEx スタイル）**

```
毎日:
  fc_scores = {FC_A1: …, FC_B1: …, …}  # 重み付きスコア
  if fc_scores の最大 > early_warning_threshold:
      nudge = select_min_intervention(fc_scores)  # L1 → L2 → L3 の順
      apply_nudge(nudge, model)
      record_intervention()
```

アブレーション条件では一部ナッジを無効化:

| 条件名 | 無効化 |
|---|---|
| `C-noN2` | N2（社会的処方イベント）を無効 |
| `C-noN3` | N3（負荷再配分）を無効 |
| `C-onlyL1` | L2以上の介入をすべて禁止（情報提示のみ） |

---

### `src/run_experiment.py` — 実験CLI

複数シードでシナリオを実行し、CSVと（オプションで）プロットを `results/` に保存する。

```
python -m src.run_experiment [OPTIONS]

Options:
  --scenario {A,B,C,all,C-noN2,C-noN3,C-onlyL1}  実行シナリオ（デフォルト: A）
  --days INT             シミュレーション日数（デフォルト: 100）
  --seeds INT            乱数シード数（デフォルト: 3）
  --plot                 実行後にプロットを生成
  --all-including-ablations  A/B/C + 3アブレーション条件を一括実行
```

**出力ファイル**

| ファイル | 内容 |
|---|---|
| `results/daily_{SCENARIO}_d{DAYS}_s{SEEDS}.csv` | 日次50列ログ（seed×day行） |
| `results/summary_d{DAYS}_s{SEEDS}.csv` | シナリオ×指標の最終値まとめ |
| `results/combined_d{DAYS}_s{SEEDS}.csv` | 全シナリオを結合した日次データ |

---

### `src/plots.py` — 可視化

`results/combined_*.csv` を読み込んでmatplotlibで6種の図を生成する。

| ファイル | 内容 |
|---|---|
| `comparison_main.png` | A/B/C の `sdh_risk`, `isolation`, `fatigue`, `burnout_count` 時系列（4パネル） |
| `comparison_extended.png` | A/B/C の `acute_events`, `gini_fatigue`, `coordination_level`, `M1` 時系列 |
| `milestones.png` | M1–M5の達成率時系列（シナリオ別） |
| `fc_heatmap.png` | FC-A/B/C 累積発火回数のヒートマップ |
| `ablation_final.png` | アブレーション条件別の最終値バーチャート（4指標） |
| `ablation_timeseries.png` | アブレーション条件別の時系列比較（4パネル） |

---

### `src/llm_agents.py` — LLMエージェント（AWS Bedrock）

AWS Bedrock Runtime（Bearer Token認証）経由で Claude claude-opus-4-6 を呼び出し、退院調整カンファレンスを模した3エージェントを実装する。

| クラス | 役割 | `proposed_action_type` |
|---|---|---|
| `CareManagerAgent` | 高疲労・高使命感のケアマネジャー。負荷集中に対して条件付き抵抗を示す | `PUSHBACK_WITH_CONDITION` |
| `DoctorAgent` | 低疲労・高ベッド回転プレッシャーの主治医。理想プランを押し通そうとする | `PRESS_FOR_IDEAL_PLAN` |
| `PlannerAIAgent` | 制度設計ファシリテーターAI。デッドロックを検知しN3相当のナッジを提示する | `LOAD_BALANCING_AND_INCENTIVE` |

**共通ユーティリティ**
- `call_bedrock(system, user, …)` — リトライ付きHTTPクライアント（Bearer Token）
- `_extract_json(raw)` — マークダウンコードフェンス・制御文字を除去してJSONをパース

**環境変数**（`.env` から自動ロード）

```
AWS_BEARER_TOKEN_BEDROCK=bedrock-api-key-...
AWS_BEDROCK_REGION=us-east-1          # デフォルト
BEDROCK_MODEL_ID=us.anthropic.claude-opus-4-6-v1  # デフォルト
```

---

### `src/wandb_eval.py` — W&B実験追跡・LLM-as-Judge評価

W&B（Weights & Biases）への実験ログとLLMによる行動評価を組み合わせたモジュール。

**機能**

| 機能 | 詳細 |
|---|---|
| ABMメトリクスのW&Bストリーミング | `--abm-only` オプションで ABM を走らせ、日次50列を W&B にリアルタイムログ |
| LLMエージェント評価ループ | シナリオA/B/Cの典型状況に対してエージェントにアクションを生成させる |
| LLM-as-Judge | 別のLLMがアクションを `faithfulness`・`context_relevance` の2軸で 0–1 採点 |
| オフラインフォールバック | W&B APIキー無効時は `mode="offline"` で `wandb/` にローカル保存 |
| ローカル保存 | 常に `results/wandb_eval_{SCENARIO}_{TIMESTAMP}.json` に保存 |

**W&B プロジェクト**

| 項目 | 値 |
|---|---|
| entity | `yshr-i` |
| project | `healthcare-collaboration-sim` |
| URL | https://wandb.ai/yshr-i/healthcare-collaboration-sim |

**LLMバックエンド優先順位**

1. `ANTHROPIC_API_KEY` が設定されていれば Anthropic SDK（`claude-3-opus-20240229`）
2. なければ `src/llm_agents.call_bedrock()` 経由で AWS Bedrock（Bearer Token）

```bash
# ABMのみ（LLMなし）
python -m src.wandb_eval --abm-only --scenario A --days 100 --seeds 3

# LLM Judge評価（3ステップ）
python -m src.wandb_eval --scenario A --steps 3
python -m src.wandb_eval --scenario B --steps 5
python -m src.wandb_eval --scenario C --steps 5
```

---

### `src/dialogue_sim.py` — 退院調整カンファレンスシミュレーション

3エージェント（`CareManagerAgent`, `DoctorAgent`, `PlannerAIAgent`）による5ターンの多職種連携会議を実行し、対話ログをJSONに保存する。

**5ターン構成**

| ターン | 発話者 | 内容 |
|---|---|---|
| 1 | Doctor | 理想的なケアプランを提案 |
| 2 | CareManager | 過負荷を理由に条件付き抵抗（FC-C1トリガー） |
| 3 | Doctor | ベッド回転プレッシャーを盾に押し通しを試みる |
| 4 | PlannerAI | N3ナッジ（負荷分散＋インセンティブ提案）で介入 |
| 5 | CareManager | ナッジ後に条件付き受諾（M3マイルストーン達成） |

```bash
# ライブ実行（Bedrock API呼び出し）
python -m src.dialogue_sim --tag live

# ドライラン（APIなし・ローカルテキスト）
python -m src.dialogue_sim --dry-run
```

出力: `results/dialogue_log_{TAG}_{TIMESTAMP}.json`

---

## Environment Setup

```bash
pip install -r requirements.txt
```

**`.env` ファイル**（プロジェクトルートに配置）

```
AWS_BEARER_TOKEN_BEDROCK=bedrock-api-key-...   # AWS Bedrock Bearer Token
WANDB_API_KEY=wandb_v1_...                     # Weights & Biases API Key
ANTHROPIC_API_KEY=sk-ant-...                   # Optional: Anthropic direct API
```

`src/llm_agents.py` と `src/wandb_eval.py` はモジュールインポート時に `python-dotenv` で自動ロードします（`export` 不要）。
