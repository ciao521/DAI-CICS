"""
dialogue_sim.py – Multi-turn discharge coordination conference simulation.

5-turn narrative:
  Turn 1  Doctor     → proposes ideal care plan
  Turn 2  CareManager→ pushback with conditions (FC-C1 trigger in C-noN3)
  Turn 3  Doctor     → presses with bed-turnover pressure
  Turn 4  PlannerAI  → N3 nudge: load-balancing + incentive  ← milestone N3
  Turn 5  CareManager→ conditional acceptance after nudge    ← D1 milestone achieved

Usage:
    # With real Bedrock API:
    export AWS_BEARER_TOKEN_BEDROCK=bedrock-api-key-...
    python -m src.dialogue_sim

    # Dry-run (mock LLM, no API calls):
    python -m src.dialogue_sim --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# ──────────────────────────────────────────────────────────────
# Simulation metadata (linked to ABM results)
# ──────────────────────────────────────────────────────────────

SIM_METADATA = {
    "abm_scenario": "C",
    "abm_day": 45,
    "abm_seed": 0,
    "cm_fatigue": 0.88,
    "gini_fatigue": 0.45,
    "cum_acute_events": 420,
    "mean_isolation": 0.81,
    "description": "急性期→退院調整カンファレンス（Day 45相当、N3ナッジ適用前後を対比）",
}

# ──────────────────────────────────────────────────────────────
# Mock responses for dry-run
# ──────────────────────────────────────────────────────────────

MOCK_TURNS: list[dict] = [
    {
        "agent": "Doctor",
        "turn": 1,
        "proposed_action_type": "PROPOSE_IDEAL_PLAN",
        "internal_reasoning": (
            "病床回転プレッシャーが0.95と極めて高い。"
            "患者の再入院を防ぐには週3回訪問と地域サロン接続が必要。"
            "ケアマネに負担が集中するが、医学的には最善のプランを提示するしかない。"
            "退院が遅延すれば病院経営に直結する問題となる。"
        ),
        "spoken_dialogue": (
            "患者さんの状態を鑑みると、社会的孤立の解消が再入院防止の鍵です。"
            "週3回の訪問看護に加え、リンクワーカーを通じた週2回の地域サロンへの同行支援と、"
            "継続的なモニタリングをお願いできますでしょうか。"
            "当院としても早期退院が求められており、ケアマネさんのご協力が不可欠です。"
        ),
    },
    {
        "agent": "CareManager",
        "turn": 2,
        "proposed_action_type": "PUSHBACK_WITH_CONDITION",
        "internal_reasoning": (
            "疲労度0.88は限界寸前。このプランを単独で引き受ければバーンアウトは確実。"
            "患者のためには最善のプランだと分かっているが（使命感0.80）、"
            "物理的に不可能な状態。拒絶ではなく条件付き合意で交渉すべき。"
            "訪問看護ステーションへのモニタリング委譲が条件になるか確認する。"
        ),
        "spoken_dialogue": (
            "先生のおっしゃるプランが患者さんにとって最善であることは十分理解しております。"
            "しかし、現在私の担当ケースは定員を大幅に超えており、"
            "疲労も限界に近い状態です。このまま週3回の訪問とモニタリング全てを"
            "私一人が担当すれば、質の担保ができないどころか離職リスクもございます。"
            "訪問看護ステーションがモニタリングの主担当を引き受けていただければ、"
            "私は週1回の総合調整役に集中できます。この条件でしたら引き受け可能です。"
        ),
        "fc_triggered": "FC-C1 (nudge rejection risk)",
    },
    {
        "agent": "Doctor",
        "turn": 3,
        "proposed_action_type": "PRESS_FOR_IDEAL_PLAN",
        "internal_reasoning": (
            "ケアマネの過労は理解できる。しかし病床は既にひっ迫しており、"
            "退院を先延ばしにする余裕はない。"
            "訪問看護への委譲案は一定の合理性があるが、"
            "モニタリング継続性の担保を確認する必要がある。押し込みながら妥協点を探る。"
        ),
        "spoken_dialogue": (
            "ケアマネさんのご状況は深刻であることは承知しています。"
            "ただ、当院のベッドは既に次の患者さんが待機しており、"
            "今週中に退院の目処を立てなければなりません。"
            "訪問看護ステーションへの一部委譲は検討できますが、"
            "患者さんの状態が急変した際の連絡体制と、"
            "月次での状態報告は引き続きケアマネさんにお願いしたい。"
            "この条件で何とかご検討いただけないでしょうか。"
        ),
    },
    {
        "agent": "PlannerAI",
        "turn": 4,
        "applied_nudge": "LOAD_BALANCING_AND_INCENTIVE",
        "system_analysis": (
            "Care_Manager_Fatigue=0.88は危険水域。医師のベッド圧力(0.95)とのデッドロックは"
            "FC-A2（退院遅延）またはFC-B2（バーンアウト連鎖）のどちらかを引き起こす。"
            "N3ナッジ（負荷再配分）とインセンティブ付与で両者の合意点を創出すべき段階。"
        ),
        "proposed_intervention": (
            "システム指標を確認しました。ケアマネジャーの疲労指数が0.88と危険水域に達しており、"
            "このまま交渉が膠着すると退院遅延（社会的入院）またはバーンアウト連鎖の"
            "いずれかが発生します。本日より以下の制度的措置を緊急適用します。\n\n"
            "①【加算措置】本ケースに「困難事例加算」を適用し、ケアマネジャーの"
            "報酬を通常の1.5倍に引き上げます。\n"
            "②【タスク委譲許可】モニタリングの主担当を訪問看護ステーションに正式委譲し、"
            "ケアマネジャーは月2回の総合調整に特化することを認めます。\n"
            "③【リンクワーカー派遣】地域包括支援センターから週1回のリンクワーカーを"
            "臨時派遣し、地域サロン接続を代行します。\n\n"
            "これらの措置により、ケアマネジャーの実質負荷は約40%削減される見込みです。"
        ),
        "nudge_level": 3,
        "milestone_hit": "N3_LOAD_REDISTRIBUTION",
    },
    {
        "agent": "CareManager",
        "turn": 5,
        "proposed_action_type": "CONDITIONAL_ACCEPTANCE",
        "internal_reasoning": (
            "プランナーAIのN3介入により実質負荷が40%削減される。"
            "困難事例加算により持続可能性も担保された。"
            "使命感(0.80)を持って患者のために最善を尽くせる条件が整った。"
            "合意することで患者の孤立解消とケアの質が担保される。これは受け入れられる。"
        ),
        "spoken_dialogue": (
            "プランナーからの制度的支援の提案、ありがとうございます。"
            "困難事例加算の適用と、訪問看護ステーションへの正式なモニタリング委譲、"
            "そしてリンクワーカーの派遣が確約いただけるのであれば、"
            "このケースの退院調整を担当することに同意いたします。"
            "私は月2回の総合調整と、急変時の連絡対応に集中することで、"
            "患者さんへの質の高いケアを維持できると考えます。"
            "先生、今週中に退院の目処を立てられるよう、"
            "連携して進めてまいりましょう。"
        ),
        "milestone_achieved": "D1_DISCHARGE_PLAN_AGREED",
    },
]


# ──────────────────────────────────────────────────────────────
# Dialogue turn builder
# ──────────────────────────────────────────────────────────────

def _build_context(history: list[dict]) -> str:
    """Build a multi-turn context string from previous turns."""
    if not history:
        return ""
    lines = ["【これまでの会議の経緯】"]
    for turn in history:
        agent = turn.get("agent", "?")
        dialogue = turn.get("spoken_dialogue") or turn.get("proposed_intervention", "")
        lines.append(f"[{agent}]: {dialogue}")
    lines.append("\n【上記の発言を踏まえて、あなたの番の発言を生成してください。】")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Main simulation runner
# ──────────────────────────────────────────────────────────────

def run_dialogue(dry_run: bool = False) -> list[dict]:
    """
    Execute the 5-turn discharge coordination conference.
    Returns list of turn dicts (full dialogue log).
    """
    if dry_run:
        print("  [DRY-RUN] Using mock responses (no API calls)")
        log: list[dict] = []
        for mock in MOCK_TURNS:
            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "turn": mock["turn"],
                "dry_run": True,
                **mock,
            }
            log.append(entry)
            _print_turn(entry)
        return log

    # Live mode: call AWS Bedrock
    from src.llm_agents import CareManagerAgent, DoctorAgent, PlannerAIAgent

    doctor = DoctorAgent()
    care_manager = CareManagerAgent()
    planner = PlannerAIAgent()
    planner.update_metrics(
        cm_fatigue=SIM_METADATA["cm_fatigue"],
        gini_fatigue=SIM_METADATA["gini_fatigue"],
        status="デッドロック寸前（医師は理想を要求し、ケアマネは過労で拒絶している状態）",
    )

    log: list[dict] = []
    history: list[dict] = []

    # Turn 1: Doctor initiates
    print("  → Turn 1 [Doctor] calling Bedrock…")
    initial_prompt = (
        "退院調整カンファレンスを開始します。"
        "患者（80代、軽度認知症、家族の介護疲れあり）の退院調整について、"
        "理想的なケアプランを提案してください。"
        "週3回の訪問看護、週2回の地域サロンへの同行支援、継続的モニタリングを含む"
        "プランについて、ケアマネジャーへの依頼として発言してください。"
    )
    turn1 = doctor.respond(initial_prompt)
    turn1["turn"] = 1
    turn1["timestamp"] = datetime.utcnow().isoformat() + "Z"
    log.append(turn1)
    history.append(turn1)
    _print_turn(turn1)
    time.sleep(1)

    # Turn 2: CareManager pushback
    print("  → Turn 2 [CareManager] calling Bedrock…")
    ctx2 = _build_context(history)
    turn2 = care_manager.respond(ctx2)
    turn2["turn"] = 2
    turn2["timestamp"] = datetime.utcnow().isoformat() + "Z"
    turn2["fc_triggered"] = "FC-C1 (nudge rejection risk, no N3 yet)"
    log.append(turn2)
    history.append(turn2)
    _print_turn(turn2)
    time.sleep(1)

    # Turn 3: Doctor presses
    print("  → Turn 3 [Doctor] calling Bedrock…")
    ctx3 = _build_context(history)
    turn3 = doctor.respond(ctx3)
    turn3["turn"] = 3
    turn3["timestamp"] = datetime.utcnow().isoformat() + "Z"
    log.append(turn3)
    history.append(turn3)
    _print_turn(turn3)
    time.sleep(1)

    # Turn 4: PlannerAI intervenes (N3 nudge)
    print("  → Turn 4 [PlannerAI] calling Bedrock (N3 nudge)…")
    ctx4 = _build_context(history)
    turn4 = planner.respond(ctx4)
    turn4["turn"] = 4
    turn4["timestamp"] = datetime.utcnow().isoformat() + "Z"
    turn4["nudge_level"] = 3
    turn4["milestone_hit"] = "N3_LOAD_REDISTRIBUTION"
    log.append(turn4)
    history.append(turn4)
    _print_turn(turn4)
    time.sleep(1)

    # Turn 5: CareManager accepts (post-nudge)
    print("  → Turn 5 [CareManager] calling Bedrock (post-nudge acceptance)…")
    ctx5 = _build_context(history)
    # Signal to care manager that conditions improved
    ctx5 += (
        "\n\n【システム状態更新】"
        "プランナーAIのN3ナッジ（負荷再配分＋インセンティブ）が適用されました。"
        "困難事例加算、モニタリング委譲、リンクワーカー派遣が確約されています。"
        "この状況を踏まえて、条件付き合意（CONDITIONAL_ACCEPTANCE）の発言を生成してください。"
        "マイルストーンD1（退院調整合意）を達成する発言にしてください。"
    )
    turn5 = care_manager.respond(ctx5)
    turn5["turn"] = 5
    turn5["timestamp"] = datetime.utcnow().isoformat() + "Z"
    turn5["milestone_achieved"] = "D1_DISCHARGE_PLAN_AGREED"
    # Update CM fatigue (slightly improved by N3)
    turn5["cm_fatigue_after_nudge"] = round(care_manager.state.fatigue - 0.05, 2)
    log.append(turn5)
    _print_turn(turn5)

    return log


def _print_turn(turn: dict) -> None:
    """Pretty-print a single dialogue turn."""
    agent = turn.get("agent", "?")
    t = turn.get("turn", "?")
    dialogue_key = "spoken_dialogue" if "spoken_dialogue" in turn else "proposed_intervention"
    dialogue = turn.get(dialogue_key, "")
    reasoning_key = "internal_reasoning" if "internal_reasoning" in turn else "system_analysis"
    reasoning = turn.get(reasoning_key, "")

    separator = "─" * 60
    print(f"\n{separator}")
    print(f"  Turn {t} │ {agent}")
    print(separator)
    if reasoning:
        print(f"  [内部思考] {reasoning[:150]}…" if len(reasoning) > 150 else f"  [内部思考] {reasoning}")
    if "fc_triggered" in turn:
        print(f"  ⚠️  FC fired: {turn['fc_triggered']}")
    if "nudge_level" in turn:
        print(f"  🎯 Nudge applied: L{turn['nudge_level']} {turn.get('applied_nudge', '')}")
    if "milestone_achieved" in turn:
        print(f"  ✅ Milestone: {turn['milestone_achieved']}")
    print(f"\n  {dialogue}\n")


# ──────────────────────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────────────────────

def save_dialogue_log(log: list[dict], tag: str = "") -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    suffix = f"_{tag}" if tag else ""
    path = RESULTS_DIR / f"dialogue_log{suffix}_{ts}.json"

    output = {
        "metadata": SIM_METADATA,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_turns": len(log),
        "model": "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "fc_triggered": [
            {"turn": t["turn"], "fc": t["fc_triggered"]}
            for t in log if "fc_triggered" in t
        ],
        "milestones_achieved": [
            {"turn": t["turn"], "milestone": t["milestone_achieved"]}
            for t in log if "milestone_achieved" in t
        ],
        "nudges_applied": [
            {"turn": t["turn"], "nudge": t.get("applied_nudge"), "level": t.get("nudge_level")}
            for t in log if "nudge_level" in t
        ],
        "turns": log,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved → {path.relative_to(ROOT)}")
    return path


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the multi-agent discharge coordination conference simulation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use mock LLM responses (no AWS Bedrock API calls)",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional tag appended to output filename",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Override BEDROCK_MODEL_ID (e.g. us.anthropic.claude-opus-4-5-20251101-v1:0)",
    )
    args = parser.parse_args()

    if args.model:
        import os
        os.environ["BEDROCK_MODEL_ID"] = args.model

    print("=" * 60)
    print("  DAI-CICS Dialogue Simulation")
    print("  多職種連携会議（退院調整カンファレンス）")
    print("=" * 60)
    print(f"  ABM link: Scenario {SIM_METADATA['abm_scenario']}, "
          f"Day {SIM_METADATA['abm_day']}, seed={SIM_METADATA['abm_seed']}")
    print(f"  CM fatigue={SIM_METADATA['cm_fatigue']}, "
          f"Gini={SIM_METADATA['gini_fatigue']}")
    print(f"  Mode: {'DRY-RUN (mock)' if args.dry_run else 'LIVE (AWS Bedrock)'}")
    print()

    try:
        log = run_dialogue(dry_run=args.dry_run)
        save_dialogue_log(log, tag=args.tag or ("dryrun" if args.dry_run else "live"))
        print("\n  Done. Dialogue simulation complete.")
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
