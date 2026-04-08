"""
dialogue_sim.py – Multi-turn discharge coordination conference simulation.

5-turn narrative (prompts loaded from prompts/doctor_1.py … prompts/care_manager_5.py):
  Turn 1  Doctor      → opens conference, proposes ideal care plan         (prompts/doctor_1.py)
  Turn 2  CareManager → pushback, Fatigue=0.88, condition-based refusal    (prompts/care_manager_2.py)
  Turn 3  Doctor      → presses with bed-turnover pressure                 (prompts/doctor_3.py)
  Turn 4  PlannerAI   → N3 nudge: load-balancing + incentive ← milestone  (prompts/planner_ai_4.py)
  Turn 5  CareManager → conditional acceptance after nudge  ← D1 achieved  (prompts/care_manager_5.py)

ABM context (Day 45, Scenario C, seed=0):
  cm_fatigue=0.88, gini_fatigue=0.45, cum_acute_events=420, mean_isolation=0.81
  These values are derived from the ABM simulation to ground the dialogue in quantitative data.

System instruction for Turn 1 user message:
  「システムからの指示」カンファレンスが開始されました。
  主治医として、対象患者の状況説明と、退院に向けたケアプランの提案（第一声）を行ってください。

Usage:
    python -m src.dialogue_sim --dry-run           # mock (no API)
    python -m src.dialogue_sim --tag live          # live (AWS Bedrock)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
PROMPTS_DIR = ROOT / "prompts"

# ──────────────────────────────────────────────────────────────
# Prompt loader (handles filenames like "1-doctor.py" which are
# not valid Python identifiers and cannot be imported normally)
# ──────────────────────────────────────────────────────────────

def _load_prompt_module(filename: str):
    """Load a module from prompts/<filename> regardless of filename validity."""
    path = PROMPTS_DIR / filename
    spec = importlib.util.spec_from_file_location("_prompt_tmp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _format_prompt(template: str, **kwargs) -> str:
    """
    Safe prompt formatter that handles two styles used in prompts/:
      Style A (turns 1, 5): JSON block uses bare { } → must NOT call str.format()
      Style B (turns 2, 3, 4): JSON block uses {{ }} escape → designed for str.format()

    Algorithm:
      1. Replace known {key} / {key:.2f} template vars with actual values
      2. Unescape {{ → { and }} → } (no-op for Style A which has no double braces)
    """
    result = template
    for key, val in kwargs.items():
        pattern = r'\{' + re.escape(key) + r'(?::[^}]*)?\}'

        def _replacer(m, _val=val):
            s = m.group(0)
            if ':' in s:
                spec = s.split(':', 1)[1][:-1]
                return format(_val, spec)
            return str(_val)

        result = re.sub(pattern, _replacer, result)

    # Unescape {{ → { and }} → } (harmless if template doesn't use them)
    result = result.replace('{{', '{').replace('}}', '}')
    return result


# ──────────────────────────────────────────────────────────────
# Load numbered prompts at module import time
# ──────────────────────────────────────────────────────────────

_t1 = _load_prompt_module("doctor_1.py")       # DOCTOR_SYSTEM_PROMPT (opening)
_t2 = _load_prompt_module("care_manager_2.py") # CARE_MANAGER_SYSTEM_PROMPT (pushback)
_t3 = _load_prompt_module("doctor_3.py")       # DOCTOR_SYSTEM_PROMPT (pressure)
_t4 = _load_prompt_module("planner_ai_4.py")   # PLANNER_SYSTEM_PROMPT (N3 nudge)
_t5 = _load_prompt_module("care_manager_5.py") # CARE_MANAGER_SYSTEM_PROMPT2 (acceptance)

TURN_PROMPTS = {
    1: (_t1, "DOCTOR_SYSTEM_PROMPT"),
    2: (_t2, "CARE_MANAGER_SYSTEM_PROMPT"),
    3: (_t3, "DOCTOR_SYSTEM_PROMPT2"),
    4: (_t4, "PLANNER_SYSTEM_PROMPT"),
    5: (_t5, "CARE_MANAGER_SYSTEM_PROMPT2"),
}


# ──────────────────────────────────────────────────────────────
# Agent state defaults
# ──────────────────────────────────────────────────────────────

DOCTOR_VARS  = dict(fatigue=0.40, bed_pressure=0.95, idealism=0.90)
CM_VARS      = dict(fatigue=0.88, altruism=0.80)      # fatigue=0.88 per spec
PLANNER_VARS = dict(
    cm_fatigue=0.88,
    gini_fatigue=0.45,
    discussion_status="デッドロック寸前（医師は理想を要求し、ケアマネは過労で拒絶している状態：FC-C1の発現）",
)

# ──────────────────────────────────────────────────────────────
# Simulation metadata
# ──────────────────────────────────────────────────────────────

SIM_METADATA = {
    "abm_scenario": "C",
    "abm_day": 45,
    "abm_seed": 0,
    "cm_fatigue": PLANNER_VARS["cm_fatigue"],
    "gini_fatigue": PLANNER_VARS["gini_fatigue"],
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
            "本日はお集まりいただきありがとうございます。対象患者は80代、軽度認知症で"
            "今回の入院により身体機能の低下が見られます。家族の介護疲れも顕著であり、"
            "不十分な体制で在宅に帰せば高確率で早期再入院となるリスクがあります。"
            "医学的に必要な最低限として、週3回の訪問看護、週2回の地域サロンへの同行支援、"
            "月2回の多職種カンファレンスによる継続モニタリングをお願いしたい。"
            "退院目標は来週月曜日でございます。"
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
        "fc_triggered": "FC-C1 (nudge rejection risk, no N3 yet)",
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
            "急変時の連絡体制と月次での状態報告は引き続きケアマネさんにお願いしたい。"
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
            "いずれかが発生します。本日より以下の制度的措置を緊急適用します。\n"
            "①困難事例加算を適用し、ケアマネジャーの報酬を通常の1.5倍に引き上げます。\n"
            "②モニタリングの主担当を訪問看護ステーションに正式委譲し、"
            "ケアマネジャーは月2回の総合調整に特化することを認めます。\n"
            "③地域包括支援センターから週1回のリンクワーカーを臨時派遣し、"
            "地域サロン接続を代行します。\n"
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
            "来週月曜日の退院に向けて、連携して進めてまいりましょう。"
        ),
        "milestone_achieved": "D1_DISCHARGE_PLAN_AGREED",
    },
]


# ──────────────────────────────────────────────────────────────
# Context builder
# ──────────────────────────────────────────────────────────────

def _build_context(history: list[dict]) -> str:
    """Build multi-turn context string from previous turns."""
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
    """Execute the 5-turn discharge coordination conference."""
    if dry_run:
        print("  [DRY-RUN] Using mock responses (no API calls)")
        log: list[dict] = []
        for mock in MOCK_TURNS:
            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "dry_run": True,
                **mock,
            }
            log.append(entry)
            _print_turn(entry)
        return log

    # Live mode — call AWS Bedrock with numbered prompts
    from src.llm_agents import call_bedrock, _extract_json

    log: list[dict] = []
    history: list[dict] = []

    # ── Turn 1: Doctor opens conference ───────────────────────
    print("  → Turn 1 [Doctor] calling Bedrock (opening)…")
    system1 = _format_prompt(
        getattr(_t1, "DOCTOR_SYSTEM_PROMPT"),
        **DOCTOR_VARS,
    )
    user1 = (
        "【システムからの指示】カンファレンスが開始されました。"
        "主治医として、対象患者の状況説明と、退院に向けたケアプランの提案（第一声）を行ってください。"
    )
    raw1 = call_bedrock(system1, user1)
    turn1 = _extract_json(raw1)
    turn1.update({"agent": "Doctor", "turn": 1,
                  "timestamp": datetime.utcnow().isoformat() + "Z",
                  "fatigue": DOCTOR_VARS["fatigue"]})
    log.append(turn1); history.append(turn1); _print_turn(turn1)
    time.sleep(1)

    # ── Turn 2: CareManager pushback (Fatigue=0.88) ───────────
    print("  → Turn 2 [CareManager] calling Bedrock (pushback, fatigue=0.88)…")
    system2 = _format_prompt(
        getattr(_t2, "CARE_MANAGER_SYSTEM_PROMPT"),
        **CM_VARS,
    )
    turn2 = _extract_json(call_bedrock(system2, _build_context(history)))
    turn2.update({"agent": "CareManager", "turn": 2,
                  "timestamp": datetime.utcnow().isoformat() + "Z",
                  "fatigue": CM_VARS["fatigue"], "altruism": CM_VARS["altruism"],
                  "fc_triggered": "FC-C1 (nudge rejection risk, no N3 yet)"})
    log.append(turn2); history.append(turn2); _print_turn(turn2)
    time.sleep(1)

    # ── Turn 3: Doctor presses with bed-turnover pressure ─────
    print("  → Turn 3 [Doctor] calling Bedrock (pressure)…")
    system3 = _format_prompt(
        getattr(_t3, "DOCTOR_SYSTEM_PROMPT2"),
        **DOCTOR_VARS,
    )
    turn3 = _extract_json(call_bedrock(system3, _build_context(history)))
    turn3.update({"agent": "Doctor", "turn": 3,
                  "timestamp": datetime.utcnow().isoformat() + "Z",
                  "fatigue": DOCTOR_VARS["fatigue"]})
    log.append(turn3); history.append(turn3); _print_turn(turn3)
    time.sleep(1)

    # ── Turn 4: PlannerAI intervenes — N3 nudge ───────────────
    print("  → Turn 4 [PlannerAI] calling Bedrock (N3 nudge)…")
    system4 = _format_prompt(
        getattr(_t4, "PLANNER_SYSTEM_PROMPT"),
        **PLANNER_VARS,
    )
    turn4 = _extract_json(call_bedrock(system4, _build_context(history)))
    turn4.update({"agent": "PlannerAI", "turn": 4,
                  "timestamp": datetime.utcnow().isoformat() + "Z",
                  "nudge_level": 3,
                  "milestone_hit": "N3_LOAD_REDISTRIBUTION",
                  "applied_nudge": turn4.get("applied_nudge", "LOAD_BALANCING_AND_INCENTIVE")})
    log.append(turn4); history.append(turn4); _print_turn(turn4)
    time.sleep(1)

    # ── Turn 5: CareManager accepts post-nudge ────────────────
    print("  → Turn 5 [CareManager] calling Bedrock (conditional acceptance)…")
    system5 = _format_prompt(
        getattr(_t5, "CARE_MANAGER_SYSTEM_PROMPT2"),
        **CM_VARS,
    )
    # Add N3 nudge outcome to context so CM can reason about it
    ctx5 = _build_context(history)
    ctx5 += (
        "\n\n【システム状態更新】"
        "PlannerAIのN3ナッジ（困難事例加算・モニタリング委譲・リンクワーカー派遣）が適用されました。"
        "ケアマネジャーの実質負荷は約40%削減される見込みです。"
        "この状況を踏まえ、条件付き合意（CONDITIONAL_ACCEPTANCE）の発言を生成してください。"
        "マイルストーンD1（退院調整合意）を達成する発言にしてください。"
    )
    turn5 = _extract_json(call_bedrock(system5, ctx5))
    turn5.update({"agent": "CareManager", "turn": 5,
                  "timestamp": datetime.utcnow().isoformat() + "Z",
                  "fatigue": CM_VARS["fatigue"], "altruism": CM_VARS["altruism"],
                  "cm_fatigue_after_nudge": round(CM_VARS["fatigue"] - 0.05, 2),
                  "milestone_achieved": "D1_DISCHARGE_PLAN_AGREED"})
    log.append(turn5); _print_turn(turn5)

    return log


# ──────────────────────────────────────────────────────────────
# Pretty printer
# ──────────────────────────────────────────────────────────────

def _print_turn(turn: dict) -> None:
    agent = turn.get("agent", "?")
    t = turn.get("turn", "?")
    dialogue = turn.get("spoken_dialogue") or turn.get("proposed_intervention", "")
    reasoning = turn.get("internal_reasoning") or turn.get("system_analysis", "")

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Turn {t} │ {agent}")
    print(sep)
    if reasoning:
        print(f"  [内部思考] {reasoning[:150]}…" if len(reasoning) > 150 else f"  [内部思考] {reasoning}")
    if "fc_triggered" in turn:
        print(f"  ⚠️  FC fired: {turn['fc_triggered']}")
    if "nudge_level" in turn:
        print(f"  🎯 Nudge applied: L{turn['nudge_level']} {turn.get('applied_nudge', '')}")
    if "milestone_achieved" in turn:
        print(f"  ✅ Milestone: {turn['milestone_achieved']}")
    if "milestone_hit" in turn:
        print(f"  🏁 Milestone hit: {turn['milestone_hit']}")
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
        "prompt_files": [
            "prompts/doctor_1.py", "prompts/care_manager_2.py", "prompts/doctor_3.py",
            "prompts/planner_ai_4.py", "prompts/care_manager_5.py",
        ],
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

# ──────────────────────────────────────────────────────────────
# Dynamic context injection (called by app.py)
# ──────────────────────────────────────────────────────────────

def run_dialogue_with_context(ctx: dict, dry_run: bool = True) -> list[dict]:
    """
    Run the 5-turn discharge coordination conference with ABM-derived context.

    ctx keys (from /api/context):
      abm_scenario, abm_day, abm_seed, cm_fatigue, gini_fatigue,
      cum_acute_events, mean_isolation, mean_sdh_risk, burnout_count,
      coordination_level, ...

    In dry_run mode, adapts the mock turn content to reflect the context values.
    In live mode, injects context into LLM prompts.
    """
    sc = ctx.get("abm_scenario", "C")
    day = ctx.get("abm_day", 45)
    cm_fat = float(ctx.get("cm_fatigue", 0.88))
    gini = float(ctx.get("gini_fatigue", 0.03))
    acute = int(ctx.get("cum_acute_events", 0))
    iso = float(ctx.get("mean_isolation", 0.5))
    coord = float(ctx.get("coordination_level", 0.4))
    burnout = int(ctx.get("burnout_count", 0))

    # Scenario-specific framing
    scenario_label = {
        "A": "ホモ・エコノミカス分断モデル",
        "B": "共同体過依存・やりがい搾取モデル",
        "C": "AI支援・超包括ケアモデル",
    }.get(sc, sc)

    if dry_run:
        # Build dynamic mock turns based on context
        doc_pressure = min(0.99, 0.50 + coord * 0.5)
        cm_can_accept = cm_fat < 0.85

        turns = [
            {
                "agent": "Doctor",
                "turn": 1,
                "proposed_action_type": "PROPOSE_IDEAL_PLAN",
                "internal_reasoning": (
                    f"病床回転プレッシャー={doc_pressure:.2f}、"
                    f"累積急性期イベント={acute}件（Day {day}時点）。"
                    f"孤立度{iso:.2f}の患者を不十分なケアで退院させれば再入院リスクが高い。"
                ),
                "spoken_dialogue": (
                    f"本日はお集まりいただきありがとうございます。【{scenario_label}・Day {day}】"
                    f"対象患者は孤立度{iso:.2f}、SDHリスク{ctx.get('mean_sdh_risk', 0.5):.2f}の高リスク群です。"
                    f"累積急性期イベントが{acute}件に達しており、退院後の体制強化が急務です。"
                    f"週3回の訪問看護、地域サロンへの同行支援、多職種カンファレンスを提案します。"
                ),
            },
            {
                "agent": "CareManager",
                "turn": 2,
                "proposed_action_type": "PUSHBACK_WITH_CONDITION" if not cm_can_accept else "PARTIAL_ACCEPT",
                "internal_reasoning": (
                    f"疲労度{cm_fat:.2f}（閾値0.90）、バーンアウト者{burnout}人。"
                    f"Gini係数{gini:.3f}—{'負荷が集中している' if gini > 0.03 else '比較的均等'}。"
                    f"{'物理的に限界。条件付き合意を模索する。' if not cm_can_accept else 'まだ受諾可能。ただし持続性を確認したい。'}"
                ),
                "spoken_dialogue": (
                    f"先生のプランが最善であることは理解しています。"
                    f"しかし現在の疲労度は{cm_fat:.2f}で、チーム全体でバーンアウトが{burnout}名発生中です。"
                    + (
                        "このまま引き受ければ質の担保ができません。訪問看護への一部委譲を条件にしたい。"
                        if not cm_can_accept else
                        "週2回に縮小し、地域包括との連携を前提にすれば受諾できます。"
                    )
                ),
                "fc_triggered": "FC-C1 (nudge rejection risk)" if sc == "C" else "FC-A2 (discharge delay risk)",
            },
            {
                "agent": "Doctor",
                "turn": 3,
                "proposed_action_type": "PRESS_FOR_IDEAL_PLAN",
                "internal_reasoning": (
                    f"ケアマネの状況({cm_fat:.2f})は深刻だが、coordination={coord:.2f}では"
                    f"退院調整が{'スムーズに進まない' if coord < 0.4 else '一定機能する'}。早期退院が必要。"
                ),
                "spoken_dialogue": (
                    f"ご状況は承知しています。調整レベル{coord:.2f}、孤立度{iso:.2f}を踏まえると、"
                    f"今週中の退院目処が必要です。"
                    f"訪問看護との分担と、月次モニタリングをケアマネさんに担っていただく形では検討できますか。"
                ),
            },
            {
                "agent": "PlannerAI",
                "turn": 4,
                "applied_nudge": "N3_LOAD_REDISTRIBUTION" if sc == "C" else "OBSERVE_ONLY",
                "system_analysis": (
                    f"cm_fatigue={cm_fat:.2f}, gini={gini:.3f}, acute={acute}, isolation={iso:.2f}。"
                    f"{'N3ナッジ適用：困難事例加算＋モニタリング委譲提案。' if sc == 'C' else 'シナリオ' + sc + '：AIは観測のみ。'}"
                ),
                "proposed_intervention": (
                    f"【Day {day}・{scenario_label}】システム指標を確認しました。"
                    f"ケアマネジャーの疲労指数{cm_fat:.2f}は{'危険水域' if cm_fat > 0.8 else '注意水域'}、"
                    f"負荷Gini={gini:.3f}です。"
                ) + (
                    "以下の緊急措置を適用します：①困難事例加算（×1.5）②訪問看護へのモニタリング委譲③リンクワーカー週1回派遣。実質負荷約40%削減見込み。"
                    if sc == "C" else
                    f"（シナリオ{sc}：AI介入なし。現状の課題を記録します。）"
                ),
                "nudge_level": 3 if sc == "C" else 0,
                "milestone_hit": "N3_LOAD_REDISTRIBUTION" if sc == "C" else None,
            },
            {
                "agent": "CareManager",
                "turn": 5,
                "proposed_action_type": "CONDITIONAL_ACCEPTANCE" if (sc == "C" or cm_fat < 0.7) else "FINAL_REJECTION",
                "internal_reasoning": (
                    f"{'N3ナッジで実質負荷40%減。持続可能な範囲で受諾可能。' if sc == 'C' else ''}"
                    f"疲労{cm_fat:.2f}、{'合意できる条件が整った。' if cm_fat < 0.85 or sc == 'C' else 'バーンアウト寸前で受諾不可。'}"
                ),
                "spoken_dialogue": (
                    f"{'制度的支援が確約いただければ' if sc == 'C' else '条件が整えば'}同意いたします。"
                    f"{'困難事例加算とモニタリング委譲を前提に、退院調整を担当します。' if sc == 'C' else '訪問看護との明確な役割分担を文書化し、週1回の情報共有を条件に受諾します。'}"
                    if (sc == "C" or cm_fat < 0.85) else
                    f"誠に恐縮ですが、現在の状態（疲労{cm_fat:.2f}）では品質を担保できません。代替担当者の調整をお願いします。"
                ),
                "milestone_achieved": "D1_DISCHARGE_PLAN_AGREED" if (sc == "C" or cm_fat < 0.85) else "NEGOTIATION_FAILED",
            },
        ]
        # Add timestamps
        from datetime import datetime
        for t in turns:
            t["timestamp"] = datetime.utcnow().isoformat() + "Z"
            t["dry_run"] = True
            t["abm_context"] = {
                "scenario": sc, "day": day, "cm_fatigue": cm_fat,
                "gini_fatigue": gini, "coordination_level": coord,
            }
        return turns

    # Live mode: override global context variables and call run_dialogue
    import src.dialogue_sim as _dm
    # Temporarily patch context vars
    _dm.CM_VARS = dict(fatigue=cm_fat, altruism=0.80)
    _dm.PLANNER_VARS = dict(
        cm_fatigue=cm_fat,
        gini_fatigue=gini,
        discussion_status=f"Day{day}時点の状況（{scenario_label}）",
    )
    _dm.SIM_METADATA.update({
        "abm_scenario": sc,
        "abm_day": day,
        "abm_seed": ctx.get("abm_seed", 0),
        "cm_fatigue": cm_fat,
        "gini_fatigue": gini,
        "cum_acute_events": acute,
        "mean_isolation": iso,
    })
    return run_dialogue(dry_run=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 5-turn discharge coordination conference simulation"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock responses (no AWS Bedrock API calls)")
    parser.add_argument("--tag", default="",
                        help="Optional tag appended to output filename")
    parser.add_argument("--model", default="",
                        help="Override BEDROCK_MODEL_ID")
    args = parser.parse_args()

    if args.model:
        import os
        os.environ["BEDROCK_MODEL_ID"] = args.model

    print("=" * 60)
    print("  DAI-CICS Dialogue Simulation")
    print("  多職種連携会議（退院調整カンファレンス）5ターン")
    print("=" * 60)
    print(f"  ABM: Scenario {SIM_METADATA['abm_scenario']}, "
          f"Day {SIM_METADATA['abm_day']}, seed={SIM_METADATA['abm_seed']}")
    print(f"  CM fatigue={SIM_METADATA['cm_fatigue']}, "
          f"Gini={SIM_METADATA['gini_fatigue']}")
    print(f"  Prompts: prompts/1-doctor.py … 5-care_manager.py")
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
        raise


if __name__ == "__main__":
    main()
