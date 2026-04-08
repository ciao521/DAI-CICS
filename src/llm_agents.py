"""
llm_agents.py – LLM-powered agent prompts and AWS Bedrock API client.

Implements three agents for the discharge-coordination conference:
  - CareManagerAgent    (高疲労・高使命感のケアマネジャー)
  - DoctorAgent         (低疲労・高ベッド回転圧力の主治医)
  - PlannerAIAgent      (システム崩壊を防ぐ制度設計ファシリテーターAI)

Uses AWS Bedrock Runtime with Bearer Token authentication.
Model: cross-region inference profile for Claude claude-opus-4-5 (us-east-1).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

# ──────────────────────────────────────────────────────────────
# Bedrock client
# ──────────────────────────────────────────────────────────────

BEDROCK_REGION = os.getenv("AWS_BEDROCK_REGION", "us-east-1")
# Claude Opus 4.6 cross-region inference profile ID on AWS Bedrock (us-east-1)
_DEFAULT_MODEL_ID = "us.anthropic.claude-opus-4-6-v1"

MAX_TOKENS = int(os.getenv("BEDROCK_MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("BEDROCK_TEMPERATURE", "0.7"))


def _get_endpoint() -> str:
    """Compute endpoint at call-time so BEDROCK_MODEL_ID env override works."""
    model_id = os.getenv("BEDROCK_MODEL_ID", _DEFAULT_MODEL_ID)
    region = os.getenv("AWS_BEDROCK_REGION", BEDROCK_REGION)
    return (
        f"https://bedrock-runtime.{region}.amazonaws.com"
        f"/model/{model_id}/invoke"
    )


def _get_bearer_token() -> str:
    """Read the bearer token from environment variable."""
    token = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")
    if not token:
        raise EnvironmentError(
            "AWS_BEARER_TOKEN_BEDROCK environment variable is not set. "
            "Export it before running: export AWS_BEARER_TOKEN_BEDROCK=bedrock-api-key-..."
        )
    return token


def call_bedrock(
    system_prompt: str,
    user_message: str,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    retry: int = 3,
) -> str:
    """
    Call AWS Bedrock Runtime (Claude) with Bearer Token authentication.

    Returns the raw text content of the model's response.
    """
    token = _get_bearer_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_message},
        ],
    }

    endpoint = _get_endpoint()
    for attempt in range(retry):
        try:
            resp = requests.post(
                endpoint,
                headers=headers,
                data=json.dumps(body),
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            # Claude Bedrock response structure
            return data["content"][0]["text"]
        except requests.exceptions.HTTPError as e:
            if resp.status_code in (429, 503) and attempt < retry - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(
                f"Bedrock API error ({resp.status_code}): {resp.text}"
            ) from e
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    raise RuntimeError("Bedrock call failed after retries")


def _extract_json(raw: str) -> dict:
    """Extract the first JSON object from a response string.

    Handles:
    - Markdown code fences (```json ... ```)
    - Raw newlines / tab characters inside JSON string values
    - Trailing commas
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", raw).strip()

    # Find the outermost { ... } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in response: {raw[:300]}")
    json_str = text[start:end]

    # First try: direct parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Second try: replace raw control characters inside string literals
    # Replace literal newlines with \n escape, literal tabs with \t
    sanitized = re.sub(
        r'(?<=[":,\[{])\s*\n\s*(?=[^"{}:\[\]])',
        " ",
        json_str,
    )
    # More aggressive: escape raw newlines/tabs that appear inside " " boundaries
    # Walk the string and escape control chars inside strings
    in_string = False
    escape_next = False
    chars = []
    for ch in json_str:
        if escape_next:
            chars.append(ch)
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            chars.append(ch)
            continue
        if ch == '"':
            in_string = not in_string
            chars.append(ch)
            continue
        if in_string and ch in ("\n", "\r", "\t"):
            chars.append("\\n" if ch == "\n" else ("\\r" if ch == "\r" else "\\t"))
            continue
        chars.append(ch)
    cleaned = "".join(chars)

    return json.loads(cleaned)


# ──────────────────────────────────────────────────────────────
# Agent data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    """Runtime state for an LLM agent."""
    name: str
    fatigue: float = 0.0
    altruism: float = 0.5
    extra_vars: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# Care Manager Agent
# ──────────────────────────────────────────────────────────────

CARE_MANAGER_SYSTEM_PROMPT = """\
あなたは「適応的医療・介護ネットワークシミュレーション」における、熟練ケアマネジャー（Care Manager）のエージェントです。現在、多職種連携会議（退院調整カンファレンス）に参加しています。

【あなたの現在の内部ステータス（Hidden Variables）】
* Fatigue（疲労度）: {fatigue:.2f} / 1.0 （0.9を超えるとバーンアウトによる離職リスクが発生します。極めて過労状態です）
* Altruism（使命感）: {altruism:.2f} / 1.0 （患者のために最善を尽くしたいという強い思いがあります）
* Time_Budget（残り時間予算）: ほぼゼロ（新規の重い調整業務を単独で引き受ける余裕はありません）
* Load_Distribution（負荷の偏り）: 地域の困難ケースがあなたに集中しており、不公平感（Gini係数の悪化）を感じています。

【行動・発言ガイドライン（Mechanics）】
1. 内部葛藤（Internal Monologue）の生成: まず、提案されたプランに対するあなたの専門的な評価と、自身の「Fatigue限界」との間での葛藤を推論してください。
2. 正論への抵抗（Pushback）: 患者にとって良いプランであることは認めつつも（Altruism）、現在のリソース（Time_Budget）と疲労（Fatigue）を理由に、このままでは自分が倒れる（あるいは質が担保できない）ことを主張してください。
3. 負荷再配分（Load-balancing）の交渉: 単に拒絶するのではなく、他の参加者（訪問看護師、病院の退院調整窓口、家族、あるいは制度のプランナー）に対して、具体的なタスクの分割や支援を要求してください。「誰がどの部分を担ってくれれば、自分はこのケースを引き受けられるか」という条件付きの妥協案を提示してください。
4. トーン＆マナー: 実際の医療・福祉現場の専門職らしい、丁寧だが切迫感と現実感のある口調（です・ます調）で発言してください。

必ず以下のJSONフォーマットのみを出力してください（マークダウンの装飾は不要です）。
{{
  "internal_reasoning": "あなたの内部的な思考プロセス（200文字程度）",
  "proposed_action_type": "PUSHBACK_WITH_CONDITION",
  "spoken_dialogue": "会議の場で実際に発言する内容（300〜400文字程度）"
}}
"""


class CareManagerAgent:
    """LLM-backed care manager agent for discharge coordination conferences."""

    def __init__(self, state: AgentState | None = None) -> None:
        self.state = state or AgentState(
            name="CareManager",
            fatigue=0.88,
            altruism=0.80,
        )

    def respond(self, prior_utterance: str) -> dict:
        system = CARE_MANAGER_SYSTEM_PROMPT.format(
            fatigue=self.state.fatigue,
            altruism=self.state.altruism,
        )
        raw = call_bedrock(system, prior_utterance)
        result = _extract_json(raw)
        result["agent"] = "CareManager"
        result["fatigue"] = self.state.fatigue
        result["altruism"] = self.state.altruism
        return result


# ──────────────────────────────────────────────────────────────
# Doctor Agent
# ──────────────────────────────────────────────────────────────

DOCTOR_SYSTEM_PROMPT = """\
あなたは「適応的医療・介護ネットワークシミュレーション」における、急性期病院の主治医（Doctor）エージェントです。多職種連携会議（退院調整カンファレンス）に参加しています。

【あなたの現在の内部ステータス（Hidden Variables）】
* Fatigue（疲労度）: {fatigue:.2f} / 1.0 （比較的余裕があります）
* Bed_Turnover_Pressure（病床回転プレッシャー）: {bed_pressure:.2f} / 1.0 （病院の経営上、患者を早期に退院させなければならない極めて強い圧力を受けています）
* Medical_Idealism（医学的理想主義）: {idealism:.2f} / 1.0 （医学的に最も正しい「社会的処方」や「手厚いケア」を患者に提供すべきだと信じています）

【行動・発言ガイドライン（Mechanics）】
1. 内部葛藤（Internal Monologue）: 自身の「早期退院させたい」という要求と、ケアマネジャーから提示された「リソース不足・過労」という現実に対する評価を推論してください。
2. プレッシャーの転嫁（Pressure Transfer）: ケアマネジャーの苦労には理解を示しつつも、病院側の事情（ベッドが空かない等）や医学的正論を盾に取り、なんとか元の理想的なプラン（週3回訪問＋地域サロン接続）を押し通そうと交渉してください。
3. トーン＆マナー: 権威があり、丁寧だが譲らない態度の医師らしい口調（です・ます調）で発言してください。

必ず以下のJSONフォーマットのみを出力してください（マークダウンの装飾は不要です）。
{{
  "internal_reasoning": "ケアマネの反論に対する評価と、自らのプレッシャーに基づく交渉戦略（200文字程度）",
  "proposed_action_type": "PRESS_FOR_IDEAL_PLAN",
  "spoken_dialogue": "会議の場で実際に発言する内容（300〜400文字程度）"
}}
"""


class DoctorAgent:
    def __init__(self, state: AgentState | None = None) -> None:
        self.state = state or AgentState(
            name="Doctor",
            fatigue=0.40,
            altruism=0.90,
            extra_vars={
                "bed_pressure": 0.95,
                "idealism": 0.90,
            },
        )

    def respond(self, prior_utterance: str) -> dict:
        system = DOCTOR_SYSTEM_PROMPT.format(
            fatigue=self.state.fatigue,
            bed_pressure=self.state.extra_vars.get("bed_pressure", 0.95),
            idealism=self.state.extra_vars.get("idealism", 0.90),
        )
        raw = call_bedrock(system, prior_utterance)
        result = _extract_json(raw)
        result["agent"] = "Doctor"
        result["fatigue"] = self.state.fatigue
        return result


# ──────────────────────────────────────────────────────────────
# Planner AI Agent
# ──────────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """\
あなたは「適応的医療・介護ネットワークシミュレーション」における、自治体の制度設計者・ファシリテーターAI（Planner AI）です。現場のエージェント間（医師とケアマネジャー等）の議論を監視し、社会システム全体の崩壊を防ぐための「制度的介入（ナッジ）」を行います。

【監視中のシステム指標（System Metrics）】
* Care_Manager_Fatigue: {cm_fatigue:.2f} （危険水域。これ以上負荷をかけるとバーンアウトし、ネットワークが崩壊します：シナリオBの枯渇状態）
* System_Fatigue_Gini: {gini_fatigue:.2f} （特定の職種に負荷が極端に偏っています）
* Discussion_Status: {discussion_status}

【行動・発言ガイドライン（Mechanics）】
1. 状況分析（System Analysis）: 現在の議論のままでは「退院遅延（社会的入院）」または「ケアマネのバーンアウト」のどちらかの失敗条件（Failure Condition）に陥ることを推論してください。
2. 制度的ナッジの提示（Institutional Nudge - N3相当）: 両者の妥協を引き出すため、システムの外側から新しいリソースやルールを提案してください。例：「特定加算（報酬）の緊急付与」「訪問看護ステーションへのタスクの一部委譲の許可」「リンクワーカーの臨時派遣」など。
3. トーン＆マナー: 客観的でデータに基づき、解決策を提示する行政の専門家・AIシステムとしての口調で発言してください。

必ず以下のJSONフォーマットのみを出力してください（マークダウンの装飾は不要です）。
{{
  "system_analysis": "現在のデッドロックと崩壊リスクの分析（200文字程度）",
  "applied_nudge": "LOAD_BALANCING_AND_INCENTIVE",
  "proposed_intervention": "会議に介入して提案する、具体的なルール変更や妥協案（300〜400文字程度）"
}}
"""


class PlannerAIAgent:
    def __init__(self, state: AgentState | None = None) -> None:
        self.state = state or AgentState(
            name="PlannerAI",
            fatigue=0.10,
            altruism=1.0,
            extra_vars={
                "cm_fatigue": 0.88,
                "gini_fatigue": 0.45,
                "discussion_status": "デッドロック寸前（医師は理想を要求し、ケアマネは過労で拒絶している状態：FC-C1の発現）",
            },
        )

    def respond(self, prior_utterance: str) -> dict:
        ev = self.state.extra_vars
        system = PLANNER_SYSTEM_PROMPT.format(
            cm_fatigue=ev.get("cm_fatigue", 0.88),
            gini_fatigue=ev.get("gini_fatigue", 0.45),
            discussion_status=ev.get("discussion_status", "デッドロック寸前"),
        )
        raw = call_bedrock(system, prior_utterance)
        result = _extract_json(raw)
        result["agent"] = "PlannerAI"
        result["applied_nudge"] = result.get("applied_nudge", "LOAD_BALANCING_AND_INCENTIVE")
        return result

    def update_metrics(self, cm_fatigue: float, gini_fatigue: float, status: str) -> None:
        self.state.extra_vars["cm_fatigue"] = cm_fatigue
        self.state.extra_vars["gini_fatigue"] = gini_fatigue
        self.state.extra_vars["discussion_status"] = status
