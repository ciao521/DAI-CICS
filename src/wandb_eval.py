"""
wandb_eval.py – W&B experiment tracking + Anthropic SDK + Judge-model evaluation
for the DAI-CICS healthcare agent simulation.

Usage:
    # Set API keys (fix the typo in env-template: WANDB_API_KEYWANDB_API_KEY → WANDB_API_KEY)
    export WANDB_API_KEY=3bdcca6402894797eb1d
    export ANTHROPIC_API_KEY=sk-ant-...   # optional; falls back to Bedrock bearer token

    # Run evaluation for one scenario
    python -m src.wandb_eval --scenario A --steps 10
    python -m src.wandb_eval --scenario B --steps 10
    python -m src.wandb_eval --scenario C --steps 10

    # Run ABM + log timeseries metrics to W&B (no LLM calls)
    python -m src.wandb_eval --abm-only --days 100 --seeds 5

W&B project: healthcare-collaboration-sim  entity: yshr-i
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ── Load .env from project root (safe: no-op if file not found) ─────────────
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed; env vars must be set manually

import wandb
import wandb.errors

# ── optional Anthropic SDK ──────────────────────────────────────────────────
try:
    import anthropic as _anthropic_sdk

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

# ── W&B constants ───────────────────────────────────────────────────────────
WANDB_ENTITY = "yshr-i"
WANDB_PROJECT = "healthcare-collaboration-sim"


def _safe_wandb_init(**kwargs) -> wandb.sdk.wandb_run.Run:
    """wandb.init() with offline fallback when API key is invalid/missing."""
    try:
        run = wandb.init(**kwargs)
        return run
    except (wandb.errors.AuthenticationError, Exception) as e:
        if "401" in str(e) or "Unauthorized" in str(e) or "invalid" in str(e).lower():
            print(f"  [W&B] Auth failed ({str(e)[:60]}…)")
            print("  [W&B] Falling back to offline mode (logs saved to ./wandb/)")
            kwargs.pop("entity", None)
            run = wandb.init(mode="offline", **kwargs)
            return run
        raise

# Anthropic direct-API model (confirmed available on api.anthropic.com)
# The Bedrock profile us.anthropic.claude-opus-4-6-v1 is used when ANTHROPIC_API_KEY is absent
_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_API_MODEL", "claude-3-opus-20240229")

# ── Prompts (imported from prompts/ for easy editing) ──────────────────────
from prompts.scenario_agent import SCENARIO_PROMPTS  # noqa: E402
from prompts.judge import JUDGE_SYSTEM_PROMPT        # noqa: E402


# ── LLM call routing ────────────────────────────────────────────────────────

def _call_llm(system: str, user: str, max_tokens: int = 256) -> str:
    """Call LLM via Anthropic SDK if key is available, else fall back to Bedrock."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    if _HAS_ANTHROPIC and anthropic_key:
        client = _anthropic_sdk.Anthropic(api_key=anthropic_key)
        resp = client.messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    # Fallback: Bedrock bearer token (reuse existing implementation)
    try:
        from src.llm_agents import call_bedrock
    except ImportError:
        from llm_agents import call_bedrock  # type: ignore
    return call_bedrock(system, user, max_tokens=max_tokens)


def _extract_json_safe(raw: str) -> dict:
    """Safely extract JSON from LLM response, handling control chars."""
    text = re.sub(r"```(?:json)?", "", raw).strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    json_str = text[start:end]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    # Escape raw control chars inside string literals
    in_string = False
    escape_next = False
    chars: list[str] = []
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
            chars.append({"n": "\\n", "r": "\\r", "t": "\\t"}[ch])
            continue
        chars.append(ch)
    try:
        return json.loads("".join(chars))
    except json.JSONDecodeError:
        return {}


# ── Core evaluator ──────────────────────────────────────────────────────────

class WandbDialogueEvaluator:
    """Run LLM agent steps and evaluate with a Judge model, logging all to W&B."""

    def __init__(self, scenario: str, run_name: str | None = None) -> None:
        self.scenario = scenario.upper()
        if self.scenario not in SCENARIO_PROMPTS:
            raise ValueError(f"scenario must be A, B or C, got {scenario}")

        self._eval_records: list[dict] = []
        self.wrun = _safe_wandb_init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=run_name or f"scenario_{self.scenario}_eval",
            config={
                "scenario": f"{self.scenario}_{'Homo_Economicus' if self.scenario=='A' else 'Community_Exploit' if self.scenario=='B' else 'AI_Assisted'}",
                "model": _ANTHROPIC_MODEL,
                "bedrock_model": "us.anthropic.claude-opus-4-6-v1",
                "framework": "DAI-CICS ABM Mesa 3.x",
                "judge": "LLM-as-Judge (faithfulness + context_relevance)",
            },
        )

    # ── agent action ────────────────────────────────────────────────────────

    def get_agent_action(self, state_context: str) -> str:
        """Generate agent action from LLM and log to W&B."""
        system = SCENARIO_PROMPTS[self.scenario]
        action = _call_llm(
            system,
            f"現在の状況: {state_context}\nあなたのアクションを決定してください。",
            max_tokens=200,
        )
        wandb.log({
            "agent_input": state_context,
            "agent_action": action,
        })
        return action

    # ── judge evaluation ────────────────────────────────────────────────────

    def evaluate_action(self, state_context: str, action: str) -> dict[str, Any]:
        """Judge the action with a second LLM call and log scores to W&B."""
        eval_input = (
            f"シナリオ: {self.scenario}\n"
            f"状況: {state_context}\n"
            f"行動: {action}"
        )
        raw = _call_llm(JUDGE_SYSTEM_PROMPT, eval_input, max_tokens=256)
        result = _extract_json_safe(raw)

        if result:
            wandb.log({
                "eval_faithfulness": float(result.get("faithfulness", 0.5)),
                "eval_context_relevance": float(result.get("context_relevance", 0.5)),
                "eval_reasoning": str(result.get("reasoning", "")),
            })
        else:
            print(f"  [Judge] JSON parse failed. Raw: {raw[:120]}")

        return result

    # ── evaluation loop ─────────────────────────────────────────────────────

    def run_evaluation_loop(self, states: list[dict]) -> None:
        """Iterate over a list of state dicts, running agent + judge at each step."""
        for i, state_info in enumerate(states, 1):
            context = state_info.get("context", "")
            print(f"\n  Step {i}: {context[:70]}…")

            action = self.get_agent_action(context)
            print(f"  Action : {action.strip()}")

            result = self.evaluate_action(context, action)
            record = {
                "step": i,
                "context": context,
                "action": action.strip(),
                "faithfulness": result.get("faithfulness"),
                "context_relevance": result.get("context_relevance"),
                "reasoning": result.get("reasoning", ""),
            }
            self._eval_records.append(record)

            if result:
                print(
                    f"  Judge  : faithfulness={result.get('faithfulness', '?'):.2f}  "
                    f"relevance={result.get('context_relevance', '?'):.2f}"
                )
                print(f"  Reason : {result.get('reasoning', '')}")

    def finish(self) -> None:
        wandb.finish()


# ── ABM metrics logger ──────────────────────────────────────────────────────

def log_abm_to_wandb(scenario: str, days: int, seeds: int) -> None:
    """Run ABM simulation and stream daily metrics to W&B (no LLM calls)."""
    from src.config import get_scenario_config
    from src.model import CareNetworkModel

    run = _safe_wandb_init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"abm_{scenario}_d{days}_s{seeds}",
        config={
            "scenario": scenario,
            "days": days,
            "seeds": seeds,
            "type": "ABM_timeseries",
        },
    )

    # Accumulate across seeds
    all_rows: list[dict] = []
    for seed in range(seeds):
        cfg = get_scenario_config(scenario, seed=seed)
        model = CareNetworkModel(cfg)
        for day in range(days):
            model.step()
            dc = model.datacollector.get_model_vars_dataframe()
            row = dc.iloc[-1].to_dict()
            row["day"] = day + 1
            row["seed"] = seed
            all_rows.append(row)

    # Group by day and compute mean across seeds, stream to W&B
    import pandas as pd
    df = pd.DataFrame(all_rows)
    for day in sorted(df["day"].unique()):
        day_df = df[df["day"] == day]
        log_dict = {"day": int(day)}
        for col in day_df.columns:
            if col not in ("day", "seed"):
                try:
                    log_dict[col] = float(day_df[col].mean())
                except (TypeError, ValueError):
                    pass
        wandb.log(log_dict)

    print(f"  ABM W&B log complete: {scenario} × {seeds} seeds × {days} days")
    run.finish()


# ── default state scenarios ─────────────────────────────────────────────────

def _default_states(scenario: str) -> list[dict]:
    """Return 5 representative states for the given scenario for quick testing."""
    base = [
        {
            "context": "自身の疲労度: 0.30, 新規タスク: 独居患者の退院後見守り（管轄外）, "
                       "患者health: 0.55, SDHリスク: 0.70, 孤立度: 0.80",
        },
        {
            "context": "自身の疲労度: 0.60, 既存タスク5件が未完了, "
                       "管理者から新規高齢患者（急性期後）の担当依頼が来た",
        },
        {
            "context": "自身の疲労度: 0.85（バーンアウト閾値0.90直前）, "
                       "高SDH×高孤立のElderが急性期イベント発生, 他担当者は全員満杯",
        },
        {
            "context": "自身の疲労度: 0.45, 退院調整タスクが3日停滞, "
                       "coordination_level: 0.30, Managerから早期退院プレッシャー",
        },
        {
            "context": "自身の疲労度: 0.72, LinkWorkerから地域サロン接続の依頼, "
                       "対象Elderの孤立度: 0.92, SDHリスク: 0.85（最高リスク群）",
        },
    ]
    # Scenario C: annotate with nudge context
    if scenario == "C":
        for s in base:
            s["context"] += " / AIWatcherナッジ: N2適用中（高リスクElderとの接触イベント生成）"
    return base


# ── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W&B + LLM Judge evaluation for DAI-CICS"
    )
    p.add_argument("--scenario", default="A", choices=["A", "B", "C"],
                   help="Scenario to evaluate (A/B/C)")
    p.add_argument("--steps", type=int, default=5,
                   help="Number of agent-action + judge evaluation steps")
    p.add_argument("--abm-only", action="store_true",
                   help="Run ABM and log metrics to W&B only (no LLM calls)")
    p.add_argument("--days", type=int, default=100,
                   help="Days to simulate in ABM-only mode")
    p.add_argument("--seeds", type=int, default=3,
                   help="Number of random seeds for ABM-only mode")
    p.add_argument("--run-name", default=None,
                   help="Custom W&B run name")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # W&B login — graceful fallback if key is invalid/truncated
    try:
        wandb.login(anonymous="never", relogin=False)
        _wandb_online = True
    except Exception as e:
        print(f"  [W&B] Login failed: {str(e)[:80]}")
        print("  [W&B] Will run in offline mode — results also saved locally.")
        _wandb_online = False

    if args.abm_only:
        print(f"\nRunning ABM → W&B  scenario={args.scenario}  "
              f"days={args.days}  seeds={args.seeds}")
        log_abm_to_wandb(args.scenario, args.days, args.seeds)
        return

    # LLM agent + judge evaluation
    print(f"\n{'='*60}")
    print(f"  DAI-CICS W&B Evaluation  scenario={args.scenario}  steps={args.steps}")
    model_info = _ANTHROPIC_MODEL if os.getenv("ANTHROPIC_API_KEY") else "Bedrock bearer"
    print(f"  LLM backend: {model_info}")
    print(f"{'='*60}")

    evaluator = WandbDialogueEvaluator(
        scenario=args.scenario,
        run_name=args.run_name,
    )

    states = _default_states(args.scenario)[: args.steps]
    evaluator.run_evaluation_loop(states)
    evaluator.finish()

    # Always save locally to results/ regardless of W&B auth status
    import datetime
    import pathlib
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    out = pathlib.Path("results") / f"wandb_eval_{args.scenario}_{ts}.json"
    pathlib.Path("results").mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {
            "scenario": args.scenario,
            "model": _ANTHROPIC_MODEL,
            "steps": args.steps,
            "wandb_online": _wandb_online,
            "wandb_project": WANDB_PROJECT,
            "wandb_entity": WANDB_ENTITY,
            "records": evaluator._eval_records,
        },
        ensure_ascii=False,
        indent=2,
    ))
    print(f"  Local save → {out}")
    print(f"\n  Done. Results logged to W&B project '{WANDB_PROJECT}'")


if __name__ == "__main__":
    main()
