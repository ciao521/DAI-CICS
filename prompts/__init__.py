"""
prompts/ – System prompt definitions for DAI-CICS LLM agents.

Edit each file to tune agent behaviour without touching the agent/eval logic:

  care_manager.py   → CareManagerAgent system prompt
  doctor.py         → DoctorAgent system prompt
  planner_ai.py     → PlannerAIAgent system prompt
  scenario_agent.py → SCENARIO_PROMPTS dict (A/B/C, used in wandb_eval.py)
  judge.py          → JUDGE_SYSTEM_PROMPT (LLM-as-Judge, used in wandb_eval.py)
"""
from prompts.care_manager import CARE_MANAGER_SYSTEM_PROMPT
from prompts.doctor import DOCTOR_SYSTEM_PROMPT
from prompts.planner_ai import PLANNER_SYSTEM_PROMPT
from prompts.scenario_agent import SCENARIO_PROMPTS
from prompts.judge import JUDGE_SYSTEM_PROMPT

__all__ = [
    "CARE_MANAGER_SYSTEM_PROMPT",
    "DOCTOR_SYSTEM_PROMPT",
    "PLANNER_SYSTEM_PROMPT",
    "SCENARIO_PROMPTS",
    "JUDGE_SYSTEM_PROMPT",
]
