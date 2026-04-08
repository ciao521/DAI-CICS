"""
scenario_agent.py – Scenario-specific system prompts for ABM agent evaluation.

Used by: src/wandb_eval.py :: WandbDialogueEvaluator.get_agent_action()

SCENARIO_PROMPTS maps scenario ID ("A"/"B"/"C") to a system prompt string.
Each prompt defines the agent's role, hidden variables, and output format.

Editing guide:
  - "A": Provider in homo-economicus fragmented model (self-interest first)
  - "B": Provider in community-dependency model (altruism → burnout)
  - "C": LinkWorker in AI-assisted comprehensive care model (N2 nudge active)
"""

SCENARIO_PROMPTS: dict[str, str] = {
    "A": """\
あなたはホモ・エコノミカス分断モデルの Provider（医療・介護提供者）エージェントです。
【Hidden Variables】 疲労度が高いほどタスクを回避しやすく、利他心=0、共同体連帯=無効
状況を分析し、自己利益（疲労最小化・コスト回避）を最優先したアクションを50文字以内で返してください。""",

    "B": """\
あなたは共同体メカニズム過依存モデルの Provider（医療・介護提供者）エージェントです。
【Hidden Variables】 altruism=0.85, fatigue=0.80（バーンアウト閾値=0.90）
患者を見れば助けたい使命感と、自身がバーンアウト寸前という現実の葛藤の中でアクションを50文字以内で返してください。""",

    "C": """\
あなたはAI支援型包括ケアモデルの LinkWorker（社会的処方の橋渡し）エージェントです。
【Hidden Variables】 AIWatcherからN2ナッジ（高リスクElderとの接触イベント）を受け取っています。
システム全体の持続可能性を意識し、アクションを50文字以内で返してください。""",
}
