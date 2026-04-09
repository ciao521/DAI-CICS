"""
config.py – Parameter management for DAI-CICS simulation.
All tuneable constants live here; scenarios override via the SCENARIO_PARAMS dict.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


# ──────────────────────────────────────────────────────────────
# 1. Population
# ──────────────────────────────────────────────────────────────
N_ELDERS: int = 40
N_PROVIDERS: int = 10
N_FAMILIES: int = 10
N_LINK_WORKERS: int = 5
N_MANAGERS: int = 2
N_AI_WATCHERS: int = 1   # always 1; scenario controls behaviour

# ──────────────────────────────────────────────────────────────
# 2. Simulation
# ──────────────────────────────────────────────────────────────
DEFAULT_DAYS: int = 100
DEFAULT_SEEDS: list[int] = list(range(10))   # seeds 0-9 for multi-run

# ──────────────────────────────────────────────────────────────
# 3. Elder state ranges / initial distributions
# ──────────────────────────────────────────────────────────────
ELDER_HEALTH_INIT_MEAN: float = 0.65
ELDER_HEALTH_INIT_STD: float = 0.15
ELDER_SDH_RISK_INIT_MEAN: float = 0.35
ELDER_SDH_RISK_INIT_STD: float = 0.15
ELDER_ISOLATION_INIT_MEAN: float = 0.30
ELDER_ISOLATION_INIT_STD: float = 0.15

# daily natural health drift
HEALTH_DECAY_RATE: float = 0.005          # no-care daily decline
HEALTH_RECOVERY_FROM_CARE: float = 0.04  # boost if preventive care received
SDH_RISK_GROW_RATE: float = 0.008        # without social support
ISOLATION_GROW_RATE: float = 0.007       # without contact

# acute_event probability base
ACUTE_EVENT_BASE_PROB: float = 0.02
ACUTE_DEPENDENCE_INCREASE: float = 0.15  # acute event → dependence up
ACUTE_DEPENDENCE_DECAY: float = 0.01     # daily recovery

# ──────────────────────────────────────────────────────────────
# 4. Provider / Family / LinkWorker initial params
# ──────────────────────────────────────────────────────────────
PROVIDER_FATIGUE_INIT_MEAN: float = 0.20
PROVIDER_FATIGUE_INIT_STD: float = 0.10
PROVIDER_CAPACITY_PER_DAY: int = 3       # max tasks per day
BURNOUT_THRESHOLD: float = 0.90          # fatigue ≥ this → burnout
FATIGUE_PER_TASK: float = 0.06           # base fatigue per task
FATIGUE_RECOVERY_PER_DAY: float = 0.03  # overnight recovery
ALTRUISM_MEAN: float = 0.50
ALTRUISM_STD: float = 0.20

# eudaimonia (meaning) changes
EUDAIMONIA_GAIN_FROM_HELP: float = 0.02
EUDAIMONIA_LOSS_OVERLOAD: float = 0.05

# ──────────────────────────────────────────────────────────────
# 5. Network
# ──────────────────────────────────────────────────────────────
COORDINATION_LEVEL_INIT: float = 0.50   # 0-1

# ──────────────────────────────────────────────────────────────
# 6. Milestone / Failure thresholds
# ──────────────────────────────────────────────────────────────
M3_FOLLOWUP_WINDOW_DAYS: int = 2          # 48 h
M4_HIGH_SDH_PERCENTILE: float = 0.75     # top 25% SDH risk
M4_HIGH_ISOLATION_PERCENTILE: float = 0.75
M5_GINI_IMPROVEMENT_THRESHOLD: float = 0.05

# FC-A
FC_A1_CONSEC_ZERO_PREVENTION: int = 5    # days without any prevention → FC
FC_A2_DISCHARGE_DELAY_DAYS: int = 3      # task stale > N days → FC
FC_A3_ACUTE_RATE_WINDOW: int = 7         # window for FC-A3
FC_A3_ACUTE_RATE_THRESHOLD: float = 0.10 # acute events / elders per day

# FC-B
FC_B1_LOAD_TOP10_THRESHOLD: float = 0.40 # top-10% carry > 40% total work
FC_B2_BURNOUT_WINDOW: int = 7
FC_B2_BURNOUT_COUNT: int = 3
FC_B3_COMMUNITY_RATE_THRESHOLD: float = 0.05  # cooperation acts/elder/day
FC_B3_COMMUNITY_RATE_WINDOW: int = 7

# FC-C
FC_C1_REJECTION_THRESHOLD: float = 0.50  # >50% nudge rejections
FC_C2_INTERVENTION_MAX: int = 80         # nudge interventions per run
FC_C3_TRADEOFF_ISOLATION_DELTA: float = 0.05  # isolation rises by this

# ──────────────────────────────────────────────────────────────
# 7. Scenario-specific overrides
# ──────────────────────────────────────────────────────────────
@dataclass
class ScenarioConfig:
    scenario: str = "A"
    days: int = DEFAULT_DAYS
    seed: int = 0

    # Scenario A flags
    community_fire_prob: float = 0.0      # A=0, B/C>0
    provider_avoidance_prob: float = 0.60 # A: high avoidance
    prevention_task_prob: float = 0.0     # A: zero prevention (spec: M1=0 continues)
    link_worker_active: bool = False       # A: inactive
    # Coordination level: scenario-specific initial value AND mean-reversion target.
    # A: low (0.30) → discharge tasks stall easily (FC-A2)
    # B/C: normal (0.50) → discharge adjustable with some probability
    coordination_level_init: float = 0.50

    # Scenario B additions
    altruism_cooperation_prob_scale: float = 0.0  # B/C: use altruism to cooperate
    exploitation_factor: float = 0.0              # B/C: extra fatigue on altruistic

    # Scenario C additions
    ai_active: bool = False               # C: AIWatcher intervenes
    nudge_min_level: int = 1             # C: minimum nudge level
    nudge_max_level: int = 4
    # Ablation flags
    nudge_n2_enabled: bool = True
    nudge_n3_enabled: bool = True
    nudge_only_l1: bool = False          # C-onlyL1 ablation

    # Extra
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "days": self.days,
            "seed": self.seed,
            "community_fire_prob": self.community_fire_prob,
            "provider_avoidance_prob": self.provider_avoidance_prob,
            "prevention_task_prob": self.prevention_task_prob,
            "link_worker_active": self.link_worker_active,
            "altruism_cooperation_prob_scale": self.altruism_cooperation_prob_scale,
            "exploitation_factor": self.exploitation_factor,
            "ai_active": self.ai_active,
            "nudge_n2_enabled": self.nudge_n2_enabled,
            "nudge_n3_enabled": self.nudge_n3_enabled,
            "nudge_only_l1": self.nudge_only_l1,
        }


def get_scenario_config(scenario: str, days: int = DEFAULT_DAYS, seed: int = 0,
                        ablation: str | None = None) -> ScenarioConfig:
    """
    Factory for scenario configs.

    scenario: "A" | "B" | "C"
    ablation: None | "C-noN2" | "C-noN3" | "C-onlyL1"
    """
    base = dict(days=days, seed=seed)

    if scenario == "A":
        cfg = ScenarioConfig(
            scenario="A",
            community_fire_prob=0.0,
            provider_avoidance_prob=0.65,
            # prevention_task_prob=0.0: spec says "予防/生活支援タスクはAではほぼ発生しない"
            # (M1=0が続く). Setting to exactly 0.0 ensures M1 never fires in A.
            prevention_task_prob=0.0,
            # coordination_level_init=0.30: A has fragmented coordination.
            # Discharge tasks stall because workers skip them at ~70% rate.
            coordination_level_init=0.30,
            link_worker_active=False,
            altruism_cooperation_prob_scale=0.0,
            exploitation_factor=0.0,
            ai_active=False,
            **base,
        )

    elif scenario == "B":
        cfg = ScenarioConfig(
            scenario="B",
            community_fire_prob=0.30,
            provider_avoidance_prob=0.20,
            prevention_task_prob=0.25,
            coordination_level_init=0.50,
            link_worker_active=True,
            altruism_cooperation_prob_scale=1.0,
            exploitation_factor=0.04,   # extra fatigue on high-altruism agents
            ai_active=False,
            **base,
        )

    elif scenario in ("C", "C-noN2", "C-noN3", "C-onlyL1"):
        cfg = ScenarioConfig(
            scenario="C",
            community_fire_prob=0.30,
            provider_avoidance_prob=0.20,
            prevention_task_prob=0.25,
            coordination_level_init=0.50,
            link_worker_active=True,
            altruism_cooperation_prob_scale=1.0,
            exploitation_factor=0.02,   # partially mitigated by AI
            ai_active=True,
            **base,
        )
        # Apply ablation
        if scenario == "C-noN2":
            cfg.nudge_n2_enabled = False
        elif scenario == "C-noN3":
            cfg.nudge_n3_enabled = False
        elif scenario == "C-onlyL1":
            cfg.nudge_only_l1 = True
        if ablation == "C-noN2":
            cfg.nudge_n2_enabled = False
        elif ablation == "C-noN3":
            cfg.nudge_n3_enabled = False
        elif ablation == "C-onlyL1":
            cfg.nudge_only_l1 = True

    else:
        raise ValueError(f"Unknown scenario: {scenario!r}")

    return cfg
