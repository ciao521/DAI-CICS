"""
agents.py – Agent class definitions for DAI-CICS simulation.

Mesa 3.x compatible: Agent.__init__(model) only; unique_id is auto-assigned.

Agent hierarchy:
  BaseAgent
  ├── Elder
  ├── WorkerAgent (Provider, Family, LinkWorker)
  ├── Manager
  └── AIWatcher
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from mesa import Agent

if TYPE_CHECKING:
    from src.model import CareNetworkModel

import src.config as C


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ──────────────────────────────────────────────────────────────
# Elder
# ──────────────────────────────────────────────────────────────

class Elder(Agent):
    """Represents a frail older adult / patient in the care network."""

    def __init__(self, model: "CareNetworkModel") -> None:
        super().__init__(model)
        rng = model.rng_py  # Python random.Random instance
        self.health: float = _clip(rng.gauss(C.ELDER_HEALTH_INIT_MEAN, C.ELDER_HEALTH_INIT_STD))
        self.sdh_risk: float = _clip(rng.gauss(C.ELDER_SDH_RISK_INIT_MEAN, C.ELDER_SDH_RISK_INIT_STD))
        self.isolation: float = _clip(rng.gauss(C.ELDER_ISOLATION_INIT_MEAN, C.ELDER_ISOLATION_INIT_STD))
        self.acute_dependence: float = 0.0

        # flags set during each step
        self.received_prevention_today: bool = False
        self.received_social_link_today: bool = False
        self.acute_event_today: bool = False
        self.discharge_task_pending: bool = False
        self.discharge_task_day: int | None = None

        # cumulative counters
        self.total_acute_events: int = 0

    # ----------------------------------------------------------
    def step(self) -> None:
        """Called once per simulation day."""
        rng = self.model.rng_py

        # 1) Reset daily flags
        self.acute_event_today = False
        self.received_prevention_today = False
        self.received_social_link_today = False

        # 2) Natural health drift
        self.health = _clip(self.health - C.HEALTH_DECAY_RATE)
        self.sdh_risk = _clip(self.sdh_risk + C.SDH_RISK_GROW_RATE)
        self.isolation = _clip(self.isolation + C.ISOLATION_GROW_RATE)

        # 3) Acute event probability
        p_acute = (
            C.ACUTE_EVENT_BASE_PROB
            + 0.10 * self.sdh_risk
            + 0.10 * self.isolation
            + 0.15 * (1.0 - self.health)
        )
        if rng.random() < p_acute:
            self.acute_event_today = True
            self.total_acute_events += 1
            self.acute_dependence = _clip(self.acute_dependence + C.ACUTE_DEPENDENCE_INCREASE)
            self.health = _clip(self.health - 0.10)
            self.discharge_task_pending = True
            self.discharge_task_day = self.model.current_day

        # 4) Dependence slow recovery
        if not self.acute_event_today:
            self.acute_dependence = _clip(self.acute_dependence - C.ACUTE_DEPENDENCE_DECAY)

    # ----------------------------------------------------------
    def apply_prevention_care(self) -> None:
        self.received_prevention_today = True
        self.health = _clip(self.health + C.HEALTH_RECOVERY_FROM_CARE)
        self.sdh_risk = _clip(self.sdh_risk - 0.015)
        self.isolation = _clip(self.isolation - 0.010)

    def apply_social_link(self) -> None:
        self.received_social_link_today = True
        self.isolation = _clip(self.isolation - 0.05)
        self.sdh_risk = _clip(self.sdh_risk - 0.03)


# ──────────────────────────────────────────────────────────────
# Worker (Provider, Family, LinkWorker share the base)
# ──────────────────────────────────────────────────────────────

class WorkerAgent(Agent):
    """Base class for care-network workers."""

    ROLE: str = "worker"

    def __init__(self, model: "CareNetworkModel") -> None:
        super().__init__(model)
        rng = model.rng_py
        self.fatigue: float = _clip(rng.gauss(C.PROVIDER_FATIGUE_INIT_MEAN, C.PROVIDER_FATIGUE_INIT_STD))
        self.altruism: float = _clip(rng.gauss(C.ALTRUISM_MEAN, C.ALTRUISM_STD))
        self.eudaimonia: float = 0.50
        self.capacity_per_day: int = C.PROVIDER_CAPACITY_PER_DAY
        self.burnout: bool = False
        self.burnout_day: int | None = None
        self.tasks_done_today: int = 0
        self.total_tasks_done: int = 0
        self.cooperation_acts_today: int = 0

    # ----------------------------------------------------------
    @property
    def available(self) -> bool:
        return (not self.burnout) and (self.tasks_done_today < self.capacity_per_day)

    def do_task(self, fatigue_multiplier: float = 1.0) -> bool:
        """Attempt to do one task. Returns True if accepted."""
        if not self.available:
            return False
        cfg = self.model.scenario_cfg
        if self.model.rng_py.random() < cfg.provider_avoidance_prob:
            return False
        cost = C.FATIGUE_PER_TASK * fatigue_multiplier
        self.fatigue = _clip(self.fatigue + cost)
        self.tasks_done_today += 1
        self.total_tasks_done += 1
        self.eudaimonia = _clip(self.eudaimonia + C.EUDAIMONIA_GAIN_FROM_HELP)
        self._check_burnout()
        return True

    def do_altruistic_task(self) -> bool:
        """Altruistic cooperation (Scenario B/C). Higher fatigue cost."""
        if not self.available:
            return False
        cfg = self.model.scenario_cfg
        extra = cfg.exploitation_factor
        cost = C.FATIGUE_PER_TASK * (1.0 + extra)
        self.fatigue = _clip(self.fatigue + cost)
        self.tasks_done_today += 1
        self.total_tasks_done += 1
        self.cooperation_acts_today += 1
        self.eudaimonia = _clip(self.eudaimonia + C.EUDAIMONIA_GAIN_FROM_HELP)
        self._check_burnout()
        return True

    def _check_burnout(self) -> None:
        if (not self.burnout) and (self.fatigue >= C.BURNOUT_THRESHOLD):
            self.burnout = True
            self.burnout_day = self.model.current_day

    def step(self) -> None:
        """Daily reset and overnight recovery."""
        self.tasks_done_today = 0
        self.cooperation_acts_today = 0
        if not self.burnout:
            self.fatigue = _clip(self.fatigue - C.FATIGUE_RECOVERY_PER_DAY)
            if self.fatigue >= C.BURNOUT_THRESHOLD:
                self._check_burnout()
        else:
            # Burnout recovery: starts after mandatory 7-day rest period.
            # Recovery rate is 3× normal to model clinical return-to-work support.
            # Re-entry condition: fatigue must drop to BURNOUT_THRESHOLD - 0.15
            # (hysteresis buffer prevents immediate re-burnout on return).
            if self.burnout_day is not None:
                days_since_burnout = self.model.current_day - self.burnout_day
                if days_since_burnout >= 7:
                    recovery = C.FATIGUE_RECOVERY_PER_DAY * 3.0
                    self.fatigue = _clip(self.fatigue - recovery)
                    if self.fatigue < C.BURNOUT_THRESHOLD - 0.15:
                        self.burnout = False
                        self.burnout_day = None


# ──────────────────────────────────────────────────────────────
# Specialised worker roles
# ──────────────────────────────────────────────────────────────

class Provider(WorkerAgent):
    ROLE = "provider"


class Family(WorkerAgent):
    ROLE = "family"

    def __init__(self, model: "CareNetworkModel") -> None:
        super().__init__(model)
        self.altruism = _clip(self.altruism + 0.10)
        self.capacity_per_day = 2


class LinkWorker(WorkerAgent):
    ROLE = "link_worker"

    def __init__(self, model: "CareNetworkModel") -> None:
        super().__init__(model)
        self.capacity_per_day = 4
        self.pending_contacts: list[int] = []

    def step(self) -> None:
        super().step()  # daily reset + overnight fatigue recovery only

    def process_pending_contacts(self) -> None:
        """Process N2-queued contacts (called after AIWatcher nudges fire)."""
        if not self.model.scenario_cfg.link_worker_active:
            return
        if self.pending_contacts:
            elder_id = self.pending_contacts.pop(0)
            elder = self.model.agents_by_uid.get(elder_id)
            if elder and isinstance(elder, Elder):
                elder.apply_social_link()


# ──────────────────────────────────────────────────────────────
# Manager
# ──────────────────────────────────────────────────────────────

class Manager(Agent):
    """Manages the care network."""

    def __init__(self, model: "CareNetworkModel") -> None:
        super().__init__(model)
        self.exploitation_scale: float = 1.0
        self.coordination_boost: float = 0.0

    def step(self) -> None:
        pass


# ──────────────────────────────────────────────────────────────
# AIWatcher
# ──────────────────────────────────────────────────────────────

class AIWatcher(Agent):
    """
    Scenario A/B: passive observer.
    Scenario C: active; evaluates milestones/FCs and applies nudges.
    """

    def __init__(self, model: "CareNetworkModel") -> None:
        super().__init__(model)
        self.nudge_history: list[dict] = []
        self.nudge_accepted_count: int = 0
        self.nudge_rejected_count: int = 0
        self.total_interventions: int = 0

    def step(self) -> None:
        if not self.model.scenario_cfg.ai_active:
            return
        from src.nudges import apply_nudges
        apply_nudges(self, self.model)

    def record_nudge(self, nudge_id: str, level: int, accepted: bool) -> None:
        self.nudge_history.append({
            "day": self.model.current_day,
            "nudge_id": nudge_id,
            "level": level,
            "accepted": accepted,
        })
        self.total_interventions += 1
        if accepted:
            self.nudge_accepted_count += 1
        else:
            self.nudge_rejected_count += 1
