"""
model.py – Mesa 3.x Model for the DAI-CICS care-network simulation.

One step = one day.
"""
from __future__ import annotations

import random as _random
from collections import deque
from typing import Any

import numpy as np
from mesa import Model

from src.agents import (
    AIWatcher, Elder, Family, LinkWorker, Manager, Provider, WorkerAgent,
)
from src.config import ScenarioConfig
import src.config as C


# ──────────────────────────────────────────────────────────────
# Task
# ──────────────────────────────────────────────────────────────

class Task:
    __slots__ = ("task_id", "kind", "elder_id", "created_day", "completed", "completed_day")

    def __init__(self, task_id: int, kind: str, elder_id: int, created_day: int) -> None:
        self.task_id = task_id
        self.kind = kind
        self.elder_id = elder_id
        self.created_day = created_day
        self.completed = False
        self.completed_day: int | None = None


# ──────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────

class CareNetworkModel(Model):
    """Multi-agent care network simulation (Mesa 3.x)."""

    def __init__(self, scenario_cfg: ScenarioConfig) -> None:
        # Mesa 3.x: pass seed to super().__init__
        super().__init__(seed=scenario_cfg.seed)
        self.scenario_cfg = scenario_cfg

        # rng_py: Python random.Random (same seed, used by agents directly)
        # In Mesa 3.x, self.random is a random.Random seeded instance
        self.rng_py = self.random  # alias for agents

        self.current_day: int = 0
        self.coordination_level: float = C.COORDINATION_LEVEL_INIT

        # Task management
        self._next_task_id: int = 0
        self.pending_tasks: list[Task] = []
        self.completed_tasks: list[Task] = []

        # Daily log
        self.daily_log: list[dict[str, Any]] = []

        # uid → agent lookup (needed by LinkWorker for N2 contacts)
        self.agents_by_uid: dict[int, Any] = {}

        # Consecutive days with no prevention
        self._consec_no_prevention: int = 0
        # History window for FC-A3
        self._acute_history: deque[int] = deque(maxlen=C.FC_A3_ACUTE_RATE_WINDOW)
        # History for FC-B3
        self._coop_history: deque[float] = deque(maxlen=C.FC_B3_COMMUNITY_RATE_WINDOW)
        # FC counters
        self.fc_counts: dict[str, int] = {
            "FC-A1": 0, "FC-A2": 0, "FC-A3": 0,
            "FC-B1": 0, "FC-B2": 0, "FC-B3": 0, "FC-B4": 0,
            "FC-C1": 0, "FC-C2": 0, "FC-C3": 0,
        }
        # Milestone cumulative hits
        self.milestone_counts: dict[str, int] = {f"M{i}": 0 for i in range(1, 6)}
        # Window queues
        self._burnout_window: deque[int] = deque(maxlen=C.FC_B2_BURNOUT_WINDOW)
        self._prev_gini: float | None = None
        self._prev_mean_isolation: float | None = None

        # Build agents (Mesa 3.x: just instantiate; auto-registered via self.register_agent)
        self._build_agents()

    # ----------------------------------------------------------
    # Agent creation
    # ----------------------------------------------------------

    def _make_agents(self, cls, n: int) -> None:
        for _ in range(n):
            a = cls(self)
            self.agents_by_uid[a.unique_id] = a

    def _build_agents(self) -> None:
        self._make_agents(Elder, C.N_ELDERS)
        self._make_agents(Provider, C.N_PROVIDERS)
        self._make_agents(Family, C.N_FAMILIES)
        n_lw = C.N_LINK_WORKERS if self.scenario_cfg.link_worker_active else 0
        self._make_agents(LinkWorker, n_lw)
        self._make_agents(Manager, C.N_MANAGERS)
        self._make_agents(AIWatcher, C.N_AI_WATCHERS)

    # ----------------------------------------------------------
    # Typed agent accessors
    # ----------------------------------------------------------

    def _get_agents_of_type(self, cls):
        return list(self.agents_by_type.get(cls, []))

    @property
    def elders(self) -> list[Elder]:
        return self._get_agents_of_type(Elder)

    @property
    def workers(self) -> list[WorkerAgent]:
        result = []
        for cls in (Provider, Family, LinkWorker):
            result.extend(self._get_agents_of_type(cls))
        return result

    @property
    def providers(self) -> list[Provider]:
        return self._get_agents_of_type(Provider)

    @property
    def families(self) -> list[Family]:
        return self._get_agents_of_type(Family)

    @property
    def link_workers(self) -> list[LinkWorker]:
        return self._get_agents_of_type(LinkWorker)

    @property
    def managers(self) -> list[Manager]:
        return self._get_agents_of_type(Manager)

    @property
    def ai_watchers(self) -> list[AIWatcher]:
        return self._get_agents_of_type(AIWatcher)

    # ----------------------------------------------------------
    # Task helpers
    # ----------------------------------------------------------

    def _new_task(self, kind: str, elder_id: int) -> Task:
        t = Task(self._next_task_id, kind, elder_id, self.current_day)
        self._next_task_id += 1
        return t

    # ----------------------------------------------------------
    # Step (= one day)
    # ----------------------------------------------------------

    def step(self) -> None:
        self.current_day += 1
        day = self.current_day

        # Phase 1: Worker overnight recovery & reset
        for w in self.workers:
            w.step()
        for m in self.managers:
            m.step()

        # Phase 2: Elder daily evolution
        for e in self.elders:
            e.step()

        # Phase 3: Task generation
        self._generate_tasks(day)

        # Phase 4: Task assignment & execution
        self._assign_and_execute_tasks(day)

        # Phase 5: Community/altruistic cooperation (B/C)
        self._community_cooperation(day)

        # Phase 6: AIWatcher (C: nudge; A/B: observe)
        for ai in self.ai_watchers:
            ai.step()

        # Phase 7: LinkWorker processes N2 pending contacts (after AIWatcher nudges)
        for lw in self.link_workers:
            lw.process_pending_contacts()

        # Phase 8: Coordination update
        self._update_coordination()

        # Phase 9: Metrics
        self._collect_metrics(day)

    # ----------------------------------------------------------
    # Task generation
    # ----------------------------------------------------------

    def _generate_tasks(self, day: int) -> None:
        cfg = self.scenario_cfg
        for e in self.elders:
            if self.rng_py.random() < cfg.prevention_task_prob:
                self.pending_tasks.append(self._new_task("prevention", e.unique_id))
            if e.discharge_task_pending:
                self.pending_tasks.append(self._new_task("discharge", e.unique_id))
                e.discharge_task_pending = False

    # ----------------------------------------------------------
    # Task assignment
    # ----------------------------------------------------------

    def _assign_and_execute_tasks(self, day: int) -> None:
        cfg = self.scenario_cfg
        use_n3 = cfg.ai_active and cfg.nudge_n3_enabled and not cfg.nudge_only_l1

        available = [w for w in self.workers if w.available]
        if use_n3:
            available.sort(key=lambda w: w.fatigue)

        still_pending: list[Task] = []
        for task in self.pending_tasks:
            elder = self.agents_by_uid.get(task.elder_id)
            if elder is None:
                continue

            if cfg.altruism_cooperation_prob_scale > 0:
                workers_sorted = sorted(available, key=lambda w: -w.altruism)
            else:
                workers_sorted = list(available)

            assigned = False
            for w in workers_sorted:
                if not w.available:
                    continue
                if task.kind == "discharge":
                    if self.rng_py.random() > self.coordination_level:
                        continue
                accepted = w.do_task()
                if accepted:
                    task.completed = True
                    task.completed_day = day
                    self.completed_tasks.append(task)
                    if task.kind == "prevention":
                        elder.apply_prevention_care()
                    elif task.kind == "discharge":
                        self.coordination_level = min(1.0, self.coordination_level + 0.02)
                    assigned = True
                    if w in available:
                        available.remove(w)
                    break

            if not assigned:
                still_pending.append(task)

        self.pending_tasks = still_pending

    # ----------------------------------------------------------
    # Community cooperation (B/C)
    # ----------------------------------------------------------

    def _community_cooperation(self, day: int) -> None:
        cfg = self.scenario_cfg
        if cfg.altruism_cooperation_prob_scale == 0:
            return
        elders_sorted = sorted(
            self.elders, key=lambda e: e.sdh_risk + e.isolation, reverse=True
        )
        for w in self.workers:
            if not w.available:
                continue
            if self.rng_py.random() < w.altruism * cfg.altruism_cooperation_prob_scale * 0.5:
                target = elders_sorted[0] if elders_sorted else None
                if target:
                    ok = w.do_altruistic_task()
                    if ok:
                        target.apply_prevention_care()

    # ----------------------------------------------------------
    # Coordination update
    # ----------------------------------------------------------

    def _update_coordination(self) -> None:
        self.coordination_level += (C.COORDINATION_LEVEL_INIT - self.coordination_level) * 0.02
        for m in self.managers:
            self.coordination_level = min(1.0, self.coordination_level + m.coordination_boost * 0.01)
            m.coordination_boost = max(0.0, m.coordination_boost - 0.1)

    # ----------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------

    def _collect_metrics(self, day: int) -> None:
        from src.metrics import compute_daily_metrics
        row = compute_daily_metrics(self, day)
        self.daily_log.append(row)

    # ----------------------------------------------------------
    # Run
    # ----------------------------------------------------------

    def run(self) -> None:
        for _ in range(self.scenario_cfg.days):
            self.step()
