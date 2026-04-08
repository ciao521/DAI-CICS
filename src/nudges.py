"""
nudges.py – Nudge dictionary and application logic for Scenario C.

Nudge levels:
  L1: Information presentation (lowest intervention)
  L2: Soft suggestion / reframing
  L3: Event generation / assignment adjustment
  L4: Hard constraint (reserved, not used by default)

Nudge catalogue:
  N1 (L1/L2): Discharge coordination information re-posting
  N2 (L3):    Social prescription – high-risk Elder × LinkWorker contact event
  N3 (L2/L3): Load redistribution proposal
  N4 (L1):    Fatigue distribution & turnover risk view for Managers

Dynamic Reflection (AgentDynEx style):
  Each day the AIWatcher evaluates current FC states (and early-warning trends)
  and selects the minimal-intervention nudge to address the most critical issue.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.agents import AIWatcher
    from src.model import CareNetworkModel

import src.config as C


# ──────────────────────────────────────────────────────────────
# Individual nudge implementations
# ──────────────────────────────────────────────────────────────

def nudge_n1(ai: "AIWatcher", model: "CareNetworkModel", level: int) -> bool:
    """
    N1 – Discharge coordination information re-post.
    Effect: small boost to coordination_level.
    Level 1: tiny boost; Level 2: moderate boost.
    """
    boost = 0.03 if level == 1 else 0.06
    model.coordination_level = min(1.0, model.coordination_level + boost)
    # Also signal managers
    for m in model.managers:
        m.coordination_boost += boost * 2
    ai.record_nudge("N1", level, accepted=True)
    return True


def nudge_n2(ai: "AIWatcher", model: "CareNetworkModel", level: int) -> bool:
    """
    N2 – Social prescription event: queue high-risk Elders for LinkWorker contact.
    Effect: LinkWorkers get pending_contacts with top-N high-risk elders.
    Level 3 only.
    """
    if not model.scenario_cfg.nudge_n2_enabled:
        ai.record_nudge("N2", level, accepted=False)
        return False
    if model.scenario_cfg.nudge_only_l1:
        ai.record_nudge("N2", level, accepted=False)
        return False

    elders = model.elders
    if not elders:
        ai.record_nudge("N2", level, accepted=False)
        return False

    # Identify top-risk elders (high SDH × isolation)
    sdh_thresh = float(np.percentile([e.sdh_risk for e in elders], 75))
    iso_thresh = float(np.percentile([e.isolation for e in elders], 75))
    high_risk = [
        e for e in elders
        if e.sdh_risk >= sdh_thresh and e.isolation >= iso_thresh
    ]

    link_workers = model.link_workers
    if not link_workers:
        ai.record_nudge("N2", level, accepted=False)
        return False

    # Queue contacts (round-robin across link workers)
    for i, elder in enumerate(high_risk[:4]):  # at most 4 per day
        lw = link_workers[i % len(link_workers)]
        if elder.unique_id not in lw.pending_contacts:
            lw.pending_contacts.append(elder.unique_id)

    ai.record_nudge("N2", level, accepted=True)
    return True


def nudge_n3(ai: "AIWatcher", model: "CareNetworkModel", level: int) -> bool:
    """
    N3 – Load redistribution proposal.
    Effect: temporarily reduce avoidance probability by adjusting model flag;
            also immediately re-sort pending tasks so least-fatigued workers get them.
    Level 2: moderate; Level 3: stronger.
    """
    if not model.scenario_cfg.nudge_n3_enabled:
        ai.record_nudge("N3", level, accepted=False)
        return False
    if model.scenario_cfg.nudge_only_l1:
        ai.record_nudge("N3", level, accepted=False)
        return False

    workers = model.workers
    if not workers:
        ai.record_nudge("N3", level, accepted=False)
        return False

    # Temporarily reduce fatigue for the most loaded workers (simulate delegation)
    fatigues = sorted(workers, key=lambda w: w.fatigue, reverse=True)
    n_help = max(1, len(fatigues) // 5)  # top 20%
    delta = 0.05 if level == 2 else 0.10
    for w in fatigues[:n_help]:
        w.fatigue = max(0.0, w.fatigue - delta)  # simulate redistribution

    ai.record_nudge("N3", level, accepted=True)
    return True


def nudge_n4(ai: "AIWatcher", model: "CareNetworkModel", level: int) -> bool:
    """
    N4 – Fatigue distribution & turnover risk view for Managers (L1).
    Effect: Managers reduce exploitation_scale, get coordination_boost.
    """
    managers = model.managers
    if not managers:
        ai.record_nudge("N4", level, accepted=False)
        return False

    for m in managers:
        m.exploitation_scale = max(0.5, m.exploitation_scale - 0.10)
        m.coordination_boost += 0.5
    ai.record_nudge("N4", level, accepted=True)
    return True


# ──────────────────────────────────────────────────────────────
# Dynamic Reflection: select & apply the right nudge
# ──────────────────────────────────────────────────────────────

def apply_nudges(ai: "AIWatcher", model: "CareNetworkModel") -> None:
    """
    Called each day by AIWatcher.step() when ai_active=True.
    Evaluates current failure conditions and applies the minimal nudge.
    """
    cfg = model.scenario_cfg
    day = model.current_day

    # Gather current FC states from last log entry
    log = model.daily_log
    if not log:
        return
    last = log[-1]

    # ── Early-warning & FC priority logic ─────────────────────
    # Priority order: acute chain > discharge stale > load > burnout > social isolation

    # 1) Discharge stale (FC-A2) or coordination low → N1
    stale_tasks = sum(
        1 for t in model.pending_tasks
        if t.kind == "discharge" and (day - t.created_day) >= C.FC_A2_DISCHARGE_DELAY_DAYS - 1
    )
    if stale_tasks > 0 or model.coordination_level < 0.45:
        _apply_with_min_level(ai, model, "N1", 1)
        return

    # 2) Load concentration (FC-B1) or upcoming burnout → N3
    fatigues = [w.fatigue for w in model.workers]
    if fatigues:
        top_k = max(1, len(fatigues) // 10)
        top_load = sorted(fatigues, reverse=True)[:top_k]
        at_risk_burnout = sum(1 for f in fatigues if f > 0.75)

        if last.get("top10_load_ratio", 0) >= C.FC_B1_LOAD_TOP10_THRESHOLD * 0.8 or at_risk_burnout >= 2:
            level = 2 if not cfg.nudge_only_l1 else 1
            _apply_with_min_level(ai, model, "N3", level)
            return

    # 3) High isolation + high SDH elders unconnected → N2 (L3)
    elders = model.elders
    if elders:
        sdh_vals = [e.sdh_risk for e in elders]
        iso_vals = [e.isolation for e in elders]
        sdh_thresh = float(np.percentile(sdh_vals, 75))
        iso_thresh = float(np.percentile(iso_vals, 75))
        high_risk_unconnected = [
            e for e in elders
            if e.sdh_risk >= sdh_thresh
            and e.isolation >= iso_thresh
            and not e.received_social_link_today
        ]
        if len(high_risk_unconnected) >= 2 and not cfg.nudge_only_l1:
            _apply_with_min_level(ai, model, "N2", 3)
            return

    # 4) Burnout or high fatigue → N4 (Manager alert, L1)
    burnout_count = sum(1 for w in model.workers if w.burnout)
    mean_fatigue = float(np.mean(fatigues)) if fatigues else 0.0
    if burnout_count >= 1 or mean_fatigue > 0.60:
        _apply_with_min_level(ai, model, "N4", 1)
        return

    # 5) No critical issue: do nothing


def _apply_with_min_level(
    ai: "AIWatcher",
    model: "CareNetworkModel",
    nudge_id: str,
    level: int,
) -> None:
    """Apply a nudge, respecting FC-C2 intervention cap."""
    if ai.total_interventions >= C.FC_C2_INTERVENTION_MAX:
        ai.record_nudge(nudge_id, level, accepted=False)
        return

    dispatch = {"N1": nudge_n1, "N2": nudge_n2, "N3": nudge_n3, "N4": nudge_n4}
    fn = dispatch.get(nudge_id)
    if fn:
        fn(ai, model, level)
