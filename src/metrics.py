"""
metrics.py – Milestone, Failure Condition, and summary metric computation.

Called once per day by CareNetworkModel._collect_metrics().
Returns a dict row that is appended to model.daily_log.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.model import CareNetworkModel

import src.config as C


# ──────────────────────────────────────────────────────────────
# Gini coefficient (for fatigue distribution)
# ──────────────────────────────────────────────────────────────

def gini(values: list[float]) -> float:
    """Compute Gini coefficient (0=equal, 1=maximally unequal)."""
    if len(values) < 2:
        return 0.0
    arr = sorted(values)
    n = len(arr)
    cumsum = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(arr, 1):
        cumsum += v
        weighted_sum += i * v
    mean = cumsum / n
    if mean == 0:
        return 0.0
    return (2 * weighted_sum / (n * cumsum)) - (n + 1) / n


# ──────────────────────────────────────────────────────────────
# Main daily metrics function
# ──────────────────────────────────────────────────────────────

def compute_daily_metrics(model: "CareNetworkModel", day: int) -> dict[str, Any]:
    from src.agents import Elder, WorkerAgent, Provider, Family, LinkWorker, AIWatcher

    elders = model.elders
    workers = model.workers
    providers = model.providers
    n_elders = len(elders) if elders else 1

    # ── Elder aggregates ──────────────────────────────────────
    mean_health = float(np.mean([e.health for e in elders])) if elders else 0.0
    mean_sdh = float(np.mean([e.sdh_risk for e in elders])) if elders else 0.0
    mean_iso = float(np.mean([e.isolation for e in elders])) if elders else 0.0
    mean_dep = float(np.mean([e.acute_dependence for e in elders])) if elders else 0.0

    # Daily acute events
    acute_today = sum(1 for e in elders if e.acute_event_today)
    # Prevention received today
    prevention_today = sum(1 for e in elders if e.received_prevention_today)
    # Social link today
    social_link_today = sum(1 for e in elders if e.received_social_link_today)

    # ── Worker aggregates ─────────────────────────────────────
    fatigues = [w.fatigue for w in workers]
    mean_fatigue = float(np.mean(fatigues)) if fatigues else 0.0
    gini_fatigue = gini(fatigues)
    burnout_count = sum(1 for w in workers if w.burnout)

    # Top-10% load ratio (tasks done today)
    tasks_today = [w.tasks_done_today for w in workers]
    if tasks_today:
        total_tasks = sum(tasks_today)
        k = max(1, len(tasks_today) // 10)
        top_k = sorted(tasks_today, reverse=True)[:k]
        top10_ratio = sum(top_k) / total_tasks if total_tasks > 0 else 0.0
    else:
        top10_ratio = 0.0

    # Cooperation acts today (altruistic)
    coop_today = sum(w.cooperation_acts_today for w in workers)
    coop_rate = coop_today / n_elders  # normalised per elder

    # ── Virtue, Eudaimonia & SWF (Modified Social Welfare) ────
    mean_virtue = float(np.mean([getattr(w, 'virtue', 0.0) for w in workers])) if workers else 0.0
    mean_eudaimonia = float(np.mean([getattr(w, 'eudaimonia', 0.5) for w in workers])) if workers else 0.0

    LAMBDA_WEIGHT = 1.0  # 徳的幸福（エウダイモニア）の重み
    GAMMA_WEIGHT = 5.0   # やりがい搾取（閾値超過）に対するペナルティ重み
    F_THRESHOLD = 0.90   # バーンアウト閾値

    if workers:
        # U_i: 経済的効用（モデル上の報酬変数を代理指標とする。デフォルトは1.0）
        u_total = sum(getattr(w, 'reward', 1.0) for w in workers)
        # E_i: エウダイモニア（徳的幸福）
        e_total = sum(getattr(w, 'eudaimonia', 0.5) for w in workers)
        # Penalty: 疲労閾値超過に対するペナルティ
        penalty_total = sum(max(0.0, w.fatigue - F_THRESHOLD) for w in workers)
        # 社会的厚生（SWF）の算出
        social_welfare = u_total + (LAMBDA_WEIGHT * e_total) - (GAMMA_WEIGHT * penalty_total)
    else:
        u_total = 0.0
        e_total = 0.0
        penalty_total = 0.0
        social_welfare = 0.0

    # ── Coordination level ────────────────────────────────────
    coord = model.coordination_level

    # ── Milestones ────────────────────────────────────────────
    # M1: prevention was executed
    m1 = int(prevention_today > 0)
    # M2: medical↔care info sharing (proxy: coordination_level > 0.55 AND discharge completed today)
    # Scenario A: coord is capped at coordination_level_init + 0.20 = 0.50 < 0.55 → M2 rarely fires
    # Scenario B/C: coord starts at 0.50 and N1/successful coordination can push it above 0.55 → M2 fires
    # This correctly differentiates fragmented (A) from community-supported (B/C) networks.
    discharge_completed_today = sum(
        1 for t in model.completed_tasks
        if t.kind == "discharge" and t.completed_day == day
    )
    m2 = int(coord > 0.55 and discharge_completed_today > 0)
    # M3: acute event followed up within 48h
    m3_hits = 0
    for t in model.completed_tasks:
        if t.kind == "discharge" and t.completed_day == day:
            delay = (t.completed_day - t.created_day)
            if delay <= C.M3_FOLLOWUP_WINDOW_DAYS:
                m3_hits += 1
    m3 = int(m3_hits > 0)
    # M4: high-SDH × high-isolation elders connected to community resources
    if elders:
        sdh_vals = [e.sdh_risk for e in elders]
        iso_vals = [e.isolation for e in elders]
        sdh_thresh = float(np.percentile(sdh_vals, C.M4_HIGH_SDH_PERCENTILE * 100))
        iso_thresh = float(np.percentile(iso_vals, C.M4_HIGH_ISOLATION_PERCENTILE * 100))
        high_risk = [
            e for e in elders
            if e.sdh_risk >= sdh_thresh and e.isolation >= iso_thresh
        ]
        m4_linked = sum(1 for e in high_risk if e.received_social_link_today)
        m4 = int(m4_linked > 0)
    else:
        m4 = 0
    # M5: Gini improved compared to previous day
    prev_gini = model._prev_gini
    m5 = int(prev_gini is not None and (prev_gini - gini_fatigue) >= C.M5_GINI_IMPROVEMENT_THRESHOLD)
    model._prev_gini = gini_fatigue

    # Update cumulative milestones
    for k_ms, v_ms in [("M1", m1), ("M2", m2), ("M3", m3), ("M4", m4), ("M5", m5)]:
        model.milestone_counts[k_ms] += v_ms

    # ── FC-A ──────────────────────────────────────────────────
    # FC-A1: consecutive days with zero prevention
    if prevention_today == 0:
        model._consec_no_prevention += 1
    else:
        model._consec_no_prevention = 0
    fca1 = int(model._consec_no_prevention >= C.FC_A1_CONSEC_ZERO_PREVENTION)
    if fca1:
        model.fc_counts["FC-A1"] += 1

    # FC-A2: discharge task stale > threshold days
    stale_discharge = sum(
        1 for t in model.pending_tasks
        if t.kind == "discharge" and (day - t.created_day) > C.FC_A2_DISCHARGE_DELAY_DAYS
    )
    fca2 = int(stale_discharge > 0)
    if fca2:
        model.fc_counts["FC-A2"] += 1

    # FC-A3: acute event rate over window
    model._acute_history.append(acute_today)
    if len(model._acute_history) == C.FC_A3_ACUTE_RATE_WINDOW:
        rate_window = sum(model._acute_history) / (C.FC_A3_ACUTE_RATE_WINDOW * n_elders)
        fca3 = int(rate_window >= C.FC_A3_ACUTE_RATE_THRESHOLD)
    else:
        fca3 = 0
    if fca3:
        model.fc_counts["FC-A3"] += 1

    # ── FC-B ──────────────────────────────────────────────────
    # FC-B1: top-10% load concentration
    fcb1 = int(top10_ratio >= C.FC_B1_LOAD_TOP10_THRESHOLD)
    if fcb1:
        model.fc_counts["FC-B1"] += 1

    # FC-B2: burnout chain in window
    model._burnout_window.append(burnout_count)
    fcb2 = int(
        len(model._burnout_window) == C.FC_B2_BURNOUT_WINDOW
        and sum(model._burnout_window) >= C.FC_B2_BURNOUT_COUNT
    )
    if fcb2:
        model.fc_counts["FC-B2"] += 1

    # FC-B3: community cooperation rate drop
    model._coop_history.append(coop_rate)
    if len(model._coop_history) == C.FC_B3_COMMUNITY_RATE_WINDOW:
        avg_coop = float(np.mean(list(model._coop_history)))
        fcb3 = int(avg_coop < C.FC_B3_COMMUNITY_RATE_THRESHOLD)
    else:
        fcb3 = 0
    if fcb3:
        model.fc_counts["FC-B3"] += 1

    # FC-B4: altruism-burnout negative correlation (compute every 10 days)
    fcb4 = 0
    if day % 10 == 0 and workers:
        altruism_vals = [w.altruism for w in workers]
        burnout_flags = [1 if w.burnout else 0 for w in workers]
        if len(set(burnout_flags)) > 1:
            # High-altruism (top 50%) burnout rate vs low-altruism burnout rate
            med_alt = float(np.median(altruism_vals))
            high_alt = [w.burnout for w in workers if w.altruism >= med_alt]
            low_alt = [w.burnout for w in workers if w.altruism < med_alt]
            rate_high = sum(high_alt) / len(high_alt) if high_alt else 0
            rate_low = sum(low_alt) / len(low_alt) if low_alt else 0
            if rate_high > rate_low + 0.05:
                fcb4 = 1
                model.fc_counts["FC-B4"] += 1

    # ── FC-C ──────────────────────────────────────────────────
    ai_watchers = model.ai_watchers
    total_nudges = sum(a.total_interventions for a in ai_watchers)
    rejected = sum(a.nudge_rejected_count for a in ai_watchers)
    fcc1 = int(total_nudges > 0 and (rejected / total_nudges) >= C.FC_C1_REJECTION_THRESHOLD)
    if fcc1:
        model.fc_counts["FC-C1"] += 1

    fcc2 = int(total_nudges > C.FC_C2_INTERVENTION_MAX)
    if fcc2 and not model.fc_counts.get("FC-C2_triggered"):
        model.fc_counts["FC-C2"] += 1
        model.fc_counts["FC-C2_triggered"] = 1  # type: ignore[assignment]

    # FC-C3: tradeoff – acute decreased but isolation rose
    prev_iso = model._prev_mean_isolation
    fcc3 = 0
    if prev_iso is not None:
        iso_delta = mean_iso - prev_iso
        if iso_delta >= C.FC_C3_TRADEOFF_ISOLATION_DELTA:
            fcc3 = 1
            model.fc_counts["FC-C3"] += 1
    model._prev_mean_isolation = mean_iso

    # ── Assemble row ──────────────────────────────────────────
    row: dict[str, Any] = {
        "day": day,
        "scenario": model.scenario_cfg.scenario,
        "seed": model.scenario_cfg.seed,
        # Elder
        "mean_health": round(mean_health, 4),
        "mean_sdh_risk": round(mean_sdh, 4),
        "mean_isolation": round(mean_iso, 4),
        "mean_acute_dependence": round(mean_dep, 4),
        "acute_events_today": acute_today,
        "prevention_acts_today": prevention_today,
        "social_links_today": social_link_today,
        # Worker
        "mean_fatigue": round(mean_fatigue, 4),
        "gini_fatigue": round(gini_fatigue, 4),
        "burnout_count": burnout_count,
        "top10_load_ratio": round(top10_ratio, 4),
        "coop_acts_today": coop_today,
        # Virtue & SWF
        "mean_virtue": round(mean_virtue, 4),
        "mean_eudaimonia": round(mean_eudaimonia, 4),
        "swf_utility": round(float(u_total), 4),
        "swf_eudaimonia": round(float(e_total), 4),
        "swf_penalty": round(float(penalty_total), 4),
        "social_welfare": round(float(social_welfare), 4),
        # Network
        "coordination_level": round(coord, 4),
        "pending_tasks": len(model.pending_tasks),
        # Milestones (daily hit)
        "M1": m1, "M2": m2, "M3": m3, "M4": m4, "M5": m5,
        # Failure conditions (daily fire)
        "FC_A1": fca1, "FC_A2": fca2, "FC_A3": fca3,
        "FC_B1": fcb1, "FC_B2": fcb2, "FC_B3": fcb3, "FC_B4": fcb4,
        "FC_C1": fcc1, "FC_C2": fcc2, "FC_C3": fcc3,
        # Cumulative
        "cum_acute_events": sum(e.total_acute_events for e in elders),
        "cum_M1": model.milestone_counts["M1"],
        "cum_M2": model.milestone_counts["M2"],
        "cum_M3": model.milestone_counts["M3"],
        "cum_M4": model.milestone_counts["M4"],
        "cum_M5": model.milestone_counts["M5"],
        "cum_FC_A1": model.fc_counts["FC-A1"],
        "cum_FC_A2": model.fc_counts["FC-A2"],
        "cum_FC_A3": model.fc_counts["FC-A3"],
        "cum_FC_B1": model.fc_counts["FC-B1"],
        "cum_FC_B2": model.fc_counts["FC-B2"],
        "cum_FC_B3": model.fc_counts["FC-B3"],
        "cum_FC_B4": model.fc_counts["FC-B4"],
        "cum_FC_C1": model.fc_counts["FC-C1"],
        "cum_FC_C2": model.fc_counts["FC-C2"],
        "cum_FC_C3": model.fc_counts["FC-C3"],
        "total_nudge_interventions": total_nudges,
    }
    return row