"""
tests/test_simulation.py – Basic smoke tests for DAI-CICS simulation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_scenario_config, ScenarioConfig
from src.model import CareNetworkModel
from src.agents import Elder, Provider, Family, LinkWorker, Manager, AIWatcher


# ──────────────────────────────────────────────────────────────
# Config tests
# ──────────────────────────────────────────────────────────────

def test_scenario_config_a():
    cfg = get_scenario_config("A", days=10, seed=0)
    assert cfg.scenario == "A"
    assert cfg.community_fire_prob == 0.0
    assert cfg.ai_active is False
    assert cfg.link_worker_active is False


def test_scenario_config_b():
    cfg = get_scenario_config("B", days=10, seed=0)
    assert cfg.scenario == "B"
    assert cfg.altruism_cooperation_prob_scale == 1.0
    assert cfg.exploitation_factor > 0


def test_scenario_config_c():
    cfg = get_scenario_config("C", days=10, seed=0)
    assert cfg.ai_active is True
    assert cfg.link_worker_active is True


def test_scenario_config_ablations():
    c_no_n2 = get_scenario_config("C-noN2", days=10, seed=0)
    assert c_no_n2.nudge_n2_enabled is False

    c_no_n3 = get_scenario_config("C-noN3", days=10, seed=0)
    assert c_no_n3.nudge_n3_enabled is False

    c_only_l1 = get_scenario_config("C-onlyL1", days=10, seed=0)
    assert c_only_l1.nudge_only_l1 is True


def test_invalid_scenario():
    with pytest.raises(ValueError):
        get_scenario_config("Z")


# ──────────────────────────────────────────────────────────────
# Model initialisation tests
# ──────────────────────────────────────────────────────────────

def test_model_init_a():
    cfg = get_scenario_config("A", days=5, seed=42)
    model = CareNetworkModel(cfg)
    assert len(model.elders) == 40
    assert len(model.providers) == 10
    assert len(model.families) == 10
    assert len(model.link_workers) == 0   # Scenario A: no link workers
    assert len(model.managers) == 2
    assert len(model.ai_watchers) == 1


def test_model_init_b():
    cfg = get_scenario_config("B", days=5, seed=42)
    model = CareNetworkModel(cfg)
    assert len(model.link_workers) == 5


def test_model_init_c():
    cfg = get_scenario_config("C", days=5, seed=42)
    model = CareNetworkModel(cfg)
    assert len(model.link_workers) == 5
    ai = model.ai_watchers[0]
    assert isinstance(ai, AIWatcher)


# ──────────────────────────────────────────────────────────────
# Single step / short run tests
# ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("scenario", ["A", "B", "C"])
def test_single_step(scenario):
    cfg = get_scenario_config(scenario, days=1, seed=0)
    model = CareNetworkModel(cfg)
    model.step()
    assert model.current_day == 1
    assert len(model.daily_log) == 1
    row = model.daily_log[0]
    assert "mean_sdh_risk" in row
    assert "mean_isolation" in row
    assert "burnout_count" in row
    assert 0.0 <= row["mean_sdh_risk"] <= 1.0
    assert 0.0 <= row["mean_isolation"] <= 1.0


@pytest.mark.parametrize("scenario", ["A", "B", "C", "C-noN2", "C-noN3", "C-onlyL1"])
def test_run_10_days(scenario):
    cfg = get_scenario_config(scenario, days=10, seed=1)
    model = CareNetworkModel(cfg)
    model.run()
    assert model.current_day == 10
    assert len(model.daily_log) == 10
    # All days present
    days = [r["day"] for r in model.daily_log]
    assert days == list(range(1, 11))


# ──────────────────────────────────────────────────────────────
# Determinism test
# ──────────────────────────────────────────────────────────────

def test_determinism():
    """Same seed should produce identical results."""
    for scenario in ["A", "B", "C"]:
        cfg1 = get_scenario_config(scenario, days=5, seed=7)
        cfg2 = get_scenario_config(scenario, days=5, seed=7)
        m1 = CareNetworkModel(cfg1)
        m2 = CareNetworkModel(cfg2)
        m1.run()
        m2.run()
        for r1, r2 in zip(m1.daily_log, m2.daily_log):
            assert r1["mean_sdh_risk"] == pytest.approx(r2["mean_sdh_risk"])
            assert r1["burnout_count"] == r2["burnout_count"]


# ──────────────────────────────────────────────────────────────
# Scenario A: prevention is near-zero
# ──────────────────────────────────────────────────────────────

def test_scenario_a_low_prevention():
    """Scenario A should have very few prevention acts over 20 days."""
    cfg = get_scenario_config("A", days=20, seed=0)
    model = CareNetworkModel(cfg)
    model.run()
    total_prevention = sum(r["prevention_acts_today"] for r in model.daily_log)
    # With prevention_task_prob=0.02 and 40 elders, expect ~16 tasks generated
    # but avoidance 0.65 further reduces execution → very low
    assert total_prevention < 30, f"Expected low prevention in A, got {total_prevention}"


# ──────────────────────────────────────────────────────────────
# Scenario B: burnout should be higher than A
# ──────────────────────────────────────────────────────────────

def test_scenario_b_more_burnout_than_a():
    """Scenario B exploitation should cause more burnout than A."""
    seeds = [0, 1, 2]
    burnout_a = []
    burnout_b = []
    for s in seeds:
        ma = CareNetworkModel(get_scenario_config("A", days=50, seed=s))
        mb = CareNetworkModel(get_scenario_config("B", days=50, seed=s))
        ma.run(); mb.run()
        burnout_a.append(ma.daily_log[-1]["burnout_count"])
        burnout_b.append(mb.daily_log[-1]["burnout_count"])
    avg_a = sum(burnout_a) / len(burnout_a)
    avg_b = sum(burnout_b) / len(burnout_b)
    assert avg_b >= avg_a, f"Scenario B burnout ({avg_b}) should be >= A ({avg_a})"


# ──────────────────────────────────────────────────────────────
# Metrics sanity
# ──────────────────────────────────────────────────────────────

def test_metrics_in_range():
    cfg = get_scenario_config("C", days=5, seed=3)
    model = CareNetworkModel(cfg)
    model.run()
    for row in model.daily_log:
        assert 0.0 <= row["mean_health"] <= 1.0
        assert 0.0 <= row["mean_sdh_risk"] <= 1.0
        assert 0.0 <= row["mean_isolation"] <= 1.0
        assert 0.0 <= row["mean_fatigue"] <= 1.0
        assert 0.0 <= row["gini_fatigue"] <= 1.0
        assert row["burnout_count"] >= 0
        assert row["acute_events_today"] >= 0


def test_milestone_counts_non_negative():
    cfg = get_scenario_config("C", days=10, seed=5)
    model = CareNetworkModel(cfg)
    model.run()
    for k, v in model.milestone_counts.items():
        assert v >= 0, f"Milestone {k} is negative"
    for k, v in model.fc_counts.items():
        if isinstance(v, int):
            assert v >= 0, f"FC {k} is negative"
