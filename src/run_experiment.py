"""
run_experiment.py – Run A/B/C scenarios (and ablations) over multiple seeds.

Usage examples:
  python -m src.run_experiment --scenario A --days 100
  python -m src.run_experiment --scenario B --days 100
  python -m src.run_experiment --scenario C --days 100
  python -m src.run_experiment --scenario all --days 100 --seeds 10
  python -m src.run_experiment --scenario C-noN2 --days 100 --seeds 10
  python -m src.run_experiment --scenario C-noN3 --days 100 --seeds 10
  python -m src.run_experiment --scenario C-onlyL1 --days 100 --seeds 10
  python -m src.run_experiment --all-including-ablations --days 100 --seeds 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

# Ensure project root is on the path when run as a module
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ScenarioConfig, get_scenario_config, DEFAULT_DAYS, DEFAULT_SEEDS
from src.model import CareNetworkModel


RESULTS_DIR = ROOT / "results"


# ──────────────────────────────────────────────────────────────
# Single run
# ──────────────────────────────────────────────────────────────

def run_single(scenario: str, days: int, seed: int, verbose: bool = False) -> pd.DataFrame:
    """Run one scenario/seed combination and return the daily DataFrame."""
    cfg = get_scenario_config(scenario, days=days, seed=seed)
    model = CareNetworkModel(cfg)
    model.run()
    df = pd.DataFrame(model.daily_log)
    # Tag with the label (distinguishes C from C-noN2 etc.)
    df["label"] = scenario
    return df


# ──────────────────────────────────────────────────────────────
# Multi-seed run
# ──────────────────────────────────────────────────────────────

def run_scenario_multi_seed(
    scenario: str,
    days: int,
    seeds: Sequence[int],
    verbose: bool = True,
) -> pd.DataFrame:
    """Run a scenario over multiple seeds and concatenate results."""
    frames = []
    for seed in seeds:
        t0 = time.perf_counter()
        df = run_single(scenario, days, seed)
        elapsed = time.perf_counter() - t0
        if verbose:
            last = df.iloc[-1]
            print(
                f"  [{scenario}] seed={seed:2d}  "
                f"acute={last['cum_acute_events']:4.0f}  "
                f"burnout={last['burnout_count']:3.0f}  "
                f"iso={last['mean_isolation']:.3f}  "
                f"({elapsed:.1f}s)"
            )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────────────────────
# Summary statistics (mean ± std over seeds for final day)
# ──────────────────────────────────────────────────────────────

def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean and std of key metrics for the final day of each seed."""
    key_cols = [
        "mean_sdh_risk", "mean_isolation", "mean_acute_dependence",
        "mean_fatigue", "gini_fatigue", "burnout_count",
        "cum_acute_events",
        "mean_virtue", "mean_eudaimonia",
        "cum_M1", "cum_M2", "cum_M3", "cum_M4", "cum_M5",
        "cum_FC_A1", "cum_FC_A2", "cum_FC_A3",
        "cum_FC_B1", "cum_FC_B2", "cum_FC_B3", "cum_FC_B4",
        "cum_FC_C1", "cum_FC_C2", "cum_FC_C3",
        "total_nudge_interventions",
    ]
    last_day = df.groupby(["label", "seed"]).last().reset_index()
    available = [c for c in key_cols if c in last_day.columns]
    mean_df = last_day.groupby("label")[available].mean().add_suffix("_mean")
    std_df = last_day.groupby("label")[available].std().add_suffix("_std")
    return pd.concat([mean_df, std_df], axis=1).sort_index(axis=1)


# ──────────────────────────────────────────────────────────────
# Save helpers
# ──────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    print(f"  Saved → {path.relative_to(ROOT)}")
    return path


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────

VALID_SCENARIOS = ["A", "B", "C", "C-noN2", "C-noN3", "C-onlyL1"]
ABLATION_SCENARIOS = ["C-noN2", "C-noN3", "C-onlyL1"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="DAI-CICS multi-agent simulation runner"
    )
    parser.add_argument(
        "--scenario",
        choices=VALID_SCENARIOS + ["all"],
        default="A",
        help="Which scenario to run (or 'all' for A+B+C)",
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument(
        "--seeds", type=int, default=10,
        help="Number of random seeds (0 … seeds-1)"
    )
    parser.add_argument(
        "--all-including-ablations", action="store_true",
        help="Run A, B, C, and all ablation conditions",
    )
    parser.add_argument("--plot", action="store_true", help="Generate plots after running")
    args = parser.parse_args(argv)

    seeds = list(range(args.seeds))
    days = args.days

    # Decide which scenarios to run
    if args.all_including_ablations:
        scenarios_to_run = VALID_SCENARIOS  # A, B, C, C-noN2, C-noN3, C-onlyL1
    elif args.scenario == "all":
        scenarios_to_run = ["A", "B", "C"]
    else:
        scenarios_to_run = [args.scenario]

    all_frames: list[pd.DataFrame] = []

    for sc in scenarios_to_run:
        print(f"\n{'='*60}")
        print(f"Running scenario: {sc}  |  days={days}  |  seeds={seeds}")
        print(f"{'='*60}")
        df = run_scenario_multi_seed(sc, days, seeds, verbose=True)
        fname = f"daily_{sc}_d{days}_s{args.seeds}.csv"
        save_csv(df, fname)
        all_frames.append(df)

    # Combined CSV
    if len(all_frames) > 1:
        combined = pd.concat(all_frames, ignore_index=True)
        save_csv(combined, f"combined_d{days}_s{args.seeds}.csv")
        # Summary
        summary = summarise(combined)
        save_csv(summary.reset_index(), f"summary_d{days}_s{args.seeds}.csv")
        print("\n── Summary (final-day means) ──")
        print(summary.to_string())

    # Optionally generate plots
    if args.plot:
        print("\nGenerating plots …")
        from src.plots import make_all_plots
        combined_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
        make_all_plots(combined_df, days=days, n_seeds=args.seeds)

    print("\nDone.")


if __name__ == "__main__":
    main()