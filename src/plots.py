"""
plots.py – Matplotlib visualisations for DAI-CICS simulation results.

Produced figures:
  1. comparison_main.png     – A/B/C time-series comparison (4 panels)
  2. comparison_extended.png – A/B/C acute & coordination (4 panels)
  3. ablation_final.png      – Ablation bar chart (final-day values, 4 metrics)
  4. ablation_timeseries.png – Ablation time-series (4 panels)
  5. fc_heatmap.png          – Failure condition cumulative counts heatmap
  6. milestones.png          – Milestone achievement rate time-series
  7. virtue_eudaimonia.png   – Virtue and Eudaimonia Dynamics
  8. swf_timeseries.png      – Modified Social Welfare Function Dynamics [追加]
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# Colour palette
PALETTE = {
    "A": "#e74c3c",
    "B": "#f39c12",
    "C": "#2ecc71",
    "C-noN2": "#3498db",
    "C-noN3": "#9b59b6",
    "C-onlyL1": "#1abc9c",
}
LINE_STYLES = {
    "A": "-",
    "B": "--",
    "C": "-",
    "C-noN2": ":",
    "C-noN3": "-.",
    "C-onlyL1": (0, (3, 1, 1, 1)),
}


# ──────────────────────────────────────────────────────────────
# Helper: compute mean (and ±1 std) over seeds per day
# ──────────────────────────────────────────────────────────────

def _agg(df: pd.DataFrame, label: str, col: str):
    sub = df[df["label"] == label]
    g = sub.groupby("day")[col]
    days = g.mean().index.values
    mean = g.mean().values
    std = g.std().fillna(0).values
    return days, mean, std


def _fill_plot(ax, days, mean, std, color, ls, label, alpha=0.15):
    ax.plot(days, mean, color=color, ls=ls, lw=1.8, label=label)
    ax.fill_between(days, mean - std, mean + std, color=color, alpha=alpha)


# ──────────────────────────────────────────────────────────────
# Figure 1: A/B/C comparison (main scenarios)
# ──────────────────────────────────────────────────────────────

def plot_abc_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    labels = [l for l in ["A", "B", "C"] if l in df["label"].unique()]
    if not labels:
        return

    metrics = [
        ("mean_sdh_risk", "Mean SDH Risk"),
        ("mean_isolation", "Mean Isolation"),
        ("mean_fatigue", "Mean Provider Fatigue"),
        ("burnout_count", "Burnout Count"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, metrics):
        for lbl in labels:
            if col not in df.columns:
                continue
            days, mean, std = _agg(df, lbl, col)
            _fill_plot(ax, days, mean, std,
                       color=PALETTE.get(lbl, "grey"),
                       ls=LINE_STYLES.get(lbl, "-"),
                       label=f"Scenario {lbl}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Day")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Scenario A/B/C Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "comparison_main.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> results/{path.name}")


# ──────────────────────────────────────────────────────────────
# Figure 2: Extended A/B/C (acute events + coordination)
# ──────────────────────────────────────────────────────────────

def plot_abc_extended(df: pd.DataFrame, out_dir: Path) -> None:
    labels = [l for l in ["A", "B", "C"] if l in df["label"].unique()]
    if not labels:
        return

    metrics = [
        ("cum_acute_events", "Cumulative Acute Events"),
        ("coordination_level", "Coordination Level"),
        ("mean_acute_dependence", "Mean Acute Dependence"),
        ("mean_health", "Mean Elder Health"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, metrics):
        for lbl in labels:
            if col not in df.columns:
                continue
            days, mean, std = _agg(df, lbl, col)
            _fill_plot(ax, days, mean, std,
                       color=PALETTE.get(lbl, "grey"),
                       ls=LINE_STYLES.get(lbl, "-"),
                       label=f"Scenario {lbl}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Day")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Scenario A/B/C – Acute & Coordination", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "comparison_extended.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> results/{path.name}")


# ──────────────────────────────────────────────────────────────
# Figure 3: Ablation bar chart (final day)
# ──────────────────────────────────────────────────────────────

def plot_ablation_bar(df: pd.DataFrame, out_dir: Path) -> None:
    ablation_labels = [l for l in ["C", "C-noN2", "C-noN3", "C-onlyL1"] if l in df["label"].unique()]
    if len(ablation_labels) < 2:
        return

    metrics = [
        ("burnout_count", "Burnout Count (final day)"),
        ("cum_acute_events", "Cumulative Acute Events"),
        ("mean_isolation", "Mean Isolation (final day)"),
        ("gini_fatigue", "Gini Fatigue (final day)"),
    ]

    last = df.groupby(["label", "seed"]).last().reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, (col, title) in zip(axes, metrics):
        if col not in last.columns:
            continue
        vals = [last[last["label"] == l][col].mean() for l in ablation_labels]
        errs = [last[last["label"] == l][col].std() for l in ablation_labels]
        colors = [PALETTE.get(l, "grey") for l in ablation_labels]
        ax.bar(ablation_labels, vals, yerr=errs, color=colors,
               capsize=4, edgecolor="black", linewidth=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(len(ablation_labels)))
        ax.set_xticklabels(ablation_labels, rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Ablation Study: Final-Day Metrics", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "ablation_final.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> results/{path.name}")


# ──────────────────────────────────────────────────────────────
# Figure 4: Ablation time-series
# ──────────────────────────────────────────────────────────────

def plot_ablation_timeseries(df: pd.DataFrame, out_dir: Path) -> None:
    ablation_labels = [l for l in ["C", "C-noN2", "C-noN3", "C-onlyL1"] if l in df["label"].unique()]
    if len(ablation_labels) < 2:
        return

    metrics = [
        ("burnout_count", "Burnout Count"),
        ("cum_acute_events", "Cumulative Acute Events"),
        ("mean_isolation", "Mean Isolation"),
        ("gini_fatigue", "Gini Fatigue"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, metrics):
        for lbl in ablation_labels:
            if col not in df.columns:
                continue
            days, mean, std = _agg(df, lbl, col)
            _fill_plot(ax, days, mean, std,
                       color=PALETTE.get(lbl, "grey"),
                       ls=LINE_STYLES.get(lbl, "-"),
                       label=lbl)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Day")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Ablation Study: Time-Series", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "ablation_timeseries.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> results/{path.name}")


# ──────────────────────────────────────────────────────────────
# Figure 5: Failure condition heatmap
# ──────────────────────────────────────────────────────────────

def plot_fc_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    fc_cols = [c for c in df.columns if c.startswith("cum_FC")]
    if not fc_cols:
        return

    labels = sorted(df["label"].unique())
    last = df.groupby(["label", "seed"]).last().reset_index()
    matrix = []
    for lbl in labels:
        sub = last[last["label"] == lbl]
        row = [sub[c].mean() for c in fc_cols]
        matrix.append(row)

    mat = np.array(matrix, dtype=float)
    fc_names = [c.replace("cum_FC_", "FC-") for c in fc_cols]

    fig, ax = plt.subplots(figsize=(max(8, len(fc_cols) * 1.0), max(4, len(labels) * 0.8)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(fc_names)))
    ax.set_xticklabels(fc_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    plt.colorbar(im, ax=ax, label="Cumulative count (mean over seeds)")

    for i in range(len(labels)):
        for j in range(len(fc_names)):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=7)

    ax.set_title("Failure Condition Cumulative Counts (mean over seeds)", fontsize=11)
    fig.tight_layout()
    path = out_dir / "fc_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> results/{path.name}")


# ──────────────────────────────────────────────────────────────
# Figure 6: Milestone achievement rate
# ──────────────────────────────────────────────────────────────

def plot_milestones(df: pd.DataFrame, out_dir: Path) -> None:
    m_cols = [f"M{i}" for i in range(1, 6)]
    labels = [l for l in ["A", "B", "C"] if l in df["label"].unique()]
    if not labels:
        return

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, mc in zip(axes, m_cols):
        for lbl in labels:
            if mc not in df.columns:
                continue
            days, mean, std = _agg(df, lbl, mc)
            rolling_mean = pd.Series(mean).rolling(7, min_periods=1).mean().values
            ax.plot(days, rolling_mean,
                    color=PALETTE.get(lbl, "grey"),
                    ls=LINE_STYLES.get(lbl, "-"),
                    lw=1.6, label=f"Scenario {lbl}")
        ax.set_title(f"Milestone {mc}\n(7-day rolling rate)", fontsize=9)
        ax.set_xlabel("Day", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Milestone Achievement Rates", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "milestones.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> results/{path.name}")


# ──────────────────────────────────────────────────────────────
# Figure 7: Virtue and Eudaimonia Dynamics 
# ──────────────────────────────────────────────────────────────

def plot_eudaimonia_timeseries(df: pd.DataFrame, out_dir: Path) -> None:
    labels = [l for l in ["A", "B", "C"] if l in df["label"].unique()]
    if not labels:
        return

    if "mean_virtue" not in df.columns or "mean_eudaimonia" not in df.columns:
        return

    metrics = [
        ("mean_virtue", "Mean Virtue (Accumulated Altruism)"),
        ("mean_eudaimonia", "Mean Eudaimonia (Sustainable Happiness)"),
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, metrics):
        for lbl in labels:
            days, mean, std = _agg(df, lbl, col)
            _fill_plot(ax, days, mean, std,
                       color=PALETTE.get(lbl, "grey"),
                       ls=LINE_STYLES.get(lbl, "-"),
                       label=f"Scenario {lbl}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Day")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Scenario A/B/C – Virtue & Eudaimonia Dynamics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "virtue_eudaimonia.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> results/{path.name}")


# ──────────────────────────────────────────────────────────────
# Figure 8: Modified Social Welfare Function Dynamics [追加]
# ──────────────────────────────────────────────────────────────

def plot_swf_timeseries(df: pd.DataFrame, out_dir: Path) -> None:
    labels = [l for l in ["A", "B", "C"] if l in df["label"].unique()]
    if not labels:
        return

    # Check if SWF columns exist
    if "social_welfare" not in df.columns:
        return

    metrics = [
        ("swf_utility", "Aggregate Utility ($U_i$)"),
        ("swf_eudaimonia", "Aggregate Eudaimonia ($E_i$)"),
        ("swf_penalty", "Exploitation Penalty (Fatigue > 0.90)"),
        ("social_welfare", "Modified Social Welfare ($W$)"),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, metrics):
        for lbl in labels:
            if col not in df.columns:
                continue
            days, mean, std = _agg(df, lbl, col)
            _fill_plot(ax, days, mean, std,
                       color=PALETTE.get(lbl, "grey"),
                       ls=LINE_STYLES.get(lbl, "-"),
                       label=f"Scenario {lbl}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Day")
        
        # Penalty is plotted inverted or just absolute, here we show absolute penalty amount
        if col == "swf_penalty":
            ax.set_ylabel("Penalty Magnitude (Lower is better)")
        
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Modified Social Welfare Function Dynamics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "swf_timeseries.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> results/{path.name}")


# ──────────────────────────────────────────────────────────────
# Master function
# ──────────────────────────────────────────────────────────────

def make_all_plots(df: pd.DataFrame, days: int = 100, n_seeds: int = 10) -> None:
    """Generate all plots from a combined DataFrame."""
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print("  No data to plot.")
        return

    print(f"\nGenerating plots (labels in data: {sorted(df['label'].unique())})")
    plot_abc_comparison(df, out_dir)
    plot_abc_extended(df, out_dir)
    plot_milestones(df, out_dir)
    plot_fc_heatmap(df, out_dir)
    plot_ablation_bar(df, out_dir)
    plot_ablation_timeseries(df, out_dir)
    plot_eudaimonia_timeseries(df, out_dir)
    plot_swf_timeseries(df, out_dir)
    print("All plots saved to results/")


# ──────────────────────────────────────────────────────────────
# CLI: python -m src.plots <combined_csv>
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.plots <combined_csv_path>")
        sys.exit(1)
    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)
    df_in = pd.read_csv(csv_path)
    make_all_plots(df_in)