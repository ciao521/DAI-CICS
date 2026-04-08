"""
app.py – FastAPI backend for DAI-CICS Web Demo

Endpoints:
  GET  /                          → serves static/index.html
  GET  /api/csv                   → combined_d100_s10.csv as JSON (grouped by scenario+day)
  GET  /api/context               → single CSV row as context dict
  POST /api/dialogue              → run 5-turn conference simulation (dry_run by default)
  POST /api/eval                  → run W&B judge evaluation steps
  GET  /api/scenarios             → available scenarios, seeds, day range

Run:
    uvicorn src.app:app --reload --port 8000
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
STATIC_DIR = ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="DAI-CICS Demo", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Static files
# ─────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ─────────────────────────────────────────────
# CSV loader (cached at startup)
# ─────────────────────────────────────────────
_CSV_CACHE: pd.DataFrame | None = None

def _get_csv() -> pd.DataFrame:
    global _CSV_CACHE
    if _CSV_CACHE is not None:
        return _CSV_CACHE
    csv_path = RESULTS_DIR / "combined_d100_s10.csv"
    if not csv_path.exists():
        raise HTTPException(404, "combined_d100_s10.csv not found. Run the experiment first.")
    df = pd.read_csv(csv_path)
    # Normalize: use 'label' as scenario if it exists and has richer info
    if "label" in df.columns:
        df["scenario"] = df["label"]
        df = df.drop(columns=["label"])

    # Merge ablation CSVs if they exist (daily_C-noN2_*.csv etc.)
    extra_dfs = []
    for ablation in ["C-noN2", "C-noN3", "C-onlyL1"]:
        # Try any matching file in results/
        matches = sorted(RESULTS_DIR.glob(f"daily_{ablation}_*.csv"))
        if matches:
            adf = pd.read_csv(matches[-1])  # use latest
            if "label" in adf.columns:
                adf["scenario"] = adf["label"]
                adf = adf.drop(columns=["label"])
            elif "scenario" not in adf.columns:
                adf["scenario"] = ablation
            extra_dfs.append(adf)

    if extra_dfs:
        df = pd.concat([df] + extra_dfs, ignore_index=True)

    _CSV_CACHE = df
    return df


# ─────────────────────────────────────────────
# API: list available scenarios / seeds
# ─────────────────────────────────────────────
@app.get("/api/scenarios")
def get_scenarios():
    df = _get_csv()
    scenarios = sorted(df["scenario"].unique().tolist())
    seeds = sorted(df["seed"].unique().tolist()) if "seed" in df.columns else [0]
    days = int(df["day"].max()) if "day" in df.columns else 100
    return {
        "scenarios": scenarios,
        "seeds": seeds,
        "days": days,
    }


# ─────────────────────────────────────────────
# API: full CSV as JSON (mean over seeds, grouped by scenario+day)
# ─────────────────────────────────────────────
@app.get("/api/csv")
def get_csv_data(scenario: str | None = None):
    df = _get_csv()
    if scenario:
        df = df[df["scenario"] == scenario]
    # Average numeric columns across seeds, group by scenario+day
    numeric_cols = df.select_dtypes("number").columns.tolist()
    # Exclude seed and groupby keys (day is a groupby key, not an aggregation target)
    non_seed = [c for c in numeric_cols if c not in ("seed", "day")]
    grouped = (
        df.groupby(["scenario", "day"])[non_seed]
        .mean()
        .reset_index()
    )
    # Return per-scenario timeseries
    result: dict[str, list] = {}
    for sc, grp in grouped.groupby("scenario"):
        rows = grp.drop(columns=["scenario"]).to_dict(orient="records")
        result[sc] = rows
    return JSONResponse(result)


# ─────────────────────────────────────────────
# API: single-row ABM context
# ─────────────────────────────────────────────
@app.get("/api/context")
def get_abm_context(scenario: str = "C", day: int = 45, seed: int = 0):
    df = _get_csv()
    mask = (df["scenario"] == scenario) & (df["day"] == day)
    if "seed" in df.columns:
        mask &= df["seed"] == seed
    rows = df[mask]
    if rows.empty:
        # Fallback: nearest day, any seed
        rows = df[df["scenario"] == scenario].sort_values(
            by="day", key=lambda s: (s - day).abs()
        ).head(1)
    if rows.empty:
        raise HTTPException(404, f"No data for scenario={scenario} day={day} seed={seed}")
    row = rows.iloc[0].to_dict()
    # Build a concise context dict
    ctx = {
        "abm_scenario": scenario,
        "abm_day": int(row.get("day", day)),
        "abm_seed": seed,
        "cm_fatigue": round(float(row.get("mean_fatigue", 0.5)), 3),
        "gini_fatigue": round(float(row.get("gini_fatigue", 0.03)), 3),
        "cum_acute_events": int(row.get("cum_acute_events", 0)),
        "mean_isolation": round(float(row.get("mean_isolation", 0.5)), 3),
        "mean_sdh_risk": round(float(row.get("mean_sdh_risk", 0.5)), 3),
        "burnout_count": int(row.get("burnout_count", 0)),
        "coordination_level": round(float(row.get("coordination_level", 0.4)), 3),
        "m1_today": int(row.get("M1", 0)),
        "m2_today": int(row.get("M2", 0)),
        "m3_today": int(row.get("M3", 0)),
        "m4_today": int(row.get("M4", 0)),
        "fc_a2_today": int(row.get("FC_A2", 0)),
        "fc_b1_today": int(row.get("FC_B1", 0)),
        "total_nudge_interventions": int(row.get("total_nudge_interventions", 0)),
    }
    return ctx


# ─────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────
class DialogueRequest(BaseModel):
    scenario: str = "C"
    day: int = 45
    seed: int = 0
    dry_run: bool = True
    # Optional manual overrides
    cm_fatigue: float | None = None
    gini_fatigue: float | None = None
    cum_acute_events: int | None = None
    mean_isolation: float | None = None

class EvalRequest(BaseModel):
    scenario: str = "A"
    day: int = 50
    seed: int = 0
    steps: int = 3
    dry_run: bool = True


# ─────────────────────────────────────────────
# API: run dialogue simulation
# ─────────────────────────────────────────────
@app.post("/api/dialogue")
def run_dialogue_endpoint(req: DialogueRequest):
    from src.dialogue_sim import run_dialogue_with_context

    # Get base context from CSV
    try:
        ctx: dict[str, Any] = get_abm_context(req.scenario, req.day, req.seed)
    except HTTPException:
        ctx = {
            "abm_scenario": req.scenario,
            "abm_day": req.day,
            "abm_seed": req.seed,
            "cm_fatigue": 0.50,
            "gini_fatigue": 0.03,
            "cum_acute_events": 0,
            "mean_isolation": 0.50,
            "mean_sdh_risk": 0.50,
            "burnout_count": 0,
            "coordination_level": 0.40,
        }

    # Manual overrides from request
    if req.cm_fatigue is not None:
        ctx["cm_fatigue"] = req.cm_fatigue
    if req.gini_fatigue is not None:
        ctx["gini_fatigue"] = req.gini_fatigue
    if req.cum_acute_events is not None:
        ctx["cum_acute_events"] = req.cum_acute_events
    if req.mean_isolation is not None:
        ctx["mean_isolation"] = req.mean_isolation

    turns = run_dialogue_with_context(ctx, dry_run=req.dry_run)
    return {"context": ctx, "turns": turns}


# ─────────────────────────────────────────────
# API: run W&B evaluation
# ─────────────────────────────────────────────
@app.post("/api/eval")
def run_eval_endpoint(req: EvalRequest):
    from src.wandb_eval import WandbDialogueEvaluator, _default_states
    from src.wandb_eval import states_from_abm_context

    # Get ABM context
    try:
        ctx: dict[str, Any] = get_abm_context(req.scenario, req.day, req.seed)
    except HTTPException:
        ctx = {"abm_scenario": req.scenario, "abm_day": req.day, "abm_seed": req.seed,
               "cm_fatigue": 0.50, "gini_fatigue": 0.03, "cum_acute_events": 0,
               "mean_isolation": 0.50, "coordination_level": 0.40}

    # Build states from ABM context
    states = states_from_abm_context(ctx, req.scenario, n=req.steps)

    if req.dry_run:
        # Return mock evaluation without calling LLM/W&B
        records = []
        for i, s in enumerate(states, 1):
            records.append({
                "step": i,
                "context": s["context"],
                "action": f"[DRY-RUN] シナリオ{req.scenario}のエージェントが状況を確認し、適切な対応を取る（Day {ctx['abm_day']}）",
                "faithfulness": round(0.70 + 0.05 * (3 - abs(i - 3)), 2),
                "context_relevance": round(0.65 + 0.08 * (3 - abs(i - 3)), 2),
                "reasoning": f"Day{ctx['abm_day']}時点のABMデータ（疲労={ctx['cm_fatigue']:.2f}, 孤立={ctx['mean_isolation']:.2f}）に基づいたシナリオ{req.scenario}の意思決定",
            })
        return {
            "context": ctx,
            "scenario": req.scenario,
            "steps": len(records),
            "dry_run": True,
            "records": records,
            "summary": {
                "mean_faithfulness": round(sum(r["faithfulness"] for r in records) / len(records), 3),
                "mean_context_relevance": round(sum(r["context_relevance"] for r in records) / len(records), 3),
            }
        }

    # Live evaluation
    try:
        evaluator = WandbDialogueEvaluator(scenario=req.scenario)
        evaluator.run_evaluation_loop(states)
        evaluator.finish()
        records = evaluator._eval_records
    except Exception as e:
        raise HTTPException(500, f"Evaluation failed: {str(e)}")

    mean_ff = sum(r.get("faithfulness") or 0.5 for r in records) / max(len(records), 1)
    mean_cr = sum(r.get("context_relevance") or 0.5 for r in records) / max(len(records), 1)
    return {
        "context": ctx,
        "scenario": req.scenario,
        "steps": len(records),
        "dry_run": False,
        "records": records,
        "summary": {
            "mean_faithfulness": round(mean_ff, 3),
            "mean_context_relevance": round(mean_cr, 3),
        }
    }
