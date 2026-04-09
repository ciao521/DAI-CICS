"""
app.py – FastAPI backend for DAI-CICS Web Demo

Endpoints:
  GET  /              → static/index.html
  GET  /api/scenarios → available scenarios, day range
  GET  /api/context   → ABM context dict for given scenario+day
  POST /api/run       → NDJSON stream: dialogue (LLM) + W&B eval (LLM)

Run:
    open http://127.0.0.1:8000 && uvicorn src.app:app --port 8000
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from threading import Thread
from typing import Any, AsyncGenerator

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
STATIC_DIR = ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="DAI-CICS Demo", version="3.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ─── CSV loader (module-level cache) ──────────────────────────
_CSV_CACHE: pd.DataFrame | None = None


def _get_csv() -> pd.DataFrame:
    global _CSV_CACHE
    if _CSV_CACHE is not None:
        return _CSV_CACHE
    csv_path = RESULTS_DIR / "combined_d100_s10.csv"
    if not csv_path.exists():
        raise HTTPException(404, "combined_d100_s10.csv not found — run experiment first.")
    df = pd.read_csv(csv_path)
    if "label" in df.columns:
        df["scenario"] = df["label"]
        df = df.drop(columns=["label"])
    extra = []
    for sc in ["C-noN2", "C-noN3", "C-onlyL1"]:
        for f in sorted(RESULTS_DIR.glob(f"daily_{sc}_*.csv")):
            adf = pd.read_csv(f)
            if "label" in adf.columns:
                adf["scenario"] = adf["label"]
                adf = adf.drop(columns=["label"])
            elif "scenario" not in adf.columns:
                adf["scenario"] = sc
            extra.append(adf)
            break
    if extra:
        df = pd.concat([df] + extra, ignore_index=True)
    _CSV_CACHE = df
    return df


# ─── GET /api/scenarios ───────────────────────────────────────
@app.get("/api/scenarios")
def get_scenarios():
    df = _get_csv()
    return {
        "scenarios": sorted(df["scenario"].unique().tolist()),
        "days": int(df["day"].max()),
    }


# ─── GET /api/context ─────────────────────────────────────────
@app.get("/api/context")
def get_abm_context(scenario: str = "C", day: int = 45) -> dict:
    df = _get_csv()
    rows = df[(df["scenario"] == scenario) & (df["day"] == day)]
    if rows.empty:
        rows = df[df["scenario"] == scenario].copy()
        rows["_dist"] = (rows["day"] - day).abs()
        rows = rows.sort_values("_dist").head(10)
    if rows.empty:
        raise HTTPException(404, f"No data for {scenario} day={day}")
    num = rows.select_dtypes("number").mean()
    return {
        "abm_scenario": scenario,
        "abm_day": int(rows["day"].iloc[0]),
        "cm_fatigue":           round(float(num.get("mean_fatigue", 0.5)), 3),
        "gini_fatigue":         round(float(num.get("gini_fatigue", 0.03)), 4),
        "cum_acute_events":     round(float(num.get("cum_acute_events", 0))),
        "mean_isolation":       round(float(num.get("mean_isolation", 0.5)), 3),
        "mean_sdh_risk":        round(float(num.get("mean_sdh_risk", 0.5)), 3),
        "burnout_count":        round(float(num.get("burnout_count", 0))),
        "coordination_level":   round(float(num.get("coordination_level", 0.4)), 3),
        "m1": round(float(num.get("M1", 0)), 2),
        "m2": round(float(num.get("M2", 0)), 2),
        "m3": round(float(num.get("M3", 0)), 2),
        "m4": round(float(num.get("M4", 0)), 2),
        "fc_a1": round(float(num.get("cum_FC_A1", 0))),
        "fc_a2": round(float(num.get("cum_FC_A2", 0))),
        "fc_b1": round(float(num.get("cum_FC_B1", 0))),
        "fc_b2": round(float(num.get("cum_FC_B2", 0))),
        "fc_c1": round(float(num.get("cum_FC_C1", 0))),
        "total_nudge_interventions": round(float(num.get("total_nudge_interventions", 0))),
    }


# ─── Request model ────────────────────────────────────────────
class RunRequest(BaseModel):
    scenario: str = "C"
    day: int = 45
    cm_fatigue: float | None = None
    gini_fatigue: float | None = None
    cum_acute_events: int | None = None
    mean_isolation: float | None = None


# ─── NDJSON helper ────────────────────────────────────────────
def _ndjson(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


# ─── POST /api/run (NDJSON streaming) ────────────────────────
@app.post("/api/run")
async def run_simulation(req: RunRequest):
    """
    Streams NDJSON events:
      {"type":"log",     "msg":"..."}
      {"type":"context", "data":{...}}
      {"type":"turn",    "data":{...}}
      {"type":"eval",    "data":{...}}
      {"type":"done"}
      {"type":"error",   "msg":"..."}
    """
    from src.dialogue_sim import run_dialogue_with_context
    from src.wandb_eval import WandbDialogueEvaluator, states_from_abm_context

    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def push(event: dict) -> None:
        asyncio.run_coroutine_threadsafe(q.put(_ndjson(event)), loop)

    def worker() -> None:
        try:
            # ① ABM context
            push({"type": "log", "msg": "① ABMメトリクス取得中…"})
            ctx: dict[str, Any] = get_abm_context(req.scenario, req.day)
            if req.cm_fatigue     is not None: ctx["cm_fatigue"]       = req.cm_fatigue
            if req.gini_fatigue   is not None: ctx["gini_fatigue"]     = req.gini_fatigue
            if req.cum_acute_events is not None: ctx["cum_acute_events"] = req.cum_acute_events
            if req.mean_isolation is not None: ctx["mean_isolation"]   = req.mean_isolation
            push({"type": "context", "data": ctx})
            push({"type": "log", "msg": "✅ ABMメトリクス取得完了"})

            # ② Dialogue
            push({"type": "log", "msg": "② 5ターン多職種カンファレンス（LLM）開始…"})

            def turn_cb(turn_data: dict) -> None:
                agent = turn_data.get("agent", "?")
                t     = turn_data.get("turn",  "?")
                push({"type": "log", "msg": f"  Turn {t} [{agent}] 完了"})
                push({"type": "turn", "data": turn_data})

            turns = run_dialogue_with_context(ctx, turn_callback=turn_cb)
            push({"type": "log", "msg": f"✅ カンファレンス完了 ({len(turns)} ターン)"})

            # ③ Eval
            base_sc = req.scenario.split("-")[0]
            if base_sc not in ("A", "B", "C"):
                base_sc = "C"
            push({"type": "log", "msg": "③ LLM Judge 評価（5ステップ）実行中…"})
            states = states_from_abm_context(ctx, base_sc, n=5)
            try:
                evaluator = WandbDialogueEvaluator(scenario=base_sc)
                evaluator.run_evaluation_loop(states)
                evaluator.finish()
                records = evaluator._eval_records
                ff = sum(r.get("faithfulness")    or 0.5 for r in records) / max(len(records), 1)
                cr = sum(r.get("context_relevance") or 0.5 for r in records) / max(len(records), 1)
                eval_data: dict[str, Any] = {
                    "records":  records,
                    "summary":  {
                        "mean_faithfulness":     round(ff, 3),
                        "mean_context_relevance": round(cr, 3),
                    },
                }
            except Exception as e:
                eval_data = {"error": str(e), "records": [], "summary": {}}

            push({"type": "eval",  "data": eval_data})
            push({"type": "log",   "msg": "✅ 評価完了"})
            push({"type": "done"})

        except Exception as e:
            push({"type": "error", "msg": str(e)})

    Thread(target=worker, daemon=True).start()

    async def generate() -> AsyncGenerator[str, None]:
        while True:
            line = await q.get()
            yield line
            evt = json.loads(line)
            if evt.get("type") in ("done", "error"):
                break

    return StreamingResponse(generate(), media_type="application/x-ndjson")
