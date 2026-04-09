"""
FastAPI server: JSON APIs for the React SPA + static file serving in production.
Run from repository root: uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from src.agent import create_agent
from src.anomaly import detect_anomalies
from src.data_loader import get_well_list, get_well_summary, load_daily_production

# --- App state (loaded at startup) ---
_state: dict[str, Any] = {}


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe records (dates ISO, NaN -> null)."""
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _filter_daily(
    daily: pd.DataFrame,
    well: Optional[str],
    start: Optional[date],
    end: Optional[date],
) -> pd.DataFrame:
    df = daily.copy()
    if start is not None:
        df = df[df["DATEPRD"].dt.date >= start]
    if end is not None:
        df = df[df["DATEPRD"].dt.date <= end]
    if well and well != "All Wells":
        df = df[df["WELL_NAME"] == well]
    return df


CHAT_HISTORY_MAX = 24


def _history_to_lc(messages: list[dict]) -> list:
    lc = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if not content or not role:
            continue
        if role == "user":
            lc.append(HumanMessage(content=content))
        elif role == "assistant":
            lc.append(AIMessage(content=content))
    if len(lc) > CHAT_HISTORY_MAX:
        lc = lc[-CHAT_HISTORY_MAX:]
    return lc


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        daily = load_daily_production()
        wells = get_well_list(daily)
        agent = create_agent(daily)
        _state["daily"] = daily
        _state["wells"] = wells
        _state["agent"] = agent
        _state["error"] = None
    except Exception as e:
        _state["daily"] = None
        _state["wells"] = []
        _state["agent"] = None
        _state["error"] = str(e)
    yield
    _state.clear()


app = FastAPI(title="Geo-Agentic RAG API", lifespan=lifespan)

_cors_origins = os.getenv("CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    ok = _state.get("daily") is not None
    return {"ok": ok, "error": _state.get("error")}


@app.get("/api/meta")
def meta():
    daily = _state.get("daily")
    if daily is None:
        raise HTTPException(503, detail=_state.get("error") or "Data not loaded")
    wells: list[str] = _state.get("wells") or []
    min_d = daily["DATEPRD"].min().date().isoformat()
    max_d = daily["DATEPRD"].max().date().isoformat()
    return {
        "wells": wells,
        "date_min": min_d,
        "date_max": max_d,
        "total_wells": len(wells),
        "production_rows": len(daily),
        "total_oil_sm3": float(daily["BORE_OIL_VOL"].sum()),
    }


@app.get("/api/production/summary")
def production_summary(
    well: Optional[str] = Query(None, description="Filter to one well, or omit for all"),
    start: Optional[date] = None,
    end: Optional[date] = None,
):
    daily = _state.get("daily")
    if daily is None:
        raise HTTPException(503, detail=_state.get("error") or "Data not loaded")
    df = _filter_daily(daily, well if well != "All Wells" else None, start, end)
    if df.empty:
        return {"rows": []}
    summary = get_well_summary(df)
    return {"rows": _df_to_records(summary)}


@app.get("/api/production/field-oil-by-well")
def field_oil_by_well(
    start: Optional[date] = None,
    end: Optional[date] = None,
):
    """Stacked / multi-series daily oil by well (Sm3)."""
    daily = _state.get("daily")
    if daily is None:
        raise HTTPException(503, detail=_state.get("error") or "Data not loaded")
    df = _filter_daily(daily, None, start, end)
    g = (
        df.groupby(["DATEPRD", "WELL_NAME"], as_index=False)["BORE_OIL_VOL"]
        .sum()
        .sort_values("DATEPRD")
    )
    g["DATEPRD"] = g["DATEPRD"].dt.strftime("%Y-%m-%d")
    return {"rows": g.to_dict(orient="records")}


@app.get("/api/production/well-detail")
def well_detail(
    well: str = Query(..., description="Exact WELL_NAME"),
    start: Optional[date] = None,
    end: Optional[date] = None,
):
    daily = _state.get("daily")
    if daily is None:
        raise HTTPException(503, detail=_state.get("error") or "Data not loaded")
    df = _filter_daily(daily, well, start, end)
    if df.empty:
        return {"rows": [], "metrics": {}}
    cols = [
        "DATEPRD",
        "WELL_NAME",
        "BORE_OIL_VOL",
        "BORE_WAT_VOL",
        "WATER_CUT_PCT",
        "AVG_WHP_P",
        "BORE_GAS_VOL",
    ]
    sub = df[[c for c in cols if c in df.columns]].copy()
    sub["DATEPRD"] = sub["DATEPRD"].dt.strftime("%Y-%m-%d")
    whp = sub["AVG_WHP_P"].replace(0, np.nan).dropna() if "AVG_WHP_P" in sub.columns else pd.Series(dtype=float)
    metrics = {
        "total_oil_sm3": float(sub["BORE_OIL_VOL"].sum()),
        "avg_water_cut_pct": float(sub["WATER_CUT_PCT"].mean()) if "WATER_CUT_PCT" in sub.columns else 0.0,
        "production_days": int((sub["BORE_OIL_VOL"] > 0).sum()) if "BORE_OIL_VOL" in sub.columns else 0,
        "avg_whp": float(whp.mean()) if not whp.empty else None,
    }
    return {"rows": _df_to_records(sub), "metrics": metrics}


@app.get("/api/anomalies")
def anomalies(well: Optional[str] = Query(None, description="Substring or exact; omit for all wells")):
    daily = _state.get("daily")
    if daily is None:
        raise HTTPException(503, detail=_state.get("error") or "Data not loaded")
    well_filter = None if not well or well == "All Wells" else well
    adf = detect_anomalies(daily, well_filter)
    if adf.empty:
        return {"rows": [], "counts": {"Critical": 0, "High": 0, "Medium": 0}}
    adf = adf.copy()
    if "DATEPRD" in adf.columns:
        adf["DATEPRD"] = adf["DATEPRD"].dt.strftime("%Y-%m-%d")
    counts = (
        adf["SEVERITY"].value_counts().to_dict()
        if "SEVERITY" in adf.columns
        else {}
    )
    return {"rows": _df_to_records(adf), "counts": counts}


class ChatMessageIn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatMessageIn] = Field(default_factory=list)


@app.post("/api/chat")
def chat(body: ChatRequest):
    agent = _state.get("agent")
    if agent is None:
        raise HTTPException(503, detail=_state.get("error") or "Agent not available")
    prior = [m.model_dump() for m in body.history]
    lc_history = _history_to_lc(prior)
    try:
        result = agent.invoke(
            {
                "input": body.message,
                "chat_history": lc_history,
            }
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e)) from e
    response = result.get("output", "")
    sources = []
    for step in result.get("intermediate_steps", []) or []:
        if len(step) < 2:
            continue
        action, observation = step[0], step[1]
        tool = getattr(action, "tool", None)
        if tool == "search_well_documents" and "Source" in str(observation):
            sources.append(
                {
                    "doc_type": "Well Document",
                    "excerpt": str(observation)[:400],
                }
            )
    return {"response": response, "sources": sources}


# --- Static SPA (production): serve Vite build after API routes ---
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DIST = _REPO_ROOT / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="spa")
