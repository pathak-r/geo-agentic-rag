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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
    # Producer wells only (WELL_TYPE == OP and has actual oil production)
    producer_wells = sorted(
        daily[
            daily["WELL_TYPE"].eq("OP") & daily["BORE_OIL_VOL"].gt(0)
        ]["WELL_NAME"].unique().tolist()
    )
    return {
        "wells": wells,
        "producer_wells": producer_wells,
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


@app.get("/api/comparison")
def comparison(
    well_a: str = Query(..., description="First well name"),
    well_b: str = Query(..., description="Second well name"),
    start: Optional[date] = None,
    end: Optional[date] = None,
):
    """Side-by-side well comparison: normalised production profiles, decline rates, divergence flags."""
    daily = _state.get("daily")
    if daily is None:
        raise HTTPException(503, detail=_state.get("error") or "Data not loaded")

    def well_series(name: str) -> pd.DataFrame | None:
        df = _filter_daily(daily, name, start, end)
        # Only producing days
        df = df[df["BORE_OIL_VOL"].notna() & (df["BORE_OIL_VOL"] > 0)].copy()
        if df.empty:
            return None
        df = df.sort_values("DATEPRD").reset_index(drop=True)
        first = df["DATEPRD"].iloc[0]
        df["day"] = (df["DATEPRD"] - first).dt.days
        return df

    def well_metrics(df: pd.DataFrame | None) -> dict:
        if df is None:
            return {}
        whp = df["AVG_WHP_P"].replace(0, np.nan).dropna() if "AVG_WHP_P" in df.columns else pd.Series(dtype=float)
        return {
            "total_oil_sm3": float(df["BORE_OIL_VOL"].sum()),
            "production_days": int(len(df)),
            "avg_water_cut_pct": float(df["WATER_CUT_PCT"].mean()) if "WATER_CUT_PCT" in df.columns else 0.0,
            "avg_whp": float(whp.mean()) if not whp.empty else None,
        }

    def exponential_decline(df: pd.DataFrame | None) -> dict | None:
        """Fit q(t) = qi * e^(-D*t) via log-linear regression."""
        if df is None or len(df) < 10:
            return None
        t = df["day"].values.astype(float)
        q = df["BORE_OIL_VOL"].values.astype(float)
        mask = q > 0
        if mask.sum() < 10:
            return None
        log_q = np.log(q[mask])
        coeffs = np.polyfit(t[mask], log_q, 1)
        D = max(-coeffs[0], 0.0)   # decline rate per day (force non-negative)
        qi = float(np.exp(coeffs[1]))
        # Generate smooth trend curve every 30 days
        t_trend = np.arange(0, int(t.max()) + 1, 30)
        q_trend = qi * np.exp(-D * t_trend)
        trend = [{"day": int(td), "q_trend": round(float(qd), 2)} for td, qd in zip(t_trend, q_trend)]
        return {
            "D_per_day": round(D, 6),
            "D_annual_pct": round(D * 365 * 100, 1),
            "qi": round(qi, 1),
            "trend": trend,
        }

    def build_series(df: pd.DataFrame | None) -> list[dict]:
        if df is None:
            return []
        rows = []
        for _, r in df.iterrows():
            whp = r.get("AVG_WHP_P")
            rows.append({
                "day": int(r["day"]),
                "date": str(r["DATEPRD"])[:10],
                "oil": round(float(r["BORE_OIL_VOL"]), 2),
                "wc": round(float(r.get("WATER_CUT_PCT") or 0), 2),
                "whp": round(float(whp), 2) if pd.notna(whp) and float(whp) > 0 else None,
            })
        return rows

    def detect_divergence(df_a: pd.DataFrame | None, df_b: pd.DataFrame | None) -> list[dict]:
        """Flag 30-day bins where oil production or water cut diverge > 1.5 std."""
        if df_a is None or df_b is None:
            return []
        max_day = max(df_a["day"].max(), df_b["day"].max())
        bins = range(0, int(max_day) + 30, 30)
        deltas_oil, deltas_wc = [], []
        records = []
        for b in bins:
            a_bin = df_a[(df_a["day"] >= b) & (df_a["day"] < b + 30)]
            b_bin = df_b[(df_b["day"] >= b) & (df_b["day"] < b + 30)]
            if a_bin.empty or b_bin.empty:
                continue
            d_oil = float(a_bin["BORE_OIL_VOL"].mean()) - float(b_bin["BORE_OIL_VOL"].mean())
            d_wc = float(a_bin["WATER_CUT_PCT"].mean()) - float(b_bin["WATER_CUT_PCT"].mean()) if "WATER_CUT_PCT" in df_a.columns else 0.0
            deltas_oil.append(d_oil)
            deltas_wc.append(d_wc)
            records.append({"day_start": b, "day_end": b + 30, "d_oil": round(d_oil, 1), "d_wc": round(d_wc, 2)})
        if not records:
            return []
        std_oil = float(np.std(deltas_oil)) if deltas_oil else 1.0
        std_wc = float(np.std(deltas_wc)) if deltas_wc else 1.0
        flags = []
        for rec in records:
            flagged = []
            if std_oil > 0 and abs(rec["d_oil"]) > 1.5 * std_oil:
                flagged.append("oil")
            if std_wc > 0 and abs(rec["d_wc"]) > 1.5 * std_wc:
                flagged.append("wc")
            if flagged:
                flags.append({**rec, "metrics": flagged})
        return flags

    df_a = well_series(well_a)
    df_b = well_series(well_b)

    return {
        "well_a": {
            "name": well_a,
            "series": build_series(df_a),
            "metrics": well_metrics(df_a),
            "decline": exponential_decline(df_a),
        },
        "well_b": {
            "name": well_b,
            "series": build_series(df_b),
            "metrics": well_metrics(df_b),
            "decline": exponential_decline(df_b),
        },
        "divergence": detect_divergence(df_a, df_b),
    }


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
    messages = lc_history + [HumanMessage(content=body.message)]
    try:
        result = agent.invoke(
            {"messages": messages},
            {"recursion_limit": 10},
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e)) from e
    # langgraph returns {"messages": [...]} — last message is the AI response
    last = result.get("messages", [])
    response = last[-1].content if last else ""
    # Extract document sources from tool messages if present
    sources = []
    for msg in result.get("messages", []):
        if hasattr(msg, "name") and msg.name == "search_well_documents":
            content = str(msg.content)
            if "Source" in content or len(content) > 50:
                sources.append({"doc_type": "Well Document", "excerpt": content[:400]})
    return {"response": response, "sources": sources}


# --- Static SPA (production): serve Vite build after API routes ---
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DIST = _REPO_ROOT / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="spa")
