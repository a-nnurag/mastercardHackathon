"""
Task 11 dashboard — FastAPI app. Per plan.md's definition of done, the
startup event precomputes everything that doesn't need a live LLM call
(signals for all 226 sessions, Task 5's CV metrics, Task 8's GNN
held-out metrics) exactly once, so page loads are instant; the one
genuinely slow path (an LLM verdict) stays on-demand per session.

Run: python3 -m uvicorn dashboard.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from dashboard.data_access import (
    build_session_index,
    compute_attack1_metrics,
    compute_attack2_metrics,
    load_mutation_rounds,
)
from defend.llm_verdict import verdict_for_session

app = FastAPI(title="Mastercard Fraud Defense Dashboard")

_state = {}


@app.on_event("startup")
def _startup():
    print("Precomputing session signals (rules, constraint drift, content, GNN)...")
    rows, lgb_lookup, gnn_lookup = build_session_index()
    _state["sessions"] = rows
    _state["lgb_lookup"] = lgb_lookup
    _state["gnn_lookup"] = gnn_lookup
    print(f"  {len(rows)} sessions ready.")

    print("Running Task 5 cross-validated metrics (attack #1: prompt injection)...")
    _state["attack1_metrics"] = compute_attack1_metrics()

    print("Running Task 8 GNN held-out evaluation (attack #2: merchant laundering)...")
    _state["attack2_metrics"] = compute_attack2_metrics()

    _state["mutation_rounds"] = load_mutation_rounds()
    print("Dashboard ready.")


@app.get("/api/sessions")
def get_sessions():
    return _state["sessions"]


@app.get("/api/sessions/{agent_id}")
def get_session(agent_id: str):
    for row in _state["sessions"]:
        if row["agent_id"] == agent_id:
            return row
    raise HTTPException(status_code=404, detail="session not found")


@app.post("/api/sessions/{agent_id}/verdict")
def get_verdict(agent_id: str):
    if not any(row["agent_id"] == agent_id for row in _state["sessions"]):
        raise HTTPException(status_code=404, detail="session not found")
    try:
        return verdict_for_session(
            agent_id, lgb_lookup=_state["lgb_lookup"], gnn_lookup=_state["gnn_lookup"]
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"verdict synthesis failed: {type(e).__name__}: {e}")


@app.get("/api/metrics")
def get_metrics():
    return {
        "attack1_prompt_injection": _state["attack1_metrics"],
        "attack2_merchant_laundering": _state["attack2_metrics"],
        "mutation_rounds": _state["mutation_rounds"],
    }


app.mount("/", StaticFiles(directory="dashboard/static", html=True), name="static")
