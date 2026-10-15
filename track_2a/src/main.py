"""LangGraph entry point served by Aegra (graph id: "agent", port 2026).

Each run processes one candidate message. The interview session (plan,
assessments, call counter, ...) is kept in the graph state, so Aegra's
Postgres checkpointer persists it per thread.

Optional input fields (e.g. from the judges' benchmark harness or the
Agent Chat UI) let a caller pin the setting of a session:
    {"messages": [...], "lang": "fr", "posting_id": "P-04",
     "candidate_id": "C-04", "scenario_id": "S-04", "feedback_mode": "per_answer"}
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any, Optional

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smartstart import coach  # noqa: E402
from smartstart.data import get_dataset  # noqa: E402

RESTART = {"/neu", "/new", "/nouveau", "/nuovo", "/restart"}


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    session: Optional[dict[str, Any]]
    lang: Optional[str]
    dialect: Optional[str]
    posting_id: Optional[str]
    candidate_id: Optional[str]
    scenario_id: Optional[str]
    feedback_mode: Optional[str]
    calls_per_answer: Optional[float]


def _last_user_text(state: State) -> str:
    for m in reversed(state.get("messages") or []):
        role = getattr(m, "type", None) or (m.get("role") if isinstance(m, dict) else None)
        if role in ("human", "user"):
            c = getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
            if isinstance(c, list):  # multimodal content blocks
                c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
            return str(c or "")
    return ""


def init_session(state: State) -> dict:
    text = _last_user_text(state)
    s = coach.new_session(
        text,
        lang=state.get("lang"),
        dialect=state.get("dialect"),
        posting_id=state.get("posting_id"),
        candidate_id=state.get("candidate_id"),
        scenario_id=state.get("scenario_id"),
        feedback_mode=state.get("feedback_mode"),
        ds=get_dataset(),
    )
    return {"session": s}


def coach_turn(state: State) -> dict:
    s = dict(state["session"])
    reply = coach.step(s, _last_user_text(state), ds=get_dataset())
    return {
        "session": s,
        "messages": [AIMessage(content=reply)],
        "calls_per_answer": coach.calls_per_answer(s),
    }


def route(state: State) -> str:
    s = state.get("session")
    text = _last_user_text(state).strip().lower()
    if not s or text in RESTART:
        return "init_session"
    return "coach_turn"


def create_graph():
    g = StateGraph(State)
    g.add_node("init_session", init_session)
    g.add_node("coach_turn", coach_turn)
    g.add_conditional_edges(START, route, {"init_session": "init_session", "coach_turn": "coach_turn"})
    g.add_edge("init_session", "coach_turn")
    g.add_edge("coach_turn", END)
    return g.compile()


graph = create_graph()
