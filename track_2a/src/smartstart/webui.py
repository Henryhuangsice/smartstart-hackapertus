"""SmartStart web app — a small, dependency-free practice UI (Python standard library only).

    LLM_BASE_URL=http://localhost:8080/v1 python -m smartstart.webui      # → http://localhost:7860

The UI talks to the same coach logic as the LangGraph/Aegra API, and additionally shows the
coach's internal assessment of every answer (rubric levels, evidence, next action) and a
"practice again" loop that re-assesses an improved answer.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import coach
from .data import get_dataset
from .planner import STAGES, stage_of

HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
SESSIONS: dict[str, dict] = {}
LOCKS: dict[str, threading.Lock] = {}


def _public_state(s: dict) -> dict:
    ds = get_dataset()
    log = []
    for i, e in enumerate(s["log"]):
        a = e["assessment"]
        log.append({
            "index": i, "question": e["question"], "answer": e["answer"], "followup": e.get("is_followup", False),
            "level": a["quality_level"], "action": a["coach_action"], "flags": a["problem_flags"],
            "evidence": a["evidence"], "strength": a["strength"], "improvement": a["improvement"],
            "scores": {ds.criterion_label(k, s["lang"]): v for k, v in (a.get("scores") or {}).items() if v},
            "exemplars": a.get("exemplars", []),
        })
    idx = min(s["idx"], len(s["plan"]) - 1)
    stage = "feedback" if s["finished"] else stage_of(ds, s["plan"][idx])
    p = ds.postings[s["posting_id"]]
    return {
        "lang": s["lang"], "posting": ds.occupation_name(p, s["lang"]),
        "company": (p.get("company") or {}).get("name"), "stage": stage, "stages": STAGES + ["feedback"],
        "question_no": min(s["idx"] + 1, len(s["plan"])), "questions": len(s["plan"]),
        "finished": s["finished"], "llm_calls": s["llm_calls"], "answers": s["answers"],
        "calls_per_answer": coach.calls_per_answer(s), "log": log,
        "violations": s.get("feedback_violations", []),
    }


def _postings() -> list[dict]:
    ds = get_dataset()
    out = []
    for pid, p in ds.postings.items():
        out.append({"id": pid, "lang": p.get("lang"), "occupation": ds.occupation_name(p, p.get("lang", "de")),
                    "company": (p.get("company") or {}).get("name"), "place": (p.get("company") or {}).get("place")})
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body, ctype: str = "application/json") -> None:
        data = body.encode("utf-8") if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:  # quiet
        pass

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            return self._send(200, HTML, "text/html")
        if self.path == "/api/postings":
            return self._send(200, _postings())
        if self.path == "/health":
            return self._send(200, {"ok": True})
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            ds = get_dataset()
            if self.path == "/api/start":
                first = req.get("text") or "Grüezi"
                s = coach.new_session(first, lang=req.get("lang") or None, posting_id=req.get("posting_id") or None,
                                      feedback_mode=req.get("feedback_mode") or None, ds=ds)
                sid = uuid.uuid4().hex
                SESSIONS[sid], LOCKS[sid] = s, threading.Lock()
                with LOCKS[sid]:
                    reply = coach.step(s, first, ds=ds)
                return self._send(200, {"sid": sid, "reply": reply, "state": _public_state(s)})
            sid = req.get("sid")
            if sid not in SESSIONS:
                return self._send(404, {"error": "session expired - please start again"})
            s = SESSIONS[sid]
            with LOCKS[sid]:
                if self.path == "/api/turn":
                    reply = coach.step(s, (req.get("text") or "").strip(), ds=ds)
                    return self._send(200, {"reply": reply, "state": _public_state(s)})
                if self.path == "/api/revise":
                    res = coach.revise_answer(s, int(req["index"]), req["text"], ds=ds)
                    return self._send(200, {"revision": res, "state": _public_state(s)})
            self._send(404, {"error": "not found"})
        except Exception as e:  # surface LLM/endpoint errors to the UI
            self._send(500, {"error": str(e)})


def main() -> None:
    port = int(os.getenv("SMARTSTART_UI_PORT", "7860"))
    print(f"SmartStart UI on http://localhost:{port}  (LLM: {os.getenv('LLM_BASE_URL', 'default')})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
