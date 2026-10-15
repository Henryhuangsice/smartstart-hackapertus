"""Experiment 2 — end-to-end simulated interviews for the 28 FHGR scenarios.

A candidate simulator (Apertus, prompted with the FHGR candidate profile and
`simulation_persona`) talks to the coach until the final feedback. For every
scenario we store the transcript, the coach's internal assessments, the number
of LLM calls per candidate answer and the guideline-checker result.

    python eval/simulate.py --out eval/results/sim               # all 28 scenarios
    python eval/simulate.py --scenarios S-01 S-04 S-05 --out ...  # subset
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from _common import ROOT, write_json

from smartstart import coach, llm
from smartstart.data import get_dataset
from smartstart.prompts import LANG_NAME

CAND_SYSTEM = """You role-play {name}, {age} years old, from {place}, in a job interview for an apprenticeship.
Stay in character. Answer ONLY as the candidate, 1-4 sentences, like a real teenager would talk
(not polished, sometimes short). Language: {lang_name}.{dialect_note}
Profile (use these facts, do not invent big new facts):
{profile}
Persona / behaviour to simulate: {persona}
When asked whether you have questions, ask one realistic question about the apprenticeship."""


def profile_text(c: dict) -> str:
    keep = ["school", "hobbies", "experiences", "career_choice_story", "strengths", "development_areas",
            "future_plans", "interview_experience"]
    return json.dumps({k: c.get(k) for k in keep if c.get(k)}, ensure_ascii=False)[:2500]


def run_scenario(ds, sid: str, max_turns: int = 30) -> dict:
    scen = ds.scenarios[sid]
    cand = ds.candidates[scen["candidate_id"]]
    lang = scen["lang"]
    dialect = scen.get("dialect_expected") if lang == "de" else None
    sys_prompt = CAND_SYSTEM.format(
        name=cand["first_name"], age=cand.get("age", 15), place=cand.get("place", ""),
        lang_name=LANG_NAME[lang],
        dialect_note=(f" Write in Swiss German dialect ({dialect}) where it feels natural." if dialect else ""),
        profile=profile_text(cand), persona=json.dumps(cand.get("simulation_persona"), ensure_ascii=False),
    )
    greet = {"de": "Grüezi, ich bin {n}.", "fr": "Bonjour, je suis {n}.", "it": "Buongiorno, sono {n}."}[lang]
    first = greet.format(n=cand["first_name"])
    s = coach.new_session(first, scenario_id=sid, ds=ds)
    transcript = [{"role": "candidate", "text": first}]
    reply = coach.step(s, first, ds=ds)
    transcript.append({"role": "coach", "text": reply})
    # Only system + one user message per call (the Apertus 1.5 chat template on vLLM
    # rejects replayed multi-turn assistant messages): the dialogue history is passed as text.
    history: list[str] = [f"INTERVIEWER: {reply}"]
    turns = 0
    t0 = time.time()
    while not s["finished"] and turns < max_turns:
        user_msg = ("Conversation so far:\n" + "\n".join(history[-10:]) +
                    "\n\nWrite ONLY your next answer as the candidate (1-4 sentences).")
        ans = llm.chat([{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}],
                       temperature=0.8, max_tokens=160, tag="candidate").strip().strip('"')
        for pre in ("CANDIDATE:", "Candidate:", "KANDIDAT:", cand["first_name"] + ":"):
            if ans.startswith(pre):
                ans = ans[len(pre):].strip()
        transcript.append({"role": "candidate", "text": ans})
        reply = coach.step(s, ans, ds=ds)
        transcript.append({"role": "coach", "text": reply})
        history += [f"CANDIDATE: {ans}", f"INTERVIEWER: {reply}"]
        turns += 1
    return {
        "scenario_id": sid,
        "lang": lang,
        "dialect": dialect,
        "difficulty": scen.get("difficulty"),
        "situation": scen.get("situation"),
        "success_criteria": scen.get("success_criteria_for_coach") or [],
        "transcript": transcript,
        "assessments": [{"qid": e["qid"], "level": e["assessment"]["quality_level"],
                         "action": e["assessment"]["coach_action"], "flags": e["assessment"]["problem_flags"]}
                        for e in s["log"]],
        "llm_calls": s["llm_calls"],
        "answers": s["answers"],
        "calls_per_answer": coach.calls_per_answer(s),
        "followups": s["followups_total"],
        "feedback_violations_first": s.get("feedback_violations_first", []),
        "feedback_violations_final": s.get("feedback_violations", []),
        "finished": s["finished"],
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="eval/results/sim")
    args = ap.parse_args()
    ds = get_dataset()
    ids = args.scenarios or sorted(ds.scenarios)
    if args.limit:
        ids = ids[: args.limit]
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    for sid in ids:
        f = out / f"{sid}.json"
        if f.exists():
            print(f"{sid}: exists, skipping")
            continue
        try:
            r = run_scenario(ds, sid)
        except Exception as e:
            print(f"{sid}: ERROR {e}")
            continue
        write_json(f, r)
        print(f"{sid} [{r['lang']}] answers={r['answers']} calls/answer={r['calls_per_answer']} "
              f"followups={r['followups']} violations={len(r['feedback_violations_final'])} ({r['seconds']}s)")


if __name__ == "__main__":
    main()
