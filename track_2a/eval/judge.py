"""Experiment 3 — LLM-as-judge on simulated interviews + judge validation.

(a) Judge validation: on the 30 FHGR good-vs-poor feedback pairs the judge rates
    each feedback (1-5) blind; we report pairwise accuracy (good > poor).
(b) Transcript judging: for every simulated scenario the judge checks the
    scenario's `success_criteria_for_coach` (pass/fail) and rates five
    dimensions 1-5 mirroring the FHGR "Performance" criterion:
    realism, adaptivity, relevance, appropriateness, feedback_quality
    (+ language_consistency).

By default the judge is Apertus itself (LLM_*); set JUDGE_BASE_URL / JUDGE_NAME /
JUDGE_API_KEY to use a different open-weights judge (documented in the report).

    python eval/judge.py --sim eval/results/sim --out eval/results
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from statistics import mean

from _common import JUDGE, ROOT, read_json, write_json

from smartstart import llm
from smartstart.data import get_dataset

FB_JUDGE = """You are an expert in feedback for young people (14-17) after job-interview practice.
Rate the FEEDBACK below from 1 (poor) to 5 (excellent) against these guidelines:
{guidelines}

Context: question: "{question}" — candidate answer: "{answer}"
FEEDBACK: "{feedback}"

Return only JSON: {{"score": 1-5, "violations": ["rule ids"], "reason": "one sentence"}}"""

TR_JUDGE = """You are evaluating an AI job-interview coach for Swiss apprenticeship candidates (14-16 years).
Below is a full practice interview (coach = interviewer, then final feedback). Interview language: {lang}.

SCENARIO SUCCESS CRITERIA FOR THE COACH:
{criteria}

FHGR FEEDBACK GUIDELINES (for the final feedback):
{guidelines}

TRANSCRIPT:
{transcript}

Return only JSON:
{{"criteria_met": [true/false for each success criterion in order],
 "realism": 1-5, "adaptivity": 1-5, "relevance": 1-5, "appropriateness": 1-5,
 "feedback_quality": 1-5, "language_consistency": 1-5,
 "comment": "two sentences: main strength and main weakness"}}"""


def judge_call(prompt: str, max_tokens: int = 400) -> dict:
    raw = llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=max_tokens, tag="judge", **JUDGE)
    return llm.parse_json(raw)


def validate_judge(ds, n: int | None = None) -> dict:
    rng = random.Random(7)
    pairs = ds.feedback_examples[: n or len(ds.feedback_examples)]
    rows = []
    for ex in pairs:
        ctx = ex.get("context") or {}
        scores = {}
        order = ["good", "poor"]
        rng.shuffle(order)
        for kind in order:
            fb = ex.get(f"{kind}_feedback") or ""
            j = judge_call(FB_JUDGE.format(guidelines=ds.guidelines_text(), question=ctx.get("question", ""),
                                           answer=ctx.get("answer", ""), feedback=fb))
            try:
                scores[kind] = float(j.get("score"))
            except (TypeError, ValueError):
                scores[kind] = None
        if None not in scores.values():
            rows.append({"id": ex["id"], "lang": ex.get("lang"), **scores})
        print(f"judge-validation {ex['id']}: {scores}")
    acc = mean(r["good"] > r["poor"] for r in rows) if rows else None
    return {"pairs": len(rows), "pairwise_accuracy": round(acc, 3) if acc is not None else None, "rows": rows}


def judge_transcripts(ds, sim_dir: Path) -> list[dict]:
    out = []
    for f in sorted(sim_dir.glob("S-*.json")):
        r = read_json(f)
        tr = "\n".join(f"{t['role'].upper()}: {t['text']}" for t in r["transcript"])
        crit = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(r["success_criteria"])) or "(none)"
        j = judge_call(TR_JUDGE.format(lang=r["lang"], criteria=crit, guidelines=ds.guidelines_text(),
                                       transcript=tr[-14000:]), max_tokens=500)
        met = [bool(x) for x in (j.get("criteria_met") or [])][: len(r["success_criteria"])]
        row = {
            "scenario_id": r["scenario_id"], "lang": r["lang"], "difficulty": r.get("difficulty"),
            "dialect": r.get("dialect"),
            "criteria_met_rate": round(mean(met), 3) if met else None,
            **{k: j.get(k) for k in ("realism", "adaptivity", "relevance", "appropriateness",
                                      "feedback_quality", "language_consistency")},
            "calls_per_answer": r["calls_per_answer"], "answers": r["answers"],
            "violations_first": len(r.get("feedback_violations_first", [])),
            "violations_final": len(r.get("feedback_violations_final", [])),
            "comment": j.get("comment"),
        }
        print(row)
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="eval/results/sim")
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--validation-n", type=int, default=0)
    args = ap.parse_args()
    ds = get_dataset()
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    sim = Path(args.sim) if Path(args.sim).is_absolute() else ROOT / args.sim
    res = {"judge": {"model": JUDGE["model"], "base_url": JUDGE["base_url"]}}
    if not args.skip_validation:
        res["validation"] = validate_judge(ds, args.validation_n or None)
    res["transcripts"] = judge_transcripts(ds, sim)
    write_json(out / "judge.json", res)
    print("saved", out / "judge.json")


if __name__ == "__main__":
    main()
