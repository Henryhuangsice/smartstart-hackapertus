"""Deterministic interview planner (no LLM).

Builds a 10–14 question plan that follows the FHGR interview flow
(introduction → motivation → strengths/weaknesses → situational →
candidate questions → closing), adapted to the apprenticeship posting
(occupational field), the candidate profile (conditional questions)
and an optional scenario (planned_question_ids).
"""

from __future__ import annotations

import random
import re

from .data import Dataset

STAGES = [
    "introduction",
    "motivation",
    "strengths_weaknesses",
    "situational",
    "candidate_questions",
    "closing",
]

_OCC_STOP = {"efz", "eba", "cfc", "afc", "afp", "aba", "fachmann", "fachfrau", "in", "e", "/"}


def _cond_ok(q: dict, cand: dict | None) -> bool:
    cond = (q.get("condition") or "").lower()
    if not cond:
        return True
    if cand is None:
        return False
    school = cand.get("school") or {}
    if "stellwerk" in cond:
        return bool(school.get("stellwerk"))
    if "absenz" in cond or "note" in cond:
        return bool(school.get("absences_note") or school.get("weak_subjects"))
    return False


def match_posting(ds: Dataset, text: str, lang: str | None = None) -> str | None:
    """Fuzzy-match a free-text wish ("Lehre als Informatiker") to a posting id."""
    toks = {t for t in re.findall(r"[a-zà-ÿ]{4,}", (text or "").lower())} - _OCC_STOP
    if not toks:
        return None
    best, best_score = None, 0.0
    for pid, p in ds.postings.items():
        names = " ".join(str(v) for k, v in (p.get("occupation") or {}).items() if k.startswith("name_")).lower()
        name_toks = set(re.findall(r"[a-zà-ÿ]{4,}", names)) - _OCC_STOP
        score = 0.0
        for t in toks:
            for n in name_toks:
                if t == n:
                    score += 2
                elif t[:6] == n[:6] and len(t) >= 6:
                    score += 1
        if lang and p.get("lang") == lang:
            score += 0.1
        if score > best_score:
            best, best_score = pid, score
    return best if best_score >= 1 else None


def default_posting(ds: Dataset, lang: str) -> str:
    for pid, p in ds.postings.items():
        if p.get("lang") == lang:
            return pid
    return next(iter(ds.postings))


def build_plan(
    ds: Dataset,
    *,
    posting_id: str,
    candidate_id: str | None = None,
    scenario_id: str | None = None,
    seed: int = 0,
    n_target: int = 12,
) -> list[str]:
    """Return an ordered list of question ids."""
    if scenario_id and scenario_id in ds.scenarios:
        planned = [q for q in ds.scenarios[scenario_id].get("planned_question_ids") or [] if q in ds.questions]
        if planned:
            return planned

    rng = random.Random(f"{posting_id}-{candidate_id}-{seed}")
    posting = ds.postings[posting_id]
    field_key = (posting.get("occupation") or {}).get("berufsfeld_key")
    cand = ds.candidates.get(candidate_id) if candidate_id else None

    def pool(prefixes: tuple[str, ...]) -> list[str]:
        out = [
            qid
            for qid, q in ds.questions.items()
            if qid.startswith(prefixes)
            and (q.get("situation") in (None, "apprenticeship_interview"))
            and _cond_ok(q, cand)
        ]
        return sorted(out)

    def pick(prefixes: tuple[str, ...], k: int, first: str | None = None) -> list[str]:
        p = pool(prefixes)
        chosen: list[str] = []
        if first and first in p:
            chosen.append(first)
            p.remove(first)
        rng.shuffle(p)
        chosen += p[: max(0, k - len(chosen))]
        return sorted(chosen) if not first else chosen

    plan: list[str] = []
    plan += pick(("Q-01-",), 1, first="Q-01-01")  # small talk
    plan += pick(("Q-02-",), 1, first="Q-02-01")  # tell me about yourself
    plan += pick(("Q-03-",), 2, first="Q-03-01")  # why this occupation
    plan += pick(("Q-04-",), 1)  # knowledge about company/occupation
    bf = [
        qid
        for qid, q in ds.questions.items()
        if qid.startswith("Q-BF") and ((q.get("berufsfeld") or {}).get("key") if isinstance(q.get("berufsfeld"), dict) else q.get("berufsfeld")) == field_key
    ]
    if bf:
        plan.append(rng.choice(sorted(bf)))
    plan += pick(("Q-05-",), 2, first="Q-05-01")  # strengths / weaknesses
    plan += pick(("Q-06-", "Q-07-"), 1)
    plan += pick(("Q-SIT-",), 1)
    plan += pick(("Q-DIF-",), 1)
    plan += pick(("Q-08-",), 1)
    plan += pick(("Q-09-",), 1, first="Q-09-01")  # do you have questions?
    plan += pick(("Q-10-",), 1, first="Q-10-01")  # closing
    # keep within the target length but never drop the frame questions
    if len(plan) > n_target + 2:
        core = {"Q-01-01", "Q-02-01", "Q-03-01", "Q-05-01", "Q-09-01", "Q-10-01"}
        extra = [q for q in plan if q not in core]
        drop = set(extra[: len(plan) - (n_target + 2)])
        plan = [q for q in plan if q not in drop]
    return plan


def stage_of(ds: Dataset, qid: str) -> str:
    return ds.questions[qid].get("flow_stage") or "situational"
