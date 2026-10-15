"""Core coaching logic, independent of LangGraph (unit-testable).

Per candidate answer the coach makes exactly 2 LLM calls:
  1. assess      – rubric-grounded, exemplar-calibrated JSON assessment
  2. interviewer – the next natural interviewer utterance (follow-up or next question)
At the end of the interview: 1 feedback call (+1 rewrite only if the rule checker
finds a guideline violation). Average ≈ 2.1–2.3 calls per answer (< 5 required).
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from . import llm
from .config import SETTINGS
from .data import Dataset, get_dataset
from .langdetect import detect
from .planner import build_plan, default_posting, match_posting, stage_of
from .prompts import (
    ADDRESS,
    ASSESS_SYSTEM,
    ASSESS_USER,
    CLOSING_NOTE,
    FEEDBACK_HEADINGS,
    FEEDBACK_SYSTEM,
    FEEDBACK_USER,
    INTERVIEWER_SYSTEM,
    INTERVIEWER_USER,
    LANG_NAME,
    OPENING_USER,
    REWRITE_USER,
)
from .retrieval import get_index
from .rules import check_feedback

CORE_CRITERIA = ["clarity", "relevance", "motivation", "self_reflection", "communication", "concrete_examples"]
ALL_CRITERIA = CORE_CRITERIA + ["demeanor", "preparation", "goal_orientation", "difficult_questions", "initiative"]

END_WORDS = {"/feedback", "/ende", "/end", "/fin", "/fine", "feedback bitte", "stop interview"}


# ======================================================================
# session handling
# ======================================================================
def new_session(
    first_message: str,
    *,
    lang: str | None = None,
    dialect: str | None = None,
    posting_id: str | None = None,
    candidate_id: str | None = None,
    scenario_id: str | None = None,
    feedback_mode: str | None = None,
    ds: Dataset | None = None,
) -> dict[str, Any]:
    ds = ds or get_dataset()
    scen = ds.scenarios.get(scenario_id) if scenario_id else None
    if scen:
        candidate_id = candidate_id or scen.get("candidate_id")
        posting_id = posting_id or scen.get("posting_id")
        lang = lang or scen.get("lang")
    cand = ds.candidates.get(candidate_id) if candidate_id else None
    det_lang, is_gsw, _ = detect(first_message, default=(cand or {}).get("interview_language", "de"))
    if lang not in ("de", "fr", "it"):
        lang = (cand or {}).get("interview_language") or (det_lang if det_lang in ("de", "fr", "it") else "de")
    if not posting_id or posting_id not in ds.postings:
        posting_id = (
            (cand or {}).get("target_posting_id")
            or match_posting(ds, first_message, lang)
            or default_posting(ds, lang)
        )
    plan = build_plan(ds, posting_id=posting_id, candidate_id=candidate_id, scenario_id=scenario_id)
    return {
        "lang": lang,
        "dialect_input": bool(is_gsw),
        "dialect": dialect,  # optional: ask questions in a Swiss-German dialect (e.g. "zh")
        "posting_id": posting_id,
        "candidate_id": candidate_id,
        "scenario_id": scenario_id,
        "feedback_mode": feedback_mode or SETTINGS.feedback_mode,
        "plan": plan,
        "idx": 0,
        "pending_followup": None,
        "followups": {},
        "followups_total": 0,
        "log": [],  # one entry per candidate answer
        "llm_calls": 0,
        "answers": 0,
        "finished": False,
        "support_mode": False,
        "feedback": None,
        "feedback_violations": [],
    }


def _persona(ds: Dataset, s: dict) -> dict[str, str]:
    p = ds.postings[s["posting_id"]]
    lang = s["lang"]
    iv = p.get("interviewer") or {}
    comp = p.get("company") or {}
    cand = ds.candidates.get(s.get("candidate_id") or "") or {}
    facts = [
        comp.get("description", ""),
        "Values: " + ", ".join(comp.get("values") or []),
        "Products/services: " + ", ".join(comp.get("products_services") or []),
        f"Employees: {comp.get('size_employees')}, apprentices: {comp.get('apprentices_total')}",
        "Offers: " + "; ".join(p.get("offers") or []) if isinstance(p.get("offers"), list) else f"Offers: {p.get('offers')}",
        f"Start: {p.get('start_date')}; BM possible: {p.get('bm_possible')}",
        f"Selection process: {p.get('selection_process')}",
        "Insider facts: " + "; ".join(p.get("insider_facts") or []) if isinstance(p.get("insider_facts"), list) else f"Insider facts: {p.get('insider_facts')}",
    ]
    return {
        "interviewer_name": iv.get("name") or "die Berufsbildnerin",
        "interviewer_role": iv.get("role") or "Berufsbildner/in",
        "style": f"{iv.get('style', 'friendly, structured')}. {iv.get('notes', '')}".strip(),
        "company": comp.get("name") or "der Lehrbetrieb",
        "place": comp.get("place") or "",
        "occupation": ds.occupation_name(p, lang),
        "candidate_name": cand.get("first_name") or "the candidate",
        "company_facts": "\n".join(f"- {f}" for f in facts if f and not f.endswith(": ")),
        "lang_name": LANG_NAME[lang],
        "address": ADDRESS[lang],
    }


def _history(s: dict, n: int = 2) -> str:
    rows = []
    for e in s["log"][-n:]:
        rows.append(f'Interviewer: "{e["question"]}"\nCandidate: "{e["answer"]}"')
    return "\n".join(rows) or "(start of interview)"


def current_question(ds: Dataset, s: dict) -> tuple[str | None, str]:
    """(question_id, text) of the question the candidate is currently answering."""
    if s.get("pending_followup"):
        fu = s["pending_followup"]
        return fu["qid"], fu["text"]
    if s["idx"] >= len(s["plan"]):
        return None, ""
    qid = s["plan"][s["idx"]]
    return qid, ds.question_text(qid, s["lang"], s.get("dialect"))


def _norm(t: str) -> str:
    import re as _re
    return " ".join(_re.findall(r"[\wÀ-ÿ']+", (t or "").lower()))


def verify_quote(quote: str, answer: str) -> bool:
    """True if the quoted evidence really occurs in the candidate's answer (whitespace/punctuation-insensitive)."""
    q, a = _norm(quote).strip(" ."), _norm(answer)
    if not q:
        return False
    if q in a:
        return True
    qt = q.split()
    at = set(a.split())
    return len(qt) >= 3 and sum(t in at for t in qt) / len(qt) >= 0.9


def _str_list(x: Any, n: int = 3) -> list[str]:
    if isinstance(x, str):
        x = [x] if x.strip() else []
    return [str(i).strip() for i in (x or []) if str(i).strip()][:n]


# ======================================================================
# LLM steps
# ======================================================================
def assess(
    ds: Dataset,
    s: dict,
    qid: str,
    question: str,
    answer: str,
    *,
    use_retrieval: bool = True,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    q = ds.questions.get(qid) or {}
    crit = [c for c in (q.get("criteria") or CORE_CRITERIA[:3]) if c in ds.criteria_by_key]
    per = _persona(ds, s)
    if use_retrieval:
        ex = get_index().retrieve(qid, answer, s["lang"], k=SETTINGS.n_exemplars, exclude=exclude)
        ex_txt = "\n\n".join(e.to_prompt() for e in ex) or "(none)"
    else:
        ex, ex_txt = [], "(none)"
    user = ASSESS_USER.format(
        lang=s["lang"],
        lang_name=LANG_NAME[s["lang"]],
        address=ADDRESS[s["lang"]],
        occupation=per["occupation"],
        company=per["company"],
        stage=q.get("flow_stage", ""),
        qid=qid,
        rubric=ds.rubric_text(crit),
        exemplars=ex_txt,
        history=_history(s),
        question=question,
        answer=answer,
        criteria_keys=", ".join(f'"{c}": 1-4|null' for c in crit),
    )
    raw = llm.chat(
        [{"role": "system", "content": ASSESS_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=450,
        tag="assess",
    )
    s["llm_calls"] += 1
    a = llm.parse_json(raw)
    if not a:  # one strict retry instead of silently inventing a default score
        raw = llm.chat(
            [{"role": "system", "content": ASSESS_SYSTEM},
             {"role": "user", "content": user + "\n\nReturn ONLY the JSON object."}],
            temperature=0.0, max_tokens=450, tag="assess",
        )
        s["llm_calls"] += 1
        a = llm.parse_json(raw)
    # --- sanitise --------------------------------------------------------
    def _lvl(x: Any) -> int | None:
        try:
            v = int(x)
            return v if 1 <= v <= 4 else None
        except (TypeError, ValueError):
            return None

    scores = {k: _lvl(v) for k, v in (a.get("scores") or {}).items() if k in ds.criteria_by_key}
    level = _lvl(a.get("quality_level"))
    if level is None:
        vals = [v for v in scores.values() if v]
        level = round(mean(vals)) if vals else None  # None = not assessable (no invented score)
    action = a.get("coach_action") if a.get("coach_action") in {"continue", "probe", "redirect", "support_and_refer"} else (
        "probe" if (level or 3) <= 2 else "continue"
    )
    observed, missing = _str_list(a.get("observed")), _str_list(a.get("missing"))
    target = str(a.get("follow_up_target") or "").strip() or (missing[0] if missing else "")
    if action == "probe" and not missing and level and level >= 3:
        action = "continue"  # nothing important missing → do not probe for the sake of probing
    evidence = str(a.get("evidence") or "")[:300]
    evidence_ok = verify_quote(evidence, answer)
    flags = [f for f in (a.get("problem_flags") or []) if isinstance(f, str)]
    if len(answer.split()) <= 2 and "one_word" not in flags:
        flags.append("one_word")
    return {
        "quality_level": level,
        "scores": scores,
        "evidence": evidence if evidence_ok else "",
        "evidence_claimed": evidence,
        "evidence_verified": evidence_ok,
        "observed": observed,
        "missing": missing,
        "follow_up_target": target,
        "strength": str(a.get("strength") or ""),
        "improvement": str(a.get("improvement") or ""),
        "coach_action": action,
        "problem_flags": flags,
        "follow_up": str(a.get("follow_up") or "").strip(),
        "candidate_question": str(a.get("candidate_question") or "").strip(),
        "tip": str(a.get("tip") or "").strip(),
        "exemplars": [e.answer_id for e in ex],
        "raw_ok": bool(a),
    }


def interviewer_turn(
    ds: Dataset,
    s: dict,
    *,
    answer: str,
    move: str,
    next_question: str,
    stage: str,
    candidate_question: str = "",
    opening: bool = False,
    last: bool = False,
) -> str:
    per = _persona(ds, s)
    system = INTERVIEWER_SYSTEM.format(**per)
    if opening:
        p = ds.postings[s["posting_id"]]
        scen = ds.scenarios.get(s.get("scenario_id") or "") or {}
        user = OPENING_USER.format(
            answer=answer, next_question=next_question, duration=scen.get("duration_min") or 30
        )
        _ = p
    else:
        cq = (
            f'The candidate asked you: "{candidate_question}". Answer it briefly and honestly using only the company facts.\n'
            if candidate_question
            else ""
        )
        sup = (
            "The candidate shows signs of insecurity/distress: respond calmly and supportively, reassure them that "
            "this is practice, and mention they can also talk to a career counsellor, teacher or parents.\n"
            if s.get("support_mode")
            else ""
        )
        situation_note = "This is the last question of the interview." if last else ""
        end_note = "" if next_question else " — or, if there is no next question, thank them and close the interview"
        user = INTERVIEWER_USER.format(
            stage=stage,
            situation_note=situation_note,
            answer=answer,
            candidate_question_note=cq,
            support_note=sup,
            move=move,
            next_question=next_question or "(none — close the interview politely)",
            end_note=end_note,
        )
    out = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.4,
        max_tokens=220,
        tag="interviewer",
    )
    s["llm_calls"] += 1
    return out.strip().strip('"')


def aggregate_levels(s: dict) -> dict[str, int | None]:
    acc: dict[str, list[int]] = {c: [] for c in ALL_CRITERIA}
    for e in s["log"]:
        for k, v in (e["assessment"].get("scores") or {}).items():
            if v and k in acc:
                acc[k].append(v)
    return {k: (round(mean(v)) if v else None) for k, v in acc.items()}


def final_feedback(ds: Dataset, s: dict) -> str:
    lang = s["lang"]
    per = _persona(ds, s)
    h = FEEDBACK_HEADINGS[lang]
    log_lines = []
    for i, e in enumerate(s["log"], 1):
        a = e["assessment"]
        log_lines.append(
            f'{i}. Q: "{e["question"]}"\n   A: "{e["answer"]}"\n'
            f'   notes: level={a["quality_level"]}, strength="{a["strength"]}", improvement="{a["improvement"]}", '
            f'evidence="{a["evidence"]}", flags={a["problem_flags"]}'
        )
    levels = aggregate_levels(s)
    lv_txt = ", ".join(f"{k}={v}" for k, v in levels.items())
    sup = (
        "IMPORTANT: the candidate showed signs of strong insecurity — be extra supportive and suggest talking to a "
        "career counsellor, teacher or parents (L3)."
        if s.get("support_mode")
        else ""
    )
    user = FEEDBACK_USER.format(
        lang_name=LANG_NAME[lang],
        address=ADDRESS[lang],
        h_strengths=h[0],
        h_improve=h[1],
        h_next=h[2],
        support_note=sup,
        occupation=per["occupation"],
        company=per["company"],
        candidate_name=per["candidate_name"],
        log="\n".join(log_lines) or "(no answers)",
        levels=lv_txt,
    )
    system = FEEDBACK_SYSTEM.format(guidelines=ds.guidelines_text())
    fb = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
        max_tokens=900,
        tag="feedback",
    )
    s["llm_calls"] += 1
    violations = check_feedback(fb, lang)
    s["feedback_violations_first"] = violations
    if violations:
        fb2 = llm.chat(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": REWRITE_USER.format(
                        violations="\n".join(f"- {v}" for v in violations), lang_name=LANG_NAME[lang], feedback=fb
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=900,
            tag="rewrite",
        )
        s["llm_calls"] += 1
        v2 = check_feedback(fb2, lang)
        if len(v2) <= len(violations):
            fb, violations = fb2, v2
    s["feedback_violations"] = violations
    s["feedback"] = fb.strip()
    return fb.strip() + "\n\n" + rubric_overview(ds, s, levels)


def rubric_overview(ds: Dataset, s: dict, levels: dict[str, int | None]) -> str:
    """Deterministic rubric table (K3: 'no evidence' instead of guessing)."""
    lang = s["lang"]
    rows = [f"### {FEEDBACK_HEADINGS[lang][3]}", "", "| | |", "|---|---|"]
    for k in ALL_CRITERIA:
        rows.append(f"| {ds.criterion_label(k, lang)} | {ds.level_label(levels.get(k), lang)} |")
    return "\n".join(rows)


# ======================================================================
# one conversational step
# ======================================================================
def step(s: dict, user_text: str, ds: Dataset | None = None) -> str:
    """Process one candidate message, mutate the session, return the coach reply."""
    ds = ds or get_dataset()
    text = (user_text or "").strip()

    if s["finished"] and s.get("feedback_failed"):
        try:
            fb = final_feedback(ds, s)
            s["feedback_failed"] = False
            return f"**{CLOSING_NOTE[s['lang']]}**\n\n" + fb
        except Exception:
            return {"de": "Das Feedback konnte noch nicht erstellt werden. Bitte versuche es gleich nochmals.",
                    "fr": "Le feedback n'a pas encore pu être créé. Réessaie dans un instant.",
                    "it": "Non è stato ancora possibile creare il feedback. Riprova tra un attimo."}[s["lang"]]
    if s["finished"]:
        restart = {"de": "Möchtest du noch einmal üben? Schreib einfach /neu.",
                   "fr": "Tu veux t'entraîner encore une fois ? Écris /nouveau.",
                   "it": "Vuoi esercitarti ancora? Scrivi /nuovo."}
        return restart[s["lang"]]

    # ---- opening turn ---------------------------------------------------
    if not s.get("started"):
        s["started"] = True
        qid, qtext = current_question(ds, s)
        return interviewer_turn(
            ds, s, answer=text, move="open the interview", next_question=qtext, stage="introduction", opening=True
        )

    # ---- candidate answered the current question -------------------------
    qid, qtext = current_question(ds, s)
    if qid is None:
        s["finished"] = True
        return final_feedback(ds, s)

    early_end = text.lower() in END_WORDS
    if not early_end:
        a = assess(ds, s, qid, qtext, text)
        s["answers"] += 1
        pf = s.get("pending_followup") or {}
        s["log"].append({"qid": qid, "question": qtext, "answer": text, "assessment": a,
                         "is_followup": bool(pf), "followup_target": pf.get("target", "")})
        if "distress_signal" in a["problem_flags"] or (
            a["coach_action"] == "support_and_refer" and (a["quality_level"] or 2) <= 2
        ):
            s["support_mode"] = True

    # ---- decide next move (deterministic policy) -------------------------
    move, next_q, cand_q = "", "", ""
    if not early_end:
        a = s["log"][-1]["assessment"]
        cand_q = a.get("candidate_question", "")
        stage = stage_of(ds, qid)
        can_follow = (
            not s.get("pending_followup")
            and s["followups"].get(qid, 0) < SETTINGS.max_followups_per_question
            and s["followups_total"] < SETTINGS.max_followups_total
            and stage not in ("closing",)
            and a["coach_action"] in ("probe", "redirect")
            and a.get("follow_up")
            and not s.get("support_mode")
            and _norm(a.get("follow_up_target")) not in s.setdefault("asked_targets", {}).get(qid, [])
        )
        if stage == "candidate_questions" and cand_q:
            # candidate asked something: answer it, then allow one more question round
            can_follow = s["followups"].get(qid, 0) < 2
            if can_follow:
                fu_text = {"de": "Hast du sonst noch eine Frage?", "fr": "Tu as encore une autre question ?",
                           "it": "Hai ancora un'altra domanda?"}[s["lang"]]
                s["followups"][qid] = s["followups"].get(qid, 0) + 1
                s["pending_followup"] = {"qid": qid, "text": fu_text}
                move = "answer the candidate's question, then ask if they have another question"
                next_q = fu_text
        elif can_follow:
            s["followups"][qid] = s["followups"].get(qid, 0) + 1
            s["followups_total"] += 1
            s["pending_followup"] = {"qid": qid, "text": a["follow_up"], "target": a.get("follow_up_target", "")}
            s["asked_targets"].setdefault(qid, []).append(_norm(a.get("follow_up_target")))
            tgt = a.get("follow_up_target") or "a concrete example"
            move = (f"ask ONE short follow-up that obtains only this missing information: {tgt}"
                    if a["coach_action"] == "probe" else "gently bring the conversation back to the question")
            next_q = a["follow_up"]
        if not move:
            s["pending_followup"] = None
            s["idx"] += 1
            nqid, next_q = current_question(ds, s)
            move = "acknowledge briefly and naturally, then move on to the next question" if nqid else "close"

    if early_end or (not next_q):
        s["finished"] = True
        closing = ""
        if not early_end:
            closing = interviewer_turn(ds, s, answer=text, move="thank the candidate and close the interview",
                                       next_question="", stage="closing", candidate_question=cand_q, last=True)
        try:
            fb = final_feedback(ds, s)
        except Exception:  # keep all answers; the next message retries the feedback
            s["feedback_failed"] = True
            fb = {"de": "Das Feedback konnte gerade nicht erstellt werden. Schreib einfach /feedback, um es nochmals zu versuchen.",
                  "fr": "Le feedback n'a pas pu être créé. Écris /feedback pour réessayer.",
                  "it": "Non è stato possibile creare il feedback. Scrivi /feedback per riprovare."}[s["lang"]]
        return (closing + "\n\n---\n" if closing else "") + f"**{CLOSING_NOTE[s['lang']]}**\n\n" + fb

    reply = interviewer_turn(
        ds, s, answer=text, move=move, next_question=next_q,
        stage=stage_of(ds, s["plan"][min(s["idx"], len(s["plan"]) - 1)]),
        candidate_question=cand_q, last=(s["idx"] == len(s["plan"]) - 1),
    )
    if s["feedback_mode"] == "per_answer" and s["log"] and s["log"][-1]["assessment"].get("tip"):
        reply = f"_💡 {s['log'][-1]['assessment']['tip']}_\n\n" + reply
    return reply


def calls_per_answer(s: dict) -> float:
    return round(s["llm_calls"] / max(1, s["answers"]), 2)


# ======================================================================
# practice loop: revise one answer after the feedback (1 LLM call)
# ======================================================================
REVISE_TEXT = {
    "de": ("Vorher", "Jetzt", "Verbessert", "Gleich geblieben", "Noch nicht besser", "Neu gezeigt", "Keine Aussage mehr"),
    "fr": ("Avant", "Maintenant", "Amélioré", "Inchangé", "Pas encore mieux", "Nouvellement montré", "Plus d'indice"),
    "it": ("Prima", "Adesso", "Migliorato", "Invariato", "Non ancora meglio", "Mostrato ora", "Nessun indizio"),
}


def revise_answer(s: dict, index: int, new_answer: str, ds: Dataset | None = None) -> dict[str, Any]:
    """Re-assess an improved version of answer `index` and compare it with the first try.

    Uses the same rubric-grounded `assess` step (1 LLM call); the comparison itself is
    deterministic, so the coach never invents progress that the rubric does not show.
    """
    ds = ds or get_dataset()
    e = s["log"][index]
    before = e["assessment"]
    after = assess(ds, s, e["qid"], e["question"], new_answer)
    lang = s["lang"] if s["lang"] in REVISE_TEXT else "de"
    t = REVISE_TEXT[lang]
    rows = []
    for k in sorted(set(before.get("scores", {})) | set(after.get("scores", {}))):
        b, a = before.get("scores", {}).get(k), after.get("scores", {}).get(k)
        if a is None and b is None:
            continue
        if b is None:
            mark = t[5]  # first evidence, not a measurable improvement
        elif a is None:
            mark = t[6]
        else:
            mark = t[2] if a > b else (t[3] if a == b else t[4])
        rows.append({"criterion": ds.criterion_label(k, lang), "before": b, "after": a, "verdict": mark})
    e.setdefault("revisions", []).append({"answer": new_answer, "assessment": after})
    return {
        "question": e["question"],
        "before": {"answer": e["answer"], "level": before["quality_level"]},
        "after": {"answer": new_answer, "level": after["quality_level"], "strength": after["strength"],
                  "missing": after.get("missing", []),
                  "improvement": after["improvement"], "evidence": after["evidence"]},
        "criteria": rows,
        "labels": {"before": t[0], "after": t[1]},
    }


_STOP = set("""ich du er sie es wir ihr der die das den dem des ein eine einen einem und oder aber auch mit von zu im in am an auf für bei
habe hat haben war ist sind wurde mich mir mein meine je tu il elle le la les un une et ou de du des à au aux avec pour dans sur
ai a ont était io tu lui lei il lo la le un una e o di da del della con per in su ho ha hanno era""".split())


def _content(t: str) -> list[str]:
    return [w for w in _norm(t).split() if w not in _STOP and len(w) > 2]


def _supported(text: str, story: str, thr: float = 0.6) -> bool:
    """A paraphrase is supported if most of its content words (or their stems) occur in the story."""
    tw, sw = [_fold(w) for w in _content(text)], [_fold(w) for w in _content(story)]
    if not tw:
        return False
    hit = sum(any(w == x or (len(w) > 4 and x.startswith(w[:5])) for x in sw) for w in tw)
    return hit / len(tw) >= thr


def _fold(w: str) -> str:
    w = w.lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss"), ("é", "e"), ("è", "e"), ("à", "a"), ("ù", "u")):
        w = w.replace(a, b)
    return w


def _restore_spelling(text: str, story: str) -> str:
    """Give words back the candidate's own spelling (umlauts, capitals) when the model flattened them."""
    import re as _re
    orig = {}
    for w in _re.findall(r"[\wÀ-ÿ]+", story):
        orig.setdefault(_fold(w), w)
        orig.setdefault(_fold(w).replace("ae", "a").replace("oe", "o").replace("ue", "u"), w)
    return _re.sub(r"[\wÀ-ÿ]+", lambda m: orig.get(_fold(m.group(0)), m.group(0)), text)


def _best_span(text: str, story: str) -> str:
    """Shortest sentence of the story that shares most content words with the fact (for display)."""
    import re as _re
    sents = [x.strip() for x in _re.split(r"(?<=[.!?])\s+", story) if x.strip()] or [story]
    tw = set(_content(text))
    return max(sents, key=lambda x: len(tw & set(_content(x))))


# ======================================================================
# A-lite: experience discovery → fact card (1 LLM call, quotes verified)
# ======================================================================
NOT_STATED = {"de": "noch nicht genannt", "fr": "pas encore précisé", "it": "non ancora detto"}


def discovery_prompt(s: dict) -> str:
    from .prompts import DISCOVER_PROMPT
    s["discovery"] = {"active": True}
    return DISCOVER_PROMPT[s["lang"]]


def make_fact_card(s: dict, story: str, ds: Dataset | None = None) -> dict[str, Any]:
    """Draft an experience card from the candidate's own story. Every fact must be backed by a
    verbatim quote from the story; unsupported facts are dropped, a missing result is shown as such."""
    from .prompts import FACT_CARD_USER
    ds = ds or get_dataset()
    _, question = current_question(ds, s)
    raw = llm.chat([{"role": "system", "content": ASSESS_SYSTEM},
                    {"role": "user", "content": FACT_CARD_USER.format(question=question, story=story,
                                                                      lang_name=LANG_NAME[s["lang"]])}],
                   temperature=0.0, max_tokens=400, tag="factcard")
    s["llm_calls"] += 1
    d = llm.parse_json(raw)
    facts, dropped = {}, []
    for k in ("situation", "action", "result"):
        f = d.get(k) or {}
        text, quote = str(f.get("text") or "").strip(), str(f.get("quote") or "").strip()
        ok_quote = verify_quote(quote, story)
        if text and (ok_quote or _supported(text, story)):
            text = _restore_spelling(text, story)
            facts[k] = {"text": text, "quote": quote if ok_quote else _best_span(text, story), "verified": True}
        else:
            if text:
                dropped.append(k)
            facts[k] = {"text": "", "quote": "", "verified": False}
    missing = [k for k, f in facts.items() if not f["text"]]
    answer = str(d.get("answer") or "").strip() if not dropped else ""
    if answer and not _supported(answer, story, thr=0.85):
        answer = ""  # the drafted answer adds content the candidate never said
    if not answer:  # fallback: the candidate's own words are always true
        answer = story.strip()
    sit = facts["situation"]
    if sit["text"] and len(sit["text"].split()) > 10 and facts["action"]["text"]:
        sit["text"] = " ".join(sit["text"].split()[:6]) + " …"  # keep the situation short, the action has its own field
    card = {"status": "draft", "question": question, "story": story, "facts": facts, "missing": missing,
            "dropped_unsupported": dropped, "skill": str(d.get("skill") or "").strip() if facts["action"]["text"] else "",
            "answer": answer, "not_stated": NOT_STATED[s["lang"]]}
    s["discovery"] = {"active": False, "card": card}
    return card


def confirm_fact_card(s: dict, facts: dict[str, str], answer: str) -> dict:
    card = (s.get("discovery") or {}).get("card") or {}
    for k, v in (facts or {}).items():
        if k in card.get("facts", {}):
            card["facts"][k]["text"] = (v or "").strip()
            card["facts"][k]["edited"] = True
    card["answer"], card["status"] = (answer or "").strip(), "confirmed"
    card["missing"] = [k for k, f in card.get("facts", {}).items() if not f["text"]]
    s.setdefault("cards", []).append(card)
    return card


# ======================================================================
# C: transfer practice — a NEW question that trains the same criterion (1 call)
# ======================================================================
def transfer_question(s: dict, index: int, ds: Dataset | None = None) -> dict[str, Any]:
    ds = ds or get_dataset()
    e = s["log"][index]
    sc = {k: v for k, v in (e["assessment"].get("scores") or {}).items()}
    weak = sorted(sc, key=lambda k: (sc[k] if sc[k] is not None else 0))[:1] or ["concrete_examples"]
    crit = weak[0]
    used = {x["qid"] for x in s["log"]} | set(s["plan"]) | {t["qid"] for t in s.get("transfers", [])}
    field = (ds.postings[s["posting_id"]].get("occupation") or {}).get("berufsfeld_key")

    def _field_ok(q: dict) -> bool:  # occupation-specific questions only from the candidate's own field
        bf = q.get("berufsfeld")
        bf = bf.get("key") if isinstance(bf, dict) else bf
        return not bf or bf == field

    pool = [qid for qid, q in ds.questions.items()
            if crit in (q.get("criteria") or []) and qid not in used and _field_ok(q)
            and q.get("situation") in (None, "apprenticeship_interview") and not q.get("condition")]
    if not pool:
        pool = [qid for qid, q in ds.questions.items()
                if qid not in used and _field_ok(q) and "concrete_examples" in (q.get("criteria") or [])]
    qid = sorted(pool)[(index * 7) % len(pool)]
    t = {"qid": qid, "question": ds.question_text(qid, s["lang"]), "criterion": crit,
         "criterion_label": ds.criterion_label(crit, s["lang"]), "source_index": index,
         "source_improvement": e["assessment"].get("improvement", "")}
    s.setdefault("transfers", []).append(t)
    return t


def answer_transfer(s: dict, text: str, ds: Dataset | None = None) -> dict[str, Any]:
    ds = ds or get_dataset()
    t = s["transfers"][-1]
    a = assess(ds, s, t["qid"], t["question"], text)  # the assessor is NOT told that progress is expected
    src = s["log"][t["source_index"]]["assessment"].get("scores", {}).get(t["criterion"])
    new = a.get("scores", {}).get(t["criterion"])
    if new is None:
        status = "no_evidence"
    elif src is None:
        status = "first_evidence"
    else:
        status = "higher" if new > src else ("same" if new == src else "lower")
    t.update({"answer": text, "assessment": a, "before": src, "after": new, "status": status})
    return t


# ======================================================================
# D: export a personal practice card (deterministic, no LLM call)
# ======================================================================
EXPORT_H = {
    "de": ("Mein Übungsblatt", "Mein Beispiel (von mir bestätigt)", "Das zeige ich schon", "Daran arbeite ich",
           "Mein nächster Schritt", "Frage fürs nächste Üben"),
    "fr": ("Ma fiche d'entraînement", "Mon exemple (confirmé par moi)", "Ce que je montre déjà", "Ce que je travaille",
           "Ma prochaine étape", "Question pour la prochaine fois"),
    "it": ("La mia scheda di esercizio", "Il mio esempio (confermato da me)", "Cosa mostro già", "Su cosa lavoro",
           "Il mio prossimo passo", "Domanda per la prossima volta"),
}


def export_card(s: dict, ds: Dataset | None = None) -> str:
    import re as _re
    ds = ds or get_dataset()
    lang = s["lang"] if s["lang"] in EXPORT_H else "de"
    h = EXPORT_H[lang]
    p = ds.postings[s["posting_id"]]
    lines = [f"# {h[0]} — {ds.occupation_name(p, lang)}", ""]
    cards = [c for c in s.get("cards", []) if c.get("status") == "confirmed"]
    if cards:
        c = cards[-1]
        fl = {"de": ("Situation", "Was ich gemacht habe", "Ergebnis / was ich gelernt habe"),
              "fr": ("Situation", "Ce que j'ai fait", "Résultat / ce que j'ai appris"),
              "it": ("Situazione", "Cosa ho fatto", "Risultato / cosa ho imparato")}[lang]
        lines += [f"## {h[1]}", ""] + [f"- **{fl[i]}:** {f['text'] or c['not_stated']}"
                                       for i, f in enumerate(c["facts"].values())] + [""]
    levels = aggregate_levels(s)
    known = {k: v for k, v in levels.items() if v}
    if known:
        best = max(known, key=known.get)
        q = next((e["assessment"]["evidence"] for e in reversed(s["log"])
                  if e["assessment"].get("evidence") and (e["assessment"].get("quality_level") or 0) >= 3
                  and (e["assessment"].get("scores") or {}).get(best) == known[best]), "")
        lines += [f"## {h[2]}", "", f"- {ds.criterion_label(best, lang)}" + (f" — «{q}»" if q else ""), ""]
        rest = {k: v for k, v in known.items() if k != best} or known
        worst = min(rest, key=rest.get)
        gap = next((e["assessment"]["missing"][0] for e in reversed(s["log"]) if e["assessment"].get("missing")), "")
        lines += [f"## {h[3]}", "", f"- {ds.criterion_label(worst, lang)}" + (f": {gap}" if gap else ""), ""]
    fb = s.get("feedback") or ""
    step = ""
    head = FEEDBACK_HEADINGS[lang][2].lower()
    for block in fb.split("## ")[1:]:
        title, _, body = block.partition("\n")
        if head in title.lower():
            step = body.split("\n\n")[0].strip()
    if not step:
        step = next((e["assessment"].get("tip") for e in reversed(s["log"]) if e["assessment"].get("tip")), "")
    if step:
        lines += [f"## {h[4]}", "", step.strip(), ""]
    nq = (s.get("transfers") or [{}])[-1].get("question")
    if not nq and s["log"]:
        nq = transfer_question(s, len(s["log"]) - 1, ds)["question"]
    if nq:
        lines += [f"## {h[5]}", "", nq, ""]
    return "\n".join(lines)
