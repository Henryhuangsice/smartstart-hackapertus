"""Rubric-grounded exemplar retrieval (no LLM, no embeddings needed).

For the current question we retrieve expert-annotated answers from the
FHGR dataset (706 answers × expert annotations) and show them to Apertus
as calibration anchors: answer → level → what the expert noticed → what
a good interviewer would ask next. Similarity is a character-trigram
Jaccard, which works across de/fr/it and Swiss-German spellings.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from .data import Dataset, get_dataset


def _grams(text: str, n: int = 3) -> set[str]:
    t = " ".join((text or "").lower().split())
    return {t[i : i + n] for i in range(max(0, len(t) - n + 1))}


def _sim(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class Exemplar:
    answer_id: str
    question: str
    answer: str
    lang: str
    level: int
    answer_class: str
    coach_action: str
    strengths: list[str]
    weaknesses: list[str]
    follow_up: str
    flags: list[str]
    score: float = 0.0

    def to_prompt(self) -> str:
        s = "; ".join(self.strengths[:2]) or "-"
        w = "; ".join(self.weaknesses[:2]) or "-"
        flags = ", ".join(self.flags) or "none"
        return (
            f'Q: "{self.question}"\nA: "{self.answer}"\n'
            f"→ level {self.level} ({self.answer_class}), coach_action={self.coach_action}, flags={flags}\n"
            f"  strengths: {s}\n  weaknesses: {w}\n  expert follow-up: {self.follow_up or '-'}"
        )


class ExemplarIndex:
    def __init__(self, ds: Dataset) -> None:
        self.ds = ds
        self.by_q: dict[str, list[Exemplar]] = defaultdict(list)
        self.by_stage: dict[str, list[Exemplar]] = defaultdict(list)
        self._grams: dict[str, set[str]] = {}
        for aid, ann in ds.annotations.items():
            a = ds.answers.get(aid)
            if not a:
                continue
            ex = Exemplar(
                answer_id=aid,
                question=a.get("question_text") or "",
                answer=a.get("text") or "",
                lang=a.get("lang") or "de",
                level=int(ann.get("quality_level") or a.get("quality_level") or 0),
                answer_class=ann.get("answer_class") or "",
                coach_action=ann.get("coach_action") or "continue",
                strengths=ann.get("strengths") or [],
                weaknesses=ann.get("weaknesses") or [],
                follow_up=ann.get("follow_up_question") or "",
                flags=ann.get("problem_flags") or [],
            )
            self._grams[aid] = _grams(ex.answer + " " + (a.get("normalized_de") or ""))
            qid = a.get("question_id")
            if qid:
                self.by_q[qid].append(ex)
                stage = (ds.questions.get(qid) or {}).get("flow_stage")
                if stage:
                    self.by_stage[stage].append(ex)

    def retrieve(
        self,
        qid: str,
        answer: str,
        lang: str,
        k: int = 3,
        exclude: set[str] | None = None,
    ) -> list[Exemplar]:
        exclude = exclude or set()
        cands = [e for e in self.by_q.get(qid, []) if e.answer_id not in exclude]
        if len(cands) < k:
            stage = (self.ds.questions.get(qid) or {}).get("flow_stage", "")
            extra = [e for e in self.by_stage.get(stage, []) if e.answer_id not in exclude and e not in cands]
            cands = cands + extra
        if not cands:
            return []
        g = _grams(answer)
        scored = []
        for e in cands:
            s = _sim(g, self._grams.get(e.answer_id, set()))
            if e.lang == lang:
                s += 0.05
            if e in self.by_q.get(qid, []):
                s += 0.10
            scored.append((s, e))
        scored.sort(key=lambda x: -x[0])
        # nearest neighbour first, then cover different quality levels (calibration)
        picked: list[Exemplar] = []
        levels: set[int] = set()
        for s, e in scored:
            if len(picked) >= k:
                break
            if not picked or e.level not in levels:
                e.score = round(s, 3)
                picked.append(e)
                levels.add(e.level)
        for s, e in scored:
            if len(picked) >= k:
                break
            if e not in picked:
                picked.append(e)
        return picked


@lru_cache(maxsize=1)
def get_index() -> ExemplarIndex:
    return ExemplarIndex(get_dataset())
