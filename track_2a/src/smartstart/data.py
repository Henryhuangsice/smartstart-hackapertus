"""Loading of the FHGR / Profolio challenge datasets (cached, read-only)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import SETTINGS

LANGS = ("de", "fr", "it")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class Dataset:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or SETTINGS.data_dir)
        r = self.root
        self.questions = {q["id"]: q for q in _jsonl(r / "questions" / "questions.jsonl")}
        self.answers = {a["id"]: a for a in _jsonl(r / "answers" / "answers.jsonl")}
        self.annotations = {a["answer_id"]: a for a in _jsonl(r / "annotations" / "annotations.jsonl")}
        self.candidates = {c["id"]: c for c in _jsonl(r / "profiles" / "candidates.jsonl")}
        self.postings = {p["id"]: p for p in _jsonl(r / "profiles" / "postings.jsonl")}
        self.scenarios = {s["id"]: s for s in _jsonl(r / "scenarios" / "scenarios.jsonl")}
        self.feedback_examples = _jsonl(r / "feedback_examples" / "feedback_examples.jsonl")
        self.criteria = json.loads((r / "rubric" / "criteria.json").read_text(encoding="utf-8"))
        self.guidelines = json.loads((r / "rubric" / "feedback_guidelines.json").read_text(encoding="utf-8"))
        self.criteria_by_key = {c["key"]: c for c in self.criteria["criteria"]}

    # ---- convenience -------------------------------------------------
    def question_text(self, qid: str, lang: str, dialect: str | None = None) -> str:
        q = self.questions[qid]
        if dialect:
            for v in q.get("gsw_variants") or []:
                if v.get("dialect") == dialect:
                    return v["text"]
        texts = q.get("text") or {}
        return texts.get(lang) or texts.get("de") or next(iter(texts.values()))

    def criterion_label(self, key: str, lang: str) -> str:
        c = self.criteria_by_key.get(key)
        if not c:
            return key
        return c["label"].get(lang) or c["label"].get("en") or key

    def level_label(self, value: int | None, lang: str) -> str:
        if value is None:
            return self.criteria["not_observed"].get(lang) or self.criteria["not_observed"]["en"]
        for s in self.criteria["scale"]:
            if s["value"] == value:
                return s.get(lang) or s["en"]
        return str(value)

    def rubric_text(self, keys: list[str]) -> str:
        """Compact English rubric for the given criteria (prompt material)."""
        out = []
        for k in keys:
            c = self.criteria_by_key.get(k)
            if not c:
                continue
            lv = "; ".join(f"{l['value']}={l['description']['en']}" for l in c["levels"])
            out.append(f"- {k}: {c['what_is_assessed']['en']} | {lv}")
        return "\n".join(out)

    def guidelines_text(self) -> str:
        rules = []
        for g in self.guidelines["groups"]:
            for r in g["rules"]:
                rules.append(f"{r['id']}: {r['en']}")
        return "\n".join(rules)

    def occupation_name(self, posting: dict, lang: str) -> str:
        occ = posting.get("occupation") or {}
        return occ.get(f"name_{lang}") or occ.get("name_de") or "Lehrstelle"


@lru_cache(maxsize=1)
def get_dataset() -> Dataset:
    return Dataset()
