"""Rule-based compliance checker for the FHGR feedback guidelines.

Checks the generated final feedback against machine-checkable rules
(T3, B3, L2, L4, K1, R3, S3, S4, A1). Only if a violation is found the
feedback is rewritten once (1 extra LLM call), so the average number of
calls stays far below the limit.
"""

from __future__ import annotations

import re

from .langdetect import detect
from .prompts import FEEDBACK_HEADINGS

_SUPERLATIVES = {
    "de": r"\b(perfekt\w*|genial\w*|fehlerlos|grossartig\w*|hervorragend\w*|schlecht\w*|katastroph\w*|gar nicht)\b",
    "fr": r"\b(parfait\w*|génial\w*|géniaux|nul(le)?|catastroph\w*|excellentissime|mauvais\w*)\b",
    "it": r"\b(perfett\w*|genial\w*|pessim\w*|catastrof\w*|scarso|scarsa)\b",
}
_LABELS = {
    "de": r"\b(schüchtern\w*|unmotiviert\w*|introvertiert\w*|faul\w*|unsicherer mensch|unreif\w*)\b",
    "fr": r"\b(timide|démotivé\w*|introverti\w*|paresseu\w*|immature)\b",
    "it": r"\b(timid\w*|demotivat\w*|introvers\w*|pigr\w*|immatur\w*)\b",
}
_FEAR = {
    "de": r"(nie(mals)? eine lehrstelle|keine chance|so wirst du nie)",
    "fr": r"(jamais (de|un) (place|apprentissage)|aucune chance)",
    "it": r"(mai (un|il) (posto|apprendistato)|nessuna possibilità)",
}
_COMPARE = {
    "de": r"(andere(n)? jugendliche|andere bewerber|besser als andere|im vergleich zu anderen)",
    "fr": r"(d'autres jeunes|autres candidat|mieux que les autres|par rapport aux autres)",
    "it": r"(altri ragazzi|altri giovani|altri candidat|meglio degli altri|rispetto agli altri)",
}
_QUOTE = re.compile(r"«[^»]{2,}»|\"[^\"]{2,}\"|“[^”]{2,}”|„[^“]{2,}“")


def check_feedback(text: str, lang: str) -> list[str]:
    """Return a list of human-readable violations (empty = compliant)."""
    v: list[str] = []
    low = (text or "").lower()
    lang = lang if lang in FEEDBACK_HEADINGS else "de"
    if re.search(_SUPERLATIVES[lang], low):
        m = re.search(_SUPERLATIVES[lang], low)
        v.append(f"T3: exaggerated praise or dramatising word '{m.group(0)}' — use neutral wording")
    if re.search(_LABELS[lang], low):
        v.append("B3: personality label / diagnosis — describe behaviour instead")
    if re.search(_FEAR[lang], low):
        v.append("L2: fear-inducing statement — remove it")
    if re.search(_COMPARE[lang], low):
        v.append("L4: comparison with other young people — remove it")
    if len(_QUOTE.findall(text or "")) < 2:
        v.append("K1: back the points with at least two quotes of the candidate's own words in «guillemets»")
    heads = FEEDBACK_HEADINGS[lang]
    for h in heads[:3]:
        if h.lower() not in low:
            v.append(f"R1/R3: missing section heading '## {h}'")
    unquoted = _QUOTE.sub(" ", text or "")
    sents = [s for s in re.split(r"[.!?]\s|\n", re.sub(r"[#*>\-|]", " ", unquoted)) if len(s.split()) >= 3]
    if sents:
        avg = sum(len(s.split()) for s in sents) / len(sents)
        if avg > 22:
            v.append(f"S3: sentences too long (avg {avg:.0f} words) — use short sentences")
    det, _, conf = detect(text or "", default=lang)
    if det != lang and conf > 0.5:
        v.append(f"S4: feedback is not written in the interview language ({lang})")
    if len((text or "").strip()) < 200:
        v.append("R2/A1: feedback is too short or incomplete")
    return v
