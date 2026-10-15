"""Tiny, dependency-free language + Swiss-German dialect detector.

No LLM call is spent on language identification: a stop-word / marker
heuristic is accurate enough for interview answers and keeps us well
below the "< 5 LLM calls per answer" budget.
"""

from __future__ import annotations

import re

_WORDS = {
    "de": """ich du der die das und ist nicht ein eine mit auf für dass wir sie es zu im den dem
    mein meine bin habe hat aber auch weil wenn sehr gerne möchte schule lehre beruf danke ja nein hallo guten tag grüezi""",
    "fr": """je tu le la les et est pas un une avec pour que nous vous il elle au du des mon ma
    suis ai mais aussi parce très bien aime voudrais école apprentissage métier merci oui non bonjour salut bonsoir madame monsieur""",
    "it": """io tu il lo la le e è non un una con per che noi voi lui lei al del della mio mia
    sono ho ma anche perché molto bene piace vorrei scuola apprendistato mestiere grazie sì no ciao buongiorno salve buonasera signora signore""",
    "en": """i you the and is not a an with for that we they it to my am have has but also
    because very like would school job thank yes no""",
}
# Swiss German markers (Mundart) — written dialect is very distinctive.
_GSW = """isch nöd nid gsi gsii öppis chli chlii mer häsch hesch hani han gha ha gseh luege lueg
mached macht's chunnt chumm gaht gäng immer gern gärn üs öis eus mini mis dini dis ufem uf
au no nüt nüd öpper hüt geschter morn bitzli halt eifach scho schaffe schaffä gfunde gfalle
wo ich schnuppere schnupperlehr lehrstell gmacht gsehe chönd chönnt würd wäri"""

_VOCAB = {k: set(v.split()) for k, v in _WORDS.items()}
_GSW_SET = set(_GSW.split())
_TOKEN = re.compile(r"[a-zA-ZÀ-ÿ']+")


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def detect(text: str, default: str = "de") -> tuple[str, bool, float]:
    """Return (lang, is_swiss_german_dialect, confidence 0..1)."""
    toks = tokens(text)
    if not toks:
        return default, False, 0.0
    scores = {k: sum(t in v for t in toks) for k, v in _VOCAB.items()}
    gsw = sum(t in _GSW_SET for t in toks)
    # dialect counts towards German
    scores["de"] += gsw * 0.8
    best = max(scores, key=scores.get)
    total = sum(scores.values()) or 1
    conf = scores[best] / total
    if scores[best] == 0:
        return default, False, 0.0
    is_gsw = best == "de" and gsw >= max(2, 0.15 * len(toks))
    return best, is_gsw, conf
