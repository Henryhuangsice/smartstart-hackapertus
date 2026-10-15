"""Offline tests (LLM_BASE_URL=mock). Run: make test"""

import os

os.environ.setdefault("LLM_BASE_URL", "mock")

from smartstart import coach  # noqa: E402
from smartstart.data import get_dataset  # noqa: E402
from smartstart.langdetect import detect  # noqa: E402
from smartstart.planner import build_plan, match_posting  # noqa: E402
from smartstart.retrieval import get_index  # noqa: E402
from smartstart.rules import check_feedback  # noqa: E402

DS = get_dataset()


def test_language_detection():
    assert detect("Bonjour, je voudrais faire un apprentissage")[0] == "fr"
    assert detect("Ciao, sono molto contento di essere qui")[0] == "it"
    lang, gsw, _ = detect("Hoi zäme, ich bi de Luca und mached gern Sport")
    assert lang == "de" and gsw


def test_plan_follows_fhgr_flow():
    plan = build_plan(DS, posting_id="P-11")
    assert plan[0] == "Q-01-01" and plan[-1] == "Q-10-01"
    assert "Q-09-01" in plan  # candidate questions
    assert any(q.startswith("Q-BF") for q in plan)  # occupation-specific question
    assert 10 <= len(plan) <= 16


def test_scenario_plan_is_used():
    assert build_plan(DS, posting_id="P-01", scenario_id="S-01") == DS.scenarios["S-01"]["planned_question_ids"]


def test_posting_matching():
    pid = match_posting(DS, "Ich möchte eine Lehre als Informatiker machen", "de")
    assert "Informatik" in DS.postings[pid]["occupation"]["name_de"]


def test_retrieval_leave_one_out():
    aid = next(iter(DS.annotations))
    a = DS.answers[aid]
    ex = get_index().retrieve(a["question_id"], a["text"], a["lang"], exclude={aid})
    assert ex and aid not in {e.answer_id for e in ex}


def test_rule_checker():
    bad = "Das war perfekt! Du bist eben schüchtern. Andere Jugendliche machen das besser."
    v = " ".join(check_feedback(bad, "de"))
    assert "T3" in v and "B3" in v and "L4" in v and "K1" in v
    good = (
        "## Deine Stärken\nDu hast gesagt: «ich baue gern Lego Technic». Das zeigt Interesse an Technik.\n"
        "## Daran kannst du arbeiten\nBei «weiss nicht» fehlt noch ein Beispiel. Erzähl von der Schnupperlehre.\n"
        "## Dein nächster Schritt\nSchreib dir zwei Beispiele auf und übe sie laut.\n"
        "Das war eine gute Übung. Bleib dran!"
    )
    assert check_feedback(good, "de") == []


def test_full_interview_under_call_budget():
    for first in ["Grüezi, ich bin Luca", "Bonjour, je suis Léa", "Ciao, sono Marco"]:
        s = coach.new_session(first)
        coach.step(s, first)
        i = 0
        while not s["finished"] and i < 40:
            coach.step(s, ["ja", "Ich habe in der Schnupperlehre viel gelernt und gerne mitgearbeitet."][i % 2])
            i += 1
        assert s["finished"]
        if s["lang"] == "de":  # the mock backend only writes German feedback
            assert s["feedback_violations"] == []
        assert coach.calls_per_answer(s) < 5  # FHGR efficiency gate
