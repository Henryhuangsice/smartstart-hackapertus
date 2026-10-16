"""Experiment 4 — consistency (FHGR judging: 25 %).

(a) Cross-language invariance: German expert-annotated answers (stratified by level) are translated
    by Apertus into French and Italian (content, length and register kept); the coach then assesses
    the same content in de / fr / it. We report how often level and next action stay the same, and the
    accuracy against the expert label per language.
(b) Repeat stability: every assessment is run twice with identical settings; we report the share of
    runs whose level differs by <= 1 and whose action is identical.

    python eval/eval_consistency.py --n 15 --out eval/results_v3
"""

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from _common import ROOT, write_json  # noqa: F401

from smartstart import coach, llm
from smartstart.data import get_dataset
from smartstart.planner import default_posting

LANGS = {"fr": "French", "it": "Italian"}
TRANSLATE = (
    "Translate the following interview answer of a 15-year-old Swiss apprenticeship candidate into {lang}. "
    "Keep the content, length, informal register and any weaknesses (vagueness, missing examples) exactly; "
    "do not improve or shorten it. Output only the translation.\n\nANSWER: \"{text}\""
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="eval/results_v3")
    args = ap.parse_args()
    ds = get_dataset()
    rng = random.Random(args.seed)
    by_lvl = defaultdict(list)
    for aid, a in sorted(ds.answers.items()):
        if a.get("lang") == "de" and aid in ds.annotations and a.get("question_id") in ds.questions:
            by_lvl[ds.annotations[aid]["quality_level"]].append(aid)
    sample: list[str] = []
    while len(sample) < args.n and any(by_lvl.values()):
        for lvl in sorted(by_lvl):
            if by_lvl[lvl] and len(sample) < args.n:
                sample.append(by_lvl[lvl].pop(rng.randrange(len(by_lvl[lvl]))))

    rows, t0 = [], time.time()
    for n, aid in enumerate(sample, 1):
        a, ann = ds.answers[aid], ds.annotations[aid]
        texts = {"de": a["text"]}
        for lg, name in LANGS.items():
            texts[lg] = llm.chat([{"role": "user", "content": TRANSLATE.format(lang=name, text=a["text"])}],
                                 temperature=0.0, max_tokens=300, tag="translate").strip().strip('"')
        row = {"answer_id": aid, "gold_level": ann["quality_level"], "gold_action": ann["coach_action"],
               "texts": texts, "runs": {}}
        for lg, txt in texts.items():
            q = ds.question_text(a["question_id"], lg) or a.get("question_text") or ""
            runs = []
            for _ in range(2):
                s = coach.new_session("", lang=lg, posting_id=default_posting(ds, lg), ds=ds)
                r = coach.assess(ds, s, a["question_id"], q, txt, exclude={aid})
                runs.append({"level": r["quality_level"], "action": r["coach_action"]})
            row["runs"][lg] = runs
        rows.append(row)
        print(f"[{n}/{len(sample)}] {aid} gold={ann['quality_level']}/{ann['coach_action']} "
              + " ".join(f"{lg}={row['runs'][lg][0]['level']}/{row['runs'][lg][0]['action']}" for lg in texts)
              + f"  ({time.time() - t0:.0f}s)")

    fu = lambda x: x in ("probe", "redirect")  # noqa: E731
    lv = lambda x: x or 0  # noqa: E731
    first = {lg: [r["runs"][lg][0] for r in rows] for lg in ("de", "fr", "it")}
    per_lang = {lg: {
        "level_acc": round(mean(lv(x["level"]) == r["gold_level"] for x, r in zip(first[lg], rows)), 3),
        "action_acc": round(mean(x["action"] == r["gold_action"] for x, r in zip(first[lg], rows)), 3),
        "followup_acc": round(mean(fu(x["action"]) == fu(r["gold_action"]) for x, r in zip(first[lg], rows)), 3),
    } for lg in first}
    cross = {}
    for lg in ("fr", "it"):
        pairs = list(zip(first["de"], first[lg]))
        cross[f"de_vs_{lg}"] = {
            "same_level": round(mean(lv(a["level"]) == lv(b["level"]) for a, b in pairs), 3),
            "level_within_1": round(mean(abs(lv(a["level"]) - lv(b["level"])) <= 1 for a, b in pairs), 3),
            "same_action": round(mean(a["action"] == b["action"] for a, b in pairs), 3),
            "same_followup_decision": round(mean(fu(a["action"]) == fu(b["action"]) for a, b in pairs), 3),
        }
    reps = [r["runs"][lg] for r in rows for lg in ("de", "fr", "it")]
    repeat = {
        "n_pairs": len(reps),
        "level_diff_le1": round(mean(abs(lv(x[0]["level"]) - lv(x[1]["level"])) <= 1 for x in reps), 3),
        "same_level": round(mean(lv(x[0]["level"]) == lv(x[1]["level"]) for x in reps), 3),
        "same_action": round(mean(x[0]["action"] == x[1]["action"] for x in reps), 3),
    }
    accs = [v["followup_acc"] for v in per_lang.values()]
    summary = {"n": len(rows), "per_language": per_lang, "cross_language": cross, "repeat": repeat,
               "followup_acc_gap_pp": round(100 * (max(accs) - min(accs)), 1)}
    out = ROOT / args.out if not args.out.startswith("/") else Path(args.out)
    write_json(Path(out) / "consistency.json", {"summary": summary, "rows": rows, "settings": vars(args)})
    print("\n== consistency ==")
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
