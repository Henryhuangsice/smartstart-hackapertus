"""Experiment 1 — assessment accuracy against FHGR expert annotations.

For a stratified sample of annotated candidate answers (de / fr / it / Swiss
German dialects) we run the coach's `assess` step (leave-one-out: the answer's
own annotation is never retrieved as an exemplar) and compare with the expert:

  * overall quality level (1-4): exact accuracy, ±1 accuracy, MAE
  * coach action (continue / probe / redirect / support_and_refer): accuracy,
    and binary "needs follow-up" accuracy (probe|redirect vs. continue)
  * per-criterion score MAE
  * distress detection recall (problem flag `distress_signal` / support_and_refer)

Run with --no-retrieval for the ablation.

    python eval/eval_assess.py --per-lang 25 --out eval/results
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from collections import defaultdict
from statistics import mean

from _common import ROOT, lang_group, write_json  # noqa: F401

from smartstart import coach
from smartstart.data import get_dataset
from smartstart.planner import default_posting


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=25, help="answers sampled per language group")
    ap.add_argument("--no-retrieval", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="eval/results")
    args = ap.parse_args()

    ds = get_dataset()
    rng = random.Random(args.seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for aid, a in ds.answers.items():
        if aid in ds.annotations and a.get("question_id") in ds.questions:
            groups[lang_group(a)].append(aid)
    sample: list[str] = []
    for g in sorted(groups):
        ids = sorted(groups[g])
        # stratify by quality level so every level is represented
        by_lvl = defaultdict(list)
        for i in ids:
            by_lvl[ds.annotations[i]["quality_level"]].append(i)
        picked: list[str] = []
        while len(picked) < min(args.per_lang, len(ids)):
            for lvl in sorted(by_lvl):
                if by_lvl[lvl] and len(picked) < args.per_lang:
                    picked.append(by_lvl[lvl].pop(rng.randrange(len(by_lvl[lvl]))))
        sample += picked

    rows = []
    t0 = time.time()
    for n, aid in enumerate(sample, 1):
        a = ds.answers[aid]
        ann = ds.annotations[aid]
        lang = a["lang"] if a["lang"] in ("de", "fr", "it") else "de"
        s = coach.new_session("", lang=lang, posting_id=default_posting(ds, lang), ds=ds)
        try:
            res = coach.assess(ds, s, a["question_id"], a.get("question_text") or "", a["text"],
                               use_retrieval=not args.no_retrieval, exclude={aid})
        except Exception as e:  # keep going on endpoint errors
            print(f"[{n}] {aid} ERROR {e}")
            continue
        gold_scores = ann.get("criteria_scores") or {}
        crit_err = [abs(res["scores"][k] - v) for k, v in gold_scores.items()
                    if v and res["scores"].get(k)]
        rows.append({
            "answer_id": aid,
            "group": lang_group(a),
            "gold_level": ann["quality_level"],
            "pred_level": res["quality_level"],
            "gold_action": ann["coach_action"],
            "pred_action": res["coach_action"],
            "gold_flags": ann.get("problem_flags") or [],
            "pred_flags": res["problem_flags"],
            "crit_mae": mean(crit_err) if crit_err else None,
            "parsed": res["raw_ok"],
        })
        print(f"[{n}/{len(sample)}] {aid} {lang_group(a)} gold={ann['quality_level']} pred={res['quality_level']} "
              f"{ann['coach_action']}/{res['coach_action']}  ({time.time() - t0:.0f}s)")

    def summarise(rs: list[dict]) -> dict:
        if not rs:
            return {}
        fu = lambda x: x in ("probe", "redirect")  # noqa: E731
        distress_gold = [r for r in rs if "distress_signal" in r["gold_flags"] or r["gold_action"] == "support_and_refer"]
        cm = [r["crit_mae"] for r in rs if r["crit_mae"] is not None]
        return {
            "n": len(rs),
            "level_acc": round(mean(r["gold_level"] == r["pred_level"] for r in rs), 3),
            "level_acc_pm1": round(mean(abs(r["gold_level"] - r["pred_level"]) <= 1 for r in rs), 3),
            "level_mae": round(mean(abs(r["gold_level"] - r["pred_level"]) for r in rs), 3),
            "action_acc": round(mean(r["gold_action"] == r["pred_action"] for r in rs), 3),
            "followup_acc": round(mean(fu(r["gold_action"]) == fu(r["pred_action"]) for r in rs), 3),
            "criteria_mae": round(mean(cm), 3) if cm else None,
            "distress_recall": round(mean(("distress_signal" in r["pred_flags"]) or r["pred_action"] == "support_and_refer"
                                          for r in distress_gold), 3) if distress_gold else None,
            "json_parse_rate": round(mean(r["parsed"] for r in rs), 3),
        }

    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    summary = {"overall": summarise(rows), **{g: summarise(v) for g, v in sorted(by_group.items())}}
    tag = "assess_noretrieval" if args.no_retrieval else "assess"
    out = ROOT / args.out if not args.out.startswith("/") else args.out
    write_json(Path(out) / f"{tag}.json", {"summary": summary, "rows": rows, "settings": vars(args)})
    print(f"\n== {tag} ==")
    for g, m in summary.items():
        print(g, m)


if __name__ == "__main__":
    main()
