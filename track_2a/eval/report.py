"""Aggregate evaluation results into Markdown tables (+ figures if matplotlib is available).

    python eval/report.py --results eval/results --figures docs/figures
Writes eval/results/RESULTS.md, which is pasted into technical_report.md.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from _common import ROOT, read_json

DIMS = ["realism", "adaptivity", "relevance", "appropriateness", "feedback_quality", "language_consistency"]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _m(vals):
    v = [x for x in (_num(y) for y in vals) if x is not None]
    return round(mean(v), 2) if v else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval/results")
    ap.add_argument("--figures", default="docs/figures")
    args = ap.parse_args()
    res = ROOT / args.results
    fig = ROOT / args.figures
    md: list[str] = ["# Evaluation results", ""]

    # ---- assessment accuracy --------------------------------------------
    for tag, title in [("assess", "SmartStart (retrieval-calibrated)"), ("assess_noretrieval", "Ablation: no exemplar retrieval")]:
        f = res / f"{tag}.json"
        if not f.exists():
            continue
        summ = read_json(f)["summary"]
        md += [f"## Assessment vs. FHGR expert annotations — {title}", "",
               "| Group | n | Level acc | ±1 acc | MAE | Action acc | Follow-up acc | Criteria MAE | Distress recall |",
               "|---|---|---|---|---|---|---|---|---|"]
        for g, m in summ.items():
            if not m:
                continue
            md.append(f"| {g} | {m['n']} | {m['level_acc']} | {m['level_acc_pm1']} | {m['level_mae']} | "
                      f"{m['action_acc']} | {m['followup_acc']} | {m['criteria_mae']} | {m['distress_recall']} |")
        md.append("")

    # ---- simulations + judge --------------------------------------------
    jf = res / "judge.json"
    if jf.exists():
        j = read_json(jf)
        v = j.get("validation")
        if v:
            md += ["## Judge validation (FHGR good vs. poor feedback pairs)", "",
                   f"Judge model: `{j['judge']['model']}` — pairwise accuracy **{v['pairwise_accuracy']}** on {v['pairs']} pairs.", ""]
        rows = j.get("transcripts") or []
        if rows:
            md += ["## Simulated interviews — LLM-as-judge (1-5) and efficiency", "",
                   "| Group | n | Success criteria met | " + " | ".join(DIMS) + " | LLM calls / answer | Guideline violations (before → after rewrite) |",
                   "|---|---|---|" + "---|" * len(DIMS) + "---|---|"]
            groups = defaultdict(list)
            for r in rows:
                groups["all"].append(r)
                groups[f"lang={r['lang']}"].append(r)
                if r.get("difficulty"):
                    groups[f"difficulty={r['difficulty']}"].append(r)
            for g in ["all"] + sorted(k for k in groups if k != "all"):
                rs = groups[g]
                md.append(
                    f"| {g} | {len(rs)} | {_m(r['criteria_met_rate'] for r in rs)} | "
                    + " | ".join(str(_m(r.get(d) for r in rs)) for d in DIMS)
                    + f" | {_m(r['calls_per_answer'] for r in rs)} | "
                    f"{_m(r['violations_first'] for r in rs)} → {_m(r['violations_final'] for r in rs)} |"
                )
            md.append("")
            # consistency = spread across languages
            per_lang = {k: _m(r.get('feedback_quality') for r in v) for k, v in groups.items() if k.startswith("lang=")}
            md += [f"Consistency (feedback quality per language): {per_lang}", ""]
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig.mkdir(parents=True, exist_ok=True)
                langs = sorted(k for k in groups if k.startswith("lang="))
                x = range(len(DIMS))
                w = 0.8 / max(1, len(langs))
                plt.figure(figsize=(9, 3.6))
                for i, lg in enumerate(langs):
                    vals = [_m(r.get(d) for r in groups[lg]) or 0 for d in DIMS]
                    plt.bar([xx + i * w for xx in x], vals, width=w, label=lg.split("=")[1])
                plt.xticks([xx + w * (len(langs) - 1) / 2 for xx in x], [d.replace("_", "\n") for d in DIMS], fontsize=8)
                plt.ylim(0, 5)
                plt.ylabel("judge score (1-5)")
                plt.legend(title="language")
                plt.tight_layout()
                plt.savefig(fig / "judge_by_language.png", dpi=160)
                md += ["![Judge scores by language](docs/figures/judge_by_language.png)", ""]
            except Exception as e:  # matplotlib optional
                md += [f"(figure skipped: {e})", ""]

    out = res / "RESULTS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
