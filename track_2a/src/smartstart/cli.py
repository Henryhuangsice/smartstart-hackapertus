"""Terminal chat with the coach (no Aegra / Docker needed).

    python -m smartstart.cli --lang de --posting P-01
    python -m smartstart.cli --scenario S-04          # French scenario
"""

from __future__ import annotations

import argparse

from . import coach
from .data import get_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["de", "fr", "it"])
    ap.add_argument("--posting")
    ap.add_argument("--candidate")
    ap.add_argument("--scenario")
    ap.add_argument("--dialect")
    ap.add_argument("--feedback-mode", choices=["end", "per_answer"])
    args = ap.parse_args()
    ds = get_dataset()
    first = input("Du / Toi / Tu > ").strip() or "Hallo"
    s = coach.new_session(first, lang=args.lang, posting_id=args.posting, candidate_id=args.candidate,
                          scenario_id=args.scenario, dialect=args.dialect, feedback_mode=args.feedback_mode, ds=ds)
    print(f"[setting: lang={s['lang']} posting={s['posting_id']} plan={len(s['plan'])} questions]\n")
    print("Coach >", coach.step(s, first, ds=ds), "\n")
    while not s["finished"]:
        try:
            msg = input("Du > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        print("\nCoach >", coach.step(s, msg, ds=ds), "\n")
    print(f"[LLM calls: {s['llm_calls']}, answers: {s['answers']}, calls/answer: {coach.calls_per_answer(s)}]")


if __name__ == "__main__":
    main()
