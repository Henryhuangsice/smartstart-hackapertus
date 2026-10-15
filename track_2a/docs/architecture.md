# Architecture

```mermaid
flowchart TD
    U([Candidate message]) --> R{session?}
    R -- no --> INIT[init_session<br/>language/dialect detection<br/>posting matching<br/>question plan]
    R -- yes --> T
    INIT --> T[coach_turn]
    T --> Q[current question<br/>from state machine]
    Q --> RET[(Exemplar retrieval<br/>706 expert-annotated answers<br/>char-trigram similarity)]
    RET --> A[["LLM call 1: ASSESS<br/>rubric levels · evidence · coach_action · flags · follow-up"]]
    A --> P{policy<br/>deterministic}
    P -- probe / redirect<br/>budget left --> FU[follow-up question]
    P -- continue --> NX[next planned question]
    P -- candidate asked --> CQ[answer from posting facts]
    FU --> I[["LLM call 2: INTERVIEWER<br/>persona, du/tu, de·fr·it"]]
    NX --> I
    CQ --> I
    I --> OUT([Coach reply])
    P -- plan finished --> F[["LLM call: FINAL FEEDBACK<br/>27 FHGR guidelines"]]
    F --> C{rule checker<br/>T3 B3 L2 L4 K1 R1/R3 S3 S4}
    C -- violations --> RW[["LLM call: REWRITE (once)"]]
    C -- ok --> TAB[rubric overview<br/>11 criteria, K3: no evidence ⇒ not rated]
    RW --> TAB
    TAB --> OUT
```

* **Graph** (`src/main.py`): `START → (init_session) → coach_turn → END`; one run per candidate
  message; the whole session (plan, assessments, LLM-call counter) lives in the LangGraph state
  and is persisted per thread by Aegra's Postgres checkpointer.
* **LLM budget**: 2 calls per answer + 1–2 calls for the final feedback ⇒ ≈ 2.1 calls/answer.
* **Where it runs**: Apertus 1.5 8B locally (MLX on Apple Silicon or vLLM on one consumer GPU);
  the coach container only talks to an OpenAI-compatible URL → on-premise, privacy-preserving.
