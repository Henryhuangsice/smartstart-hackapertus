# Evaluation results

## Assessment vs. FHGR expert annotations — SmartStart (retrieval-calibrated)

| Group | n | Level acc | ±1 acc | MAE | Action acc | Follow-up acc | Criteria MAE | Distress recall |
|---|---|---|---|---|---|---|---|---|
| overall | 100 | 0.65 | 0.99 | 0.36 | 0.83 | 0.87 | 0.455 | 1 |
| de | 25 | 0.52 | 1 | 0.48 | 0.76 | 0.76 | 0.567 | None |
| fr | 25 | 0.72 | 1 | 0.28 | 0.84 | 0.84 | 0.42 | 1 |
| gsw | 25 | 0.68 | 0.96 | 0.36 | 0.84 | 0.92 | 0.486 | None |
| it | 25 | 0.68 | 1 | 0.32 | 0.88 | 0.96 | 0.347 | 1 |

## Assessment vs. FHGR expert annotations — Ablation: no exemplar retrieval

| Group | n | Level acc | ±1 acc | MAE | Action acc | Follow-up acc | Criteria MAE | Distress recall |
|---|---|---|---|---|---|---|---|---|
| overall | 100 | 0.59 | 0.97 | 0.45 | 0.44 | 0.79 | 0.544 | 0 |
| de | 25 | 0.48 | 1 | 0.52 | 0.36 | 0.72 | 0.617 | None |
| fr | 25 | 0.68 | 1 | 0.32 | 0.44 | 0.84 | 0.533 | 0 |
| gsw | 25 | 0.56 | 0.92 | 0.56 | 0.52 | 0.8 | 0.6 | None |
| it | 25 | 0.64 | 0.96 | 0.4 | 0.44 | 0.8 | 0.427 | 0 |

## Judge validation (FHGR good vs. poor feedback pairs)

Judge model: `swiss-ai/Apertus-v1.5-8B` — pairwise accuracy **0.897** on 29 pairs.

## Simulated interviews — LLM-as-judge (1-5) and efficiency

| Group | n | Success criteria met | realism | adaptivity | relevance | appropriateness | feedback_quality | language_consistency | LLM calls / answer | Guideline violations (before → after rewrite) |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 28 | 1.0 | 4.96 | 4.96 | 4.96 | 4.96 | 4.96 | 5.0 | 2.12 | 0.07 → 0.04 |
| difficulty=easy | 5 | 1.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 2.13 | 0.2 → 0.0 |
| difficulty=hard | 5 | 1.0 | 4.8 | 4.8 | 4.8 | 4.8 | 4.8 | 5.0 | 2.11 | 0.0 → 0.0 |
| difficulty=medium | 18 | 1.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 2.11 | 0.06 → 0.06 |
| lang=de | 20 | 1.0 | 4.95 | 4.95 | 4.95 | 4.95 | 4.95 | 5.0 | 2.12 | 0.1 → 0.05 |
| lang=fr | 5 | 1.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 2.11 | 0.0 → 0.0 |
| lang=it | 3 | 1.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 2.11 | 0.0 → 0.0 |

Consistency (feedback quality per language): {'lang=de': 4.95, 'lang=fr': 5.0, 'lang=it': 5.0}

(figure skipped: No module named 'matplotlib')
