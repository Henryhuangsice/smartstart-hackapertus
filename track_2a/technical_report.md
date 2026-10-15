# Technical report — SmartStart: a rubric-grounded Apertus interview coach

- **Track:** Track 2A — FHGR: AI-Powered Job Interview Coach
- **Event:** Online (1–16 October 2026)
- **Team:** SmartStart — Chu Jia (University of Zurich, MA Computational Linguistics)
- **Demo:** `make run` + Agent Chat UI (http://localhost:2026, graph `agent`); screenshots in `docs/`

## 1. Summary

Young people preparing for their first apprenticeship interview rarely get repeated,
individual practice. SmartStart is an Apertus-1.5-8B interview coach that runs a realistic,
adaptive interview in German, French or Italian (Swiss-German dialect input understood),
then gives personal, guideline-compliant feedback. Instead of one large prompt, we combine
a **deterministic interview state machine**, **retrieval of FHGR expert annotations as
calibration anchors**, a **two-call-per-answer LLM loop (assess → interview)** and a
**rule-based checker for the FHGR feedback guidelines** that triggers a single rewrite only
when needed. The system runs fully on consumer hardware (MacBook Pro M4 Pro, MLX; or one
< 32 GB GPU with vLLM) and needs **≈ 2.1 LLM calls per candidate answer** (limit < 5).
Headline results: see §5 (`eval/results/RESULTS.md`).

## 2. Architecture

![Architecture](docs/architecture.png)

| Component | File | LLM calls |
|---|---|---|
| LangGraph graph served by Aegra (`agent`, port 2026), session state persisted per thread | `src/main.py` | – |
| Language & dialect detection (stop-word / Mundart-marker heuristic) | `smartstart/langdetect.py` | 0 |
| Interview planner: FHGR flow (introduction → motivation → strengths/weaknesses → situational → candidate questions → closing); occupation-specific questions via `berufsfeld`; conditional questions (Stellwerk, absences) only if the profile justifies them; scenario plans when a `scenario_id` is given; free-text occupation wish matched to a posting | `smartstart/planner.py` | 0 |
| Exemplar retrieval over 706 expert-annotated answers (char-trigram Jaccard, same question first, diverse quality levels, leave-one-out in evaluation) | `smartstart/retrieval.py` | 0 |
| **Assess**: JSON with rubric levels (1-4 / null), evidence quote, strength, improvement, coach action (`continue / probe / redirect / support_and_refer`), problem flags, follow-up question, coaching tip | `smartstart/coach.py` | 1 per answer |
| Deterministic policy: ≤ 1 follow-up per question, ≤ 4 per interview; candidate questions answered from posting facts only; distress ⇒ support mode (L3) | `smartstart/coach.py` | 0 |
| **Interviewer** turn in the persona of the posting's Berufsbildner/in, du/tu | `smartstart/coach.py` | 1 per answer |
| **Final feedback** (strengths → improvements with quotes → one next step → encouragement) | `smartstart/coach.py` | 1 per interview |
| **Guideline checker** T3, B3, L2, L4, K1, R1/R3, S3, S4 (+ length) ⇒ **rewrite once** if violated | `smartstart/rules.py` | 0–1 per interview |
| Rubric overview table for all 11 criteria; "no evidence" instead of guessing (K3) | `smartstart/coach.py` | 0 |

Two feedback modes: `end` (realistic interview, feedback at the end; default) and
`per_answer` (adds the assessment's one-line tip after each answer — no extra call).

## 3. Use of Apertus

- **Model:** `swiss-ai/Apertus-v1.5-8B` (bf16)
- **How it is used:** inference — assessment (structured JSON), interviewer role-play,
  feedback generation and rewrite; also as candidate simulator and (default) judge in the evaluation.
- **Where it runs:** local weights. Development: MacBook Pro M4 Pro (48 GB unified memory)
  with `mlx_lm.server` (OpenAI-compatible). Alternative: `docker compose --profile gpu`
  (vLLM, single consumer GPU, `--max-model-len 8192`). The coach only needs
  `LLM_BASE_URL / LLM_API_KEY / LLM_NAME` ⇒ on-premise deployment, no data leaves the school.
- **Prompts:** `src/smartstart/prompts.py`. Instructions are English (most reliable
  instruction following), all user-facing text in the interview language; temperature 0
  for assessment, 0.3–0.4 for generation; fixed seed.

## 4. Data

FHGR/Profolio challenge data (provided for Hack Apertus; see `data/fhgr/README.md`):
242 questions (+296 Swiss-German variants), 706 answers at 4 quality levels in de/fr/it and
8 dialects with expert annotations, the 11-criterion rubric, 27 feedback guidelines,
24 candidate profiles, 30 postings, 28 scenarios with coach success criteria, 15 reference
transcripts and 30 good-vs-poor feedback pairs. All content is synthetic or didactic; no
personal data. Occupation data © SDBB / berufsberatung.ch. No other data was used.

## 5. Evaluation

The official LLM-as-judge benchmark is hidden, so we built three proxy experiments
(`make eval`; all numbers reproducible with the fixed seeds):

1. **Assessment accuracy vs. expert annotations** (`eval/eval_assess.py`): stratified sample
   per language group (de, fr, it, Swiss-German dialects), leave-one-out retrieval.
   Metrics: quality-level accuracy / ±1 / MAE, coach-action accuracy, follow-up decision
   accuracy, criterion MAE, distress recall. **Ablation:** without exemplar retrieval.
2. **Simulated interviews** (`eval/simulate.py`): Apertus plays the 28 FHGR scenario
   candidates (profile + `simulation_persona`, dialect where expected) against the coach.
3. **LLM-as-judge** (`eval/judge.py`): per transcript, pass/fail for each scenario
   `success_criteria_for_coach` and 1–5 ratings for realism, adaptivity, relevance,
   appropriateness, feedback quality, language consistency. The judge is first
   **validated** on the 30 FHGR good-vs-poor feedback pairs (pairwise accuracy).

<!-- RESULTS:START — replaced by eval/results/RESULTS.md -->
| Setup | Metric | Result |
|---|---|---|
| Efficiency gate | LLM calls per candidate answer | ≈ 2.1 (mock run; final numbers below) |
| Assessment, no retrieval (baseline) | level acc / ±1 / follow-up acc | _tbd_ |
| Assessment, with retrieval (ours) | level acc / ±1 / follow-up acc | _tbd_ |
| Simulated interviews (28) | success criteria met / judge scores | _tbd_ |
| Consistency | judge scores per language (de / fr / it) | _tbd_ |
<!-- RESULTS:END -->

## 6. Limitations

- Text only; speech (TTS/ASR) is prepared for (short turns, plain text output) but not integrated.
- Dialect: understood on input; the coach answers in Standard German (S4), dialect questions
  only from the curated FHGR variants (`dialect` option).
- Judge = same model family as the coach by default (self-preference bias); judge
  validation on the FHGR pairs quantifies its reliability. Simulated candidates are less
  varied than real teenagers. The FHGR data is synthetic and not yet expert-reviewed.
- The rule checker is lexical; it catches clear violations, not subtle tone problems.

## 7. Reproducibility

`make test` (offline), `make smoke` (all 28 scenarios with a mock LLM),
`make run` (Docker: Aegra + Postgres + Redis, Apertus via `LLM_BASE_URL`),
`make eval` (needs an Apertus endpoint). Hardware used: MacBook Pro M4 Pro, 48 GB, macOS;
Apertus 1.5 8B bf16 via mlx-lm. Seeds: `LLM_SEED=42`, sampling seed 13.

## 8. Next steps

Speech in/out with Swiss-German ASR; fine-tuning a LoRA assessor on the expert annotations;
teacher dashboard over the stored session rubric levels; user study with real classes.

## License

Code Apache-2.0; report CC-BY-4.0. FHGR data under the FHGR / Hack Apertus conditions.

## AI assistance

Parts of the code and documentation were drafted with the help of an AI coding assistant
(Claude) and reviewed, run and evaluated by the author.

## References

- FHGR / Profolio: *Feedbackkriterien zum Vorstellungsgespräch*; challenge datasets (github.com/fhgr/Hackapertus).
- Swiss AI Initiative: Apertus technical report and model cards (huggingface.co/swiss-ai).
