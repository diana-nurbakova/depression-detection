# MentalRiskES 2026 — LLM Configuration Reference

**Team:** INSALyon
**Date:** 2026-05-12
**Purpose:** complete inventory of every LLM provider, model checkpoint, sampling parameter, retry policy, and runtime constant used across the submitted system and the post-hoc Gemma branch. Sourced from `config/*.yaml`, `src/mentalriskes/llm_client.py`, the post-hoc runner CLI defaults, and the data-preparation simulator.

This document is a paper-appendix reference for reproducibility. For an account of *why* each setting was chosen, see [docs/mentalriskes_task1_solution_description.md](mentalriskes_task1_solution_description.md), [docs/mentalriskes_task2_solution_description.md](mentalriskes_task2_solution_description.md), and [docs/mentalriskes_gemma_branch_description.md](mentalriskes_gemma_branch_description.md).

---

## 1. Submitted Task 1 system (live evaluation, 2026-04-13 → 2026-04-20)

| Setting | Value | Notes |
|---|---|---|
| Provider | `huggingface` (HF Inference API, serverless) | live submission [config/mentalriskes.yaml](../config/mentalriskes.yaml) at submission time |
| Model checkpoint | `meta-llama/Llama-3.3-70B-Instruct` | full FP16, not the Turbo / FP8 variant |
| Temperature | `0.1` | identical for all 3 runs (Run 0 A5-T3, Run 1 A3-T2, Run 2 A1-T2) |
| Top-p | not set — provider default (HF: 1.0) | no override anywhere in the pipeline |
| Max new tokens | `8192` | CompACT-10 CoT requires ~5–6K output tokens |
| Stop sequences | none | system relies on JSON parser; no stop strings |
| Random seed | not set — provider default | non-deterministic; HF Inference API does not accept a seed parameter |
| Streaming | no | uses `InferenceClient.chat_completion` (non-streaming) |
| `response_format` / JSON mode | not requested | parser tolerates Markdown fences and prose-array fallbacks |
| Timeout | 180 s | per call |
| Retry policy | 5 attempts, exponential back-off | `HFInferenceClient.max_retries=5`, `rate_limit_delay=2.0` |
| HTTP 429 back-off | `max(10, 2^(attempt+1))` s | longer than other transient errors |
| Other transient back-off | `2^attempt` s | catches `ConnectionError`, `Timeout`, stream truncation |
| Fallback chain | TogetherAI `meta-llama/Llama-3.3-70B-Instruct-Turbo` | enabled in `combined_server.py` when `--fallback` (default `true`) |
| Concurrency / parallelism | 1 (sequential per-instrument per-session) | `pipeline.parallel_assessment: false` |
| LLM calls per round (Run 2) | 3 (PHQ-9 + GAD-7 + CompACT-10 assessors) | A1-T2 substrate: anchors only, no Level B / C |
| LLM calls per round (Run 1) | 3 + 0–2 conditional Level B violations are non-LLM rules | A3-T2 substrate: rules don't add LLM calls |
| LLM calls per round (Run 0) | 3 + 1 conditional Level C agent call when triggered | A5-T3 substrate: agent gated on Level B violations or CompACT total > expected + 5 |

---

## 2. Submitted Task 2 system (live evaluation, 2026-04-13 → 2026-04-20)

| Setting | Value | Notes |
|---|---|---|
| Provider | `huggingface` | live submission [config/mentalriskes_task2.yaml](../config/mentalriskes_task2.yaml) at submission time |
| Model checkpoint | `meta-llama/Llama-3.3-70B-Instruct` | non-Turbo |
| Temperature | `0.1` | identical for all 3 runs |
| Top-p | not set | |
| Max new tokens | `4096` | smaller than Task 1 (no long CompACT-10 CoT) |
| Stop sequences | none | |
| Random seed | not set | |
| Streaming | no | |
| `response_format` / JSON mode | not requested | parser handles fences, Spanish/English key variants, accent-insensitive matching |
| Timeout | 300 s | per call (longer than Task 1 because of permutation voting in Run 0) |
| Retry policy | 5 attempts, exponential back-off | same `HFInferenceClient` defaults |
| Fallback chain | TogetherAI `Llama-3.3-70B-Instruct-Turbo` | |
| Run 0 config | `B / FUNC / PERM / W3` | permutation voting → ×3 selection calls per round |
| Run 1 config | `B / FUNC / FIX / W3` | ×2 calls per round (state update + selection) |
| Run 2 config | `B+ / HYB / FIX / W3` | ×3 calls per round (state + characterisation + selection) |
| Lookback window | `W3` — last 3 rounds full transcript injected | longer windows tested and rejected (W1: −10pp, W5: −5pp) |
| Permutation voting orderings (Run 0) | 3: `(1,2,3)`, `(2,3,1)`, `(3,1,2)` | majority vote with original-order tiebreak |
| Calibration tiebreaker | **disabled** | tested in [output/mentalriskes_task2/ablation/B+_*_es_FUNC_FIX_W3_CAL.jsonl](../output/mentalriskes_task2/ablation/B+_Llama-3.3-70B-Instruct-Turbo_es_FUNC_FIX_W3_CAL.jsonl), rejected (−5.6 pp accuracy on trial) |

---

## 3. Full-test replay (2026-04-30 → 2026-05-01, DeepInfra)

Used for the truncation-bug fix re-runs (Layer 0) and the W cross-cohort analyses. **Same prompts as the submitted system**; only the provider changed to handle the 82-round workload that HF was rate-limiting.

| Setting | Task 1 replay | Task 2 replay |
|---|---|---|
| Provider | `deepinfra` (OpenAI-compatible, streaming SSE) | `deepinfra` |
| Model checkpoint | `meta-llama/Llama-3.3-70B-Instruct` | `meta-llama/Llama-3.3-70B-Instruct` |
| Base URL | `https://api.deepinfra.com/v1/openai` | same |
| Temperature | `0.1` | `0.1` |
| Top-p | not set | not set |
| Max new tokens | `8192` | `4096` |
| Stop sequences | none | none |
| Random seed | not set | not set |
| Streaming | yes (SSE with `stream_options.include_usage=true`) | yes |
| `response_format` / JSON mode | not requested | not requested |
| Timeout (read / connect) | 180 s read clamped to `max(180, 300)` = 300 s effective / 30 s connect | 300 s read / 30 s connect |
| Retry policy | 3 attempts, exponential back-off `2^attempt` s | same |
| Catches | HTTP 429, 5xx, `ConnectionError`, `Timeout`, `ChunkedEncodingError`, `IncompleteRead`, `ProtocolError`, `RemoteDisconnected` | same |
| Rate-limit delay | 0.3 s between calls | 0.3 s |
| Config file | [config/mentalriskes_test_replay.yaml](../config/mentalriskes_test_replay.yaml) | [config/mentalriskes_task2_test_replay.yaml](../config/mentalriskes_task2_test_replay.yaml) |
| Output location | `output/mentalriskes_test_replay/` | `output/mentalriskes_task2_test_replay/` |

---

## 4. Gemma GAD-7 post-hoc (Arm A of the Gemma branch)

Specs: [specs/MentalRiskES/gemma_gad7_prompt_spec.md](../specs/MentalRiskES/gemma_gad7_prompt_spec.md) (v1), [gemma_gad7_prompt_v2.md](../specs/MentalRiskES/gemma_gad7_prompt_v2.md) (v2). Runner: [analysis/MentalRiskES_test/posthoc_P_gemma_gad7.py](../analysis/MentalRiskES_test/posthoc_P_gemma_gad7.py).

| Setting | Value | Notes |
|---|---|---|
| Provider | `openrouter` (OpenAI-compatible) via the official `openai` Python SDK | |
| Base URL | `https://openrouter.ai/api/v1` | |
| Model checkpoints | `google/gemma-3-27b-it`, `google/gemma-4-31b-it`, `google/gemma-4-26b-a4b-it` (MoE) | paid endpoints, not `:free`; free-tier hits upstream Google AI Studio 429 cap at 20 req / min |
| Temperature | `0.1` | hard-coded in `client.chat.completions.create(...)` |
| Top-p | not set — provider default | |
| Max new tokens | `1500` (default `--max-tokens`); `1800` for v2 runs | reflects severe-anxiety reasoning |
| Stop sequences | none | |
| Random seed | not set — provider default | |
| Streaming | no (single-shot completion) | |
| `response_format` / JSON mode | `{"type": "json_object"}` requested **only for non-Gemma models** | Gemma on Google AI Studio backend rejects it ("Developer instruction is not enabled"); parser strips Markdown fences as fallback |
| Message structure (Gemma) | single user message with system prompt merged in | Google AI Studio rejects separate `system` role |
| Message structure (non-Gemma) | `[{role:system,content:...}, {role:user,content:...}]` | standard OpenAI format |
| Timeout | OpenAI SDK default (~600 s) | not overridden |
| Retry policy | 5 attempts (full runs) / 3 (smoke tests) | manual loop, not OpenAI SDK retries |
| Per-attempt back-off | `min(60, 5 × attempt)` s | hard cap at 60 s |
| Rate-limit delay | 2.0 s between calls (paid tier) / 3.5 s (free tier, ≈ 17 req/min ceiling) | |
| Auto-resume | on | skips `(session, round)` pairs already in `raw.jsonl` |
| Prompt versions | `v1`, `v2` | selected via `--prompt-version {v1,v2}` CLI flag |
| Cohort flag | `--cohort {test,trial,simulated}` | output dir is suffixed with cohort label |

Headline numbers (test set, n = 10 sessions, final-round per session):

| Model / prompt | GAD-7 MAE_items | Signed total bias | Band accuracy |
|---|---|---|---|
| Gemma 3 27B v1 | 0.814 | −4.9 | 0.30 |
| Gemma 3 27B v2 | 0.743 | −3.8 | 0.20 |
| Gemma 4 31B v1 | 0.786 | −4.7 | 0.30 |
| Gemma 4 26B MoE v1 | 0.743 | −4.0 | 0.20 |
| **Gemma 4 26B MoE v2** | **0.714** | **−3.4** | **0.50** |

---

## 5. Task 2 bare-LLM post-hoc (Arm B of the Gemma branch)

Spec: [specs/MentalRiskES/task2_improvement_spec.md](../specs/MentalRiskES/task2_improvement_spec.md). Runner: [analysis/MentalRiskES_test/posthoc_S_task2_bare_llm.py](../analysis/MentalRiskES_test/posthoc_S_task2_bare_llm.py).

| Setting | Value | Notes |
|---|---|---|
| Provider | `openrouter` via `openai` SDK | |
| Base URL | `https://openrouter.ai/api/v1` | |
| Model checkpoints | `google/gemma-4-31b-it` (headline), `google/gemma-3-27b-it`, `meta-llama/llama-3.3-70b-instruct` | |
| Temperature | `0.1` | hard-coded |
| Top-p | not set | |
| Max new tokens | varies by mode: **S / S2** = 200, **R2** = 400, **S3** = 200 per perm × 6, **S4** = 80 per pairwise call | reflects expected output length |
| Stop sequences | none | |
| Random seed | not set | |
| `response_format` / JSON mode | `{"type": "json_object"}` for non-Gemma; omitted for Gemma (same Google AI Studio constraint) | |
| Message structure | same Gemma-specific merging as Arm A | |
| Timeout | OpenAI SDK default | |
| Retry policy | 4 attempts, `min(60, 5 × attempt)` s back-off | catches all exceptions in the inner loop |
| Rate-limit delay | 2.0 s | |
| Auto-resume | on | |

Mode definitions:

| Mode | LLM calls / round | Description |
|---|---|---|
| `S` | 1 | bare 3-way "trust your clinical intuition" prompt |
| `S2` | 1 | bare + 4-line anti-bias guardrails block ("don't prefer longer responses", "don't always pick option 2", "sometimes the simplest validation is best", "what a skilled therapist would actually say") |
| `R2` | 1 | full 3-way ranking, then take rank-1 |
| `S3` | 6 | all 6 permutations of `(option_1, option_2, option_3)`, majority vote mapped back to original numbering |
| `S4` | 3 | pairwise comparisons `(A vs B, B vs C, A vs C)`, Condorcet winner (ties broken by most wins) |

Headline numbers (test set, n = 568 patient-rounds):

| Mode (model) | Accuracy | Macro F1 | Pred dist (1/2/3) |
|---|---|---|---|
| Submitted Run 2 (HYB B+) | 0.247 | — | 20 / 45 / 35 |
| **S2** Gemma 4 31B | **0.470** | 0.454 | 53 / 23 / 24 |
| S Gemma 4 31B | 0.412 | 0.402 | 48 / 27 / 25 |
| S3 Gemma 4 31B | 0.400 | 0.399 | 37 / 34 / 30 |
| S4 Gemma 4 31B | 0.354 | 0.353 | 36 / 35 / 29 |
| R2 Gemma 3 27B (rank-1 pick) | 0.287 | 0.225 | 15 / 77 / 7 |
| S Gemma 3 27B | 0.290 | 0.263 | 56 / 23 / 21 |
| S Llama-3.3-70B | 0.257 | 0.234 | 54 / 29 / 16 |

---

## 6. Pre-submission ablation runs (offline, TogetherAI)

Used to motivate run selection. The ablation JSONLs in [output/mentalriskes_task2/ablation/](../output/mentalriskes_task2/ablation/) and [runs/mentalriskes_ablation/](../runs/mentalriskes_ablation/) were generated here.

| Setting | Value | Notes |
|---|---|---|
| Provider | `together` (OpenAI-compatible, streaming SSE) | |
| Base URL | `https://api.together.xyz/v1` | |
| Primary model | `meta-llama/Llama-3.3-70B-Instruct-Turbo` (FP8-quantised) | the variant available on Together |
| Comparison model | `claude-sonnet-4-20250514` | only P1_C2 and P1_C4 (Task 2 ablation) |
| Temperature | `0.1` | |
| Top-p | not set | |
| Max new tokens | `4096` (Task 2) / `8192` (Task 1) | |
| Stop sequences | none | |
| Random seed | not set | |
| Streaming | yes (SSE) | |
| Timeout (read / connect) | 180–300 s read / 30 s connect | |
| Retry policy | 3 attempts, exponential back-off | `LLMClient.max_retries=3`, `rate_limit_delay=0.5` |
| Rate-limit delay | 0.5 s | |

**Important reading note for the paper:** the pre-submission ablation JSONLs (e.g. `B_Llama-3.3-70B-Instruct-Turbo_es_FUNC_PERM_W3.jsonl`) record predictions made by the **Turbo / FP8** variant — those are the configuration-selection evidence. The actually submitted runs in `output/mentalriskes_task2/server_submissions/round*_run{R}.json` were generated by the **non-Turbo / FP16** variant via HuggingFace. Small numerical differences between the two variants on individual rounds are expected.

---

## 7. Synthetic-persona simulator (data preparation)

Generator for the simulated cohort. Source: [src/mentalriskes/data_prep/simulator.py](../src/mentalriskes/data_prep/simulator.py).

| Setting | Value | Notes |
|---|---|---|
| Provider | `openai` interface against TogetherAI endpoint | |
| Model checkpoint | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | |
| Temperature | `0.7` | **higher than inference (0.1)** — explicit choice to introduce naturalistic variation in patient turns |
| Top-p | not set | |
| Max new tokens | `512` | turn-length cap |
| Random seed | **set via `--seed N`** (defaults to `None`) | controls option shuffling, distractor sampling, and the `session_id` randomiser; only Python-side randomness, not LLM completions |
| Timeout | 60 s | |
| Retry policy | 3 attempts, exponential back-off | |

Personas generated: 6 (Task 1) + 7 (Task 2) — see [output/mentalriskes/data_prep/simulated/](../output/mentalriskes/data_prep/simulated/) and the persona inventory table in §4 of the Task 1 / Task 2 solution descriptions.

---

## 8. Cross-cutting `LLMClient` behaviour

Defaults baked into the base client class ([src/mentalriskes/llm_client.py](../src/mentalriskes/llm_client.py)). Anything not overridden per-config falls back here.

| Default | Value |
|---|---|
| `LLMClient.temperature` | `0.1` |
| `LLMClient.max_tokens` | `4096` |
| `LLMClient.max_retries` | `3` |
| `LLMClient.rate_limit_delay` | `1.0` s (overridden per provider: 0.3 DeepInfra, 0.5 Together) |
| `LLMClient.timeout` | `180` s (clamped to `max(timeout, 300)` for read in the streaming path) |
| `HFInferenceClient.max_retries` | `5` |
| `HFInferenceClient.rate_limit_delay` | `2.0` s |
| HTTP 429 back-off | `max(10, 2^(attempt+1))` s before retry |
| Other transient back-off | `2^attempt` s |
| Streaming providers | `ollama`, `together`, `deepinfra` (SSE `stream_options.include_usage=true`) |
| Non-streaming providers | `openai`-named generic + `huggingface` |
| Stream truncation error handling | catches `ChunkedEncodingError`, `IncompleteRead`, `ProtocolError`, `RemoteDisconnected`, `ConnectionResetError`, then retries |
| JSON parser | tolerates ```code fences, embedded JSON via outermost-brace bracket matching, truncated JSON via close-bracket repair (`parse_json_response` in `llm_client.py`) |

---

## 9. Non-LLM runtime parameters

| Setting | Value | Where it lives |
|---|---|---|
| `--max-rounds` (live server loop, pre-fix until 2026-04-30) | `30` (the truncation bug) | [src/mentalriskes/combined_server.py](../src/mentalriskes/combined_server.py) — the cause of the missing R31–82 in the official submission |
| `--max-rounds` (live server loop, post-fix) | `200` with hard-error exit if the cap fires while the server is still serving | [src/mentalriskes/combined_server.py](../src/mentalriskes/combined_server.py) |
| Lookback window (Task 2 transcript injection) | `W3` (last 3 rounds full transcript) | submitted config; ablation rejected W1, W5 |
| Permutation voting orderings (submitted Task 2 Run 0) | 3: `(1,2,3)`, `(2,3,1)`, `(3,1,2)` | submission config |
| Permutation orderings (post-hoc Task 2 S3) | 6: all permutations of `(1,2,3)` | bare-LLM post-hoc only |
| Wasserstein anomaly threshold (T2 / T3 aggregation) | `W1 > μ + 2σ` discards round (PHQ-9 / GAD-7 only); flagged but retained for CompACT-10 | Task 1 |
| T2 early-decay weight | rounds 1–5 carry `2.0`, rounds 6+ carry `1.0` | Task 1 |
| T3 stability threshold | per-item `std < 0.5` → use last-round prediction; else fall back to T2 | Task 1 CompACT-10 in Run 0 (A5-T3) |
| Level B rules | 7 cross-instrument consistency rules (C1–C7) | Task 1; see [docs/mentalriskes_task1_solution_description.md §2.3](mentalriskes_task1_solution_description.md) |
| Level C agent gating | invoked iff Level B violation (severity = high or medium) OR CompACT-10 total > expected ceiling + 5 | Task 1 Run 0 only |
| Self-contradiction guard (Level B rule C4) | OtE mean < 2.5 → suppress VA correction | Task 1; preserves the 19 % clinical pattern from Chinese LPA study |

---

## 10. What is *not* set anywhere

To be explicit about negatives:

- **Top-p / nucleus sampling:** never overridden. Every provider uses its default (HF / TogetherAI / DeepInfra / OpenRouter all default to `1.0`).
- **Random seed (LLM):** never set. HF / TogetherAI / DeepInfra all expose a `seed` parameter but our client doesn't pass one. Outputs are non-deterministic at temperature 0.1. **The simulator's `--seed` flag only controls Python-side shuffles**, not LLM completions.
- **Stop sequences:** never set anywhere. All termination is via `max_tokens` or the model's natural EOS.
- **Frequency / presence penalty:** not set.
- **`logit_bias`:** not set.
- **Tool / function calling:** not used.
- **Vision / multimodal:** not used.
- **System fingerprint / determinism flags:** not used.

---

## 11. Quick lookup — which config produced which artefact?

| Artefact | LLM / provider | Reference §  |
|---|---|---|
| [output/mentalriskes/predictions/round{N}_run{R}.json](../output/mentalriskes/predictions/) (R1–30) | Llama-3.3-70B-Instruct, HF Inference API | §1 |
| [output/mentalriskes_task2/server_submissions/round{N}_run{R}.json](../output/mentalriskes_task2/server_submissions/) (R1–30) | Llama-3.3-70B-Instruct, HF Inference API | §2 |
| [output/mentalriskes_test_replay/predictions/](../output/mentalriskes_test_replay/predictions/) (R1–82) | Llama-3.3-70B-Instruct, DeepInfra | §3 |
| [output/mentalriskes_task2_test_replay/server_submissions/](../output/mentalriskes_task2_test_replay/server_submissions/) (R1–82) | Llama-3.3-70B-Instruct, DeepInfra | §3 |
| [output/mentalriskes_gemma_gad7/{model}__{vN}{__cohort}/](../output/mentalriskes_gemma_gad7/) | Gemma 3/4, OpenRouter | §4 |
| [output/mentalriskes_task2_bare_llm/{model}__{mode}{__cohort}/](../output/mentalriskes_task2_bare_llm/) | Gemma 3/4 or Llama-3.3-70B, OpenRouter | §5 |
| [output/mentalriskes_task2/ablation/*.jsonl](../output/mentalriskes_task2/ablation/) | Llama-3.3-70B-Instruct-Turbo, TogetherAI | §6 |
| [output/mentalriskes_task2/simulated_ablation/*/sim_*.jsonl](../output/mentalriskes_task2/simulated_ablation/) | Llama-3.3-70B-Instruct-Turbo, TogetherAI | §6 |
| [runs/mentalriskes_ablation/A*_rounds.jsonl](../runs/mentalriskes_ablation/) | Llama-3.3-70B-Instruct, TogetherAI (mixed history; see file timestamps) | §6 |
| [output/mentalriskes/data_prep/simulated/](../output/mentalriskes/data_prep/simulated/) | Llama-3.3-70B-Instruct-Turbo, TogetherAI | §7 |

---

## 12. Reproducibility notes

- All outputs are **non-deterministic** because no seed is passed to the LLM. Re-running the same prompt against the same model is expected to produce slightly different results (token-level differences at temperature 0.1; same overall ranking expected on the aggregate metrics).
- The Llama-3.3-70B-Instruct checkpoint is the same FP16 weights on HuggingFace and DeepInfra; results between the two providers should be numerically equivalent up to floating-point noise. The TogetherAI Turbo variant is FP8-quantised and produces measurably different outputs on a small fraction of rounds.
- For exact reproduction of any specific number in our SUMMARY tables, point the relevant runner at the exact provider / model / config combination listed in §1–§7 and accept the random variation. The submitted predictions are archived as-is in their respective `output/mentalriskes*/server_submissions/` and `output/mentalriskes*/predictions/` directories.
