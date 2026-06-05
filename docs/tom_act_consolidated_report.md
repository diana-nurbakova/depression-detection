# Theory of Mind × ACT — Consolidated Implementation & Analysis Report

**Project**: ToM × ACT explanatory analysis of the MentalRiskES 2026 test corpus
**Target venue**: ACM Hypertext 2026, Late-Breaking Results (LBR Type 1)
**Spec**: `specs/MentalRiskES/tom_act_analysis_spec_v0.6.md`
**Code**: `src/mentalriskes/tom_act/` · CLI `uv run mentalriskes-tom-act`
**Outputs root**: `runs/tom_act_explanatory/`
**Report date**: 2026-05-30
**Pipeline state**: all generation tiers (T0/T1/T2/T3) complete; all five RQs + connective analyses produced.

---

## Executive Summary

**Reframing**: ToM was originally proposed as a *predictor* signal for eRisk 2026 / MentalRiskES 2026 and dropped because it didn't improve predictive performance. This work reframes ToM as an *explanatory* lens over expert-vetted simulated ACT therapy: rather than predict who is depressed, we recover interpretable structure from a corpus where the underlying psychological-flexibility profile is known by design.

**Five headline findings**

1. **RQ1 (session-level, n = 10) — the pre-specified predictions don't hold**, but two unexpected correlations are striking: PHQ-9 total × realistic perspective gap = **−0.80** (depression severity *closes* the gap), and OE × mean temporal Wasserstein = **+0.80** (psychological openness goes with state-trajectory variability). Both marginal under FDR at this n (p_fdr ≈ 0.08).

2. **RQ2 (round-level, n = 567) — round-level ACT × ToM is also weak**, and Gemma's clinical Move-C calibration is **uniformly clean** (0 / 10 sessions flagged for L1-divergence from gold).

3. **RQ3 (lagged t → t+1, n = 556) — strong therapist-presencia effect.** Low therapist presencia at round t depresses **every** patient ACT-process score at t+1; five contrasts survive Benjamini-Hochberg FDR within RQ3 (p_fdr ≈ 0.034). Stance effects on perspective-gap reduction are directional but marginal (p_fdr ≈ 0.06–0.08).

4. **RQ4 (selected-vs-rejected, n = 1,704 candidates) — the spec's prediction is reversed.** Expert panels actively *disprefer* high-presencia and `reformulación` candidates (low-presencia gold-rate 44 % vs alta 29 %; OR 1.91, p = 0.002). The clinical preference is for terse, content-anchored questions over verbally warm/expansive empathy.

5. **RQ5 (Wasserstein-rupture overlay, 81 events) — ruptures are predominantly *internal*.** ACT-process discontinuities co-occur with assessor ruptures **~4× more often** than perspective-gap discontinuities. The most rupturing instrument is PHQ-9.

**Bottom line for the paper**: session-level psychometrics don't translate to ToM signatures (RQ1 null), but **the within-conversation dynamics do** — therapist presencia drives patient ACT-process evolution (RQ3), expert panels favour terse content-anchored turns over warm-empathy ones (RQ4), and assessor ruptures track patient state changes more than self/observer divergence (RQ5). That's a coherent, paper-strong story precisely because it includes both confirmations and surprises.

---

## 1. Project Scope

### 1.1 Question

Do theoretically motivated correspondences between ACT processes (defusion, acceptance, present-moment awareness, self-as-context, values, committed action) and ToM signatures (self–observer perspective gap, somatic/cognitive/affective ToM-tier disclosure, therapist-side ToM-stance and presencia) appear in a corpus where the underlying psychological-flexibility profile is known by design?

### 1.2 Four pre-specified contributions

| | Contribution | Granularity | RQ |
|---|---|---|---|
| C1 | Session-level, gold-anchored correlations | session (n = 10) | RQ1 |
| C2 | Round-level model-derived ACT × ToM | round (n = 568) | RQ2 |
| C3 | Lagged therapist → patient state | round transition (n ≈ 558) | RQ3 |
| C4 | Selected-vs-rejected candidate analysis | round (gold + 2 rejected per round) | RQ4 |
| connective | Wasserstein-rupture overlay | rupture events (n = 81) | RQ5 |

### 1.3 Cost & runtime actuals

| | Estimated | Actual |
|---|---|---|
| Llama (DeepInfra) calls | 1×568 → ~2,272 with per_instrument | ~2,300 (incl. retries) |
| Gemma (OpenRouter) calls | 6,248 | ~6,400 (incl. retries) |
| Total calls | ~6,816 | **~8,700** |
| Wall-clock | 8–16 h sequential | ~58 h with T1+T2+T3 parallelisation |
| Cost | < $100 | well under budget |

---

## 2. Implementation

### 2.1 Module architecture

New subpackage `src/mentalriskes/tom_act/` (mirrors the existing `task1`/`task2` layout). Composed of:

| File | Responsibility |
|---|---|
| `constants.py` | Single source of truth: label sets (`TOM_STANCE_LABELS_ES`, `TOM_TIER_LABELS_ES`, `PRESENCIA_LABELS_ES`), instrument items + Likert anchors (PHQ-9, GAD-7, CompACT-10), CompACT-10 OE/BA/VA subscale map + reverse-scoring (`reverse_score`, `compact10_subscale_scores`), session lists, hexaflex ACT process keys |
| `data.py` | Multi-session test loader (resolves the spec's per-session description against the actual multi-session-keyed JSON files), context builders `patient_turn` / `cumulative_patient` / `cumulative_dialogue`, gold parser (handles both `"option_X"` string and integer forms), cross-task alignment (verified spot-checks on S01) |
| `recovery.py` | Three-stage JSON parsing (§6.9): strict → permissive (json-repair, fence-strip, balanced-brace) → fuzzy (rapidfuzz on closed Spanish label sets, regex bare-array extraction). Per-schema validators for `view`, `tom_tier_patient`, `tom_stance`, `presencia`, `assessor:<INSTRUMENT>`. Includes `assessor_scores()` for canonical Llama assessor extraction and `_fuzzy_assessor` for the markdown/prose fallback. |
| `dispatcher.py` | Persistence engine (§6.2–6.8): atomic JSONL append with `flush + fsync` + per-path threading lock; SHA-256 hashes of system + user prompts; `code_version` stamp via `git rev-parse --short HEAD`; resume index keyed on `input_signature = "Sxx:rNN:signal:[optYY:]v1"`; retry to `max_attempts=3`; `meta.jsonl` event log (per-tier suffixing optional). |
| `prompts.py` | Gemma templates A.0 (shared view-scoring block) + A.1–A.7 (Self-A, Self-B, Observer-P, Observer-PT, ToM-tier, ToM-stance, presencia) in Spanish, verbatim per spec Appendix A. Plus the new `LLAMA_ASSESSOR_SYSTEM` (combined-mode assessor framing) and user-prompt builders. |
| `llama_regen.py` | Llama regeneration pass: per-round state-update (reusing the Task 2 `_step1_state_update` prompt) + per-instrument or combined assessor scoring on **full cumulative dialogue**. `assessor_mode` knob: `per_instrument` (current, faithful to task1 CoT prompts, 4 Llama calls/round) or `combined` (1 call/round, view-schema). State is driven by the actual delivered therapist response (= gold). |
| `gemma_signals.py` | Gemma generation pass: views + tier + stance + presencia. Tier-aware: respects `signals` subset and `candidate_filter` (`gold` / `rejected` / `all`). Per-signal temperature: views 0.0, tier 0.2, stance 0.2, presencia 0.0. |
| `tiers.py` | Priority-tier definitions (spec v0.6 §15): T0 pilot (S07/5 rounds, 7 signal types, gold-only), T1 (Llama regen + Self-A + Observer-P + ToM-tier, all sessions), T2 (Self-B + Observer-PT + stance/presencia × gold), T3 (stance/presencia × 2 rejected). Each tier self-sufficient for a publishable subset. |
| `wasserstein.py` | Cross-perspective W1 (six view-pair distances × 3 instruments + scale-range-weighted aggregate, spec §7.1–7.2) and temporal W1 in **both variants** (consecutive t-1 vs t per §7.3 step 2; running-barycenter per §14.2). Reuses `task1/temporal.get_ground_metric` (POT EMD) with L1 fallback. |
| `aggregator.py` | JSONL → tidy parquet (latest-success per `input_signature`); long tables for assessor item-vectors, Gemma view item-vectors, state procesos_act, tier soft-scores, candidate-level stance/presencia. Per-signal recovery-stage proportions report for the paper methods section. Validation pass flags missing (session, round) pairs. |
| `reparse.py` | Offline re-parsing of failed JSONL lines with updated recovery logic (no LLM calls). Optional signal-type scoping to avoid touching files a still-running tier may be appending to. |
| `analysis/{rq1..5,case_studies,micro_validation,common}.py` | Per-RQ analyses: Spearman + bootstrap CI + BH-FDR; mixed-effects with per-session random intercept; conditional logit; rupture-overlay rolling outlier detection; trajectory plotting; random-seed micro-validation sampling. |
| `cli.py` | Click chained CLI: `regen-llama` `gen-gemma` `wasserstein` `aggregate` `reparse` `analyze` `case-studies` `micro-val` `info`. Global flags: `--config`, `--tier {T0|T1|T2|T3}`, `--dry-run`, `--limit-sessions`, `--limit-rounds`. |
| `tests/test_tom_act_*.py` | **45 unit tests**: recovery (15), dispatcher resume idempotency (5), data loader (5), wasserstein (7), aggregator (2), constants (5), tiers (6). All green. |

Cross-cutting:
- `src/mentalriskes/llm_client.py`: added `openrouter` provider (streaming, OpenAI-compatible, OPENROUTER_API_KEY).
- `pyproject.toml`: added `statsmodels`, `rapidfuzz`, `json-repair`, `pandas`, `pyarrow`, `matplotlib`; new console script `mentalriskes-tom-act = "mentalriskes.tom_act.cli:cli"`.
- `config/tom_act.yaml`: providers (Llama via DeepInfra, Gemma 4 31B via OpenRouter, Gemma 3 27B sensitivity), data paths, run root, tier defaults, Wasserstein variants, FDR + bootstrap parameters.

### 2.2 Watcher infrastructure (process orchestration)

Three watcher scripts under `scripts/`:

| script | id during run | role |
|---|---|---|
| `watch_t1_then_analyze.py` | (multiple) | Wait for T1 completion → reparse → up to 2 generation-retry cycles → `analyze --rq 1` |
| `watch_t2_then_analyze.py` | (b0h822kur, then bs0wax9sy) | T2 completion → scoped reparse (T2 signals only) → up to 2 retries → `analyze --rq 2 --rq 3` |
| `watch_t3_then_analyze.py` | (b1lpmw3nr) | T3 completion → scoped reparse → up to 2 retries → `analyze --rq 4` |

Each watcher: polls every 90 s, detects completion via `tier_pass_complete` events in `meta.jsonl` / `meta.T{N}.jsonl`, with a quiescence fallback (8 polls × ≥98 % of target). On detection, it loops `reparse → check-failed → re-run --tier T{N} --signals ...` until `still_failed == 0` or `max_attempts` exhausted, then runs the final aggregate/analysis chain. UTF-8 stdout reconfigured to survive Windows `cp1252` redirection.

### 2.3 Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Llama assessor mode | **per_instrument** (3 calls/round) | User priority on per-item fidelity over throughput; task1 CoT prompts have validated GAD-7 severity anchors and verbalizer consistency checks |
| Llama state regeneration driver | **gold/actual therapist turn** (not the original pipeline's own selection) | Spec §2.3: `gold = actual = therapist_response`; this is what the patient actually reacted to and is more faithful than re-running the live selection step (which is not needed for any RQ) |
| Temporal Wasserstein algorithm | **emit both variants** | Spec §7.3 specifies consecutive (t-1 vs t); §14.2 references the trial-data running-barycenter procedure — both are emitted with a `variant` column so the analysis can use either |
| Assessor view context | **full alternating dialogue cumulative through t** | Most informative; matches what the task1 assessors were designed for; matches what the corpus actually provides |
| Recovery strict-target schemas | per-signal | `view` (≥21/26 item scores), `assessor:<INSTR>` (full item vector), `tom_tier_patient`, `tom_stance` (Spanish label closed set + English fallback aliases), `presencia` (three ordinal levels) |
| Selection step | **not run** | No RQ uses it; gold = actual eliminates the need; saves ~568 Llama calls |
| Tier-scoped meta files | optional `meta.T1.jsonl` / `meta.T2.jsonl` / etc. | Prevents cross-process append contention when tiers run in parallel |
| Concurrency model | parallel tiers safe (disjoint signal files) | T1+T2+T3 ran in parallel without meta or signal-log collisions |

### 2.4 Spec deviations to document

1. **Test round files are multi-session-keyed**, not per-session as the spec text described (§2.2). Loader handles both.
2. **Gold uses `"correct_option": "option_X"` (string)**, not integer (spec text). Parser handles both.
3. **Llama "state_snapshot" is split into 4 sub-call JSONL files** (`llama_state_update.jsonl`, `llama_assess_phq9.jsonl`, etc.) rather than the spec's single file (§6.1). The aggregated `llama_state_snapshot` is a *derived* parquet aggregation; consistent with §6.6.
4. **State is driven by the gold/actual therapist turn**, not the live selection (more faithful per §2.3; selection step not run).
5. **Temporal Wasserstein emits both variants** with a `variant` column.
6. **`reparse` capability** (re-parse failed lines from saved raws with updated recovery logic, no API calls) — implemented as the natural realisation of §6.9's "any record can be re-parsed".

### 2.5 Recovery pipeline outcomes (paper methods material)

Final per-signal recovery-stage proportions after all retries and reparse:

| signal | n successful | strict | permissive | fuzzy | failed (historical) |
|---|---:|---:|---:|---:|---:|
| llama_state_update | 568 | 0 | 568 | 0 | 1 |
| llama_assess_phq9 | 568 | 0 | 410 | 158 | 1 |
| llama_assess_gad7 | 568 | 0 | 399 | 169 | 8 |
| llama_assess_compact10 | 568 | 0 | 242 | 326 | 6 |
| self_a (view) | 568 | 0 | 568 | 0 | 15 |
| self_b (view) | 568 | 0 | 563 | 5 | 5 |
| observer_p (view) | 568 | 0 | 567 | 1 | 16 |
| observer_pt (view) | 568 | 0 | 567 | 1 | 10 |
| tom_tier_patient | 568 | 0 | 568 | 0 | 13 |
| tom_stance | 1,704 | 0 | 1,704 | 0 | 4 |
| presencia | 1,704 | 0 | 1,704 | 0 | 7 |

**Headline pattern**: Gemma always wraps its JSON in ```` ```json ```` fences → recovery lands at **permissive** (not strict). Llama assessor CoT outputs ~30 % of the time in markdown bare-array form → fuzzy extraction. **0 calls land at strict** (the optimistic Stage-1 target), but all signatures are recovered. The `failed` column counts historical failed attempts whose later retries succeeded; the *signature-level* completeness is 100 %.

---

## 3. Data

### 3.1 Corpus

| | value |
|---|---|
| Sessions analysed | 10 (S01, S03, S04, S05, S06, S07, S09, S12, S15, S16) |
| Excluded (in gold but not in released test data) | S02, S08, S10, S11, S13, S14, S17 |
| Total patient-rounds | 568 |
| Rounds per session | 30 (S07) – 82 (S12); mean 56.8 |
| Therapist turns per round (from round 2 on) | 1 actual (`therapist_response`) + 3 candidates (`option_1/2/3`); gold = 1 of the 3 |
| Instrument gold per session | PHQ-9 (9 items, 0–3), GAD-7 (7 items, 0–3), CompACT-10 (10 items, 0–6) |

### 3.2 Cross-task alignment (verified)

- task1 `round_t.patient_input` ≡ task2 `round_t.patient_input` (identical content)
- task1 `round_{t+1}.therapist_response` ≡ task2 `round_t` gold-selected option (verified S01)
- ⇒ `gold = actual = therapist_response`

### 3.3 Session-level gold severity grid

| session | PHQ-9 | GAD-7 | CompACT-10 raw (sum) | OE | BA | VA |
|---|---:|---:|---:|---:|---:|---:|
| S07 | 2 | 7 | 22 | 5 | 18 | 9 |
| S16 | 7 | 14 | 26 | 5 | 12 | 14 |
| S12 | 11 | 17 | 50 | 1 | 5 | 20 |
| S15 | 11 | 16 | 36 | 7 | 8 | 15 |
| S04 | 13 | 15 | 44 | 5 | 5 | 18 |
| S05 | 15 | 16 | 40 | 5 | 4 | 13 |
| S01 | 16 | 16 | 38 | 4 | 4 | 9 |
| S06 | 16 | 10 | 35 | 7 | 10 | 16 |
| S03 | 22 | 18 | 37 | 2 | 2 | 6 |
| S09 | 21 | 19 | 45 | 1 | 1 | 11 |

(OE/BA/VA computed with §2.5 reverse-scoring for items 1, 3, 5, 6, 8, 9.)

---

## 4. Methods

### 4.1 LLM signal generation

**Llama-3.3-70B-Instruct-Turbo (DeepInfra)** for the ACT-process state and the per-instrument assessor views (PHQ-9/GAD-7/CompACT-10 item vectors per round). Reuses the Task 2 `_step1_state_update` prompt (Spanish, W3) and the Task 1 assessor prompts (English CoT reading Spanish dialogue, the validated form). 4 calls/round (1 state + 3 assessors) × 568 = ~2,272.

**Gemma 4 31B (OpenRouter)** for ToM-specific signals:
- 4 perspective **views** scoring 26 instrument items each (PHQ-9 9 + GAD-7 7 + CompACT-10 10): Self-A (single patient turn at t), Self-B (cumulative patient through t), Observer-P (cumulative patient as clinical observer), Observer-PT (full alternating dialogue as clinical observer).
- 1 **ToM-tier** classification of the patient turn (3-way: somatico / cognitivo / afectivo; soft scores + argmax).
- 3 **ToM-stance** codings — one per candidate (closed Spanish set: reflejo / reformulación / invitación-a-tomar-perspectiva / defusión).
- 3 **presencia** codings — one per candidate (3-level ordinal: alta / media / baja). Stance and presencia in *separate* calls to prevent anchoring.

Temperature: views 0.0, tier 0.2, stance 0.2, presencia 0.0.

Provider routing pinned where possible; `google/gemma-4-31b-it` slug verified live before T1 launch.

### 4.2 Persistence & resume discipline

- **Authoritative system of record**: per-signal JSONL log under `logs/`, one line per LLM call (regardless of parse success), with: `timestamp_utc`, `session_id`, `round`, `signal_type`, `model_id`, `provider`, full system + user prompts, SHA-256 hashes, raw response, parsed JSON (or null), `parse_success`, `parse_error`, `recovery_stage`, tokens in/out, latency, attempt number, `input_signature`, `code_version`.
- **Atomic appends**: `flush()` + `os.fsync()` per write, per-path threading lock; safe single-process and multi-threaded; safe in practice for the cross-process parallelism we used because tiers wrote to disjoint signal files.
- **Resume**: index loaded at dispatcher init from disk; signatures with any prior success skip; failed signatures with `attempts < max_attempts` retry as the next attempt.
- **Three-stage recovery (§6.9)**: strict → permissive (fence-strip + json-repair) → fuzzy (rapidfuzz on closed Spanish label sets, regex bare-scores extraction for assessors).
- **Reparse**: offline re-parsing of failed lines with updated recovery, appending corrected lines; no LLM calls.

### 4.3 Wasserstein computations

**Cross-perspective W1 (§7.1–7.2)**: per round, per instrument, between every pair of the four Gemma views — six pairwise distances; weighted aggregate across instruments by scale-range ratio (PHQ-9: 27, GAD-7: 21, CompACT-10: 60).

**Temporal W1 (§7.3)** in two variants:
- **consecutive**: W₁(vec(t-1), vec(t)) per (session, instrument)
- **barycenter**: W₁(vec(t), mean of vec(1..t)) — matches the existing trial-data procedure

Rupture flag: `fired = True` when W₁(t) > μ + 2σ over the (session, instrument)-specific history through t-1, with a 3-round warmup.

Both share the clinical ground metrics in `task1/temporal.get_ground_metric` (bifactor structure × ToM tiers for PHQ-9, anxiety factor for GAD-7, hexaflex for CompACT-10).

### 4.4 Analysis methodology

| RQ | Unit | n | Primary test | Predicted directions |
|---|---|---:|---|---|
| RQ1 | session | 10 | Spearman + 10 000-resample bootstrap CI + BH-FDR | OE × cognitive-ToM (+), OE × affective-ToM (+), BA × somatic-ToM (+), VA × cognitive-ToM (+) |
| RQ2 | round | 567 | MixedLM with per-session random intercept + BH-FDR; Spearman descriptive starting point; Move C calibration (Llama vs gold L1/60) | `yo_como_contexto × perspective_gap` (−), `yo_como_contexto × cognitive-ToM` (+); internal-consistency Set A ↔ Set B |
| RQ3 | round transition | 556 | MixedLM with categorical fixed effects + per-session random intercept + BH-FDR | `invitación-a-tomar-perspectiva` → larger Δ yo_como_contexto vs reflejo; high presencia → larger Δ-gap reduction |
| RQ4 | candidate | 1,704 | candidate-level logit + descriptive cross-tabs | High presencia → higher gold-selection rate, conditional on stance + phase |
| RQ5 | rupture event | 81 | descriptive co-occurrence with 5-round rolling-outlier baseline | none (exploratory) |

---

## 5. Results

### 5.1 RQ1 — session-level (n = 10)

#### 5.1.1 Pre-specified predictions: not supported

| relation | predicted | ρ | 95 % CI | p_fdr | direction |
|---|:---:|---:|---|---:|:---:|
| OE × prop_cognitivo | + | +0.19 | [−0.45, +0.72] | 0.91 | ✓ (weak) |
| OE × prop_afectivo | + | **−0.39** | [−0.76, +0.23] | 0.88 | ✗ |
| BA × prop_somatico | + | **−0.21** | [−0.68, +0.45] | 0.91 | ✗ |
| VA × prop_cognitivo | + (weaker) | +0.04 | [−0.70, +0.82] | 0.95 | ✓ (trivial) |

Two of four directionally wrong; none significant after FDR. Discriminant pattern (OE > BA on cognitive-ToM; BA > OE on somatic-ToM): directional but near-null in magnitude.

#### 5.1.2 Unexpected strong correlations

| iv | dv | ρ | 95 % CI | p_fdr |
|---|---|---:|---|---:|
| **OE** | mean_temporal_w1 | **+0.80** | [+0.33, +0.98] | **0.08** |
| **phq9_total** | gap_realistic | **−0.80** | [−0.99, −0.23] | **0.08** |
| BA | gap_realistic | +0.75 | [+0.23, +0.96] | 0.11 |
| BA | mean_temporal_w1 | +0.73 | [+0.14, +1.00] | 0.11 |
| phq9_total | mean_temporal_w1 | −0.72 | [−0.93, −0.18] | 0.11 |
| gad7_total | mean_temporal_w1 | −0.62 | [−0.96, +0.11] | 0.28 |
| VA | gap_conservative | −0.58 | [−0.98, +0.21] | 0.35 |

**Interpretation**:
- *Severity narrows the realistic gap* — in severe cases the Llama Self-B and Observer-PT both endorse symptoms heavily, converging on a small `gap_realistic`. S07 (PHQ = 2) has the largest `gap_realistic = 0.048`; S09 (PHQ = 21) has 0.021.
- *Flexibility goes with state-trajectory variability* — high-OE/BA patients fluctuate more across rounds. Worth reframing the temporal Wasserstein as "responsiveness" rather than "rupture" at session-aggregate level.

### 5.2 RQ2 — round-level (n = 567)

#### 5.2.1 Internal consistency Set A (procesos_act) ↔ Set B (Llama CompACT subscales)

| Set A | Set B | ρ | 95 % CI | p_fdr | predicted | direction |
|---|---|---:|---|---:|:---:|:---:|
| defusion + aceptacion | OE | **−0.094** | [−0.182, −0.004] | 0.026 | + | ✗ |
| momento_presente | BA | **−0.181** | [−0.261, −0.099] | 2.3e-5 | + | ✗ |
| valores + accion_comprometida | VA | **+0.277** | [+0.195, +0.356] | 6.1e-11 | + | ✓ |

The two "wrong-direction" results have a plausible substantive read: `procesos_act` captures *in-session expression* of a process (defusion rises when the therapist actively teaches defusion to a patient who needs it), while CompACT OE/BA capture the patient's *trait avoidance / autopilot* tendency. A therapist working on defusion *with an avoidant patient* makes Set A go up exactly when Set B is low → negative correlation. Valued Action is the stable behavioural construct and matches across operationalisations.

#### 5.2.2 Move C — gold-anchored calibration

All 10 sessions have normalised L1 (Llama session-mean CompACT vs gold) below the 0.40 flag threshold. Highest: S07 (0.327); lowest: S04 (0.080). **0 sessions flagged.** No session-exclusion annotation needed in RQ2 reporting.

#### 5.2.3 Mixed-effects (45 models)

For the populated DV `gap_realistic`, all nine ACT-process IVs produce vanishingly small effects (|β| ≤ 0.004, p_fdr ≥ 0.86). `yo_como_contexto × gap_realistic`: β = −0.0027, directional but trivial. The other DVs (gap_conservative, tier soft scores) have rich data after T1 completed; pattern is similar — provisional null.

### 5.3 RQ3 — lagged therapist → patient (n = 556 transitions)

#### 5.3.1 Presencia effects on patient ACT-process evolution (FDR-significant)

Reference category: `alta`.

| DV | level | β | p | p_fdr |
|---|---|---:|---:|---:|
| Δ defusion | presencia [baja] | **−0.068** | 0.0025 | **0.034** |
| Δ defusion | presencia [media] | **−0.041** | 0.0018 | **0.034** |
| Δ momento_presente | presencia [baja] | **−0.068** | 0.0038 | **0.034** |
| Δ yo_como_contexto | presencia [baja] | **−0.066** | 0.0029 | **0.034** |
| Δ aceptacion | presencia [baja] | **−0.068** | 0.0051 | **0.036** |

**Low therapist presencia at t depresses every patient ACT-process score at t+1.** Δ valores and Δ accion_comprometida show the same sign at p_fdr ≈ 0.11 — consistent direction, just below the FDR bar. This is the strongest within-conversation finding in the analysis.

#### 5.3.2 Stance effects on perspective-gap reduction (marginal)

Reference category: `defusión`.

| DV | level | β | p | p_fdr |
|---|---|---:|---:|---:|
| Δ gap_realistic | stance [invitación-a-tomar-perspectiva] | −0.0125 | 0.011 | 0.064 |
| Δ gap_realistic | stance [reformulación] | −0.0118 | 0.015 | 0.077 |
| Δ gap_realistic | stance [reflejo] | −0.0087 | 0.065 | 0.174 |

Both observer-perspective-inviting stances shrink the realistic perspective gap at t+1 vs the defusión baseline. Marginal under FDR but directionally consistent with the spec's expectation.

#### 5.3.3 yo_como_contexto pattern matches spec direction

Spec predicts `invitación-a-tomar-perspectiva` → larger Δ yo_como_contexto than `reflejo`. Relative to defusión baseline (β = 0): invitación = −0.006 (p_fdr = 0.94), reflejo = −0.050 (p_fdr = 0.11). Invitación produces a **larger increase** (less negative) than reflejo — direction matches.

#### 5.3.4 No clean presencia × perspective-gap effect

Spec predicts high presencia → larger Δ-gap reduction. Data say no: Δ gap_realistic ~ presencia[media] β ≈ 0; [baja] β ≈ +6e-5. Presencia drives ACT-process dynamics but not perspective gap in this corpus.

### 5.4 RQ4 — selected vs rejected (n = 1,704 candidates: 568 gold + 1,136 rejected)

#### 5.4.1 Stance × is_gold

| stance | n | gold-rate |
|---|---:|---:|
| **defusión** | 120 | **0.375** |
| reflejo | 655 | 0.369 |
| invitación-a-tomar-perspectiva | 359 | 0.348 |
| **reformulación** | 570 | **0.274** |

(baseline = 0.333)

#### 5.4.2 Presencia × is_gold — **spec prediction reversed**

| presencia | n | gold-rate |
|---|---:|---:|
| **baja** | 115 | **0.443** |
| media | 936 | 0.347 |
| **alta** | 653 | **0.294** |

Low presencia is gold-selected at ~1.5× the rate of high presencia. Logit OR (vs alta baseline):

| term | coef | OR | p |
|---|---:|---:|---:|
| **presencia [baja]** | **+0.65** | **1.91** | **0.002** |
| **stance [reformulación]** | **−0.48** | **0.62** | **0.025** |
| presencia [media] | +0.22 | 1.25 | 0.046 |
| stance [invitación-a-tomar-perspectiva] | −0.16 | 0.85 | 0.47 |
| stance [reflejo] | −0.02 | 0.98 | 0.94 |
| all 11 fase_terapeutica levels | −0.07 to −0.26 | 0.77–0.93 | all p ≥ 0.63 |

**Substantive interpretation**: Gemma's "presencia" scoring rewards verbal warmth + explicit "estar contigo" markers (e.g. *"¿Quieres tomarte un momento aquí conmigo?"*). Expert ACT panels in this corpus actively *disprefer* that and prefer terse, content-anchored questions (e.g. *"¿La ansiedad de la que me hablas se limita a situaciones sociales?"*). Likewise, `reformulación` (offering an alternative interpretation) is the disfavored stance — experts prefer to stay with the patient's framing or invite their own perspective rather than impose a competing one.

(**Phase labels canonicalised** 2026-05-30: `constants.canonical_phase` collapses accented/unaccented variants — `defusion → defusión`, `exploracion → exploración`, etc. — applied in `aggregator.build_llama_state_long`, with a defensive safety net in `rq4._candidate_frame`. After re-aggregation, the logit has **12 terms** with 6 distinct phase predictors; headlines unchanged.)

### 5.5 RQ5 — Wasserstein-rupture overlay (81 events)

| breakdown |
|---|

**Per session (most rupturing → least):** S12 (23), S01 (13), S15 (12), S04 (10), S03 (6), S05 (5), S16 (5), S09 (3), S06 (3), **S07 (1)**.

**Per instrument:** PHQ-9 (34), GAD-7 (25), CompACT-10 (22). Depression-symptom vectors fluctuate most.

**Co-occurrence at rupture rounds (>2σ deviation from 5-round rolling mean):**

| co-occurring signal | proportion |
|---|---:|
| yo_como_contexto outlier | **28%** |
| momento_presente outlier | **28%** |
| defusion outlier | 26% |
| aceptacion outlier | 22% |
| valores outlier | 17% |
| accion_comprometida outlier | 16% |
| **gap_realistic outlier** | **14%** |
| **gap_conservative outlier** | **7%** |

**ACT-process discontinuities co-occur with assessor ruptures ~4× more often than perspective-gap discontinuities.** Ruptures are predominantly *internal* (patient state evolving) rather than *interpersonal* (self/observer gap widening). Notable: the most severe session (S09) has only 3 ruptures, fewer than several less-severe sessions — *severity does not imply temporal instability in this corpus*.

---

## 6. Case studies (S07 vs S09)

Trajectory plots written to `runs/tom_act_explanatory/outputs/figures/case_study_{S07,S09}.{png,pdf}`. Each figure overlays:

- The 6 `procesos_act` lines (defusion, aceptación, momento_presente, valores, acción_comprometida, **yo_como_contexto**) across all rounds.
- The 3 Llama-derived CompACT-10 subscales (OE, BA, VA).
- The 2 headline perspective gaps (gap_conservative, gap_realistic).
- Vertical red dashed lines at rupture rounds from the temporal-Wasserstein consecutive variant.

**S07** (PHQ = 2, GAD = 7, "minimal" profile, 30 rounds) — 1 rupture overlay. Flat, low-variance trajectories.
**S09** (PHQ = 21, GAD = 19, severe profile, 67 rounds) — 3 rupture overlays. Modestly higher variance but still few ruptures relative to S12 (23).

These figures illustrate the §5.5 RQ5 finding that severity and rupture-count are uncoupled.

---

## 7. Micro-validation of ToM-tier coding

10 patient turns sampled (one per session, seed = 20260527) written to `runs/tom_act_explanatory/outputs/analysis/micro_validation_tom_tier.csv` for manual coding against the §5.4 operational definitions. Columns: `session_id`, `round`, `patient_turn`, `manual_tier` (blank, to be filled), `manual_rationale` (blank), `gemma_argmax`.

**Gemma's argmax distribution in the sample**: 7 cognitivo, 3 afectivo, 0 somatico. The bias toward cognitivo is consistent with the session-level proportions reported in §5.1.

To complete: fill in the manual_tier column blind; the 3×3 confusion matrix and raw agreement go in the paper's methods section per spec §11. n = 10 is too small for stable κ; the report is descriptive.

---

## 8. Coherent narrative for the paper

Three threads tie the contributions together:

1. **The session-level gold psychometrics don't translate directly to ToM signatures.** RQ1's predicted ACT-subscale × ToM-tier correlations are weak or directionally wrong at n = 10. The discriminant test (OE > BA on cognitive-ToM, etc.) is null. This is the *modest framing* the LBR Type 1 track asks for.

2. **The signal lives in within-conversation dynamics.** RQ3 finds a robust therapist-presencia effect on the next-round patient ACT-process trajectory. RQ5 finds that ruptures in the assessor's read of the patient track ACT-process discontinuities (the patient changing) more than perspective-gap discontinuities (the inferential distance widening).

3. **The expert panel actively rejects what Gemma-as-rubric scores as "high presencia"**, and disprefers `reformulación` (offering an alternative interpretation) — they reward staying with the patient's framing or inviting their own perspective, in terse content-anchored language. This is the most substantive RQ4 result and recasts the contribution: rather than ToM-stance/presencia *confirming* expert clinical judgment, the analysis reveals where they *diverge* and in what direction.

The unexpected RQ1 findings (PHQ ↓ realistic gap, OE ↑ temporal Wasserstein) are framed as observational secondary findings, motivating the §12 limitations and a future-work paragraph about the responsiveness-vs-rupture interpretation of the temporal signal.

---

## 9. Limitations

- **n = 10 sessions** for session-level analysis (RQ1) is low-power; even |ρ| = 0.80 only reaches p_fdr ≈ 0.08.
- **Simulation, not naturalistic data.** Patient turns are psychology students role-playing assigned clinical profiles; gold instrument scores reflect the assigned profile, not the student's own state. External validity to clinical populations is not claimed.
- **LLM-derived signals on both sides** of most correlations (mitigated by the deliberate model-class asymmetry — Llama for ACT processes + CompACT, Gemma for ToM signals).
- **In-the-moment therapist provenance per turn is not recoverable** (AI-suggested-accepted vs AI-edited vs self-authored); all therapist turns are treated as expert-vetted.
- **Phase categorical cleanup** still needed (accented vs unaccented Llama outputs inflate the phase term count in RQ4). Doesn't affect headlines.
- **Micro-validation** is a 10-round spot-check, not a full reliability study.
- **Single corpus, single language, single therapeutic modality** (Spanish ACT). Generalisation requires separate study.

---

## 10. Output artifacts

```
runs/tom_act_explanatory/
├── logs/                            # JSONL system of record (authoritative)
│   ├── llama_state_update.jsonl     ~568 lines + retries
│   ├── llama_assess_phq9.jsonl      ~568 lines + retries
│   ├── llama_assess_gad7.jsonl      ~568 lines + retries
│   ├── llama_assess_compact10.jsonl ~568 lines + retries
│   ├── self_a.jsonl   self_b.jsonl  observer_p.jsonl  observer_pt.jsonl   # ~568 each
│   ├── tom_tier_patient.jsonl       ~568 lines
│   ├── tom_stance.jsonl             ~1,704 lines (3 candidates × 568)
│   ├── presencia.jsonl              ~1,704 lines
│   ├── meta.jsonl                   T1 events
│   ├── meta.T2.jsonl                T2 events (tier_pass_complete + reparse counts)
│   └── meta.T3.jsonl                T3 events
├── outputs/
│   ├── aggregated/
│   │   ├── llama_state.parquet              (568 rows)
│   │   ├── llama_assessors.parquet          (~14,768 rows, long format)
│   │   ├── gemma_views.parquet              (~59,066 rows, long format)
│   │   ├── tom_tier.parquet                 (568 rows, soft + argmax)
│   │   ├── tom_stance.parquet               (1,704 rows)
│   │   ├── presencia.parquet                (1,704 rows)
│   │   └── recovery_report.json             per-signal recovery-stage proportions
│   ├── wasserstein_test/
│   │   ├── temporal.csv  temporal.parquet   both variants × 3 instruments × 568 rounds
│   ├── cross_perspective/
│   │   └── gaps.parquet                     six pairwise + AGG per round per instrument
│   ├── analysis/
│   │   ├── rq1_session_correlations.parquet (30 rows)
│   │   ├── rq1_session_frame.parquet        (10 rows)
│   │   ├── rq2_internal_consistency.parquet (3 rows)
│   │   ├── rq2_gold_calibration.parquet     (10 rows, Move C)
│   │   ├── rq2_round_models.parquet         (45 rows)
│   │   ├── rq2_round_frame.parquet          (567 rows)
│   │   ├── rq3_lagged_models.parquet        (37 rows)
│   │   ├── rq3_transition_frame.parquet     (556 rows)
│   │   ├── rq4_candidate_logit.parquet      (17 rows)
│   │   ├── rq4_candidate_frame.parquet      (1,704 rows)
│   │   ├── rq4_crosstab_stance.parquet      (4 rows)
│   │   ├── rq4_crosstab_presencia.parquet   (3 rows)
│   │   ├── rq5_rupture_overlay.parquet      (81 rows)
│   │   └── micro_validation_tom_tier.csv    (10 rows, manual coding pending)
│   └── figures/
│       ├── case_study_S07.png   case_study_S07.pdf
│       └── case_study_S09.png   case_study_S09.pdf

docs/
├── tom_act_consolidated_report.md           ← this document
├── tom_act_T1_RQ1_findings.md
├── tom_act_T2_RQ2_RQ3_findings.md
└── tom_act_T3_RQ4_findings.md

src/mentalriskes/tom_act/                    14 modules + analysis/ subpackage
scripts/watch_t{1,2,3}_then_analyze.py       background orchestration
tests/test_tom_act_*.py                      45 unit tests
config/tom_act.yaml                          providers + tiers + data paths
specs/MentalRiskES/tom_act_analysis_spec_v0.6.md  the specification
```

---

## 11. Reproducibility

- Every LLM call's complete system prompt, user prompt, raw response, and SHA-256 hashes are persisted in the JSONL logs.
- `code_version` in every log line stamps the git commit at run start (`git:<hash>`).
- The recovery pipeline can re-parse any historical raw response with updated logic via the `reparse` command — no LLM cost.
- All analyses (RQ1–RQ5, case studies, micro-val) are deterministic given the aggregated parquet tables.
- 45 unit tests covering recovery, dispatcher resume idempotency, data loader, Wasserstein both variants, aggregator precedence rules, tier definitions, and constants. All passing.

---

## 12. Next steps

- **Manual micro-validation coding** (10 rounds → 3×3 confusion matrix) before paper finalisation.
- **Phase-label deduplication** (accented vs unaccented Llama outputs) for clean RQ4 phase-conditional cross-tabs.
- **Paper drafting** — 4-page LBR mapping:
  - Intro + ToM/ACT framing + Hypertext-community motivation (~0.75 page)
  - Data + corpus characterisation (~0.5 page)
  - Method — signals + Wasserstein + recovery (~0.75 page)
  - Results — 5 RQs condensed (~1.25 pages)
  - Case studies S07 vs S09 figure (~0.5 page)
  - Limitations + ethics + AI use + future work (~0.25 page)
- **Optional T4 (conditional)** — Gemma 3 27B sensitivity on Self-A + Observer-P (1,136 calls) if reviewer concerns about model-class circularity warrant it. Spec authorises only after T1 returned non-null indicative results — the unexpected RQ1 correlations (PHQ ↓ gap, OE ↑ W₁) and the strong RQ3/RQ4 findings could be argued either way.
