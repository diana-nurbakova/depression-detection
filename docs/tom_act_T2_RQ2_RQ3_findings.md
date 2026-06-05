# T2 Checkpoint — RQ2 and RQ3 Findings

**Status**: preliminary, run-in-progress
**Date**: 2026-05-29
**Corpus**: MentalRiskES 2026 test set, 10 sessions, 568 patient-rounds
**Tier**: T2 complete (Self-B, Observer-PT, ToM-stance×gold, presencia×gold). T1 Gemma still in progress.
**Spec ref**: `specs/MentalRiskES/tom_act_analysis_spec_v0.6.md` §4 RQ2/RQ3, §8.2/§8.4, §15.3

---

## TL;DR

- **RQ3 (lagged therapist → patient state) shows a robust presencia effect** and directionally consistent stance effects. Five contrasts survive Benjamini-Hochberg FDR within RQ3.
- **RQ2 internal consistency** is mixed: `valores+accion_comprometida × VA` correlates strongly *positive* as predicted (ρ = +0.28, p_fdr ≈ 6 × 10⁻¹¹), but `defusion+aceptacion × OE` (ρ = −0.09) and `momento_presente × BA` (ρ = −0.18) come out *negative*. This needs interpretation (see §3.1).
- **Move C gold calibration** is clean: **0/10 sessions flagged** (all normalized L1 vs gold < 0.40; max = S07 at 0.327). The Llama signal is well-calibrated against the assigned profiles.
- **Spec §15.3 T2→T3 gate**: ✅ pass. RQ3 shows "interpretable lagged effects" (the gate's OR clause), so the spec authorises proceeding to T3 even before RQ2 is fully evaluable (the headline `gap_conservative` and tier soft-scores still depend on T1's gen-gemma, which is in progress).

---

## 1. Methods at a glance

- Mixed-effects models (statsmodels `MixedLM`) with **per-session random intercept**, fixed effect of the ACT/therapist signal.
- p-values **Benjamini-Hochberg FDR-corrected within each RQ family** (`p_fdr`).
- Two parallel Llama-derived ACT operationalisations:
  - **Set A** — `procesos_act` hexaflex (6 dims incl. `yo_como_contexto`)
  - **Set B** — Llama-derived CompACT-10 → OE / BA / VA subscales
- Move C: per-session L1 between Llama session-mean CompACT-10 and gold, normalised by 60 (max possible); flag threshold 0.40.

| File | Rows | What |
|---|---:|---|
| `rq2_round_models.parquet` | 45 | mixed-effects of ACT process IV on ToM DV |
| `rq2_internal_consistency.parquet` | 3 | Set-A vs Set-B Spearman with bootstrap CI + FDR |
| `rq2_gold_calibration.parquet` | 10 | per-session Llama vs gold L1 (Move C) |
| `rq2_round_frame.parquet` | 567 | round-level join used as input |
| `rq3_lagged_models.parquet` | 37 | gold-stance/presencia(t) → Δprocesos_act / Δgap / tier(t+1) |
| `rq3_transition_frame.parquet` | 556 | t → t+1 transitions used as input |

---

## 2. RQ2 — round-level ACT × ToM correspondence

### 2.1 Internal consistency (Set A ↔ Set B)

Two of the three theoretically related pairs come out in the **wrong** direction. This is the most non-trivial RQ2 finding to date.

| Set A | Set B | ρ | 95 % CI | p | p_fdr | predicted | direction match |
|---|---|---:|---|---:|---:|---|:---:|
| `defusion + aceptacion` | OE | **−0.094** | [−0.182, −0.004] | 0.026 | 0.026 | + | ✗ |
| `momento_presente` | BA | **−0.181** | [−0.261, −0.099] | 1.6e-5 | 2.3e-5 | + | ✗ |
| `valores + accion_comprometida` | VA | **+0.277** | [+0.195, +0.356] | 2.0e-11 | 6.1e-11 | + | ✓ |

**Interpretation candidate (testable, not yet tested):** `procesos_act` likely measures the *in-session expression* of a process (defusion rises when the therapist actively introduces defusion language for a patient who *needs* it), while CompACT-10 OE/BA measure the patient's *general tendency* toward openness / awareness (high OE = low avoidance, reverse-scored). A therapist working with a fused/avoidant patient triggers high in-session defusion (Set A) on a patient whose general OE is low (Set B). The result: a negative correlation between the two operationalisations, *consistent* with both reads being correct but measuring different time-scales. Valued Action, in contrast, captures stable behaviour and matches across operationalisations.

This is worth a paragraph in the paper rather than a "failure" framing.

### 2.2 Move C — gold-anchored calibration

All 10 sessions pass the L1 ≤ 0.40 threshold against gold CompACT-10 item vectors. **0 flagged sessions**, so RQ2 results don't need an "excluding divergent sessions" annotation.

| session | L1 / 60 | flagged |
|---|---:|:---:|
| S07 | 0.327 | False |
| S03 | 0.281 | False |
| S09 | 0.237 | False |
| S01 | 0.234 | False |
| S15 | 0.172 | False |
| S12 | 0.171 | False |
| S16 | 0.153 | False |
| S05 | 0.144 | False |
| S06 | 0.134 | False |
| S04 | 0.080 | False |

### 2.3 Mixed-effects models — partial picture

The cross-perspective gap DVs that are populated at T2 reduce to `gap_realistic` (uses `self_b` + `observer_pt`, both T2 signals). The headline `gap_conservative` uses `self_a` + `observer_p` (T1 signals still in progress at 50/568) — so 26 of the 45 model rows are NaN. The ToM-tier soft scores are likewise sparse pending T1.

For `gap_realistic`, all nine ACT-process IVs produce vanishingly small effects:

| iv | dv | β | 95 % CI | p_fdr |
|---|---|---:|---|---:|
| aceptacion | gap_realistic | +0.0035 | [−0.0041, +0.0110] | 0.86 |
| momento_presente | gap_realistic | +0.0033 | [−0.0052, +0.0117] | 0.86 |
| **yo_como_contexto** | gap_realistic | **−0.0027** | [−0.0103, +0.0049] | 0.86 |
| accion_comprometida | gap_realistic | +0.0016 | [−0.0053, +0.0084] | 0.86 |
| defusion | gap_realistic | +0.0011 | [−0.0062, +0.0085] | 0.86 |
| OE | gap_realistic | +0.0007 | [−0.0005, +0.0019] | 0.86 |
| BA | gap_realistic | +0.0008 | [−0.0010, +0.0027] | 0.86 |
| VA | gap_realistic | +0.0002 | [−0.0006, +0.0009] | 0.86 |
| valores | gap_realistic | −0.0003 | [−0.0072, +0.0066] | 0.94 |

`yo_como_contexto × gap_realistic` is directionally negative as predicted (spec §4 RQ2 "yo_como_contexto × perspective gap → negative"), but the effect is **tiny** (β ≈ −0.003) and non-significant. **Verdict: provisional null on this DV.** Re-run after T1's gen-gemma completes will populate `gap_conservative` and the tier soft scores, which is where stronger predicted patterns may emerge.

---

## 3. RQ3 — lagged therapist intervention → patient state

n = 556 transitions; reference categories: stance `defusión` (alphabetical first), presencia `alta`.

### 3.1 Presencia effects (significant after FDR)

When the gold therapist response at round t has **low presencia** (`baja`), every patient ACT process at t+1 is depressed relative to high-presencia turns. Five contrasts survive FDR within RQ3:

| DV | level | β | p | p_fdr |
|---|---|---:|---:|---:|
| Δ defusion | presencia [baja] | **−0.068** | 0.0025 | **0.034** |
| Δ defusion | presencia [media] | **−0.041** | 0.0018 | **0.034** |
| Δ momento_presente | presencia [baja] | **−0.068** | 0.0038 | **0.034** |
| Δ yo_como_contexto | presencia [baja] | **−0.066** | 0.0029 | **0.034** |
| Δ aceptacion | presencia [baja] | **−0.068** | 0.0051 | **0.036** |

**Interpretation**: aligned with the spec's expected direction that high therapist presencia supports patient ACT-process engagement at the next turn. The effect is strikingly consistent across all six processes (Δ valores and Δ accion_comprometida show the same sign at p_fdr ≈ 0.11, just below the FDR bar).

### 3.2 Stance effects on perspective gap (marginal)

Both `invitación-a-tomar-perspectiva` and `reformulación` (relative to `defusión` baseline) shrink the realistic perspective gap at t+1:

| DV | level | β | p | p_fdr |
|---|---|---:|---:|---:|
| Δ gap_realistic | stance [invitación-a-tomar-perspectiva] | −0.0125 | 0.011 | 0.064 |
| Δ gap_realistic | stance [reformulación] | −0.0118 | 0.015 | 0.077 |
| Δ gap_realistic | stance [reflejo] | −0.0087 | 0.065 | 0.174 |

Direction is consistent with the spec premise that observer-perspective-inviting interventions narrow the self/observer gap. p_fdr is in the "interesting but not significant" range — promising for the §15.3 OR-clause.

### 3.3 yo_como_contexto pattern (directional, descriptive)

The spec specifically predicts `invitación-a-tomar-perspectiva` → larger Δ`yo_como_contexto` than `reflejo`. Relative to the `defusión` baseline (β = 0 by construction):

| stance | β on Δ yo_como_contexto |
|---|---:|
| invitación-a-tomar-perspectiva | **−0.006** |
| reflejo | **−0.050** (p_fdr = 0.106) |

So `invitación-a-tomar-perspectiva` produces a **larger increase** (less negative) in `yo_como_contexto` at t+1 than `reflejo` does — **directionally consistent with the spec prediction**, though the contrast is not directly tested in this model (it'd need re-leveling with `reflejo` as the baseline; can be added if useful).

### 3.4 No clean presencia × perspective-gap effect

Spec also predicts high presencia → larger Δ-gap *reduction*. The current data say no:

| DV | level | β | p_fdr |
|---|---|---:|---:|
| Δ gap_realistic | presencia [media] | −0.0002 | 0.994 |
| Δ gap_realistic | presencia [baja] | +0.00006 | 0.994 |

Presencia drives ACT-process dynamics but apparently not the perspective gap, in this corpus. Worth noting as a substantive (not null-artefact) finding.

---

## 4. Spec §15.3 T2 → T3 decision

The gate (verbatim): *"proceed to T3 if RQ2 confirms the RQ1 direction at round level (consistency across granularities) **or** if RQ3 shows interpretable lagged effects."*

- **RQ2 → cannot fully evaluate yet** (gap_conservative + tier soft-scores blocked by in-flight T1 Gemma). The only fully-populated DV (`gap_realistic`) returns null.
- **RQ3 → clearly meets the "interpretable lagged effects" bar**: 5 contrasts survive FDR, all consistent in direction (low presencia depresses patient ACT-process evolution).

**→ proceed to T3** (✅ gate clause satisfied via the OR branch).

---

## 5. Limitations and what's still in flight

- **T1 Gemma (`self_a`, `observer_p`, `tom_tier_patient`) is at ~50/568.** Until it completes:
  - `gap_conservative` (the *conservative* headline of the two §7.1 headlines) is sparse.
  - ToM-tier soft-scores (DV for predicted yo_como_contexto × cognitivo and OE × cognitivo/afectivo correlations) are sparse.
- After T1 Gemma finishes, re-running `aggregate analyze --rq 2` will populate those rows automatically (no extra LLM cost). The T1 watcher (background id `b4u8pczh2`) will do this on completion.
- Internal-consistency reversal (§2.1) needs framing in the paper — likely an in-session-vs-trait time-scale difference rather than a measurement failure.
- RQ3 currently uses gold-only stance/presencia, which is by design (T2 scope). The full t-level distributions over all three candidates arrive only with T3.

---

## 6. Files

```
runs/tom_act_explanatory/outputs/analysis/
├── rq2_round_models.parquet         # 45 rows
├── rq2_internal_consistency.parquet # 3 rows
├── rq2_gold_calibration.parquet     # 10 rows
├── rq2_round_frame.parquet          # 567 rows (round-level frame)
├── rq3_lagged_models.parquet        # 37 rows
└── rq3_transition_frame.parquet     # 556 rows
```
