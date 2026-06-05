# T1 Final — RQ1 Findings (Session-level, Gold-anchored)

**Status**: T1 complete (all 7 signals at 100 % unique-success coverage)
**Date**: 2026-05-30
**Spec ref**: `specs/MentalRiskES/tom_act_analysis_spec_v0.6.md` §4 RQ1, §8.1
**Unit**: session (n = 10)

---

## TL;DR

- **The four pre-specified RQ1 predictions don't hold at session-level.** Two of the four are directionally *opposite* to prediction; the other two are correct in sign but trivial in magnitude. None survive Benjamini-Hochberg FDR within the predicted family.
- **The strongest correlations are unexpected ones** (|ρ| ≈ 0.72–0.80, p_fdr 0.08–0.12), and they tell a coherent story:
  - **PHQ-9 total × realistic perspective gap = −0.80** — depression severity *closes* the realistic gap (both self-view and clinical-observer view converge on heavy symptom endorsement).
  - **OE × mean temporal Wasserstein = +0.80** — higher psychological openness goes with *more* state-trajectory fluctuation across rounds.
  - **BA × realistic gap = +0.75** — higher behavioural awareness goes with *larger* self/observer gap.
- **The discriminant test the spec proposed is null** — OE and BA correlate similarly (and weakly) with the three ToM-tier proportions; the subscale structure is not informative for tier-prediction at n = 10.

Combined with the T2/T3 picture this is a coherent paper story: *session-level patterns are weak or inverted from prediction; the real signal lives in within-conversation lagged dynamics (RQ3) and in the structure of expert clinical preference (RQ4).*

---

## 1. The pre-specified predictions

| relation | predicted | ρ | 95 % CI | p_fdr | direction match |
|---|:---:|---:|---|---:|:---:|
| OE × prop_cognitivo | + | +0.19 | [−0.45, +0.72] | 0.91 | ✓ (weak) |
| OE × prop_afectivo | + | **−0.39** | [−0.76, +0.23] | 0.88 | **✗** |
| BA × prop_somatico | + | **−0.21** | [−0.68, +0.45] | 0.91 | **✗** |
| VA × prop_cognitivo | + (weaker than OE) | +0.04 | [−0.70, +0.82] | 0.95 | ✓ (trivial) |

Two of four directionally wrong; none significant. **The hypothesis that ACT-flexibility subscales map onto ToM-tier proportions at session-level is not supported.**

### Discriminant patterns (the §4 RQ1 consistency check)

The spec also asks: *"OE × somatic-ToM should be weaker than BA × somatic-ToM"* and *"BA × cognitive-ToM should be weaker than OE × cognitive-ToM"*. Both checks come out near-null:

| iv | dv | ρ |
|---|---|---:|
| OE | prop_somatico | −0.09 |
| BA | prop_somatico | −0.21 ← *more negative* than OE (consistent direction; tiny) |
| OE | prop_cognitivo | +0.19 ← *more positive* than BA (consistent direction; tiny) |
| BA | prop_cognitivo | +0.10 |

The directional asymmetry is in the predicted direction, but the magnitudes are so close that **the discriminant signal is effectively null**.

---

## 2. The strong correlations — unexpected but clinically interpretable

The 12 strongest correlations across all 5 IVs (3 subscales + PHQ-9 + GAD-7) × 6 DVs (3 tier props + 2 gaps + temporal W1) are:

| iv | dv | ρ | 95 % CI | p_fdr |
|---|---|---:|---|---:|
| **OE** | **mean_temporal_w1** | **+0.80** | [+0.33, +0.98] | **0.08** |
| **phq9_total** | **gap_realistic** | **−0.80** | [−0.99, −0.23] | **0.08** |
| **BA** | **gap_realistic** | **+0.75** | [+0.23, +0.96] | 0.11 |
| **BA** | **mean_temporal_w1** | **+0.73** | [+0.14, +1.00] | 0.11 |
| **phq9_total** | **mean_temporal_w1** | **−0.72** | [−0.93, −0.18] | 0.11 |
| gad7_total | mean_temporal_w1 | −0.62 | [−0.96, +0.11] | 0.28 |
| VA | gap_conservative | −0.58 | [−0.98, +0.21] | 0.35 |
| gad7_total | gap_realistic | −0.47 | [−0.91, +0.33] | 0.63 |

Three patterns emerge:

1. **Symptom severity narrows the realistic gap.** PHQ-9 and (less strongly) GAD-7 totals are negatively correlated with `gap_realistic`. Mechanism candidate: in severe cases the Llama Self-B and Observer-PT both score symptoms high (congruent → small gap); in mild cases the patient under-reports relative to the clinical-observer inference (divergent → larger gap). The lowest-PHQ session, S07 (PHQ = 2), has the *largest* `gap_realistic = 0.048`; the highest-PHQ session, S09 (PHQ = 21), has `gap_realistic = 0.021`.

2. **Psychological flexibility goes with state-trajectory variability.** OE × mean_temporal_w1 = +0.80 and BA × mean_temporal_w1 = +0.73. More flexible patients fluctuate more across rounds — the Llama assessor sees their state shift more between consecutive turns. This is *not* what the rupture-flag-as-pathology framing of §7.3 implies; it might warrant reframing the temporal Wasserstein as "responsiveness" rather than "rupture" at session-aggregate level.

3. **Behavioural awareness goes with a *larger* observer gap.** BA × gap_realistic = +0.75. The patients judged most "behaviourally aware" (low autopilot) have the *biggest* self/observer scoring discrepancy. Worth one paragraph in the paper to explore — likely a phrasing artefact (highly self-aware patients articulate their state precisely, but the clinical observer reading the full dialogue infers a different severity profile).

---

## 3. The full session frame (raw data)

| session | OE | BA | VA | PHQ-9 | GAD-7 | prop_som | prop_cog | prop_aff | gap_cons | gap_real | mean_w1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S01 | 4 | 4 | 9 | 16 | 16 | 0.13 | 0.83 | 0.05 | 0.041 | 0.029 | 0.048 |
| S03 | 2 | 2 | 6 | 22 | 18 | 0.18 | 0.50 | 0.32 | 0.085 | 0.017 | 0.031 |
| S04 | 5 | 5 | 18 | 13 | 15 | 0.25 | 0.48 | 0.28 | 0.069 | 0.020 | 0.054 |
| S05 | 5 | 4 | 13 | 15 | 16 | 0.25 | 0.71 | 0.04 | 0.074 | 0.023 | 0.068 |
| S06 | 7 | 10 | 16 | 16 | 10 | 0.12 | 0.71 | 0.17 | 0.079 | 0.035 | 0.060 |
| S07 | 5 | 18 | 9 | 2 | 7 | 0.23 | 0.53 | 0.23 | 0.088 | 0.048 | 0.064 |
| S09 | 1 | 1 | 11 | 21 | 19 | 0.16 | 0.60 | 0.24 | 0.080 | 0.021 | 0.030 |
| S12 | 1 | 5 | 20 | 11 | 17 | 0.11 | 0.71 | 0.18 | 0.024 | 0.044 | 0.052 |
| S15 | 7 | 8 | 15 | 11 | 16 | 0.09 | 0.81 | 0.10 | 0.077 | 0.043 | 0.077 |
| S16 | 5 | 12 | 14 | 7 | 14 | 0.14 | 0.74 | 0.12 | 0.069 | 0.041 | 0.068 |

(OE/BA/VA scored from gold CompACT-10 with reverse-scoring; tier proportions and gaps aggregated across all rounds per session.)

---

## 4. Limitations

- **n = 10 sessions.** Even |ρ| = 0.80 only reaches p_fdr ≈ 0.08 after BH correction; the report leans on consistency of direction across multiple related signals, as the spec anticipated (§8.1 power note).
- **Tier proportions are dominated by `cognitivo`** in most sessions (0.48–0.83). The somatic and affective categories have small denominators; the ToM-tier signal at session-level is essentially "how cognitive-ToM-leaning are the patient turns overall."
- **The discriminant null** can't distinguish "the ACT subscales genuinely don't map to ToM tiers in this corpus" from "n = 10 is too small to detect a real difference." Round-level RQ2 results (n = 567) are the better statistical lens — and they're consistent with the null direction (β ≈ 0 for ACT-process × tier soft scores).

---

## 5. Putting the three tiers together

| tier | RQ | headline |
|---|---|---|
| T1 | **RQ1** | predicted relations don't hold; strongest signals are *unexpected* (PHQ↓gap, OE↑W₁) |
| T2 | **RQ2** | round-level ACT × ToM also weak; Move C calibration uniformly clean (0/10 flagged) |
| T2 | **RQ3** | **strong presencia effect** (low presencia depresses next-round ACT processes, p_fdr ≈ 0.03 across 5 contrasts) |
| T3 | **RQ4** | **panel disprefers high presencia and reformulación** — spec's RQ4 prediction reversed (OR = 1.91 for low-presencia, p = 0.002) |

The coherent story is: *session-level psychometrics don't translate to ToM signatures (RQ1 null), but the within-conversation dynamics do — therapist presencia drives patient state evolution (RQ3) and expert panels actively favour terse, content-anchored responses over verbally warm ones (RQ4)*.

---

## 6. Files

```
runs/tom_act_explanatory/outputs/analysis/
├── rq1_session_correlations.parquet   # 30 rows (5 IV x 6 DV)
└── rq1_session_frame.parquet          # 10 rows (per-session aggregates)
```

---

## 7. Remaining work

- **RQ5 (Wasserstein-rupture overlay)** has not been computed in the watcher chain — run `analyze --rq 5` to produce it.
- **Case-study figures (S07 vs S09)** — run `case-studies` to produce the trajectory PDFs.
- **Micro-validation ToM-tier sampling** — run `micro-val`.
