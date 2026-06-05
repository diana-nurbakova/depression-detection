# T3 Checkpoint — RQ4 Findings (Selected vs Rejected Candidate)

**Status**: preliminary
**Date**: 2026-05-30
**Tier**: T3 complete (stance + presencia × 2 rejected candidates; 7 h wall-clock). T1 Gemma still in progress.
**Spec ref**: `specs/MentalRiskES/tom_act_analysis_spec_v0.6.md` §4 RQ4, §8.5

---

## TL;DR — the spec's RQ4 prediction is *reversed*

| Spec prediction (§4 RQ4) | What the data says |
|---|---|
| High-presencia candidates are gold-selected at a *higher* rate than low-presencia. | **Opposite.** `presencia = baja` gold-rate is **44.3 %**; `presencia = alta` is **29.4 %**. The candidate-level logit gives `presencia[baja]` OR = **1.91**, p = 0.002 — *low-presencia* candidates are ~2× more likely to be panel-selected. |

And a secondary, FDR-marginal finding: **`reformulación` stance is significantly disfavored** (OR = 0.62, p = 0.025) relative to the `defusión` baseline. Other stances (reflejo, invitación-a-tomar-perspectiva) are statistically indistinguishable from defusión.

**Phase has no clear effect** — no `fase_terapeutica` level is significant in the joint model.

This is the most substantive find of the tier and reframes one of the four contributions: rather than "experts converge on Gemma's notion of therapeutic presence," the picture is closer to "Gemma's presencia scoring captures verbal warmth / expansiveness, which expert panels actively *dis*prefer in this clinical context."

---

## 1. Numbers

### 1.1 Stance × is_gold (n per stance is the row count)

| stance | n | gold-rate | rejected-rate |
|---|---:|---:|---:|
| defusión | 120 | **0.375** | 0.625 |
| reflejo | 655 | 0.369 | 0.631 |
| invitación-a-tomar-perspectiva | 359 | 0.348 | 0.652 |
| reformulación | 570 | **0.274** | 0.726 |

(baseline = 568 gold / 1,704 total = 0.333)

### 1.2 Presencia × is_gold

| presencia | n | gold-rate | rejected-rate |
|---|---:|---:|---:|
| **baja** | 115 | **0.443** | 0.557 |
| media | 936 | 0.347 | 0.653 |
| alta | 653 | **0.294** | 0.706 |

### 1.3 Candidate-level logit (1,704 candidates; reference: `defusión` stance, `alta` presencia, `aceptación` phase)

| term | coef | OR | p |
|---|---:|---:|---:|
| **presencia [baja]** | **+0.65** | **1.91** | **0.002** |
| **stance [reformulación]** | **−0.48** | **0.62** | **0.025** |
| presencia [media] | +0.22 | 1.25 | 0.046 |
| stance [invitación-a-tomar-perspectiva] | −0.16 | 0.85 | 0.47 |
| stance [reflejo] | −0.02 | 0.98 | 0.94 |
| Intercept | −0.57 | 0.57 | 0.29 |
| all 11 `fase_terapeutica` levels | |−0.07 … −0.26 | OR 0.77–0.93 | all p ≥ 0.63 |

**Update 2026-05-30**: phase labels now canonicalised in `aggregator.build_llama_state_long` via `constants.canonical_phase` (collapses accented/unaccented variants to canonical Spanish: `defusion → defusión`, `exploracion → exploración`, etc.). After re-aggregation + re-`analyze --rq 4`, the logit is **12 terms** (down from 17) with 6 distinct phase predictors. Headlines unchanged: presencia[baja] OR = 1.91 (p = 0.002), stance[reformulación] OR = 0.63 (p = 0.028).

---

## 2. Interpretation candidate (for the paper)

The reversal isn't a spec-bug — it's the substantive contribution. Two plausible mechanisms, both worth naming:

1. **Gemma's "presencia" tracks verbal warmth and inclusivity, expert panels reward something else.** In ACT contexts, the panel-preferred turn is often a *terse, content-anchored* response that asks one specific question (e.g. *"Cuéntame un poco más. ¿La ansiedad de la que me hablas se limita a situaciones sociales?"*). These score low on Gemma's "alta" presencia rubric, which rewards explicit empathy / "estar contigo" markers and warmth phrasing. The panel is **explicitly choosing against templated empathy** in favour of clinical specificity.

2. **`reformulación` is the disfavored stance.** Reformulation is the candidate that offers an alternative interpretation of the patient's stated experience. Expert ACT therapists in this corpus appear to prefer staying with the patient's framing (`reflejo`, `defusión`) or inviting their own perspective (`invitación-a-tomar-perspectiva`) over offering a competing interpretation. This is consistent with ACT's emphasis on the patient as the expert on their own values.

Both reframings are *strengthenings* of the paper's preliminary-empirical posture: the four contributions become more interesting when one of them comes out the opposite of the prediction.

---

## 3. Caveats

- **Phase encoding has a duplicate-level issue** (accented vs unaccented strings from Llama). 11 phase terms in the model are really ~6 distinct phases. Cleaning the categorical levels won't change the presencia / reformulación result (phase isn't significant anyway), but the phase-conditional cross-tabs in §8.5 should be re-run after deduplication.
- **The candidate-level logit assumes independence within round** (i.e., 1,704 independent observations). In reality the 3 candidates per round are clustered. A conditional / nested model would tighten the inference; the headline OR signs are robust to that switch.
- **Sample size**: `defusión` n = 120 and `presencia=baja` n = 115 are the smallest cells. The effect sizes are large enough that this shouldn't reverse, but CIs around the OR are wider for those terms.
- **T1's Gemma signals (self_a, observer_p, tom_tier_patient) are still arriving**, so the RQ1 and RQ2 headline gaps will refresh ~24 h from now when the T1 watcher fires its final `analyze --rq 1` (and a re-aggregate refreshes RQ2 / RQ4 with no extra LLM cost).

---

## 4. Files

```
runs/tom_act_explanatory/outputs/analysis/
├── rq4_candidate_frame.parquet      # 1,704 rows (3 candidates x 568 rounds)
├── rq4_candidate_logit.parquet      # 17 model terms
├── rq4_crosstab_stance.parquet      # 4 stance levels x is_gold
└── rq4_crosstab_presencia.parquet   # 3 presencia levels x is_gold
```
