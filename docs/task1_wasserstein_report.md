# eRisk 2026 Task 1 — Wasserstein Distance (W₁): Report & Counterfactual Analysis

**Team.** INSA-Lyon. **Task.** Conversational depression detection (20 LoRA personas; predict BDI-II total + top-4 symptoms).
**Scope of this note.** What the W₁ metric measures, how it was (and was *not*) used in the submitted system, a post-hoc retrofit over the official test logs, a worked example, and a counterfactual analysis of what W₁ *could* have changed had it been live.

**Artifacts.**
- Reproducible retrofit: [`analysis/eda_task1/wasserstein_retrofit.py`](../analysis/eda_task1/wasserstein_retrofit.py)
- Full results JSON: [`analysis/eda_task1/wasserstein_retrofit.json`](../analysis/eda_task1/wasserstein_retrofit.json)
- Live code paths: [`src/erisk_task1/tom.py`](../src/erisk_task1/tom.py), [`src/erisk_task1/orchestrator.py`](../src/erisk_task1/orchestrator.py)

---

## 0. TL;DR

1. **W₁ was never computed live.** POT was not installed during the official runs (`pot_available: false` in *every* log), so no Wasserstein value was produced. The submitted runs' adaptive interviewing was driven by **symptom-coverage** signals, not W₁.
2. Even with POT, the turn-over-turn metric **`W_self` is never read by the orchestrator** — only `W_align` (interviewer↔persona alignment) and coverage gaps are. So `W_self` was a *diagnostic by design*.
3. We retrofit W₁ from the logged per-turn symptom profiles. It works only for **personas 12–19** (Run 3, ToM era); personas 1–11 (incl. **Daniel**) and Runs 1–2 logged empty ToM.
4. **Counterfactual is partially answerable.** We can reconstruct the *signal* the live policy would have seen (the `W_align` hint, and fire-vs-termination timing) from logged data. We **cannot** measure the *effect* on final predictions without re-running — and the personas would respond differently to redirected questions.
5. The reconstructed `W_align` hint flags exactly the two pathological conversations (Javier's looping interviewer, Priya's over-probed minimal case). The clearest actionable case is **Linda**, cut off at `min_turns` with symptom mass still rising one round after a disclosure spike.

---

## 1. What the metric measures

After each persona turn *t*, the assessors produce a 21-dim **expressed symptom profile** `E(t)` (BDI-II item scores 0–3; un-scored items = 0; [tom.py `update_expressed`](../src/erisk_task1/tom.py)). W₁ is the **1-Wasserstein (earth-mover) distance** between two such profiles under a fixed **clinical ground metric** `M` (21×21 inter-item cost; [`get_cost_matrix`](../src/erisk_task1/tom.py)). Both profiles are **L1-normalised** before transport, so W₁ measures a shift in the *shape* of the symptom distribution — which symptoms dominate — independent of total severity mass ([`wasserstein_balanced`](../src/erisk_task1/tom.py)).

We compute two conventions:

| Convention | Definition | Source | Role |
|---|---|---|---|
| **`W_self`** | W₁(`E(t)`, `E(t−1)`) | eRisk `tom.compute_wasserstein_metrics` (`W_self[t][1]`) | turn-over-turn shift; the eRisk-native quantity |
| **barycenter** | W₁(`E(t)`, mean(`E(1..t)`)) | `mentalriskes.task1.temporal.compute_w1_trajectory` | round-vs-running-mean anomaly detector |

**Firing rule** (both conventions, matching `temporal.detect_anomalous_rounds`): a round *fires* iff `W₁ > running_μ + 2·running_σ`, where μ/σ are computed over the trajectory strictly **before** the current round; the first **3** points never fire (warmup); σ is floored with `+1e-6`.

A *fire* is meant to mark a **moment of disclosure** — a round where the persona reveals new symptom content that moves the distribution.

---

## 2. Status in the submitted system: post-hoc, not live

This is the central honesty point.

- **No live computation.** Every `runs/task1/**/internal_*.json` carries `pot_available: false` with empty `W_self`/`W_align`. `compute_wasserstein_metrics` returns early without POT ([tom.py](../src/erisk_task1/tom.py)). The `backfill_wasserstein.py` script exists precisely because the values had to be recomputed afterward.
- **`W_self` is not wired to behaviour.** The orchestrator's only ToM hook, [`get_orchestrator_context`](../src/erisk_task1/tom.py), injects two things into the reasoning prompt ([orchestrator.py:246](../src/erisk_task1/orchestrator.py#L246)): **coverage-gap descriptions** (derived from `E_profiles`, *no POT needed*) and a **`W_align` interpretation**. `W_self` appears only in the serialized summary.
- **What actually steered the live interviews** was the coverage machinery — including the hard-coded **SOMATIC coverage override** at [orchestrator.py:281](../src/erisk_task1/orchestrator.py#L281) (force a daily-routine probe if somatic is uncovered after turn 3). This is consistent with the analysis doc's wording, "ToM tracking on for orchestrator guidance only."

**Consequence for the working notes.** Any W₁ firing example is a **post-hoc diagnostic** (results section), *not* a description of live policy. We must **not** claim "W₁ fired → the orchestrator acted." It did not.

---

## 3. Retrofit method & coverage

[`wasserstein_retrofit.py`](../analysis/eda_task1/wasserstein_retrofit.py) reloads the logged per-turn `E_profiles` (and `I_profiles`, for `W_align`) and recomputes both conventions + firing flags under the live clinical ground metric.

**Coverage limit.** Only the ToM-era Run-3 logs saved per-turn profiles:

| Retrofittable (personas 12–19) | Not retrofittable |
|---|---|
| Javier, Laura, Linda, Marco, Maria, Maya, Noah, Priya | personas 1–11 (incl. **Daniel** pid 6) + persona 20 (Sofia); all of Runs 1–2 |

Personas 1–11 and Runs 1–2 logged empty ToM, so W₁ cannot be reconstructed without re-running the assessor pipeline (LLM).

---

## 4. Results — all retrofittable personas

W₁ trajectories under both conventions; columns show which rounds *fire*.

| Persona | gold BDI | pred BDI | rounds | **barycenter** fires | **`W_self`** fires |
|---|---:|---:|---:|---|---|
| Javier | 15 | 24 | 7 | **t4** (strong) | **t7** |
| Laura | 23 | 21 | 7 | — | — |
| Linda | 28 | 19 | 5 | **t4** | — |
| Marco | 38 | 18 | 7 | **t4** | — |
| Maria | 40 | 14 | 7 | — | — |
| Maya | 6 | 10 | 7 | t4 | — |
| Noah | 5 | 7 | 7 | — | — |
| Priya | 7 | 11 | 7 | — | — |

**Two structural findings.**
- **`W_self` essentially never fires.** Disclosure is **front-loaded** (largest shifts at t2–t3) and the trajectory decays; by the time the 3-round warmup is satisfied, `W_self` has fallen below threshold. The one `W_self` fire (Javier t7) is an artifact (see §5.3).
- **The *strong* barycenter fires come from instability, not disclosure** (Javier). Genuine disclosures (Marco, Linda) fire only **marginally** — because the early-round disclosure inflates the running baseline μ+2σ. This is honest behaviour on short (5–7 round) conversations.

---

## 5. Worked example

### 5.1 Primary — Marco (pid 15, gold-BDI 38, predicted 18)

Marco is the requested anchor (Daniel is not retrofittable). Clean monotonic disclosure over 7 rounds; **barycenter W₁ fires at t4**; **`W_self` never fires** — the two conventions shown side by side.

| round | W₁ (bary) | μ | σ | thr = μ+2σ | **fired** | W₁ (self) | new symptoms expressed |
|---|---:|---:|---:|---:|:--:|---:|---|
| t1 | 0.000 | – | – | – | · | 0.000 | Loss of energy, Sleep, Fatigue |
| t2 | 0.000 | 0.000 | – | – | · | 0.000 | (somatic deepens) |
| t3 | 0.333 | 0.000 | 0.000 | 0.000 | · | 0.667 | Sadness, Pessimism (grief: lost sister) |
| **t4** | **0.432** | 0.111 | 0.157 | **0.425** | **✓** | 0.410 | **Loss of pleasure, Self-dislike, Loss of interest, Worthlessness** |
| t5 | 0.399 | 0.191 | 0.194 | 0.580 | · | 0.173 | Loss of interest, Indecisiveness |
| t6 | 0.239 | 0.233 | 0.193 | 0.618 | · | 0.121 | Loss of pleasure, Concentration |
| t7 | 0.256 | 0.234 | 0.176 | 0.586 | · | 0.176 | Past failure, Self-criticalness, Worthlessness |

**Interpretable cause (t4).** Rounds 1–2 are purely somatic (energy/sleep/fatigue); t3 adds sadness/grief; at **t4 Marco discloses the anhedonia + withdrawal + worthlessness cluster** —

> *"No, I don't really have time for that. I just focus on work and staying tired. It's easier than trying to pretend I'm okay, I guess."*

— which moves the symptom-distribution centre of mass away from the somatic-dominated barycenter, so barycenter W₁ spikes and (just) clears μ+2σ.

**Honesty note.** The fire is *marginal* (0.432 vs 0.425). `W_self` misses the same disclosure entirely — the eRisk-native metric's failure mode on a front-loaded, short conversation.

### 5.2 Secondary — Linda (pid 14, gold-BDI 28, predicted 19)

Barycenter fires at **t4** on a self-identity/cognitive disclosure:

> *"I used to be the one with the answers. I had a plan. Now... I don't know. I feel like I've lost that part of myself."*

→ Past failure, Self-criticalness, Indecisiveness, Concentration appear. (Linda is central to the counterfactual — §6.3.)

### 5.3 Failure mode — Javier (pid 12): spurious fire

Honest negative result. Barycenter fires **strongly** at t4 (0.586 vs 0.234) and `W_self` at t7 — but **not because of disclosure**. The interviewer **loops** (t5/t6 are near-identical questions *and* answers) and the assessor drops then re-adds whole symptom clusters, so expressed mass oscillates **2 → 6 → 15 → 6 → 15 → 6 → 24**. The W₁ spikes track assessor instability, not the persona → a **false positive** that fails the interpretable-cause test.

---

## 6. Counterfactual — what W₁ could have changed in the live system

> **Is this analysis possible?** *Partially.* We can reconstruct, from logged data, the **signal** the live policy would have received. We **cannot** measure its **effect** on final BDI/symptom predictions, because (a) acting on the signal changes which questions are asked, and (b) the LoRA personas would then respond differently — an unobservable counterfactual rollout. Below, reconstructed signal is **fact**; downstream effect is **hypothesis**, clearly labelled.

### 6.1 The two live levers W₁ could have pulled

Both are real code paths:

1. **`W_align` hint → question selection.** `get_orchestrator_context` maps the latest `W_align` to a textual hint (`>1.0` = "high misalignment: interviewer probing different domains than expressed"; `>0.5` = "moderate"; else "good alignment"), injected into the orchestrator prompt ([orchestrator.py:246](../src/erisk_task1/orchestrator.py#L246)). Had POT been on, this hint would have nudged the next question toward under-probed domains.
2. **A `W_self`/barycenter fire → termination & re-probe.** Not currently wired, but the orchestrator's `should_terminate` allows early termination once the band is stable with confidence ≥0.5 after `min_turns=5` ([orchestrator.py:305](../src/erisk_task1/orchestrator.py#L305)). A fire-aware guard ("do not early-terminate within K rounds of a disclosure spike") is a plausible add.

### 6.2 Reconstructed `W_align` hint (fact)

W₁(interviewer attention `I(t)`, expressed `E(t)`), recomputed from logged profiles. Hint level per round:

| Persona | t1 | t2 | t3 | t4 | t5 | t6 | t7 | misalign rounds |
|---|---|---|---|---|---|---|---|---:|
| Javier | MOD | good | good | **HIGH** | **HIGH** | **HIGH** | MOD | 5 |
| Laura | MOD | MOD | MOD | MOD | good | good | good | 4 |
| Linda | MOD | good | good | MOD | good | — | — | 2 |
| Marco | **HIGH** | MOD | good | good | good | good | good | 2 |
| Maria | MOD | good | good | good | good | good | good | 1 |
| Maya | **HIGH** | **HIGH** | MOD | good | good | good | good | 3 |
| Noah | **HIGH** | MOD | MOD | good | good | MOD | good | 4 |
| Priya | **HIGH** | MOD | MOD | MOD | MOD | **HIGH** | **HIGH** | 7 |

**What the signal says (fact).**
- For most personas, `W_align` **decreases** as the interview proceeds — the (coverage-driven) interviewer *converges* onto the expressed profile (Marco 1.33 → 0.20; Maya 1.52 → 0.32). Healthy.
- **The two conversations where `W_align` stays/grows HIGH are exactly the two pathologies:** **Javier** (HIGH t4–t6 — the looping interviewer kept asking somatic questions while the persona had moved to self-criticism) and **Priya** (HIGH t6–t7 — a *minimal* case, gold 7, over-predicted to 11, where the interviewer kept probing depression domains the persona was not expressing).

**Hypothesised effect (unverified).** A live `W_align` hint would have signalled "stop re-probing the same domain" to the orchestrator at Javier t4–t6 (potentially breaking the loop) and "you are over-probing" at Priya t6–t7 (potentially curbing the over-prediction). Whether the orchestrator LLM would have acted correctly on the hint is untested.

### 6.3 Fire-vs-termination timing (fact) and the premature-cutoff hypothesis

| Persona | rounds | end reason | mass trajectory | rising at end? | fire | gap to fix |
|---|---:|---|---|:--:|---|---|
| **Linda** | **5** | **`min_turns` early-term** | 7,13,11,15,**19** | **yes** | bary **t4** | under-pred 28→19 |
| Marco | 7 | stable-band term | 3,6,9,13,16,17,**18** | yes | bary t4 | under-pred 38→18 |
| Maria | 7 | stable-band term | 5,7,11,15,13,12,**14** | yes (mild) | none | under-pred 40→14 |
| Laura | 7 | stable-band term | …,22,21 | no | none | ok (23→21) |

**The clearest actionable case is Linda (fact):** she terminated at exactly `min_turns = 5`, **one round after a disclosure spike (t4)**, with **symptom mass still rising (15 → 19)**. A fire-aware termination guard would have deferred cutoff and likely surfaced more symptoms. Linda is under-predicted by a full band (28 moderate → 19 mild), so this is the case with the most plausible upside.

**But the dominant error mode is *not* fixable by W₁ (fact + reasoning).** The severe personas Marco (38→18) and Maria (40→14) ran their course (7 rounds, mass plateauing) yet were massively under-predicted. Per the results analysis (§4 of [`task1_results_analysis.md`](task1_results_analysis.md)), severe under-prediction is an **assessor scoring** failure (the model under-scores score-3 items), not a conversation-length failure. Extending the interview on a W₁ fire would not change how the assessor scores the evidence already gathered. **W₁ is a coverage/timing instrument, not a severity-calibration instrument.**

### 6.4 Net counterfactual verdict

- **Reconstructable signal, real value:** the `W_align` hint cleanly separates the two pathological conversations (Javier loop, Priya over-probe) from the six healthy ones. As a *diagnostic*, W₁ earns its place.
- **Plausible but unverified live impact:** breaking Javier's loop, curbing Priya's over-prediction, and deferring Linda's premature cutoff. Each is a hypothesis requiring a re-run to confirm.
- **No impact on the headline error:** severe under-prediction is an assessor-calibration problem orthogonal to W₁.

---

## 7. Run 3's "ToM-calibrated" tail — what does it actually consume?

The official Run 3 is described as *mixed: `flat_minus_3` for pid 2–12, **ToM-calibrated** for pid 13–20* ([`task1_results_analysis.md`](task1_results_analysis.md) §1). Since this report establishes that no Wasserstein value existed at run time, it is worth asking precisely what the "ToM-calibrated" tail consumed. The answer reframes the §5 cross-calibration comparison.

### 7.1 The calibration consumes assessor confidence + somatic presence — not W₁

The "ToM calibration" is the **C1 + C2** correction in [`tom_corrections.py`](../src/erisk_task1/tom_corrections.py), wired at [pipeline.py:186–200](../src/erisk_task1/pipeline.py#L186):

- **C1 — confidence gate.** Drops scored items with assessor `confidence < 0.5`. *Consumes per-item assessor confidence + score.*
- **C2 — somatic boost.** Adds `+9` (scaled for partial coverage) when the somatic items {11,15,16,18,20,21} are under-covered **and** `gated_total ≥ 20`. *Consumes the somatic-item presence count + the gated total.*

Both read the **final `item_scores`** — not the ToM tracker's `E_profiles`, trajectory, or any Wasserstein distance. The *only* W₁ hook is the optional `walign_threshold` filter on C2 (tom_corrections.py:147), and it is **explicitly disabled** in the Run 3 config:

```yaml
# config/task1_colab_run3_tom.yaml
# walign_threshold: null  # W_align filter disabled (no benefit over standard C1+C2)
```

So the "ToM" label is conceptual (the somatic-item set is motivated by the *Somatic/Low-ToM* category hypothesis), not computational. The calibration never touches a Wasserstein distance.

### 7.2 …and in practice it changed almost nothing

Reconstructing C1 + C2 from the logged item-scores (the reconstructed final matches each logged BDI exactly):

| persona | gold | final BDI | C1 items gated | somatic evidence | C2 boost | **C1+C2 net Δ** |
|---|---:|---:|---:|---:|---:|:--:|
| Laura | 23 | 21 | 0 | 3/6 | 0 | **0** |
| Linda | 28 | 19 | 0 | 3/6 | 0 | **0** |
| Marco | 38 | 18 | 0 | 3/6 | 0 | **0** |
| Maria | 40 | 14 | 1 | 4/6 | 0 | **−1** |
| Maya | 6 | 10 | 0 | 3/6 | 0 | **0** |
| Noah | 5 | 7 | 0 | 3/6 | 0 | **0** |
| Priya | 7 | 11 | 0 | 0/6 | 0 | **0** |

**The tail calibration was a near-total no-op: 6/7 personas unchanged, Maria −1.** The serialized `correction` field reads `strategy=none, delta=0` because that is the *separate* standard post-hoc stage (`run3: "none"`, [pipeline.py:210–215](../src/erisk_task1/pipeline.py#L210)); the C1/C2 effect is folded into `raw_total` upstream and its audit goes only to the run log.

### 7.3 Two material consequences for the working notes

1. **C2 has a catch-22 on the cases it targets.** The somatic boost is meant to rescue severe patients who under-disclose somatic symptoms, but it requires `gated_total ≥ 20` — and the severe personas never reach 20 *precisely because* the assessor under-scores them (the §4 error mode). Marco (gold 38) gated to 18, Maria (gold 40) to 14; both below the boost threshold. The mechanism built for severe cases is structurally unable to fire on them. (C2 also requires somatic *absence*, yet every persona expressed 3–4/6 somatic items.)

2. **The §5 A/B comparison is confounded.** The head (pid 2–12) used `flat_minus_3` (Δ = −3, confirmed: persona8 21→18, persona11 17→14); the tail ≈ **uncorrected raw assessor scores**. So "ToM-calibrated vs `flat_minus_3`" is really **"no correction vs minus-3."** The §5 conclusion *"ToM-calibration helps moderate personas"* is more honestly **"not subtracting 3 helps moderate personas"** — for Laura (23) and Linda (28) `flat_minus_3` over-subtracts, and leaving the score untouched lands closer to gold. ToM played no role in that effect.

**Recommended correction to §5 of the results analysis:** restate the tail as an *uncorrected* baseline, and report the moderate-band win as evidence that `flat_minus_3` over-corrects the moderate band — not as evidence for ToM calibration.

---

## 8. Recommendations

1. **Working notes framing.** Present W₁ as a **post-hoc diagnostic** (results section). State plainly: *"Wasserstein distances were computed post-hoc from the logged per-turn symptom profiles; POT was unavailable during the official runs, so W₁ informed our analysis but not the live interviewing policy."* Do not claim live firing.
2. **Use the Marco slice (§5.1)** as the worked example, paired with the `W_self`-never-fires contrast; cite Javier (§5.3) as the honest failure mode.
3. **Future work, low cost / high value:** install POT and wire the **`W_align` hint** (already plumbed) — its retrofit value is demonstrated. Add a **fire-aware termination guard** (`min_turns` + "no early-term within K rounds of a fire") — would have helped Linda.
4. **Prefer the barycenter convention** for any live anomaly detector; turn-over-turn `W_self` is too easily defeated by the warmup on short conversations.
5. **Do not expect W₁ to fix severe under-prediction** — address that in the assessor/calibration stage.

---

## 9. Reproducibility

```bash
unset VIRTUAL_ENV
uv run python analysis/eda_task1/wasserstein_retrofit.py
# -> analysis/eda_task1/wasserstein_retrofit.json
#    (full per-round trajectories for personas 12-19, both conventions,
#     firing flags, disclosure annotations, and the highlighted example slices)
```

Method parameters: clinical ground metric `tom.get_cost_matrix`; L1-normalised (shape-only) balanced W₁; firing `μ + 2σ` over the strictly-prior trajectory; 3-round warmup; σ floor `+1e-6`. `W_align` reconstructed as W₁(`I(t)`, `E(t)`) with the live interpretation thresholds (0.5 / 1.0).
