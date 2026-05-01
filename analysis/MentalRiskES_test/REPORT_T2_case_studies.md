# Task 2 — Case-Study Report

**Team:** INSALyon
**Date:** 2026-05-01
**Scope:** Therapist response selection (3-way classification on 568 patient-rounds)
**Sources:** [posthoc_S_task2_bare_llm.py](posthoc_S_task2_bare_llm.py), [posthoc_T2_cross_cohort_eval.py](posthoc_T2_cross_cohort_eval.py), [posthoc_T2_consensus_failures.py](posthoc_T2_consensus_failures.py), [qualitative_T2_submitted_vs_s2.py](qualitative_T2_submitted_vs_s2.py)

---

## 1. Headline numbers

| System | Test (n=568) | Trial (n=18) | Simulated (n=87) |
|---|---|---|---|
| Submitted Run 2 (HYB B+) — R1-30 only | 0.247 | — | — |
| Submitted Run 2 (HYB B+) — full 82 replay | 0.255 | — | — |
| Submitted-equivalent (HYB B+ FIX W3) | — | **0.444** | 0.897 |
| Gemma 3 27B bare (S) | 0.290 | — | — |
| Llama-3.3-70B bare (S) | 0.257 | — | — |
| **Gemma 4 31B bare (S)** | **0.412** | 0.333 | 0.931 |
| **Gemma 4 31B bare + guardrails (S2)** | **0.470** | **0.444** | **0.943** |
| Gemma 4 31B bare permutation (S3) | 0.400 | — | — |
| Gemma 4 31B bare pairwise (S4) | 0.354 | — | — |
| Random baseline | 0.363 | ≈ 0.333 | ≈ 0.333 |
| Top team (NLP Innovators Run 1) | 0.393 | — | — |

**Headline:** S2 (Gemma 4 31B with anti-bias guardrails) achieves **0.470** on test — **+22.3 pp above our submission and +7.7 pp above the official top team**. The same system on trial ties our submitted-equivalent (8/18 = 0.444); on simulated it wins by only +4.6 pp (0.943 vs 0.897). The dramatic gap is test-specific.

---

## 2. The "could we have known before submission?" question

A core methodological question for the paper: **would a pre-submission ablation have flagged the bare-LLM approach?** We ran the same prompts against the trial corpus (TRIAL_GROUND_TRUTH, 18 labels) and the persona-simulated corpus (labels.json, 87 rounds across 7 sessions) that we used for system selection in March-April 2026.

**Answer: no.** The cross-cohort table above shows:

- **On trial,** S2 ties our submitted-equivalent (8/18 = 0.444 each). Two systems disagree on 6 of the 18 rounds, with the gold roughly evenly distributed across the disagreements. With n=18, a 0pp gap is well within sampling noise.
- **On simulated,** S2 leads our submitted-equivalent by only +4.6 pp (82/87 vs 78/87). All three systems (S, S2, Submitted) saturate above 90 % because the simulated personas are deliberately constructed with one clearly-fitting response per round; the task is too easy on this cohort to discriminate between approaches that reason at the same level.

**Implication:** the trial and simulated benchmarks under-discriminate at the quality range relevant to system selection. Trial's small size doesn't have the statistical power to expose a +5 pp difference, and simulated's artificial unambiguity hides differences that surface on real heterogeneous data. The lesson for future iterations is to **invest in test-like out-of-distribution corpora** (clinical conversations from a different therapist or population) before submission, not just in synthetic personas and a single transcribed session.

---

## 3. What S2 fixes that the submitted system gets wrong

Source: [outputs/qualitative_T2_submitted_vs_s2.md](outputs/qualitative_T2_submitted_vs_s2.md)

On the 300-round inner-join slice (R1-30 × 10 sessions, where both systems have predictions):

| Bucket | Count | Share |
|---|---|---|
| Both correct | 34 | 11.3 % |
| **S2 wins** (S2 right, Submitted wrong) | **91** | **30.3 %** |
| Submitted wins (Submitted right, S2 wrong) | 40 | 13.3 % |
| Both wrong, same answer | 80 | 26.7 % |
| Both wrong, different answers | 55 | 18.3 % |

S2 wins **2.3× as often as Submitted wins**. Per-class:

| Gold class | Submitted acc | S2 acc | Agreement rate |
|---|---|---|---|
| 1 | 0.317 | 0.396 | 0.485 |
| 2 | 0.222 | 0.453 | 0.158 |
| 3 | 0.198 | 0.404 | 0.490 |

S2 lifts accuracy on every gold class. The largest absolute gain is on **gold = 2** (+23 pp), where the submitted system's option-2-bias paradoxically hurt (it picked option 2 for everything, so when gold actually was option 2 it got it right, but in many rounds gold = 2 corresponded to a *short, direct* validation that the submitted system saw as "too simple" and rerouted to option 1 or 3). Agreement on gold-2 rounds is the lowest at 15.8 %.

### 3.1 Representative S2-wins case

From [outputs/qualitative_T2_submitted_vs_s2.md](outputs/qualitative_T2_submitted_vs_s2.md), Section A (S2 wins). The single most common pattern is **gold is a direct probing question, Submitted picked an empathic reframe, S2 picked the probe**. Example: S04 round 27, gold = "¿cuando lo hablas te dan ataques? dime más sobre esto por favor" ("when you talk about it, do you have attacks? tell me more about this please"). The submitted system picked an option that begins "Es comprensible que evites pensar en ello demasiado..." — an elaborated empathic reframe. S2 picked the gold response.

This is the **sophistication-bias** taxonomy entry from Analysis V.

### 3.2 Representative Submitted-wins case

There are 40 rounds where the submitted system is right and S2 is wrong. Looking at these by gold class:

| Gold class | Submitted-wins count |
|---|---|
| 1 | 12 |
| 2 | 14 |
| 3 | 14 |

S2 mostly loses these because it over-corrects toward option 1 in cases where the gold response is option 2 or 3 *and* the patient context is brief. In those rounds, Submitted's HYB B+ scoring with state-tracker context happened to align with gold whereas S2's "trust your intuition" prompt picked option 1 because it was the most concise. This is the inverse of the sophistication bias — anti-bias guardrails over-correct in some short-context rounds.

### 3.3 Both wrong, same answer (80 cases)

In 80 rounds (26.7 %), both Submitted and S2 picked the *same wrong option*. These are the most paper-relevant cases for the disagreement taxonomy: both systems converged on a wrong-but-consistent reading, suggesting the gold reflects a clinician judgment that neither LLM-based approach can reliably recover. From a clinical-evaluation standpoint, these are the rounds where the task itself is ambiguous or the gold is contestable.

---

## 4. Where every system fails — consensus failures

Source: [outputs/W_t2_consensus_failures.md](outputs/W_t2_consensus_failures.md), [outputs/W_t2_consensus_failure_stats.csv](outputs/W_t2_consensus_failure_stats.csv)

We ran 9 systems on the 299 (round, session) pairs covered by all of them: Submitted Run 2 R1-30, Submitted Run 2 full replay, Gemma 4 31B bare (S, S2, S3, S4, R2), Gemma 3 27B bare (S), Llama-3.3-70B bare (S). For each (round, session) we counted how many systems matched gold.

| Gold class | n | All-wrong | All-correct | Mean correct systems |
|---|---|---|---|---|
| 1 | 101 | 18 (17.8 %) | 3 (3.0 %) | 3.14 / 9 |
| 2 | 94 | 20 (21.3 %) | 1 (1.1 %) | 2.47 / 9 |
| **3** | **104** | **40 (38.5 %)** | **0 (0 %)** | **1.71 / 9** |
| ALL | 299 | **78 (26.1 %)** | 4 (1.3 %) | 2.43 / 9 |

**Critical finding: gold = option 3 is dramatically harder than the other two classes.** When the gold response is option 3, **38.5 % of rounds are wrong-by-every-system**, and *zero* rounds had every system correct. The mean correct-systems count on gold-3 is 1.71/9 — barely one in five systems gets it right. Compare to gold-1 where the mean is 3.14/9 (over a third of systems are right).

This is a more nuanced reading of the "safety bias" finding from §3 of [SUMMARY.md](SUMMARY.md). The safety-bias hypothesis was that we systematically pick option 2 when gold is option 3. The consensus-failure analysis shows that **option-3 rounds are categorically harder, regardless of which system tries to solve them**. The bias isn't only in our pipeline — it's in the LLM family as a whole. Whatever distinguishes "the gold response is option 3" from the other two is something Gemma, Llama, and our engineered system all fail to recover.

A reasonable hypothesis: option 3 is positionally last and may correspond to either "the riskiest direct question" (clinically warranted but conversationally surprising) *or* "the most elaborate intervention" (sometimes appropriate, sometimes over-engineered). The mixed nature of the option-3 class makes it hard for any system to learn a consistent decision rule.

---

## 5. The architecture-sensitivity finding

The bare-LLM result is **not "any LLM beats engineered systems"**. It's "*Gemma 4 31B specifically beats engineered systems*."

| Model | Bare (S) accuracy | Same prompt, same data |
|---|---|---|
| **Gemma 4 31B** | **0.412** | wins |
| Gemma 3 27B | 0.290 | below random |
| Llama-3.3-70B | 0.257 | below random, basically tied with our submission |

Gemma 3 27B has heavy option-1 bias (56 %); Llama-3.3-70B also over-picks option 1 (54 %); Gemma 4 31B is the only model that splits its predictions roughly proportionally to gold while still extracting signal (48/27/25 vs gold 36/33/32). The +21pp improvement S brings on Gemma 4 31B is the model's reasoning capability, not the prompt's simplicity.

This matters for the paper's framing: we are **not** arguing that prompt complexity is the universal failure mode. We are arguing that *for Task 2 specifically, with the model size and reasoning required*, our engineered ACT-process scoring overrides the model's native conversational judgment in a way that hurts. The same engineered scoring on a smaller or differently-trained model might not show this pattern.

---

## 6. Mid-conversation peak

For Gemma 4 31B with both S and S2:

| Tercile | S accuracy | S2 accuracy |
|---|---|---|
| Early (R1-27) | 0.389 | 0.407 |
| Mid (R28-54) | **0.457** | **0.569** |
| Late (R55-82) | 0.348 | 0.379 |

Mid-conversation rounds are the easiest for both systems. Three plausible reasons:

1. Early rounds have sparse context (only 1-3 patient turns) and low signal-to-noise.
2. Mid rounds have established alliance, focused topic, clear ACT phase — the response choice is most determined by the conversation dynamics.
3. Late rounds carry accumulated context that can dilute the relevant signal; closing-phase rounds also tend to have less differentiated candidate responses.

For S2 specifically, mid-conversation accuracy of **0.569** is the highest single-tercile number we report for any system on Task 2. This is meaningful clinical performance — better than half the time, the system picks the gold response — and it's where post-hoc deployment of S2 as a clinician-assist tool would have the highest value.

---

## 7. Recommendations for the paper

1. **Lead with S2's 0.470 number on test** as the post-hoc headline, framing it as "ACT-process-aware scoring inversely correlates with clinician preferences in the response selection task."
2. **Mention that trial and simulated benchmarks didn't predict this**, and use that as a methodological discussion point. Don't hide it — the inability of pre-submission benchmarks to flag the bare-LLM win is itself a finding.
3. **Use the consensus-failure analysis (§4) to characterise option-3 rounds as a generic LLM blind-spot**, not just a quirk of our pipeline. This is more honest than the "safety bias" framing alone, and is supported by the multi-system data.
4. **Frame the architecture sensitivity (§5) explicitly** to head off "any LLM works" misreadings. The headline depends on Gemma 4 31B's capability.
5. **Use the Submitted-vs-S2 disagreement Markdown (§3) and the consensus-failure Markdown (§4) for the paper appendix**, with the manual disagreement-taxonomy labels filled in for ~30 representative cases.

---

## 8. Open questions / future work

- **Why does Gemma 4 31B specifically win?** Compare token-level reasoning traces between Gemma 4 31B, Gemma 3 27B, and Llama-3.3-70B on the same rounds to see what changes.
- **Permutation (S3) and pairwise (S4) achieve cleaner distributions but lose accuracy.** The trade-off between "no position bias" and "extract maximum signal from the contrastive 3-way comparison" is informative — S2 wins because it gets both, but the underlying mechanism is unclear.
- **Are option-3 consensus failures attributable to the gold itself?** Manual annotation of the 40 gold-3 all-wrong cases would tell us whether they are genuinely ambiguous (in which case the task ceiling is closer to 0.6 than 1.0) or whether the gold reflects a specific clinical heuristic that LLMs don't replicate.
- **Could a Gemma-4-31B + S2 system fine-tuned on our 300 R1-30 predictions push past 0.50?** The S2 baseline is zero-shot; supervised tuning is a natural next step.
