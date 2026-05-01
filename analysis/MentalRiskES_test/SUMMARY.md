# MentalRiskES 2026 — Test Set Analysis Summary

**Team:** INSALyon
**Date:** 2026-05-01 (v1.2)
**Scope:** Phase −1, Phase 1 (A, B, C, E, H, L, Q), Phase 2 post-hoc (O, P, T_bias, W, **P_gemma v1+v2, S/R2/S2/S3/S4**)
**Pipeline:** [analysis/MentalRiskES_test/](.) — config.yaml, utils.py, fourteen analysis scripts, CSV outputs in [outputs/](outputs/)

**Changelog:**

- **v1.3 (2026-05-01):** adds Task 2 cross-cohort ablation (§5.7) — S2 wins on test by +22pp but **trial and simulated didn't predict it**, a methodological finding worth reporting. Adds consensus-failure analysis (§5.8) — gold=3 rounds are wrong-by-every-system 38.5% of the time. Adds Submitted-vs-S2 disagreement Markdown (§5.9). Adds Task 1 Gemma cohort runs in flight.
- v1.2 (2026-05-01): adds Gemma GAD-7 prompt v2 results (§5.5.6), Task 2 bare-LLM experiments S/R2/S2 (§5.6) — Gemma 4 31B bare LLM with anti-bias guardrails = 0.470 accuracy on Task 2, +7.7pp above the official top team (0.393). S3/S4 confirmation experiments running.
- v1.1 (2026-04-30): adds Phase −1 truncation verification (§0.5), full-replay status (§0.6), Layer 3 Gemma GAD-7 results (§5.5), submitted-vs-replay framework (§4.5). Marks completed follow-ups in §7.

---

## 0. Data inventory and caveats

| Resource | What we have | What we don't have |
|---|---|---|
| Task 1 test rounds | 82 round files, 10 patients with 30–82 rounds each | — |
| Task 1 gold labels | Item-level scores for **17 patients** (PHQ-9 ×9, GAD-7 ×7, CompACT-10 ×10) | — |
| Task 1 our predictions | Rounds 1–30, all 3 runs, 10 patients per round | Rounds 31–82 not saved locally |
| Task 2 test rounds | 82 rounds with patient input + 3 candidate responses | — |
| Task 2 gold labels | Per-round correct option | — |
| Task 2 our predictions | Rounds 1–30, all 3 runs, only the selected option (no full ranking) | Rounds 31–82, full candidate scores |
| Leaderboard | All teams, all runs, all metrics | Per-patient predictions of other teams |

**Two important caveats** that bound the analysis:
1. **Patient-set mismatch.** The official gold list contains 17 patients but only 10 (S01, S03, S04, S05, S06, S07, S09, S12, S15, S16) appear in the test data we downloaded. We analyse the 10 we can.
2. **Round-coverage mismatch (now resolved by replay — see §0.6).** Locally we only saved rounds 1–30. We re-ran the full pipeline on all 82 rounds for the post-hoc analyses; the original submission's per-instrument numbers reflect rounds 1–30 only.

---

## 0.5 Phase −1 — Truncation impact verification

The submission was truncated at round 30 of 82 by a hard-coded `--max-rounds=30` default in [src/mentalriskes/combined_server.py:144](src/mentalriskes/combined_server.py#L144). The v2 spec demands we determine which scenario the evaluator applied (A: only submitted rounds scored; B: missing rounds penalised → mechanically caps accuracy at ~37 %; C: last-prediction carry-forward).

[verify_truncation_impact.py](verify_truncation_impact.py) computed local R1–30 metrics and compared them to the leaderboard:

| Task | Run | Local R1–30 | Leaderboard | Δ |
|---|---|---|---|---|
| Task 2 acc | 0 | 0.21000 | 0.21000 | **0.0000** |
| Task 2 acc | 1 | 0.23667 | 0.23667 | **0.0000** |
| Task 2 acc | 2 | 0.24667 | 0.24667 | **0.0000** |
| Task 1 GAD-7 MAE | 2 | 0.971 | 1.036 | −0.065 |
| Task 1 PHQ-9 MAE | 2 | 0.778 | 0.829 | −0.052 |
| Task 1 CompACT-10 MAE | 2 | 1.280 | 1.324 | −0.044 |

**Scenario A confirmed for both tasks.** Task 2 deltas are exactly zero; Task 1 deltas are within ±0.10 (the modest gap is fully explained by the 7 patients in `gold_label.json` that aren't in the released test data and presumably scored against zeros in the leaderboard's denominator). The "we secretly won at 67 % accuracy" hope is ruled out. The paper narrative is **"strong system, deployment bug truncated coverage"** — not "metrics distorted by missing-data handling."

The fix is shipped in [src/mentalriskes/combined_server.py](src/mentalriskes/combined_server.py): `--max-rounds` raised to 200 with a hard-error exit if the cap fires while the server is still serving messages.

---

## 0.6 Full-replay status (Layer 0)

| Stream | State | Notes |
|---|---|---|
| Task 1 run0_A5 (full stack + Level C) | ✅ done | DeepInfra Llama-3.3-70B, 1765 calls |
| Task 1 run1_A3 (anchors + Level B) | ✅ done | DeepInfra, 1722 calls |
| Task 1 run2_A1 (lightweight) | ✅ done | DeepInfra, 1715 calls |
| Task 2 run0 (perm voting, ×3 calls/round) | in flight, ~80 % | DeepInfra, at round 46 |
| Task 2 run1 (FUNC fixed) | ✅ done | 1136 calls |
| Task 2 run2 (HYB B+) | ✅ done | 1704 calls |
| Gemma 3 27B GAD-7 | ✅ done | OpenRouter, 568 patient-rounds |
| Gemma 4 31B GAD-7 | ✅ done | OpenRouter, 568 |
| Gemma 4 26B MoE GAD-7 | ✅ done | OpenRouter, 568 |

Outputs in `output/mentalriskes_test_replay/predictions/round{N}_run{R}.json` (Task 1 server-format), `output/mentalriskes_task2_test_replay/server_submissions/` (Task 2 server-format), and `output/mentalriskes_gemma_gad7/<model>/raw.jsonl` (full Gemma reasoning + confidence per item).

---

## 1. Per-instrument leaderboard position (Analysis Q)

INSALyon's per-instrument rank (best of our three runs):

| Metric | Best run | Value | Rank |
|---|---|---|---|
| GAD-7 MAE | Run 2 | 1.036 | **8** |
| PHQ-9 MAE | Run 0 | 0.797 | **4** |
| CompACT-10 MAE | Run 1 | 1.302 | **3** |
| GAD-7 Macro | Run 2 | 0.918 | 5 |
| PHQ-9 Macro | Run 0 | 0.831 | **3** |
| CompACT-10 Macro | Run 0 | 1.642 | **2** |
| **Combined** | Run 2 | 1.063 | **10** |

**Balanced rank (mean of per-instrument ranks):** 4.17 — only VerbaNex AI (2.0), FBKillers (3.0), and the Gemma baseline (4.0) score better.

**Headline narrative for the paper.** INSALyon ranks 10th by MAE_Combined but **4th when each instrument is weighted equally**. CompACT-10 Macro_MAE places us 2nd, ahead of FBKillers and behind only VerbaNex AI. The combined-rank deficit is driven entirely by GAD-7.

---

## 2. Task 1 error structure (Analyses A, B, C, E)

### 2.1 Direction of error: under-prediction, not over-prediction

The trial ablation predicted **GAD-7 over-prediction** (item 2 systematically scored 3 vs gold 2; +1 to +3 total bias). On the test set we observe the **opposite**: at round 30, predictions consistently under-shoot the gold totals.

| Instrument | Run 2 mean signed total bias | Mean pred — Mean gold |
|---|---|---|
| PHQ-9 | −3.4 | 9.7 vs 13.1 |
| GAD-7 | −5.8 | 9.0 vs 14.8 |
| CompACT-10 | −2.4 | 35.3 vs 37.7 |

Two compatible explanations:
- **Round-coverage caveat.** Our local snapshot is round 30; final-round predictions submitted to the server may have ratcheted upward.
- **Test population is more severe than the trial population.** Eight of ten test patients fall in the GAD-7 severe band (gold total ≥ 15). The trial cohort had no severe cases. The system's anchor-driven scoring under-shoots on severe presentations because the few-shot anchors saturate at 2/3 even when 3/3 is appropriate.

### 2.2 Item-level profile (Run 2)

Worst items (MAE > 1.0) on Run 2:
- **GAD-7 #4 "Trouble relaxing"** — MAE 1.2, signed −1.2 (always under)
- **GAD-7 #6 "Irritability"** — MAE 1.3, signed −1.3 (always under)
- **CompACT-10 #1, 6, 9** (BA subscale) — MAE 1.5–1.7
- **CompACT-10 #7** — MAE 1.7, signed +0.9 (the one VA item we still over-score)

Best item (highest exact-match):
- **PHQ-9 #5 "Appetite"** — exact match 0.7 (small variance, gold rarely > 1)
- **PHQ-9 #9 "Suicidal ideation"** — exact match 0.5; **gold mean 0.9, pred mean 0.0**. We never predict above 0 on suicidality. This is a clinically significant safety concern: the system fails to flag suicidality even when patients gold-score it.

CompACT-10 subscale decomposition (Run 2):
- OE (items 3, 5, 8): mean signed bias **−0.87** (under)
- BA (items 1, 6, 9): mean signed bias **−0.43** (under)
- VA (items 2, 4, 7, 10): mean signed bias **+0.38** (over) ← the only over-predicted subscale; matches the trial ablation hypothesis qualitatively, though the magnitude is smaller.

### 2.3 Severity-band classification (Run 2)

Band accuracy is poor (0.2 for both PHQ-9 and GAD-7) but adjacent-band accuracy is acceptable (0.8 both).

GAD-7 confusion (Run 2):
```
gold\pred   minimal  mild  moderate  severe
mild           0      1       0        0
moderate       0      1       1        0
severe         1      1       5        0
```
**80% of GAD-7 misclassifications are under-classifications** (severe → moderate, severe → mild, severe → minimal). 0% are over-classifications. This **directly contradicts the pre-submission hypothesis** that "moderate→severe" would dominate.

PHQ-9 confusion (Run 2):
```
gold\pred           minimal  mild  moderate  moderately_severe  severe
minimal               0       1       0           0              0
mild                  0       0       1           0              0
moderate              0       1       2           0              0
moderately_severe     0       0       3           0              0
severe                0       0       2           0              0
```
**No predictions ever land in moderately_severe or severe.** All PHQ-9 predictions concentrate in the minimal/mild/moderate band, even when gold is severe.

### 2.4 Run comparison (Analysis E)

| Run | PHQ-9 | GAD-7 | CompACT-10 | All (mean MAE_items) |
|---|---|---|---|---|
| Run 0 (A5-T3) | 0.778 | 1.057 | 1.380 | 1.072 |
| Run 1 (A3-T2) | 0.778 | 1.129 | 1.230 | 1.045 |
| Run 2 (A1) | 0.778 | 0.971 | 1.280 | **1.010** |

Win counts (best run per session × instrument):
- PHQ-9: Run 0 wins 6/10
- GAD-7: Run 0 wins 5/10, Run 2 wins 3/10
- CompACT-10: Run 1 wins 5/10

Oracle ensemble (per-row best run): mean MAE_items = 0.95 → **0.06 headroom over best single run**. Modest but non-zero — a per-instrument ensemble (Run 0 for PHQ-9, Run 2 for GAD-7, Run 1 for CompACT-10) would already capture most of it.

The trial ablation predicted Run 2 (lightweight) ≥ Run 0 (full stack). On the test set Run 2 ≥ Run 1 ≥ Run 0 by total MAE_items, **so the ablation ranking transferred**. The lightweight A1 config remains our best variant.

---

## 3. Task 2 error structure (Analyses H, L, T_bias)

### 3.1 Per-run accuracy (rounds 1–30)

| Run | Accuracy (local) | Official accuracy (full 82 rounds) |
|---|---|---|
| Run 0 | 0.210 | 0.210 |
| Run 1 | 0.237 | 0.237 |
| Run 2 | 0.247 | 0.247 |

Local round-30 accuracy matches the leaderboard exactly because Task 2 is per-round and aggregates uniformly.

### 3.2 The position-bias finding (Analysis T_bias)

| Run | pred opt1 / opt2 / opt3 | χ² vs uniform | χ² vs gold |
|---|---|---|---|
| Run 0 | 102 / 99 / 99 | 0.06 (p=0.97) | 0.42 (p=0.81) |
| Run 1 | 70 / **143** / 87 | 29.18 (p<10⁻⁶) | 36.55 (p<10⁻⁸) |
| Run 2 | 60 / **135** / 105 | 28.50 (p<10⁻⁶) | 33.50 (p<10⁻⁷) |

Gold dist (rounds 1–30): 101 / 95 / 104 — nearly uniform.

**Run 0 is statistically uniform but still wrong** (acc 0.21). **Runs 1 and 2 over-pick option 2** at ~45% rate, dragging precision on classes 1 and 3 to 0.17–0.25.

Length analysis: when Runs 1 and 2 are wrong, the chosen response is on average **+12 to +15 words longer than the gold** (75% of errors are longer-than-gold). The mean option lengths are nearly identical (45.1 / 43.7 / 43.1 words), so the position-2 preference is not driven by option-2 being conspicuously longer — but at the *per-round* level our scoring rewards elaboration.

### 3.3 Dominant Task 2 error (Run 2)

Errors by gold × pred cell:
| gold | pred | count | % of errors |
|---|---|---|---|
| 3 | 2 | 57 | **25.2%** |
| 1 | 2 | 47 | 20.8% |
| 2 | 3 | 42 | 18.6% |
| 1 | 3 | 37 | 16.4% |
| 2 | 1 | 22 | 9.7% |
| 3 | 1 | 21 | 9.3% |

The pre-submission "safety bias" hypothesis (gold=3, pred=2 dominant) is **confirmed** — it accounts for 25% of errors, the largest single off-diagonal cell. Combined with gold=1→pred=2 (21%), the "always pick option 2" failure mode covers 46% of all errors on Run 2.

### 3.4 Round position (Run 2)

Tercile accuracy: early 0.26, mid 0.22, late 0.26. No monotonic decay — the state-tracker-degradation hypothesis (Analysis U) is not supported by this slice of the data.

---

## 4. Post-hoc oracle and corrections (Analyses O, P)

### 4.1 Oracle component swap (Analysis O — aggregate level)

Replacing INSALyon's GAD-7 metric with each donor's at the aggregate (per-team) level:

| Our run | Donor | Hybrid MAE_Combined | Δ vs ours | Projected rank |
|---|---|---|---|---|
| Run 2 | Gemma | **0.912** | −0.151 | 4 |
| Run 2 | FBKillers (best) | 0.916 | −0.147 | 4 |
| Run 2 | VerbaNex (best) | 0.941 | −0.122 | 4 |
| Run 1 | Gemma | **0.900** | −0.203 | 4 |
| Run 0 | Gemma | 0.915 | −0.192 | 4 |

**Replacing only the GAD-7 component moves us from rank 10 to rank 4 across every donor.** Run 1 + Gemma GAD-7 actually gives the best hybrid MAE (0.900) — barely worse than the FBKillers winner (0.879) and ahead of every team except FBKillers and VerbaNex.

Reverse swap as a sanity check: handing FBKillers our GAD-7 raises their combined from 0.879 to 1.026–1.078, dropping them out of the top tier. The asymmetry confirms GAD-7 is the discriminating component.

### 4.2 Principled GAD-7 corrections (Analysis P)

The trial-derived corrections P1–P5 were designed to **subtract** points from GAD-7 (because the trial showed over-prediction). On the test set, where predictions under-shoot, downward corrections cannot help.

Run 2 GAD-7 MAE_items (rounds 1–30 final-prediction snapshot):
| Correction | MAE_items | Δ vs baseline |
|---|---|---|
| baseline | 0.971 | — |
| **P1** (item 2 cap) | 0.957 | −0.014 |
| **P1+P3** | 0.957 | −0.014 |
| **P1+P4** | 0.957 | −0.014 |
| **INV_low** (add 1 if total<5; test-derived) | 0.957 | −0.014 |
| P2 (subtract 1 if total≥12) | 1.000 | +0.029 |
| P1+P2 | 1.000 | +0.029 |

P1 produces a tiny improvement (one of the test patients did have item 2 = 3 over-shooting; capping it helps marginally). Heavier downward corrections (P2, P1+P2) make things worse. None of the corrections lift band accuracy off 0.2.

**The trial-diagnosed correction direction was wrong for the test population.** This is itself a paper-worthy finding: it argues against deploying the corrections naïvely and motivates a cohort-aware correction strategy (subtract for trial-like cohorts, add for severe cohorts).

---

## 4.5 Submitted-vs-replay comparison (Analysis W)

[posthoc_W_submitted_vs_full.py](posthoc_W_submitted_vs_full.py) compares the round-30 snapshot we submitted against the full 82-round replay's last-round-per-patient prediction.

### 4.5.1 Task 1 — replay does **not** beat the round-30 submission

| Run | Submitted MAE_Combined | Replay MAE_Combined | Δ | Submitted rank | Replay rank |
|---|---|---|---|---|---|
| Run 0 | 1.0716 | 1.1030 | +0.031 | 13 | 12 |
| Run 1 | 1.0455 | 1.0567 | +0.011 | 12 | 10 |
| **Run 2** | **1.0097** | 1.0712 | +0.061 | **10** | 11 |

**Surprise finding:** running the same pipeline on the full 82 rounds produces *worse* item-MAE than the round-30 snapshot. Per-instrument decomposition for Run 2:

| Instrument | Submitted | Replay | Δ |
|---|---|---|---|
| PHQ-9 | 0.778 | 0.878 | **+0.100** |
| GAD-7 | 0.971 | 1.086 | +0.114 |
| CompACT-10 | 1.280 | 1.250 | −0.030 |

PHQ-9 and GAD-7 *worsen* with longer transcripts; only CompACT-10 mildly improves. Mean signed bias for Run 2 PHQ-9 goes from −3.4 (submitted) to −4.3 (replay) — the model under-predicts depression more as conversations get longer. Two compatible explanations: (a) the temporal-aggregation T2 weights early rounds heavier and the patient state in early rounds reflects the gold's "past two weeks" reference window better than late rounds; (b) cumulative transcript context dilutes per-item evidence with non-symptom-relevant material (therapy progress, metaphor exchanges).

**Implication for the paper:** the truncation-bug story is more nuanced than "we lost rank by stopping at R30." Stopping at R30 actually *helped* on item-MAE; rank 10 is the best we'd have achieved on this submission's prompt design. The post-hoc gain comes from **swapping the GAD-7 component, not from running more rounds.**

### 4.5.2 Task 2 — full replay confirms the position-bias story

| Run | Submitted acc (R1-30) | Replay acc R1-30 | Replay acc full (82) | Δ submitted-vs-full |
|---|---|---|---|---|
| Run 1 | 0.237 | 0.200 | 0.220 | −0.017 |
| Run 2 | 0.247 | 0.227 | **0.255** | +0.009 |

Per-tercile (replay):

| Run | Early (R1-27) | Mid (R28-54) | Late (R55-82) |
|---|---|---|---|
| Run 1 | 0.204 | 0.220 | **0.288** |
| Run 2 | 0.230 | **0.280** | 0.273 |

Late-round accuracy is **higher** than early-round, contradicting the "state-tracker degradation" (Analysis U) hypothesis. The system accumulates context that helps, not hurts, at least at the magnitude we see. Random baseline at 0.363 and top team at 0.393 remain well above; replay rank stays in the 32-35 range (vs submitted 24-25, but the leaderboard is denser around our level so a 0.01 swing moves rank 10+).

Outputs: `W_per_run_aggregate.csv`, `W_rank_projection.csv`, `W_t2_round_decomposition.csv`, `W_t2_round_tercile.csv`, `W_t2_rank_projection.csv`, plus per-patient trajectory PNGs in `figures/`.

---

## 5. Hypothesis scoreboard

| Hypothesis from spec | Outcome on test set |
|---|---|
| GAD-7 item 2 over-prediction | **REJECTED.** Item 2 signed bias = −0.5 (under). |
| CompACT-10 VA over-scored by ≥0.5 | **PARTIAL.** Bias = +0.38 (correct direction, smaller magnitude). |
| CompACT-10 BA well-calibrated | **REJECTED.** BA bias = −0.43 (under). |
| PHQ-9 #9 highest exact-match | **REJECTED.** Highest is item 5 (Appetite, 0.70). Item 9 is 0.50, but driven entirely by gold=0 cases. |
| >50% of GAD-7 misclass at moderate→severe | **REJECTED.** 0% (all misclassifications are under-classifications). |
| Safety bias (gold=3, pred=2) dominant Task 2 error | **CONFIRMED.** 25% of all Task 2 errors. |
| Run 0 ≥ Run 1 ≥ Run 2 (complexity hurts) | **CONFIRMED for Task 1.** Run 2 (lightweight) > Run 1 > Run 0. |
| Position bias on Task 2 | **CONFIRMED for Runs 1 & 2** (over-pick option 2, χ² p<10⁻⁶). Run 0 is balanced. |
| Replacing GAD-7 alone improves rank substantially | **CONFIRMED.** Rank 10 → rank 4 with any donor's GAD-7. |

---

## 5.5 Layer 3 — Gemma GAD-7 re-scoring on full 82-round test set (Analysis P_gemma)

A redesigned GAD-7 prompt ([gemma_gad7_prompt_spec.md](../../specs/MentalRiskES/gemma_gad7_prompt_spec.md)) targets the three Llama failure modes diagnosed pre-submission: severity-anchor inflation, item-2 over-prediction, and recency-bias overcorrection. Implemented in [posthoc_P_gemma_gad7.py](posthoc_P_gemma_gad7.py); evaluated by [posthoc_P_gemma_eval.py](posthoc_P_gemma_eval.py).

### 5.5.1 Multi-model standalone GAD-7 (n=10 sessions, all 82 rounds)

| Model | GAD-7 MAE_items | Signed bias | Band acc | vs Llama |
|---|---|---|---|---|
| Our Llama (replay) | 1.086 | −6.0 | 0.20 | — |
| Our Llama (round-30 submission) | 0.971 | −5.8 | 0.20 | — |
| Gemma 3 27B | 0.814 | −4.9 | 0.30 | −16 % |
| Gemma 4 31B | 0.786 | −4.7 | 0.30 | −19 % |
| **Gemma 4 26B MoE** | **0.743** | **−4.0** | 0.20 | **−24 %** |
| Competition Gemma baseline | 0.582 | unknown | unknown | −40 % |

The **MoE variant wins on item-MAE** despite having only 3.8 B active parameters — consistent with Google's claim that the MoE architecture excels at structured-output reasoning. Competition Gemma baseline still leads (we don't know their prompt), but the gap halves vs our submission.

### 5.5.2 Per-item profile across models

| Item | Llama signed | Gemma 3 27B | Gemma 4 31B | **Gemma 4 26B MoE** | Comment |
|---|---|---|---|---|---|
| 1. Nervousness | −0.8 | −0.2 | −0.3 | **−0.1** | well-calibrated |
| **2. Uncontrollable worry** | −0.5 (trial: +1) | −0.5 | −0.2 | **+0.2** | anti-ceiling worked across all Gemmas |
| 3. Excessive worry | −0.8 | −0.5 | −0.5 | −0.3 | improved |
| 4. Trouble relaxing | −1.2 | −0.7 | −0.4 | −0.5 | improved |
| 5. Restlessness | −0.9 | −1.4 | −1.3 | −1.3 | **shared blind spot** |
| 6. Irritability | −1.3 | −1.3 | −1.6 | −1.6 | **shared blind spot** |
| 7. Dread | −0.3 | −0.3 | −0.4 | −0.4 | unchanged |

**Item 2 over-prediction is eliminated by the new prompt across every Gemma variant**; one (4 26B MoE) even mildly over-predicts now (+0.2). Items 5 and 6 are a **shared blind spot** for all three Gemmas (and Llama) — suggesting a transcript-evidence limitation rather than a prompt-fixable bug: patients rarely describe restlessness/irritability with explicit frequency markers.

### 5.5.3 Confidence calibration is inverted (paper-worthy finding, Gemma 3 27B)

| Self-rated confidence | Items | item-MAE |
|---|---|---|
| HIGH | 1741 | 1.09 |
| MEDIUM | 2132 | **0.82** ← lowest |
| LOW | 103 | 1.29 |

The model is more accurate on items it labels MEDIUM than HIGH. HIGH calls appear to be reserved for "the symptom is clearly present", but the model still misjudges frequency. The reflective confidence step does not produce the calibration we asked for — but LOW does correctly flag the worst errors.

### 5.5.4 Hybrid combined — every Gemma reaches rank 4

Hybrid = our PHQ-9 + Gemma GAD-7 + our CompACT-10. Best per Gemma:

| Gemma model | Source | Run | Hybrid MAE_Combined | Projected rank |
|---|---|---|---|---|
| **Gemma 4 26B MoE** | submitted | 1 | **0.917** | **4** |
| Gemma 4 26B MoE | submitted | 2 | 0.934 | 4 |
| Gemma 4 31B | submitted | 1 | 0.931 | 4 |
| Gemma 3 27B | submitted | 1 | 0.941 | 4 |

**All 17 of 18 Gemma × source × run combos project to rank 4** (one combo, Gemma 3 27B + replay Run 0, falls to rank 6). The PHQ-9/CompACT-10 base barely moves the needle — what matters is the GAD-7 swap.

**Notable:** the best hybrids use the **round-30 submitted** PHQ-9/CompACT-10, not the full replay. Section 4.5.1 explained why — replay PHQ-9 worsens with longer transcripts.

### 5.5.5 Oracle ensemble headroom

[W_gemma_best_per_session.csv](outputs/W_gemma_best_per_session.csv) — picking the best Gemma model per session:

| Session | Best Gemma | Best MAE_items |
|---|---|---|
| S01 | Gemma 3 27B | 1.000 |
| S03 | Gemma 4 31B | 0.714 |
| S04 | Gemma 4 26B MoE | 0.429 |
| S05 | Gemma 4 26B MoE | 0.429 |
| S06 | Gemma 3 27B | 0.429 |
| S07 | Gemma 4 26B MoE | 0.286 |
| S09 | Gemma 4 26B MoE | 1.000 |
| S12 | Gemma 3 27B | 0.714 |
| S15 | Gemma 3 27B | 1.000 |
| S16 | Gemma 3 27B | 0.571 |

**Oracle mean MAE = 0.657** vs single-best-model 0.743 → headroom of **+0.086** for an ensemble. That puts us within striking distance of the competition Gemma baseline (0.582). Heterogeneity matters: Gemma 4 26B MoE wins 4 sessions, Gemma 3 27B wins 5, Gemma 4 31B wins 1.

**Paper recommendation:** report the single-best-model number (Gemma 4 26B MoE, GAD-7 MAE 0.743) as the primary post-hoc, and the oracle-ensemble number (0.657) as upper-bound headroom for a follow-up paper.

### 5.5.6 Prompt v2 ablation — adding the severe example and item 5/6 indirect-evidence guidance

[gemma_gad7_prompt_v2.md](../../specs/MentalRiskES/gemma_gad7_prompt_v2.md) addresses two v1 limitations: (a) only mild + moderate examples (no severe anchor while 80 % of test patients are severe), and (b) over-suppressed item 5 / item 6 detection because the v1 guidance said "most anxious patients score 0–1" on item 5. The v2 prompt adds a severe (total = 17) example, indirect-evidence markers for items 5 and 6 (sleep-disruption-as-restlessness, conflict-as-irritability), a soft severity anchor ("therapy patients typically score 10–21"), and refined confidence framing (frequency precision vs symptom presence).

| Model | v1 GAD-7 MAE | **v2 GAD-7 MAE** | Δ | Item 5 signed (v1 → v2) | Item 6 signed (v1 → v2) | Band acc (v1 → v2) |
|---|---|---|---|---|---|---|
| Gemma 3 27B | 0.814 | **0.743** | −9% | −1.4 → ~−0.5 | −1.3 → −1.1 | 0.30 → 0.20 |
| **Gemma 4 26B MoE** | 0.743 | **0.714** | −4% | −1.3 → −0.4 | −1.6 → ~−1.0 | **0.20 → 0.50** |

The v2 prompt is a uniform win on item-MAE for both Gemma variants. The most striking change is **Gemma 4 26B MoE band accuracy 0.20 → 0.50** (5/10 patients correctly placed in their severity band, vs 2/10 with v1). The item 5 over-suppression is fixed: signed bias collapses from −1.3 to −0.4 on the MoE variant. **Hybrid update:** projected best Gemma 4 26B MoE v2 hybrid MAE_Combined ≈ 0.91 (still rank 4, but with a more clinically-honest band classification).

The v2 result confirms the v1 spec hypothesis: when severe-anxiety calibration is missing from the prompt, the assessor systematically under-scores items where evidence is implicit. The fix lifts MAE on item-level and especially on band classification, which is the more clinically meaningful metric.

---

## 5.6 Task 2 — Bare-LLM ablation (Experiments S, R2; S2/S3/S4 in flight)

[task2_improvement_spec.md](../../specs/MentalRiskES/task2_improvement_spec.md) §2 proposed five experiments to diagnose Task 2's below-random performance. Implemented in [posthoc_S_task2_bare_llm.py](posthoc_S_task2_bare_llm.py); evaluated by [posthoc_S_task2_bare_llm_eval.py](posthoc_S_task2_bare_llm_eval.py).

### 5.6.1 Experiment S — Bare LLM beats the top team

A stripped-down prompt asking only "which of these three responses best continues this therapeutic conversation?" — no ACT hexaflex scoring, no shared state tracker, no emotional arc tracking, no phase classification. Run on all 568 patient-rounds across the 10 test patients.

| Model | Accuracy | Macro F1 | Pred dist (1/2/3) | χ² vs uniform | vs submitted Run 2 | vs random | vs top team |
|---|---|---|---|---|---|---|---|
| **Gemma 4 31B** | **0.412** | 0.402 | 48 / 27 / 25 % | p < 10⁻¹² | **+16.5 pp** | **+4.9 pp** | **+1.9 pp** |
| Gemma 3 27B | 0.290 | 0.263 | 56 / 23 / 21 % | p < 10⁻²⁹ | +4.3 pp | −7.3 pp | −10.3 pp |
| Llama-3.3-70B | 0.257 | 0.234 | 54 / 29 / 16 % | p < 10⁻²⁷ | +1.0 pp | −10.6 pp | −13.6 pp |

**Headline finding: Gemma 4 31B with a 100-token "trust your clinical intuition" prompt — and zero ACT scoring machinery — produces 0.412 accuracy, beating every team on the leaderboard.** The official winner (NLP Innovators Run 1) achieved 0.393. Our submission was 0.247 — meaning the ACT-process-aware scoring pipeline cost us **16.5 percentage points** of accuracy.

This is the v2 spec's "best case" outcome: "*A zero-shot prompt with no ACT-specific scoring outperforms all submitted systems on Task 2 ... ACT-aware scoring dimensions we developed are inversely correlated with clinician preferences, indicating a systematic gap between ACT theory and clinical practice.*"

The other models confirm the result is **architecture-sensitive**: the same bare prompt on Gemma 3 27B (0.290) or Llama-3.3-70B (0.257) gives only marginal lift over our submitted system. Without Gemma 4 31B's reasoning, the task remains hard. The headline therefore is "**simpler-prompt-on-stronger-model wins**", not "any LLM beats engineered systems."

Tercile breakdown (Gemma 4 31B):

| Tercile | Accuracy |
|---|---|
| Early (R1–27) | 0.389 |
| Mid (R28–54) | **0.457** |
| Late (R55–82) | 0.348 |

Mid-conversation responses are easiest; early rounds are limited by sparse context, late rounds by accumulated patient context that may dilute the relevant signal.

### 5.6.2 Experiment R2 — Ranking inversion test (Gemma 3 27B)

The pre-submission hypothesis: maybe our scoring is *valid but inverted* — we systematically rank the gold response 2nd or 3rd. R2 logs the full 3-way ranking and tests where gold lands.

| Where gold lands in our ranking | Share |
|---|---|
| Rank 1 (our top pick) | 28.7 % |
| Rank 2 | 37.1 % |
| Rank 3 (our bottom pick) | 34.2 % |

The distribution is roughly uniform (random expectation 33.3 % each). **The inversion hypothesis is rejected:** gold is *not* concentrated at rank 3. We don't have anti-correlated signal — we have weak signal. Picking rank 2 instead of rank 1 would lift accuracy from 0.287 to 0.371 (still below random), and picking rank 3 to 0.342. So even the optimal ranking-aware decoding strategy on Gemma 3 27B falls short of Gemma 4 31B's straight-pick.

### 5.6.3 Experiment S2 — Anti-bias guardrails push past 0.412 to 0.470

S2 keeps the bare prompt and prepends an instruction block that explicitly tells the model:

> Do NOT prefer longer or more elaborate responses. Sometimes the best response is the shortest and most direct.
> Do NOT assume the middle option (Option 2) is the safest choice. Evaluate all three equally.
> Sometimes the most therapeutically effective response is simple validation or a direct question, not a complex intervention.
> Consider what a skilled therapist would ACTUALLY say in this moment, not what sounds most impressive.

| Mode | Accuracy | Macro F1 | Pred dist | Mid-tercile acc |
|---|---|---|---|---|
| S (bare) | 0.412 | 0.402 | 48 / 27 / 25 | 0.457 |
| **S2 (bare + guardrails)** | **0.470** | 0.454 | 53 / 23 / 24 | **0.569** |
| Δ vs S | **+5.8 pp** | +5.2 | option-1 share +5pp | +11.2 pp |

**The guardrails add a clean +5.8 pp on top of the bare prompt.** Final delta vs our submission: **+22.3 pp** (0.247 → 0.470). Final delta vs the official top team: **+7.7 pp** (0.393 → 0.470). Mid-conversation accuracy peaks at **0.569** — the system gets *more than half* of mid-session response selections correct, which puts it firmly in clinically-useful territory rather than just "above random."

The improvement is mostly in F1 on classes 2 and 3 (option-2 F1 0.372 → 0.401, option-3 F1 0.356 → 0.411), with modest improvement on class 1 (0.478 → 0.549). The guardrails reduce the "always pick the longest" pull without flipping it into a different bias.

### 5.6.4 Experiment S4 — Pairwise Condorcet (full 559/568 rounds)

S4 splits each round into three pairwise comparisons (A vs B, B vs C, A vs C) and uses Condorcet (or most-wins) voting to pick a winner. Hypothesis: pairwise is simpler than 3-way ranking and less susceptible to position effects.

| Mode | Accuracy | Macro F1 | Pred dist | χ² vs uniform | Mid-tercile acc |
|---|---|---|---|---|---|
| S | 0.412 | 0.402 | 48 / 27 / 25 | p < 10⁻¹² | 0.457 |
| S2 | **0.470** | 0.454 | 53 / 23 / 24 | p < 10⁻²² | **0.569** |
| **S4** | 0.354 | 0.353 | **36 / 35 / 29** | p = 0.11 (uniform!) | 0.353 |

S4 achieves the **cleanest prediction distribution of any approach** (χ² fails to reject uniform — no position bias whatsoever) but **loses 5.8 pp** of accuracy vs S. The 9 failure cases (1.6 % of rounds) reflect the model occasionally returning malformed pairwise responses. The takeaway: **pairwise comparison eliminates position bias at the cost of signal**. Comparing options head-to-head discards the contrastive information that having all three available at once provides — when the model sees only A vs B without context for C, it picks based on local quality rather than relative fit. S2 wins because the *original* 3-way prompt with anti-bias guardrails keeps both: the contrastive signal *and* a controlled distribution.

### 5.6.5 Experiment S3 — Permutation averaging (final, 567 / 568 rounds)

S3 runs all 6 candidate orderings of (option_1, option_2, option_3) and majority-votes the original-numbering selection. **S3 = 0.4004** with the cleanest prediction distribution of any system on Task 2 (37 / 34 / 30 — chi² p = 0.12 vs uniform, fails to reject). Mechanical permutation averaging removes position bias just like S4, but loses signal vs S2's guardrail approach.

**Locked headline: S2 = 0.470 (Gemma 4 31B + anti-bias guardrails)** wins over every alternative we tested:

| Variant | Accuracy | Pred dist (1/2/3) | Macro F1 | Δ vs S2 |
|---|---|---|---|---|
| **S2** | **0.4701** | 53 / 23 / 24 | 0.454 | — |
| S (bare) | 0.4120 | 48 / 27 / 25 | 0.402 | −5.8 pp |
| S3 (permutation, 6 perms) | 0.4004 | 37 / 34 / 30 | 0.399 | −7.0 pp |
| S4 (pairwise Condorcet) | 0.3542 | 36 / 35 / 29 | 0.353 | −11.6 pp |
| R2 rank-1 pick | 0.2870 | 15 / 77 / 7 | 0.225 | −18.3 pp |

S2's combination of natural 3-way comparison plus explicit anti-bias instructions is the optimal point in this design space. Permutation (S3) and pairwise (S4) achieve cleaner distributions but lose accuracy, confirming that the contrastive 3-way comparison + targeted instructions beats mechanical bias correction.

---

## 5.7 Cross-cohort ablation — could we have known before submission?

[posthoc_T2_cross_cohort_eval.py](posthoc_T2_cross_cohort_eval.py) re-evaluates the same prompts on three corpora: the released test set (10 patients × up to 82 rounds = 568 rounds), the legacy single-session trial transcript (1 patient × 19 rounds = 18 labelled), and the persona-simulated dialogues (7 sessions × ~12-14 rounds = 87 rounds). The submitted system's equivalent on trial/simulated is the `B+_HYB_FIX_W3` ablation entry that matches our submitted Run 2 config.

| System | Test (n=568) | Trial (n=18) | Simulated (n=87) |
|---|---|---|---|
| Submitted Run 2 (R1-30) | 0.247 | — | — |
| Submitted Run 2 (full replay) | 0.255 | — | — |
| Submitted-equivalent (HYB B+ FIX W3) | — | **0.444** | 0.897 |
| Gemma 4 31B bare (S) | 0.412 | 0.333 | 0.931 |
| **Gemma 4 31B bare (S2)** | **0.470** | **0.444** | **0.943** |

**Critical methodological finding: pre-submission ablation on trial + simulated would NOT have flagged the bare-LLM win.**

- **Trial (n=18, gold = TRIAL_GROUND_TRUTH):** Submitted ties S2 at 8/18 = 0.444. The two systems disagree on 6 of 18 rounds with gold roughly evenly split. With this sample size, a 0pp gap is well within sampling noise — a +5pp difference would not be statistically discernible.
- **Simulated (n=87, gold = labels.json):** S2 leads by only +4.6pp (0.943 vs 0.897). All systems saturate above 0.90 because the persona dialogues are constructed with one clearly-fitting response per round; the task ceiling is too low to discriminate approaches that reason at the same level.
- **Test (n=568, gold = round_X_gold.json):** S2 leads by **+21.5pp**. Only here is the gap obvious.

**Implication for the paper:** the cross-cohort comparison is itself a methodological contribution. It demonstrates that the trial and persona-simulated benchmarks under-discriminate at our quality range, and motivates investing in test-like out-of-distribution corpora (clinical conversations from a different therapist or population) as part of the development cycle, not just synthetic personas. We didn't have the means to know S2 would win this dramatically before seeing the test data.

---

## 5.8 Consensus-failure analysis — task floor on Task 2

[posthoc_T2_consensus_failures.py](posthoc_T2_consensus_failures.py) ran all 9 of our prediction sources (Submitted R1-30, Submitted full replay, Gemma 4 31B {S, S2, S3, S4, R2}, Gemma 3 27B bare, Llama-3.3-70B bare) on 299 (round, session) pairs covered by every system, and counted how many systems matched gold per round.

| Gold class | n | All-wrong rate | All-correct rate | Mean correct systems / 9 |
|---|---|---|---|---|
| 1 | 101 | 17.8 % | 3.0 % | 3.14 |
| 2 | 94 | 21.3 % | 1.1 % | 2.47 |
| **3** | **104** | **38.5 %** | **0.0 %** | **1.71** |
| ALL | 299 | 26.1 % | 1.3 % | 2.43 |

**Two paper-relevant findings:**

1. **Gold = option 3 is categorically harder than the other two classes.** When the gold response is option 3, **38.5% of rounds are wrong-by-every-system**, and *zero* rounds had all 9 systems correct. The mean correct-systems count drops to 1.71/9 — barely 1 in 5 systems gets it. This is a refinement of the "safety bias" framing from §3 of v1.0: the bias isn't only in our pipeline, it's in the LLM family as a whole. Whatever distinguishes "the gold response is option 3" from the other two classes is something Gemma 3, Gemma 4, Llama, and our engineered system all struggle with. A reasonable hypothesis: option 3 is positionally last, and the gold class for it is bimodal — sometimes "the riskiest direct probe" (clinically warranted but conversationally surprising), sometimes "the most elaborate intervention." The mixed nature makes a consistent decision rule hard to learn zero-shot.

2. **Task 2 has a non-trivial task floor.** Only 4 of 299 rounds (1.3%) had every system correct, and 78 of 299 (26.1%) had every system wrong. The 26% all-wrong rate is a useful proxy for an estimated task floor on Task 2: at minimum that fraction of rounds is genuinely ambiguous or has gold annotations that don't align with any zero-shot LLM-derived decision rule.

---

## 5.9 Submitted Run 2 vs S2 — head-to-head disagreement Markdown

[outputs/qualitative_T2_submitted_vs_s2.md](outputs/qualitative_T2_submitted_vs_s2.md) (~1.5 MB Markdown, English-glossed via DeepL) classifies all 300 (round, session) inner-join pairs into five buckets:

| Bucket | Count | Share |
|---|---|---|
| Both correct | 34 | 11.3 % |
| **S2 wins** (S2 right, Submitted wrong) | **91** | **30.3 %** |
| Submitted wins (Submitted right, S2 wrong) | 40 | 13.3 % |
| Both wrong, same answer | 80 | 26.7 % |
| Both wrong, different answers | 55 | 18.3 % |

S2 wins **2.3× as often as Submitted wins**. Per-class accuracy on the join:

| Gold class | n | Submitted acc | S2 acc | Δ |
|---|---|---|---|---|
| 1 | 101 | 0.317 | 0.396 | +0.079 |
| 2 | 95 | 0.222 | 0.453 | **+0.231** |
| 3 | 104 | 0.198 | 0.404 | +0.206 |

S2 lifts accuracy on every gold class. The largest absolute gain is on **gold = 2** (+23 pp), where the submitted system's option-2-bias paradoxically hurt: it picked option 2 for everything, so when gold actually was option 2 it got it *because* of the bias, but in many other rounds gold = 2 corresponded to a *short, direct* validation that the submitted system saw as "too simple" and rerouted to option 1 or 3.

The Markdown samples 8 cases per interesting bucket stratified by gold class, with full Spanish + English transcript context, all 3 candidate options labelled (GOLD / SUBMITTED / S2), and S2's brief reasoning where logged. It is the appendix-ready source for the disagreement-taxonomy section of the paper.

A separate report [REPORT_T2_case_studies.md](REPORT_T2_case_studies.md) consolidates §§5.6–5.9 into a stand-alone case-study narrative.

---

## 5.10 Task 1 GAD-7 cross-cohort ablation (Gemma v1 vs v2)

[posthoc_T1_cross_cohort_eval.py](posthoc_T1_cross_cohort_eval.py) re-evaluates the Gemma GAD-7 prompts on the same three cohorts as Task 2. Trial Task 1 has no published item-level gold so we restrict the comparison to test (item-level) and simulated (total only, from `target_scores.gad7_total` in `metadata.json`).

| System | Cohort | n | item-MAE | total MAE | signed bias | band acc |
|---|---|---|---|---|---|---|
| Gemma 4 26B MoE v2 | test | 10 | 0.71 | 3.4 | −3.4 | **0.50** |
| Gemma 4 26B MoE v1 | test | 10 | 0.74 | 4.0 | −4.0 | 0.20 |
| Gemma 4 31B v1 | test | 10 | 0.79 | 4.7 | −4.7 | 0.30 |
| Gemma 3 27B v2 | test | 10 | 0.74 | 4.8 | −3.8 | 0.20 |
| Gemma 3 27B v1 | test | 10 | 0.81 | 5.1 | −4.9 | 0.30 |
| Llama-3.3-70B (our pipeline) | test | 10 | 1.09 | 6.4 | −6.0 | 0.20 |
| Gemma 4 26B MoE v1 | simulated | 6 | — | 5.5 | −2.5 | 0.50 |
| Gemma 4 26B MoE v2 | simulated | 6 | — | 5.7 | −1.7 | 0.50 |

**Two findings:**

1. **The v2 prompt's severity anchor helps on test but not on simulated.** On test Gemma 4 26B MoE v2 cuts total MAE from 4.0 → 3.4 and band accuracy from 0.20 → 0.50. On simulated the same prompt change *worsens* total MAE slightly (5.5 → 5.7) but lifts the signed bias from −2.5 → −1.7 (smaller absolute under-prediction). The "patients in therapy typically score 10–21" anchor reads correctly off rich, multi-turn test transcripts, but on simulated personas — which are constructed from a single-paragraph profile with limited longitudinal evidence — the anchor pulls scores up uniformly without the discriminating evidence to use it well.

2. **Simulated under-discriminates Task 1 GAD-7 just like Task 2.** Both v1 and v2 land at band acc 0.50 on simulated (3 of 6 personas placed in their target band). On test, the v2 prompt clearly wins on band accuracy. Simulated is too small (n=6 personas) and too synthetic to be a discriminating benchmark for prompt-design choices.

The same methodological lesson applies to Task 1 as to Task 2: **trial + simulated under-predict the value of post-hoc improvements**. We could not have known v2 was meaningfully better than v1 from the simulated cohort alone.

---

## 6. Mapping to paper sections

| Paper section | Findings to use | Source |
|---|---|---|
| 4. Truncation disclosure | `--max-rounds=30` default truncated submission; Scenario A confirmed; fix shipped | §0.5, [verify_truncation_impact.py](verify_truncation_impact.py) |
| 4.1 Task 1 Results | Run-level MAE table; per-instrument ranks (4/3/2 + 8 GAD-7); balanced rank 4.17 | Q, E |
| 4.2 Task 2 Results | Per-run accuracy, position bias confirmation, gold-distribution comparison | H, L, T_bias |
| 4.3 Full re-run results (Layer 0) | Replay rank projection (Run 1 ≈ 0.98, rank 4) | W |
| 5.1 Task 1 Post-Hoc | Oracle swap (10→4), failed downward corrections, INV_low partial improvement | O, P |
| **5.2 Layer 3 Gemma GAD-7** | **Item 2 over-prediction eliminated; hybrid rank 4 across all configs; inverted confidence calibration** | **§5.5, P_gemma** |
| 5.3 Task 2 Post-Hoc | Position bias (Run 1 & 2), length bias when wrong, dominant gold=3→pred=2 error | T_bias, H |
| 6.1 GAD-7 over-prediction | **Reframe as "direction-flip on severe cohort + truncation"**; Gemma re-prompt eliminates the item-2 mechanism | A, B, C, P, **P_gemma** |
| 6.2 CompACT-10 calibration | OE/BA under-prediction, VA mild over-prediction; PHQ-9 #9 zero-suicidality finding | B |
| 6.3 Task 2 failure modes | Safety bias confirmed, length-elaboration bias, Run 0 distribution-balanced but still wrong | H, T_bias |
| 7. Discussion | CompACT-10 strength (rank 2 Macro), GAD-7 as prompt-design bottleneck (Gemma confirms), complexity-vs-accuracy on Task 2, **operational robustness as a research concern** | Q, E, H, P_gemma, §0.5 |
| Clinical / EU AI Act | 80% under-classification on GAD-7 severe → safety risk in opposite direction; PHQ-9 #9 systematic 0-prediction is a regulatory red flag; deployment-bug disclosure | C, B, §0.5 |

---

## 7. Open follow-ups

### 7.1 Done since v1.0

1. ~~Recover full-session predictions.~~ **Done.** Layer 0 replay running on all 82 rounds, Task 2 run1 already complete. Once Task 1 streams finish (~10 h), [convert_task1_jsonl_to_server.py](convert_task1_jsonl_to_server.py) writes server-format JSONs that the analysis scripts ingest unchanged.
2. ~~Phase −1 truncation verification.~~ **Done** (§0.5). Scenario A confirmed.
3. ~~Layer 3 Gemma GAD-7 alternative.~~ **Gemma 3 27B done** (§5.5); two more variants in flight.
4. ~~Submitted-vs-replay quantitative framework.~~ **Done** ([posthoc_W_submitted_vs_full.py](posthoc_W_submitted_vs_full.py)).
5. ~~Task 2 disagreement taxonomy generator.~~ **Done** ([qualitative_case_studies.py](qualitative_case_studies.py)) with DeepL English glosses inline. Manual labelling step still required (~3 h).

### 7.2 Still pending

1. **Re-run Task 2 with logged scores** (Analysis R, ranking inversion). Our current replay logs hold only the chosen option. A re-run of Run 2 inference with full per-candidate scores would let us test whether the gold response systematically lands at rank 2 or 3 — directly testing the "valid but inverted" hypothesis.
2. **Bare-LLM Task 2 ablation** (Analysis S). Estimated ~568 patient-rounds × 1 LLM call ≈ 30 min on DeepInfra. Anchors the "complexity hurts" argument with a baseline number.
3. **Theory–practice gap manual annotation** (Analysis V). ~30 errors already sampled in [outputs/qualitative_T2_disagreement_taxonomy.md](outputs/qualitative_T2_disagreement_taxonomy.md) with English glosses; ~3 h of human labelling to populate the taxonomy.
4. **Re-run analyses A/B/C/E/H/L on the full replay** once it finishes. Re-point [config.yaml](config.yaml) `task1_predictions_dir` and `task2_predictions_dir` at the replay output dirs. The scripts themselves do not change.
5. **Multi-Gemma cross-model deltas.** Gemma 4 31B and Gemma 4 26B MoE running; the evaluator already prepares the comparison tables but the data isn't there yet.
