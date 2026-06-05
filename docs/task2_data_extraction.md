# Task 2 — DUET paper data extraction

**Date:** 2026-05-31
**Team:** INSA-Lyon (5 runs R0–R4)
**Spec:** [specs/task-2/data-extraction-request.md](../specs/task-2/data-extraction-request.md)
**Companion JSON:** [docs/task2_data_extraction.json](task2_data_extraction.json)
**Local-data compute script:** [scripts/compute_task2_alert_metrics.py](../scripts/compute_task2_alert_metrics.py)
**Local-only JSON output:** [docs/task2_data_extraction_local.json](task2_data_extraction_local.json)

## Sources

- **Official:** `docs/eRisk_2026__Preliminary_results-with-Task3.pdf` (Tables 5–7 for Task 2).
- **Local re-scoring:** `runs/task2/train/decisions/run_{0..4}_round_{0000..0499}.json` against `data/eRisk-2026/.../risk_golden_truth_t2_2026.txt` (522 subjects: 91 positive, 431 negative).
- The `train/` folder name is misleading: the decision files there correspond to the **522 2026 test subjects** matched against the 2026 gold truth (matches `analysis/eda_task2/outputs/example_candidates.json`).

## Three spec-assumption corrections to flag in the paper

| Spec said | Actual |
|---|---|
| "field of 63" runs | **70** runs (17 teams, see Table 5) |
| "best of our portfolio = 0.76" for F1 | **0.80** (R0 and R4 tied) |
| "NDCG@100 ... 0.87 (already known)" as field max | **0.91** (DeepCare run 0 at 500 writings) |

These don't change extraction; they only mean the §5 phrasing the spec assumed should be re-checked against the corrected numbers.

---

## G1.1 — Ranking metrics per run at 500 writings (HIGH priority)

Source: PDF Table 7, INSA-Lyon rows at 500 writings.

| Run | P@10 | NDCG@10 | NDCG@100 |
|---|---|---|---|
| R0 | 1.00 | 1.00 | **0.89** |
| R1 | 1.00 | 1.00 | **0.89** |
| R2 | 1.00 | 1.00 | 0.85 |
| R3 | 1.00 | 1.00 | **0.89** |
| R4 | 1.00 | 1.00 | **0.89** |

### Bonus — trajectory at 1 / 100 / 250 writings

| Run | 1w P@10 | 1w NDCG@10 | 1w NDCG@100 | 100w P@10 | 100w NDCG@10 | 100w NDCG@100 | 250w P@10 | 250w NDCG@10 | 250w NDCG@100 |
|---|---|---|---|---|---|---|---|---|---|
| R0 | 0.80 | 0.87 | 0.64 | 1.00 | 1.00 | 0.84 | 1.00 | 1.00 | 0.87 |
| R1 | 0.80 | 0.87 | 0.64 | 1.00 | 1.00 | 0.84 | 1.00 | 1.00 | 0.87 |
| R2 | 0.90 | 0.81 | 0.55 | 1.00 | 1.00 | 0.80 | 1.00 | 1.00 | 0.83 |
| R3 | 0.80 | 0.87 | 0.64 | 1.00 | 1.00 | 0.84 | 1.00 | 1.00 | 0.87 |
| R4 | 0.80 | 0.87 | 0.63 | 1.00 | 1.00 | 0.84 | 1.00 | 1.00 | 0.87 |

**Stabilisation note:** R0/R1/R3/R4 reach P@10 = 1.00 and NDCG@10 = 1.00 by 100 writings and stay there; NDCG@100 climbs 0.64 → 0.84 → 0.87 → 0.89. R2 trails by ~0.04 NDCG@100 throughout. The "ranking quality stabilises early" framing is supported.

---

## G1.2 — Field-rank and field-leader values (LOW priority — only if field-comparison sentences are restored)

Source: PDF Tables 6 and 7. Field size = **70 runs** (17 teams).

| Metric | INSA-Lyon best | INSA-Lyon rank in field of 70 | Field leader |
|---|---|---|---|
| F1 | **0.80** (R0, R4) | tied at **5** of 70 | HUGETIME run 0 at **0.83** |
| NDCG@100 (500w) | **0.89** (R0, R1, R3, R4) | tied at **3** of 70 | DeepCare run 0 at **0.91** |
| ERDE_5 (lower is better) | **0.14** (R2) | tied at **23** of 70 | HUTECH-NLP run 3 at **0.08** |
| ERDE_5 — R1 specifically (the run referenced in spec, value 0.15) | 0.15 | tied at **30** of 70 | (same) |

Runs above us:

- **F1 > 0.80**: HUGETIME 0 (0.83), UNED-GELP 2 (0.82), NoirByte 0 (0.82), MindAILab 0 (0.82).
- **NDCG@100 > 0.89**: DeepCare 0 (0.91), MindAILab 0 (0.90). Nine runs share 0.89: 5 NoirByte runs + 4 INSA-Lyon runs.

**Editorial note:** Field-leader-team names were a source of past drafting errors. The values above are correct; whether to name HUGETIME / DeepCare / MindAILab in §5 text is an editorial choice, not required by these numbers.

---

## G1.3 — Decision-metric Table 5 columns (MEDIUM priority — unblocks `$^\ddagger$` cells)

Source: PDF Table 6, INSA-Lyon rows. All values direct from the official PDF (no `$^\ddagger$` needed).

| Run | P | R | F1 | ERDE_5 | ERDE_50 | latency_TP | speed | F_latency |
|---|---|---|---|---|---|---|---|---|
| R0 | **0.76** | **0.85** | 0.80 | 0.16 | **0.07** | 13.00 | 0.95 | **0.76** |
| R1 | **0.69** | **0.89** | 0.78 | 0.15 | 0.05 | 9.00 | 0.97 | **0.75** |
| R2 | **0.46** | **0.95** | 0.62 | 0.14 | **0.08** | 7.50 | 0.97 | **0.60** |
| R3 | **0.63** | **0.92** | 0.75 | 0.16 | **0.06** | 11.50 | 0.96 | **0.72** |
| R4 | **0.75** | **0.85** | 0.80 | 0.16 | **0.07** | 14.00 | 0.95 | **0.76** |

The columns the spec called out as `$^\ddagger$`-pending are now filled (P, R, F_latency for all 5 runs; ERDE_50 for R0/R2/R3/R4 — R1 = 0.05 was already filled).

---

## G2.1 — Per-run alert counts (HIGH priority — unblocks §5.1 ¶2)

Source: local re-scoring (`scripts/compute_task2_alert_metrics.py`).

| Run | Total alerts | True positives | False positives |
|---|---|---|---|
| R0 | **101** | 77 | 24 |
| R1 | **117** | 81 | 36 |
| R2 | 188 | 86 | 102 |
| R3 | 133 | 84 | 49 |
| R4 | 102 | 77 | 25 |

**Verification:** Each row exactly reproduces what Table 6's official P and R imply (e.g., R0: TP = R × 91 positives = 0.85 × 91 ≈ 77; total = TP / P = 77 / 0.76 ≈ 101). So the local data is the correct backing for the "101 alerts" figure.

**The "Run 0's final 101 alerts" claim is confirmed.** The Phase 1 / Phase 2 split sentence ("50 in Phase 1, 51 in Phase 2") is also consistent with the local data: 50 R0 alerts first fire in rounds ≤ 50, the remaining 51 fire in rounds 51–499.

---

## G2.2 — Median TP first-alert latency (MEDIUM priority — unblocks §5.4 ¶2 + abstract/§1.3/§6 framing)

**Two numbers per run depending on which authority is used.**

| Run | Official Table 6 (rounds) | Local re-scoring (rounds) |
|---|---|---|
| R0 | **13.00** | 11.0 |
| R1 | **9.00** | 7.0 |
| R2 | **7.50** | 5.5 |
| R3 | **11.50** | 9.5 |
| R4 | **14.00** | 12.0 |

Local values are exactly 2 rounds below official across all 5 runs — a consistent round-indexing offset (local files index rounds 0–499; official counts writings 1-indexed and likely adds the initial context post).

**Recommendation for the paper:** use the **official Table 6** values (13, 9, 7.5, 11.5, 14) since the paper cites Table 6 throughout. The "median 9 for R1 vs 13 for R0" sentence the spec asked about restores cleanly: R1 alerts a median of **4 rounds** earlier than R0 on true positives (9 vs 13). The same gap shows up at 11 vs 7 in the local re-scoring, so the framing holds either way.

---

## G2.3 — Per-subject latency examples (LOW priority — optional §5.1 ¶2 illustration)

Source: local re-scoring + `analysis/eda_task2/outputs/example_candidates.json`.

| Tag | Real ID | Gold | R0 alert round | R1 alert round | Gap (R0 − R1) | Role |
|---|---|---|---|---|---|---|
| Subject A | qMXSpL4 | 1 | 41 | 4 | 37 | TP, larger-gap exemplar (referenced as §1.1 example: sparse target text + community concern) |
| Subject B | kcdgN0X | 1 | 50 | 29 | **21** | TP, **median-gap exemplar** for the 28-subject cohort |
| Subject C | OvzSZuo | 0 | never | 7 | n/a | Control. R0 = TN; R1 = FP at round 7. Spec marked this row "n/a (control)". |
| Subject D | Pop2mTP | 1 | 116 | 21 | 95 | TP, very-large-gap example |

### Early-alert cohort summary statistics

**Definition:** TPs (gold = 1) where both R0 and R1 alert AND R1 alerts ≥ 5 rounds before R0.

- **Number of qualifying TPs:** **28** (out of 77 R0 TPs)
- **Median gap (rounds) for that set:** **21**
- Mean gap: 38.5
- Gap distribution: min = 5, p25 = 9, **p50 = 21**, p75 = 45.75, max = 154

These numbers let the §5.1 ¶2 paragraph be restored: "Across 28 TPs where R1 alerts at least 5 rounds before R0, the median gap is 21 rounds. Subject B (gap 21) is the median case; Subject A (gap 37) and Subject D (gap 95) sit further up the distribution."

---

## G2.4 — Per-run training-mean alert round (LOWEST priority) — **PARTIAL / UNVERIFIABLE**

| Run | Spec's previous draft value | What we have locally |
|---|---|---|
| R0 | (previously had 158) | **not reproducible** from current files |
| R1 | (previously had 114) | **not reproducible** from current files |

**Why this is a gap:** the prior-draft numbers 158 and 114 must come from a separate **training-simulation log over the 2025 train corpus (909 users)**, which is not in `runs/task2/`. The only decision files present (`runs/task2/train/decisions/run_*_round_*.json`) score the **522 2026 test subjects**, not the 909-user 2025 train set.

What we *can* report — mean first-alert round on the test cohort (for context only, **not** the training-mean the spec asked for):

| Run | Mean TP first-alert round (test) | Mean first-alert round across all alerts (test) |
|---|---|---|
| R0 | 36.58 | 51.83 |
| R1 | 30.59 | 46.94 |
| R2 | 33.08 | 68.27 |
| R3 | 34.74 | 71.53 |
| R4 | 39.47 | 56.16 |

**Recommendation:** either drop the train/test comparison from §5.4 ¶2 entirely, or re-run R0 and R1 over the 909-user 2025 training corpus to regenerate the 158 / 114 numbers. The current files can't substitute.

---

## Summary — what unblocks what

| Spec group | Status | What it unblocks |
|---|---|---|
| G1.1 | ✅ Filled (Table 7) | §5.2 placeholder paragraph |
| G1.2 | ✅ Filled (with 3 spec-assumption corrections) | Optional field-comparison framing |
| G1.3 | ✅ All 15 `$^\ddagger$` cells filled (Table 6) | Table 5 verification status |
| G2.1 | ✅ Filled — **R0 = 101 confirmed** | §5.1 ¶2 Phase 1/2 specificity |
| G2.2 | ✅ Filled (official + local both available; recommend official) | §5.4 ¶2 + abstract / §1.3 / §6 framing |
| G2.3 | ✅ Filled (4 subjects + 28-cohort summary) | §5.1 ¶2 illustrative paragraph |
| G2.4 | ⚠️ **Unverifiable from current files** | Recommend dropping train/test comparison or re-running over 2025 train data |
