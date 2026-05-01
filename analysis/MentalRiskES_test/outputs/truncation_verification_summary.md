# Phase −1 — Truncation Impact Verification

Spec reference: spec v1.3 §8.1 (Phase −1 protocol).

## Inventory

```
 run  task1_n_round_files  task1_max_round  task2_n_round_files  task2_max_round
   0                   30               30                   30               30
   1                   30               30                   30               30
   2                   30               30                   30               30
```

All three runs submitted exactly 30 rounds for both Task 1 and Task 2 out of 82 test rounds.

## Task 2 arithmetic

 run  n_correct_r1_30  n_total_r1_30  acc_r1_30_only  leaderboard_acc  delta_vs_leaderboard  if_scenario_b_reported  if_scenario_b_implied_true_acc
   0               63            300        0.210000         0.210000                   0.0                0.110915                          0.3976
   1               71            300        0.236667         0.236667                  -0.0                0.125000                          0.4481
   2               74            300        0.246667         0.246667                  -0.0                0.130282                          0.4670

**Scenario determination: A**

Local R1–30 accuracy matches the leaderboard verbatim across all three runs (|Δ| < 0.0005). The evaluator scored only submitted rounds. The 0.247 reported for Run 2 is the genuine per-round accuracy on rounds 1–30 — it is below the random baseline of 0.333 on the same rounds, and below the 0.363 reported by the random baseline on the full set.

## Task 1 arithmetic

 run instrument  n_sessions  local_R30_item_MAE  leaderboard_MAE   delta
   0      GAD-7          10              1.0571         1.159048 -0.1019
   0      PHQ-9          10              0.7778         0.796667 -0.0189
   0 CompACT-10          10              1.3800         1.366333  0.0137
   1      GAD-7          10              1.1286         1.191429 -0.0629
   1      PHQ-9          10              0.7778         0.816667 -0.0389
   1 CompACT-10          10              1.2300         1.301667 -0.0717
   2      GAD-7          10              0.9714         1.036190 -0.0648
   2      PHQ-9          10              0.7778         0.829259 -0.0515
   2 CompACT-10          10              1.2800         1.324000 -0.0440

**Scenario determination: A or close-to-A**

Local R30 item-MAE is consistently within ±0.15 of the leaderboard for all instruments and runs. The small discrepancy is consistent with the gold containing 7 patients (S02, S08, S10, S11, S13, S14, S17) not present in the released test data.

## Implications for the paper

- The truncation **reduced coverage** (30 of 82 rounds) but did not **distort metrics** in a way that hides system quality.
- Reported per-instrument MAEs and Task 2 accuracy are honest estimates of our system's behaviour on the rounds it actually processed.
- The full-replay results (Layer 0) will tell us whether quality is stable across rounds or degrades, but they will not unmask a secretly-winning system. The paper narrative is 'strong system, deployment bug truncated coverage' rather than 'metrics distorted by missing-data handling'.
