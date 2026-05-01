# Task 1 GAD-7 — Cross-cohort comparison

Gold sources: test = item-level `gold_label.json`; trial = (none, no item gold); simulated = `target_scores.gad7_total` (totals only).

**MAE_total** is reported uniformly across cohorts; item-level MAE is in the `GAD7_MAE_items` column for the test cohort only.

| System | Cohort | n | item-MAE | total MAE | signed bias | band acc |
| --- | --- | --- | --- | --- | --- | --- |
| gemma-4-26b-a4b-it v2 | test | 10 | 0.714 | 3.40 | -3.40 | 0.50 |
| gemma-4-26b-a4b-it v1 | test | 10 | 0.743 | 4.00 | -4.00 | 0.20 |
| gemma-4-31b-it v1 | test | 10 | 0.786 | 4.70 | -4.70 | 0.30 |
| gemma-3-27b-it v2 | test | 10 | 0.743 | 4.80 | -3.80 | 0.20 |
| gemma-3-27b-it v1 | test | 10 | 0.814 | 5.10 | -4.90 | 0.30 |
| Llama-3.3-70B (replay, our pipeline) | test | 10 | 1.086 | 6.40 | -6.00 | 0.20 |
| gemma-4-26b-a4b-it v1 | simulated | 6 | — | 5.50 | -2.50 | 0.50 |
| gemma-4-26b-a4b-it v2 | simulated | 6 | — | 5.67 | -1.67 | 0.50 |