# Task 2 — Cross-cohort comparison

Each cell = accuracy (n). Sources of gold: test = released `round_X_gold.json`; trial = `TRIAL_GROUND_TRUTH` (18 rounds, single session); simulated = `labels.json` per persona session.

| System | test | trial | simulated |
| --- | --- | --- | --- |
| Gemma 4 31B bare (S) | **0.412** (n=568) | **0.333** (n=18) | **0.931** (n=87) |
| Gemma 4 31B bare (S2) | **0.470** (n=568) | **0.444** (n=18) | **0.943** (n=87) |
| Gemma 4 31B bare (S3) | **0.400** (n=567) | — | — |
| Gemma 4 31B bare (S4) | **0.354** (n=559) | — | — |
| Submitted Run 2 (HYB B+, R1-30) | **0.247** (n=300) | — | — |
| Submitted Run 2 replay (HYB B+, full) | **0.255** (n=568) | — | — |
| Submitted-equivalent (HYB B+ FIX W3) | — | **0.444** (n=18) | **0.897** (n=87) |
| gemma-3-27b-it bare (S) | **0.290** (n=568) | — | — |
| llama-3.3-70b-instruct bare (S) | **0.257** (n=568) | — | — |