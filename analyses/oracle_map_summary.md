# Task 3 Symptom-Routing Oracle MAP

Computed per `specs/task-3/analysis-oracle-map.md`. Inputs: per-symptom AP from `runs/task3_all_results.json` (option (b)); cross-checked against TREC-file recomputation for two (run, qrels) pairs.

## Headline

| qrels | oracle_MAP | best_actual_run | best_actual_MAP | headroom |
|---|---:|---|---:|---:|
| majority | 0.1911 | Ensemble | 0.159 | +0.0321 |
| unanimity | 0.1719 | LLM cascade | 0.134 | +0.0379 |

## Per-symptom oracle selection

| # | short name | majority pick | AP (maj) | unanimity pick | AP (unan) |
|---:|---|---|---:|---|---:|
| 1 | Wrap-up details | BiEnc baseline | 0.1591 | BiEnc baseline | 0.1271 |
| 2 | Order | BiEnc baseline | 0.1986 | Ensemble | 0.2013 |
| 3 | Remember appts | LLM cascade | 0.3298 | LLM cascade | 0.3162 |
| 4 | Avoid effortful tasks | BiEnc baseline | 0.2212 | BiEnc baseline | 0.1856 |
| 5 | Fidget hands/feet | Ensemble | 0.3092 | Ensemble | 0.3477 |
| 6 | Over-active drive | BiEnc baseline | 0.1176 | BiEnc baseline | 0.2470 |
| 7 | Careless mistakes | LLM cascade *tie* | 0.0000 | LLM cascade *tie* | 0.0000 |
| 8 | Sustain attention | CADRE full | 0.0347 | CADRE full | 0.0075 |
| 9 | Concentrate on speech | LLM cascade *tie* | 0.0000 | LLM cascade *tie* | 0.0000 |
| 10 | Misplace items | DepTransfer | 0.0001 | DepTransfer | 0.0001 |
| 11 | Distracted by noise | BiEnc baseline | 0.0185 | LLM cascade *tie* | 0.0103 |
| 12 | Leave seat | CADRE full | 0.4126 | Ensemble | 0.2518 |
| 13 | Restless/fidgety | BiEnc baseline | 0.2644 | BiEnc baseline | 0.3096 |
| 14 | Hard to unwind | Ensemble | 0.0188 | Ensemble | 0.0237 |
| 15 | Talk too much | LLM cascade | 0.2673 | LLM cascade | 0.2898 |
| 16 | Finish sentences | LLM cascade | 0.4425 | CADRE full | 0.2646 |
| 17 | Wait turn | CADRE full | 0.1238 | CADRE full | 0.1239 |
| 18 | Interrupt others | LLM cascade | 0.5213 | LLM cascade | 0.3881 |

## Per-run selection counts

| run | majority | unanimity |
|---|---:|---:|
| LLM cascade | 6 | 6 |
| CADRE full | 3 | 3 |
| Ensemble | 2 | 4 |
| DepTransfer | 1 | 1 |
| BiEnc baseline | 6 | 4 |

## Sanity checks

- Oracle ≥ best actual: majority=True, unanimity=True
- Pick counts sum to 18: majority=True, unanimity=True
- Expected majority counts (from `numerical-claims.md`): {'BiEnc_baseline': 6, 'HiPerT_full': 5, 'LLM_cascade': 4, 'Ensemble': 2, 'DepTransfer': 1}
- Actual majority counts: {'LLM_cascade': 6, 'HiPerT_full': 3, 'Ensemble': 2, 'DepTransfer': 1, 'BiEnc_baseline': 6}
- Item 13 ties: none recorded at machine precision.
- Cross-check BiEnc_baseline/majority: recomputed MAP 0.1165 vs cached 0.1165; per-symptom mismatches: 0.
- Cross-check LLM_cascade/unanimity: recomputed MAP 0.1343 vs cached 0.1343; per-symptom mismatches: 0.
