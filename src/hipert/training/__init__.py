"""Encoder training infrastructure for HiPerT-ADHD.

v1 (bi-encoder):
- Stage A: Depression pre-training (BDI-Sen + eRisk 2025 T1)
- Stage B: ADHD silver-label fine-tuning (curriculum learning)

v2 (cross-encoder):
- Single-stage distillation from LLM silver labels
- CORAL ordinal regression or ListMLE listwise ranking
- Leave-symptom-out 5-fold cross-validation

Encoder models: MentalRoBERTa, ClinicalBERT, all-mpnet-base-v2
"""
