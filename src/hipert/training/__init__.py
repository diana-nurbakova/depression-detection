"""Encoder training infrastructure for HiPerT-ADHD.

Training stages:
- Stage A: Depression pre-training (BDI-Sen + eRisk 2025 T1)
- Stage B: ADHD silver-label fine-tuning (curriculum learning)
- Stage C: Cross-condition bridging (optional)

Encoder models: MentalRoBERTa, ClinicalBERT, all-mpnet-base-v2
"""
