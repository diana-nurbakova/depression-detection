# Early Prediction of Depression: State-of-the-Art Review for eRisk Challenge Participation

**The eRisk challenge has become the premier benchmark for early depression detection, yet recent editions reveal that traditional ML approaches like SVM with TF-IDF can match or outperform deep learning methods**—a critical insight for researchers planning novel contributions. This comprehensive review synthesizes eight years of eRisk methodology, LLM advances through 2025, and identifies concrete opportunities for competitive system development.

The challenge has evolved from simple chunk-based processing to sophisticated conversational detection with LLM personas, while evaluation has shifted toward metrics that jointly optimize earliness and accuracy. Successful approaches consistently combine sentence transformers with careful decision policies, though the field remains constrained by privacy concerns, demographic biases, and a persistent clinical validation gap.

---

## The eRisk challenge structure and evaluation methodology

The **Early Risk Prediction on the Internet (eRisk)** challenge, running since 2017 as a CLEF Lab, focuses on detecting mental health conditions from sequential social media posts before crisis points occur. Organized by researchers from the University of A Coruña, University of Santiago de Compostela, and Università della Svizzera italiana, the challenge has addressed depression, anorexia, self-harm, pathological gambling, and eating disorders.

Recent editions (2022-2024) structured tasks around three core paradigms: **early sequential detection** (processing posts chronologically to make timely predictions), **symptom-level ranking** (identifying sentences relevant to specific BDI-II depression symptoms), and **severity estimation** (predicting questionnaire responses for eating disorders via EDE-Q). The 2024 edition attracted **84 registered teams** with **17 submitting results** across 87 total runs.

The signature metric **ERDE (Early Risk Detection Error)** penalizes both errors and delays through a latency cost function: `lc_o(k) = 1 - 1/(1 + e^(k-o))`, where parameter *o* controls penalty steepness. **ERDE5** promotes rapid decisions (penalty approaches 1.0 quickly after 5 posts), while **ERDE50** tolerates moderate delays. Complementary metrics include **latency-weighted F1** (F_latency = F1 × speed factor) and standard precision/recall for classification tasks, with **MAP, NDCG, and P@10** for ranking tasks.

### Dataset characteristics shape viable approaches

Datasets derive primarily from Reddit, using self-disclosure methodology where users explicitly state diagnoses (e.g., "I was diagnosed with depression"). The 2024 Task 1 sentence ranking dataset contains **551,311 users** and **15.5 million sentences** averaging 17.98 words each. Task 2 (anorexia detection) included **92 anorexia users** (304.8 average posts) versus **692 controls** (489.6 average posts), highlighting the characteristic class imbalance that affects all eRisk tasks.

Ground truth creation relies on:
- **Self-disclosure extraction** for binary detection tasks
- **BDI-II (Beck Depression Inventory-II)** for depression severity (21 symptoms, 0-63 scale)
- **EDE-Q (Eating Disorder Examination Questionnaire)** for eating disorder severity (28 items, 4 subscales)

Data access requires signed usage agreements, with test data released sequentially post-by-post via REST API—a constraint that eliminates batch processing approaches and requires genuine streaming systems.

---

## Top-performing systems reveal surprising insights (2022-2024)

Analysis of winning approaches across three editions reveals a counterintuitive finding: **sophisticated deep learning often underperforms carefully tuned traditional methods**, though transformer embeddings remain essential for semantic representation.

### 2024 winners and their methods

For **Task 1 (depression symptom ranking)**, NUS-IDS achieved best results (MAP=0.375, NDCG=0.631) using an **ensemble of three pre-trained sentence transformers**: all-mpnet-base-v2, all-MiniLM-L12-v2, and all-distilroberta-v1, with expanded BDI symptom descriptions as queries. The approach emphasizes that embedding diversity outperforms single-model optimization.

For **Task 2 (anorexia detection)**, NLP-UNED achieved F1=0.79 and F_latency=0.75 using **Approximate Nearest Neighbors (ANN) combined with contrastive learning** to refine embeddings. Notably, Riewe-Perla achieved the best ERDE5 (0.07) and ERDE50 (0.02) using a hybrid recommender system (LightFM) with SBERT embeddings—demonstrating that recommendation algorithms can outperform pure classification approaches for early detection timing.

For **Task 3 (eating disorder severity)**, SCaLAR-NITK dominated using **SVM with Word2Vec embeddings** plus PCA and back-translation augmentation, winning 7 of 8 metrics despite the relative simplicity compared to transformer fine-tuning approaches.

### Consistent success patterns

| Strategy | Evidence | Recommended Application |
|----------|----------|------------------------|
| Sentence transformer ensembles | NUS-IDS (2024), Formula-ML (2023) top rankings | Symptom ranking tasks |
| SVM + TF-IDF | ELiRF-UPV achieved **perfect precision** (1.00) on gambling detection (2023) | Binary early detection |
| Contrastive learning for embeddings | NLP-UNED (2024) best F1 | Improving domain-specific representations |
| Dual-model architecture | Separate positive/negative classifiers improve ERDE >10% | When earliness critical |

Traditional BOW approaches—specifically TF-IDF with 5,000 features and SVM—achieved state-of-the-art results in 2022-2023, particularly for pathological gambling where ELiRF-UPV reached F1=0.938. NLPGroup-IISERB explicitly demonstrated that BOW matched or exceeded Longformer, BERT, and RoBERTa for depression detection when properly tuned.

---

## LLMs and transformers redefine depression detection capabilities

### Domain-specific pre-trained models outperform general transformers

**MentalBERT** and **MentalRoBERTa** represent the leading domain-adapted transformers, achieving F1 scores of **93.1% for binary depression recognition** and **76.7% accuracy** on the SWMH benchmark for psychological well-being classification. These models use continued pretraining on mental health-specific Reddit corpora with lexicon-guided masked language modeling.

Comparative studies consistently show RoBERTa-based models outperforming BERT baselines:

| Model | Dataset | Accuracy | F1-Score |
|-------|---------|----------|----------|
| RoBERTa | Reddit depression (632K tweets) | 98.1% | 0.97 |
| RoBERTa-BiLSTM hybrid | Twitter/Reddit | 99.4% | 0.973 |
| BERT baseline | Reddit/Twitter | 87.6% | 0.857 |
| MentalBERT | SWMH | 76.7% | — |

### Instruction-tuned LLMs achieve competitive performance

The landmark **Mental-LLM study** (Xu et al., 2024) demonstrated that instruction-fine-tuned models dramatically outperform zero-shot approaches. Mental-Alpaca and Mental-FLAN-T5 exceeded GPT-3.5 by **10.9% on balanced accuracy** and GPT-4 by **4.8%**, despite being 250× and 150× smaller respectively.

**MentalLLaMA** represents the first open-source instruction-following LLM for interpretable mental health analysis, trained on the **IMHI dataset (105K instruction samples)** covering 8 mental health tasks. The family includes MentalLLaMA-7B through 33B variants, approaching state-of-the-art discriminative methods on 7 of 10 test sets while generating clinician-quality explanations.

For diary-based detection, **fine-tuned GPT-3.5 with chain-of-thought prompting** achieved 90.2% accuracy and 0.685 F1, while separate studies report fine-tuned GPT-3.5/LLaMA2-7B reaching **~96% accuracy** on social media depression detection.

### LLM strategies specifically used in eRisk

Several teams have integrated LLMs into eRisk submissions:

- **MeVer-REBECCA (2024)**: Combined BGE-M3 transformer embeddings with GPT-4 prompt-based refinement for symptom ranking
- **BLUE (2023)**: Used ChatGPT to generate synthetic queries for BDI-II symptom expansion (mixed results—sometimes overly specific)
- **DS@GT (2025 pilot)**: Employed diverse LLMs conducting BDI-II assessments with JSON structured outputs, achieving 2nd place on the conversational leaderboard

The 2025 edition introduced a **pilot conversational task** using LLM personas fine-tuned on diverse user histories, released on Hugging Face. This paradigm shift requires participants to interact with personas within limited conversational windows to identify depressive symptoms—a fundamentally different challenge than post classification.

### Fine-tuning versus zero-shot: when each approach wins

Systematic comparison using LLaMA 3 reveals stark performance differences:

| Approach | Emotion Classification | Mental Health | Computational Cost |
|----------|----------------------|---------------|-------------------|
| Fine-tuning | **91%** | **80%** | High |
| Zero-shot | 49% | 68% | Low |
| Few-shot | <Zero-shot | <Zero-shot | Low |
| RAG | 40-68% | 40-68% | Medium |

**Fine-tuning is optimal** for complex multi-class classification, domain-specific terminology, and when accuracy is critical. **Zero-shot works** for simpler binary tasks, rapid prototyping, and resource-constrained settings—carefully crafted prompts can leverage pre-trained knowledge effectively.

The **MentalQLM (0.5B parameters)** demonstrates that lightweight approaches with dual LoRA adaptation can outperform larger counterparts on Dreaddit and MultiWD datasets by 2-4%, suggesting parameter efficiency remains viable.

---

## Methodological techniques for early detection systems

### Sequential decision-making determines challenge success

The core algorithmic challenge is deciding *when* to emit a prediction—too early risks false positives, too late incurs ERDE penalties. Successful stopping strategies include:

**Threshold-based approaches**: Emit decisions when prediction confidence exceeds θ, with dual thresholds for positive versus negative classifications. UNSL's "historic stop policy" uses rolling windows with adaptive thresholds that lower over time to ensure eventual decisions.

**Dual-model architecture** (Cacheda et al., 2019): Train separate Random Forest classifiers—one optimized for detecting depressed users, another for identifying non-depressed users. This approach improved state-of-the-art ERDE by >10% by allowing each classifier to specialize.

**Reinforcement learning**: Model the problem as MDP where states encode accumulated evidence and confidence, actions are {wait, classify_positive, classify_negative}, and rewards derive from ERDE. SARSA and DQN have been applied for learning optimal stopping policies.

### Temporal modeling captures depression trajectory

RNN variants remain effective for processing sequential posts:
- **BiLSTM** combined with RoBERTa achieves 99.4% accuracy and 97.3% F1
- **DABLNet** uses cross-attention between BiLSTM text processing and LSTM temporal features
- **BERT + LSTM hybrid** with 4 transformer layers plus LSTM head reaches 84.15% weighted F1

Key temporal features include posting frequency changes (>50% increase/decrease signals crisis risk), late-night activity patterns, and time-span distributions. Effective encoding uses sine/cosine transformations for cyclical time and personalized inter-post interval features.

### Feature engineering remains competitive with deep learning

**LIWC-based linguistic features** achieve strong standalone performance:
- Elevated first-person singular pronouns ("I", "me", "myself")
- Increased negative emotion and sadness words
- Higher absolutist language ("always", "never", "completely")
- Elevated anxiety words and negations

**Cognitive distortion detection** based on Beck's 15 patterns (filtering, polarized thinking, catastrophizing, etc.) shows BERT-based models achieving F1=0.62-0.88, comparable to trained clinicians (F1=0.63).

**Class imbalance handling** critically affects performance. SMOTE and variants (ADASYN, Borderline-SMOTE, K-SMOTE) help with moderate-dimensional features, while focal loss and Dice loss work better for deep learning. Note that SMOTE becomes less effective on high-dimensional embeddings—undersampling may be preferable.

---

## Ethical considerations constrain deployment possibilities

### Privacy requirements exceed platform consent

Simply agreeing to Reddit's terms does not constitute informed consent for mental health research. GDPR requires researchers to establish independent legal basis, with the "public interest" exemption demanding demonstrated societal benefit. Pseudonymization alone is insufficient—mental health data remains personal under GDPR even without identifiers.

**Re-identification risks** persist: studies generating synthetic clinical data found rare information from original data leaks into generated samples. The mental data protection gap means emotions and moods not related to diagnosed conditions may lack Article 9 protections.

### Documented biases undermine generalizability

A 2024 Nature Scientific Reports study across four populations found standard ML approaches "regularly present biased behaviors" across sex, ethnicity, nationality, age, income, and health conditions. A 2025 systematic review of 47 studies identified critical biases:
- **63.8% relied exclusively on Twitter**, only 17% combined platforms
- **>90% used English-language content only**
- **~80% used non-probability sampling**
- Only 23% explicitly addressed negation and sarcasm handling

Depression manifestation differs across genders and cultures, with multimodal features (acoustic, textual, visual) and inter-modal relations varying significantly between US and Chinese datasets.

### Clinical validation remains the critical gap

Most models train on self-disclosed diagnoses rather than clinical verification. The NLPxMHI systematic review found **only 4 studies** used external validation or out-of-domain testing. The CMD-1 deployment case study demonstrated that clinical teams determined false negatives are **20× more costly** than false positives—threshold decisions should be made by end-users, not model builders.

Successful clinical integration at Stanford reduced crisis response time from 10 hours to 10 minutes while maintaining human review of all AI-surfaced messages. The lesson: AI should aid but never replace clinical judgment.

---

## Research gaps reveal concrete opportunities

### Underexplored areas for novel contributions

1. **Contextualized multi-turn detection**: eRisk 2025's new paradigm analyzes full conversational contexts rather than isolated posts—few existing approaches model dialogue structure
2. **Cross-lingual transfer**: M3L Framework with LoRA achieves 88.7% English → 81.1% Chinese accuracy, but most languages lack any evaluation
3. **Missing modality robustness**: Real-world scenarios have incomplete data; handling graceful degradation remains unsolved
4. **Federated learning**: Privacy-preserving distributed training on mental health data has not been attempted for eRisk
5. **Active learning with clinical feedback**: Iterative improvement with clinician input represents unexplored territory

### Technical gaps in eRisk specifically

- **512-token limitations** of RoBERTa/BERT constrain longitudinal analysis—Longformer adoption remains limited
- **Symptom sparsity** in limited conversational windows challenges detection
- **Depression level estimation** achieves only ~40% hit rates in early editions
- **LLM persona interaction** (2025 pilot) has no established methodology

### Promising architectural directions

**Multimodal fusion** shows strong results: SBT-Net uses text-audio cross-attention, while EEG + interview audio integration improves accuracy by 4.7%. Text-guided cross-modal feature reconstruction handles missing modalities.

**Lightweight efficient models** like MentalQLM (0.5B) with dual LoRA demonstrate that massive parameters aren't required. This opens opportunities for on-device deployment and reduced computational barriers.

---

## Actionable recommendations for eRisk participation

### For symptom ranking tasks
Use **ensemble sentence transformers** (MiniLM, mpnet, distilroberta) with expanded BDI-II symptom descriptions as queries. Apply cosine similarity for initial ranking, optionally refine with LLM-based re-ranking. NUS-IDS's approach provides a strong template.

### For early detection tasks
Start with **SVM + TF-IDF baseline**—it may outperform complex deep learning. Then experiment with:
- Contrastive learning to refine domain-specific embeddings
- Dual-model architecture (separate positive/negative classifiers)
- Time-aware decision policies optimizing ERDE directly
- Custom loss functions addressing class imbalance

### For severity estimation tasks
Combine **sentence transformer embeddings** with classical ML (SVM, XGBoost, Random Forest). Apply PCA for dimensionality reduction and back-translation for data augmentation. BERTopic can capture user-level patterns effectively.

### Novel contribution opportunities
Consider **federated learning** for privacy-preserving training, **cross-lingual approaches** for underserved languages, **conversational strategies** for the LLM persona pilot task, or **explainable AI methods** that provide clinician-interpretable outputs. The gap between research performance and clinical validation represents perhaps the highest-impact opportunity for contributions that advance the field beyond benchmark optimization.

---

## Conclusion

The eRisk challenge represents a mature yet evolving benchmark where traditional ML methods remain surprisingly competitive against deep learning approaches. The key insight for new participants is that **sentence transformer ensembles with careful decision policies consistently outperform more complex architectures**, while **instruction-tuned LLMs offer a promising but underexplored direction**.

The most significant opportunities lie not in incremental accuracy improvements but in addressing fundamental limitations: cross-lingual generalization, multimodal integration, privacy-preserving training, and clinical validation. The 2025-2026 shift toward conversational LLM personas signals the organizers' recognition that real-world deployment requires moving beyond post classification toward interactive assessment.

For researchers entering this space, the ethical landscape demands as much attention as technical innovation. Systems achieving high benchmark performance may still exhibit demographic biases, privacy violations, and clinical invalidity that preclude responsible deployment. The ultimate measure of success will be whether detection systems actually improve patient outcomes—a question the field has largely avoided answering.