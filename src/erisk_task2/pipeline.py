"""Main pipeline orchestration for training, evaluation, and live runs."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from erisk_task2.config import Task2Config
from erisk_task2.models import (
    DEFAULT_RUNS,
    NUM_SYMPTOMS,
    RunConfig,
    RunUserState,
    Thread,
    UserProfile,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction for a single thread (shared by train/eval/live)
# ---------------------------------------------------------------------------

def process_thread(
    thread: Thread,
    profile: UserProfile,
    encoder,
    symptom_scorer,
    thompson,
    config: Task2Config,
):
    """Process one thread and update user profile in-place.

    Handles Layer 1 (embeddings, symptoms), Layer 2 (sentiment, concern, position),
    and bandit updates. Heavy components (emotion, BERTopic, ToM LLM) are
    computed separately.
    """
    from erisk_task2.features.layer1 import update_embedding_running_mean
    from erisk_task2.features.layer2 import compute_reply_sentiment, detect_concern

    profile.rounds_seen += 1
    profile.last_active_round = thread.round_number
    profile.is_author_flags.append(thread.target_is_author)

    target_texts = thread.target_texts
    has_text = len(target_texts) > 0

    if not has_text:
        profile.target_silent_rounds += 1
        profile.symptom_activations.append(np.zeros(NUM_SYMPTOMS))
        profile.reply_sentiments.append(0.0)
        profile.concern_flags.append(False)
        profile.reply_depths.append(0)
        profile.thread_topic_sims.append(np.zeros(NUM_SYMPTOMS))
        return

    # -- Layer 1: Embeddings --
    embeddings = encoder.encode(target_texts)  # (n_texts, 1920)
    round_mean = embeddings.mean(axis=0)

    profile.embedding_sum, profile.embedding_weight = update_embedding_running_mean(
        profile.embedding_sum,
        profile.embedding_weight,
        round_mean,
        decay_lambda=config.embedding.decay_lambda,
    )

    # Symptom scores
    activations = symptom_scorer.score(round_mean)
    profile.symptom_activations.append(activations)

    # Accumulate texts
    profile.all_target_texts.extend(target_texts)
    for t in target_texts:
        profile.text_word_counts.append(len(t.split()))

    # Rolling buffer for BERTopic
    profile.rolling_text_buffer.extend(target_texts)
    buf_size = config.bertopic.rolling_buffer_size
    if len(profile.rolling_text_buffer) > buf_size:
        profile.rolling_text_buffer = profile.rolling_text_buffer[-buf_size:]

    # -- Layer 2: Reply sentiment --
    try:
        sentiment, _ = compute_reply_sentiment(thread)
        profile.reply_sentiments.append(sentiment)
    except Exception:
        profile.reply_sentiments.append(0.0)

    # Concern detection
    concern, _ = detect_concern(thread)
    profile.concern_flags.append(concern)

    # Reply depth
    depths = []
    parent_depth_map = {thread.submission_id: 0}
    for c in thread.comments:
        d = parent_depth_map.get(c.parent_id, 0) + 1
        parent_depth_map[c.comment_id] = d
        if c.is_target:
            depths.append(d)
    profile.reply_depths.append(int(np.mean(depths)) if depths else 0)

    # Thread topic similarity
    title_text = thread.title or thread.body or ""
    if title_text.strip():
        title_emb = encoder.encode([title_text])[0]
        topic_sim = symptom_scorer.score(title_emb)
        profile.thread_topic_sims.append(topic_sim)
    else:
        profile.thread_topic_sims.append(np.zeros(NUM_SYMPTOMS))

    # -- Bandit update --
    if profile.bandit_alphas is None:
        profile.bandit_alphas, profile.bandit_betas = thompson.init_posteriors()
    profile.bandit_alphas, profile.bandit_betas = thompson.update(
        profile.bandit_alphas, profile.bandit_betas, activations,
    )


def compute_final_features(
    profile: UserProfile,
    thompson,
    feature_mask: Optional[list[str]] = None,
) -> np.ndarray:
    """Compute full feature vector for a user after processing rounds."""
    from erisk_task2.classification.feature_assembler import assemble_feature_vector
    from erisk_task2.distances.wasserstein import compute_all_wasserstein

    # Wasserstein on symptom activations
    wasserstein_features = compute_all_wasserstein(
        symptom_activations=profile.symptom_activations,
        emotion_entropies=[0.0] * len(profile.symptom_activations),
        embedding_history=[],
        topic_entropies=[0.0] * len(profile.symptom_activations),
    )

    # Bandit features
    bandit_features = None
    if profile.bandit_alphas is not None and profile.symptom_activations:
        latest_act = profile.symptom_activations[-1]
        bandit_features = thompson.compute_features(
            profile.bandit_alphas, profile.bandit_betas, latest_act,
        )

    return assemble_feature_vector(
        profile,
        wasserstein_features=wasserstein_features,
        mahalanobis_features=None,
        tom_features=None,
        bandit_features=bandit_features,
        emotion_features=None,
        topic_features=None,
        feature_mask=feature_mask,
    )


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train_pipeline(config: Task2Config):
    """Offline training pipeline.

    1. Load training data
    2. Initialize feature extractors
    3. Process each user's threads (subsampled for speed)
    4. Extract feature vectors
    5. Fit Mahalanobis on control users
    6. Train classifiers with 5-fold CV
    7. Report metrics
    8. Save models
    """
    from erisk_task2.bandits.thompson import ThompsonSampler
    from erisk_task2.classification.classifiers import create_classifier
    from erisk_task2.data.loader import load_training_data
    from erisk_task2.decision.policy import compute_erde, compute_f_latency
    from erisk_task2.distances.mahalanobis import MahalanobisScorer
    from erisk_task2.features.layer1 import EmbeddingEncoder, SymptomScorer

    logger.info("=== TRAINING PIPELINE ===")

    output_dir = Path(config.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load data ----
    users, labels = load_training_data(config.training_data_dir, config.labels_path)
    logger.info("Loaded %d users, %d labels", len(users), len(labels))

    # ---- Step 2: Initialize feature extractors ----
    logger.info("Loading sentence transformers...")
    encoder = EmbeddingEncoder(
        model_names=config.embedding.models,
        device=config.embedding.device,
        batch_size=config.embedding.batch_size,
    )
    encoder.load()

    symptom_scorer = SymptomScorer(encoder)
    symptom_scorer.build_references()
    logger.info("Symptom references built (21 x %dd)", encoder.total_dim)

    thompson = ThompsonSampler(tau_active=config.symptom.activation_threshold)

    # ---- Step 3: Process all users ----
    logger.info("Processing %d users...", len(users))
    profiles: dict[str, UserProfile] = {}
    user_ids = sorted(users.keys())

    for uid in tqdm(user_ids, desc="Feature extraction"):
        threads = users[uid]
        profile = UserProfile(subject_id=uid)

        # Subsample threads for speed: keep up to 100 evenly spaced
        n = len(threads)
        if n > 100:
            indices = np.linspace(0, n - 1, 100, dtype=int)
        else:
            indices = range(n)

        for i in indices:
            process_thread(threads[i], profile, encoder, symptom_scorer, thompson, config)

        profiles[uid] = profile

    logger.info("All users processed. Extracting feature vectors...")

    # ---- Step 4: Extract feature vectors ----
    X_list = []
    y_list = []
    subject_ids = []

    for uid in user_ids:
        if uid not in labels:
            continue
        profile = profiles[uid]
        if profile.rounds_seen == 0:
            continue

        fv = compute_final_features(profile, thompson)
        X_list.append(fv)
        y_list.append(labels[uid])
        subject_ids.append(uid)

    X = np.stack(X_list)
    y = np.array(y_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(
        "Feature matrix: %d users x %d features (dep=%d, ctrl=%d)",
        X.shape[0], X.shape[1], int(y.sum()), int((y == 0).sum()),
    )

    # ---- Step 5: Fit Mahalanobis ----
    logger.info("Fitting Mahalanobis scorer...")
    mahalanobis = MahalanobisScorer(n_pca_components=config.mahalanobis.n_pca_components)
    mahalanobis.fit(X[y == 0], X[y == 1])
    mahalanobis.save(output_dir / "mahalanobis.pkl")

    # Inject Mahalanobis scores into feature matrix
    # Offset: embedding(1920) + sym_max(21) + sym_mean(21) + sym_stats(147) + lex(4)
    #   + sent(3) + concern(1) + conv(3) + thread_topic(21) + emotion(9) + bertopic(41)
    #   + wasserstein(72) = 2263
    maha_offset = 1920 + 21 + 21 + 147 + 4 + 3 + 1 + 3 + 21 + 9 + 41 + 72
    for i in range(len(X)):
        maha_scores = mahalanobis.score(X[i])
        X[i, maha_offset:maha_offset + 3] = maha_scores
        # Combined distributional score (next 2 slots)
        X[i, maha_offset + 3] = maha_scores[0]  # D_M_control
        X[i, maha_offset + 4] = X[i, maha_offset - 72:maha_offset].mean()  # W_mean

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Step 6: Cross-validated training ----
    logger.info("Training classifiers with %d-fold CV...", config.cv_folds)

    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=42)

    classifier_types = sorted(set(
        rc.classifier_type.value for rc in DEFAULT_RUNS
    ))

    results = {}

    for clf_type in classifier_types:
        logger.info("--- %s ---", clf_type)

        fold_f1s = []
        all_val_decisions = {}
        all_val_alert_rounds = {}

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            clf = create_classifier(clf_type)
            clf.fit(X_train, y_train)

            probs = clf.predict_proba(X_val)
            preds = (probs >= 0.5).astype(int)
            f1 = f1_score(y_val, preds)
            fold_f1s.append(f1)

            for j, idx in enumerate(val_idx):
                uid = subject_ids[idx]
                all_val_decisions[uid] = int(preds[j])
                if preds[j] == 1:
                    all_val_alert_rounds[uid] = max(1, int((1 - probs[j]) * 20))

            logger.info("  Fold %d: F1=%.4f", fold + 1, f1)

        mean_f1 = float(np.mean(fold_f1s))
        std_f1 = float(np.std(fold_f1s))

        eval_labels = {uid: labels[uid] for uid in all_val_decisions if uid in labels}
        erde5 = compute_erde(all_val_decisions, all_val_alert_rounds, eval_labels, 5)
        erde50 = compute_erde(all_val_decisions, all_val_alert_rounds, eval_labels, 50)
        f_lat = compute_f_latency(all_val_alert_rounds, eval_labels, all_val_decisions, 50)

        results[clf_type] = {
            "mean_f1": mean_f1, "std_f1": std_f1,
            "erde5": erde5, "erde50": erde50, "f_latency": f_lat,
        }

        logger.info(
            "  %s: F1=%.4f+/-%.4f  ERDE5=%.4f  ERDE50=%.4f  F_lat=%.4f",
            clf_type, mean_f1, std_f1, erde5, erde50, f_lat,
        )

    # ---- Step 7: Train final models on full data ----
    logger.info("Training final models on full dataset...")
    for clf_type in classifier_types:
        clf = create_classifier(clf_type)
        clf.fit(X, y)
        model_path = output_dir / f"classifier_{clf_type}.pkl"
        clf.save(model_path)
        logger.info("Saved %s -> %s", clf_type, model_path)

    # Save symptom references
    np.save(output_dir / "symptom_references.npy", symptom_scorer.reference_embeddings)

    # Save feature matrix for debugging
    np.savez_compressed(
        output_dir / "features.npz",
        X=X, y=y, subject_ids=np.array(subject_ids),
    )

    # Save results
    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- Summary ----
    logger.info("=== TRAINING COMPLETE ===")
    logger.info("Output: %s", output_dir)
    for clf_type, m in results.items():
        logger.info(
            "  %s: F1=%.4f  ERDE5=%.4f  ERDE50=%.4f  F_lat=%.4f",
            clf_type, m["mean_f1"], m["erde5"], m["erde50"], m["f_latency"],
        )


# ---------------------------------------------------------------------------
# Evaluation pipeline (round-by-round simulation)
# ---------------------------------------------------------------------------

def evaluate_pipeline(config: Task2Config):
    """Simulate round-by-round processing with decision policies on training data."""

    from erisk_task2.bandits.thompson import ThompsonSampler
    from erisk_task2.classification.classifiers import create_classifier
    from erisk_task2.data.loader import load_training_data
    from erisk_task2.decision.policy import apply_decision, compute_erde, compute_f_latency
    from erisk_task2.distances.mahalanobis import MahalanobisScorer
    from erisk_task2.features.layer1 import EmbeddingEncoder, SymptomScorer

    logger.info("=== EVALUATION PIPELINE ===")

    output_dir = Path(config.logging.output_dir)

    # Load data
    users, labels = load_training_data(config.training_data_dir, config.labels_path)

    # Load models
    encoder = EmbeddingEncoder(
        model_names=config.embedding.models,
        device=config.embedding.device,
        batch_size=config.embedding.batch_size,
    )
    encoder.load()

    symptom_scorer = SymptomScorer(encoder)
    ref_path = output_dir / "symptom_references.npy"
    if ref_path.exists():
        symptom_scorer.reference_embeddings = np.load(ref_path)
    else:
        symptom_scorer.build_references()

    thompson = ThompsonSampler(tau_active=config.symptom.activation_threshold)

    # Load classifiers
    classifiers = {}
    for rc in DEFAULT_RUNS:
        ctype = rc.classifier_type.value
        model_path = output_dir / f"classifier_{ctype}.pkl"
        if model_path.exists() and ctype not in classifiers:
            clf = create_classifier(ctype)
            clf.load(model_path)
            classifiers[ctype] = clf

    if not classifiers:
        logger.error("No trained classifiers found in %s. Run 'train' first.", output_dir)
        return

    mahalanobis = MahalanobisScorer()
    maha_path = output_dir / "mahalanobis.pkl"
    if maha_path.exists():
        mahalanobis.load(maha_path)

    # Initialize state
    run_states: dict[int, dict[str, RunUserState]] = {}
    for rc in DEFAULT_RUNS:
        run_states[rc.run_number] = {uid: RunUserState() for uid in users}

    profiles: dict[str, UserProfile] = {
        uid: UserProfile(subject_id=uid) for uid in users
    }

    max_rounds = max(len(threads) for threads in users.values())

    # Process round by round, evaluate every eval_step rounds
    eval_step = max(1, max_rounds // 50)

    logger.info(
        "Simulating %d rounds (step=%d) for %d users, %d runs",
        max_rounds, eval_step, len(users), len(DEFAULT_RUNS),
    )

    maha_offset = 1920 + 21 + 21 + 147 + 4 + 3 + 1 + 3 + 21 + 9 + 41 + 72

    for round_num in tqdm(range(0, max_rounds, eval_step), desc="Eval rounds"):
        # Process threads
        for uid in users:
            threads = users[uid]
            if round_num >= len(threads):
                continue
            process_thread(
                threads[round_num], profiles[uid],
                encoder, symptom_scorer, thompson, config,
            )

        # Classification + decision for each run
        for rc in DEFAULT_RUNS:
            clf = classifiers.get(rc.classifier_type.value)
            if clf is None:
                continue

            for uid in users:
                state = run_states[rc.run_number][uid]
                if state.alert_emitted:
                    continue

                profile = profiles[uid]
                if profile.rounds_seen == 0:
                    continue

                fv = compute_final_features(profile, thompson, feature_mask=rc.feature_mask)
                fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)

                if mahalanobis.pca is not None:
                    maha = mahalanobis.score(fv)
                    fv[maha_offset:maha_offset + 3] = maha

                prob = float(clf.predict_proba(fv.reshape(1, -1))[0])
                decision, state = apply_decision(prob, round_num, rc, state)
                state.last_score = prob
                run_states[rc.run_number][uid] = state

    # Report metrics
    logger.info("=== EVALUATION RESULTS ===")
    eval_results = {}

    for rc in DEFAULT_RUNS:
        decisions = {}
        alert_rounds = {}

        for uid in users:
            if uid not in labels:
                continue
            state = run_states[rc.run_number][uid]
            decisions[uid] = 1 if state.alert_emitted else 0
            if state.alert_emitted and state.alert_round is not None:
                alert_rounds[uid] = state.alert_round

        eval_labels = {uid: labels[uid] for uid in decisions if uid in labels}

        erde5 = compute_erde(decisions, alert_rounds, eval_labels, 5)
        erde50 = compute_erde(decisions, alert_rounds, eval_labels, 50)
        f_lat = compute_f_latency(alert_rounds, eval_labels, decisions, 50)

        tp = sum(1 for u in eval_labels if eval_labels[u] == 1 and decisions[u] == 1)
        fp = sum(1 for u in eval_labels if eval_labels[u] == 0 and decisions[u] == 1)
        fn = sum(1 for u in eval_labels if eval_labels[u] == 1 and decisions[u] == 0)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        mean_ar = float(np.mean(list(alert_rounds.values()))) if alert_rounds else 0

        eval_results[f"run_{rc.run_number}"] = {
            "classifier": rc.classifier_type.value,
            "f1": f1, "precision": precision, "recall": recall,
            "erde5": erde5, "erde50": erde50, "f_latency": f_lat,
            "alerts": len(alert_rounds), "mean_alert_round": mean_ar,
        }

        logger.info(
            "Run %d (%s): F1=%.4f P=%.3f R=%.3f ERDE5=%.4f ERDE50=%.4f "
            "F_lat=%.4f alerts=%d mean_round=%.1f",
            rc.run_number, rc.classifier_type.value,
            f1, precision, recall, erde5, erde50, f_lat,
            len(alert_rounds), mean_ar,
        )

    with open(Path(config.logging.output_dir) / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)


# ---------------------------------------------------------------------------
# Live competition pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config: Task2Config):
    """Live competition pipeline against eRisk server (Section 17.2)."""

    from erisk_task2.bandits.thompson import ThompsonSampler
    from erisk_task2.classification.classifiers import create_classifier
    from erisk_task2.data.loader import parse_server_response
    from erisk_task2.decision.policy import apply_decision
    from erisk_task2.distances.mahalanobis import MahalanobisScorer
    from erisk_task2.features.layer1 import EmbeddingEncoder, SymptomScorer
    from erisk_task2.server.client import ERiskClient

    logger.info("=== LIVE PIPELINE ===")

    output_dir = Path(config.logging.output_dir)

    # Initialize client
    client = ERiskClient(config)
    client.initialize()

    # Load feature extractors
    encoder = EmbeddingEncoder(
        model_names=config.embedding.models,
        device=config.embedding.device,
        batch_size=config.embedding.batch_size,
    )
    encoder.load()

    symptom_scorer = SymptomScorer(encoder)
    ref_path = output_dir / "symptom_references.npy"
    if ref_path.exists():
        symptom_scorer.reference_embeddings = np.load(ref_path)
    else:
        symptom_scorer.build_references()

    thompson = ThompsonSampler(tau_active=config.symptom.activation_threshold)

    # Load classifiers
    classifiers = {}
    for rc in DEFAULT_RUNS:
        ctype = rc.classifier_type.value
        model_path = output_dir / f"classifier_{ctype}.pkl"
        if model_path.exists() and ctype not in classifiers:
            clf = create_classifier(ctype)
            clf.load(model_path)
            classifiers[ctype] = clf

    # Load Mahalanobis
    mahalanobis = MahalanobisScorer()
    maha_path = output_dir / "mahalanobis.pkl"
    if maha_path.exists():
        mahalanobis.load(maha_path)

    maha_offset = 1920 + 21 + 21 + 147 + 4 + 3 + 1 + 3 + 21 + 9 + 41 + 72
    round_number = client.current_round

    while True:
        logger.info("--- Round %d ---", round_number)
        t0 = time.monotonic()

        # 1. GET discussions
        response_data = client.get_discussions()
        if response_data is None:
            logger.error("GET failed, retrying in 30s...")
            time.sleep(30)
            continue

        if len(response_data) == 0:
            logger.info("Empty response — all rounds complete")
            break

        client.log_server_response(response_data)
        threads = parse_server_response(response_data)

        if round_number == 0:
            client.capture_master_list(threads)

        # 2. Feature extraction
        for uid, thread in threads.items():
            profile = client.profiles.get(uid)
            if profile is None:
                continue
            process_thread(thread, profile, encoder, symptom_scorer, thompson, config)

        # 3. Classification + Decision per run
        for rc in DEFAULT_RUNS:
            clf = classifiers.get(rc.classifier_type.value)

            for uid in client.master_user_list:
                state = client.run_states[rc.run_number][uid]
                profile = client.profiles[uid]

                if state.alert_emitted:
                    if profile.rounds_seen > 0 and clf is not None:
                        fv = compute_final_features(profile, thompson, feature_mask=rc.feature_mask)
                        fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
                        state.last_score = float(clf.predict_proba(fv.reshape(1, -1))[0])
                    continue

                if profile.rounds_seen == 0 or clf is None:
                    continue

                fv = compute_final_features(profile, thompson, feature_mask=rc.feature_mask)
                fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)

                if mahalanobis.pca is not None:
                    maha = mahalanobis.score(fv)
                    fv[maha_offset:maha_offset + 3] = maha

                prob = float(clf.predict_proba(fv.reshape(1, -1))[0])
                decision, state = apply_decision(prob, round_number, rc, state)
                state.last_score = prob
                client.run_states[rc.run_number][uid] = state

            # Submit
            payload = client.build_submission(rc.run_number)
            success = client.submit_run(rc.run_number, payload)
            if success:
                client.log_decisions(rc.run_number, payload)
            else:
                logger.error("Failed to submit run %d", rc.run_number)

        # 4. Checkpoint
        client.current_round = round_number
        client.save_round_state()

        elapsed = time.monotonic() - t0
        n_alerts = sum(
            1 for u in client.master_user_list
            if client.run_states[0][u].alert_emitted
        )
        logger.info(
            "Round %d: %d active, %d alerts (run0), %.1fs",
            round_number, len(threads), n_alerts, elapsed,
        )
        round_number += 1

    logger.info("=== PIPELINE COMPLETE ===")
