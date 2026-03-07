"""Main pipeline orchestration for training, evaluation, and live runs."""

from __future__ import annotations

import logging

from erisk_task2.config import Task2Config

logger = logging.getLogger(__name__)


def train_pipeline(config: Task2Config):
    """Offline training pipeline.

    1. Load training data
    2. Extract features for all users across all rounds
    3. Train classifiers with cross-validation
    4. Fit Mahalanobis reference distributions
    5. Fit BERTopic
    6. Run ablation studies
    7. Save trained models
    """
    from erisk_task2.data.loader import load_training_data

    logger.info("=== TRAINING PIPELINE ===")

    # Load data
    users, labels = load_training_data(config.training_data_dir, config.labels_path)
    logger.info("Loaded %d users, %d labels", len(users), len(labels))

    # TODO: Implement full training loop
    # - Initialize feature extractors (sentence transformers, BERTopic, etc.)
    # - Process each user's threads sequentially (simulating rounds)
    # - Accumulate features in UserProfile objects
    # - Extract feature vectors at final round for each user
    # - Train classifiers
    # - Evaluate with cross-validation
    # - Save models

    logger.info("Training pipeline not yet fully implemented")


def evaluate_pipeline(config: Task2Config):
    """Offline evaluation: simulate round-by-round processing on training data.

    For each user, process threads one at a time (simulating server rounds),
    apply decision policy, and compute ERDE/F1/F_latency metrics.
    """
    logger.info("=== EVALUATION PIPELINE ===")

    # TODO: Implement
    # - Load trained models
    # - Simulate round-by-round processing
    # - Apply decision policies for all 5 runs
    # - Compute metrics (ERDE5, ERDE50, F1, F_latency)

    logger.info("Evaluation pipeline not yet fully implemented")


def run_pipeline(config: Task2Config):
    """Live competition pipeline against eRisk server.

    Main round loop as specified in Section 17.2.
    """
    from erisk_task2.data.loader import parse_server_response
    from erisk_task2.models import DEFAULT_RUNS
    from erisk_task2.server.client import ERiskClient

    logger.info("=== LIVE PIPELINE ===")

    client = ERiskClient(config)
    client.initialize()

    round_number = client.current_round

    while True:
        logger.info("--- Round %d ---", round_number)

        # 1. GET discussions
        response_data = client.get_discussions()
        if response_data is None:
            logger.error("Failed to get discussions, pausing...")
            continue

        if len(response_data) == 0:
            logger.info("Empty response — all rounds complete")
            break

        # Save raw response
        client.log_server_response(response_data)

        # Parse threads
        threads = parse_server_response(response_data)

        # Capture master list on round 0
        if round_number == 0:
            client.capture_master_list(threads)

        # 2. Feature extraction (shared across all runs)
        for uid, thread in threads.items():
            profile = client.profiles.get(uid)
            if profile is None:
                continue
            profile.rounds_seen += 1
            profile.last_active_round = round_number

            # TODO: Full feature extraction
            # - Layer 1: embeddings, symptoms, lexical
            # - Layer 2: sentiment, concern, position
            # - Layer 3: emotion, topic
            # - ToM assessment
            # - Distance computation
            # - Bandit updates

        # 3. Classification + Decision (per run)
        for rc in DEFAULT_RUNS:
            for uid in client.master_user_list:
                state = client.run_states[rc.run_number][uid]

                if state.alert_emitted:
                    # Keep decision=1, update score
                    continue

                # TODO: Run classifier, apply decision policy
                # For now, default to no alert
                state.last_score = 0.0

            # Build and submit
            payload = client.build_submission(rc.run_number)
            success = client.submit_run(rc.run_number, payload)
            if success:
                client.log_decisions(rc.run_number, payload)
                logger.info("Run %d submitted (%d users)", rc.run_number, len(payload))
            else:
                logger.error("Failed to submit run %d", rc.run_number)

        # 4. Checkpoint
        client.current_round = round_number
        client.save_round_state()

        round_number += 1

    logger.info("=== PIPELINE COMPLETE ===")
