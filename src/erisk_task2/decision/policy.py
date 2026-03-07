"""Decision policy for alert timing (Spec Section 10).

Adaptive threshold with ERDE-aware decay and consecutive confirmation.
"""

from __future__ import annotations

import math

import numpy as np

from erisk_task2.models import RunConfig, RunUserState


def lc_o(k: int, o: int) -> float:
    """ERDE latency cost function: lc_o(k) = 1 - 1/(1 + e^(k-o))."""
    return 1.0 - 1.0 / (1.0 + math.exp(k - o))


def compute_threshold(
    round_k: int,
    theta_init: float,
    theta_floor: float,
    erde_o: int,
) -> float:
    """Compute adaptive threshold at round k.

    theta(k) = theta_init - (theta_init - theta_floor) * lc_o(k)
    """
    decay = lc_o(round_k, erde_o)
    return theta_init - (theta_init - theta_floor) * decay


def apply_decision(
    probability: float,
    round_k: int,
    run_config: RunConfig,
    state: RunUserState,
) -> tuple[int, RunUserState]:
    """Apply decision policy to determine if alert should fire.

    Args:
        probability: P(depressed) from classifier
        round_k: current round number
        run_config: run-specific parameters
        state: current per-user per-run state

    Returns:
        (decision, updated_state) where decision is 0 or 1
    """
    # Already alerted — keep decision=1
    if state.alert_emitted:
        state.last_probability = probability
        return 1, state

    # Compute dynamic threshold
    threshold = compute_threshold(
        round_k, run_config.theta_init, run_config.theta_floor, run_config.erde_o
    )

    state.last_probability = probability

    # Check if above threshold
    if probability >= threshold:
        state.consecutive_positives += 1
    else:
        state.consecutive_positives = 0

    # Fire alert if enough consecutive positives
    if state.consecutive_positives >= run_config.t_con:
        state.alert_emitted = True
        state.alert_round = round_k
        return 1, state

    return 0, state


def compute_erde(
    decisions: dict[str, int],
    alert_rounds: dict[str, int],
    labels: dict[str, int],
    o: int,
) -> float:
    """Compute ERDE_o metric for evaluation.

    Args:
        decisions: subject_id -> final decision (0 or 1)
        alert_rounds: subject_id -> round when alerted (only for decision=1)
        labels: subject_id -> ground truth (0 or 1)
        o: ERDE parameter (5 or 50)
    """
    total = 0.0
    n = len(labels)

    for uid, label in labels.items():
        decision = decisions.get(uid, 0)

        if label == 1:  # truly depressed
            if decision == 1:  # true positive
                k = alert_rounds.get(uid, 0)
                total += lc_o(k, o)
            else:  # false negative
                total += 1.0
        else:  # truly control
            if decision == 1:  # false positive
                total += 1.0
            # true negative: cost = 0

    return total / max(n, 1)


def compute_f_latency(
    alert_rounds: dict[str, int],
    labels: dict[str, int],
    decisions: dict[str, int],
    o: int = 50,
) -> float:
    """Compute F_latency metric.

    F_latency = F1 × speed, where speed = 1 - median(lc_o(k)) over true positives.
    """
    # True positives
    tp_rounds = []
    tp = fp = fn = 0

    for uid, label in labels.items():
        decision = decisions.get(uid, 0)
        if label == 1 and decision == 1:
            tp += 1
            tp_rounds.append(alert_rounds.get(uid, 0))
        elif label == 0 and decision == 1:
            fp += 1
        elif label == 1 and decision == 0:
            fn += 1

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    latencies = [lc_o(k, o) for k in tp_rounds]
    speed = 1.0 - float(np.median(latencies))

    return f1 * speed
