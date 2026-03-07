"""Data loading and normalization for eRisk Task 2.

Handles two formats:
1. Training data: per-user JSON files with {submission: {}, comments: []} per thread
2. Server data: flat JSON with {submissionId, body, author, date, comments: [...]}

Both are normalized into the common Thread model.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from erisk_task2.models import Comment, Thread

logger = logging.getLogger(__name__)


def load_labels(labels_path: str | Path) -> dict[str, int]:
    """Load ground truth labels from space-separated file.

    Returns dict mapping subject_id -> 0 (control) or 1 (depressed).
    """
    labels = {}
    with open(labels_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels[parts[0]] = int(parts[1])
    return labels


def _normalize_date(date_str: str) -> str:
    """Normalize date strings to ISO 8601 format.

    Training format: "2024-05-08 02:55:38 UTC"
    Server format:   "2024-05-08T02:55:38.000+00:00"
    """
    if not date_str or date_str == "None":
        return ""
    # Already ISO
    if "T" in date_str:
        return date_str
    # Training format: "2024-05-08 02:55:38 UTC"
    try:
        dt = datetime.strptime(date_str.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%dT%H:%M:%S.000+00:00")
    except ValueError:
        return date_str


def _is_target(value) -> bool:
    """Check target field which can be bool, str 'True'/'False', or missing."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return False


def _is_deleted(text: str | None) -> bool:
    """Check if text is a deletion marker."""
    if not text:
        return True
    return text.startswith("[deleted") or text.startswith("[removed")


def _clean_text(text: str | None) -> str:
    """Clean text, returning empty string for deleted/missing content."""
    if not text or text == "None":
        return ""
    if _is_deleted(text):
        return ""
    return text.strip()


# ---------------------------------------------------------------------------
# Training data loading
# ---------------------------------------------------------------------------

def parse_training_thread(
    raw_thread: dict,
    target_subject: str,
    round_number: int,
) -> Thread:
    """Parse a single training data thread into normalized Thread.

    Training format:
        {
            "submission": {user_id, target, title, body, created_utc, submission_id, ...},
            "comments": [{user_id, target, body, created_utc, comment_id, parent_id, ...}, ...]
        }
    """
    sub = raw_thread.get("submission", {})

    submission_id = sub.get("submission_id", "")
    title = _clean_text(sub.get("title", ""))
    body = _clean_text(sub.get("body", ""))
    author = sub.get("user_id", "")
    created_utc = _normalize_date(sub.get("created_utc", ""))

    comments = []
    for c in raw_thread.get("comments", []):
        user_id = c.get("user_id", "")
        comment = Comment(
            comment_id=c.get("comment_id", ""),
            author=user_id,
            body=_clean_text(c.get("body", "")),
            parent_id=c.get("parent_id", ""),
            created_utc=_normalize_date(c.get("created_utc", "")),
            is_target=_is_target(c.get("target")) or user_id == target_subject,
        )
        comments.append(comment)

    return Thread(
        submission_id=submission_id,
        title=title,
        body=body,
        author=author,
        created_utc=created_utc,
        comments=comments,
        target_subject=target_subject,
        round_number=round_number,
    )


def load_training_user(json_path: str | Path) -> list[Thread]:
    """Load all threads for a single user from their training data JSON file.

    The filename encodes the target subject ID (e.g., subject_01ZzrIT.json).
    List index = round number.
    """
    json_path = Path(json_path)
    target_subject = json_path.stem  # e.g. "subject_01ZzrIT"

    with open(json_path, "r", encoding="utf-8") as f:
        raw_threads = json.load(f)

    if not isinstance(raw_threads, list):
        raw_threads = [raw_threads]

    threads = []
    for round_num, raw in enumerate(raw_threads):
        thread = parse_training_thread(raw, target_subject, round_num)
        threads.append(thread)

    return threads


def load_training_data(
    data_dir: str | Path,
    labels_path: str | Path | None = None,
) -> tuple[dict[str, list[Thread]], dict[str, int]]:
    """Load full training dataset.

    Returns:
        users: dict mapping subject_id -> list of Thread (one per round)
        labels: dict mapping subject_id -> 0/1 (empty if labels_path is None)
    """
    data_dir = Path(data_dir)
    labels = load_labels(labels_path) if labels_path else {}

    users: dict[str, list[Thread]] = {}
    json_files = sorted(data_dir.glob("*.json"))

    logger.info("Loading %d user files from %s", len(json_files), data_dir)

    for i, json_path in enumerate(json_files):
        subject_id = json_path.stem
        try:
            threads = load_training_user(json_path)
            users[subject_id] = threads
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse %s: %s", json_path.name, e)
            continue

        if (i + 1) % 100 == 0:
            logger.info("Loaded %d/%d users", i + 1, len(json_files))

    logger.info(
        "Loaded %d users (%d depressed, %d control, %d unlabeled)",
        len(users),
        sum(1 for sid in users if labels.get(sid) == 1),
        sum(1 for sid in users if labels.get(sid) == 0),
        sum(1 for sid in users if sid not in labels),
    )
    return users, labels


# ---------------------------------------------------------------------------
# Server data loading (for live competition)
# ---------------------------------------------------------------------------

def parse_server_thread(raw: dict) -> Thread:
    """Parse a single server response thread into normalized Thread.

    Server format:
        {
            "submissionId": "...",
            "body": "...",
            "author": "...",
            "title": "...",
            "date": "2024-05-08T02:55:38.000+00:00",
            "number": 5,
            "targetSubject": "subject_xyz",
            "comments": [
                {"commentId": "...", "author": "...", "body": "...",
                 "date": "...", "parent": "...", ...},
                ...
            ]
        }
    """
    target_subject = raw.get("targetSubject", "")

    comments = []
    for c in raw.get("comments", []):
        author = c.get("author", "")
        comment = Comment(
            comment_id=c.get("commentId", ""),
            author=author,
            body=_clean_text(c.get("body", "")),
            parent_id=c.get("parent", ""),
            created_utc=c.get("date", ""),
            is_target=(author == target_subject),
        )
        comments.append(comment)

    return Thread(
        submission_id=raw.get("submissionId", ""),
        title=_clean_text(raw.get("title", "")),
        body=_clean_text(raw.get("body", "")),
        author=raw.get("author", ""),
        created_utc=raw.get("date", ""),
        comments=comments,
        target_subject=target_subject,
        round_number=raw.get("number", 0),
    )


def parse_server_response(response_json: list[dict]) -> dict[str, Thread]:
    """Parse full server GET response into dict of target_subject -> Thread.

    Each round returns one thread per active target user.
    """
    threads: dict[str, Thread] = {}
    for raw in response_json:
        thread = parse_server_thread(raw)
        if thread.target_subject:
            threads[thread.target_subject] = thread
        else:
            logger.warning(
                "Thread %s has no targetSubject", thread.submission_id
            )
    return threads
