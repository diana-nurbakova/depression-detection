"""Server interaction client for eRisk Task 2 (Spec Section 17).

GET/POST loop with retry, checkpointing, and full round orchestration.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

from erisk_task2.config import Task2Config
from erisk_task2.data.loader import parse_server_response
from erisk_task2.models import DEFAULT_RUNS, RunConfig, RunUserState, UserProfile
from erisk_task2.server.checkpoint import load_latest_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)


class ERiskClient:
    """HTTP client for eRisk server with retry and state management."""

    def __init__(self, config: Task2Config):
        self.config = config
        self.base_url = config.server.base_url.rstrip("/")
        self.token = config.server.team_token
        self.max_retries = config.server.max_retries
        self.initial_delay = config.server.initial_delay
        self.backoff_factor = config.server.backoff_factor
        self.timeout = config.server.timeout
        self._session = requests.Session()

        # State
        self.master_user_list: list[str] = []
        self.profiles: dict[str, UserProfile] = {}
        self.run_states: dict[int, dict[str, RunUserState]] = {}
        self.current_round: int = 0

        # Log directories
        self.log_dir = Path(config.logging.output_dir)
        self.checkpoint_dir = self.log_dir / "checkpoint"
        self.server_log_dir = self.log_dir / "server_responses"
        self.decision_log_dir = self.log_dir / "decisions"
        self.submit_log_dir = self.log_dir / "submit_responses"
        for d in [self.log_dir, self.checkpoint_dir, self.server_log_dir,
                  self.decision_log_dir, self.submit_log_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_discussions(self) -> Optional[list[dict]]:
        """GET discussions for the current round.

        Returns list of thread dicts, or None on failure, or empty list if done.
        """
        url = f"{self.base_url}/getdiscussions/{self.token}"
        return self._get_with_retry(url)

    def submit_run(self, run_number: int, payload: list[dict]) -> tuple[bool, str]:
        """POST submission for a single run.

        Returns (success, response_body).
        """
        url = f"{self.base_url}/submit/{self.token}/{run_number}"
        return self._post_with_retry(url, payload)

    def _get_with_retry(self, url: str) -> Optional[list[dict]]:
        delay = self.initial_delay
        for attempt in range(self.max_retries):
            try:
                resp = self._session.get(url, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    return data if isinstance(data, list) else []
                logger.warning("GET %s returned %d (attempt %d)", url, resp.status_code, attempt + 1)
            except requests.RequestException as e:
                logger.warning("GET %s failed (attempt %d): %s", url, attempt + 1, e)
            if attempt < self.max_retries - 1:
                time.sleep(delay)
                delay *= self.backoff_factor
        logger.error("All GET attempts failed for %s", url)
        return None

    def _post_with_retry(self, url: str, payload: list[dict]) -> tuple[bool, str]:
        delay = self.initial_delay
        for attempt in range(self.max_retries):
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                body = resp.text
                if resp.status_code == 200:
                    logger.info("POST %s → 200: %s", url, body[:200])
                    return True, body
                logger.warning("POST %s returned %d (attempt %d): %s", url, resp.status_code, attempt + 1, body[:500])
            except requests.RequestException as e:
                logger.warning("POST %s failed (attempt %d): %s", url, attempt + 1, e)
                body = str(e)
            if attempt < self.max_retries - 1:
                time.sleep(delay)
                delay *= self.backoff_factor
        logger.error("All POST attempts failed for %s", url)
        return False, body

    def initialize(self, run_configs: list[RunConfig] | None = None):
        """Initialize or restore state.

        Checks for existing checkpoint first. If none, starts fresh.
        """
        run_configs = run_configs or DEFAULT_RUNS

        # Try loading checkpoint
        checkpoint = load_latest_checkpoint(self.checkpoint_dir)
        if checkpoint is not None:
            self.current_round, self.master_user_list, self.profiles, self.run_states = checkpoint
            self.current_round += 1  # resume from next round
            logger.info(
                "Resumed from checkpoint: round %d, %d users",
                self.current_round, len(self.master_user_list),
            )
            return

        # Fresh start — initialize run states
        for rc in run_configs:
            self.run_states[rc.run_number] = {}

    def capture_master_list(self, threads: dict[str, object]):
        """Capture master user list from round 0 response."""
        self.master_user_list = sorted(threads.keys())
        # Initialize profiles and run states for all users
        for uid in self.master_user_list:
            if uid not in self.profiles:
                self.profiles[uid] = UserProfile(subject_id=uid)
            for run_num in self.run_states:
                if uid not in self.run_states[run_num]:
                    self.run_states[run_num][uid] = RunUserState()
        logger.info("Captured master user list: %d users", len(self.master_user_list))

    def build_submission(self, run_number: int) -> list[dict]:
        """Build submission payload for a run.

        Every user in master_user_list must appear.
        """
        submission = []
        for uid in self.master_user_list:
            state = self.run_states[run_number][uid]
            submission.append({
                "nick": uid,
                "decision": 1 if state.alert_emitted else 0,
                "score": round(state.last_score, 6),
            })
        return submission

    def save_round_state(self):
        """Save checkpoint and log server response."""
        save_checkpoint(
            self.checkpoint_dir,
            self.current_round,
            self.master_user_list,
            self.profiles,
            self.run_states,
        )

    def log_server_response(self, response_data: list[dict]):
        """Save raw server response."""
        path = self.server_log_dir / f"round_{self.current_round:04d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(response_data, f, ensure_ascii=False)

    def log_decisions(self, run_number: int, payload: list[dict]):
        """Save submitted decisions."""
        path = self.decision_log_dir / f"run_{run_number}_round_{self.current_round:04d}.json"
        with open(path, "w") as f:
            json.dump(payload, f)

    def log_submit_response(self, run_number: int, success: bool, response_body: str):
        """Save server response to our POST submission."""
        path = self.submit_log_dir / f"run_{run_number}_round_{self.current_round:04d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"success": success, "response": response_body}, f)

    def log_round_interaction(self, round_number: int, n_active_users: int,
                              submissions: dict[int, list[dict]]):
        """Append a full round summary to the interaction log (JSONL)."""
        path = self.log_dir / "interactions.jsonl"
        record = {
            "round": round_number,
            "active_users": n_active_users,
            "total_users": len(self.master_user_list),
            "runs": {},
        }
        for run_num, payload in submissions.items():
            alerts = [e for e in payload if e["decision"] == 1]
            record["runs"][run_num] = {
                "total_alerts": len(alerts),
                "new_alerts": [
                    e["nick"] for e in alerts
                    if self.run_states[run_num][e["nick"]].alert_round == round_number
                ],
                "scores": {e["nick"]: e["score"] for e in payload},
            }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
