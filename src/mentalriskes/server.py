"""Server client for MentalRiskES competition server.

Handles GET (fetch messages) and POST (submit predictions) with retry logic.
Adapted from the organizer's ClientServer.ipynb.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter, Retry

from .config import ServerConfig

logger = logging.getLogger(__name__)


@dataclass
class MentalRiskESClient:
    """HTTP client for MentalRiskES competition server."""

    base_url: str = ""
    token: str = ""
    use_trial: bool = True
    retries: int = 5
    backoff: float = 0.1

    @classmethod
    def from_config(cls, cfg: ServerConfig) -> MentalRiskESClient:
        return cls(
            base_url=cfg.base_url,
            token=cfg.token,
            use_trial=cfg.use_trial,
            retries=cfg.retries,
            backoff=cfg.backoff,
        )

    def _make_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=self.retries,
            backoff_factor=self.backoff,
            status_forcelist=[500, 502, 503, 504],
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.mount("http://", HTTPAdapter(max_retries=retry))
        return session

    def _get_endpoint(self, kind: str) -> str:
        """Return the correct URL based on trial/test mode."""
        task = "task1"
        if kind == "get":
            suffix = "getmessages_trial" if self.use_trial else "getmessages"
            return f"{self.base_url}/{task}/{suffix}/{self.token}"
        else:
            # POST endpoint — run index appended by caller
            suffix = "submit_trial" if self.use_trial else "submit"
            return f"{self.base_url}/{task}/{suffix}/{self.token}"

    def get_messages(self) -> dict:
        """
        Fetch the current round's messages from the server.

        Returns:
            dict keyed by session_id, each containing 'round', 'patient_input',
            and optionally 'therapist_response'. Empty dict when all rounds done.
        """
        url = self._get_endpoint("get")
        session = self._make_session()

        try:
            response = session.get(url, timeout=30)
            if response.status_code != 200:
                logger.error("GET failed (status %d): %s", response.status_code, response.text)
                return {}
            data = response.json()
            logger.info("GET round messages: %d sessions", len(data))
            return data
        except Exception as e:
            logger.error("GET failed: %s", e)
            return {}

    def submit_predictions(
        self,
        run_index: int,
        predictions: list[dict],
        emissions: dict,
    ) -> bool:
        """
        Submit predictions for a single run.

        Args:
            run_index: 0-based run index.
            predictions: list of prediction dicts, each with 'id', 'round', 'prediction'.
            emissions: CodeCarbon emissions dict.

        Returns:
            True if submission succeeded.
        """
        base_url = self._get_endpoint("post")
        url = f"{base_url}/{run_index}"
        session = self._make_session()

        payload = [{"predictions": predictions, "emissions": emissions}]

        try:
            response = session.post(url, json=payload, timeout=30)
            if response.status_code != 200:
                logger.error("POST run %d failed (status %d): %s",
                             run_index, response.status_code, response.text)
                return False
            logger.info("POST run %d: %s", run_index, response.text)
            return True
        except Exception as e:
            logger.error("POST run %d failed: %s", run_index, e)
            return False

    def submit_all_runs(
        self,
        predictions_per_run: list[list[dict]],
        emissions: dict,
        save_dir: Path | None = None,
        round_number: int = 0,
    ) -> None:
        """Submit predictions for all runs, optionally saving locally."""
        for run_idx, preds in enumerate(predictions_per_run):
            self.submit_predictions(run_idx, preds, emissions)

            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                local_path = save_dir / f"round{round_number}_run{run_idx}.json"
                payload = [{"predictions": preds, "emissions": emissions}]
                with open(local_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                logger.debug("Saved predictions to %s", local_path)
