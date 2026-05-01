"""Server client for Task 2: adapts Task 1 server client for task2 endpoints."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter, Retry

from ..config import ServerConfig

logger = logging.getLogger(__name__)


@dataclass
class Task2Client:
    """HTTP client for MentalRiskES Task 2 competition server."""

    base_url: str = ""
    token: str = ""
    use_trial: bool = True
    retries: int = 5
    backoff: float = 0.1

    @classmethod
    def from_config(cls, cfg: ServerConfig) -> Task2Client:
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
        """Build endpoint URL.

        Trial:  GET  {base_url}/task2/getmessages_trial/{token}
                POST {base_url}/task2/submit_trial/{token}/{run_index}
        Test:   GET  {base_url}/task2/getmessages/{token}
                POST {base_url}/task2/submit/{token}/{run_index}
        """
        task = "task2"
        suffix = "_trial" if self.use_trial else ""
        if kind == "get":
            return f"{self.base_url}/{task}/getmessages{suffix}/{self.token}"
        else:
            # run_index appended by caller
            return f"{self.base_url}/{task}/submit{suffix}/{self.token}"

    def get_round(self) -> dict:
        """Fetch current round data from server.

        Returns:
            dict with session data, or empty dict if done/error.
            Expected format per session: {round, patient_input, option_1, option_2, option_3}
        """
        url = self._get_endpoint("get")
        session = self._make_session()
        try:
            response = session.get(url, timeout=30)
            logger.debug("GET task2 response (status %d): %s",
                         response.status_code, response.text[:500])
            if response.status_code != 200:
                logger.error("GET failed (status %d): %s", response.status_code, response.text)
                return {}
            data = response.json()
            logger.info("GET round data: %d sessions", len(data) if isinstance(data, dict) else 1)
            return data
        except Exception as e:
            logger.error("GET failed: %s", e)
            return {}

    def submit_selection(
        self,
        run_index: int,
        predictions: list[dict],
        emissions: dict,
    ) -> bool:
        """Submit selection predictions for a run.

        Args:
            run_index: 0-based run index (max 2 for 3 runs).
            predictions: list of {id, round, prediction} where prediction is 1/2/3.
            emissions: CodeCarbon emissions dict.
        """
        base_url = self._get_endpoint("post")
        url = f"{base_url}/{run_index}"
        session = self._make_session()

        # Server expects prediction as "option_1", "option_2", "option_3"
        formatted_preds = [
            {**p, "prediction": f"option_{p['prediction']}"} for p in predictions
        ]

        # Server requires all emissions fields
        if not emissions or "duration" not in emissions:
            emissions = {
                "duration": 0.0, "emissions": 0.0, "cpu_energy": 0.0,
                "gpu_energy": 0.0, "ram_energy": 0.0, "energy_consumed": 0.0,
                "cpu_count": 0, "gpu_count": 0, "cpu_model": "",
                "gpu_model": "", "ram_total_size": 0.0, "country_iso_code": "FRA",
            }

        payload = [{"predictions": formatted_preds, "emissions": emissions}]

        try:
            response = session.post(url, json=payload, timeout=30)
            logger.debug("POST task2 run %d response (status %d): %s",
                         run_index, response.status_code, response.text[:500])
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
        """Submit predictions for all runs."""
        for run_idx, preds in enumerate(predictions_per_run):
            self.submit_selection(run_idx, preds, emissions)
            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                path = save_dir / f"round{round_number}_run{run_idx}.json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump([{"predictions": preds, "emissions": emissions}],
                              f, ensure_ascii=False, indent=2)
