"""Theory of Mind module (Spec Section 6).

Implements all 3 options:
- Option A: Embedding-based dual representation
- Option B: Response category classification
- Option C: LLM-based explicit mentalizing
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from erisk_task2.formatting.thread_formatter import format_thread
from erisk_task2.models import Thread, UserProfile
from erisk_task2.tom.llm_client import OllamaClient
from erisk_task2.tom.prompts import (
    PROMPT1_SYSTEM,
    PROMPT1_USER,
    PROMPT2A_SYSTEM,
    PROMPT2A_USER,
    PROMPT2B_SYSTEM,
    PROMPT2B_USER,
    PROMPT4_SYSTEM,
    PROMPT4_USER,
    get_symptom_variant,
)

logger = logging.getLogger(__name__)

NUM_SYMPTOMS = 21
RESPONSE_CATEGORIES = [
    "CONCERN", "ADVICE", "EMOTIONAL_SUPPORT", "NORMALIZATION",
    "SHARED_EXPERIENCE", "PRACTICAL_SUPPORT", "CASUAL",
]


class ToMModule:
    """Theory of Mind module with configurable implementation options."""

    def __init__(
        self,
        method: str = "option_c",
        chained: bool = False,
        symptom_variant: str = "C",
        encoder=None,  # EmbeddingEncoder for Option A
        llm_client: Optional[OllamaClient] = None,  # for Option C
    ):
        self.method = method
        self.chained = chained
        self.symptom_variant = symptom_variant
        self.encoder = encoder
        self.llm_client = llm_client
        self._symptom_defs = get_symptom_variant(symptom_variant)

        # Pre-format system prompts (byte-identical for caching)
        self._sys_prompt1 = PROMPT1_SYSTEM.format(symptom_definitions=self._symptom_defs)
        self._sys_prompt2a = PROMPT2A_SYSTEM.format(symptom_definitions=self._symptom_defs)
        self._sys_prompt2b = PROMPT2B_SYSTEM.format(symptom_definitions=self._symptom_defs)

    def assess(
        self,
        thread: Thread,
        profile: UserProfile,
    ) -> dict:
        """Run ToM assessment for a thread.

        Returns dict with self_view, observer_view, and gap_metrics.
        """
        if self.method == "option_a":
            return self._assess_embedding(thread, profile)
        elif self.method == "option_b":
            return self._assess_categories(thread, profile)
        else:
            return self._assess_llm(thread, profile)

    # ---- Option A: Embedding-based ----

    def _assess_embedding(self, thread: Thread, profile: UserProfile) -> dict:
        if self.encoder is None:
            return _empty_result()

        result: dict = {"method": "option_a"}

        # Self-view: target user texts embedding
        target_texts = thread.target_texts
        if target_texts:
            self_emb = self.encoder.encode(target_texts).mean(axis=0)
            result["self_view"] = {"embedding_norm": float(np.linalg.norm(self_emb))}
        else:
            self_emb = None
            result["self_view"] = None

        # Observer-view: all thread comments embedding
        other_texts = [c.body for c in thread.other_comments if c.body]
        if other_texts:
            obs_emb = self.encoder.encode(other_texts).mean(axis=0)
            result["observer_view"] = {"embedding_norm": float(np.linalg.norm(obs_emb))}
        else:
            obs_emb = None
            result["observer_view"] = {"embedding_norm": 0.0}

        # Cosine distance
        if self_emb is not None and obs_emb is not None:
            norm_s = np.linalg.norm(self_emb)
            norm_o = np.linalg.norm(obs_emb)
            if norm_s > 1e-8 and norm_o > 1e-8:
                cos_sim = float(np.dot(self_emb, obs_emb) / (norm_s * norm_o))
                result["gap_metrics"] = {"cosine_distance": 1.0 - cos_sim}
            else:
                result["gap_metrics"] = {"cosine_distance": 0.0}
        else:
            result["gap_metrics"] = {"cosine_distance": 0.0}

        return result

    # ---- Option B: Response categories ----

    def _assess_categories(self, thread: Thread, profile: UserProfile) -> dict:
        result: dict = {"method": "option_b"}

        if self.llm_client is None:
            result["observer_view"] = {"category_distribution": np.zeros(7).tolist()}
            return result

        direct_replies = thread.direct_replies_to_target
        target_texts = thread.target_texts
        target_snippet = target_texts[0][:200] if target_texts else "[no text]"

        categories = np.zeros(len(RESPONSE_CATEGORIES))
        for reply in direct_replies[:10]:  # limit to 10 replies
            user_prompt = PROMPT4_USER.format(
                target_text=target_snippet,
                reply_text=reply.body[:200],
            )
            response, _ = self.llm_client.generate(PROMPT4_SYSTEM, user_prompt)
            label = response.strip().upper()
            for i, cat in enumerate(RESPONSE_CATEGORIES):
                if cat in label:
                    categories[i] += 1
                    break

        total = categories.sum()
        if total > 0:
            categories = categories / total

        result["observer_view"] = {"category_distribution": categories.tolist()}
        return result

    # ---- Option C: LLM-based ----

    def _assess_llm(self, thread: Thread, profile: UserProfile) -> dict:
        result: dict = {"method": "option_c"}

        if self.llm_client is None:
            return _empty_result()

        # Self-view (only if target has text)
        self_view = None
        if thread.has_target_text:
            target_texts = "\n\n".join(thread.target_texts)
            user_prompt = PROMPT1_USER.format(target_user_texts=target_texts)
            self_view, elapsed = self.llm_client.generate_json(
                self._sys_prompt1, user_prompt,
            )
            if self_view:
                result["self_view"] = self_view
                result["self_view_ms"] = int(elapsed * 1000)
            else:
                result["self_view"] = None
        else:
            result["self_view"] = None

        # Observer-view
        formatted, _ = format_thread(thread)

        if self.chained and self_view is not None:
            import json
            user_prompt = PROMPT2B_USER.format(
                self_view_json=json.dumps(self_view),
                formatted_thread=formatted,
            )
            obs_view, elapsed = self.llm_client.generate_json(
                self._sys_prompt2b, user_prompt,
            )
        else:
            user_prompt = PROMPT2A_USER.format(formatted_thread=formatted)
            obs_view, elapsed = self.llm_client.generate_json(
                self._sys_prompt2a, user_prompt,
            )

        if obs_view:
            result["observer_view"] = obs_view
            result["observer_view_ms"] = int(elapsed * 1000)
        else:
            result["observer_view"] = None

        # Compute gap metrics
        result["gap_metrics"] = _compute_gap(result.get("self_view"), result.get("observer_view"))

        return result


def extract_tom_features(tom_result: dict, n_symptoms: int = NUM_SYMPTOMS) -> np.ndarray:
    """Extract numerical feature vector from ToM assessment result.

    Returns (~47,) array:
        - 21d self-view symptom vector
        - 21d observer-view symptom vector
        - 1d self depression_probability
        - 1d observer depression_probability
        - 1d insight_gap
        - 1d observer_concern_level
        - 1d community_response_type_encoded
    """
    features = np.zeros(47)

    # Self-view symptoms (21d)
    self_view = tom_result.get("self_view")
    if self_view and isinstance(self_view, dict):
        symptoms = self_view.get("active_symptoms", {})
        for name, info in symptoms.items():
            idx = _symptom_name_to_index(name)
            if idx >= 0 and isinstance(info, dict):
                features[idx] = info.get("score", 0) / 3.0
        features[42] = self_view.get("depression_probability", 0.0)

    # Observer-view symptoms (21d)
    obs_view = tom_result.get("observer_view")
    if obs_view and isinstance(obs_view, dict):
        symptoms = obs_view.get("perceived_symptoms", {})
        for name, info in symptoms.items():
            idx = _symptom_name_to_index(name)
            if idx >= 0 and isinstance(info, dict):
                features[21 + idx] = info.get("score", 0) / 3.0
        features[43] = obs_view.get("depression_probability", 0.0)
        features[45] = obs_view.get("observer_concern_level", 0) / 3.0

        # Encode community response type
        resp_type = obs_view.get("community_response_type", "casual")
        type_map = {"concern": 1.0, "support": 0.8, "advice": 0.6, "mixed": 0.5, "normalization": 0.3, "casual": 0.0}
        features[46] = type_map.get(resp_type, 0.0)

    # Gap metrics
    gap = tom_result.get("gap_metrics", {})
    features[44] = gap.get("insight_gap", 0.0)

    return features


def _compute_gap(self_view: Optional[dict], observer_view: Optional[dict]) -> dict:
    """Compute gap metrics between self and observer views."""
    if not self_view or not observer_view:
        return {"insight_gap": 0.0}

    self_probs = self_view.get("active_symptoms", {})
    obs_probs = observer_view.get("perceived_symptoms", {})

    self_scores = np.zeros(NUM_SYMPTOMS)
    obs_scores = np.zeros(NUM_SYMPTOMS)

    for name, info in self_probs.items():
        idx = _symptom_name_to_index(name)
        if idx >= 0 and isinstance(info, dict):
            self_scores[idx] = info.get("score", 0) / 3.0

    for name, info in obs_probs.items():
        idx = _symptom_name_to_index(name)
        if idx >= 0 and isinstance(info, dict):
            obs_scores[idx] = info.get("score", 0) / 3.0

    gap = float(np.mean(np.abs(self_scores - obs_scores)))
    return {"insight_gap": gap}


# Symptom name -> 0-indexed lookup
_SYMPTOM_NAME_MAP = {
    "sadness": 0, "pessimism": 1, "past failure": 2, "past_failure": 2,
    "loss of pleasure": 3, "loss_of_pleasure": 3,
    "guilty feelings": 4, "guilty_feelings": 4,
    "punishment feelings": 5, "punishment_feelings": 5,
    "self-dislike": 6, "self_dislike": 6,
    "self-criticalness": 7, "self_criticalness": 7,
    "suicidal thoughts": 8, "suicidal_thoughts": 8,
    "crying": 9, "agitation": 10,
    "loss of interest": 11, "loss_of_interest": 11,
    "indecisiveness": 12, "worthlessness": 13,
    "loss of energy": 14, "loss_of_energy": 14,
    "sleep changes": 15, "sleep_changes": 15,
    "irritability": 16,
    "appetite changes": 17, "appetite_changes": 17,
    "concentration difficulty": 18, "concentration_difficulty": 18,
    "tiredness/fatigue": 19, "tiredness_fatigue": 19, "tiredness": 19, "fatigue": 19,
    "loss of interest in sex": 20, "loss_of_interest_in_sex": 20,
}


def _symptom_name_to_index(name: str) -> int:
    return _SYMPTOM_NAME_MAP.get(name.lower().strip(), -1)


def _empty_result() -> dict:
    return {
        "method": "none",
        "self_view": None,
        "observer_view": None,
        "gap_metrics": {"insight_gap": 0.0},
    }
