"""Tests for tom_act.dispatcher — atomic logging + resume idempotency."""

import json
from pathlib import Path

from mentalriskes.tom_act.dispatcher import Dispatcher, input_signature


class FakeClient:
    """Minimal LLM client stub matching the dispatcher's call contract."""

    def __init__(self, content):
        self.content = content
        self.calls = 0

    def chat_completion(self, messages, temperature=None, max_tokens=None):
        self.calls += 1
        return {"choices": [{"message": {"content": self.content}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3}}

    @staticmethod
    def _get_content(resp):
        return resp["choices"][0]["message"]["content"]


def _read_lines(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _process(disp, client, schema="tom_stance"):
    return disp.process(
        signal_type="tom_stance", session_id="S01", round_n=3,
        system_prompt="sys", user_prompt="usr", client=client,
        model_id="m", provider="p", schema=schema, candidate=2,
    )


def test_signature_format():
    assert input_signature("S01", 3, "tom_stance", 2) == "S01:r03:tom_stance:opt2:v1"
    assert input_signature("S01", 3, "self_a") == "S01:r03:self_a:v1"


def test_success_logs_one_line_with_provenance(tmp_path):
    disp = Dispatcher(tmp_path, max_attempts=3)
    client = FakeClient('{"stance": "reflejo", "rationale_es": "x"}')
    parsed = _process(disp, client)
    assert parsed["stance"] == "reflejo"

    lines = _read_lines(tmp_path / "logs" / "tom_stance.jsonl")
    assert len(lines) == 1
    rec = lines[0]
    assert rec["parse_success"] is True
    assert rec["prompt_system"] == "sys" and rec["prompt_user"] == "usr"
    assert rec["prompt_system_hash"].startswith("sha256:")
    assert rec["candidate"] == 2
    assert rec["input_signature"] == "S01:r03:tom_stance:opt2:v1"
    assert "code_version" in rec


def test_resume_skips_completed_same_instance(tmp_path):
    disp = Dispatcher(tmp_path, max_attempts=3)
    client = FakeClient('{"stance": "reflejo"}')
    _process(disp, client)
    _process(disp, client)  # should skip — no second LLM call
    assert client.calls == 1
    assert len(_read_lines(tmp_path / "logs" / "tom_stance.jsonl")) == 1


def test_resume_skips_across_new_instance(tmp_path):
    Dispatcher(tmp_path).process(
        signal_type="tom_stance", session_id="S01", round_n=3,
        system_prompt="s", user_prompt="u",
        client=FakeClient('{"stance": "defusión"}'),
        model_id="m", provider="p", schema="tom_stance", candidate=2)
    # Fresh dispatcher loads index from disk and skips.
    client2 = FakeClient('{"stance": "reflejo"}')
    parsed = _process(Dispatcher(tmp_path), client2)
    assert client2.calls == 0
    assert parsed["stance"] == "defusión"


def test_failed_parse_retries_until_exhausted(tmp_path):
    disp = Dispatcher(tmp_path, max_attempts=3)
    bad = FakeClient("no json here at all")
    assert _process(disp, bad) is None         # attempt 1
    assert _process(disp, bad) is None          # attempt 2
    assert _process(disp, bad) is None          # attempt 3
    assert _process(disp, bad) is None          # exhausted -> skip, no call
    assert bad.calls == 3
    lines = _read_lines(tmp_path / "logs" / "tom_stance.jsonl")
    assert len(lines) == 3
    assert all(l["parse_success"] is False for l in lines)
