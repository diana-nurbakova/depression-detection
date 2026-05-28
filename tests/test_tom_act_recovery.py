"""Tests for tom_act.recovery — three-stage JSON parsing/recovery."""

from mentalriskes.tom_act import recovery


def _view_json(score=1):
    items9 = ", ".join(f'{{"item": {i}, "score": {score}, "label_es": "x"}}' for i in range(1, 10))
    items7 = ", ".join(f'{{"item": {i}, "score": {score}, "label_es": "x"}}' for i in range(1, 8))
    items10 = ", ".join(f'{{"item": {i}, "score": {score}, "label_es": "x"}}' for i in range(1, 11))
    return (
        '{"phq9": {"items": [' + items9 + '], "rationale_es": "r"},'
        ' "gad7": {"items": [' + items7 + '], "rationale_es": "r"},'
        ' "compact10": {"items": [' + items10 + '], "rationale_es": "r"}}'
    )


def test_strict_categorical():
    r = recovery.recover('{"stance": "defusión", "rationale_es": "x"}', "tom_stance")
    assert r.success and r.stage == "strict"
    assert r.parsed["stance"] == "defusión"


def test_permissive_code_fence_and_trailing_comma():
    raw = '```json\n{"presencia": "alta", "rationale_es": "x",}\n```'
    r = recovery.recover(raw, "presencia")
    assert r.success and r.stage in ("permissive", "strict")
    assert r.parsed["presencia"] == "alta"


def test_fuzzy_accent_insensitive_stance():
    # Missing accent + English alias should both normalise to canonical Spanish.
    r = recovery.recover('{"stance": "defusion"}', "tom_stance")
    assert r.success
    assert r.parsed["stance"] == "defusión"


def test_english_alias_fallback():
    r = recovery.recover('{"stance": "reframing"}', "tom_stance")
    assert r.success and r.parsed["stance"] == "reformulación"


def test_fuzzy_extraction_from_broken_text():
    raw = 'El paciente... la postura es claramente reflejo aquí. (sin JSON válido'
    r = recovery.recover(raw, "tom_stance")
    assert r.success and r.stage == "fuzzy"
    assert r.parsed["stance"] == "reflejo"


def test_tier_strict():
    raw = '{"argmax": "cognitivo", "soft_scores": {"somatico":0.3,"cognitivo":0.5,"afectivo":0.2}}'
    r = recovery.recover(raw, "tom_tier_patient")
    assert r.success and r.parsed["argmax"] == "cognitivo"


def test_view_strict_full():
    r = recovery.recover(_view_json(), "view")
    assert r.success and r.stage == "strict"


def test_view_insufficient_items_fails():
    raw = '{"phq9": {"items": [{"item":1,"score":1}]}, "gad7": {"items": []}, "compact10": {"items": []}}'
    r = recovery.recover(raw, "view")
    assert not r.success


def test_assessor_direct_array_ok():
    raw = '{"PHQ-9": [0,1,2,3,0,1,2,1,0]}'
    r = recovery.recover(raw, "assessor:PHQ-9")
    assert r.success
    assert recovery.assessor_scores(r.parsed, "PHQ-9") == [0, 1, 2, 3, 0, 1, 2, 1, 0]


def test_assessor_truncated_cot_fails():
    # Only step_0/step_1, no scores — the real round-1 failure mode.
    raw = '{"step_0_category_scan": {"a": 1}, "step_1_detection": {"item_1": "x"}}'
    r = recovery.recover(raw, "assessor:PHQ-9")
    assert not r.success and r.error == "fuzzy_extraction_insufficient"


def test_assessor_step2_items_ok():
    items = ", ".join(f'"item_{i}": {{"score": 1}}' for i in range(1, 8))
    raw = '{"step_2_temporal": {' + items + '}}'
    r = recovery.recover(raw, "assessor:GAD-7")
    assert r.success
    assert recovery.assessor_scores(r.parsed, "GAD-7") == [1] * 7


def test_assessor_recovers_markdown_bare_array():
    # Real Llama failure mode: markdown CoT ending in a bare scores array.
    raw = ("## Step 0: Triflex scan ...\n- Item 10: 4\n\n"
           "## CompACT-10\n\n[3, 3, 4, 3, 4, 3, 4, 3, 4, 4]\n\nThis assessment ...")
    r = recovery.recover(raw, "assessor:CompACT-10")
    assert r.success and r.stage == "fuzzy"
    assert recovery.assessor_scores(r.parsed, "CompACT-10") == [3, 3, 4, 3, 4, 3, 4, 3, 4, 4]


def test_assessor_recovers_prose_labeled_array():
    raw = "Reasoning about anxiety ...\nGAD-7: [2, 3, 1, 1, 0, 1, 2]\nEnd."
    r = recovery.recover(raw, "assessor:GAD-7")
    assert r.success
    assert r.parsed["GAD-7"] == [2, 3, 1, 1, 0, 1, 2]


def test_empty_response():
    r = recovery.recover("", "tom_stance")
    assert not r.success and r.error == "empty_response"


def test_unparseable_categorical_fails():
    r = recovery.recover("totalmente irrelevante zzz", "presencia")
    assert not r.success
