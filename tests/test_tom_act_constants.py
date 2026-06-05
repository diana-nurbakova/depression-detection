"""Tests for tom_act.constants — CompACT-10 reverse-scoring and subscales."""

from mentalriskes.tom_act import constants as C


def test_reverse_score():
    assert C.reverse_score(0) == 6
    assert C.reverse_score(6) == 0
    assert C.reverse_score(2) == 4


def test_subscale_mapping_covers_all_items():
    all_items = sorted(i for items in C.COMPACT10_SUBSCALES.values() for i in items)
    assert all_items == list(range(1, 11))
    # OE + BA are exactly the reverse-scored items.
    rev = set(C.COMPACT10_SUBSCALES["OE"]) | set(C.COMPACT10_SUBSCALES["BA"])
    assert rev == C.COMPACT10_REVERSE_ITEMS


def test_compact10_subscale_scores_reverse_applied():
    # All raw zeros: OE/BA reverse to 6 each; VA stays 0.
    sub = C.compact10_subscale_scores([0] * 10)
    assert sub["OE"] == 6 * 3   # 3 reverse items, each 6-0
    assert sub["BA"] == 6 * 3
    assert sub["VA"] == 0

    # A known gold array (spec S07): [0,2,5,1,4,0,2,4,0,4]
    sub = C.compact10_subscale_scores([0, 2, 5, 1, 4, 0, 2, 4, 0, 4])
    # OE items 3,5,8 raw=5,4,4 -> reversed 1,2,2 = 5
    assert sub["OE"] == (6 - 5) + (6 - 4) + (6 - 4)
    # BA items 1,6,9 raw=0,0,0 -> reversed 6,6,6 = 18
    assert sub["BA"] == 18
    # VA items 2,4,7,10 raw=2,1,2,4 = 9 (direct)
    assert sub["VA"] == 2 + 1 + 2 + 4


def test_instrument_item_counts():
    assert len(C.PHQ9_ITEMS) == 9
    assert len(C.GAD7_ITEMS) == 7
    assert len(C.COMPACT10_ITEMS) == 10
    assert len(C.SESSIONS) == 10


def test_stance_label_roundtrip():
    for es in C.TOM_STANCE_LABELS_ES:
        en = C.TOM_STANCE_EN_FROM_ES[es]
        assert C.TOM_STANCE_ES_FROM_EN[en] == es


def test_canonical_phase_handles_accent_variants():
    # Accented and unaccented variants collapse to the accented canonical.
    assert C.canonical_phase("defusión") == "defusión"
    assert C.canonical_phase("defusion") == "defusión"
    assert C.canonical_phase("EXPLORACIÓN") == "exploración"
    assert C.canonical_phase("exploracion") == "exploración"
    assert C.canonical_phase("Activacion") == "activación"
    assert C.canonical_phase("integración") == "integración"
    assert C.canonical_phase("crisis") == "crisis"
    assert C.canonical_phase("cierre") == "cierre"


def test_canonical_phase_pass_through_and_unknown():
    assert C.canonical_phase(None) is None
    assert C.canonical_phase("") == ""
    # Unknown phase falls through to accent-stripped lowercase form (no lookup hit).
    assert C.canonical_phase("Misceláneo") == "miscelaneo"
