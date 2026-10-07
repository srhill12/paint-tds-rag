"""Tests for the rule-based safety-question router."""

from safety_router import SAFETY_TERMS_V2, build_response, is_safety_question

POSITIVE_QUESTIONS = [
    "Is this product safe to use in enclosed spaces?",
    "What PPE do I need when spraying V201?",
    "Is Aura safe around pets?",
    "What happens if paint gets in my eyes?",
    "Is this flammable?",
]

NEGATIVE_QUESTIONS = [
    "What is the dry time for Regal Select?",
    "What is the VOC content of N549?",
    "How many square feet does a gallon cover?",
    "What is the minimum application temperature?",
]

# Near-misses are documented with the actual router result. They are not
# acceptance cases the term list was tuned to pass or fail.
NEAR_MISSES = [
    {
        "question": "Can I recoat in a well-ventilated room after 4 hours?",
        "triggered": False,
        "matched_terms": [],
        "reason": (
            "well-ventilated is a different word from ventilation, so "
            "whole-word matching does not fire"
        ),
    },
    {
        "question": "What safety yellow colors does AC-14 come in?",
        "triggered": False,
        "matched_terms": [],
        "reason": (
            "safety is not in the term list, and the phrase safe to use "
            "does not match safety yellow"
        ),
    },
]


def test_positive_questions_trigger():
    for question in POSITIVE_QUESTIONS:
        triggered, matched_terms = is_safety_question(question)
        assert triggered, f"expected a safety route for {question!r}"
        assert matched_terms, f"expected matched terms for {question!r}"


def test_negative_questions_do_not_trigger():
    for question in NEGATIVE_QUESTIONS:
        triggered, matched_terms = is_safety_question(question)
        assert not triggered, (
            f"did not expect a safety route for {question!r}, "
            f"matched {matched_terms}"
        )
        assert matched_terms == []


def test_near_misses_document_actual_classification():
    """Record how near-misses classify; do not tune SAFETY_TERMS to flip them."""
    for case in NEAR_MISSES:
        triggered, matched_terms = is_safety_question(case["question"])
        assert triggered == case["triggered"], (
            f"{case['question']!r} classified triggered={triggered} "
            f"terms={matched_terms}; documented triggered={case['triggered']} "
            f"because {case['reason']}"
        )
        assert matched_terms == case["matched_terms"]


def test_substring_ventilation_does_not_match_ventilated():
    triggered, matched_terms = is_safety_question(
        "Can I recoat in a well-ventilated room after 4 hours?"
    )
    assert "ventilation" not in matched_terms
    assert triggered is False


def test_response_object_includes_routing_fields():
    result = build_response("Is this flammable?", "tds answer", [])
    assert result["safety_routed"] is True
    assert result["matched_terms"] == ["flammable"]
    assert result["answer"] == "tds answer"
    assert result["question"] == "Is this flammable?"

    result = build_response("What is the dry time for Regal Select?", "ok", [])
    assert result["safety_routed"] is False
    assert result["matched_terms"] == []


def test_v2_terms_gated_by_flag():
    question = "will the smell bother my newborn?"
    off, terms_off = is_safety_question(question, router_v2=False)
    assert off is False
    assert terms_off == []
    on, terms_on = is_safety_question(question, router_v2=True)
    assert on is True
    assert "smell" in terms_on
    assert "newborn" in terms_on
    default, _ = is_safety_question(question)
    assert default is True


def test_v2_shipped_term_list_matches_specified():
    assert SAFETY_TERMS_V2 == (
        "smell",
        "odor",
        "odour",
        "newborn",
        "infant",
        "baby",
        "nursery",
        "headache",
        "dizzy",
        "nausea",
        "breathing",
        "breathe",
        "asthma",
        "allergic",
        "allergy",
    )
    assert "nauseous" not in SAFETY_TERMS_V2
    routed, terms = is_safety_question(
        "I feel nauseous after using Corotech V201.", router_v2=True
    )
    assert "nauseous" not in terms
    odour_hit, odour_terms = is_safety_question(
        "Does the odour linger?", router_v2=True
    )
    assert odour_hit is True
    assert "odour" in odour_terms


def test_v1_terms_still_match_when_v2_enabled():
    triggered, matched = is_safety_question("Is this flammable?", router_v2=True)
    assert triggered is True
    assert "flammable" in matched
