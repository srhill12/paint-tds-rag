"""Schema and scoring smoke tests. Does not call Ollama."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from eval.scoring import (
    CATEGORIES,
    MISATTRIBUTED,
    NOT_APPLICABLE,
    SOURCE_CORRECT,
    TESTSET_FIELDS,
    asks_which_product,
    citation_hit,
    extract_quantities,
    has_attribution,
    is_decline,
    is_hedged,
    numeric_match,
    score_row,
    top1_match,
    unsupported_values,
    value_support_decisions,
)

ROOT = Path(__file__).resolve().parent.parent
TESTSET_PATH = ROOT / "eval" / "testset_v1.jsonl"

TARGET_COUNTS = {
    "answerable": 16,
    "unanswerable": 8,
    "confusion": 16,
    "family": 4,
    "underspecified": 5,
    "safety": 8,
    "safety_negative": 5,
}

FINDING_QUERIES = {
    "F-001": "Is this product safe to use in enclosed spaces?",
    "F-002": "what is the voc content of n549?",
    "F-003": "does regal select give off fumes while drying?",
    "F-004": "what is the drying time of regal select?",
    "F-005": "will the smell bother my newborn?",
}


def load_testset() -> list[dict]:
    rows = []
    with TESTSET_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_testset_schema_and_counts():
    rows = load_testset()
    assert rows, "testset is empty"
    counts = Counter()
    ids = []
    unverified = []
    for row in rows:
        missing = [field for field in TESTSET_FIELDS if field not in row]
        assert not missing, f"{row.get('id')} missing {missing}"
        assert row["category"] in CATEGORIES
        verified = str(row.get("verified_by") or "").strip()
        if not verified:
            unverified.append(row.get("id"))
        else:
            assert re.fullmatch(r"Steven Hill \d{4}-\d{2}-\d{2}", verified), row.get("id")
        assert isinstance(row["acceptable_skus"], list)
        ids.append(row["id"])
        counts[row["category"]] += 1
    assert not unverified, f"unverified ids: {unverified}"
    assert len(ids) == len(set(ids))
    assert counts == TARGET_COUNTS
    assert sum(TARGET_COUNTS.values()) == 62


def test_finding_queries_are_verbatim():
    rows = load_testset()
    by_ref = {row["finding_ref"]: row for row in rows if row.get("finding_ref")}
    for ref, question in FINDING_QUERIES.items():
        assert ref in by_ref, f"missing {ref}"
        assert by_ref[ref]["question"] == question


def test_paraphrased_safety_count():
    rows = load_testset()
    safety = [row for row in rows if row["category"] == "safety"]
    paraphrased = [row for row in safety if "paraphrased" in (row.get("notes") or "").lower()]
    assert len(paraphrased) == 3
    assert len(safety) - len(paraphrased) == 5


def test_numeric_match_range_and_units():
    assert numeric_match("Coverage is 400 – 450 sq. ft. per gallon.", "400-450", "sq ft/gal")
    assert numeric_match("VOC is 320 g/L (2.67 lbs/gal).", "320", "g/L")
    assert numeric_match("VOC is 2.67 lbs/gal.", "320", "g/L")
    assert numeric_match("Dry to touch in 1 hour.", "1", "hours")
    assert numeric_match("To touch: 30 minutes.", "30", "minutes")
    assert numeric_match("Apply at a minimum of 35°F (1.7°C).", "35", "F")
    assert numeric_match("Do not apply below 1.7 °C.", "35", "F")
    assert numeric_match("VOC < 50 g/L", "<50", "g/L")
    assert numeric_match("Zero g/L", "0", "g/L")
    assert not numeric_match("Recoat in 12 hours.", "4", "hours")


def test_decline_matches_system_prompt_phrasing():
    assert is_decline(
        "That answer is not in the provided context, so I will not guess."
    )
    assert is_decline(
        "The provided documents do not contain pricing information."
    )
    assert not is_decline("N549 has a VOC content of less than 50 g/L.")


# Product-aware una-01 answer from eval/results/20261007T204608Z_67c76a9.
UNA01_PRODUCT_AWARE_ANSWER = (
    "Product: Regal® Select Premium Interior Paint & Primer Eggshell Finish N549 (N549)\n"
    "The Technical Data Sheet does not list the price of Regal® Select \uf0aa Eggshell N549. "
    "It states the weight per gallon is 11.5 lbs."
)


def test_una01_does_not_list_is_hedged_decline():
    assert is_decline(UNA01_PRODUCT_AWARE_ANSWER)
    row = {
        "category": "unanswerable",
        "expected_value": None,
        "expected_unit": None,
        "expected_sku": "",
        "acceptable_skus": [],
        "notes": "",
    }
    scored = score_row(
        row,
        UNA01_PRODUCT_AWARE_ANSWER,
        ["N549"],
        ["Weight Per Gallon\n11.5 lbs."],
        False,
    )
    assert scored["decline"] is True
    assert any("11.5" in q and "lb" in q.lower() for q in scored["quantities"])
    assert scored["hedged"] is True
    assert scored["passed"] is False
    assert scored["failure_reason"] == "hedged"


def test_lbs_per_gallon_attested_by_sheet_form():
    """conf-07a: '3.29 lbs. per gallon' is VOC, attested by '3.29 Lbs./Gallon'."""
    answer = (
        "The VOC content of Corotech Alkyd Shop Coat Primer V133 is "
        "394 grams per liter or 3.29 lbs. per gallon."
    )
    chunks = ["394 Grams/Liter \n3.29 Lbs./Gallon"]
    qs = extract_quantities(answer)
    voc_lbs = [q for q in qs if "3.29" in q.raw]
    assert voc_lbs, f"expected to extract 3.29 lbs per gallon, got {[q.raw for q in qs]}"
    assert voc_lbs[0].family == "voc_g_l"
    decisions = value_support_decisions(
        answer, chunks, chunk_skus=["V133"], expected_sku="V133"
    )
    hit = next(d for d in decisions if "3.29" in d["value"])
    assert hit["supported"] is True
    assert hit["source"] == SOURCE_CORRECT
    assert "V133" in hit["attesting_skus"]


def test_lbs_per_gallon_unsupported_when_absent_from_chunks():
    answer = "VOC is 3.29 lbs. per gallon."
    chunks = ["Weight Per Gallon\n11.5 lbs."]
    missing = unsupported_values(answer, chunks)
    assert any("3.29" in item for item in missing), missing


def test_hedged_answer_fails_every_category():
    answer = (
        "The VOC is not in the provided context, but it is typically 49 g/L."
    )
    assert is_hedged(answer)
    row = {
        "category": "unanswerable",
        "expected_value": None,
        "expected_unit": None,
        "expected_sku": "",
        "acceptable_skus": [],
        "notes": "",
    }
    scored = score_row(row, answer, ["N549"], ["VOC < 50 g/L"], False)
    assert scored["hedged"] is True
    assert scored["passed"] is False
    assert scored["failure_reason"] == "hedged"

    row["category"] = "answerable"
    row["expected_value"] = "49"
    row["expected_unit"] = "g/L"
    row["expected_sku"] = "N549"
    scored = score_row(row, answer, ["N549"], ["VOC 49 g/L"], False)
    assert scored["passed"] is False
    assert scored["failure_reason"] == "hedged"


def test_unsupported_value_detected():
    answer = "N549 dries in 8 hours."
    chunks = ["To Recoat: 1 – 2 hours", "To Touch: 1 hour"]
    missing = unsupported_values(answer, chunks)
    assert missing
    assert any("8" in item for item in missing)


# Exact To Touch / To Recoat block from cleaned_texts/263_TDS_US.txt (F-004).
F004_263_DRY_SECTION = (
    "Dry Time @ 77 °F \n"
    "(25 °C) @ 50% RH \n"
    "To Touch:  \n"
    "1 hour \n"
    "To Recoat:  \n"
    "4 hours \n"
)
F004_ANSWER = (
    "Regal Select dries tack-free in 2 hours and can be recoated in 8 hours."
)


def test_f004_unsupported_values_against_263_dry_section():
    path = ROOT / "cleaned_texts" / "263_TDS_US.txt"
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "To Touch:" in text
        assert "1 hour" in text
        assert "To Recoat:" in text
        assert "4 hours" in text
        start = text.index("Dry Time @ 77 °F")
        end = text.index("4 hours", start) + len("4 hours")
        chunk = text[start:end]
    else:
        chunk = F004_263_DRY_SECTION

    missing = unsupported_values(F004_ANSWER, [chunk])
    missing_norm = {item.lower().replace(" ", "") for item in missing}
    assert "8hours" in missing_norm, f"8 hours must be unsupported, got {missing}"
    assert "2hours" in missing_norm, f"2 hours must be unsupported, got {missing}"

    decisions = value_support_decisions(F004_ANSWER, [chunk])
    by_raw = {d["value"].lower().replace(" ", ""): d for d in decisions}
    assert by_raw["8hours"]["supported"] is False
    assert by_raw["2hours"]["supported"] is False


def test_unsupported_ignores_bare_number_with_other_unit():
    """A bare 8 (or 8 mils) must not support '8 hours'."""
    answer = "Recoat in 8 hours."
    chunks = ["Recommended film thickness: 8 mils. To Recoat: 4 hours"]
    missing = unsupported_values(answer, chunks)
    assert any("8" in item and "hour" in item.lower() for item in missing)


def test_f004_pattern_value_is_misattributed():
    """A value attested only by a non-target SKU chunk is misattributed."""
    row = {
        "category": "family",
        "expected_value": None,
        "expected_unit": None,
        "expected_sku": "263",
        "acceptable_skus": [],
        "notes": "",
    }
    other_chunk = (
        "DRYING TIME: Dries tack free in 2 hours. Can be recoated in 8 hours."
    )
    scored = score_row(
        row,
        F004_ANSWER,
        ["263", "V440"],
        [F004_263_DRY_SECTION, other_chunk],
        False,
    )
    by_value = {
        item["value"].lower().replace(" ", ""): item
        for item in scored["misattributed_values"]
    }
    assert "2hours" in by_value
    assert "8hours" in by_value
    assert "V440" in by_value["2hours"]["attesting_skus"]
    assert "V440" in by_value["8hours"]["attesting_skus"]
    assert "263" not in by_value["2hours"]["attesting_skus"]
    assert scored["has_misattributed"] is True
    support = {
        item["value"].lower().replace(" ", ""): item for item in scored["value_support"]
    }
    assert support["2hours"]["source"] == MISATTRIBUTED
    assert support["8hours"]["source"] == MISATTRIBUTED


def test_source_labels_not_applicable_without_expected_sku():
    """No expected_sku outside family/underspecified: attested values are N/A."""
    row = {
        "category": "unanswerable",
        "expected_value": None,
        "expected_unit": None,
        "expected_sku": "",
        "acceptable_skus": [],
        "notes": "",
    }
    scored = score_row(
        row,
        "The documents do not contain pricing. VOC is < 50 g/L.",
        ["N549"],
        ["VOC < 50 g/L"],
        False,
    )
    assert scored["has_source_correct"] is False
    assert scored["has_misattributed"] is False
    assert scored["value_support"]
    assert scored["value_support"][0]["source"] == NOT_APPLICABLE
    assert scored["source_correct_values"] == []
    assert scored["misattributed_values"] == []

    family = {
        "category": "family",
        "expected_value": None,
        "expected_unit": None,
        "expected_sku": "",
        "acceptable_skus": [],
        "notes": "",
    }
    family_scored = score_row(
        family,
        "Regal Select can be recoated in 4 hours.",
        ["N400"],
        ["To Recoat: 4 Hours"],
        False,
    )
    assert family_scored["has_misattributed"] is True
    assert family_scored["value_support"][0]["source"] == MISATTRIBUTED


def test_value_attested_by_target_sku_is_source_correct():
    row = {
        "category": "answerable",
        "expected_value": "1-2",
        "expected_unit": "hours",
        "expected_sku": "N549",
        "acceptable_skus": ["N549"],
        "notes": "",
    }
    scored = score_row(
        row,
        "N549 can be recoated in 1 – 2 hours.",
        ["N549", "N547"],
        ["To Recoat:\n1 – 2 hours", "To Recoat: 4 hours"],
        False,
    )
    assert scored["has_source_correct"] is True
    assert scored["has_misattributed"] is False
    assert scored["source_correct_values"]
    assert "N549" in scored["source_correct_values"][0]["attesting_skus"]
    assert scored["value_support"][0]["source"] == SOURCE_CORRECT


def test_supported_value_not_flagged():
    answer = "N549 can be recoated in 1 – 2 hours."
    chunks = ["To Recoat:\n1 – 2 hours"]
    assert unsupported_values(answer, chunks) == []


def test_citation_and_top1():
    assert citation_hit(["HP2200", "N549"], "N549", [])
    assert citation_hit(["CHP2200"], "HP2200", ["CHP2200"])
    assert not citation_hit(["HP2200"], "CHP2200", [])
    assert top1_match(["N549", "N547"], "N549")
    assert not top1_match(["N547", "N549"], "N549")


def test_family_requires_clarify_and_rejects_numeric():
    row = {
        "category": "family",
        "expected_value": None,
        "expected_unit": None,
        "expected_sku": "",
        "acceptable_skus": [],
        "notes": "",
    }
    clarify = "Which product in the Regal Select line do you mean?"
    scored = score_row(row, clarify, ["N549"], ["To Recoat: 1 – 2 hours"], False)
    assert scored["passed"] is True
    numeric = "Regal Select can be recoated in 4 hours."
    scored = score_row(row, numeric, ["N400"], ["To Recoat: 4 Hours"], False)
    assert scored["passed"] is False
    assert scored["failure_reason"] == "numeric_specification"
    assert asks_which_product(clarify)


def test_attribution_detects_sku_in_answer():
    assert has_attribution(
        "According to the N549 sheet, VOC is < 50 g/L.",
        expected_sku="N549",
        retrieved_skus=["N549"],
        retrieved_products=["Regal Select Interior Eggshell"],
    )
    assert not has_attribution(
        "VOC is < 50 g/L.",
        expected_sku="N549",
        retrieved_skus=["N549"],
        retrieved_products=["Regal Select Interior Eggshell N549"],
    )


def test_extract_quantities_handles_zero_and_gloss():
    voc = extract_quantities("VOC Zero g/L")
    assert voc and voc[0].low == 0
    gloss = extract_quantities("Sheen 20 – 35 @ 60°")
    assert gloss and gloss[0].family == "gloss"
    assert gloss[0].low == 20
    assert gloss[0].high == 35


def test_score_answerable_numeric_pass():
    row = {
        "category": "answerable",
        "expected_value": "<50",
        "expected_unit": "g/L",
        "expected_sku": "N549",
        "acceptable_skus": ["N549"],
        "notes": "",
    }
    answer = "Regal Select Interior Eggshell N549 has a VOC of < 50 g/L."
    scored = score_row(row, answer, ["N549"], ["VOC\n< 50 g/L"], False)
    assert scored["passed"] is True
    assert scored["numeric_match"] is True
    assert scored["citation"] is True
    assert scored["attribution"] is True
