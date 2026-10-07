"""
Rule-based safety-question router.

Technical Data Sheets are not the authoritative source for hazard
information. This module flags questions that should be directed to the
product Safety Data Sheet (SDS) before any TDS-derived answer is shown.
"""

from __future__ import annotations

import re

from config import ROUTER_V2_ENABLED

# Auditable term list. Matching is whole-word or whole-phrase only.
SAFETY_TERMS: tuple[str, ...] = (
    "hazard",
    "hazardous",
    "toxic",
    "toxicity",
    "ventilation",
    "enclosed space",
    "confined space",
    "respirator",
    "PPE",
    "flammable",
    "flammability",
    "fumes",
    "inhalation",
    "inhale",
    "skin contact",
    "eye contact",
    "eyes",
    "ingestion",
    "swallow",
    "swallowed",
    "children",
    "kids",
    "pets",
    "pregnant",
    "pregnancy",
    "first aid",
    "health effects",
    "poisoning",
    "safe to use",
)

# Additional terms gated by ROUTER_V2_ENABLED. SAFETY_TERMS (v1) is unchanged.
SAFETY_TERMS_V2: tuple[str, ...] = (
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

SDS_SEARCH_URL = "https://www.benjaminmoore.com/en-us/documentation_nocards"

SAFETY_NOTICE = (
    "Technical Data Sheets are not the authoritative source for safety "
    "information; consult the Safety Data Sheet (SDS) for the specific "
    "product and base. Search Benjamin Moore documentation: "
    f"{SDS_SEARCH_URL}"
)

def _compile_terms(terms: tuple[str, ...]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    return tuple(
        (
            term,
            re.compile(
                rf"(?<!\w){re.escape(term)}(?!\w)",
                flags=re.IGNORECASE,
            ),
        )
        for term in terms
    )


_TERM_PATTERNS = _compile_terms(SAFETY_TERMS)
_V2_TERM_PATTERNS = _compile_terms(SAFETY_TERMS_V2)


def is_safety_question(
    text: str,
    router_v2: bool | None = None,
) -> tuple[bool, list[str]]:
    """Return whether text is a safety question, plus the terms that matched.

    Matching is case-insensitive and requires a whole word or whole phrase.
    Substring hits such as "ventilated" for "ventilation" do not count.
    When router_v2 is true (or ROUTER_V2_ENABLED if omitted), v2 terms are
    matched in addition to SAFETY_TERMS.
    """
    enabled = ROUTER_V2_ENABLED if router_v2 is None else router_v2
    patterns = _TERM_PATTERNS + _V2_TERM_PATTERNS if enabled else _TERM_PATTERNS
    matched: list[str] = []
    for term, pattern in patterns:
        if pattern.search(text or ""):
            matched.append(term)
    return (bool(matched), matched)


def build_response(
    question: str,
    answer: str,
    source_docs: list,
    router_v2: bool | None = None,
) -> dict:
    """Build the assistant response object, including the routing decision."""
    safety_routed, matched_terms = is_safety_question(question, router_v2=router_v2)
    return {
        "question": question,
        "answer": answer,
        "source_docs": source_docs,
        "safety_routed": safety_routed,
        "matched_terms": matched_terms,
    }
