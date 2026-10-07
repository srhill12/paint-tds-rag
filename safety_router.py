"""
Rule-based safety-question router.

Technical Data Sheets are not the authoritative source for hazard
information. This module flags questions that should be directed to the
product Safety Data Sheet (SDS) before any TDS-derived answer is shown.
"""

from __future__ import annotations

import re

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

SDS_SEARCH_URL = "https://www.benjaminmoore.com/en-us/documentation_nocards"

SAFETY_NOTICE = (
    "Technical Data Sheets are not the authoritative source for safety "
    "information; consult the Safety Data Sheet (SDS) for the specific "
    "product and base. Search Benjamin Moore documentation: "
    f"{SDS_SEARCH_URL}"
)

_TERM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (
        term,
        re.compile(
            rf"(?<!\w){re.escape(term)}(?!\w)",
            flags=re.IGNORECASE,
        ),
    )
    for term in SAFETY_TERMS
)


def is_safety_question(text: str) -> tuple[bool, list[str]]:
    """Return whether text is a safety question, plus the terms that matched.

    Matching is case-insensitive and requires a whole word or whole phrase.
    Substring hits such as "ventilated" for "ventilation" do not count.
    """
    matched: list[str] = []
    for term, pattern in _TERM_PATTERNS:
        if pattern.search(text or ""):
            matched.append(term)
    return (bool(matched), matched)


def build_response(
    question: str,
    answer: str,
    source_docs: list,
) -> dict:
    """Build the assistant response object, including the routing decision."""
    safety_routed, matched_terms = is_safety_question(question)
    return {
        "question": question,
        "answer": answer,
        "source_docs": source_docs,
        "safety_routed": safety_routed,
        "matched_terms": matched_terms,
    }
