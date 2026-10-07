"""Deterministic scoring for the Paint TDS eval harness. No LLM judge."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

TESTSET_FIELDS = (
    "id",
    "category",
    "question",
    "expected_value",
    "expected_unit",
    "expected_sku",
    "acceptable_skus",
    "source_file",
    "source_page",
    "evidence_quote",
    "finding_ref",
    "verified_by",
    "notes",
)

CATEGORIES = (
    "answerable",
    "unanswerable",
    "confusion",
    "family",
    "underspecified",
    "safety",
    "safety_negative",
)

# Phrasing instructed by the system prompt: "If the answer is not in the
# provided context, say so clearly rather than guessing."
DECLINE_PATTERNS = (
    r"not in the provided context",
    r"not (?:present|found|available|contained) in the (?:provided |retrieved )?(?:context|excerpts?|documents?)",
    r"(?:provided|retrieved) (?:context|excerpts?|documents?) (?:do not|don't|does not|doesn't) (?:contain|include|cover|mention|have)",
    r"(?:the )?(?:documents?|excerpts?|context) (?:do not|don't|does not|doesn't) (?:contain|include|cover|mention|have)",
    r"do(?:es)? not (?:contain|include|cover|mention) (?:that |this |the )?(?:information|data|value|answer)",
    r"(?:that |this |the )?(?:information|answer) is not (?:in|available|present)",
    r"no information (?:is |was )?(?:available|provided|found|in)",
    r"cannot (?:find|determine|answer|tell|provide)",
    r"can'?t (?:find|determine|answer|tell|provide)",
    r"unable to (?:find|determine|answer|tell|provide)",
    r"i don'?t (?:know|have)",
    r"not specified in (?:the )?(?:provided |retrieved )?(?:context|documents?|excerpts?)",
    r"do(?:es)? not have (?:that |this |the )?(?:information|data)",
    r"isn'?t (?:in|available in) the (?:provided |retrieved )?(?:context|documents?|excerpts?)",
    r"not available in the (?:provided |retrieved )?(?:context|documents?|excerpts?)",
)

CLARIFY_PRODUCT_PATTERNS = (
    r"which product",
    r"which sku",
    r"specify (?:the |a |which )?product",
    r"what product",
    r"which (?:one|item|sku)",
    r"need to know which",
    r"please (?:specify|clarify|identify)(?: the)? product",
    r"could you (?:specify|clarify|tell me) which product",
    r"product (?:are you asking about|do you mean|you mean|you are asking)",
    r"which (?:benjamin moore )?product",
    r"clarify which",
    r"identify (?:the |which )?product",
    r"tell me which product",
    r"ask(?:ing)? which product",
    r"more specific(?:ally)? (?:about )?which product",
)

WORD_ZERO_PATTERN = re.compile(r"\bzero\b", flags=re.IGNORECASE)

# Number (or range) followed by a unit. Used for hedged / unsupported / family fails.
NUMBER_WITH_UNIT_PATTERN = re.compile(
    r"""
    (?P<cmp><\s*|<=\s*|less\s+than\s+|under\s+|below\s+)?
    (?P<num>\d+(?:\.\d+)?)
    (?:
        \s*[–—−\-]\s*
        (?P<num2>\d+(?:\.\d+)?)
    )?
    \s*
    (?P<unit>
        g\s*/\s*l|
        grams?\s*/\s*liters?|
        grams?\s+per\s+lit(?:er|re)|
        lbs?\.?\s*/\s*gal(?:lon)?s?|
        pounds?\s+per\s+gallon|
        sq\.?\s*ft\.?(?:\s*/\s*gal(?:lon)?)?|
        square\s+feet(?:\s+per\s+gallon)?|
        hours?|hrs?|
        minutes?|mins?|
        weeks?|
        °\s*[fc]|
        deg(?:rees?)?\s*[fc]|
        [fc](?=\b)|
        %|percent|
        gloss\s+units?|\bgu\b
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

GLOSS_RANGE_PATTERN = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*[–—−\-]\s*(?P<num2>\d+(?:\.\d+)?)\s*@\s*\d+\s*°",
    flags=re.IGNORECASE,
)

PM_PERCENT_PATTERN = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*±\s*(?P<span>\d+(?:\.\d+)?)\s*%",
    flags=re.IGNORECASE,
)

LBS_PER_GAL_TO_G_L = 119.83
SQ_M_TO_SQ_FT = 10.7639

UNIT_FAMILY_VOC = "voc_g_l"
UNIT_FAMILY_COVERAGE = "coverage_sqft"
UNIT_FAMILY_DURATION = "duration_hours"
UNIT_FAMILY_TEMP = "temp_f"
UNIT_FAMILY_PERCENT = "percent"
UNIT_FAMILY_GLOSS = "gloss"

_UNIT_ALIASES: tuple[tuple[re.Pattern[str], str, float], ...] = (
    (re.compile(r"^g\s*/\s*l$", re.I), UNIT_FAMILY_VOC, 1.0),
    (re.compile(r"^grams?\s*/\s*liters?$", re.I), UNIT_FAMILY_VOC, 1.0),
    (re.compile(r"^grams?\s+per\s+lit(?:er|re)$", re.I), UNIT_FAMILY_VOC, 1.0),
    (re.compile(r"^lbs?\.?\s*/\s*gal(?:lon)?s?$", re.I), UNIT_FAMILY_VOC, LBS_PER_GAL_TO_G_L),
    (re.compile(r"^pounds?\s+per\s+gallon$", re.I), UNIT_FAMILY_VOC, LBS_PER_GAL_TO_G_L),
    (re.compile(r"^sq\.?\s*ft\.?(?:\s*/\s*gal(?:lon)?)?$", re.I), UNIT_FAMILY_COVERAGE, 1.0),
    (re.compile(r"^square\s+feet(?:\s+per\s+gallon)?$", re.I), UNIT_FAMILY_COVERAGE, 1.0),
    (re.compile(r"^hours?$", re.I), UNIT_FAMILY_DURATION, 1.0),
    (re.compile(r"^hrs?$", re.I), UNIT_FAMILY_DURATION, 1.0),
    (re.compile(r"^minutes?$", re.I), UNIT_FAMILY_DURATION, 1.0 / 60.0),
    (re.compile(r"^mins?$", re.I), UNIT_FAMILY_DURATION, 1.0 / 60.0),
    (re.compile(r"^weeks?$", re.I), UNIT_FAMILY_DURATION, 168.0),
    (re.compile(r"^°\s*f$", re.I), UNIT_FAMILY_TEMP, 1.0),
    (re.compile(r"^deg(?:rees?)?\s*f$", re.I), UNIT_FAMILY_TEMP, 1.0),
    (re.compile(r"^f$", re.I), UNIT_FAMILY_TEMP, 1.0),
    (re.compile(r"^°\s*c$", re.I), UNIT_FAMILY_TEMP, "c_to_f"),  # type: ignore[arg-type]
    (re.compile(r"^deg(?:rees?)?\s*c$", re.I), UNIT_FAMILY_TEMP, "c_to_f"),  # type: ignore[arg-type]
    (re.compile(r"^c$", re.I), UNIT_FAMILY_TEMP, "c_to_f"),  # type: ignore[arg-type]
    (re.compile(r"^%$", re.I), UNIT_FAMILY_PERCENT, 1.0),
    (re.compile(r"^percent$", re.I), UNIT_FAMILY_PERCENT, 1.0),
    (re.compile(r"^gloss\s+units?$", re.I), UNIT_FAMILY_GLOSS, 1.0),
    (re.compile(r"^gu$", re.I), UNIT_FAMILY_GLOSS, 1.0),
)

# Expected-unit strings used in the testset (not always identical to extracted text).
EXPECTED_UNIT_TO_FAMILY = {
    "g/l": UNIT_FAMILY_VOC,
    "g / l": UNIT_FAMILY_VOC,
    "grams/liter": UNIT_FAMILY_VOC,
    "grams per liter": UNIT_FAMILY_VOC,
    "lbs/gal": UNIT_FAMILY_VOC,
    "lbs./gal": UNIT_FAMILY_VOC,
    "sq ft/gal": UNIT_FAMILY_COVERAGE,
    "sq. ft./gal": UNIT_FAMILY_COVERAGE,
    "sq ft": UNIT_FAMILY_COVERAGE,
    "hours": UNIT_FAMILY_DURATION,
    "hour": UNIT_FAMILY_DURATION,
    "minutes": UNIT_FAMILY_DURATION,
    "minute": UNIT_FAMILY_DURATION,
    "f": UNIT_FAMILY_TEMP,
    "°f": UNIT_FAMILY_TEMP,
    "c": UNIT_FAMILY_TEMP,
    "°c": UNIT_FAMILY_TEMP,
    "%": UNIT_FAMILY_PERCENT,
    "percent": UNIT_FAMILY_PERCENT,
    "gu": UNIT_FAMILY_GLOSS,
    "sheen": UNIT_FAMILY_GLOSS,
}


@dataclass(frozen=True)
class Quantity:
    low: float
    high: float
    family: str
    comparator: str  # "", "<", ">"
    raw: str


def _compile_any(patterns: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns), flags=re.IGNORECASE)


_DECLINE_RE = _compile_any(DECLINE_PATTERNS)
_CLARIFY_RE = _compile_any(CLARIFY_PRODUCT_PATTERNS)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("\xa0", " ").replace("−", "-").replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def is_decline(text: str) -> bool:
    return bool(_DECLINE_RE.search(normalize_text(text)))


def asks_which_product(text: str) -> bool:
    return bool(_CLARIFY_RE.search(normalize_text(text)))


def _c_to_f(value: float) -> float:
    return value * 9.0 / 5.0 + 32.0


def _unit_to_family(unit: str) -> Optional[tuple[str, float | str]]:
    token = re.sub(r"\s+", " ", (unit or "").strip())
    for pattern, family, factor in _UNIT_ALIASES:
        if pattern.match(token):
            return family, factor
    key = token.lower().replace("°", "")
    # expected-unit aliases
    family = EXPECTED_UNIT_TO_FAMILY.get(token.lower()) or EXPECTED_UNIT_TO_FAMILY.get(key)
    if family:
        factor: float | str = 1.0
        if family == UNIT_FAMILY_VOC and "lb" in token.lower():
            factor = LBS_PER_GAL_TO_G_L
        if family == UNIT_FAMILY_TEMP and token.lower().endswith("c"):
            factor = "c_to_f"
        return family, factor
    return None


def _scale(value: float, factor: float | str) -> float:
    if factor == "c_to_f":
        return _c_to_f(value)
    return value * float(factor)


def _skip_geometry_at(text: str, start: int) -> bool:
    """Ignore temperatures that are gloss-geometry angles ('@ 85°')."""
    prefix = text[max(0, start - 3) : start]
    return bool(re.search(r"@\s*$", prefix))


def extract_quantities(text: str) -> list[Quantity]:
    """Extract number-with-unit values and normalize units."""
    text = normalize_text(text)
    if not text:
        return []
    found: list[Quantity] = []
    occupied: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < b and end > a for a, b in occupied)

    for match in PM_PERCENT_PATTERN.finditer(text):
        center = float(match.group("num"))
        span = float(match.group("span"))
        found.append(
            Quantity(
                low=center - span,
                high=center + span,
                family=UNIT_FAMILY_PERCENT,
                comparator="",
                raw=match.group(0),
            )
        )
        occupied.append(match.span())

    for match in GLOSS_RANGE_PATTERN.finditer(text):
        if overlaps(*match.span()):
            continue
        found.append(
            Quantity(
                low=float(match.group("num")),
                high=float(match.group("num2")),
                family=UNIT_FAMILY_GLOSS,
                comparator="",
                raw=match.group(0),
            )
        )
        occupied.append(match.span())

    for match in NUMBER_WITH_UNIT_PATTERN.finditer(text):
        if overlaps(*match.span()):
            continue
        if _skip_geometry_at(text, match.start()):
            continue
        unit = match.group("unit")
        mapped = _unit_to_family(unit)
        if not mapped:
            continue
        family, factor = mapped
        low = _scale(float(match.group("num")), factor)
        high = low
        if match.group("num2"):
            high = _scale(float(match.group("num2")), factor)
            if high < low:
                low, high = high, low
        cmp_raw = (match.group("cmp") or "").strip().lower()
        comparator = "<" if cmp_raw else ""
        found.append(
            Quantity(
                low=low,
                high=high,
                family=family,
                comparator=comparator,
                raw=match.group(0),
            )
        )
        occupied.append(match.span())

    # Word "zero" next to a VOC unit, e.g. "Zero g/L".
    for match in re.finditer(
        r"\bzero\b\s*(?P<unit>g\s*/\s*l|grams?\s*/\s*liters?)",
        text,
        flags=re.IGNORECASE,
    ):
        if overlaps(*match.span()):
            continue
        found.append(
            Quantity(
                low=0.0,
                high=0.0,
                family=UNIT_FAMILY_VOC,
                comparator="",
                raw=match.group(0),
            )
        )
        occupied.append(match.span())

    return found


def parse_expected_quantity(
    expected_value: Any,
    expected_unit: Any,
) -> Optional[Quantity]:
    if expected_value is None or str(expected_value).strip() == "":
        return None
    raw = normalize_text(str(expected_value))
    unit = normalize_text(str(expected_unit or ""))
    family_info = _unit_to_family(unit) if unit else None
    family = family_info[0] if family_info else None
    factor: float | str = family_info[1] if family_info else 1.0

    if raw.lower() in {"zero", "0"}:
        if family is None:
            family = UNIT_FAMILY_VOC
        return Quantity(0.0, 0.0, family, "", raw)

    comparator = ""
    if raw.startswith("<") or raw.lower().startswith("less than"):
        comparator = "<"
        raw = re.sub(r"^(?:<|less than)\s*", "", raw, flags=re.IGNORECASE)

    pm = re.match(
        rf"^(?P<num>\d+(?:\.\d+)?)\s*±\s*(?P<span>\d+(?:\.\d+)?)(?:\s*%)?$",
        raw,
    )
    if pm:
        center = float(pm.group("num"))
        span = float(pm.group("span"))
        if family is None:
            family = UNIT_FAMILY_PERCENT
        return Quantity(
            _scale(center - span, factor),
            _scale(center + span, factor),
            family,
            "",
            expected_value,
        )

    range_match = re.match(
        rf"^(?P<a>\d+(?:\.\d+)?)\s*-\s*(?P<b>\d+(?:\.\d+)?)$",
        raw,
    )
    if range_match:
        a = float(range_match.group("a"))
        b = float(range_match.group("b"))
        if family is None:
            return None
        low, high = _scale(a, factor), _scale(b, factor)
        if high < low:
            low, high = high, low
        return Quantity(low, high, family, comparator, str(expected_value))

    num_match = re.match(r"^(?P<num>\d+(?:\.\d+)?)$", raw)
    if num_match and family:
        value = _scale(float(num_match.group("num")), factor)
        return Quantity(value, value, family, comparator, str(expected_value))

    # Fall back to extracting from the combined string.
    extracted = extract_quantities(f"{expected_value} {expected_unit}".strip())
    return extracted[0] if extracted else None


def _values_close(a: float, b: float, rel_tol: float = 0.02) -> bool:
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= max(rel_tol * scale, 0.05)


def quantities_match(expected: Quantity, observed: Quantity, rel_tol: float = 0.02) -> bool:
    """Match an expected testset value against a quantity extracted from an answer."""
    if expected.family != observed.family:
        return False

    if expected.low != expected.high or observed.low != observed.high:
        overlap = min(expected.high, observed.high) >= max(expected.low, observed.low) - 0.05
        if overlap:
            return True
        return _values_close(expected.low, observed.low, rel_tol) and _values_close(
            expected.high, observed.high, rel_tol
        )

    if expected.comparator == "<":
        if observed.comparator == "<" and _values_close(observed.low, expected.low, rel_tol):
            return True
        return observed.low <= expected.low + 0.05 and _values_close(
            observed.low, expected.low, rel_tol
        )

    return _values_close(expected.low, observed.low, rel_tol)


def quantity_attested(item: Quantity, chunk_q: Quantity, rel_tol: float = 0.02) -> bool:
    """True only if the answer quantity's number+unit pair is attested in a chunk.

    Extracted chunk quantities already require a number and unit together, so a
    bare number elsewhere in the chunk (different unit, or no unit) cannot
    support the value. Point values match a chunk range only at an endpoint,
    where that number appears with the unit; interior overlap is not enough.
    """
    if item.family != chunk_q.family:
        return False
    item_range = not _values_close(item.low, item.high, rel_tol)
    chunk_range = not _values_close(chunk_q.low, chunk_q.high, rel_tol)
    if item_range and chunk_range:
        return _values_close(item.low, chunk_q.low, rel_tol) and _values_close(
            item.high, chunk_q.high, rel_tol
        )
    if not item_range and chunk_range:
        return _values_close(item.low, chunk_q.low, rel_tol) or _values_close(
            item.low, chunk_q.high, rel_tol
        )
    if item_range and not chunk_range:
        return _values_close(chunk_q.low, item.low, rel_tol) or _values_close(
            chunk_q.low, item.high, rel_tol
        )
    if item.comparator and chunk_q.comparator and item.comparator != chunk_q.comparator:
        return False
    return _values_close(item.low, chunk_q.low, rel_tol)


def numeric_match(answer: str, expected_value: Any, expected_unit: Any) -> bool:
    expected = parse_expected_quantity(expected_value, expected_unit)
    if expected is None:
        return False
    observed = extract_quantities(answer)
    return any(quantities_match(expected, item) for item in observed)


def is_hedged(answer: str) -> bool:
    return is_decline(answer) and bool(extract_quantities(answer))


def citation_hit(
    retrieved_skus: Iterable[str],
    expected_sku: str,
    acceptable_skus: Optional[Iterable[str]] = None,
) -> bool:
    wanted = {s.strip().upper() for s in ([expected_sku] if expected_sku else [])}
    if acceptable_skus:
        wanted.update(s.strip().upper() for s in acceptable_skus if s)
    wanted.discard("")
    have = {s.strip().upper() for s in retrieved_skus if s}
    return bool(wanted & have)


def top1_match(retrieved_skus: Iterable[str], expected_sku: str) -> bool:
    skus = [s.strip().upper() for s in retrieved_skus if s and s.strip()]
    if not skus or not expected_sku:
        return False
    return skus[0] == expected_sku.strip().upper()


def expected_in_topk(retrieved_skus: Iterable[str], expected_sku: str) -> bool:
    if not expected_sku:
        return False
    target = expected_sku.strip().upper()
    return target in {s.strip().upper() for s in retrieved_skus if s}


SOURCE_CORRECT = "source_correct"
MISATTRIBUTED = "misattributed"
UNSUPPORTED = "unsupported"


def target_sku_set(
    expected_sku: str = "",
    acceptable_skus: Optional[Iterable[str]] = None,
) -> set[str]:
    wanted = {s.strip().upper() for s in ([expected_sku] if expected_sku else [])}
    if acceptable_skus:
        wanted.update(s.strip().upper() for s in acceptable_skus if s)
    wanted.discard("")
    return wanted


def classify_value_source(
    attesting_skus: Iterable[str],
    expected_sku: str = "",
    acceptable_skus: Optional[Iterable[str]] = None,
) -> str:
    """Label a value as source_correct, misattributed, or unsupported."""
    attested = [s for s in attesting_skus]
    if not attested:
        return UNSUPPORTED
    targets = target_sku_set(expected_sku, acceptable_skus)
    attested_norm = {s.strip().upper() for s in attested if s and str(s).strip()}
    if targets and attested_norm & targets:
        return SOURCE_CORRECT
    return MISATTRIBUTED


def value_support_decisions(
    answer: str,
    chunk_texts: Iterable[str],
    chunk_skus: Optional[Iterable[str]] = None,
    expected_sku: str = "",
    acceptable_skus: Optional[Iterable[str]] = None,
) -> list[dict[str, Any]]:
    """For each number-with-unit in the answer, which chunks/SKUs attest it."""
    chunks = list(chunk_texts)
    sku_list = list(chunk_skus) if chunk_skus is not None else [""] * len(chunks)
    if len(sku_list) < len(chunks):
        sku_list.extend([""] * (len(chunks) - len(sku_list)))
    parsed_chunks = [
        (extract_quantities(chunk), chunk, sku_list[i])
        for i, chunk in enumerate(chunks)
    ]
    decisions: list[dict[str, Any]] = []
    for item in extract_quantities(answer):
        attesting_skus: list[str] = []
        matching_chunks: list[str] = []
        for quantities, chunk, sku in parsed_chunks:
            if any(quantity_attested(item, chunk_q) for chunk_q in quantities):
                attesting_skus.append(sku)
                matching_chunks.append(chunk)
        source = classify_value_source(attesting_skus, expected_sku, acceptable_skus)
        decisions.append(
            {
                "value": item.raw,
                "supported": source != UNSUPPORTED,
                "source": source,
                "attesting_skus": attesting_skus,
                "matching_chunk": matching_chunks[0] if matching_chunks else "",
                "matching_chunks": matching_chunks,
            }
        )
    return decisions


def unsupported_values(
    answer: str,
    chunk_texts: Iterable[str],
) -> list[str]:
    """Return raw answer quantities that do not appear in any retrieved chunk."""
    return [
        decision["value"]
        for decision in value_support_decisions(answer, chunk_texts)
        if not decision["supported"]
    ]


def has_attribution(
    answer: str,
    expected_sku: str = "",
    retrieved_skus: Optional[Iterable[str]] = None,
    retrieved_products: Optional[Iterable[str]] = None,
) -> bool:
    text = normalize_text(answer)
    if not text:
        return False
    candidates: list[str] = []
    if expected_sku:
        candidates.append(expected_sku)
    if retrieved_skus:
        candidates.extend(s for s in retrieved_skus if s)
    if retrieved_products:
        candidates.extend(p for p in retrieved_products if p and len(p.strip()) >= 4)
    for candidate in candidates:
        token = normalize_text(str(candidate))
        if token and re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text, flags=re.IGNORECASE):
            return True
        # SKUs like N549 / HP2200 often appear without word boundaries around digits.
        compact = re.sub(r"[\s\-_/]", "", token)
        if compact and re.search(compact, re.sub(r"[\s\-_/]", "", text), flags=re.IGNORECASE):
            return True
    return False


def is_paraphrased_safety(row: dict) -> bool:
    notes = (row.get("notes") or "").lower()
    return "paraphrased" in notes


def score_row(
    row: dict,
    answer: str,
    retrieved_skus: list[str],
    chunk_texts: list[str],
    safety_routed: bool,
    retrieved_products: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Score one eval row. Hedged answers fail in every category."""
    category = row.get("category") or ""
    expected_sku = row.get("expected_sku") or ""
    acceptable = row.get("acceptable_skus") or []
    expected_value = row.get("expected_value")
    expected_unit = row.get("expected_unit")

    decline = is_decline(answer)
    clarify = asks_which_product(answer)
    quantities = extract_quantities(answer)
    hedged = decline and bool(quantities)
    support = value_support_decisions(
        answer,
        chunk_texts,
        chunk_skus=retrieved_skus,
        expected_sku=expected_sku,
        acceptable_skus=acceptable,
    )
    missing_values = [item["value"] for item in support if item["source"] == UNSUPPORTED]
    source_correct_values = [
        {"value": item["value"], "attesting_skus": item["attesting_skus"]}
        for item in support
        if item["source"] == SOURCE_CORRECT
    ]
    misattributed_values = [
        {"value": item["value"], "attesting_skus": item["attesting_skus"]}
        for item in support
        if item["source"] == MISATTRIBUTED
    ]
    attributed = has_attribution(answer, expected_sku, retrieved_skus, retrieved_products)
    cited = citation_hit(retrieved_skus, expected_sku, acceptable)
    top1 = top1_match(retrieved_skus, expected_sku)
    in_topk = expected_in_topk(retrieved_skus, expected_sku)
    numeric = numeric_match(answer, expected_value, expected_unit)

    passed = False
    failure_reason = ""

    if hedged:
        passed = False
        failure_reason = "hedged"
    elif category == "answerable":
        passed = numeric
        if not passed:
            failure_reason = "numeric_mismatch"
    elif category == "unanswerable":
        passed = decline
        if not passed:
            failure_reason = "no_decline"
    elif category == "confusion":
        passed = top1
        if not passed:
            failure_reason = "wrong_top_sku" if retrieved_skus else "no_sources"
    elif category in {"family", "underspecified"}:
        if quantities:
            passed = False
            failure_reason = "numeric_specification"
        elif not clarify:
            passed = False
            failure_reason = "did_not_ask_product"
        else:
            passed = True
    elif category == "safety":
        passed = bool(safety_routed)
        if not passed:
            failure_reason = "safety_not_routed"
    elif category == "safety_negative":
        passed = not bool(safety_routed)
        if not passed:
            failure_reason = "safety_false_route"
    else:
        failure_reason = f"unknown_category:{category}"

    return {
        "passed": passed,
        "failure_reason": failure_reason,
        "hedged": hedged,
        "decline": decline,
        "asks_which_product": clarify,
        "numeric_match": numeric,
        "citation": cited,
        "top1_match": top1,
        "expected_in_topk": in_topk,
        "wrong_top_sku": bool(expected_sku) and not top1,
        "safety_routed": bool(safety_routed),
        "paraphrased_safety": is_paraphrased_safety(row),
        "quantities": [q.raw for q in quantities],
        "value_support": support,
        "unsupported_values": missing_values,
        "source_correct_values": source_correct_values,
        "misattributed_values": misattributed_values,
        "has_source_correct": bool(source_correct_values),
        "has_misattributed": bool(misattributed_values),
        "has_unsupported": bool(missing_values),
        "has_value": bool(quantities),
        "attribution": attributed if quantities else None,
    }
