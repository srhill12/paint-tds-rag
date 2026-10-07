"""Product resolution and product-aware planning tests. No Ollama."""

from product_index import (
    KIND_FAMILY,
    KIND_NONE,
    KIND_SINGLE,
    SPEC_TERMS,
    has_spec_intent,
    is_deictic,
    plan_query,
    resolve_product,
)
from safety_router import is_safety_question


def test_spec_terms_are_one_constant():
    required = {
        "dry",
        "dry time",
        "recoat",
        "voc",
        "coverage",
        "spread rate",
        "application temperature",
        "sheen",
        "gloss",
        "solids",
        "viscosity",
        "flash point",
        "thinning",
        "film thickness",
    }
    have = {term.lower() for term in SPEC_TERMS}
    assert required <= have


def test_sku_case_and_whole_token():
    lower = resolve_product("n549")
    upper = resolve_product("N549")
    assert lower.kind == KIND_SINGLE and lower.sku == "N549"
    assert upper.kind == KIND_SINGLE and upper.sku == "N549"
    v201 = resolve_product("What is the VOC of V201?")
    cv201 = resolve_product("What is the VOC of CV201?")
    assert v201.kind == KIND_SINGLE and v201.sku == "V201"
    assert cv201.kind == KIND_SINGLE and cv201.sku == "CV201"
    assert v201.sku != cv201.sku


def test_leading_zeros_only_when_manifest_confirms():
    """0549 is not a manifest SKU and is not the same product as N549."""
    ref = resolve_product("0549")
    assert ref.kind == KIND_NONE
    assert ref.sku != "N549"
    # Numeric SKU 532 does exist; a leading-zero form should resolve to it.
    padded = resolve_product("0532")
    assert padded.kind == KIND_SINGLE
    assert padded.sku == "532"


def test_regal_select_family_versus_full_name():
    family = resolve_product("What is the recoat time for Regal Select?")
    assert family.kind == KIND_FAMILY
    assert family.family == "Regal Select"
    assert "N549" in family.skus
    assert len(family.skus) > 1
    single = resolve_product(
        "Regal Select Premium Interior Paint & Primer Eggshell Finish"
    )
    assert single.kind == KIND_SINGLE
    assert single.sku == "N549"


def test_deictic_this_product_and_it():
    this_product = resolve_product("Is this product safe indoors?")
    assert this_product.kind == KIND_NONE
    assert this_product.deictic is True
    on_it = resolve_product("What is the dry time on it?")
    assert on_it.kind == KIND_NONE
    assert on_it.deictic is True
    assert is_deictic("the paint looks done")


def test_aura_pets_is_family_without_spec_intent():
    ref = resolve_product("Is Aura safe around pets?")
    assert ref.kind == KIND_FAMILY
    assert ref.family == "Aura"
    assert has_spec_intent("Is Aura safe around pets?") is False
    plan = plan_query("Is Aura safe around pets?")
    assert plan.action == "unfiltered"
    routed, terms = is_safety_question("Is Aura safe around pets?")
    assert routed is True
    assert "pets" in terms


def test_aura_cost_is_family_without_spec_intent():
    ref = resolve_product("What does Aura cost?")
    assert ref.kind == KIND_FAMILY
    assert has_spec_intent("What does Aura cost?") is False
    assert plan_query("What does Aura cost?").action == "unfiltered"


def test_plan_single_filters_to_sku():
    plan = plan_query("what is the voc content of n549?")
    assert plan.action == "filtered"
    assert plan.ref.sku == "N549"
    assert plan.spec_intent is True


def test_plan_family_with_spec_clarifies():
    plan = plan_query("what is the drying time of regal select?")
    assert plan.action == "clarify"
    assert plan.ref.kind == KIND_FAMILY
    assert "N549" in plan.ref.skus
    assert "Which product" in plan.clarify_text
    assert "N549" in plan.clarify_text
    assert plan.spec_intent is True


def test_plan_none_with_spec_or_deictic_clarifies():
    spec = plan_query("What is the VOC?")
    assert spec.action == "clarify"
    assert spec.ref.kind == KIND_NONE
    assert "product or SKU" in spec.clarify_text
    deictic = plan_query("Is this product washable?")
    assert deictic.action == "clarify"
    assert deictic.ref.deictic is True


def test_plan_family_or_none_without_spec_is_unfiltered():
    family = plan_query("What does Aura cost?")
    assert family.action == "unfiltered"
    none = plan_query("What's the weather this weekend for painting outdoors?")
    assert none.action == "unfiltered"
    assert none.ref.kind == KIND_NONE
