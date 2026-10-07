"""SKU-sheet context selection for product_aware single(sku) retrieval."""

from __future__ import annotations

from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from config import (
    CONTEXT_BUDGET_FRAC,
    NUM_CTX,
    PROMPT_TEMPLATE_PRODUCT_AWARE,
    SKU_FALLBACK_K,
)


def estimate_tokens(text: str) -> int:
    """Character/4 estimate. Not a model tokenizer."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def format_prompt(
    question: str,
    docs: Iterable[Any],
    prompt_template: str = PROMPT_TEMPLATE_PRODUCT_AWARE,
) -> str:
    context = "\n\n".join(
        d if isinstance(d, str) else (getattr(d, "page_content", None) or "")
        for d in docs
    )
    return PromptTemplate.from_template(prompt_template).format(
        context=context, question=question
    )


def context_budget_tokens(num_ctx: int = NUM_CTX, frac: float = CONTEXT_BUDGET_FRAC) -> int:
    return int(num_ctx * frac)


def choose_sku_docs(
    question: str,
    all_docs: list,
    similar_docs: list,
    prompt_template: str = PROMPT_TEMPLATE_PRODUCT_AWARE,
    num_ctx: int = NUM_CTX,
    budget_frac: float = CONTEXT_BUDGET_FRAC,
    fallback_k: int = SKU_FALLBACK_K,
) -> tuple[list, dict[str, Any]]:
    """Use all SKU chunks in given order unless the prompt would exceed the budget.

    If the full-sheet prompt estimate is above budget_frac * num_ctx, use the
    first fallback_k similar_docs instead. Does not search for other cutoffs.
    """
    budget = context_budget_tokens(num_ctx, budget_frac)
    tokens_all = estimate_tokens(format_prompt(question, all_docs, prompt_template))
    fallback = tokens_all > budget
    docs = list(similar_docs[:fallback_k]) if fallback else list(all_docs)
    tokens = estimate_tokens(format_prompt(question, docs, prompt_template))
    return docs, {
        "prompt_tokens_est": tokens,
        "prompt_tokens_full_sheet_est": tokens_all,
        "sku_context_fallback": fallback,
        "prompt_hit_cap": tokens > budget,
        "sku_chunk_count": len(docs),
        "sku_chunk_count_full": len(all_docs),
        "context_budget_tokens": budget,
    }


def prompt_stats_for_docs(
    question: str,
    docs: list,
    prompt_template: str,
    num_ctx: int = NUM_CTX,
    budget_frac: float = CONTEXT_BUDGET_FRAC,
) -> dict[str, Any]:
    budget = context_budget_tokens(num_ctx, budget_frac)
    tokens = estimate_tokens(format_prompt(question, docs, prompt_template))
    return {
        "prompt_tokens_est": tokens,
        "prompt_tokens_full_sheet_est": tokens,
        "sku_context_fallback": False,
        "prompt_hit_cap": tokens > budget,
        "sku_chunk_count": len(docs),
        "sku_chunk_count_full": len(docs),
        "context_budget_tokens": budget,
    }


def sku_docs_in_index_order(vector_store, sku: str) -> list:
    """Return every chunk for sku in FAISS index order (document order)."""
    target = sku.strip().upper()
    n = int(getattr(vector_store.index, "ntotal", 0) or 0)
    id_map = getattr(vector_store, "index_to_docstore_id", {}) or {}
    store = getattr(vector_store, "docstore", None)
    out: list = []
    for i in range(n):
        if isinstance(id_map, dict):
            doc_id = id_map.get(i)
        else:
            try:
                doc_id = id_map[i]
            except (KeyError, IndexError, TypeError):
                doc_id = None
        if not doc_id or store is None:
            continue
        doc = store.search(doc_id) if hasattr(store, "search") else None
        if not isinstance(doc, Document) and hasattr(store, "_dict"):
            doc = store._dict.get(doc_id)
        if not isinstance(doc, Document):
            continue
        if (doc.metadata.get("sku") or "").strip().upper() == target:
            out.append(doc)
    return out
