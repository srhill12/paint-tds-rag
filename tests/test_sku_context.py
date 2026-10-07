"""SKU full-sheet context selection. No Ollama."""

from types import SimpleNamespace

from langchain_core.documents import Document

from config import CONTEXT_BUDGET_FRAC, NUM_CTX, SKU_FALLBACK_K
from sku_context import (
    choose_sku_docs,
    context_budget_tokens,
    estimate_tokens,
    sku_docs_in_index_order,
)


class _Store:
    def __init__(self, docs: dict[str, Document]):
        self._dict = docs

    def search(self, doc_id: str):
        return self._dict.get(doc_id, f"ID {doc_id} not found")


def _docs(n: int, size: int, prefix: str = "c") -> list[Document]:
    return [
        Document(page_content=f"{prefix}{i} " + ("x" * size), metadata={"i": i})
        for i in range(n)
    ]


def test_num_ctx_and_fallback_k_are_fixed():
    assert NUM_CTX == 8192
    assert CONTEXT_BUDGET_FRAC == 0.8
    assert SKU_FALLBACK_K == 12
    assert context_budget_tokens() == int(8192 * 0.8)


def test_under_budget_keeps_all_docs_in_order():
    all_docs = _docs(5, 20)
    similar = list(reversed(all_docs))
    chosen, stats = choose_sku_docs("what is voc?", all_docs, similar)
    assert [d.metadata["i"] for d in chosen] == [0, 1, 2, 3, 4]
    assert stats["sku_context_fallback"] is False
    assert stats["sku_chunk_count"] == 5
    assert stats["sku_chunk_count_full"] == 5
    assert stats["prompt_hit_cap"] is False


def test_over_budget_falls_back_to_top_similar():
    all_docs = _docs(20, 2000)
    similar = list(reversed(all_docs))
    chosen, stats = choose_sku_docs("what is voc?", all_docs, similar)
    assert stats["sku_context_fallback"] is True
    assert len(chosen) == 12
    assert [d.metadata["i"] for d in chosen] == [19 - i for i in range(12)]
    assert stats["prompt_tokens_full_sheet_est"] > stats["context_budget_tokens"]


def test_estimate_tokens_is_char_quarter():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 8) == 2


def test_fallback_still_over_budget_flags_cap():
    all_docs = _docs(20, 5000)
    chosen, stats = choose_sku_docs(
        "q",
        all_docs,
        all_docs,
        num_ctx=128,
        budget_frac=0.8,
        fallback_k=12,
    )
    assert stats["sku_context_fallback"] is True
    assert stats["prompt_hit_cap"] is True
    assert len(chosen) == 12


def test_index_order_keeps_sku_chunks_in_faiss_order():
    a0 = Document(page_content="a0", metadata={"sku": "N549"})
    b0 = Document(page_content="b0", metadata={"sku": "N539"})
    a1 = Document(page_content="a1", metadata={"sku": "n549"})
    store = SimpleNamespace(
        index=SimpleNamespace(ntotal=3),
        index_to_docstore_id={0: "id0", 1: "id1", 2: "id2"},
        docstore=_Store({"id0": a0, "id1": b0, "id2": a1}),
    )
    docs = sku_docs_in_index_order(store, "N549")
    assert [d.page_content for d in docs] == ["a0", "a1"]
