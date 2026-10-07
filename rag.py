"""Shared retrieval and generation path used by the Streamlit app and eval."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    PROMPT_TEMPLATE,
    PROMPT_TEMPLATE_PRODUCT_AWARE,
    RETRIEVAL_MODE,
    SEED,
    TEMPERATURE,
    TOP_K,
    VECTOR_STORE_PATH,
)
from product_index import get_index
from safety_router import build_response


def load_vector_store():
    """Load FAISS, LLM, and the baseline RAG chain.

    Returns a namespace with chain, retriever, vector_store, and llm,
    or None if the store is missing.
    """
    if not Path(VECTOR_STORE_PATH).exists():
        return None

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOllama(model=CHAT_MODEL, temperature=TEMPERATURE, seed=SEED)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return SimpleNamespace(
        chain=chain,
        retriever=retriever,
        vector_store=vector_store,
        llm=llm,
    )


def _generate(question: str, docs, llm, prompt_template: str) -> str:
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = PromptTemplate.from_template(prompt_template)
    return (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )


def _retrieve_sku(vector_store, question: str, sku: str) -> list:
    target = sku.strip().upper()

    def matches(metadata: dict) -> bool:
        return (metadata.get("sku") or "").strip().upper() == target

    fetch_k = int(getattr(vector_store.index, "ntotal", 0) or 0) or 20
    return vector_store.similarity_search(
        question,
        k=TOP_K,
        filter=matches,
        fetch_k=fetch_k,
    )


def _product_label(ref) -> str:
    name = ref.product_name or ref.sku or ""
    sku = ref.sku or ""
    if name and sku:
        return f"Product: {name} ({sku})"
    if sku:
        return f"Product: ({sku})"
    return ""


def run_baseline(question: str, resources) -> dict:
    answer = resources.chain.invoke(question)
    source_docs = resources.retriever.invoke(question)
    return build_response(question, answer, source_docs)


def run_product_aware(question: str, resources) -> dict:
    index = get_index()
    plan = index.plan(question)
    product_label = ""
    source_docs: list = []

    if plan.action == "clarify":
        answer = plan.clarify_text
    elif plan.action == "filtered" and plan.ref.sku:
        source_docs = _retrieve_sku(resources.vector_store, question, plan.ref.sku)
        answer = _generate(
            question,
            source_docs,
            resources.llm,
            PROMPT_TEMPLATE_PRODUCT_AWARE,
        )
        product_label = _product_label(plan.ref)
    else:
        source_docs = resources.retriever.invoke(question)
        answer = _generate(
            question,
            source_docs,
            resources.llm,
            PROMPT_TEMPLATE_PRODUCT_AWARE,
        )

    result = build_response(question, answer, source_docs)
    result["product_label"] = product_label
    result["query_action"] = plan.action
    result["resolved_sku"] = plan.ref.sku
    result["resolved_kind"] = plan.ref.kind
    return result


def run_query(question: str, resources, retrieval_mode: str | None = None) -> dict:
    """Run retrieval and generation; return a response object for UI and eval."""
    mode = retrieval_mode or RETRIEVAL_MODE
    if resources is None:
        raise ValueError("RAG resources are not loaded")
    if mode == "baseline":
        result = run_baseline(question, resources)
        result.setdefault("product_label", "")
        result.setdefault("query_action", "unfiltered")
        result.setdefault("resolved_sku", None)
        result.setdefault("resolved_kind", None)
        return result
    return run_product_aware(question, resources)
