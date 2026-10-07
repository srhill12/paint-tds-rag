"""Shared retrieval and generation path used by the Streamlit app and eval."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    PROMPT_TEMPLATE,
    SEED,
    TEMPERATURE,
    TOP_K,
    VECTOR_STORE_PATH,
)
from safety_router import build_response


def load_vector_store():
    """Load FAISS and build the RAG chain. Returns (chain, retriever) or (None, None)."""
    if not Path(VECTOR_STORE_PATH).exists():
        return None, None

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
    return chain, retriever


def run_query(question: str, chain, retriever) -> dict:
    """Run retrieval and generation; return a response object for UI and eval."""
    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)
    return build_response(question, answer, source_docs)
