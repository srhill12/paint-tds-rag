"""
Paint TDS Assistant: RAG-powered Q&A for Benjamin Moore Technical Data Sheets
Runs locally using Ollama (no data leaves your machine)
Built on: LangChain + FAISS + nomic-embed-text + Gemma 3
"""

import streamlit as st
from pathlib import Path
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from safety_router import SDS_SEARCH_URL, SAFETY_NOTICE, build_response

# ── Configuration ─────────────────────────────────────────────────────────────
VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL   = "nomic-embed-text"
CHAT_MODEL        = "gemma3:4b"
TOP_K_RESULTS     = 4

def load_resources():
    """Load the vector store and build the RAG chain. Cached after first load."""
    if not Path(VECTOR_STORE_PATH).exists():
        return None, None

    embeddings   = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )

    prompt = PromptTemplate.from_template("""You are a knowledgeable assistant \
for Benjamin Moore paint products. Use the provided Technical Data Sheet \
excerpts to answer the question accurately and concisely.

If the answer is not in the provided context, say so clearly rather than \
guessing. Always cite which product the information comes from.

Context from Technical Data Sheets:
{context}

Question: {question}

Answer:""")

    llm = ChatOllama(model=CHAT_MODEL, temperature=0.1)

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


def render_sources(source_docs) -> None:
    with st.expander("📄 Source Documents Retrieved", expanded=False):
        for i, doc in enumerate(source_docs, 1):
            product = doc.metadata.get("product", "Unknown")
            st.markdown(f"**Source {i}: {product}**")
            st.text(
                doc.page_content[:500] + "..."
                if len(doc.page_content) > 500
                else doc.page_content
            )
            st.divider()


def render_response(result: dict) -> None:
    if result["safety_routed"]:
        st.warning(SAFETY_NOTICE)
        st.markdown(
            f"[Search Benjamin Moore SDS documentation]({SDS_SEARCH_URL})"
        )
        st.success("Secondary context from Technical Data Sheet")
        st.write(result["answer"])
    else:
        st.success("Answer")
        st.write(result["answer"])
    render_sources(result["source_docs"])


def main() -> None:
    st.set_page_config(
        page_title="Paint TDS Assistant",
        page_icon="🎨",
        layout="centered"
    )

    st.title("🎨 Paint TDS Assistant")
    st.caption(
        "Ask questions about Benjamin Moore Technical Data Sheets: product "
        "specifications, application instructions, and technical requirements. "
        "All processing is local, and no data leaves this machine."
    )

    st.divider()

    st.info(
        "**Privacy Notice:** This assistant runs entirely on your local machine "
        "using Ollama. No queries or responses leave the machine at runtime. "
        "Local processing keeps the questions staff and customers ask on the "
        "machine. Those questions can reveal customer names, job sites, "
        "projects, and purchasing context, and no third-party data processor "
        "is involved. The only network step is `scripts/fetch_sources.py`.",
        icon="🔒"
    )

    cached_load = st.cache_resource(load_resources)

    if not Path(VECTOR_STORE_PATH).exists():
        st.error(
            "Vector store not found. Run `python build_vector_store.py` first "
            "to index the TDS documents."
        )
        st.stop()

    chain, retriever = cached_load()

    if chain is None:
        st.error("Failed to load RAG chain. Check that Ollama is running.")
        st.stop()

    st.subheader("Example Questions")
    example_questions = [
        "What is the drying time for Regal Select Interior?",
        "What surface preparation is required before applying exterior paint?",
        "What sheens are available for Regal Select Interior?",
        "What is the VOC content of Aura Interior paint?",
        "How many square feet does a gallon of Ben Interior cover?",
        "What is the minimum application temperature for exterior products?",
        "Is Aura Interior paint safe around pets?",
    ]

    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        if cols[i % 2].button(question, use_container_width=True):
            st.session_state.selected_question = question

    st.divider()
    st.subheader("Ask a Question")

    default_val = st.session_state.get("selected_question", "")
    user_question = st.text_input(
        "Type your question about Benjamin Moore products:",
        value=default_val,
        placeholder="e.g. What is the recoat time for Regal Select Exterior?"
    )

    if st.button("Ask", type="primary", use_container_width=True):
        if user_question.strip():
            with st.spinner("Searching TDS documents and generating answer..."):
                try:
                    result = run_query(user_question, chain, retriever)
                    render_response(result)
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    st.info("Make sure Ollama is running: `ollama serve`")
        else:
            st.warning("Please enter a question.")

    st.divider()
    st.caption(
        "**Paint TDS Assistant** | Built with LangChain + FAISS + Ollama | "
        "Corpus: Benjamin Moore Technical Data Sheets | "
        "Local inference: Gemma 3 4B | "
        "Embeddings: nomic-embed-text"
    )
    st.caption(
        "**Governance note:** The model is instructed to answer only from "
        "retrieved TDS excerpts and to say so when the answer is not in them. "
        "It can still be wrong. Check answers against the source documents "
        "before relying on them."
    )


# Streamlit runs this file as __main__. Eval harnesses can import run_query
# without launching the UI.
if __name__ == "__main__":
    main()
