"""
Paint SDS Assistant — RAG-powered Q&A for Benjamin Moore Technical Data Sheets
Local deployment using Ollama (no data leaves your machine)
Built on: LangChain + FAISS + nomic-embed-text + Gemma 3
"""

import streamlit as st
from pathlib import Path
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Configuration ─────────────────────────────────────────────────────────────
VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL   = "nomic-embed-text"
CHAT_MODEL        = "gemma3:4b"
TOP_K_RESULTS     = 4

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Paint SDS Assistant",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Paint SDS Assistant")
st.caption(
    "Ask questions about Benjamin Moore product specifications, "
    "safety data, application instructions, and technical requirements. "
    "All processing is local — no data leaves this machine."
)

st.divider()

# ── Privacy Notice ────────────────────────────────────────────────────────────
st.info(
    "**Privacy Notice:** This assistant runs entirely on your local machine "
    "using Ollama. No queries, product data, or responses are sent to any "
    "external server. This was a deliberate design decision to protect "
    "proprietary product information.",
    icon="🔒"
)

# ── Load Vector Store ─────────────────────────────────────────────────────────
@st.cache_resource
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

# ── Check for vector store ────────────────────────────────────────────────────
if not Path(VECTOR_STORE_PATH).exists():
    st.error(
        "Vector store not found. Run `python build_vector_store.py` first "
        "to index the TDS documents."
    )
    st.stop()

chain, retriever = load_resources()

if chain is None:
    st.error("Failed to load RAG chain. Check that Ollama is running.")
    st.stop()

# ── Example Questions ─────────────────────────────────────────────────────────
st.subheader("Example Questions")
example_questions = [
    "What is the drying time for Regal Select Interior?",
    "What surface preparation is required before applying exterior paint?",
    "Is this product safe to use in enclosed spaces?",
    "What is the VOC content of Aura Interior paint?",
    "How many square feet does a gallon of Ben Interior cover?",
    "What is the minimum application temperature for exterior products?",
]

cols = st.columns(2)
for i, question in enumerate(example_questions):
    if cols[i % 2].button(question, use_container_width=True):
        st.session_state.selected_question = question

# ── Chat Interface ────────────────────────────────────────────────────────────
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
                # Get answer
                answer = chain.invoke(user_question)

                # Get source documents separately
                source_docs = retriever.invoke(user_question)

                st.success("Answer")
                st.write(answer)

                # Show source documents
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

            except Exception as e:
                st.error(f"Error generating answer: {e}")
                st.info("Make sure Ollama is running: `ollama serve`")
    else:
        st.warning("Please enter a question.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Paint SDS Assistant** | Built with LangChain + FAISS + Ollama | "
    "Corpus: Benjamin Moore Technical Data Sheets | "
    "Local inference: Gemma 3 4B | "
    "Embeddings: nomic-embed-text"
)
st.caption(
    "**Governance note:** This system retrieves answers only from indexed "
    "TDS documents. It will not hallucinate product specifications — if "
    "the answer is not in the corpus, it says so. Human review recommended "
    "before acting on safety-critical information."
)