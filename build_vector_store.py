"""
Build Vector Store from Benjamin Moore Technical Data Sheets
Runs once to index all cleaned TDS documents using Ollama embeddings + FAISS.
"""

import os
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── Configuration ─────────────────────────────────────────────────────────────
CLEANED_TEXTS_DIR = Path("cleaned_texts")
VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL   = "nomic-embed-text"
CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 50

def load_documents(directory: Path) -> list[Document]:
    """Load all .txt TDS files from the cleaned_texts directory."""
    documents = []
    files = list(directory.glob("*.txt"))
    print(f"Found {len(files)} TDS documents to index...")

    for filepath in files:
        try:
            text = filepath.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": filepath.name,
                        "product": filepath.stem
                    }
                )
                documents.append(doc)
        except Exception as e:
            print(f"  Skipping {filepath.name}: {e}")

    print(f"Loaded {len(documents)} documents successfully.")
    return documents

def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks

def build_vector_store(chunks: list[Document]) -> FAISS:
    """Embed chunks and build FAISS vector store."""
    print(f"Initializing Ollama embeddings ({EMBEDDING_MODEL})...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print("Building FAISS vector store...")
    print("This may take several minutes for 200+ documents...")

    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def main():
    print("=" * 60)
    print("Paint SDS RAG — Vector Store Builder")
    print("Using: Ollama nomic-embed-text + FAISS")
    print("Privacy: all processing stays local, no data leaves machine")
    print("=" * 60)
    print()

    # Load documents
    documents = load_documents(CLEANED_TEXTS_DIR)
    if not documents:
        print("No documents found. Check cleaned_texts/ directory.")
        return

    # Split into chunks
    chunks = split_documents(documents)

    # Build vector store
    vector_store = build_vector_store(chunks)

    # Save to disk
    vector_store.save_local(VECTOR_STORE_PATH)
    print()
    print(f"Vector store saved to: {VECTOR_STORE_PATH}/")
    print("Ready to run: streamlit run app.py")

if __name__ == "__main__":
    main()