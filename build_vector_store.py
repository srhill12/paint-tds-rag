"""
Build Vector Store from Benjamin Moore Technical Data Sheets
Runs once to index all cleaned TDS documents using Ollama embeddings + FAISS.
"""

import csv
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CLEANED_TEXTS_DIR,
    EMBEDDING_MODEL,
    MANIFEST_PATH,
    VECTOR_STORE_PATH,
)

def load_manifest(path: Path) -> list[dict[str, str]]:
    """Return manifest rows in file order."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            row for row in csv.DictReader(handle)
            if (row.get("filename") or "").strip()
        ]


def load_documents(directory: Path) -> tuple[list[Document], int, int]:
    """Load cleaned texts for current-index TDS rows; skip NON_TDS."""
    documents = []
    excluded = 0
    rows = load_manifest(MANIFEST_PATH)
    print(f"Found {len(rows)} manifest rows...")

    for row in rows:
        pdf_name = (row.get("filename") or "").strip()
        if (row.get("classification") or "").strip() == "NON_TDS":
            excluded += 1
            continue
        filepath = directory / (Path(pdf_name).stem + ".txt")
        if not filepath.exists():
            print(f"  Missing text for {pdf_name}")
            continue
        try:
            text = filepath.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                product_name = (row.get("product_name") or "").strip()
                sku = (row.get("sku") or "").strip()
                discontinued = (row.get("discontinued") or "false").strip().lower() == "true"
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": filepath.name,
                        "product": product_name or filepath.stem,
                        "filename": pdf_name,
                        "sku": sku,
                        "product_name": product_name,
                        "discontinued": discontinued,
                    }
                )
                documents.append(doc)
        except Exception as e:
            print(f"  Skipping {filepath.name}: {e}")

    return documents, len(documents), excluded

def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    return chunks

def build_vector_store(chunks: list[Document]) -> FAISS:
    """Embed chunks and build FAISS vector store."""
    print(f"Initializing Ollama embeddings ({EMBEDDING_MODEL})...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print("Building FAISS vector store...")
    print("This may take several minutes for 200+ documents...")

    # Ollama rejects a single embed request covering the full corpus.
    batch_size = 32
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        vectors.extend(embeddings.embed_documents(batch))
        done = min(start + batch_size, len(texts))
        print(f"  Embedded {done}/{len(texts)}")

    vector_store = FAISS.from_embeddings(
        list(zip(texts, vectors)),
        embeddings,
        metadatas=metadatas,
    )
    return vector_store

def main():
    print("=" * 60)
    print("Paint TDS RAG: Vector Store Builder")
    print("Using: Ollama nomic-embed-text + FAISS")
    print("Privacy: all processing stays local, no data leaves machine")
    print("=" * 60)
    print()

    # Load documents
    documents, indexed, excluded = load_documents(CLEANED_TEXTS_DIR)
    if not documents:
        print("No documents found. Check cleaned_texts/ directory.")
        return

    # Split into chunks
    chunks = split_documents(documents)
    print(f"Documents indexed: {indexed}")
    print(f"Documents excluded: {excluded}")
    print(f"Chunks created: {len(chunks)}")

    # Build vector store
    vector_store = build_vector_store(chunks)

    # Save to disk
    vector_store.save_local(VECTOR_STORE_PATH)
    print()
    print(f"Vector store saved to: {VECTOR_STORE_PATH}/")
    print("Ready to run: streamlit run app.py")

if __name__ == "__main__":
    main()
