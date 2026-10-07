"""Shared runtime settings for the app and the evaluation harness."""

from pathlib import Path

# ChatOllama previously used temperature=0.1 in app.py. Eval and the app
# now use 0 with a fixed seed so runs are reproducible.
CHAT_MODEL = "gemma3:4b"
EMBEDDING_MODEL = "nomic-embed-text"
TEMPERATURE = 0.0
PRIOR_TEMPERATURE = 0.1
SEED = 42
TOP_K = 4

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

VECTOR_STORE_PATH = "vector_store"
MANIFEST_PATH = Path("sources") / "manifest.csv"
CLEANED_TEXTS_DIR = Path("cleaned_texts")
FAMILIES_PATH = Path("sources") / "families.csv"

TESTSET_VERSION = "v1"
TESTSET_PATH = Path("eval") / "testset_v1.jsonl"
RESULTS_DIR = Path("eval") / "results"

# "baseline" is the original unfiltered similarity search. The app uses
# product_aware. Eval --mode selects which path to run.
RETRIEVAL_MODE = "product_aware"
RETRIEVAL_MODES = ("baseline", "product_aware")

# Router v2 extra terms. The app and product_aware eval default to on.
# Eval --router-v2 off still disables v2 for a comparison run.
ROUTER_V2_ENABLED = True

PROMPT_TEMPLATE = """You are a knowledgeable assistant \
for Benjamin Moore paint products. Use the provided Technical Data Sheet \
excerpts to answer the question accurately and concisely.

If the answer is not in the provided context, say so clearly rather than \
guessing. Always cite which product the information comes from.

Context from Technical Data Sheets:
{context}

Question: {question}

Answer:"""

PROMPT_TEMPLATE_PRODUCT_AWARE = """You are a knowledgeable assistant \
for Benjamin Moore paint products. Use the provided Technical Data Sheet \
excerpts to answer the question accurately and concisely.

If the answer is not in the provided context, say so clearly rather than \
guessing. Every specification value you report must name the product and \
SKU it comes from. Never present a value from one product as applying to \
another product.

Context from Technical Data Sheets:
{context}

Question: {question}

Answer:"""
