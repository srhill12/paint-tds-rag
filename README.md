# Paint SDS Assistant — Local RAG System

A fully local Retrieval-Augmented Generation (RAG) system that enables 
natural language queries against a corpus of Benjamin Moore Technical 
Data Sheets — built to solve a real operational problem at Hill Country 
Paints without sending proprietary product data to external servers.

---

## The Business Problem

Paint store staff regularly need to answer customer questions about 
product specifications — drying times, VOC content, surface preparation 
requirements, application temperatures, coverage rates. Looking up this 
information manually across hundreds of Technical Data Sheets is 
time-consuming and error-prone.

The obvious solution — a cloud-based AI assistant — creates a data 
privacy problem: proprietary product knowledge and customer queries 
would be transmitted to external servers. For a business relationship 
with a brand like Benjamin Moore, that's not acceptable.

**The governance-driven solution:** A fully local RAG system where all 
processing stays on-premise. No data leaves the machine.

---

## Architecture

User Question (natural language)
↓
nomic-embed-text (Ollama) — converts question to vector embedding
↓
FAISS Vector Store — semantic similarity search across 8,655 chunks
↓
Top 4 relevant TDS excerpts retrieved
↓
Gemma 3 4B (Ollama) — generates answer grounded in retrieved context
↓
Answer + Source Citations displayed in Streamlit UI

**Why this is RAG, not just a chatbot:**
The model never answers from training data alone. Every response is 
grounded in retrieved excerpts from actual Benjamin Moore TDS documents. 
If the answer isn't in the corpus, the system says so rather than 
hallucinating product specifications.

---

## Corpus

- **348 Technical Data Sheets** extracted from Benjamin Moore PDFs
- **8,655 text chunks** indexed in FAISS vector store
- Coverage includes: interior paints, exterior paints, primers, 
  specialty coatings, wood finishes, industrial products
- PDF extraction pipeline built with PyMuPDF (fitz) with MD5 
  hash-based deduplication to avoid reprocessing

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Embeddings | nomic-embed-text (Ollama) |
| Vector store | FAISS (Facebook AI Similarity Search) |
| LLM | Gemma 3 4B (Ollama) |
| Orchestration | LangChain |
| Interface | Streamlit |
| PDF extraction | PyMuPDF (fitz) |
| Deployment | Fully local — no external API calls |

---

## Setup

### Prerequisites
- [Ollama](https://ollama.ai) installed and running
- Python 3.10+

### 1. Pull required models

```bash
ollama pull nomic-embed-text
ollama pull gemma3:4b
```

### 2. Install dependencies

```bash
pip install langchain langchain-community langchain-ollama langchain-core \
            langchain-text-splitters faiss-cpu streamlit pymupdf
```

### 3. Build the vector store

Run once to index all TDS documents:

```bash
python build_vector_store.py
```

This embeds 348 documents into 8,655 chunks. Takes 3-5 minutes on first run.

### 4. Launch the assistant

```bash
streamlit run app.py
```

---

## Example Queries

- *"What is the VOC content of Aura Interior paint?"*
- *"What surface preparation is required before applying exterior paint?"*
- *"Is this product safe to use in enclosed spaces?"*
- *"What is the minimum application temperature for exterior products?"*
- *"How many square feet does a gallon of Ben Interior cover?"*

---

## Governance Design Decisions

This project was built with explicit governance constraints that shaped 
every technical decision:

**1. Local-only deployment**
All embeddings, retrieval, and inference run on-device via Ollama. 
No queries or product data are transmitted to external APIs. This 
protects proprietary product knowledge and customer interaction data.

**2. Retrieval-grounded responses**
The LLM is constrained to answer only from retrieved TDS excerpts. 
The prompt explicitly instructs the model to acknowledge when 
information is not available rather than generating plausible-sounding 
but unverified specifications. In safety-critical contexts (VOC content, 
application temperatures, enclosed space warnings), hallucinated 
answers cause real harm.

**3. Source transparency**
Every answer displays the source TDS documents used for retrieval. 
Staff can verify the model's answer against the original document. 
This is human-in-the-loop design — the assistant supports decisions, 
it doesn't make them.

**4. Scope restriction**
The system can only query indexed TDS documents. It has no access to 
pricing, customer data, inventory, or any other system. Scope 
restriction is a primary governance control.

**5. Honest uncertainty**
When the retrieved documents don't contain the answer, the system 
says so explicitly. A system that says "I don't know" is safer than 
one that confidently answers incorrectly.

---

## Known Limitations

- **Semantic matching on SKU numbers:** Queries using product names 
  ("Regal Select Interior") may retrieve semantically similar products 
  rather than exact matches. Queries using SKU codes (e.g., "N549") 
  improve precision.
- **Chunk boundary issues:** Some TDS specifications span page breaks 
  in ways that chunk splitting may separate. Increasing chunk size or 
  overlap would improve recall on multi-value specifications.
- **Single-document retrieval:** The system retrieves the top 4 chunks 
  globally. A production version would implement product-specific 
  filtering to constrain retrieval to the correct SKU before semantic 
  search.

---

## Origin

This project originated during my tenure as Director of Employee 
Development at Hill Country Paints, where I identified the need for 
staff to query product safety and specification data quickly during 
customer interactions — without sending proprietary data to cloud 
services. The initial prototype used fine-tuned GPT-2; this version 
rebuilds the system using a modern RAG architecture with local 
inference via Ollama.

---

## Author

**Steven Hill**
AI Ethics & Policy Professional | Purdue University MSAI
[LinkedIn](https://linkedin.com/in/stevenrhill) |
[GitHub](https://github.com/srhill12)