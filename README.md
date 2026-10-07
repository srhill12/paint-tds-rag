# Paint TDS Assistant: Local RAG System

A fully local Retrieval-Augmented Generation (RAG) system that enables 
natural language queries against a corpus of Benjamin Moore Technical 
Data Sheets (TDS). It was built to solve a real operational problem at 
Hill Country Paints without sending proprietary product data to external 
servers.

---

## The Business Problem

Paint store staff regularly need to answer customer questions about 
product specifications such as drying times, VOC content, surface 
preparation requirements, application temperatures, and coverage rates. 
Looking up this information manually across hundreds of Technical Data 
Sheets is time-consuming and error-prone.

The obvious solution is a cloud-based AI assistant, but that creates a 
data privacy problem: proprietary product knowledge and customer queries 
would be transmitted to external servers. For a business relationship 
with a brand like Benjamin Moore, that's not acceptable.

**The governance-driven solution:** A fully local RAG system where all 
processing stays on-premise. No data leaves the machine.

---

## Architecture

User Question (natural language)
↓
nomic-embed-text (Ollama): converts question to vector embedding
↓
FAISS Vector Store: semantic similarity search across 8,655 chunks
↓
Top 4 relevant TDS excerpts retrieved
↓
Gemma 3 4B (Ollama): generates answer grounded in retrieved context
↓
Answer + Source Citations displayed in Streamlit UI

**Why this is RAG, not just a chatbot:**
The model is prompted to answer from retrieved excerpts of Benjamin Moore 
TDS documents rather than from its training data, and to say so when the 
answer is not in those excerpts. This is an instruction to the model, not 
a guarantee. The model can still misread or go beyond the retrieved text, 
which is why every answer shows its sources.

---

## Corpus

- **348 Technical Data Sheets** extracted from Benjamin Moore PDFs
- **8,655 text chunks** indexed in FAISS vector store
- Coverage includes: interior paints, exterior paints, primers, 
  specialty coatings, wood finishes, industrial products
- PDF extraction pipeline built with PyMuPDF (fitz) with MD5 
  hash-based deduplication to avoid reprocessing
- Source PDFs are not stored in this repository. They are listed in 
  `sources/manifest.csv` and downloaded with `scripts/fetch_sources.py`.

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
| Runtime | Fully local with no external API calls |

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
pip install -r requirements.txt
```

### 3. Fetch the source TDS PDFs

Fill in `sources/manifest.csv` with one row per product 
(`product_name`, `sku`, `interior_exterior`, `tds_url`), then run:

```bash
python scripts/fetch_sources.py
```

PDFs are saved to `pdfs/`. Files that already exist are skipped, so the 
script is safe to re-run after adding rows.

### 4. Extract text from the PDFs

```bash
python extract_text_from_pdfs.py
```

This writes one `.txt` file per PDF to `cleaned_texts/`.

### 5. Build the vector store

Run once to index all TDS documents:

```bash
python build_vector_store.py
```

On the original 348-document corpus this produced 8,655 chunks and took 
3-5 minutes on first run.

### 6. Launch the assistant

```bash
streamlit run app.py
```

---

## Example Queries

- *"What is the VOC content of Aura Interior paint?"*
- *"What surface preparation is required before applying exterior paint?"*
- *"What sheens are available for Regal Select Interior?"*
- *"What is the minimum application temperature for exterior products?"*
- *"How many square feet does a gallon of Ben Interior cover?"*

---

## Governance Design Decisions

This project was built with explicit governance constraints that shaped 
every technical decision:

**1. Local-only processing**
All embeddings, retrieval, and inference run on-device via Ollama. 
No queries or product data are transmitted to external APIs. This 
protects proprietary product knowledge and customer interaction data. 
The only network step is the one-time download of public TDS PDFs by 
`scripts/fetch_sources.py`.

**2. Retrieval-grounded responses**
The LLM is instructed to answer only from retrieved TDS excerpts. 
The prompt explicitly tells the model to acknowledge when information 
is not available rather than generating plausible-sounding but 
unverified specifications. Staff rely on these answers for product 
specifications such as VOC content, application temperatures, and 
recoat times, where an invented value leads to a failed job or a wrong 
recommendation to a customer. Technical Data Sheets describe product 
performance and application. They are not a source for safety or 
hazard questions, which belong with the product's Safety Data Sheet.

**3. Source transparency**
Every answer displays the source TDS excerpts used for retrieval, so 
staff can verify the model's answer against the original document. The 
design supports human verification. No human approves outputs before 
use, so checking the sources is the responsibility of the person 
relying on the answer.

**4. Scope restriction**
The system can only query indexed TDS documents. It has no access to 
pricing, customer data, inventory, or any other system. Scope 
restriction is a primary governance control.

**5. Honest uncertainty**
The prompt instructs the model to say so when the retrieved documents 
don't contain the answer. A system that says "I don't know" is safer 
than one that confidently answers incorrectly. How consistently the 
model follows this instruction should be confirmed by evaluation.

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
  globally. A future version could add product-specific filtering to 
  constrain retrieval to the correct SKU before semantic search.

---

## Origin

This project originated during my tenure as Director of AI Governance & 
Enablement at Hill Country Paints, where I identified the need for staff 
to query product specification data quickly during customer interactions 
without sending proprietary data to cloud services. The initial prototype 
was built during that tenure and used fine-tuned GPT-2. This version is a 
later rebuild using a modern RAG architecture with local inference via 
Ollama.

---

## Author

**Steven Hill**
AI Governance Professional | AIGP | ISO/IEC 42001 Lead Auditor
[LinkedIn](https://linkedin.com/in/stevenrhill) |
[GitHub](https://github.com/srhill12)
