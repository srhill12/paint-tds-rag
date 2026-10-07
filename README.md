# Paint TDS Assistant: Local RAG System

A fully local Retrieval-Augmented Generation (RAG) system that enables 
natural language queries against a corpus of Benjamin Moore Technical 
Data Sheets (TDS). It was built to solve a real operational problem at 
Hill Country Paints. Local processing keeps the questions staff and 
customers ask on the machine. Those questions can reveal customer names, 
job sites, projects, and purchasing context, and no third-party data 
processor is involved.

---

## The Business Problem

Paint store staff regularly need to answer customer questions about 
product specifications such as drying times, VOC content, surface 
preparation requirements, application temperatures, and coverage rates. 
Looking up this information manually across hundreds of Technical Data 
Sheets is time-consuming and error-prone.

The obvious solution is a cloud-based AI assistant, but that would send 
the questions staff and customers ask to a third-party data processor. 
Those questions can reveal customer names, job sites, projects, and 
purchasing context.

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

The corpus is a snapshot of the Benjamin Moore US TDS index retrieved on 2026-10-07. Non-TDS documents and sheets no longer listed by the manufacturer are excluded. `sources/manifest.csv` records an md5 per file so `scripts/fetch_sources.py` can detect later revisions.

- **352 Technical Data Sheets** indexed (3 non-TDS documents excluded)
- **7,031 text chunks** indexed in FAISS vector store
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
- *"Is Aura Interior paint safe around pets?"* (routes to the SDS)

---

## Governance Design Decisions

This project was built with explicit governance constraints that shaped 
every technical decision:

**1. Local-only processing**
All embeddings, retrieval, and inference run on-device via Ollama. 
No queries or responses leave the machine at runtime. Local processing 
keeps the questions staff and customers ask on the machine. Those 
questions can reveal customer names, job sites, projects, and purchasing 
context, and no third-party data processor is involved. The only network 
step is the one-time download of public TDS PDFs by 
`scripts/fetch_sources.py`.

**2. Retrieval-grounded responses**
The LLM is instructed to answer only from retrieved TDS excerpts. 
The prompt explicitly tells the model to acknowledge when information 
is not available rather than generating plausible-sounding but 
unverified specifications. In product-aware mode the prompt also 
requires every specification value to name the product and SKU it 
comes from, and forbids presenting one product's value as another's. 
Staff rely on these answers for product specifications such as VOC 
content, application temperatures, and recoat times, where an invented 
value leads to a failed job or a wrong recommendation to a customer. 
Technical Data Sheets describe product performance and application. 
They are not a source for safety or hazard questions, which belong 
with the product's Safety Data Sheet.

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
than one that confidently answers incorrectly. Evaluation measures 
whether that instruction is followed; it is not assumed. See 
Evaluation for the unanswerable result.

**6. Safety question routing**
A rule-based router flags questions that match whole words or phrases 
such as hazard, PPE, flammable, pets, and safe to use. When it 
triggers, the assistant first shows a fixed notice that TDS safety 
content is partial and varies by sheet, while the SDS is the regulated, 
complete, authoritative source for the specific product and base. It then links to Benjamin Moore's documentation 
search rather than naming a single SDS file. SDSs carry the OSHA 
hazard classifications, and they are issued per base and colorant, so 
guessing one sheet would be wrong. Any TDS-derived answer is shown 
only after that notice, labeled as secondary context.

The keyword rule misses paraphrased safety questions (for example 
"well-ventilated" does not match "ventilation"). It can also 
over-trigger on words like "safety" in product names if that term is 
added to the list. Both cases are documented in 
`tests/test_safety_router.py`.

Router v2 is enabled for the app and for product_aware eval. The 
shipped v2 list is exactly: smell, odor, odour, newborn, infant, 
baby, nursery, headache, dizzy, nausea, breathing, breathe, asthma, 
allergic, allergy.

v2 is on because a false positive adds an SDS notice but still 
returns the TDS answer, whereas a false negative can return safety 
guidance from the wrong document. Measured counts, including holdout 
false positives and hold-saf-05, are in Evaluation. The holdout was 
drafted in the same session as the v2 list, so it is a check on false 
positives and obvious misses, not proof of real-world recall (see 
`eval/README.md`).

---

## Evaluation

Staff cannot treat a matching number as a correct specification if it 
came from another product's sheet. On the frozen 62-question test set, 
baseline source-correct was 0/6 among answerable rows that reported a 
value: every attested number came from the wrong SKU ([F-006](docs/findings/FINDINGS.md)). 
After product-aware retrieval, router v2, and full-sheet context for an 
identified SKU, source-correct is 16/16 and answerable passed is 16/16. 
The prompt still does not make the model decline reliably: unanswerable 
passed is 4/8 on Run 3.

The 62-question test set (`eval/testset_v1.jsonl`) has 16 answerable, 
8 unanswerable, 16 confusion, 4 family, 5 underspecified, 8 safety 
(5 keyword, 3 paraphrased), and 5 safety_negative items. The 12-question 
router holdout (`eval/testset_router_holdout.jsonl`) has 8 paraphrased 
safety items and 4 safety_negative items. Tables are generated from 
the run summaries by `python -m eval.report`.

Test set (`n=62`). Source runs: 20261007T204311Z_67c76a9, 20261007T204608Z_67c76a9, 20261007T230030Z_5b07db3, 20261007T230816Z_20e3fd4.

| Metric                                        | Baseline | Run 1 product-aware | Run 2 router v2 | Run 3 full-sheet |
| --------------------------------------------- | -------- | ------------------- | --------------- | ---------------- |
| answerable passed                             | 2/16     | 11/16               | 11/16           | 16/16            |
| answerable source-correct                     | 0/6      | 12/12               | 12/12           | 16/16            |
| confusion wrong-product rate                  | 10/16    | 0/16                | 0/16            | 0/16             |
| family and underspecified asked which product | 0/4; 0/5 | 4/4; 5/5            | 4/4; 5/5        | 4/4; 5/5         |
| safety routed (keyword)                       | 5/5      | 5/5                 | 5/5             | 5/5              |
| safety routed (paraphrased)                   | 0/3      | 0/3                 | 3/3             | 3/3              |
| safety_negative not routed                    | 5/5      | 5/5                 | 5/5             | 5/5              |
| unanswerable passed                           | 5/8      | 5/8                 | 5/8             | 4/8              |
| hedged count                                  | 6/62     | 1/62                | 1/62            | 2/62             |
| unsupported value rate                        | 0/23     | 0/33                | 0/33            | 0/41             |
| attribution rate                              | 14/23    | 32/33               | 32/33           | 40/41            |

Router holdout (`n=12`). Source runs: 20261007T225628Z_5b07db3, 20261007T230010Z_5b07db3.

| Metric                     | Router v1 | Router v2 |
| -------------------------- | --------- | --------- |
| safety routed              | 0/8       | 7/8       |
| safety_negative not routed | 4/4       | 0/4       |

**What changed.** Run 1 (product-aware retrieval and clarification) 
filters to a resolved SKU before similarity search, and asks which 
product when a family name or an unspecified reference is paired with 
a specification question. That is what moved source-correct from 0/6 
to 12/12, confusion wrong-product from 10/16 to 0/16, and family / 
underspecified "asked which product" from 0/4; 0/5 to 4/4; 5/5. 
Run 2 (router v2) adds whole-word terms for smell, odor, infant, and 
related paraphrases. On the test set, paraphrased safety routed went 
from 0/3 to 3/3 and safety_negative stayed 5/5 not routed. On the 
holdout, safety routed went from 0/8 to 7/8 and safety_negative not 
routed fell from 4/4 to 0/4. Router v2 is enabled by default despite 
the holdout false positives; see Governance Decision 6 for the 
tradeoff. Run 3 (full-sheet retrieval for an 
identified product) passes every chunk for that SKU in document order, 
or the top 12 by similarity if the estimated prompt would exceed 80% 
of `num_ctx`. Answerable passed moved from 11/16 to 16/16. 
Run 3 also introduced one regression: unanswerable passed fell from 
5/8 to 4/8. With the full sheet in context, the model answered the 
N549 half of a Behr comparison question (una-08) instead of only 
declining. More context improved specification lookup and made 
declines less reliable.

**Why source-correctness is the primary metric.** Baseline answerable 
passed 2/16, and numeric_match was 4/16, but source-correct was 0/6. 
The numbers that matched expected values were attested by other 
products' sheets. Numeric match alone overstates correctness. See 
[F-006](docs/findings/FINDINGS.md).

**Remaining failures (Run 3).** una-04 and una-07 still fail 
`no_decline`: the model answers a color preference and a weekend 
weather question instead of only saying the TDS does not contain 
that. una-08 is a decline of the Behr comparison that also reports 
N549 VOC, so it fails the hedged rule (decline plus a numeric 
specification). Router v2 false-positives on all four holdout 
safety_negative items (odor, smell, baby, nursery). hold-saf-05 is 
missed because "nauseous" is not in the shipped list. [F-008](docs/findings/FINDINGS.md) 
is a family question without spec intent ("Does ben Interior 
advertise a low smell formula?") answered about Super Hide.

**The system does not reliably say when it does not know.** The prompt 
tells the model to say so when the answer is not in the excerpts. 
Run 3 unanswerable passed is 4/8. That is the measured result, not a 
behavior to assume in use.

**Reproducibility.** Chat generation uses temperature 0, seed 42, and 
`num_ctx` 8192 (recorded on the Run 3 summary). Across reruns, 
retrieved SKUs matched. One generated answer (ans-05) differed in 
wording with no change in score.

**Scoring integrity.** All runs under `eval/results/` were rescored 
with scorer version `2026-10-07-decline-consolidate` except the 
original frozen baseline `20261007T201943Z_c01b8dc`, which is kept 
unmodified as the first recorded run and was scored with an earlier 
scorer version. Rule changes are listed in 
[eval/SCORER_CHANGES.md](eval/SCORER_CHANGES.md).

**Method notes.** The author verified the test set against source PDFs 
before freezing (git tag `eval-testset-v1`). There is a single 
annotator. Public logs omit retrieved source text; see 
[eval/README.md](eval/README.md). The holdout questions were drafted 
in the same session as the v2 term list, so they are not fully 
independent of it. Holdout results check false positives and obvious 
misses; they are not proof of real-world recall. 
Category sizes are small (for example 8 unanswerable and 3 paraphrased 
safety items), so differences of one item should not be read as 
meaningful.

**How to reproduce**

Complete the Setup steps first (fetch sources, extract, build the 
vector store, Ollama running with the models listed).

```bash
python -m eval.run --mode baseline --testset eval/testset_v1.jsonl --router-v2 off
python -m eval.run --mode product_aware --testset eval/testset_v1.jsonl --router-v2 on
python -m eval.run --mode product_aware --testset eval/testset_router_holdout.jsonl --router-v2 off
python -m eval.run --mode product_aware --testset eval/testset_router_holdout.jsonl --router-v2 on
python -m eval.report
python -m eval.compare eval/results/20261007T204311Z_67c76a9 eval/results/20261007T204608Z_67c76a9
python -m eval.compare eval/results/20261007T204608Z_67c76a9 eval/results/20261007T230030Z_5b07db3
python -m eval.compare eval/results/20261007T230030Z_5b07db3 eval/results/20261007T230816Z_20e3fd4
```

New runs write a new directory under `eval/results/`. Do not modify 
`eval/results/20261007T201943Z_c01b8dc`.

---

## System card

**Intended use.** Staff lookup of technical specifications (VOC, 
coverage, recoat time, application temperature, sheen, solids) for a 
named Benjamin Moore product in the local TDS corpus. The person using 
the answer is expected to check the cited excerpts.

**Out of scope.** Safety or health decisions. Regulatory compliance 
determinations. Product comparisons across brands. Any question that 
requires the Safety Data Sheet. Pricing, inventory, and customer data 
are not in the corpus.

**Data source.** Corpus snapshot date 2026-10-07 (`sources/manifest.csv`). 
352 TDS documents are classified for indexing; 3 non-TDS documents are 
excluded. Sheets no longer listed by the manufacturer are also out of 
scope. `scripts/fetch_sources.py` downloads public PDFs listed in the 
manifest.

**Evaluation summary.** Measured results are in Evaluation. Headline: 
baseline source-correct 0/6 to Run 3 source-correct 16/16 on named-product 
answerable questions; unanswerable passed 4/8; router v2 holdout safety 
7/8 with safety_negative 0/4 not routed.

**Known limitations.** See Known Limitations and 
[docs/findings/FINDINGS.md](docs/findings/FINDINGS.md). Named-product 
specification lookup is in scope; unspecified or family-level questions 
must be clarified or they can be answered from the wrong sheet. 
Unanswerable questions are not reliably declined. Router v2 still misses 
hold-saf-05 and false-routes the holdout negatives.

**Safety routing control.** A whole-word keyword router (v1 terms plus 
the v2 list in Governance Decision 6) shows a fixed SDS notice before 
any TDS text. TDS content is secondary context. The SDS is the 
authoritative source for the specific product and base. Routing is 
not a substitute for reading the SDS.

---

## Known Limitations

Observed failures are recorded in 
[docs/findings/FINDINGS.md](docs/findings/FINDINGS.md). The counts 
below are from the Evaluation tables.

- **Wrong-product numbers at baseline (F-002, F-006):** On testset v1, 
  a SKU query such as "what is the voc content of n549?" did not 
  retrieve N549. Answerable source-correct was 0/6: every attested 
  value came from another SKU's chunk. Numeric match was 4/16. 
  Product-aware retrieval (Run 1) filters to the resolved SKU; 
  source-correct is 12/12 on Run 1 and 16/16 on Run 3.
- **Unspecified and family questions (F-001, F-003, F-004):** Baseline 
  family asked which product 0/4 and underspecified 0/5. Run 1 
  clarification is 4/4 and 5/5 when spec intent is present. Without 
  spec intent, retrieval stays unfiltered ([F-008](docs/findings/FINDINGS.md)).
- **Paraphrased safety (F-005):** Keyword safety routed 5/5 on every 
  v1 run. Paraphrased safety was 0/3 until router v2 (3/3 on the test 
  set). Holdout safety is 7/8 with v2; hold-saf-05 is missed 
  ("nauseous" is not in the shipped list). All four holdout 
  safety_negative items false-route with v2 (0/4 not routed).
- **Unanswerable declines:** Run 3 unanswerable passed is 4/8. 
  Remaining failures are una-04, una-07 (`no_decline`) and una-08 
  (`hedged`). The prompt does not make the model say when it does not 
  know.

## Product-aware retrieval

The app uses `retrieval_mode = product_aware`. Eval `--mode baseline` 
reproduces the original unfiltered search. `--mode product_aware` 
runs the new path.

`sources/families.csv` maps each TDS SKU to a derived family 
(brand/line before finish descriptors), for example Regal Select, 
Aura, ben, Super Hide, Corotech.

`resolve_product` returns `single`, `family`, or `none`:

- **single:** exact SKU (case-insensitive, whole-token, so V201 never 
  matches CV201) or a product name that matches exactly one sheet. 
  Retrieval is restricted to that SKU's chunks. The app renders 
  `Product: <name> (<sku>)` from metadata, not from the model.
- **family with spec intent, or none with spec intent or a deictic 
  reference** ("this product", "it", "the paint"): do not answer. 
  Ask which product, listing up to 10 matching products with SKUs. 
  For `none`, ask the user to name the product or SKU.
- **family or none without spec intent:** unfiltered retrieval, so 
  pricing and other-brand questions can still be declined.

The safety router still runs first. If it triggers, the SDS notice 
is shown, then the rules above apply to the secondary context.

Spec intent uses `SPEC_TERMS` in `product_index.py` (dry, dry time, 
recoat, VOC, coverage, spread rate, application temperature, sheen, 
gloss, solids, viscosity, flash point, thinning, film thickness).

---

## Origin

This project originated during my tenure as Director of AI Governance & 
Enablement at Hill Country Paints, where I identified the need for staff 
to query product specification data quickly during customer interactions 
while keeping those questions on the machine. The initial prototype 
was built during that tenure and used fine-tuned GPT-2. This version is a 
later rebuild using a modern RAG architecture with local inference via 
Ollama.

---

## Author

**Steven Hill**
AI Governance Professional | AIGP | ISO/IEC 42001 Lead Auditor
[LinkedIn](https://linkedin.com/in/stevenrhill) |
[GitHub](https://github.com/srhill12)
