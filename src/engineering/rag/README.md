# Agenty — Local RAG Engineering Tutor for Edge Devices

A fully offline AI engineering tutor running on edge hardware (NVIDIA Jetson Orin Nano / Radxa Rock 5B), combining a fine-tuned Qwen 2.5 7B INT4 model with a local Retrieval-Augmented Generation (RAG) pipeline built over a technical knowledge base of 22 books and papers across AI, Robotics, Electronics, and Data Science.

> No cloud. No API keys. No internet required at inference time.

---

## What it does

The engineering tutor answers technical questions in English and Spanish by retrieving relevant passages from a local FAISS vector index built from ~3.7 million words of technical literature, then grounding the LLM response in that context to reduce hallucinations on precise technical topics.

```
User question (EN or ES)
        │
        ▼
Semantic domain check     ← MiniLM multilingual embedding, ~100ms
        │
   In domain?
        │
       Yes → FAISS search over 9,415 vectors   ← cosine similarity, <10ms
                │
           score ≥ 0.40?
                │
               Yes → inject context → LLM (Qwen 2.5 7B Q4_K_M)
                No → LLM answers directly from parametric knowledge
        │
       No  → LLM answers directly
```

---

## Knowledge Base

| Domain | Sources |
|--------|---------|
| Deep Learning / AI | Bishop & Bishop (2024), Géron (2022), Chip Huyen (2023), Attention is All You Need, BERT, LoRA, QLoRA, ReAct, RAG paper, Sutton & Barto RL |
| Computer Vision | Szeliski — CV Algorithms and Applications |
| Recommender Systems | Bischof & Yee, Neural CF, BERT4Rec, Wide & Deep, Message-Passing CF |
| Robotics | Lynch & Park — Modern Robotics, Murphy — AI Robotics, Choset et al. — Principles of Robot Motion, Probabilistic Robotics |
| Electronics | Horowitz & Hill — The Art of Electronics |
| Data Science | Wes McKinney — Python for Data Analysis |

**Total index:** 9,415 chunks · 3,759,655 words · 38.8 MB on disk

---

## Architecture

```
src/
├── main.py                          ← entry point, intent routing
├── orchestrator/
│   └── orchestrator.py             ← intent detection, session management
├── engineering/
│   ├── engineering_session.py      ← tutor loop with RAG integration
│   └── rag/
│       ├── build_index.py          ← PDF/EPUB/TXT → chunks → FAISS index
│       ├── rag_engine.py           ← query engine: domain check + FAISS search
│       ├── docs/                   ← source documents (not committed)
│       └── index/                  ← generated FAISS index (not committed)
├── english/                        ← English language tutor session
└── agent/                          ← tool use / agentic session
```

### Key design decisions

**Single embedding per query** — `query_with_domain_check()` embeds the user question once and reuses the same vector for both the domain filter and the FAISS search. No redundant inference.

**Two-gate filtering** — Domain threshold (0.15) acts as a cheap semantic pre-filter before touching FAISS. Relevance threshold (0.40) is the quality gate that decides whether context is injected into the LLM prompt. This separation avoids polluting the LLM with borderline-relevant chunks.

**Multilingual embeddings** — `paraphrase-multilingual-MiniLM-L12-v2` was chosen over the English-only `all-MiniLM-L6-v2` to align Spanish queries against English-language technical books. The multilingual model maps semantically equivalent text across languages into the same vector space.

**Context injected as user message** — RAG context is prepended to the user turn, not the system prompt. This causes the LLM to treat it as grounding information rather than a behavioral instruction, which produces more faithful citations.

**IndexFlatIP over IVF** — Exact cosine search (inner product on L2-normalized vectors) with no approximation. At 9,415 vectors, exact search takes <10ms — no need for IVF clustering complexity.

---

## Embedding Models Comparison

| Model | Dims | Size | EN Retrieval | ES Retrieval | Chosen |
|-------|------|------|-------------|-------------|--------|
| `all-MiniLM-L6-v2` | 384 | 90 MB | ✅ Good | ❌ Poor | No |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 470 MB | ✅ Good | ✅ Good | **Yes** |

---

## Evaluation Results

### Retrieval Quality (eval_retrieval.py)

Golden set of 19 questions across all domains, in English and Spanish, with 5 negative (out-of-domain) cases.

| Metric | Value |
|--------|-------|
| Precision@3 (English) | **6/6 = 100%** |
| Precision@3 (Spanish) | **8/8 = 100%** |
| Overall Precision@3 | **14/14 = 100%** |
| False positive rate | **0/5 = 0%** |
| Avg relevance score (hits) | **0.68** |

Score distribution of confirmed hits:

| Question | Score |
|----------|-------|
| Dropout regularization | 0.763 |
| Kalman filter | 0.754 |
| Transformer attention | 0.728 |
| LoRA fine-tuning | 0.714 |
| PID control | 0.710 |
| Backpropagation | 0.692 |
| Batch normalization (ES) | 0.684 |
| Kalman filter (ES) | 0.755 |
| Transistor (ES) | 0.657 |
| PID controller (ES) | 0.624 |

### Threshold Calibration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `DOMAIN_THRESHOLD` | 0.15 | Permissive pre-filter; FAISS threshold is the real gate |
| `RELEVANCE_THRESHOLD` | 0.40 | Eliminates all false positives while keeping all legitimate hits |
| `TOP_K` | 3 | Enough context without overwhelming the LLM prompt |
| `MAX_CONTEXT_WORDS` | 500 | Respects Qwen 2.5's context window budget |

### Response Quality (human evaluation)

14 responses generated with RAG active were manually evaluated across three dimensions:

- **Factual correctness**: All responses technically accurate, no hallucinations detected
- **RAG grounding**: Responses with RAG correctly incorporated retrieved context
- **Language handling**: Both English and Spanish responses fluent and domain-appropriate

One known limitation: the word "transformer" in Spanish queries occasionally retrieves chunks about electrical transformers from *The Art of Electronics* due to lexical ambiguity. Reformulating as "transformer de deep learning" or "transformer arquitectura" resolves this.

---

## Setup

### 1. Requirements

```bash
# Python 3.10+
pip install sentence-transformers faiss-cpu pymupdf
pip install openai requests                          # Ollama client
pip install ocrmypdf                                 # optional, for scanned PDFs
```

### 2. Run Ollama with the fine-tuned model

```bash
# Create the model from the GGUF
ollama create asistente -f model/gguf/Modelfile

# Verify
ollama list  # should show asistente:latest
```

### 3. Add documents and build the index

Place PDFs, EPUBs, or TXT files in `src/engineering/rag/docs/` (subdirectories supported), then:

```bash
python src/engineering/rag/build_index.py
```

Expected output for ~22 documents:
```
✅ Index built successfully
   Chunks:     9415
   Documents:  23
   Index size: 38.8 MB
   Time:       33.5s
```

### 4. Run the assistant

```bash
cd ~/Desktop/Agenty-Edge-Assitant
python src/main.py
```

### 5. Run the test suite

```bash
# Retrieval quality
python src/engineering/test/eval_retrieval.py

# Generate tutor responses for faithfulness evaluation
python src/engineering/test/generate_responses.py

# NLI-based faithfulness scoring
python src/engineering/test/eval_faithfulness.py
```

---

## Hardware Targets

| Device | CPU | RAM | Expected embed latency | Expected LLM latency |
|--------|-----|-----|----------------------|---------------------|
| NVIDIA Jetson Orin Nano | Cortex-A78AE | 8 GB | ~40-80ms (DLA) | ~3-6s |
| Radxa Rock 5B | Cortex-A76 | 16 GB | ~80-120ms | ~5-10s |
| Dev machine (Predator) | i7/Ryzen | 16+ GB | ~100ms | ~10-15s |

The RAG overhead (~100ms embedding + <10ms FAISS) is negligible compared to LLM inference time on all target devices.

---

## Configuration Reference

All tunable parameters are at the top of their respective files:

**`src/engineering/rag/build_index.py`**
```python
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE      = 400    # words per chunk
CHUNK_OVERLAP   = 80     # overlap between consecutive chunks
```

**`src/engineering/rag/rag_engine.py`**
```python
EMBEDDING_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"
RELEVANCE_THRESHOLD = 0.40   # min cosine similarity to inject context
DOMAIN_THRESHOLD    = 0.15   # min similarity to domain description
TOP_K               = 3      # chunks retrieved per query
MAX_CONTEXT_WORDS   = 500    # hard cap on injected context
```

---

## .gitignore

```
src/engineering/rag/index/    # generated, rebuild with build_index.py
src/engineering/rag/docs/     # PDFs are large and may be copyrighted
model/                        # GGUF files are large
src/engineering/test/data/    # generated evaluation outputs
```

---

## License

MIT