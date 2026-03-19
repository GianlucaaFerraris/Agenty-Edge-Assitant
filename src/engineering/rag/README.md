# RAG — Retrieval-Augmented Generation for the Engineering Tutor

This module gives the engineering tutor access to a local knowledge base
built from your own documents (PDFs, books, papers, notes).

When a question falls within the indexed domain, the tutor retrieves
the most relevant passages and uses them to ground its answer —
reducing hallucinations on technical topics.

---

## How It Works

```
User question
      ↓
should_use_rag(question)   ← fast keyword check, ~0ms, no LLM call
      ↓
   Yes → rag_engine.query(question)
              ↓
         Embed question (MiniLM, ~100ms)
              ↓
         Search FAISS index (<10ms)
              ↓
         Score ≥ threshold?
              ↓
         Yes → inject context into prompt
               "Un momento, déjame revisar en mis libros..."
               LLM answers with grounded context
              ↓
         No  → answer directly (nothing relevant found)
      ↓
   No  → answer directly (out of domain)
```

---

## File Structure

```
src/engineering/rag/
  build_index.py        ← run once to build the index from your docs
  rag_engine.py         ← query engine used by engineering_session.py
  docs/                 ← put your PDFs, EPUBs, TXTs here
    deep_learning/
    robotics/
    electronics/
    physics/
  index/                ← auto-generated, do NOT add to git
    faiss.index         ← the vector index (~50MB per 500-page book)
    metadata.json       ← chunk text + source mapping
```

---

## Setup

### 1. Install dependencies

```bash
pip install sentence-transformers faiss-cpu pymupdf
# Optional (for EPUB support):
pip install ebooklib beautifulsoup4
```

### 2. Add your documents

Put PDFs, EPUBs, TXTs in `src/engineering/rag/docs/`.
Subdirectories are supported — the builder scans recursively.

Recommended first document:
- **Bishop (2025) — Deep Learning: Foundations and Concepts**
  Free official PDF at: https://www.bishopbook.com

### 3. Build the index

```bash
python src/engineering/rag/build_index.py
```

Expected output:
```
════════════════════════════════════════════════════════════
  RAG Index Builder
════════════════════════════════════════════════════════════

[DOCS] Found 1 file(s):
  • deep_learning_bishop_2025.pdf  (45.2 MB)

[EXTRACT] deep_learning_bishop_2025.pdf...
  → 1842 chunks from 245,031 words

[EMB] Loading model 'all-MiniLM-L6-v2'...
      (downloads ~90MB on first run, cached afterwards)
[EMB] Embedding 1842 chunks in batches of 64...

[FAISS] Index built: 1842 vectors, 384 dimensions

════════════════════════════════════════════════════════════
  ✅ Index built successfully
     Chunks:     1842
     Documents:  1
     Index size: 6.1 MB
     Time:       47.3s
════════════════════════════════════════════════════════════
```

### 4. Test the index

```bash
python src/engineering/rag/rag_engine.py "What is backpropagation?"
python src/engineering/rag/rag_engine.py "How do transformers work?"
python src/engineering/rag/rag_engine.py "What is the capital of France?"
```

The last query should return `relevant: False` — out of domain.

### 5. Run the assistant

```bash
python src/main.py
```

The engineering tutor will automatically use the RAG when relevant.

---

## Rebuilding the index

When you add new documents:

```bash
python src/engineering/rag/build_index.py --force
```

---

## Configuration

All parameters are at the top of each file:

### build_index.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 400 | Words per chunk |
| `CHUNK_OVERLAP` | 80 | Overlap words between chunks |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model |

**Tuning chunk size:**
- Smaller chunks (200-300 words): better precision, more chunks, larger index
- Larger chunks (500-600 words): more context per result, fewer chunks, may dilute relevance
- 400 words is a good default for technical books

### rag_engine.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOP_K` | 3 | Chunks retrieved per query |
| `RELEVANCE_THRESHOLD` | 0.30 | Minimum cosine similarity (0-1) |
| `MAX_CONTEXT_WORDS` | 500 | Hard cap on injected context |

**Tuning the threshold:**
- Too low (< 0.20): injects irrelevant context, confuses the LLM
- Too high (> 0.50): misses relevant content, RAG rarely triggers
- 0.30 is a good starting point — test with your documents and adjust

---

## Latency Impact

With `all-MiniLM-L6-v2` on CPU (ARM or x86):

| Step | Time |
|------|------|
| Keyword domain check | ~0ms |
| Embedding the question | ~50-150ms |
| FAISS search (exact, flat index) | <10ms |
| LLM inference with extra context | +3-5s vs. baseline |

Total overhead when RAG triggers: **~4-6 seconds** on top of baseline inference.
When RAG does not trigger (out of domain or low score): **~0ms overhead**.

---

## Adding New Domains

To extend the keyword list that triggers RAG lookup, edit `RAG_DOMAINS`
in `rag_engine.py`:

```python
RAG_DOMAINS = {
    # Add your new keywords here
    "quantum computing", "cryptography", "topology",
    ...
}
```

---

## What Goes in the .gitignore

The `index/` folder should not be committed — it's generated from your docs
and can be large. Add this to your `.gitignore`:

```
src/engineering/rag/index/
src/engineering/rag/docs/
```

The docs themselves (PDFs, books) also shouldn't be committed due to size
and copyright. Each developer builds their own index locally.

---

## Troubleshooting

**"Index not found — run build_index.py first"**
→ You haven't built the index yet, or you're running from the wrong directory.
  Always run from the project root: `python src/engineering/rag/build_index.py`

**RAG never triggers**
→ Your question keywords don't match `RAG_DOMAINS`. Add relevant keywords
  or lower `RELEVANCE_THRESHOLD` slightly.

**RAG triggers but answers are worse**
→ Your documents don't cover that topic well, or chunk size is too small.
  Try `--chunk-size 500` when rebuilding.

**Very slow on first run**
→ The embedding model (`all-MiniLM-L6-v2`, ~90MB) is downloading.
  Subsequent runs use the cached version in `~/.cache/huggingface/`.

**Out of memory during build**
→ Reduce batch size: edit `embed_chunks(..., batch_size=32)` in `build_index.py`