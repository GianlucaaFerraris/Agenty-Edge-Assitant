"""
build_index.py — Builds the FAISS vector index from documents in the docs/ folder.

Run this script once after adding new documents:
    cd ~/Desktop/Agenty-Edge-Assitant
    python src/engineering/rag/build_index.py

Supported formats: PDF, TXT, MD, DOCX, EPUB

What it does:
    1. Scans all documents in src/engineering/rag/docs/
    2. Extracts text from each file
    3. Splits text into overlapping chunks
    4. Generates an embedding vector for each chunk
    5. Saves the FAISS index + metadata to src/engineering/rag/index/
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Generator

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent
DOCS_DIR   = _HERE / "docs"
INDEX_DIR  = _HERE / "index"
INDEX_FILE = INDEX_DIR / "faiss.index"
META_FILE  = INDEX_DIR / "metadata.json"

# ── Chunking config ───────────────────────────────────────────────────────────
CHUNK_SIZE    = 400   # words per chunk — balance between context and speed
CHUNK_OVERLAP = 80    # words of overlap between consecutive chunks
                      # overlap ensures concepts split across chunks are still findable

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# ~470MB model, 384-dim vectors, multilingual (50+ languages including Spanish)
# Trained to align semantically equivalent text across languages in the same vector space
# Downloads automatically to ~/.cache/huggingface/ on first run


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf(path: Path) -> str:
    """Extracts text from a PDF using PyMuPDF. Handles multi-column layouts."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Install PyMuPDF: pip install pymupdf")

    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        text = page.get_text("text")  # preserves reading order
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def extract_txt(path: Path) -> str:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="replace")


def extract_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")
    doc = Document(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_epub(path: Path) -> str:
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Install: pip install ebooklib beautifulsoup4")

    book  = epub.read_epub(str(path))
    parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n")
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def extract_text(path: Path) -> str | None:
    """Dispatches to the correct extractor based on file extension."""
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            return extract_pdf(path)
        elif ext in (".txt", ".md"):
            return extract_txt(path)
        elif ext == ".docx":
            return extract_docx(path)
        elif ext == ".epub":
            return extract_epub(path)
        else:
            print(f"  [SKIP] Unsupported format: {path.name}")
            return None
    except Exception as e:
        print(f"  [ERROR] Could not extract {path.name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Splits text into overlapping word-based chunks.

    Why word-based instead of character-based?
    → More consistent chunk sizes across different languages and layouts.

    Why overlap?
    → A concept explained across two pages won't be split and lost.
       The overlapping region appears in both chunks, so a query about
       that concept will find at least one of them.

    Returns list of dicts:
        {
            "text":   str,   ← the chunk text
            "source": str,   ← filename
            "chunk":  int,   ← chunk index within the document
            "words":  int,   ← word count
        }
    """
    # Clean whitespace
    text  = " ".join(text.split())
    words = text.split()

    if not words:
        return []

    chunks = []
    start  = 0
    idx    = 0

    while start < len(words):
        end        = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])

        chunks.append({
            "text":   chunk_text,
            "source": source,
            "chunk":  idx,
            "words":  end - start,
        })

        idx   += 1
        start += chunk_size - overlap  # step forward with overlap

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────

def load_embedding_model():
    """Loads the sentence-transformers model. Downloads on first run (~90MB)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Install: pip install sentence-transformers")

    print(f"  [EMB] Loading model '{EMBEDDING_MODEL}'...")
    print(f"        (downloads ~90MB on first run, cached afterwards)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  [EMB] Model loaded. Vector dimensions: {model.get_sentence_embedding_dimension()}")
    return model


def embed_chunks(chunks: list[dict], model,
                 batch_size: int = 64) -> np.ndarray:
    """
    Generates embeddings for all chunks in batches.

    Batching is important: sending all chunks at once may OOM on low-RAM devices.
    batch_size=64 is safe for 4GB RAM devices.

    Returns numpy array of shape (n_chunks, 384).
    """
    texts = [c["text"] for c in chunks]
    print(f"  [EMB] Embedding {len(texts)} chunks in batches of {batch_size}...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize → cosine similarity via dot product
    )
    return embeddings.astype("float32")


# ─────────────────────────────────────────────────────────────────────────────
# FAISS index
# ─────────────────────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray):
    """
    Builds a FAISS flat index (IndexFlatIP = Inner Product on normalized vectors).

    Why IndexFlatIP instead of IndexFlatL2?
    → When vectors are L2-normalized, inner product = cosine similarity.
      This gives more meaningful similarity scores for text (0 = unrelated, 1 = identical).

    Why Flat and not IVF?
    → IndexFlatIP does exact search — no approximation.
      For <100k chunks it's fast enough (<10ms per query).
      IVF adds complexity and requires training; not needed at this scale.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("Install: pip install faiss-cpu")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  [FAISS] Index built: {index.ntotal} vectors, {dim} dimensions")
    return index


def save_index(index, chunks: list[dict]) -> None:
    """Saves the FAISS index and metadata to disk."""
    import faiss

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))
    print(f"  [SAVE] FAISS index → {INDEX_FILE}")

    # Metadata: maps chunk id (= FAISS row index) to source info + text
    metadata = {str(i): c for i, c in enumerate(chunks)}
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  [SAVE] Metadata    → {META_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def scan_docs(docs_dir: Path) -> list[Path]:
    """Returns all supported files in docs/ recursively."""
    supported = {".pdf", ".txt", ".md", ".docx", ".epub"}
    files = [
        p for p in docs_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in supported
    ]
    files.sort()
    return files


def build(force: bool = False) -> None:
    """
    Full pipeline: scan → extract → chunk → embed → index → save.

    Args:
        force: If True, rebuilds even if index already exists.
    """
    print("\n" + "═" * 60)
    print("  RAG Index Builder")
    print("═" * 60)

    # Check if rebuild is needed
    if INDEX_FILE.exists() and META_FILE.exists() and not force:
        print(f"\n[INFO] Index already exists at {INDEX_FILE}")
        print("  Use --force to rebuild.")
        return

    # Scan documents
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    files = scan_docs(DOCS_DIR)

    if not files:
        print(f"\n[ERROR] No documents found in {DOCS_DIR}")
        print("  Add PDFs, TXTs, or EPUBs and run again.")
        return

    print(f"\n[DOCS] Found {len(files)} file(s):")
    for f in files:
        size_mb = f.stat().st_size / 1e6
        print(f"  • {f.relative_to(DOCS_DIR)}  ({size_mb:.1f} MB)")

    # Extract + chunk
    all_chunks = []
    t0 = time.perf_counter()

    for path in files:
        print(f"\n[EXTRACT] {path.name}...")
        text = extract_text(path)
        if not text or len(text.strip()) < 100:
            print(f"  [SKIP] Too short or empty.")
            continue

        chunks = chunk_text(text, source=path.name)
        all_chunks.extend(chunks)
        print(f"  → {len(chunks)} chunks from {len(text.split()):,} words")

    if not all_chunks:
        print("\n[ERROR] No chunks generated. Check your documents.")
        return

    total_words = sum(c["words"] for c in all_chunks)
    print(f"\n[CHUNK] Total: {len(all_chunks)} chunks ({total_words:,} words)")
    print(f"        Avg chunk size: {total_words // len(all_chunks)} words")

    # Embed
    model      = load_embedding_model()
    embeddings = embed_chunks(all_chunks, model)

    # Build + save FAISS index
    print("\n[FAISS] Building index...")
    index = build_faiss_index(embeddings)
    save_index(index, all_chunks)

    elapsed = time.perf_counter() - t0
    size_mb = (INDEX_FILE.stat().st_size + META_FILE.stat().st_size) / 1e6

    print(f"\n{'═' * 60}")
    print(f"  ✅ Index built successfully")
    print(f"     Chunks:    {len(all_chunks)}")
    print(f"     Documents: {len(files)}")
    print(f"     Index size: {size_mb:.1f} MB")
    print(f"     Time:       {elapsed:.1f}s")
    print(f"{'═' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAG index from documents")
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild index even if it already exists"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help=f"Words per chunk (default: {CHUNK_SIZE})"
    )
    parser.add_argument(
        "--overlap", type=int, default=CHUNK_OVERLAP,
        help=f"Overlap words between chunks (default: {CHUNK_OVERLAP})"
    )
    args = parser.parse_args()

    CHUNK_SIZE    = args.chunk_size
    CHUNK_OVERLAP = args.overlap

    build(force=args.force)