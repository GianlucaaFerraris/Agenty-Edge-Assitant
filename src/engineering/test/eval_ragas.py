"""
eval_ragas.py — Evaluación end-to-end con RAGAS v0.4.3 usando modelo local (Ollama).

Métricas (ninguna requiere ground truth):
    - Faithfulness:     ¿La respuesta está respaldada por el contexto RAG?
    - AnswerRelevancy:  ¿La respuesta responde la pregunta?
    - ContextRelevance: ¿Los chunks recuperados son relevantes para la pregunta?

API de RAGAS v0.4.3:
    - Faithfulness(llm=llm)                  ← llm es argumento posicional obligatorio
    - AnswerRelevancy(llm=llm, embeddings=e)  ← embeddings necesario para similitud
    - ContextRelevance(llm=llm)

Requiere haber corrido generate_responses.py primero.

Instalación:
    pip install ragas==0.4.3 langchain-ollama langchain-huggingface datasets

Correr desde la raíz del proyecto:
    python src/engineering/test/eval_ragas.py

Input:  src/engineering/test/data/tutor_responses.json
Output: src/engineering/test/data/ragas_results.json
"""

import json
import sys
import time
import warnings
from pathlib import Path

# Suprimir DeprecationWarnings de RAGAS/LangChain — son informativos, no críticos
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT_DIR = Path(__file__).resolve().parents[3]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR       = Path(__file__).parent / "data"
RESPONSES_FILE = DATA_DIR / "tutor_responses.json"
RESULTS_FILE   = DATA_DIR / "ragas_results.json"

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_LLM_MODEL = "asistente"
OLLAMA_EMB_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


# ─────────────────────────────────────────────────────────────────────────────
# Verificaciones
# ─────────────────────────────────────────────────────────────────────────────

def check_dependencies() -> bool:
    missing = []
    for pkg in ("ragas", "langchain_ollama", "langchain_huggingface", "datasets"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Faltan dependencias: {', '.join(missing)}")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True


def check_ollama() -> bool:
    try:
        import requests
        resp  = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        names = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        if OLLAMA_LLM_MODEL not in names:
            print(f"[ERROR] Modelo '{OLLAMA_LLM_MODEL}' no encontrado.")
            print(f"  Disponibles: {names}")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] Ollama no está corriendo: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Setup — wrappers de RAGAS v0.4.3
# ─────────────────────────────────────────────────────────────────────────────

def setup_ragas_local():
    """
    RAGAS v0.4.3 usa sus propios wrappers internos para LLM y embeddings.
    LangchainLLMWrapper sigue funcionando (con DeprecationWarning suprimido).
    HuggingFaceEmbeddings de ragas.embeddings es la forma nativa recomendada.
    """
    from langchain_ollama import ChatOllama
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    print(f"  [RAGAS] LLM judge  → {OLLAMA_LLM_MODEL} (Ollama local)")
    llm = LangchainLLMWrapper(
        ChatOllama(
            model=OLLAMA_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
            num_predict=512,
        )
    )

    print(f"  [RAGAS] Embeddings → {OLLAMA_EMB_MODEL}")
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=OLLAMA_EMB_MODEL)
    )

    return llm, embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(responses: list[dict]) -> tuple[list, list]:
    rag_items, skipped = [], []

    for item in responses:
        if not item.get("rag_used") or not item.get("rag_context"):
            skipped.append(item["question"])
            continue

        contexts = [c["text"] for c in item.get("rag_chunks", []) if c.get("text")]
        if not contexts:
            skipped.append(item["question"])
            continue

        rag_items.append({
            "question":     item["question"],
            "answer":       item["response"],
            "contexts":     contexts,
            "top_score":    item.get("top_score", 0.0),
            "domain_score": item.get("domain_score", 0.0),
        })

    return rag_items, skipped


# ─────────────────────────────────────────────────────────────────────────────
# Evaluación — API exacta de RAGAS v0.4.3
# ─────────────────────────────────────────────────────────────────────────────

def run_ragas_evaluation(items: list[dict], llm, embeddings) -> dict:
    """
    RAGAS v0.4.3:
        - Faithfulness(llm)              → llm es argumento posicional
        - AnswerRelevancy(llm, embeddings) → necesita embeddings para similitud coseno
        - ContextRelevance(llm)           → solo LLM

    evaluate() recibe un EvaluationDataset o Dataset de HuggingFace.
    """
    from datasets import Dataset as HFDataset
    from ragas import evaluate
    from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRelevance

    # Instanciar métricas con la firma correcta de v0.4.3
    faith   = Faithfulness(llm=llm)
    relev   = AnswerRelevancy(llm=llm, embeddings=embeddings)
    ctx_rel = ContextRelevance(llm=llm)

    dataset = HFDataset.from_dict({
        "question": [i["question"] for i in items],
        "answer":   [i["answer"]   for i in items],
        "contexts": [i["contexts"] for i in items],
    })

    print(f"\n  [RAGAS] Evaluando {len(items)} respuestas...")
    print(f"          Estimado: {len(items) * 2}-{len(items) * 4} minutos\n")

    t0     = time.perf_counter()
    result = evaluate(dataset=dataset, metrics=[faith, relev, ctx_rel])
    elapsed = round(time.perf_counter() - t0, 1)

    return {"result": result, "elapsed": elapsed}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    print("\n" + "═" * 60)
    print("  Evaluación RAGAS v0.4.3 — LLM local (Ollama)")
    print("═" * 60)

    if not check_dependencies():
        sys.exit(1)
    if not check_ollama():
        sys.exit(1)

    if not RESPONSES_FILE.exists():
        print(f"\n[ERROR] No se encontró {RESPONSES_FILE}")
        print("  Corré generate_responses.py primero.")
        sys.exit(1)

    with open(RESPONSES_FILE, "r", encoding="utf-8") as f:
        responses = json.load(f)

    print(f"\n[INFO] Respuestas cargadas:   {len(responses)}")

    items, skipped = build_dataset(responses)
    print(f"[INFO] Con RAG (evaluables):  {len(items)}")
    print(f"[INFO] Sin RAG (omitidas):    {len(skipped)}")

    if not items:
        print("\n[ERROR] No hay respuestas con RAG para evaluar.")
        sys.exit(1)

    print(f"\n{'─' * 60}")
    for i, item in enumerate(items, 1):
        print(f"  [{i:02d}] {item['question'][:55]}...")
        print(f"        score={item['top_score']:.2f} | "
              f"domain={item['domain_score']:.2f} | "
              f"{len(item['contexts'])} chunks")
    print(f"{'─' * 60}\n")

    llm, embeddings = setup_ragas_local()

    try:
        eval_output = run_ragas_evaluation(items, llm, embeddings)
    except Exception as e:
        print(f"\n[ERROR] Falló la evaluación RAGAS: {e}")
        print("\n  Tips:")
        print("  · Evaluá menos items cambiando items[:5] en run_ragas_evaluation()")
        print("  · Verificá que Ollama tenga suficiente RAM libre")
        sys.exit(1)

    result  = eval_output["result"]
    elapsed = eval_output["elapsed"]
    scores  = result.to_pandas()

    # Detectar columnas reales (pueden tener nombres con espacios o guiones bajos)
    cols = list(scores.columns)
    print(f"  [DEBUG] Columnas disponibles: {cols}")

    def find_col(keyword):
        for c in cols:
            if keyword.lower() in c.lower():
                return c
        return None

    c_faith = find_col("faith")
    c_relev = find_col("answer_relev") or find_col("relevancy") or find_col("relevance")
    c_ctx   = find_col("context_relev") or find_col("context_rel")

    faith_avg = float(scores[c_faith].mean()) if c_faith else 0.0
    relev_avg = float(scores[c_relev].mean()) if c_relev else 0.0
    ctx_avg   = float(scores[c_ctx].mean())   if c_ctx   else 0.0

    active    = [v for v in [faith_avg, relev_avg, ctx_avg] if v > 0]
    avg_score = sum(active) / max(len(active), 1)

    print(f"\n{'═' * 60}")
    print(f"  RESUMEN RAGAS v0.4.3")
    print(f"{'─' * 60}")
    if c_faith: print(f"  Faithfulness (respaldo en contexto):    {faith_avg:.1%}")
    if c_relev: print(f"  Answer relevancy (pertinencia):         {relev_avg:.1%}")
    if c_ctx:   print(f"  Context relevance (calidad chunks):     {ctx_avg:.1%}")
    print(f"{'─' * 60}")
    print(f"  Score promedio:  {avg_score:.1%}")
    print(f"  Tiempo total:    {elapsed}s")
    print(f"{'─' * 60}")

    if avg_score >= 0.70:
        print("  🟢 Sistema LISTO para producción")
    elif avg_score >= 0.50:
        print("  🟡 Sistema ACEPTABLE — revisar casos de baja fidelidad")
    else:
        print("  🔴 Sistema REQUIERE mejoras")
    print(f"{'═' * 60}\n")

    print("  Detalle por pregunta:")
    print(f"{'─' * 60}")
    for i in range(len(items)):
        q  = items[i]["question"][:50]
        f  = float(scores[c_faith][i]) if c_faith else 0.0
        ar = float(scores[c_relev][i]) if c_relev else 0.0
        cr = float(scores[c_ctx][i])   if c_ctx   else 0.0
        icon = "✅" if min(f, ar, cr) >= 0.5 else "⚠️ "
        print(f"  {icon} [{i+1:02d}] {q}...")
        print(f"        F={f:.2f} | AR={ar:.2f} | CTX={cr:.2f}")
    print(f"{'─' * 60}\n")

    # Guardar
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "ragas_version":          "0.4.3",
        "avg_faithfulness":       round(faith_avg, 3),
        "avg_answer_relevancy":   round(relev_avg, 3),
        "avg_context_relevance":  round(ctx_avg, 3),
        "avg_overall":            round(avg_score, 3),
        "total_evaluated":        len(items),
        "elapsed_seconds":        elapsed,
        "columns_found":          cols,
        "per_question": [
            {
                "question":          items[i]["question"],
                "faithfulness":      round(float(scores[c_faith][i]), 3) if c_faith else None,
                "answer_relevancy":  round(float(scores[c_relev][i]), 3) if c_relev else None,
                "context_relevance": round(float(scores[c_ctx][i]),   3) if c_ctx   else None,
                "rag_top_score":     items[i]["top_score"],
            }
            for i in range(len(items))
        ],
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  Resultados guardados → {RESULTS_FILE}")


if __name__ == "__main__":
    run()