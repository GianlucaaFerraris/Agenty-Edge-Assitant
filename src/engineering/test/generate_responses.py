"""
generate_responses.py — Genera respuestas reales del tutor con RAG y las guarda en JSON.

Corre el pipeline completo (RAG + LLM) sin interfaz interactiva.
El output se usa como input para eval_faithfulness.py.

Correr desde la raíz del proyecto:
    python src/engineering/test/generate_responses.py

Output:
    src/engineering/test/data/tutor_responses.json
"""

import os
import sys
import json
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT_DIR = Path(__file__).resolve().parents[3]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import requests
from openai import OpenAI

from src.engineering.rag.rag_engine import get_engine

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL     = "http://localhost:11434/v1"
MODEL          = "asistente"
FALLBACK_MODEL = "qwen2.5:7b"
OUTPUT_DIR     = Path(__file__).parent / "data"
OUTPUT_FILE    = OUTPUT_DIR / "tutor_responses.json"

client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")

SYSTEM_ENGINEERING = (
    "You are a brilliant, warm retired engineer and scientist with decades of experience "
    "across software, AI, electronics, physics, chemistry, and systems engineering. "
    "You explain concepts with clarity, depth, and genuine enthusiasm — like a mentor who "
    "loves sharing knowledge. You NEVER write code, never do numerical calculations, and "
    "never solve logic puzzles. Instead, you explain the intuition, the trade-offs, the "
    "history, the analogies, and the real-world implications of technical concepts. "
    "You are direct and concrete — no filler phrases like 'Great question!' or 'Sure!'. "
    "If asked in Spanish, answer in Spanish. If asked in English, answer in English. "
    "No bullet-point lists unless the question is explicitly a comparison or enumeration. "
    "No Chinese characters."
)

SYSTEM_WITH_CONTEXT = (
    SYSTEM_ENGINEERING +
    "\n\nWhen you receive a RELEVANT CONTEXT FROM KNOWLEDGE BASE section, "
    "use it to ground your answer. Cite the source naturally if relevant. "
    "If the context contradicts your prior knowledge, trust the context. "
    "If the context is only partially relevant, use what's useful and fill "
    "the rest with your expertise."
)

# ── Preguntas de evaluación ───────────────────────────────────────────────────
# Mezcla de inglés y español, distintos dominios
# Incluye preguntas donde el RAG debería activarse Y donde no debería
EVAL_QUESTIONS = [
    # AI / Deep Learning — inglés
    "What is backpropagation and why does it work?",
    "Explain the intuition behind dropout regularization.",
    "How does the attention mechanism in transformers decide what to focus on?",
    "What is the difference between LoRA and full fine-tuning?",

    # AI / Deep Learning — español
    "¿Qué es la retropropagación y por qué funciona?",
    "¿Cómo funciona la normalización por lotes en entrenamiento?",
    "¿Qué hace que los transformers sean mejores que los RNNs?",
    "¿Qué es el ajuste fino con LoRA y cuándo conviene usarlo?",

    # Robótica — inglés
    "Explain PID control and its three components.",
    "What is a Kalman filter used for in robotics?",

    # Robótica — español
    "¿Qué es un controlador PID y cómo se ajustan sus parámetros?",
    "¿Para qué sirve el filtro de Kalman en robótica?",

    # Electrónica
    "How does a MOSFET work as a switch?",
    "¿Cómo funciona un transistor?",

    # Fuera de dominio — el RAG NO debería activarse
    "¿Cuál es la capital de Francia?",
    "Tell me a joke about engineers.",
]


def _resolve_model() -> str:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        names = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        return MODEL if MODEL in names else FALLBACK_MODEL
    except Exception:
        return FALLBACK_MODEL


def _generate(messages: list[dict], max_tokens: int = 400) -> tuple[str, float]:
    """Genera una respuesta completa (no streaming) y devuelve (texto, segundos)."""
    model = _resolve_model()
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:  
        text = f"[ERROR] {e}"
    elapsed = round(time.perf_counter() - t0, 2)
    return text, elapsed


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 60)
    print("  Generador de respuestas del tutor")
    print("═" * 60)

    # Verificar Ollama
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        print("\n[ERROR] Ollama no está corriendo. Arrancalo con: ollama serve")
        sys.exit(1)

    model = _resolve_model()
    print(f"[OK] Modelo: {model}")

    # Cargar RAG engine
    print("[OK] Cargando RAG engine...")
    rag = get_engine()
    if not rag.is_ready:
        print("[ERROR] RAG index no encontrado. Corré build_index.py primero.")
        sys.exit(1)

    stats = rag.get_stats()
    print(f"[OK] RAG listo — {stats['vectors']} vectores\n")

    results = []

    for i, question in enumerate(EVAL_QUESTIONS, 1):
        print(f"[{i:02d}/{len(EVAL_QUESTIONS)}] {question[:60]}...")

        # RAG lookup
        t_rag = time.perf_counter()
        rag_result = rag.query_with_domain_check(question)
        t_rag = round(time.perf_counter() - t_rag, 2)

        rag_used    = rag_result.relevant
        rag_context = rag_result.context if rag_used else ""
        rag_chunks  = rag_result.chunks  if rag_used else []

        # Construir prompt
        if rag_used:
            augmented = (
                f"{rag_context}\n\n"
                f"USER QUESTION: {question}\n\n"
                f"Answer based on the context above and your expertise."
            )
            messages = [
                {"role": "system", "content": SYSTEM_WITH_CONTEXT},
                {"role": "user",   "content": augmented},
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_ENGINEERING},
                {"role": "user",   "content": question},
            ]

        # Generar respuesta
        response, t_llm = _generate(messages)

        status = f"RAG(score={rag_result.top_score:.2f})" if rag_used else "sin RAG"
        print(f"       → {status} | LLM={t_llm}s | {len(response.split())} palabras")

        results.append({
            "question":     question,
            "rag_used":     rag_used,
            "rag_context":  rag_context,
            "rag_chunks":   rag_chunks,
            "top_score":    rag_result.top_score,
            "domain_score": rag_result.domain_score,
            "response":     response,
            "t_rag_s":      t_rag,
            "t_llm_s":      t_llm,
        })

    # Guardar
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    rag_count = sum(1 for r in results if r["rag_used"])
    print(f"\n{'═' * 60}")
    print(f"  ✅ {len(results)} respuestas generadas")
    print(f"     Con RAG:   {rag_count}")
    print(f"     Sin RAG:   {len(results) - rag_count}")
    print(f"     Guardado → {OUTPUT_FILE}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    run()