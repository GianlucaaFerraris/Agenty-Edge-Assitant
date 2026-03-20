"""
eval_faithfulness.py — Evalúa si las respuestas del tutor están respaldadas
por el contexto RAG recuperado.

Usa un modelo NLI (Natural Language Inference) para clasificar cada oración
de la respuesta como ENTAILMENT, NEUTRAL, o CONTRADICTION respecto al contexto.

Requiere haber corrido generate_responses.py primero.

Correr desde la raíz del proyecto:
    python src/engineering/test/eval_faithfulness.py

Input:
    src/engineering/test/data/tutor_responses.json

Output en terminal + src/engineering/test/data/faithfulness_results.json
"""

import json
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT_DIR = Path(__file__).resolve().parents[3]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR         = Path(__file__).parent / "data"
RESPONSES_FILE   = DATA_DIR / "tutor_responses.json"
RESULTS_FILE     = DATA_DIR / "faithfulness_results.json"

# ── NLI config ────────────────────────────────────────────────────────────────
NLI_MODEL        = "cross-encoder/nli-deberta-v3-small"
MIN_SENTENCE_LEN = 25   # caracteres — oraciones más cortas se descartan
CONTEXT_CHARS    = 800  # cuántos chars del contexto usar por oración


def load_nli():
    """Carga el modelo NLI. Usa GPU si está disponible, CPU si no."""
    try:
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device == 0 else "CPU"
        print(f"  [NLI] Cargando {NLI_MODEL} en {device_name}...")
        nli = pipeline("text-classification", model=NLI_MODEL, device=device)
        print(f"  [NLI] Modelo cargado.")
        return nli
    except ImportError:
        print("[ERROR] Instalá transformers: pip install transformers torch")
        sys.exit(1)


def split_sentences(text: str) -> list[str]:
    """
    Divide la respuesta en oraciones filtrando las muy cortas.
    Usa punto como separador — suficiente para respuestas técnicas en prosa.
    """
    raw = [s.strip() for s in text.replace("\n", " ").split(".")]
    return [s for s in raw if len(s) >= MIN_SENTENCE_LEN]


def score_sentence(nli, sentence: str, context: str) -> dict:
    """
    Evalúa una oración contra el contexto con el modelo NLI.

    El contexto RAG es la premisa, la oración de la respuesta es la hipótesis.
    Retorna label (ENTAILMENT/NEUTRAL/CONTRADICTION), score de confianza, y
    si la oración es fiel al contexto.
    """
    input_text = f"{context[:CONTEXT_CHARS]} [SEP] {sentence}"
    result = nli(input_text, truncation=True)[0]
    label  = result["label"].upper()

    return {
        "sentence": sentence,
        "label":    label,
        "score":    round(result["score"], 4),
        "faithful": label == "ENTAILMENT",
    }


def faithfulness_score(nli, response: str, context: str) -> dict:
    """
    Calcula el faithfulness score de una respuesta completa.
    Devuelve fracción de oraciones ENTAILMENT + detalle por oración.
    """
    sentences = split_sentences(response)

    if not sentences:
        return {
            "score": 0.0, "sentences": 0,
            "entailed": 0, "neutral": 0, "contradicted": 0,
            "details": []
        }

    details      = [score_sentence(nli, s, context) for s in sentences]
    entailed     = sum(1 for d in details if d["label"] == "ENTAILMENT")
    neutral      = sum(1 for d in details if d["label"] == "NEUTRAL")
    contradicted = sum(1 for d in details if d["label"] == "CONTRADICTION")

    return {
        "score":        round(entailed / len(details), 3),
        "sentences":    len(details),
        "entailed":     entailed,
        "neutral":      neutral,
        "contradicted": contradicted,
        "details":      details,
    }


def interpret(score: float) -> str:
    if score >= 0.75:
        return "✅ ALTA fidelidad"
    elif score >= 0.50:
        return "⚠️  MEDIA fidelidad"
    else:
        return "❌ BAJA fidelidad"


def run() -> None:
    print("\n" + "═" * 60)
    print("  Evaluación de Faithfulness (NLI)")
    print("═" * 60)

    if not RESPONSES_FILE.exists():
        print(f"\n[ERROR] No se encontró {RESPONSES_FILE}")
        print("  Corré generate_responses.py primero.")
        sys.exit(1)

    with open(RESPONSES_FILE, "r", encoding="utf-8") as f:
        responses = json.load(f)

    # Solo las respuestas que usaron RAG tienen contexto para evaluar
    rag_responses = [r for r in responses if r["rag_used"] and r["rag_context"]]
    no_rag        = [r for r in responses if not r["rag_used"]]

    print(f"\n[INFO] Total respuestas: {len(responses)}")
    print(f"       Con RAG (evaluables): {len(rag_responses)}")
    print(f"       Sin RAG (skip):       {len(no_rag)}\n")

    if not rag_responses:
        print("[ERROR] No hay respuestas con RAG para evaluar.")
        sys.exit(1)

    nli = load_nli()
    print()

    results      = []
    total_scores = []

    for i, item in enumerate(rag_responses, 1):
        question = item["question"]
        response = item["response"]
        context  = item["rag_context"]

        print(f"[{i:02d}/{len(rag_responses)}] {question[:55]}...")

        faith = faithfulness_score(nli, response, context)
        total_scores.append(faith["score"])

        print(f"       → {interpret(faith['score'])} "
              f"({faith['score']:.0%}) | "
              f"{faith['entailed']}E / {faith['neutral']}N / {faith['contradicted']}C "
              f"de {faith['sentences']} oraciones")

        # Mostrar oraciones contradictorias si las hay
        for d in faith["details"]:
            if d["label"] == "CONTRADICTION":
                print(f"         ⚠️  CONTRADICCIÓN: \"{d['sentence'][:80]}\"")

        results.append({
            "question":         question,
            "rag_top_score":    item["top_score"],
            "faithfulness":     faith,
            "response_preview": response[:200],
        })

    # ── Resumen ───────────────────────────────────────────────────────────────
    avg    = sum(total_scores) / len(total_scores)
    high   = sum(1 for s in total_scores if s >= 0.75)
    medium = sum(1 for s in total_scores if 0.50 <= s < 0.75)
    low    = sum(1 for s in total_scores if s < 0.50)

    print(f"\n{'═' * 60}")
    print(f"  RESUMEN FAITHFULNESS")
    print(f"{'─' * 60}")
    print(f"  Score promedio:    {avg:.1%}")
    print(f"  Alta fidelidad:    {high}/{len(rag_responses)} respuestas (≥75%)")
    print(f"  Media fidelidad:   {medium}/{len(rag_responses)} respuestas (50-74%)")
    print(f"  Baja fidelidad:    {low}/{len(rag_responses)} respuestas (<50%)")
    print(f"{'─' * 60}")

    if avg >= 0.70:
        print(f"  🟢 Sistema LISTO para producción")
    elif avg >= 0.50:
        print(f"  🟡 Sistema ACEPTABLE — revisar respuestas de baja fidelidad")
    else:
        print(f"  🔴 Sistema REQUIERE mejoras — contexto o threshold mal calibrado")

    print(f"{'═' * 60}\n")

    # Guardar resultados
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "avg_faithfulness": round(avg, 3),
        "total_evaluated":  len(rag_responses),
        "high_fidelity":    high,
        "medium_fidelity":  medium,
        "low_fidelity":     low,
        "per_question":     results,
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  Resultados guardados → {RESULTS_FILE}")


if __name__ == "__main__":
    run()