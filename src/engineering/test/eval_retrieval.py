import os
import sys
from pathlib import Path

# Mismo patrón que main.py — agrega la raíz del proyecto al path
_ROOT_DIR = Path(__file__).resolve().parents[3]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))


# eval_retrieval.py
from src.engineering.rag.rag_engine import get_engine

GOLDEN_SET = [
    # Casos existentes corregidos
    ("What is backpropagation?",
        "Deep Learning Foundations", 0.50),
    ("How does dropout regularization work?",
        "Deep Learning Foundations", 0.45),
    ("What is the transformer attention mechanism?",
        "Deep Learning Foundations", 0.50),   # ← corregido, no el paper
    ("What is a Kalman filter?",
        "Probabilistic Robotics",    0.45),
    ("How does a MOSFET work?",
        "The Art of Electronics",    0.40),
    ("What is LoRA fine-tuning?",
        "LoRA",                      0.50),

    # Nuevos — en español (el idioma real del usuario)
    ("¿Qué es la retropropagación?",
        "Deep Learning Foundations", 0.40),
    ("¿Cómo funciona la normalización por lotes?",
        "Deep Learning Foundations", 0.40),
    ("¿Qué es el filtro de Kalman?",
        "Probabilistic Robotics",    0.40),
    ("¿Cómo funciona un transistor?",
        "The Art of Electronics",    0.35),
    ("¿Qué es el ajuste fino con LoRA?",
        "LoRA",                      0.40),
    ("¿Cómo aprende una red neuronal?",
        "Deep Learning Foundations", 0.40),

    # Nuevos — PID (después del OCR)
    ("Explain PID control for robotics",
        "Modern Robotics",           0.40),
    ("¿Qué es un controlador PID?",
        "Modern Robotics",           0.35),

    # Negativos ampliados
    ("What is the capital of France?",  None, None),
    ("Tell me a joke",                  None, None),
    ("¿Cuánto es 2 más 2?",             None, None),
    ("Who won the World Cup in 2022?",  None, None),
    ("¿Cuál es la mejor película de Nolan?", None, None),
]

engine = get_engine()
hits, misses, false_positives = 0, 0, 0

for question, expected_source, min_score in GOLDEN_SET:
    result = engine.query(question)
    
    if expected_source is None:
        # Caso negativo — no debería ser relevante
        if result.relevant:
            print(f"❌ FALSE POSITIVE: '{question}' → score={result.top_score:.3f}")
            false_positives += 1
        else:
            print(f"✅ Correctly rejected: '{question}'")
        continue
    
    if not result.relevant:
        print(f"❌ MISS: '{question}' → no context retrieved")
        misses += 1
        continue
    
    # Verificar que la fuente correcta esté entre los chunks recuperados
    sources = [c["source"] for c in result.chunks]
    source_hit = any(expected_source.lower() in s.lower() for s in sources)
    score_ok   = result.top_score >= min_score
    
    if source_hit and score_ok:
        print(f"✅ HIT: '{question[:50]}' → score={result.top_score:.3f}, source OK")
        hits += 1
    else:
        print(f"⚠️  PARTIAL: '{question[:50]}' → score={result.top_score:.3f}, "
              f"source_hit={source_hit}, sources={[s[:30] for s in sources]}")
        misses += 1

total_positive = sum(1 for _, s, _ in GOLDEN_SET if s is not None)
print(f"\n{'─'*50}")
print(f"Retrieval Precision@3:  {hits}/{total_positive} = {hits/total_positive:.0%}")
print(f"False positives:        {false_positives}")