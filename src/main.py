"""
main.py — Punto de entrada del asistente edge.

Correr SIEMPRE desde la raíz del proyecto:
    cd ~/Desktop/Agenty-Edge-Assitant
    python src/main.py

Flujo:
  1. Inicia scheduler de recordatorios
  2. Saluda y detecta intent
  3. Lanza el modo correspondiente:
     - english    → levanta LT, TutorSession, apaga LT al salir
     - engineering → EngineeringSession
     - agent       → AgentSession (tool directa)
  4. Desde cualquier modo activo:
     - Tool liviana → ejecuta y vuelve al modo anterior
     - search_web   → confirma, entra en modo búsqueda (sin retorno)
"""

import os
import sys
import datetime
import time

import requests
from openai import OpenAI

# ── Path setup: permite correr como `python src/main.py` desde la raíz ───────
_THIS_FILE = os.path.abspath(__file__)
_SRC_DIR   = os.path.dirname(_THIS_FILE)
_ROOT_DIR  = os.path.dirname(_SRC_DIR)

# Agregar raíz al path para que `from src.X import Y` funcione
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# ── Imports del proyecto ──────────────────────────────────────────────────────
from src.orchestrator.orchestrator import (
    detect_intent,
    detect_intent_from_active_mode,
    generate_greeting,
    generate_clarification,
    generate_return_prompt,
    _resolve_model,
)
from src.orchestrator.context_manager import ContextManager
from src.agent.agent_session import AgentSession
from src.agent import reminder_manager as reminders
from src.english.language_tool_server import ensure_running, ensure_stopped

OLLAMA_URL     = "http://localhost:11434/v1"
MODEL          = "asistente"
FALLBACK_MODEL = "qwen2.5:7b"

client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")


# ── I/O hooks (reemplazar por Whisper/TTS en producción) ─────────────────────

def listen() -> str:
    try:
        return input("[VOS]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return "salir"


def speak(text: str) -> None:
    print(f"\n[ASISTENTE]: {text}\n")


# ── Verificaciones de inicio ──────────────────────────────────────────────────

def _check_ollama() -> bool:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
        return True
    except Exception:
        return False


def _show_alerts() -> None:
    alerts = reminders.pop_alerts()
    if not alerts:
        return
    print("\n" + "⏰" * 20)
    for a in alerts:
        dt = datetime.datetime.fromisoformat(a["remind_at"])
        print(f"  ⏰  {a['title']} — {dt.strftime('%H:%M')}")
    print("⏰" * 20 + "\n")


# ── MODO ENGLISH ──────────────────────────────────────────────────────────────

def _run_english_with_interrupts(ctx: ContextManager) -> None:
    from src.english.tutor_session import (
        TutorSession, classify_intent, extract_proposed_topic,
        check_errors, generate_feedback,
    )

    ctx.set_active("english")
    eng_ctx = ctx.get("english")

    # Levantar LanguageTool
    print("  [LT] Iniciando servidor LanguageTool...")
    lt_ok = ensure_running()

    # Crear o retomar sesión
    if eng_ctx.session is None:
        eng_ctx.session = TutorSession()
        eng_ctx.session.lt_ok = lt_ok
        eng_ctx.session._ask_topic_preference()
    else:
        speak("Retomando el tutor de inglés donde lo dejaste.")
        eng_ctx.session.lt_ok = lt_ok

    eng_ctx.session._start_topic()
    agent = AgentSession()

    try:
        while True:
            _show_alerts()
            student_text = eng_ctx.session.listen()

            if not student_text:
                continue
            if student_text.lower() in ("salir", "exit", "quit", "menu"):
                speak("Cerrando el tutor de inglés.")
                eng_ctx.session._save_log()
                eng_ctx.session = None
                ctx.set_active("idle")
                break

            # ¿Interrupción de agente, cambio de modo o salida?
            interruption = detect_intent_from_active_mode(student_text, "english")

            if interruption == "exit":
                speak("Cerrando el tutor de inglés.")
                eng_ctx.session._save_log()
                eng_ctx.session = None
                ctx.set_active("idle")
                break

            if interruption == "switch":
                eng_ctx.session._save_log()
                eng_ctx.session = None
                ctx.set_active("idle")
                speak("Dale, pasamos al tutor de ingeniería.")
                _run_engineering_with_interrupts(ctx)
                break

            if interruption == "agent":
                result = agent.run_turn(student_text, return_mode="english")

                if result["action"] == "web_search":
                    speak(
                        "Para buscar en internet tengo que pausar el tutor de inglés "
                        "y no vas a poder retomarlo. "
                        f"{result['text']}\n¿Querés continuar con la búsqueda?"
                    )
                    confirm = listen()
                    if confirm.lower() in ("sí", "si", "dale", "yes"):
                        eng_ctx.session._save_log()
                        eng_ctx.session = None
                        ctx.set_active("idle")
                        agent._enter_web_mode(result["search_data"])
                        break
                    else:
                        speak("Entendido, seguimos con el tutor de inglés.")
                        continue

                # Tool liviana — mostrar resultado y preguntar si retomar
                speak(result["text"])
                return_q = generate_return_prompt("english", client, _resolve_model())
                speak(return_q)
                answer = listen()

                intent_ans = detect_intent(answer)
                if intent_ans == "english" or answer.lower() in ("sí", "si", "dale", "yes", "claro"):
                    eng_ctx.session._start_topic()
                    continue
                elif intent_ans == "agent":
                    continue
                else:
                    eng_ctx.session._save_log()
                    eng_ctx.session = None
                    ctx.set_active("idle")
                    break

            else:
                # Turno normal del tutor
                intent_tutor = classify_intent(student_text)
                print(f"  [intent tutor: {intent_tutor}]")

                if intent_tutor == "exit":
                    eng_ctx.session.speak(
                        "It was great practicing English with you! Keep it up!"
                    )
                    eng_ctx.session._save_log()
                    eng_ctx.session = None
                    ctx.set_active("idle")
                    break

                if intent_tutor == "change_topic":
                    eng_ctx.session._pick_new_topic()
                    eng_ctx.session._start_topic()
                    continue

                if intent_tutor == "propose_topic":
                    eng_ctx.session.topic = extract_proposed_topic(student_text)
                    eng_ctx.session.used_topics.add(eng_ctx.session.topic)
                    eng_ctx.session.history = []
                    print(f"  [INFO] Tema propuesto: {eng_ctx.session.topic}")
                    eng_ctx.session._start_topic()
                    continue

                # Respuesta normal
                eng_ctx.session._run_normal_turn(student_text)

    finally:
        # Apagar LT siempre al salir del modo inglés
        print("  [LT] Apagando servidor LanguageTool...")
        ensure_stopped()


# ── MODO ENGINEERING ──────────────────────────────────────────────────────────

def _run_engineering_with_interrupts(ctx: ContextManager) -> None:
    from src.engineering.engineering_session import (
        SYSTEM_ENGINEERING, _resolve_model as eng_resolve, _chat_stream,
    )

    ctx.set_active("engineering")
    eng_ctx = ctx.get("engineering")
    agent   = AgentSession()

    print(f"\n{'═'*60}")
    print(f"  TUTOR DE INGENIERÍA  |  Modelo: {eng_resolve()}")
    print(f"  'salir' → menú  |  'limpiar' → nuevo contexto")
    print(f"{'═'*60}\n")

    if not eng_ctx.history:
        speak("Hola, soy tu tutor de ingeniería. ¿Sobre qué concepto querés hablar?")
    else:
        speak("Retomando donde lo dejamos. ¿Qué más querés explorar?")

    while True:
        _show_alerts()

        try:
            user_text = input("[VOS]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            ctx.set_active("idle")
            break

        if not user_text:
            continue
        if user_text.lower() in ("limpiar", "clear"):
            eng_ctx.history = []
            print("  [INFO] Contexto limpiado.\n")
            continue

        # ¿Interrupción de agente, cambio de modo o salida?
        interruption = detect_intent_from_active_mode(user_text, "engineering")

        if interruption == "exit":
            ctx.set_active("idle")
            break

        if interruption == "switch":
            ctx.set_active("idle")
            speak("Dale, pasamos al tutor de inglés.")
            _run_english_with_interrupts(ctx)
            break

        if interruption == "agent":
            result = agent.run_turn(user_text, return_mode="engineering")

            if result["action"] == "web_search":
                speak(
                    "Para buscar en internet tengo que pausar el tutor de ingeniería "
                    "y no vas a poder retomarlo. "
                    f"{result['text']}\n¿Querés continuar con la búsqueda?"
                )
                confirm = listen()
                if confirm.lower() in ("sí", "si", "dale", "yes"):
                    ctx.set_active("idle")
                    agent._enter_web_mode(result["search_data"])
                    break
                else:
                    speak("Entendido, seguimos con el tutor.")
                    continue

            # Tool liviana
            speak(result["text"])
            return_q = generate_return_prompt("engineering", client, _resolve_model())
            speak(return_q)
            answer = listen()

            intent_ans = detect_intent(answer)
            if intent_ans == "engineering" or answer.lower() in ("sí", "si", "dale", "yes", "claro"):
                speak("Dale, ¿qué más querés explorar?")
                continue
            elif intent_ans == "agent":
                continue
            else:
                ctx.set_active("idle")
                break

        else:
            # Turno normal
            eng_ctx.history.append({"role": "user", "content": user_text})
            messages = [{"role": "system", "content": SYSTEM_ENGINEERING}] + eng_ctx.history[-8:]

            print("\n[INGENIERO]: ", end="", flush=True)
            response, ttft, total = _chat_stream(messages, temperature=0.3, max_tokens=600)
            print(f"\n  ⏱  TTFT={ttft}s | Total={total}s\n")

            eng_ctx.history.append({"role": "assistant", "content": response})
            if len(eng_ctx.history) > 12:
                eng_ctx.history = eng_ctx.history[-12:]


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  AGENTY — Asistente Edge")
    print(f"  {datetime.datetime.now().strftime('%A %d/%m/%Y %H:%M')}")
    print("═" * 60)

    # Verificar Ollama
    if not _check_ollama():
        print("\n[ERROR] Ollama no está corriendo.")
        print("  Arrancalo con: ollama serve")
        sys.exit(1)

    model = _resolve_model()
    print(f"[OK] Ollama conectado | Modelo: {model}\n")

    # Iniciar scheduler de recordatorios en background
    reminders.start_scheduler(interval_minutes=30)

    ctx = ContextManager()

    # Saludo inicial
    greeting = generate_greeting(client, model)
    speak(greeting)

    while True:
        _show_alerts()

        user_text = listen()
        if not user_text:
            continue
        if user_text.lower() in ("salir", "exit", "chau", "quit"):
            speak("¡Hasta luego!")
            ensure_stopped()  # apagar LT si quedó corriendo
            break

        intent = detect_intent(user_text)
        print(f"  [intent: {intent}]")

        if intent == "unclear":
            clarification = generate_clarification(user_text, client, model)
            speak(clarification)
            user_text2 = listen()
            intent = detect_intent(user_text2)
            if intent == "unclear":
                speak("No pude entender qué querés hacer. ¿Querés practicar inglés, consultar algo técnico, o hacer una tarea?")
                continue
            user_text = user_text2

        if intent == "english":
            speak("¡Perfecto, arrancamos con el tutor de inglés!")
            _run_english_with_interrupts(ctx)

        elif intent == "engineering":
            speak("Dale, te conecto con el tutor de ingeniería.")
            _run_engineering_with_interrupts(ctx)

        elif intent == "agent":
            agent = AgentSession()
            result = agent.run_turn(user_text, return_mode=None)

            if result["action"] == "web_search":
                speak(result["text"])
                confirm = listen()
                if confirm.lower() in ("sí", "si", "dale", "yes"):
                    agent._enter_web_mode(result["search_data"])
                else:
                    speak("Entendido, cancelé la búsqueda. ¿Qué más necesitás?")
            else:
                speak(result["text"])

        print()  # línea en blanco antes del próximo prompt


if __name__ == "__main__":
    main()