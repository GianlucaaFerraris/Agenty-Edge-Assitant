"""
language_tool_server.py — Lifecycle del servidor LanguageTool.

Levanta el servidor LT como subproceso al entrar al tutor de inglés
y lo apaga limpiamente al salir.

Uso:
    from src.english.language_tool_server import LTServer

    with LTServer() as lt:
        if lt.running:
            # LT disponible
            errors = check_errors(text)

O manualmente:
    lt = LTServer()
    lt.start()
    ...
    lt.stop()

Requisitos:
    - Java instalado (java -version debe funcionar)
    - JAR en ~/languagetool/LanguageTool-*/languagetool-server.jar
      (o configurar LT_JAR_PATH en variables de entorno)
"""

import os
import subprocess
import time
import requests
import glob
import signal

LT_PORT    = 8081
LT_URL     = f"http://localhost:{LT_PORT}/v2/languages"
LT_TIMEOUT = 30  # segundos máximos para esperar que arranque

# Busca el JAR automáticamente en ubicaciones comunes
def _find_jar() -> str | None:
    # 1. Variable de entorno explícita
    env_path = os.environ.get("LT_JAR_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # 2. ~/languagetool/ (instalación típica en Linux)
    patterns = [
        os.path.expanduser("~/languagetool/LanguageTool-*/languagetool-server.jar"),
        os.path.expanduser("~/LanguageTool-*/languagetool-server.jar"),
        "/opt/languagetool/languagetool-server.jar",
        "/usr/local/lib/languagetool/languagetool-server.jar",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return sorted(matches)[-1]  # versión más nueva si hay varias

    return None


class LTServer:
    """
    Maneja el ciclo de vida del servidor LanguageTool.
    Soporta context manager (with statement).
    """

    def __init__(self, jar_path: str = None):
        self.jar_path = jar_path or _find_jar()
        self._process: subprocess.Popen | None = None
        self.running  = False
        self._already_running = False  # si LT ya estaba corriendo antes de nosotros

    def _is_up(self) -> bool:
        """Chequea si el servidor ya está respondiendo."""
        try:
            resp = requests.get(LT_URL, timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def start(self) -> bool:
        """
        Arranca el servidor. Si ya está corriendo, no hace nada.
        Retorna True si el servidor está disponible.
        """
        # Si ya estaba corriendo (ej: iniciado manualmente), usarlo
        if self._is_up():
            print("  [LT] Servidor ya estaba corriendo — usando instancia existente.")
            self._already_running = True
            self.running = True
            return True

        if not self.jar_path:
            print("  [LT] ⚠ JAR no encontrado. Opciones:")
            print("       1. Descargar desde https://languagetool.org/download/")
            print("       2. Extraer en ~/languagetool/")
            print("       3. O setear: export LT_JAR_PATH=/ruta/al/languagetool-server.jar")
            print("  [LT] El tutor funcionará SIN detección de errores.")
            self.running = False
            return False

        jar_dir = os.path.dirname(self.jar_path)
        print(f"  [LT] Arrancando servidor desde: {self.jar_path}")

        try:
            self._process = subprocess.Popen(
                [
                    "java", "-cp", self.jar_path,
                    "org.languagetool.server.HTTPServer",
                    "--port", str(LT_PORT),
                    "--allow-origin", "*",
                    "--public",
                ],
                cwd=jar_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,  # grupo de procesos para matar limpio
            )
        except FileNotFoundError:
            print("  [LT] ⚠ Java no encontrado. Instalalo con: sudo apt install default-jre")
            self.running = False
            return False
        except Exception as e:
            print(f"  [LT] ⚠ Error al arrancar: {e}")
            self.running = False
            return False

        # Esperar a que responda
        print(f"  [LT] Esperando que el servidor levante (máx {LT_TIMEOUT}s)...", end="", flush=True)
        deadline = time.time() + LT_TIMEOUT
        while time.time() < deadline:
            if self._is_up():
                print(" ✅")
                self.running = True
                return True
            time.sleep(1)
            print(".", end="", flush=True)

        print(" ❌ Timeout")
        self.stop()
        self.running = False
        return False

    def stop(self) -> None:
        """Apaga el servidor si fue iniciado por nosotros."""
        if self._already_running:
            # No apagamos algo que no arrancamos
            print("  [LT] Servidor externo — no se apaga.")
            self.running = False
            return

        if self._process is None:
            self.running = False
            return

        try:
            # Matar todo el grupo de procesos (Java puede tener hijos)
            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            self._process.wait(timeout=5)
            print("  [LT] Servidor apagado.")
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
            except Exception:
                pass
        except Exception as e:
            print(f"  [LT] Error al apagar: {e}")
        finally:
            self._process = None
            self.running  = False

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False  # no suprimir excepciones


# ── Singleton global para main.py ─────────────────────────────────────────────
_lt_server: LTServer | None = None


def get_server() -> LTServer:
    """Retorna la instancia global del servidor LT."""
    global _lt_server
    if _lt_server is None:
        _lt_server = LTServer()
    return _lt_server


def ensure_running() -> bool:
    """Arranca el servidor si no está corriendo. Retorna True si está disponible."""
    return get_server().start()


def ensure_stopped() -> None:
    """Apaga el servidor si fue iniciado por nosotros."""
    global _lt_server
    if _lt_server is not None:
        _lt_server.stop()
        _lt_server = None