# Edge AI Assistant: Multi-SBC Agent for Learning & Other Tasks

Este proyecto consiste en el desarrollo e implementación de un asistente inteligente de respuesta agéntica, diseñado para funcionar íntegramente en el "Edge" (computación en el borde). El sistema actúa como un tutor de inglés (ESL) e ingeniería capaz de corregir gramática, mantener conversaciones fluidas y gestionar tareas locales, priorizando la privacidad y la baja latencia. Además, cuenta con otras herramientas de agencia como hablar a través de Whatsapp, navegar en la web, etc.

## 🚀 Visión General
A diferencia de los asistentes basados en la nube, este agente utiliza hardware de alto rendimiento como la **Radxa Rock 5B (RK3588)** o la **NVIDIA Jetson Orin Nano**. El enfoque principal es la optimización de modelos de lenguaje (LLMs) mediante técnicas de cuantización y fine-tuning para superar las limitaciones físicas de los dispositivos embebidos.

## 🛠️ Stack Tecnológico
- **Modelos:** Qwen 2.5 (7B), Llama 3.1 (8B) - Formatos GGUF / EXL2.
- **Hardware:** Radxa Rock 5B (NPU 6 TOPS) | NVIDIA Jetson Orin Nano.
- **Inferencia:** Llama.cpp / RKNN-Toolkit2.
- **Audio Multimodal:** Faster-Whisper (STT) y Piper (TTS).
- **Orquestación:** Agentes basados en Python con soporte para MCP (Model Context Protocol).
- **Documentación:** Sistema de gestión de conocimiento en Obsidian (Markdown + LaTeX).

## 📊 Arquitectura y Fundamentos Técnicos
El proyecto se fundamenta en un análisis profundo de la arquitectura de computadores aplicada a la IA:

* **Memory Bound Inferencia:** Optimización basada en el ancho de banda de memoria (LPDDR4x vs LPDDR5) para maximizar los tokens por segundo.
* **Compute Performance:** Evaluación de capacidad mediante TFLOPS (FP16) y TOPS (INT8).
* **Edge AI Strategy:** Implementación de cuantización de 4-bits para reducir la carga en el bus de datos y evitar el "Memory Wall".

## 📋 Plan de Trabajo (Hitos)
1.  **Fase 1: Auditoría de Hardware:** Benchmarking de latencia (TTFT/TPOT) y capacidad de cómputo.
2.  **Fase 2: Fine-Tuning y RAG:** Ajuste fino con QLoRA para especialización en corrección lingüística y memoria semántica local.
3.  **Fase 3: Pipeline Multimodal:** Integración de STT/TTS de baja latencia con streaming de audio.
4.  **Fase 4: Despliegue en el Borde:** Optimización final de inferencia y gestión térmica del SoC.
5.  **Fase 5: Diseño Industrial:** Integración física, control de periféricos (OLED/VLM) y chasis en impresión 3D.

## 📂 Estructura del Repositorio
- `/docs`: Documentación técnica detallada (Notas de Obsidian).
- `/src`: Código fuente del agente y el pipeline de audio.
- `/benchmarks`: Logs de rendimiento y comparativas de hardware.
- `/models`: Scripts de conversión y cuantización.

## 🧠 Documentación en Obsidian
Este repositorio está diseñado para ser navegado como una bóveda de Obsidian. Los archivos en la carpeta `/docs` contienen explicaciones explayadas.

---
**Autor:** Gianluca Ferraris
**Institución:** Fundación Fulgor - Tarpuy
**Estado:** En Desarrollo - Fase 1
