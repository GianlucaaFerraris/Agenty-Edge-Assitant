
#### <font color="#de7802">Fase 1: Auditoría de Hardware y Benchmarking (Semanas 1-2)</font>

- [ ] **Análisis de cómputo**: Estudio de los TFLOPS y ancho de banda de memoria en tokens/segundo de la Rock Model 5B de Radxa (NPU/GPU) o Orin Jetson Nano de Nvidia.
    
- [ ] **Configuración de entorno**: Instalación de sistemas operativos optimizados (Ubuntu Rockchip con Kernel 6.1+). Configuración de herramientas de aceleración: Rockchip MPP (Media Process Platform) para video, RKNN-Toolkit 2 para la NPU, o TensorRT en caso de utilizar la Jetson.
    
- [ ] **Selección de Modelo Base**: Selección de modelo LLM base desde Hugging Face Llama 3.1 o Qwen 2.5. Pruebas de latencia y perplejidad utilizando Safe Tensors para el procesamiento en la computadora y GGUF para inferencia en la CPU/GPU de la placa que va a hostear el LLM.
    

---
#### <font color="#de7802">Fase 2: Entrenamiento con ecosistema Hugging Face (Semanas 3-5)</font>

- [ ] **Fine-Tuning Técnico con PEFT y QLoRA**: Entrenamiento mediante QLoRA en la RTX 4050 mediante las librerías transformers, peft y bitsandbytes de HF para realizar fine-tunning (4-bits). El entrenamiento se especializa en:
	- <u>Strict Instruction Following</u>: Mejora en el uso de herramientas (Function Calling) mediante datasets en formato Alpaca/Share GPT.
	- <u>ESL Coaching Layer</u>: Entrenamiento para la detección de errores gramaticales y sugerencias de corrección en tiempo real.**

- [ ] **Arquitectura RAG (Retrieval-Augmented Generation)**: implementación de un vector DB local (ChromaDB o FAISS). Integración con la librería sentence-transformers para gestionar la memoria semántica persistente del asistente, optimizando la ventana de contexto sin saturar la RAM.
    
- [ ] **Lógica de Estados**: Diseño de una máquina de estados finita (FMS) que orqueste los modos de operación (conversación casual, práctica de idioma, búsqueda técnica o gestión de tareas).
    
---
#### <font color="#de7802">Fase 3: Interfaz de Usuario y Conectividad Agéntica (Semanas 6-8)</font>

- [ ] **Pipeline de Audio de Baja Latencia**: Integración de Faster-Whisper (STT) y Piper (TTS) con manejo de un sistema asincrónico en Python para procesar el streaming de tokens del modelo y enviarlos al motor de forma iterativa.
    
- [ ] **Desarrollo de Agentes**: Configuración de herramientas mediante MCP (Model Context Protocol) u OpenClaw para conexión a Google Calendar, APIs de búsqueda y mensajería (WhatsApp/Telegram).
    
---
#### <font color="#de7802">Fase 4: Integración y Despliegue Híbrido en el Edge (Semanas 9-11)</font>

- [ ] **Optimización de Inferencia (SBC)**: Conversión de los adaptadores LoRA entrenados en la Fase 2 a formato GGUF compatible. Despliegue mediante llama-cpp-python para aprovechar la aceleración de hardware de la Rock 5B.
    
- [ ] **Eficiencia Energética y Térmica**: Desarrollo de un script de monitoreo que gestione el escalado de frecuencia de los núcleos Cortex-A76/A55, evitando el thermal throttling durante el procesamiento intensivo de lenguaje natural.
    
- [ ] **Interacción Física**: Control de periféricos vía I2C/SPI para una pantalla OLED (feedback visual de estados). Implementación de detección de Wake Word local para activación por voz "hands-free".
    
---
#### <font color="#de7802">Fase 5: Pruebas, Documentación y Visión (Semanas 12+)</font>

- [ ] **Visión Computacional y Percepción**: Integración de modelos de visión ligeros (VLM) como Moondream 2 para tareas de descripción de objetos y lectura de documentos vía cámara USB.
    
- [ ] **Diseño y Fabricación de Carcasa (CAD/3D Printing)**:
	- Diseño en Fusion 360 de un chasis integrado para la Rock 5B, SSD NVMe, pantalla OLED, altavoces y micrófono.
	- Modelado de conductos de aire para ventilación activa.
	- Impresión 3D en materiales de ingeniería (PLA/PETG).

- [ ] **QA y Documentación Final**: Medición de métricas clave (TTFT, tokens por segundo, tasa de error en correcciones). Redacción del manual técnico y memoria del proyecto.
    
