## <font color="#de7802">1. Definición General</font>

La **latencia** es el tiempo transcurrido entre el estímulo (entrada del usuario) y la respuesta (salida del sistema). En un sistema de voz interactivo como el nuestro, la latencia total es la suma de los retrasos de cada componente:

$$\text{Latencia Total} = L_{STT} + L_{Inferencia} + L_{TTS} + L_{Hardware}$$

---
## <font color="#de7802">2. Desglose de Componentes</font>

1. **Latencia STT (Speech-to-Text):** Tiempo que tarda Faster-Whisper en transcribir el audio a texto. Depende de la longitud del audio y de la potencia del procesador.
2. **Latencia de Red (Opcional):** En este proyecto es **0 ms**, ya que al ser Edge AI, no hay llamadas a nubes externas.
3. **Latencia TTS (Text-to-Speech):** Tiempo que tarda Piper en convertir la respuesta del LLM en ondas sonoras.
4. **Latencia de E/S (I/O):** Retraso de los drivers de audio y el bus del sistema.

---
## <font color="#de7802">3. Implicación en el Proyecto</font>

El éxito de un tutor de inglés depende de la **Latencia Percibida**. 
* Si la latencia total supera los **2 segundos**, el usuario siente una interrupción incómoda.
* **Optimización:** Implementaremos **Streaming**. El TTS empezará a hablar apenas el LLM genere las primeras palabras, "escondiendo" el resto de la latencia de inferencia mientras el audio ya suena.