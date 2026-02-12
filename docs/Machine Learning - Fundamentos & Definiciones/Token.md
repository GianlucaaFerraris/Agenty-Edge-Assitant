## <font color="#de7802">1. ¿Qué es un Token? (Definición Atómica)</font>

Un modelo de lenguaje no "lee" palabras ni letras; lee números. El **Token** es la unidad mínima de procesamiento semántico. Un token puede ser una palabra completa (ej: "casa"), una parte de una palabra (ej: "ingeni" e "ería") o incluso un solo carácter o signo de puntuación.

## <font color="#de7802">2. El Proceso de Tokenización</font>

Para que nuestro modelo [QWen 2.5 7B](../LLM%20-%20Conceptos%20&%20Modelos/QWen%202.5%207B.md) entienda el texto, pasa por un _Tokenizer_ (usualmente **Byte Pair Encoding - BPE**):

1. **Segmentación:** Divide el texto según patrones estadísticos de frecuencia.
2. **ID Mapping:** Cada segmento se asocia a un número único en un vocabulario (ej: "ingeniero" = ID 4521).
3. **Embeddings:** Ese ID se convierte en un vector de alta dimensión (ej: 4096 dimensiones) que representa su significado en un espacio vectorial.

## <font color="#de7802">3. ¿Por qué se usan en ML?</font>

- **Compresión de Vocabulario:** Permite que el modelo entienda millones de palabras usando solo un vocabulario de ~150,000 tokens mediante la combinación de sub-palabras.
- **Manejo de Idiomas:** Facilita que el modelo aprenda raíces lingüísticas comunes (prefijos y sufijos).
- **Eficiencia de Cómputo:** Los tokens estandarizan la entrada. El modelo siempre recibe una matriz de dimensiones fijas, lo que facilita el procesamiento paralelo en la NPU.

## <font color="#de7802">4. Relevancia en el Cálculo de Performance</font>

En nuestro proyecto, medimos el rendimiento en **Tokens/segundo** porque es la métrica real de "velocidad de pensamiento" de la IA. Un token equivale aproximadamente a **0.75 palabras** en inglés. Si nuestra Rock 5B entrega 6 tokens/s, está entregando unas 4.5 palabras por segundo, lo cual es la velocidad ideal de una locución humana clara.