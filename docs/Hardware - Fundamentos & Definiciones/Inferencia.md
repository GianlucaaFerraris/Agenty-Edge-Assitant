## <font color="#de7802">1. Definición Técnica</font>

La **inferencia** es la fase de ejecución de un modelo de IA ya entrenado. A diferencia del entrenamiento, donde los pesos del modelo se ajustan mediante [[Retropropagación]], en la inferencia el modelo es de "solo lectura" y utiliza sus parámetros aprendidos para predecir el siguiente [[Token]] basado en una entrada ([[Prompt]]).

En términos matemáticos, la inferencia en un LLM es una serie masiva de operaciones de **Álgebra Lineal**, específicamente multiplicaciones de matrices y vectores:
$$y = \text{softmax}(W \cdot x + b)$$

## <font color="#de7802">2. El Proceso de Generación (Autoregresión)</font>

La inferencia en LLMs es **autoregresiva**. Esto significa que para generar una respuesta de 10 palabras, el modelo debe realizar 10 pasadas completas por toda la red neuronal. Cada palabra generada se añade al final de la secuencia y sirve como entrada para la siguiente iteración.

## <font color="#de7802">3. Implicación en el Proyecto</font>

Para nuestro agente de IA, la eficiencia de la inferencia determina la "naturalidad" de la charla.
* **Carga Computacional:** Generar una respuesta larga requiere billones de operaciones de coma flotante (FLOPs).
* **Estrategia en Edge AI:** Dado que el hardware es limitado, utilizaremos **Inferencia Cuantizada**. Al reducir la precisión de los pesos de FP16 a INT4, disminuimos la carga de cómputo y el consumo de RAM, permitiendo que el modelo [[QWen 2.5 7B]] sea viable.

## <font color="#de7802">Recursos para comprender mejor el tema</font>

[What is AI Inference by IBM](https://youtu.be/XtT5i0ZeHHE?si=VGFndbPnUy3f5wpM)
