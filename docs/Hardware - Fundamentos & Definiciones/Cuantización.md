## <font color="#de7802">1. Fundamentos: De la Precisión a la Eficiencia</font>

La cuantización es el proceso de reducir la precisión de los números que representan los pesos (_weights_) del modelo. Originalmente, los modelos se entrenan en **FP32** (32 bits por número). Sin embargo, esto es inviable para [Edge AI](../Machine%20Learning%20-%20Fundamentos%20&%20Definiciones/Edge%20AI.md).

## <font color="#de7802">2. Niveles de Cuantización</font>

- **FP16/BF16 (16 bits):** El estándar de oro para entrenamiento. Alta fidelidad, pero ocupa mucha memoria (2 Bytes por parámetro).
- **INT8 (8 bits):** Reduce el peso a la mitad (1 Byte por parámetro). Es el "punto dulce" nativo para la NPU del RK3588 (6 TOPS).
- **INT4 / Q4 (4 bits):** Reduce el peso a ~0,7 Bytes por parámetro. Permite meter un modelo de 7B (7 mil millones de parámetros) en solo 4,8 GB de RAM.

## <font color="#de7802">3. El Proceso Matemático (Mapping)</font>

Cuantizar no es simplemente "borrar decimales". Se utiliza un factor de escala ($S$) y un punto cero ($Z$) para mapear un rango de valores flotantes a un rango entero:

$$W_{int} = \text{round}\left(\frac{W_{float}}{S} + Z\right)$$

Esto intenta preservar la distribución estadística de los pesos para que el modelo no pierda su "inteligencia" en pocas palabras.

## <font color="#de7802">4. Tipos de Cuantización en el Proyecto (RKLLM)</font>

Nuestro stack tecnológico utiliza principalmente:

- **w8a8:** Pesos en 8 bits, Activaciones en 8 bits. Máxima precisión para el Tutor de Ingeniería.
- **w4a16:** Pesos en 4 bits, Activaciones en 16 bits. Esta técnica guarda los datos en 4 bits para que viajen rápido por la RAM, pero cuando la NPU los procesa, los eleva a 16 bits para que el cálculo matemático sea más preciso.

## <font color="#de7802">5. Ventajas y Degradación</font>

1. **Velocidad:** Al reducir los bits, reducimos la cantidad de datos que deben cruzar el bus de memoria (el cuello de botella). Por lo tanto la velocidad de transporte de datos aumenta.
2. **Eficiencia Energética:** Mover menos bits consume menos energía.
3. **Pérdida de Calidad** ([Perplexity](../Machine%20Learning%20-%20Fundamentos%20&%20Definiciones/Perplexity.md)): Cuantos menos bits usamos, más "ruido" introducimos. Un modelo INT4 puede empezar a confundir conceptos técnicos muy específicos que un modelo FP16 entendería perfectamente.