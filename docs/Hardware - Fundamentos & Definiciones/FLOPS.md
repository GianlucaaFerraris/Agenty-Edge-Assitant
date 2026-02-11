## <font color="#de7802">1. Definición FLOPS vs FLOPs</font>

**FLOPS** o *Floating Point Operations per Second* es una métrica estandar usada para medir la capacidad computacional de un procesador (fijarse que es **FLOPS** con *S* mayuscula por *Second*). Cuenta cuantos calculos puede realizar una computadora en cierto tiempo en segundos. Mientras mayor sea el **FLOPS** del computador mas adaptado a correr NN mas complejas va a estar.   

![[Pasted image 20260210171240.png]]
Es el calculo para obtener el numero de FLOPS de un procesador, calcular FLOPs es algo mas complejo.

Por otro lado, la complejidad computacional de una [[Red Neuronal]] se mide por **FLOPs** (esta es **FLOPs** con *s* minúscula por simplemente *Floating Point Operations* en plural). Cuenta específicamente el numero de calculos matemáticos  (el numero de sumas y multiplicaciones) que una NN (Neural Network) debe realizar para procesar una entrada, en nuestro caso un [[Token]]. Por esto mismo, es el standar dentro de los modelos de [[Deep Learning]] para poder determinar que tan "pesado" para el hardware un modelo es. Mientras mayor sea la métrica de FLOPs, mas energía y costo computacional va a requerir el modelo para ser ejecutado. Por otro lado, mientras 

## <font color="#de7802">2. Factores relevantes para FLOPS y FLOPs</font>

Los factores que afectan a la métrica son:
- **Arquitectura del procesador**: el diseño del CPU o GPU determina que tan eficientemente se realizan las operaciones de punto flotante. Nuevos procesadores incluso cuentan con una unidad llamada Unidad de Procesos Flotantes FPUs para encargarse de esta tarea y poder tener un FLOPS mayor.
- **Clock Speed**: mientras mayor sea la velocidad del clock del procesador, mayor seran la cantidad de operaciones por segundo que este pueda realizar y por lo tanto mayor el FLOPS.
- **Paralelismo**: Si nuestro procesador cuenta con varios núcleos puede ejecutar mas operaciones en simultaneo; por esto las GPUs tienen mejor [[Benchmark]] que otras unidades de proceso no dedicadas a esto como las CPUs.
- [[Ancho de Banda de Memoria]]: Si nuestra data no puede ser suministrada al procesador rapidamente tambien va a afectar la performance, siendo un cuello de botella del flujo de datos y por lo tanto menor el numero de nuestra métrica FLOPS.
- **Eficiencia Algorítmica**: si nuestra red neuronal está bien construida va a necesitar de menos calculos para poder procesar un token y por lo tanto menor su FLOPs (mas eficiente para correr localmente).

## <font color="#de7802">3. Otros Datos</font>

El benchmark utilizado academicamente para poder obtener el numero de FLOPS se lo llama **LINPACK**, el cual resuelve un sistema denso de ecuaciones lineales. Generalmente el resultado está en **TFLOPS** por la magnitud del numero en **teras** que es *10E12*). 

La métrica FLOPs nos permite saber de manera independiente al hardware la [[Latencia de Inferencia]] del modelo. De esta forma, podemos escoger sabiamente el [[LLM]] o red neuronal que mas nos convenga. Ya sea para hacer computacion en la nube, o en nuestro caso [[Edge AI]], el cual requiere FLOPs mas pequeños porque la métrica FLOPS de nuestras placas son menores. 