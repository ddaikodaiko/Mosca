# Cuándo debería importar la topología del cableado y cómo comprobarlo con justicia

Notas de investigación (octubre de 2026). Estado de cada fuente: **[PR]** revisada por pares, **[PP]** preprint, **[clásico]** anterior a 2016 o de fuera de la ventana, **[2ª]** dato tomado de una fuente secundaria. Los números son los que da la fuente; si no los he podido comprobar en el texto original, lo digo.

Contexto del proyecto (de los registros del proyecto, no es una fuente externa): Mosca entrena **pesos por arista dentro de una máscara** (ley de Dale, N = 500-4.000, puertos elegidos por grado con un solape fijo de 92) contra estos controles: ER, recableado que conserva grados, denso con los mismos parámetros, `sin_nucleo` y MLP. Empata en todo, y el núcleo apenas le gana a `sin_nucleo`.

---

## 1. Pesos entrenables frente a pesos fijos: ¿cuánto importa la máscara concreta y cuánto la densidad o la distribución de grados?

### Takeaway
Cuando los pesos se entrenan desde cero dentro de una máscara dispersa, la literatura de redes dispersas y de cableado aleatorio coincide: importan la densidad, el reparto de parámetros entre capas y las estadísticas globales del grafo (generador, agrupamiento, longitud de camino), pero apenas la máscara concreta. Las ventajas del conectoma aparecen casi solo donde los pesos de las aristas están **fijos** (reservorios, FlyGM), y aun ahí a menudo vienen de los pesos o de la normalización espectral más que de la topología. Mosca está en el régimen donde lo esperable de antemano es un empate.

### Cited Findings
**Redes dispersas entrenadas desde la inicialización (análogo directo de Mosca)**
- Poda en la inicialización [PR, ICLR 2021]. Con SNIP, GraSP, SynFlow y poda por magnitud aplicadas al inicio, la exactitud se mantiene o **mejora** si la máscara se baraja al azar dentro de cada capa o si se reinicializan los pesos. La poda por magnitud *después* de entrenar sí pierde exactitud con las mismas manipulaciones. Conclusión de los autores: solo cuenta la fracción podada por capa, no qué pesos concretos se conservan — [Frankle et al. 2021](https://arxiv.org/abs/2009.08576v2).
- La explicación propuesta es que barajar o reinicializar apenas cambia la distribución de los pesos no podados de cada capa. El barajado conserva el número de pesos por capa — [Singh & Liu 2021, PP](https://arxiv.org/pdf/2107.01808).
- Poda aleatoria en la inicialización seguida de entrenamiento disperso desde cero [PR, ICLR 2022]. Alcanza la exactitud de la red densa, y una subred aleatoria de Wide ResNet-50 llega a superar a la densa en ImageNet. Los dos factores decisivos son el tamaño de la red y las razones de dispersión por capa elegidas de antemano, no el patrón — [Liu et al. 2022](https://arxiv.org/abs/2202.02643v1).
- Aprendizaje por refuerzo profundo [PR, ICML 2022]:
  - Con el mismo número de parámetros, las redes dispersas superan a las densas.
  - Se entrena con un 80-90 % de dispersión casi sin pérdida.
  - El crecimiento de conexiones guiado por gradiente (RigL) apenas aporta frente al crecimiento aleatorio (SET), por la baja relación señal/ruido de los gradientes en RL.
  - Lo que sí importa es repartir los parámetros: dar la mayoría al crítico y usar la distribución ERK de dispersión por capa.
  - [Graesser et al. 2022](https://proceedings.mlr.press/v162/graesser22a.html)

**Cableado aleatorio (pesos entrenables y grafo fijo)**
- RandWire, en ImageNet [PR, ICCV 2019]:
  - La desviación típica entre 5 instancias del mismo generador es de **0,2-0,4 %**, frente al 0,1-0,2 % de reentrenar la misma ResNet-50.
  - Entre generadores sí hay saltos claros: WS(4, 0,75) da 73,8 % de media y BA(1) 70,7 %, unos 3 puntos menos.
  - WS sin recablear (P = 0, regular) es peor que su versión recableada para cualquier K.
  - Conclusión de los autores: importa el diseño del generador, no la instancia concreta.
  - [Xie et al. 2019](https://ar5iv.labs.arxiv.org/html/1904.01569)
- Grafos relacionales con la estructura fija por enmascaramiento y pesos entrenables [PR, ICML 2020]:
  - El rendimiento es una función suave (en U) del coeficiente de agrupamiento C y de la longitud media de camino L, con un punto óptimo en C ∈ [0,43; 0,50] y L ∈ [1,82; 2,28].
  - En CIFAR-10, el mejor grafo supera al grafo completo por 1,4 puntos (error 32,05 ± 0,14 frente a 33,34 ± 0,36). En ImageNet, por 0,5-1,2 puntos.
  - Se evaluaron 3.942 grafos con 5 semillas cada uno.
  - Los grafos biológicos caen cerca del óptimo pero **no superan al mejor sintético**: corteza de gato 33,01 ± 0,22; corteza visual de macaco 32,78 ± 0,21; corteza completa de macaco 32,77 ± 0,14.
  - [You et al. 2020](https://arxiv.org/html/2007.06559v2)
- Capacidad de las RNN [PR, ICLR 2017]. Almacenan unos **5 bits por parámetro** y las arquitecturas comunes llegan a cotas de capacidad casi idénticas cuando se entrenan bien. Muchas diferencias publicadas entre arquitecturas son de entrenabilidad, no de capacidad. Cada unidad oculta guarda aproximadamente un número real del historial de entrada — [Collins, Sohl-Dickstein & Sussillo 2017](https://www.arxiv.org/abs/1611.09913).
- Universalidad [PR, NeurIPS 2019]. En miles de RNN de arquitecturas distintas entrenadas en tareas de neurociencia, la **topología de puntos fijos** y la dinámica linealizada son universales; la geometría de las representaciones varía según la arquitectura — [Maheswaranathan et al. 2019](https://arxiv.org/abs/1907.08549).

**Conectomas como reservorio (pesos recurrentes fijos, solo se entrena la lectura)**
- Conectomas de macaco, tití y humano en redes de estado de eco [PR, PLOS CB 2022]:
  - Topología empírica con pesos aleatorios, `Bio (no-rank)`: **rinde igual** que los controles aleatorios.
  - Topología empírica con pesos ordenados según la fuerza del conectoma, `Bio (rank)`: es la **única condición claramente peor** en capacidad de memoria y en recuerdo de secuencias, en los tres conectomas.
  - La diferencia la causa la colocación de los pesos, no la topología.
  - [Damicelli, Hilgetag & Goulas 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9710781/)
- Conectomas humanos de difusión como reservorio [PR, Nat Mach Intell 2021]. Frente a dos modelos nulos, la topología y la organización modular aumentan la capacidad de memoria **solo con dinámica crítica**; la misma arquitectura da capacidades muy distintas según el régimen dinámico — [Suárez et al. 2021](https://colab.ws/articles/10.1038/s42256-021-00376-1?locale=en); [preprint](https://www.biorxiv.org/content/10.1101/2020.11.10.350876.full.pdf). No pude leer las cifras: bioRxiv devolvió 429 y Nature pide autenticación.
- Red de estado de eco con FlyWire [PR, MDPI Biomimetics 2025]. Más resistente al sobreajuste que una estándar. Usa controles híbridos (topología del conectoma con pesos aleatorios, y pesos del conectoma con topología aleatoria) que sugieren que contribuyen las dos cosas — [Costi et al. 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12109256/). Un revisor objetó que el radio espectral, «parámetro crítico» en reservorios, no estaba bien controlado — [informe de revisión](https://www.mdpi.com/2313-7673/10/5/341/review_report).
- «Fruit Fly ESN» [PP, 2022]. Mejor o con menos varianza que una red de estado de eco estándar, pero sin controles de grado ni de radio espectral que yo haya podido ver — [Morra & Daley 2022](https://arxiv.org/abs/2201.09359).
- Seis conectomas como reservorio [PP, junio 2026]. La inicialización aleatoria sobre la **misma topología** rinde de forma fiable por debajo de los pesos biológicos. Los autores lo leen como que los pesos son un sesgo inductivo que la topología sola no recupera. No hay nulos de topología — [Guragain et al. 2026](https://arxiv.org/abs/2606.09902).
- C. elegans, 22 mapas de conectoma, pesos congelados y ρ = 1 [PP, septiembre 2026; cifras 2ª].
  - El cableado real solo gana en **capacidad de memoria**, y lo hace contra los tres controles (coeficientes β: +0,305 al recablear, +0,657 al barajar la entrada/salida, +0,951 con ambos).
  - En la tarea de Hénon pierde (−1,342 / −1,086 / −1,426).
  - En decisión perceptual no hay diferencias significativas.
  - El signo depende de cómo se midan los pesos: con el mapa de área de contacto, los aleatorios ganan en los 7 puntos de edad.
  - También depende de la métrica: con RMSE el original no gana en ningún punto.
  - [Reimers et al. 2026 vía Pebblous](https://blog.pebblous.ai/blog/celegans-connectome-reservoir-benchmark/en/) (primario: arXiv 2609.30508, sin comprobar).

**Los dos trabajos sobre la mosca que el proyecto contrasta**
- FlyGM [PP, v3 del 14-06-2026]:
  - Pesos sinápticos **fijos** con signo: W = N_exc − N_inh. Se entrenan un descriptor por neurona, un codificador, una compuerta aferente, una MLP de actualización compartida, el decodificador y el crítico.
  - Error angular (°): FlyGM 4,96 ± 0,09 frente a 7,84 ± 0,08 del recableado que conserva grados (DP) y 12,11 ± 0,23 de ER, en la condición fácil. En la difícil (v = 3, ψ = 7): 8,29 ± 0,21, 13,55 ± 0,69 y 125,36 ± 8,96.
  - El MLP 4×512, con más parámetros, se queda en 6,76 en la fácil y 13,90 en la difícil.
  - «Varias ejecuciones» sin número concreto, sin pruebas estadísticas y sin contar parámetros.
  - [Jin et al. 2026](https://arxiv.org/html/2602.17997v3)
- Dhiman [PP, abril 2026], sobre flyvis:
  - 45.669 nodos y 1.513.231 aristas. Solo hay **734 parámetros entrenables**, compartidos por tipo celular, más 2.959 fijos.
  - Con controles débiles (checkpoint entrenado sobre el conectoma y nulo ingenuo), Δpérdida = 0,184 a los 5 pasos.
  - Con inicialización aleatoria compartida, Δ = −0,002. Con el nulo que conserva grados, Δ = 0,0003 (5 pasos) y −0,0018 (10 pasos).
  - Conjunto de 5 nulos DP × 3 semillas: 0,5155 ± 0,0067 (conectoma) frente a 0,5172 ± 0,0061 (control).
  - [Dhiman 2026](https://arxiv.org/html/2604.04033v1)

### Inferences
- Hay **tres regímenes**, y la literatura solo encuentra ventajas fiables en los dos primeros:
  - **(a)** grafo y pesos de arista fijos, entrenando solo la lectura (reservorio);
  - **(b)** grafo y magnitudes fijos, entrenando parámetros por neurona o por tipo celular (Lappalainen, Beiran, Duan; FlyGM se parece a este);
  - **(c)** grafo fijo con **pesos por arista entrenables** (Mosca, RandWire, poda en la inicialización).
- En (c), el optimizador redistribuye la magnitud entre las aristas disponibles. Según Frankle 2021, Liu 2022 y Xie 2019, la máscara concreta se vuelve intercambiable con otra de la misma densidad y la misma distribución de grados. El empate mosca = aleatorio = recableado de Mosca es justo lo que predice esta literatura, no una anomalía.
- Cuenta de grados de libertad. En (c), Mosca tiene unos 160 k-185 k pesos libres sobre la máscara. En (b), el conectoma deja unos 2N parámetros libres (unos 8.000 para N = 4.000; Beiran lo formula como 2N frente a N²). La restricción que impone el cableado es mucho mayor en (b).
- Las ventajas que sobreviven en (a) suelen venir de los **pesos** y no del soporte:
  - en Damicelli, la topología con pesos aleatorios empata con el aleatorio;
  - en Guragain, los pesos biológicos son esenciales;
  - en la ablación de FlyGM sin pesos, el error pasa de 8,29 a 11,00 en la condición difícil (aunque en la fácil la versión sin pesos va ligeramente mejor: 4,91 contra 4,96).
- Implicación para un rediseño. Si Mosca quiere resolución, el cambio más fundamentado es pasar al régimen (b):
  - fijar |w| proporcional al número de sinapsis de FlyWire y fijar el signo;
  - entrenar solo ganancias, sesgos y constantes de tiempo por neurona, o escalas por par de tipos celulares;
  - comparar contra un DP que **arrastre el peso con la arista** y contra un barajado de pesos sobre la misma topología. Así se separa «soporte» de «pesos».
- FlyGM no es una RNN escalar. Cada neurona tiene un estado de 32 dimensiones y hace 4 rondas de paso de mensajes con una MLP compartida. Con profundidad fija, el grafo decide **por dónde y en cuántos saltos** llega la información de los aferentes a los eferentes, y eso no se compensa entrenando aristas porque no se entrenan. El colapso de ER (125°) apunta a un fallo de enrutado entre los puertos. Es una hipótesis mía, no está medida en el paper.

### Gaps
- No encontré ningún estudio que entrene **pesos por arista** dentro de una máscara de FlyWire completa con un nulo DP en una tarea que exija memoria. Es el diseño exacto de Mosca, y parece que nadie más lo ha publicado.
- FlyGM no da el número de semillas, ni los parámetros por modelo, ni pruebas estadísticas.
- No hay cifras de Suárez 2021 ni de conn2res 2024 (Nat Commun) sobre la diferencia empírico-contra-nulo por régimen α: no pude acceder al texto.
- Reimers 2026 solo está verificado por una fuente secundaria.

---

## 2. Modelos nulos: ER, conservación de grados (Maslov–Sneppen), nulos espaciales y barajado de pesos. ¿Qué propiedades más allá del grado importan? ¿Hace falta igualar el radio espectral?

### Takeaway
ER es un nulo débil: confunde el «cableado concreto» con la heterogeneidad de grados, y esta cambia el radio espectral. El estándar mínimo es el DP con varias instancias y bien mezclado. Por encima hay nulos que conservan además la reciprocidad, el signo por neurona, el tipo celular o la geometría. Igualar o barrer el radio espectral no es opcional: tanto con pesos fijos como en la inicialización de redes entrenables, parte de las «ventajas de arquitectura» publicadas viene del reescalado espectral.

### Cited Findings
- Revisión de referencia [PR, Nat Rev Neurosci 2022]. Taxonomía de nulos, de «liberales» (controlan pocas propiedades) a «conservadores» (reproducen muchas), por aleatorización o por modelos generativos, con nulos específicos para redes espacialmente incrustadas y anotadas — [Váša & Mišić 2022](https://kclpure.kcl.ac.uk/portal/en/publications/null-models-in-network-neuroscience/).
- El recableado por intercambio de aristas que conserva grados procede de [Maslov & Sneppen 2002](https://www.science.org/doi/10.1126/science.1065103) [clásico; no releído en esta sesión].
- Escalera de controles de Dhiman [PP 2026]:
  - El nulo ingenuo, con el mismo número de nodos, aristas y autobucles, produce una «ventaja de actividad» espuria (Δactividad = 1,2056) que desaparece con el DP (−0,0106).
  - El DP usa intercambios dobles dirigidos y conserva exactamente los grados de entrada y de salida; se hicieron **5 grafos independientes con 250.000 intercambios aceptados cada uno (≈ 0,17 por arista)** y una tasa de aceptación de 0,992.
  - El autor recomienda igualar el radio espectral y probar nulos estructurados (mundo pequeño, anillo).
  - [Dhiman 2026](https://arxiv.org/html/2604.04033v1)
- En FlyGM [PP], el orden es conectoma < DP < ER en las cuatro condiciones, y la distancia al DP crece con la dificultad: 2,88° en la fácil y 5,26° en la difícil — [Jin et al. 2026](https://arxiv.org/html/2602.17997v3).
- Además del grado, la longitud de camino y el agrupamiento predicen el rendimiento con pesos entrenables, aunque con efectos de 0,5-1,4 puntos — [You et al. 2020](https://arxiv.org/html/2007.06559v2). El recableado de WS (P > 0) mejora frente al anillo regular — [Xie et al. 2019](https://ar5iv.labs.arxiv.org/html/1904.01569).
- La modularidad importa en reservorios, pero solo en el régimen crítico — [Suárez et al. 2021](https://colab.ws/articles/10.1038/s42256-021-00376-1?locale=en).
- Radio espectral y heterogeneidad de grados:
  - En grafos aleatorios con grados esperados dados, el autovalor máximo de la adyacencia es casi seguro (1 + o(1))·max{d̃, √m}, con d̃ el grado medio de segundo orden (Σd²/Σd) y m el grado máximo. Una cola de grados pesada **sube** λ_max con la misma densidad — [Chung, Lu & Vu 2003](https://pmc.ncbi.nlm.nih.gov/articles/PMC164443) [clásico].
  - λ_max gobierna varios procesos dinámicos en redes — [Restrepo, Ott & Hunt 2007](https://arxiv.org/abs/0705.4503) [clásico].
- Reservorios de conectoma [PP, septiembre 2026]. Hay un compromiso rendimiento-robustez que sigue al **radio espectral bruto antes de normalizar**: al reescalar todas las matrices al mismo radio objetivo, las de radio bruto menor reciben más amplificación global. Parte de las diferencias entre arquitecturas es un efecto del reescalado, no de la estructura — [Churchland et al. 2026](https://arxiv.org/abs/2609.07355).
- Con pesos entrenables y la ley de Dale, el **espectro de valores singulares en la inicialización** influye más en el rendimiento que las restricciones de signo — [Li et al. 2023](https://nips.cc/virtual/2023/poster/70321) [PR].
- Barajar los nodos de entrada/salida es un nulo aparte, y puede pesar más que barajar el cableado: +0,657 frente a +0,305 en capacidad de memoria — [Reimers et al. 2026, 2ª](https://blog.pebblous.ai/blog/celegans-connectome-reservoir-benchmark/en/).
- Penalizar solo la longitud de las conexiones en 3D basta para que una RNN entrenada desarrolle redes modulares de mundo pequeño parecidas a las corticales [PR, Nat Mach Intell 2023]. Parte de lo que parece «diseño» puede ser un subproducto de la geometría — [Achterberg et al. 2023](https://www.repository.cam.ac.uk/handle/1810/362473).

### Inferences
- **El DP de Dhiman estaba poco mezclado.** Con 250.000 intercambios sobre unos 1,5 M de aristas no autobucle, como mucho cambia ≈ 33 % de las aristas, así que al menos ≈ 67 % del cableado original sigue igual. Su «empate bajo DP» es en parte esperable, y su nulo no es independiente. Es un cálculo mío a partir de sus cifras; los autores no lo comentan. En Mosca hay que comprobar cuántos intercambios por arista hace `connectome/controls.py`. La práctica habitual con Maslov–Sneppen es del orden de 10 o más por arista; la cifra no la he verificado en esta sesión.
- La mosca tiene grado máximo 710 frente a 123 de ER con la misma densidad, así que con la **misma escala de pesos** en la inicialización su radio espectral será mayor (Chung-Lu-Vu). Hasta que no se iguale la ganancia efectiva, cualquier diferencia mosca-ER es en parte una diferencia de régimen dinámico. El DP conserva los grados y, por tanto, aproximadamente d̃. Lo hace solo «aproximadamente» porque las correlaciones de grado sí cambian, y Restrepo indica que afectan a λ_max.
- Escalera de nulos que propongo para Mosca, de liberal a conservador (cada peldaño aísla una hipótesis):
  1. ER con la misma densidad y la misma fracción inhibidora (ya existe).
  2. DP dirigido bien mezclado, con varias instancias (ya existe; hay que verificar la mezcla).
  3. DP que conserve también el **signo por neurona presináptica**: intercambiar solo aristas con presinápticas del mismo signo. Así se conservan los grados excitador e inhibidor por nodo y la ley de Dale.
  4. DP que conserve la reciprocidad (los pares bidireccionales).
  5. Nulo espacial: recableado que conserva la distribución de longitudes, usando las coordenadas de FlyWire v783 que el proyecto ya tiene.
  6. Nulo de tipo celular: barajar las aristas dentro de cada par de tipos celulares. Prueba si importa la identidad de la neurona o solo la estadística entre tipos.
  7. Barajado de pesos sobre la topología real, solo si se pasa al régimen de pesos fijos.
- En todos los peldaños, normalizar el radio espectral (o la norma) en la inicialización, o mejor **barrer la ganancia** y comparar curvas, como hacen los reservorios con α. Además, registrar la entropía SVD normalizada de Li et al.

### Gaps
- No encontré cifras sobre cuánto cambia el rendimiento de una RNN *entrenada* al conservar motivos o reciprocidad además del grado.
- La regla práctica de «≥ 10 intercambios por arista» y la referencia de Rubinov 2016 sobre «spandrels» espaciales no se han verificado en esta sesión.
- No pude extraer las cifras de nulos de Suárez 2021 ni de conn2res.

---

## 3. Ley de Dale y estructura E/I: efecto sobre la entrenabilidad y el rendimiento

### Takeaway
Imponer Dale por columnas (ColEI), que es lo que hace un núcleo con signo fijo por neurona presináptica, degrada el aprendizaje sobre todo por el **espectro patológico de la matriz en la inicialización**, no por el signo en sí. Ese efecto puede tapar cualquier diferencia de topología. Hay recetas que lo corrigen: DANN, inicialización equilibrada y diagnóstico por entropía SVD.

### Cited Findings
- Marco estándar para entrenar RNN excitadoras-inhibidoras con Dale en tareas cognitivas — [Song, Yang & Wang 2016](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004792) [PR; no releído en esta sesión].
- Normalmente se prescinde de Dale porque imponerlo perjudica el aprendizaje. Las DANN (Dale's ANNs) aprenden **tanto como una red estándar** si se inicializan de modo que la inhibición centre y estandarice la actividad excitadora y si las actualizaciones inhibidoras se escalan con una corrección basada en la matriz de información de Fisher — [Cornford et al. 2021, ICLR](https://discovery-pp.ucl.ac.uk/id/eprint/10167375) [PR].
- La partición E/I simple (ColEI) da una distribución de valores singulares **multimodal y dispersa**; las RNN estándar y las DANN recurrentes la dan unimodal y compacta. El espectro en la inicialización influye **más que las restricciones de signo**, y la entropía SVD normalizada sigue al rendimiento — [Li, Cornford, Ghosh & Richards 2023, NeurIPS](https://www.biorxiv.org/content/10.1101/2023.06.28.546924.full.pdf) [PR].
- En reservorios de conectoma, los barridos de equilibrio E/I de las aristas muestran que el compromiso rendimiento-robustez sigue al radio espectral bruto — [Churchland et al. 2026](https://arxiv.org/abs/2609.07355) [PP].
- En el sistema visual de la mosca, la preferencia de contraste (FRI) solo se predice bien cuando se conocen los **signos** sinápticos medidos — [Lappalainen et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/) [PR].
- En FlyGM, los pesos con signo ayudan solo en la condición difícil (8,29 con pesos frente a 11,00 sin ellos) y no en la fácil (4,96 frente a 4,91) — [Jin et al. 2026](https://arxiv.org/html/2602.17997v3) [PP].

### Inferences
- En Mosca, la fracción inhibidora cambia entre subgrafos (0,36-0,50) y los grados son muy heterogéneos. Con Dale por columnas, el espectro inicial de la mosca, del recableado y de ER puede ser muy distinto aunque la densidad y la fracción inhibidora coincidan. Si uno de los seis modelos parte de un espectro peor, la comparación mide entrenabilidad, no cableado.
- Recomendación para el rediseño:
  - informar del radio espectral y de la entropía SVD normalizada de W en la inicialización para los seis modelos;
  - usar una inicialización equilibrada por neurona (cada fila con suma excitadora e inhibidora compensada) o al estilo DANN;
  - hacer un control «Dale apagado» en los seis por igual. Si el empate se mantiene sin Dale, el freno no es el signo.
- En el régimen de pesos fijos, los signos de FlyWire (`Excitatory`, constantes por presináptica según los registros del proyecto) son información biológica que sí entra en juego. Es otro argumento para el régimen (b).

### Gaps
- No verifiqué en esta sesión las cifras concretas de Li et al. 2023 sobre cómo cambia el rendimiento de ColEI con el tamaño de la red y la proporción E:I.
- No encontré estudios que crucen Dale con una máscara de conectoma real frente a nulos DP.

---

## 4. Control positivo y resolución del experimento: demostrar que la tarea necesita recurrencia o arquitectura antes de comparar cableados

### Takeaway
Antes de comparar cableados hay que demostrar dos cosas: que la tarea **necesita** al núcleo recurrente (núcleo ≫ `sin_nucleo`, con un margen muy por encima del ruido entre semillas) y que el método **detectaría** una ventaja de cableado si la hubiera, plantándola en un experimento maestro-alumno. Las tareas con demanda de memoria explícita (capacidad de memoria, n-bit flip-flop, emparejamiento con demora, integración de trayectoria) son donde aparecen los efectos de topología, también en reservorios.

### Cited Findings
- En C. elegans, el cableado real solo gana contra los tres controles en **capacidad de memoria**; en decisión perceptual no hay diferencias, y en Go/No-go la ventaja se invierte al barajar la entrada/salida — [Reimers et al. 2026, 2ª](https://blog.pebblous.ai/blog/celegans-connectome-reservoir-benchmark/en/).
- Las diferencias de conectoma en reservorios se miden con tareas de memoria: capacidad de memoria y recuerdo de secuencias de longitud L = 5-25 — [Damicelli et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9710781/). La modularidad solo cuenta en el régimen crítico — [Suárez et al. 2021](https://colab.ws/articles/10.1038/s42256-021-00376-1?locale=en).
- conn2res acepta tareas de NeuroGym y devuelve **curvas de rendimiento a lo largo del parámetro de régimen** (estable, crítico, caótico) en vez de un número suelto — [conn2res](https://conn2res.readthedocs.io); [Nat Commun 2024](https://ideas.repec.org/a/nat/natcom/v15y2024i1d10.1038_s41467-024-44900-4.html) [PR].
- Cada unidad oculta guarda aproximadamente un número real del historial de entrada — [Collins et al. 2017](https://www.arxiv.org/abs/1611.09913). Eso da una escala para dimensionar la demanda de memoria de una tarea.
- Paradigma maestro-alumno [PP bioRxiv v2; COSYNE 2025]:
  - Un alumno con el conectoma del maestro y parámetros por neurona desconocidos reproduce la salida de la tarea, pero **no** la actividad de cada neurona. El error baja mucho al registrar M > 30 neuronas de 300.
  - Los alumnos con pesos entrenables no predicen las neuronas no registradas para ningún M.
  - El número de registros necesario lo fija la dimensionalidad de la dinámica, no el tamaño de la red. En redes aleatorias caóticas crece en proporción a N.
  - [Beiran & Litwin-Kumar](https://www.biorxiv.org/content/10.1101/2024.02.22.581667.full); [COSYNE 2025](https://world-wide.org/eposter/cosyne-25/predicting-neural-activity-connectome-3guagi1t)
- Calibración con estructura plantada en Lappalainen [PR]. Con 25 redes verdad-base por nivel de dispersión (MNIST), los modelos que solo conocen la conectividad dan una correlación mediana de 0,85 con el 10 % de conectividad y de 0,38 con el 80 %. El cableado restringe más cuanto más disperso es — [Lappalainen et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/).
- Arquitecturas distintas, entrenadas en tareas con estructura dinámica, convergen en la misma topología de puntos fijos — [Maheswaranathan et al. 2019](https://arxiv.org/abs/1907.08549). Si la tarea admite una solución universal, el cableado no tiene margen para diferenciarse.

### Inferences
- Puerta de resolución que propongo, en tres pasos y fijada antes de mirar resultados:
  - **(i) Necesidad de recurrencia.** Núcleo frente a `sin_nucleo`, con un tamaño de efecto mínimo fijado de antemano y varias veces mayor que la desviación entre semillas. En Mosca esto falla en fluidos (0,0994 frente a 0,0998) y en el NES (`sin_nucleo` empata con la mosca), así que la pregunta del cableado no tiene resolución en esas tareas.
  - **(ii) Sensibilidad (estructura plantada).** Generar la tarea con un maestro cuya máscara sea el subgrafo de la mosca, con los pesos fijados, y comprobar que el alumno con la máscara verdadera gana al alumno con la máscara DP. Si ni siquiera así se separan, el procedimiento (pesos por arista entrenables, puertos, tamaño) no puede detectar ventajas de cableado. Es un análisis de potencia ejecutable.
  - **(iii) Diferenciación de arquitecturas.** Un control estructurado conocido, por ejemplo un anillo de atractores escrito a mano para orientación, debe ganar a ER en su tarea. Si no gana, la tarea no discrimina estructuras.
- Tareas candidatas, con demanda de memoria regulable: barrido de capacidad de memoria (retardos de 1 a 50), n-bit flip-flop, emparejamiento con demora y distractores, integración de velocidad angular o de trayectoria. La última encaja con el complejo central (ver la pregunta 5).
- El caso de Mario muestra que la memoria es necesaria (clonado con ventana: 0,98 frente a 0,32 de `sin_nucleo`) pero **no discrimina cableados**: mosca 0,981, aleatorio 0,979, recableado 1,000. Pasar el criterio (i) es necesario y no suficiente; también hace falta el (ii).

### Gaps
- No encontré ningún trabajo de NeuroAI que publique explícitamente una «puerta de resolución» o un análisis de potencia para comparaciones entre cableados. Parece un hueco metodológico real.
- No verifiqué la lista exacta de tareas de Maheswaranathan 2019.

---

## 5. La hipótesis de la tarea ecológica: ¿el cableado biológico ayuda en tareas parecidas a las computaciones naturales del organismo?

### Takeaway
Las pruebas a favor son indirectas y dependen del régimen. Con tareas ecológicas y el conectoma fijo (locomoción en FlyGM, flujo óptico en el lóbulo óptico, dirección de la cabeza en el complejo central), el cableado **predice la biología** y en FlyGM supera a DP. Pero con flyvis, la misma tarea ecológica deja de dar ventaja frente a DP en cuanto la inicialización es justa. Y en el cuerpo seta, el diseño biológico *es* aleatorio por construcción. «Ecológico» parece necesario pero no suficiente.

### Cited Findings
- Locomoción ecológica (FlyGM) [PP]:
  - Inicio de la marcha, marcha recta a 3 cm/s, giro (10 rad/s) y vuelo a 20 cm/s.
  - La ventaja sobre DP crece con la dificultad: de 2,88° a 5,26°.
  - La política se inicializa **por imitación** de expertos MLP de flybody antes de PPO.
  - [Jin et al. 2026](https://arxiv.org/html/2602.17997v3)
- Lóbulo óptico, flujo óptico en Sintel [PR]:
  - 64 tipos celulares y 734 parámetros libres (65 potenciales de reposo, 65 constantes de tiempo y 604 escalas sinápticas); los números de sinapsis y los signos están fijos.
  - Conjunto de 50 modelos. Predice la preferencia ON/OFF de los 32 tipos conocidos, T4 como ON y T5 como OFF, y 12 de 19 tipos con entradas asimétricas como selectivos de dirección.
  - Un modelo con parámetros aleatorios sobre el conectoma completo predice el contraste pero **no** la selectividad de dirección.
  - El error de la tarea se correlaciona con la exactitud de la selectividad de dirección: r = −0,60, P = 2,6 × 10⁻⁶.
  - Los modelos con solo conectividad por tipo celular predicen mal.
  - [Lappalainen et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/)
- Sobre ese mismo flyvis, la ventaja de *rendimiento* frente a DP desaparece con inicialización compartida — [Dhiman 2026](https://arxiv.org/html/2604.04033v1) [PP]. Ojo: se evaluó tras solo 5 y 10 pasos de entrenamiento.
- Dirección de la cabeza, complejo central [PP bioRxiv; NeurIPS 2025]. Partiendo del conectoma medido y ajustando solo ganancias, sesgos y constantes de tiempo por tipo celular, con una pérdida autosupervisada (que la burbuja siga a la velocidad de entrada), el circuito funciona como atractor de anillo **aunque el conectoma sea asimétrico e irregular** — [Duan, Dong & Fiete 2025](https://www.biorxiv.org/content/10.1101/2025.05.26.655406.full.pdf); [NeurIPS](https://neurips.cc/virtual/2025/poster/115017). Otro preprint obtiene atractores de anillo exactos a partir de cuatro conectomas de mosca — [bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2024.11.01.621596.full.pdf). No encontré en estos trabajos ninguna comparación contra cableados aleatorios entrenados.
- Cuerpo seta:
  - La teoría predice el número óptimo de entradas por célula de Kenyon y por célula granular del cerebelo, y lo que predice coincide con lo observado.
  - Con cableado **aleatorio**, la conectividad dispersa (< 10 entradas) supera a la densa. Si los pesos son plásticos, el óptimo se desplaza hacia el cableado denso.
  - [Litwin-Kumar, charla](https://neuroscience.stanford.edu/node/2463)
  - En la larva, la mayoría de las células de Kenyon combinan entradas al azar y un subconjunto recibe entradas estereotipadas; esa organización maximiza la discriminación en un modelo — [Eichler et al., conectoma del cuerpo seta larvario](https://integ04-cnspfv.archimed.fr/doc/OAI_1/oai-HAL-hal-04193636v1) [PR; resumen 2ª].
- Cerebro entero con pesos fijos (LIF, más de 127.400 neuronas y más de 50 M de sinapsis) [PR]. Predice las neuronas de alimentación y aseo, validadas con optogenética — [Shiu et al. 2024, Nature](https://ideas.repec.org/a/nat/nature/v634y2024i8032d10.1038_s41586-024-07763-9.html). Según una revisión, el 91 % de 164 predicciones resultaron coherentes con los experimentos — [State of Brain Emulation 2025, 2ª](https://arxiv.org/pdf/2510.15745).
- Abstract de COSYNE 2025: los reservorios de conectoma concentran la tarea en pocos subconjuntos de neuronas muy selectivos, mientras que los aleatorios reparten la implicación de forma más «general» — [COSYNE 2025](https://world-wide.org/eposter/cosyne-25/sparse-neural-engagement-connectome-ybm245ip) [abstract, 2ª].

### Inferences
- Todos los casos ecológicos positivos comparten **régimen (b) o (a)**: pesos fijos o restringidos y pocos parámetros libres. Ninguno usa pesos por arista libres. La variable que mejor separa éxitos de fracasos en la literatura revisada parece el régimen de parámetros, más que «ecológico frente a ajeno». Es una inferencia de patrón, no hay un estudio que lo cruce.
- En el cuerpo seta, un nulo ER con la densidad y el grado de entrada correctos *es* casi el modelo biológico. Una tarea olfativa de discriminación con ese subcircuito probablemente empataría con ER **por diseño**, así que sirve mal para la pregunta del proyecto.
- El complejo central es el mejor candidato de tarea ecológica para Mosca: integración de velocidad angular o de trayectoria, con el subcircuito de unas 2.875 neuronas CX anotadas en los datos locales (EPG, PEN, Delta7). Hay prueba de que el conectoma, con pocos parámetros por tipo celular, ya implementa la computación (Duan/Fiete). Falta el contraste contra DP. Sería una contribución nueva y responde a la pregunta del proyecto con una predicción a priori clara: si la topología importa en alguna parte, debería ser aquí.
- Navier–Stokes y Mario son ajenos *y* apenas recurrentes. El empate actual no distingue entre «la topología no importa» y «la tarea no la pone a prueba».

### Gaps
- No encontré ningún estudio de CX ni de cuerpo seta que compare el conectoma entrenado contra nulos DP o ER entrenados con el mismo protocolo.
- No verifiqué la cifra del 91 % de Shiu en el texto original.
- No sé si Dhiman probó horizontes más largos que 10 pasos; el resumen solo da 5 y 10.

---

## 6. Escala: ¿la ventaja aparece solo con el cerebro entero, o con puertos sensoriales y motores anatómicos en vez de puertos elegidos por grado?

### Takeaway
No hay ablaciones directas de escala (subgrafo frente a cerebro entero) en los trabajos positivos. Lo que sí hay indica que **dónde se colocan la entrada y la salida** puede pesar tanto o más que el cableado. En C. elegans, barajar la entrada/salida resta más que recablear, y FlyGM usa la clasificación anatómica de flujo de FlyWire. La escala por sí sola no está demostrada como factor.

### Cited Findings
- FlyGM usa el cerebro entero (FAFB v783). Los puertos aferente, intrínseco y eferente siguen la clasificación de *flow* de FlyWire: la observación se codifica en 32 dimensiones y se difunde a todos los aferentes, y las acciones se leen de los eferentes. **No hay ablaciones de puertos ni de escala** — [Jin et al. 2026](https://arxiv.org/html/2602.17997v3).
- En C. elegans, con 180 neuronas:
  - la entrada y salida biológica (sensoriales hacia dentro, músculos de la pared corporal hacia fuera) frente a una aleatoria vale +0,657 en capacidad de memoria, frente a +0,305 por recablear;
  - con ambas manipulaciones, +0,951;
  - Go/No-go pasa a favorecer a los aleatorios cuando se baraja la entrada/salida (−0,386).
  - [Reimers et al. 2026, 2ª](https://blog.pebblous.ai/blog/celegans-connectome-reservoir-benchmark/en/)
- conn2res trata la elección de los nodos de entrada y de lectura como uno de los grados de libertad centrales del reservorio de conectoma — [conn2res](https://conn2res.readthedocs.io).
- Sobre el tamaño:
  - La restricción que ejerce el conectoma depende de la dimensionalidad de la dinámica, no del número de neuronas. En redes de rango 2 el requisito de registros apenas cambia con N — [Beiran & Litwin-Kumar](https://www.biorxiv.org/content/10.1101/2024.02.22.581667.full).
  - El cableado restringe más cuanto más dispersa es la red — [Lappalainen et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/).
  - En redes dispersas aleatorias, la ventaja relativa de la máscara aleatoria crece con el tamaño de la red — [Liu et al. 2022](https://arxiv.org/abs/2202.02643v1).
- En flyvis, con 45.669 nodos, la ventaja frente a DP desaparece igualmente con controles estrictos — [Dhiman 2026](https://arxiv.org/html/2604.04033v1). Que el grafo sea grande no basta.

### Inferences
- Elegir los puertos por grado en Mosca tiene un efecto sutil:
  - en el DP, los grados se conservan, así que **las mismas neuronas** son sensoriales y motoras en la mosca y en el recableado. Bien para la comparación, pero se pierde cualquier información anatómica de los puertos;
  - en ER, las neuronas puerto son otras, con otra posición en el grafo.
- Con puertos anatómicos (aferentes sensoriales y neuronas descendentes de las anotaciones de FlyWire), el DP conserva los grados, pero **no** el orden de capas sensorial→motor ni la distancia en saltos entre puertos. Ahí es donde el cableado real puede llevar información que el DP destruye.
- En FlyGM, con 4 rondas de paso de mensajes, la ventaja frente al DP podría venir de la distancia aferente→eferente. Es una hipótesis comprobable: medir la distribución de longitudes de camino entre los conjuntos de puertos en el conectoma y en los nulos.
- Rediseño:
  - **(i)** puertos anatómicos, con el mismo conjunto de neuronas puerto en los seis modelos;
  - **(ii)** un control adicional de «puertos aleatorios» sobre el cableado real, como en Reimers;
  - **(iii)** registrar en cada modelo la distancia sensorial→motora. Según los registros del proyecto, en el diseño actual las motoras están a 1 salto de las sensoriales: si la entrada y la salida son adyacentes, la tarea puede resolverse sin usar la red profunda, y eso refuerza el empate.
- No hay pruebas de que pasar de 4.000 al cerebro entero cambie nada sin cambiar también el régimen de pesos y los puertos. Subir N por sí solo es caro y está poco motivado.

### Gaps
- Ningún trabajo que encontré compara la misma tarea con un subgrafo de N neuronas frente al cerebro entero y nulos equivalentes.
- No hay ablación de puertos en FlyGM.
- Las cifras de Reimers no están verificadas en la fuente primaria.

---

## 7. Práctica estadística: semillas, pruebas por nivel frente a por semilla, prerregistro en ML/NeuroAI y publicación de resultados nulos

### Takeaway
Las comparaciones entre cableados tienen datos anidados: instancia del grafo nulo × semilla de entrenamiento × nivel o episodio. Con pocas ejecuciones, las estimaciones puntuales engañan. Lo recomendado es tratar la instancia del nulo y la semilla como efectos aleatorios (bootstrap jerárquico), dar intervalos y métricas robustas (IQM), varias instancias de cada nulo y un protocolo prerregistrado. Los trabajos positivos más citados (FlyGM) no cumplen ese estándar; el trabajo negativo (Dhiman) lo cumple a medias.

### Cited Findings
- Aprendizaje por refuerzo con pocas ejecuciones [PR, NeurIPS 2021, Outstanding Paper]. Las estimaciones puntuales de media y mediana con pocas ejecuciones son poco fiables. Los autores recomiendan intervalos para las métricas agregadas, perfiles de rendimiento y la media intercuartílica (IQM). En Atari 100k, el análisis completo cambia conclusiones publicadas — [Agarwal et al. 2021](https://arxiv.org/abs/2108.13264).
- Dos grupos de 5 semillas del mismo algoritmo pueden dar curvas que parecen de distribuciones distintas, y los artículos revisados usaban 5 semillas o menos — [Henderson et al. 2018](https://ar5iv.labs.arxiv.org/html/1709.06560) [PR, AAAI; la cifra de 2 × 5 viene de una fuente secundaria].
- Datos anidados: tratar cada punto como independiente subestima la incertidumbre y el valor p. Alrededor del 53 % de los estudios en revistas de neurociencia destacadas tenía datos anidados. Los autores proponen el bootstrap jerárquico, que remuestrea en cada nivel — [Saravanan, Berman & Sober 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7906290) [PR, NBDT].
- Dhiman [PP] usa 5 grafos DP × 3 semillas (15 combinaciones), bootstrap de la diferencia control − conectoma e inicialización compartida, y recomienda documentar qué conserva cada nulo — [Dhiman 2026](https://arxiv.org/html/2604.04033v1).
- La variación entre instancias del mismo generador (0,2-0,4 %) es comparable al efecto de muchas diferencias entre arquitecturas: hacen falta varias instancias por generador — [Xie et al. 2019](https://ar5iv.labs.arxiv.org/html/1904.01569). You et al. usaron 5 semillas por grafo, y una submuestra de 52 grafos reproduce con r = 0,90 las tendencias de los 3.942 — [You et al. 2020](https://arxiv.org/html/2007.06559v2).
- FlyGM dice «varias ejecuciones» sin número y no hace pruebas — [Jin et al. 2026](https://arxiv.org/html/2602.17997v3).
- La conclusión de Reimers cambia con la medida de los pesos (número de sinapsis, tamaño o área de contacto) y con la métrica (Pearson o RMSE). Es un ejemplo claro de «jardín de senderos que se bifurcan» — [Reimers et al. 2026, 2ª](https://blog.pebblous.ai/blog/celegans-connectome-reservoir-benchmark/en/).
- Prerregistro en ML. Los talleres de NeurIPS de 2020 (piloto, PMLR v148) y 2021 (PMLR v181) aceptaban artículos **sin resultados**, solo el protocolo, para combatir que los resultados negativos no se publiquen — [NeurIPS 2020](https://neurips.cc/virtual/2020/workshop/16158); [NeurIPS 2021](https://neurips.cc/virtual/2021/workshop/21885); [PMLR v148](https://proceedings.mlr.press/v148/).

### Inferences
- Unidad de análisis en Mosca:
  - con 4 niveles de test en el NES, la p exacta por nivel no baja de 0,125, y con 3 semillas la p por semilla no baja de 0,1 (ya anotado en los registros del proyecto);
  - esto es un **techo de potencia estructural**: ninguna prueba honesta puede dar significación con esas unidades;
  - el rediseño necesita más unidades independientes en el nivel donde vive la variación: niveles procedurales (cientos), más semillas (≥ 5-10), y **varias instancias de cada nulo** (ER y DP) como efecto aleatorio cruzado con la semilla.
- Para declarar un *empate* hace falta una prueba de **equivalencia** con un tamaño de efecto mínimo de interés fijado de antemano, no un p > 0,05. Ejemplo: «mosca − DP dentro de ±0,02 de progreso». El método estándar es TOST (Lakens 2017); la referencia no está verificada en esta sesión. El plan de la 6c (`analiza_6c.py` antes de los datos) ya va en esa línea; habría que añadir el margen de equivalencia y la puerta de resolución de la pregunta 4.
- Plantilla de informe mínima para cada comparación de cableados:
  - qué conserva cada nulo, y la mezcla del DP (intercambios por arista);
  - radio espectral y entropía SVD en la inicialización;
  - número de instancias de nulo × semillas;
  - IQM con IC por bootstrap jerárquico;
  - prueba de equivalencia;
  - resultado del control positivo (núcleo frente a `sin_nucleo`, estructura plantada).
- Un resultado nulo con este estándar se puede publicar y tiene valor. Un empate sin la puerta de resolución, no: es exactamente la diferencia entre «H0 confirmada» y «el experimento no podía responder», ya anotada en los registros del proyecto.

### Gaps
- No encontré guías específicas de NeuroAI para comparaciones conectoma-contra-nulo más allá de las recomendaciones de Dhiman.
- No hay datos publicados sobre los resultados de los pilotos de prerregistro de NeurIPS.
- No verifiqué el texto original de Henderson para la cifra de 2 × 5 semillas.
