# Diseño: una red con el cableado de la mosca que predice fluidos

**Estado:** borrador de diseño, v0.1
**Fecha:** 11 de septiembre de 2026

## 1. Objetivo

Queremos saber si el cableado real del cerebro de *Drosophila melanogaster* sirve como arquitectura para aprender dinámica de fluidos. Para ello usaremos el conectoma de FlyWire como máscara de conectividad de una red neuronal recurrente. La red se entrenará para predecir la evolución de un flujo turbulento en 2D gobernado por las ecuaciones de Navier–Stokes, y se comparará con redes de igual tamaño cableadas de otras formas.

La pregunta concreta es esta: **¿el cableado específico de la mosca aporta algo frente a un cableado aleatorio con las mismas estadísticas?**

## 2. Qué NO es este proyecto

No entrenamos una mosca real. Solo usamos su diagrama de conexiones. Tampoco intentamos resolver el problema del milenio sobre Navier–Stokes, que exige una demostración matemática. Nuestro objetivo es un modelo sustituto (*surrogate*) que aprende a predecir flujos a partir de datos simulados. Por último, no esperamos batir a los modelos especializados. El FNO sirve de referencia superior, no de rival.

## 3. Hipótesis

| ID | Hipótesis | Cómo se contrasta |
|----|-----------|-------------------|
| H0 | El cableado de la mosca rinde igual que uno aleatorio de la misma densidad | Mosca vs. aleatorio (Erdős–Rényi) |
| H1 | La ventaja, si la hay, se explica por la distribución de grados | Mosca vs. recableado con grados conservados |
| H2 | La ventaja se debe a la estructura específica (motivos, módulos) | Mosca > recableado, con significación estadística |

Nuestra predicción previa es H0 o H1. Registramos esa predicción antes de ver los resultados para no autoengañarnos.

## 4. Datos del conectoma

**Fuente:** repositorio `philshiu/Drosophila_brain_model` (Shiu et al., 2024, *Nature*), basado en FlyWire materialización v783.

| Archivo | Tamaño | Contenido |
|---------|--------|-----------|
| `Completeness_783.csv` | 3,3 MB | Lista de ~139.000 neuronas con sus IDs |
| `Connectivity_783.parquet` | 101 MB | Pares presináptica→postsináptica, número de sinapsis y signo |

El signo (excitadora o inhibidora) viene de los neurotransmisores predichos por FlyWire. Consideramos excitadora la acetilcolina e inhibidores GABA y glutamato, la misma convención que Shiu et al.

**Limitación conocida:** este repositorio no incluye las anotaciones de región cerebral. Para seleccionar regiones concretas, como el complejo central, necesitaríamos los datos de anotación de FlyWire Codex, que requieren registro.

### 4.1 Selección del subgrafo

Entrenar las ~139.000 neuronas no es viable en CPU, así que extraemos un subgrafo inducido de N neuronas. Hay dos estrategias posibles:

**DECIDIDO: N = 4.000, cobertura de todo el cerebro mediante crecimiento por manchas.**

No cogemos 4.000 neuronas al azar de toda la ciudad: la mayoría quedaría aislada, porque sus conexiones apuntan a neuronas que dejamos fuera, y una red desconectada no aprende nada. Tampoco cogemos solo las de mayor grado, porque eso sesga hacia un núcleo muy denso que no representa al cerebro.

El procedimiento es:

1. Plantar S ≈ 20 semillas repartidas por el grafo (muestreo estratificado por grado, para que haya neuronas muy conectadas y poco conectadas).
2. Desde cada semilla, crecer por vecindad (BFS sobre el grafo no dirigido) hasta repartir las 4.000 neuronas entre las manchas.
3. Tomar el subgrafo inducido y quedarse con la **componente débilmente conexa mayor**.
4. Podar neuronas con grado interno 0.

**Criterio de aceptación:** al menos el 90 % de las 4.000 neuronas deben quedar en la componente principal. Si no, aumentar el tamaño de mancha y reducir S.

Al extraer el subgrafo guardaremos su densidad, la distribución de grados, la fracción inhibidora y el número de componentes, porque los controles tienen que reproducir esas cifras.

## 5. Simulador de fluidos

Resolvemos Navier–Stokes 2D incompresible en forma de vorticidad:

```
∂ω/∂t + (u·∇)ω = ν ∇²ω + f(x) − α ω
u = ∇⊥ ψ,   −∇²ψ = ω
```

| Parámetro | Valor propuesto | Motivo |
|-----------|-----------------|--------|
| Dominio | [0, 2π]², periódico | Permite método espectral |
| Malla | 64 × 64 | Suficiente para turbulencia visible, barato en CPU |
| Método | Pseudoespectral + RK4, desaliasing 2/3 | Estándar, preciso |
| Forzado | Kolmogorov, f = sin(4y) | Turbulencia sostenida, banco de pruebas conocido |
| Viscosidad ν | 1e-3 | Régimen turbulento moderado |
| Arrastre α | 0,1 | Evita acumulación de energía en escalas grandes |
| Trayectorias | ~200 entrenamiento, 40 validación, 40 test | Condiciones iniciales aleatorias |
| Longitud | 200 pasos guardados por trayectoria | Tras descartar el transitorio inicial |

El conjunto de test usa semillas separadas desde el principio para evitar filtraciones entre entrenamiento y evaluación.

## 6. Arquitectura del modelo

```
campo ω(t) ──► FFT ──► modos bajos (K≈256 números reales)
                              │
                     W_in (entrenable)
                              ▼
            ┌──────────────────────────────────┐
            │  Núcleo recurrente de N neuronas  │
            │  h ← tanh(W_in·x + (M⊙W)·h + b)   │  × T_int pasos internos
            │  M = máscara del conectoma        │
            └──────────────────────────────────┘
                              │
                     W_out (entrenable)
                              ▼
               Δ modos ──► iFFT ──► ω(t+Δt)
```

**Entrada y salida.** Unas 256 neuronas del subgrafo reciben la entrada (neuronas "sensoriales") y otras 256 distintas se leen para producir la salida. Se asignan según su posición en el grafo: las de más salidas hacia el resto actúan como entrada y las de más entradas desde el resto como salida. Este es un análogo burdo de la jerarquía sensorial→motora. La red predice el *incremento* del campo (predicción residual), no el campo completo, porque así se entrena mucho más fácilmente.

**Núcleo.** Hay tres restricciones biológicas:

1. **Conectividad fija:** solo existen las conexiones de M. Las demás son siempre cero.
2. **Ley de Dale:** cada neurona es siempre excitadora o siempre inhibidora. Se implementa como W = signo · softplus(θ), donde solo θ es entrenable.
3. **Inicialización:** las magnitudes arrancan proporcionales al número de sinapsis y se reescalan para que el radio espectral quede cerca de 1, evitando que la actividad explote o se apague.

**Pasos internos.** Proponemos T_int = 4, es decir, la red "piensa" 4 ciclos por cada paso del fluido.

## 7. Controles

Todos los controles comparten codificador, decodificador, número de neuronas, número de parámetros entrenables y receta de entrenamiento. Solo cambia la máscara M.

| Modelo | Máscara | Qué aísla |
|--------|---------|-----------|
| **Mosca** | Subgrafo de FlyWire | — |
| **Aleatorio** | Erdős–Rényi, misma densidad y fracción inhibidora | Efecto de la estructura en general |
| **Recableado** | Intercambio de aristas conservando grados de entrada y salida | Efecto de la estructura *más allá* de los grados |
| **Denso** | Red densa con igual número de parámetros (menos neuronas) | Si la dispersión ayuda o estorba |
| **FNO** | Fourier Neural Operator estándar | Techo de rendimiento |
| **Persistencia** | ω(t+Δt) = ω(t) | Suelo: el modelo que no aprende nada |

## 8. Entrenamiento

| Fase | Objetivo | Detalle |
|------|----------|---------|
| 1 | Predicción a un paso | Pérdida L2 relativa, Adam, lr 1e-3 con decaimiento coseno |
| 2 | Ajuste con despliegue | Se despliega el modelo 5 → 20 pasos (currículo) con retropropagación a través del tiempo |

Usaremos recorte de gradiente (norma 1,0), parada temprana según validación y **3 semillas por modelo como mínimo**.

## 9. Evaluación

| Métrica | Qué mide |
|---------|----------|
| Error L2 relativo a 1 paso | Precisión inmediata |
| Error en despliegue de 50 pasos | Estabilidad a largo plazo |
| Tiempo hasta decorrelación | Pasos hasta que la correlación con la verdad cae por debajo de 0,8 |
| Espectro de energía E(k) | Si respeta la física estadística, aunque diverja punto a punto |

**Análisis estadístico:** media ± desviación típica entre semillas, con prueba de permutación entre Mosca y cada control. Una diferencia solo cuenta si supera la variabilidad entre semillas.

## 10. Estructura del código

```
mosca_ns/
├── data/
│   ├── download_flywire.py    # descarga y cachea los .parquet/.csv
│   └── simulate_ns.py         # genera el conjunto de trayectorias
├── connectome/
│   ├── subgraph.py            # selección del subgrafo
│   └── controls.py            # máscaras aleatoria y recableada
├── models/
│   ├── masked_rnn.py          # núcleo con máscara + ley de Dale
│   └── fno.py                 # referencia
├── train.py
├── evaluate.py
└── configs/                   # un YAML por experimento
```

## 11. Riesgos

| Riesgo | Probabilidad | Mitigación |
|--------|--------------|------------|
| Inestabilidad de la RNN (explosión/desvanecimiento) | Alta | Reescalado espectral, recorte de gradiente, predicción residual |
| Coste de CPU excesivo | Media | Empezar con N=1.000 y malla 32×32, escalar después |
| Diferencias dentro del ruido | Alta | Varias semillas, prueba de permutación, informar del resultado nulo |
| Sesgo del subgrafo "más conectado" | Media | Documentarlo; repetir con la estrategia B si hay acceso a anotaciones |

## 12. Decisiones tomadas

1. **Subgrafo:** cobertura de todo el cerebro por crecimiento en manchas (ver 4.1). No se restringe a una región funcional, así que no hacen falta las anotaciones de FlyWire Codex.
2. **Tamaño:** N = 4.000 neuronas.
3. **Alcance:** comparación completa contra los cinco controles de la sección 7.

## 13. Trabajo futuro: transferencia a otras tareas

El núcleo con máscara del conectoma es agnóstico a la tarea. Solo hay que cambiar el codificador, el decodificador y el objetivo de entrenamiento. Esto permite contrastar si el cableado de la mosca sirve para algo en general o solo para fluidos.

**Orden propuesto (después de tener resultados en fluidos):**

| Tarea | Entorno | Entrada | Salida | Entrenamiento |
|-------|---------|---------|--------|---------------|
| **Plataformas (Mario)** | MarioAI / Infinite Mario Bros, o `procgen` | Rejilla de casillas alrededor del jugador (suelo, enemigo, hueco, tubería) | Botones (izq., der., salto, correr) | PPO |
| **Fútbol** | Google Research Football | Vector de estado (posiciones de jugadores y balón) | Acción discreta | PPO |

Ambos son control sensoriomotor en tiempo real, que es lo que el cerebro de la mosca hace de verdad. Si el cableado aporta algo en alguna parte, es aquí más que en fluidos.

**Notas de viabilidad.** Se usan entornos con estado simbólico y no píxeles, porque aprender desde imagen multiplica el coste por varios órdenes de magnitud y aquí no hay GPU. Se descartan emuladores de ROMs comerciales por derechos de autor, y se descarta automatizar FIFA por las condiciones de uso de EA y porque requiere el juego instalado con pantalla. En Mario es obligatorio usar **niveles generados aleatoriamente**: con un nivel fijo, la red puede memorizar la secuencia de saltos y la métrica dejaría de medir aprendizaje.

## Referencias

- Dorkenwald et al. (2024). *Neuronal wiring diagram of an adult brain.* Nature.
- Shiu et al. (2024). *A Drosophila computational brain model reveals sensorimotor processing.* Nature.
- Lappalainen et al. (2024). *Connectome-constrained networks predict neural activity across the fly visual system.* Nature.
- Li et al. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR.
- Kochkov et al. (2021). *Machine learning–accelerated computational fluid dynamics.* PNAS.
