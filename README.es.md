# Mosca — una IA cableada como el cerebro de una mosca, prediciendo un fluido

[English](README.md)

<p align="center">
  <img src="docs/media/fly_fluid.gif" alt="Neuronas de la red cableada como la mosca encendiéndose junto al campo de vorticidad real y el predicho" width="100%">
</p>

Tomamos el diagrama de conexiones del cerebro de *Drosophila* (conectoma de FlyWire, unas
139.000 neuronas), recortamos un trozo y construimos una red recurrente artificial que
**solo puede usar las conexiones que existen en la mosca**, respetando qué neuronas excitan y
cuáles inhiben (ley de Dale). Después la ponemos a predecir un flujo de Navier–Stokes en 2D
y a jugar a un juego de plataformas.

**La pregunta:** ¿el cableado real aporta algo frente a un cableado aleatorio con las mismas
estadísticas?

> **No es una mosca.** Nada de esto es una mosca ni un modelo de cómo piensa una mosca. El
> conectoma se usa como plano de una red artificial.

## En corto (21-10-2026)

- **En cinco tareas, el cableado de la mosca no le gana a ningún cableado aleatorio.** En la
  única con resolución suficiente para separarlos (plataformas con niveles nunca vistos),
  **pierde** contra el aleatorio, con un análisis escrito antes de tener los datos.
- **Encontramos un sesgo del diseño que castiga a la mosca.** Para que todos los cableados empiecen
  en el mismo régimen dinámico, el diseño escala los pesos hasta que el autovalor mayor
  vale 0,95. La mosca tiene un núcleo de pares de neuronas conectadas en los dos sentidos,
  con muchas sinapsis, que dispara su autovalor mayor; al escalarlo, todas las demás
  conexiones quedan **27 veces más débiles que en el aleatorio** (73 veces con 4.000
  neuronas). Su recurrencia está casi apagada, lo que encaja con que empate con un control
  sin conexiones. "Mismo radio espectral" no es "mismo trato".
- <!-- 6E --> **Pero no es lo que la hace perder.** Reentrenada con una normalización que
  iguala la corriente de entrada de cada neurona (la recurrencia de la mosca pasa de estar
  27 veces más apagada a 4), la mosca mejora un poco (0,462 → 0,503 de progreso) y el
  aleatorio mejora más (0,568 → 0,653): la distancia no se cierra, crece. Con el análisis
  escrito antes de los datos, el reescalado queda descartado como causa. Por qué pierde
  sigue sin estar medido. <!-- /6E -->
- **Lo que falta:** el experimento original de fluidos a escala real (4.000 neuronas, malla
  64×64) va por 8 de 18 entrenamientos, en pausa. Con una semilla, ahí la mosca sí le gana
  al control sin conexiones.

Todas las cifras, también las que salieron mal, están en [`RESULTADOS.md`](RESULTADOS.md).
La comparación con todo lo publicado está en [`docs/literatura/informe.md`](docs/literatura/informe.md).

## Predicción registrada de antemano

Escrita antes de ver ningún resultado ([diseño, §3](diseno_mosca_navier_stokes.md)):

| | Hipótesis |
|---|---|
| **H0** | El cableado de la mosca rinde igual que uno aleatorio de la misma densidad |
| **H1** | Cualquier ventaja se explica por la distribución de grados (mosca ≈ recableado con grados conservados) |
| **H2** | La estructura específica importa (mosca > recableado, con significación) |

Esperamos **H0 o H1**, y que un Fourier Neural Operator (FNO) gane a todas las redes
cableadas. Un resultado nulo es un resultado válido.

## Los modelos

Entre los modelos cableados solo cambia la máscara de conectividad; codificador,
decodificador, receta de entrenamiento y presupuesto de parámetros son comunes. Todos los
pesos de la máscara se entrenan; el signo de cada neurona viene de los datos y no cambia.

| Modelo | Núcleo recurrente |
|---|---|
| `mosca` | Subgrafo de FlyWire, signos reales |
| `aleatorio` | Erdős–Rényi, misma densidad y fracción inhibidora |
| `recableado` | Grafo de la mosca con aristas barajadas, **grados conservados exactamente** |
| `denso` | Todas contra todas, menos neuronas, mismo número de pesos entrenables |
| `sin_nucleo` | Mismas neuronas, signos y puertos que la mosca, **cero conexiones** (control) |
| `mlp` | Red sin máscara con el mismo número de parámetros (solo en Mario) |
| `fno` | Fourier Neural Operator (referencia superior, solo en fluidos) |
| `persistencia` | Predice "no cambia nada" (referencia inferior, solo en fluidos) |

`sin_nucleo` es el control que dice si hay algo que discutir: si la mosca no le gana, lo que
hace no sale de su cableado, sino del codificador y el decodificador que comparten todos.
No hemos encontrado ningún otro trabajo que lo use.

## Lo que ha salido, tarea por tarea

| Tarea | Aprende con | Mosca contra aleatorio / recableado | Núcleo contra `sin_nucleo` |
|---|---|---|---|
| Navier–Stokes, ensayo (500 neuronas, 32×32) | supervisado | empate / empate | 0,0004 menos de error: nada |
| Navier–Stokes, escala real (4.000, 64×64) | supervisado | **pendiente**: 8 de 18 entrenamientos | semilla 0: −4 % de error en despliegue |
| Plataformas propio | PPO desde cero | todos atascados en "saltar a la derecha"; cerrada sin resolución | — |
| Super Mario Bros (NES), nivel 1-1 | imitación con memoria | empate (mosca 0,98, aleatorio 0,98, recableado 1,00) | lo decide la memoria: 0,98 contra 0,32 |
| NES, niveles no vistos | PPO | empate / no resuelto | empate |
| Plataformas procedurales, 200 niveles no vistos | imitación + PPO | **pierde** (0,462 contra 0,568) / no resuelto | empate |

En la última, los seis modelos llegan a la meta en niveles que no han visto y le ganan al
azar y a cualquier política constante, así que la tarea sí separa. Y en esa tarea el
aleatorio y el recableado **sí** le ganan a `sin_nucleo` (+0,091 y +0,049, p < 0,0001): la
recurrencia sirve cuando está viva. La que no la usa es la mosca.

## Mismo radio espectral no es mismo trato (pero no es por eso por lo que pierde)

| | mosca | recableado | aleatorio |
|---|---|---|---|
| autovalor mayor antes de escalar (500 neuronas) | 74,5 | 16,4 | 10,4 |
| grueso del espectro tras escalar a 0,95 (mediana de \|autovalor\|) | **0,013** | 0,102 | 0,356 |
| lo mismo con 4.000 neuronas | **0,004** | 0,075 | 0,284 |
| fracción de conexiones con su recíproca | 0,35 | 0,09 | 0,04 |
| progreso en los niveles no vistos | 0,462 | 0,527 | 0,568 |
| grueso del espectro con normalización por filas (`row`) | 0,146 | 0,498 | 0,595 |
| progreso con `row` | 0,503 | 0,432 | 0,653 |

El orden del grueso del espectro es el orden del resultado, y el entrenamiento no lo
corrige: después de PPO la mosca sigue en 0,013. Se mide con
[`diagnostico_cableado.py`](diagnostico_cableado.py), sin entrenar nada.

Esto no dice que el conectoma sea malo: dice que la forma habitual de igualar cableados
(igualar el radio espectral, que es lo habitual en reservorios) castiga a los grafos con un núcleo recíproco fuerte, y el de
la mosca lo tiene.

Lo que no dice es que eso explique la derrota. Se comprobó con la fase 6e (exploratoria,
análisis escrito antes de los datos): la misma tanda con `core.normalisation: row`, que
iguala la corriente de entrada de cada neurona antes de escalar. La recurrencia de la mosca
se despierta (grueso 0,146) y la mosca mejora, pero el aleatorio mejora más, y la mosca
sigue perdiendo contra él (−0,150, sus tres semillas por debajo). El veredicto registrado
es "sin efecto" contra los dos controles: el reescalado queda descartado como causa.

## Las tareas

### Fluidos

Recorrido: campo de vorticidad → FFT → modos de Fourier más bajos → neuronas *sensoriales*
(mayor grado de salida) → núcleo recurrente con máscara (4 subpasos internos) → neuronas
*motoras* (mayor grado de entrada) → modos → FFT inversa → **incremento** que se suma al campo.

### Mario

El mismo núcleo cableado, jugando en vez de prediciendo ([diseño, §13](diseno_mosca_navier_stokes.md)).
Dos entornos con **la misma observación simbólica**, así que la política es la misma en los
dos: un motor de plataformas propio con niveles procedurales, y el Super Mario Bros de NES
real con emulador, leyendo el mapa de casillas de la RAM. El visor anatómico
(`visor_nes.py`) pone el juego a un lado y, al otro, las 139.000 neuronas del conectoma con
las del subgrafo encendiéndose encima, en sus coordenadas reales de FlyWire.

<p align="center">
  <img src="docs/media/mosca_juega_nes.gif" alt="La red cableada como la mosca jugando a Super Mario Bros, con la actividad de sus neuronas al lado" width="100%">
</p>

Esta demo (300 iteraciones de PPO, política muestreada) entrena y mide en el mismo nivel, le
gana al salto constante y **no se distingue de apretar botones al azar** (p = 0,25).

Toda cifra se lee contra las políticas de referencia (azar, siempre a la derecha, salto
constante, salto corriendo y una heurística escrita a mano) o no se lee. PPO desde cero cae
solo en la política degenerada de saltar siempre a la derecha, y un MLP sin máscara
también: por eso la versión que separa cableados arranca imitando a la heurística y luego
afina con PPO, y usa la regla del NES de soltar el botón para volver a saltar, que hunde a
todas las políticas constantes.

Por imitación, la red cableada como la mosca se pasa el 1-1 de NES
([vídeo](docs/media/mosca_clonada_1-1.mp4)). Es una demo, no ciencia: mismo nivel y copiado
de una maestra. Con controles, el mérito es de la memoria, no del cableado: el aleatorio y
el recableado la clonan igual de bien, y los modelos sin recurrencia se atascan en la
primera tubería.

## Lo que hemos aprendido por el camino

- **Un bug en el aumento de datos**: la traslación libre se aplicaba sobre el eje que lleva
  el forzado `sin(4y)`, así que los pares aumentados tenían un 4 % de contenido físicamente
  falso cuando la señal que hay que aprender es ~10 % del campo. Lo detectó un test contra el
  simulador; se descartó todo lo entrenado antes con aumento.
- **Puertos injustos**: dejar que las neuronas de entrada y de salida se solaparan libremente
  daba a la mosca 159 atajos entrada→salida y al aleatorio 18. Ahora el solape es el mismo
  en todos los modelos.
- **Un crítico ciego**: en Mario, el estimador de valor no veía nada porque tenía que dar
  números de 30-60 con pesos pequeños. Arreglarlo era necesario, pero no era lo que frenaba
  a PPO: era la exploración.
- **Ruido de evaluación**: con una partida por nivel, el orden entre modelos se daba la
  vuelta solo. Ahora todos los modelos juegan los mismos niveles con los mismos números
  aleatorios, y las pruebas se hacen por nivel y por semilla.
- **El análisis, antes que los datos**: desde la fase 6c, el análisis se escribe como código
  y se commitea antes de lanzar los entrenamientos. Por eso el "pierde" de la mosca no se
  puede reinterpretar a posteriori.

## Frente a la literatura

Que sepamos, nadie había predicho una ecuación de fluidos con un conectoma, nadie usa un
control sin conexiones, y ningún trabajo reúne todos estos controles a la vez. Los
resultados positivos publicados con conectomas de mosca están en otros regímenes (pesos
fijos con pocos parámetros por neurona, o el grafo como único camino entre entrada y
salida) y casi todos son preprints de 2026. Los proyectos de "la mosca juega a X" de
septiembre de 2026 evalúan en el mismo nivel en que entrenan y casi sin controles. Informe
completo, con cada fuente etiquetada por su estado de revisión:
[`docs/literatura/informe.md`](docs/literatura/informe.md).

## Estado

**Fluidos a escala real:** 8 de 18 entrenamientos hechos (mosca ×3, aleatorio ×3,
recableado s0, `sin_nucleo` s0), en pausa desde el 23-09-2026. Faltan unas 3 horas de GPU.
Es la pregunta original del proyecto y la única tarea a 4.000 neuronas.

<!-- RESULTADOS -->

## Ejecutarlo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # en GPU, instalar la rueda de torch con CUDA
pytest -q

python -m data.download_flywire           # conectoma, 105 MB
python -m data.download_annotations       # posiciones de las neuronas para el visor, 32 MB
python -m data.simulate_ns   --config configs/ns_32_proto.yaml
python -m connectome.subgraph --config configs/conectoma_500_proto.yaml
python -m connectome.controls --config configs/conectoma_500_proto.yaml
python benchmark.py --config configs/ensayo_32.yaml          # coste medido antes de lanzar
python train.py    --config configs/ensayo_32.yaml
python evaluate.py --config configs/ensayo_32.yaml
python visor.py    --config configs/ensayo_32.yaml --models mosca aleatorio   # HTML interactivo
python video.py    --config configs/ensayo_32.yaml --models mosca              # el vídeo de arriba
```

Mario, con la receta que separa cableados (imitación + PPO, niveles procedurales):

```bash
C=configs/mario_proto_6c_soltar.yaml
python clonado_rl.py  --config $C --ruido 0.1 --guarda-pesos --semilla 0
python train_rl.py    --config $C --model mosca --seed 0 \
    --desde artifacts/mario_proto_6c_soltar/clonado/mosca_s0.pt
python evaluate_rl.py --config $C --levels 200 \
    --checkpoint artifacts/mario_proto_6c_soltar/desde_clonado/mosca_s0.pt
python diagnostico_cableado.py --config $C        # el espectro de cada cableado
pip install nes-py gym-super-mario-bros           # solo para el entorno de NES
python visor_nes.py   --config configs/mario_nes_demo.yaml     # juego + cerebro anatómico
```

Todo se configura por YAML; las semillas y la configuración se escriben en cada salida. Los
entrenamientos de verdad no caben en un portátil: la receta para la GPU y para Google Cloud
está en [`TRASPASO.md`](TRASPASO.md) y en `scripts/gcp/`.

## Datos y créditos

- Conectoma: FlyWire, materialización v783 — Dorkenwald et al., *Nature* 2024; Schlegel et
  al., *Nature* 2024. Conectividad y signos tal como los empaqueta Shiu et al., *Nature* 2024
  ([philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)).
- Posiciones de las neuronas: [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations).
- Los datos descargados no se redistribuyen en este repositorio.

Licencia MIT.
