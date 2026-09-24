# Mosca — una IA cableada como el cerebro de una mosca, prediciendo un fluido

[English](README.md)

<p align="center">
  <img src="docs/media/fly_fluid.gif" alt="Neuronas de la red cableada como la mosca encendiéndose junto al campo de vorticidad real y el predicho" width="100%">
</p>

Tomamos el diagrama de conexiones del cerebro de *Drosophila* (conectoma de FlyWire, unas
139.000 neuronas), recortamos un trozo y construimos una red recurrente artificial que
**solo puede usar las conexiones que existen en la mosca**, respetando qué neuronas excitan y
cuáles inhiben (ley de Dale). Después la entrenamos para predecir un flujo de Navier–Stokes
en 2D un paso hacia delante.

**La pregunta:** ¿el cableado real aporta algo frente a un cableado aleatorio con las mismas
estadísticas?

> **No es una mosca.** Nada de esto es una mosca ni un modelo de cómo piensa una mosca. El
> conectoma se usa como plano de una red artificial.

## Predicción registrada de antemano

Escrita antes de ver ningún resultado ([diseño, §3](diseno_mosca_navier_stokes.md)):

| | Hipótesis |
|---|---|
| **H0** | El cableado de la mosca rinde igual que uno aleatorio de la misma densidad |
| **H1** | Cualquier ventaja se explica por la distribución de grados (mosca ≈ recableado con grados conservados) |
| **H2** | La estructura específica importa (mosca > recableado, con significación) |

Esperamos **H0 o H1**, y que un Fourier Neural Operator (FNO) gane a todas las redes
cableadas. Un resultado nulo es un resultado válido.

## Los siete modelos

Entre los modelos cableados solo cambia la máscara de conectividad; codificador,
decodificador, receta de entrenamiento y presupuesto de parámetros son comunes.

| Modelo | Núcleo recurrente |
|---|---|
| `mosca` | Subgrafo de FlyWire, signos reales |
| `aleatorio` | Erdős–Rényi, misma densidad y fracción inhibidora |
| `recableado` | Grafo de la mosca con aristas barajadas, **grados conservados exactamente** |
| `denso` | Todas contra todas, menos neuronas, mismo número de pesos entrenables |
| `sin_nucleo` | Mismas neuronas, signos y puertos que la mosca, **cero conexiones** (control) |
| `fno` | Fourier Neural Operator (referencia superior) |
| `persistencia` | Predice "no cambia nada" (referencia inferior) |

Recorrido: campo de vorticidad → FFT → modos de Fourier más bajos → neuronas *sensoriales*
(mayor grado de salida) → núcleo recurrente con máscara (4 subpasos internos) → neuronas
*motoras* (mayor grado de entrada) → modos → FFT inversa → **incremento** que se suma al campo.

`sin_nucleo` es el control que dice si hay algo que discutir: si la mosca no le gana, lo que
predice no sale de su cableado, sino del codificador y el decodificador que comparten todos.

## La segunda tarea: Mario

El mismo núcleo cableado, jugando en vez de prediciendo ([diseño, §13](diseno_mosca_navier_stokes.md)).
Dos entornos con **la misma observación simbólica de 302 números**, así que la política es la
misma en los dos: un motor de plataformas propio con niveles procedurales, y el Super Mario
Bros de NES real con emulador, leyendo el mapa de casillas de la RAM. El visor anatómico
(`visor_nes.py`) pone el juego a un lado y, al otro, las 139.000 neuronas del conectoma con
las del subgrafo encendiéndose encima, en sus coordenadas reales de FlyWire.

<p align="center">
  <img src="docs/media/mario_referencias.gif" alt="Las cuatro políticas de referencia jugando en el entorno de plataformas" width="100%">
</p>

Y la red cableada como la mosca jugando al Super Mario Bros de NES, con su cerebro al lado
(300 iteraciones de PPO, política muestreada). Honestamente: esta demo entrena y mide en el
mismo nivel, le gana al salto constante (p = 0,008) y **no se distingue de apretar botones
al azar** (p = 0,25).

<p align="center">
  <img src="docs/media/mosca_juega_nes.gif" alt="La red cableada como la mosca jugando a Super Mario Bros, con la actividad de sus neuronas al lado" width="100%">
</p>

Toda cifra se lee contra cuatro líneas de referencia (azar, siempre a la derecha, salto
constante, heurística escrita a mano) o no se lee: **PPO cae solo en la política degenerada**
de saltar siempre a la derecha, que ya recorre el 38 % del nivel. En el ensayo de portátil
los seis modelos empatan y ninguno le gana a esa política.

**El experimento de Mario se cierra sin resolución.** El núcleo *sí* puede representar una
buena política: clonada de la heurística llega al 0,82-0,84 del nivel, y PPO la conserva si
arranca de ahí. Pero desde cero PPO no la encuentra nunca, **y un MLP libre tampoco**. El
freno es de exploración, no de cableado, así que un empate entre seis modelos pegados a la
misma política degenerada no diría nada del conectoma. No se lanzan los 36 entrenamientos
en GPU de esta tarea.

Por imitación, la red cableada como la mosca sí se pasa el 1-1 de NES (0,947 del nivel,
bandera en 4 de 12 partidas): [vídeo](docs/media/mosca_clonada_1-1.mp4). Es una demo, no
ciencia: mismo nivel, copiado de una maestra escrita a mano, sin controles.

## Estado

**En curso. Todavía no hay resultado final.** El experimento real (N = 4.000 neuronas, malla
64×64, 3 semillas por modelo) está corriendo en GPU, con 8 de 18 entrenamientos hechos. La
primera semilla ya le da la vuelta a la lectura del ensayo: a escala real el núcleo de la
mosca **sí** se despega de no hacer nada, un 11 % mejor que la persistencia y un 4 % mejor
que el control sin conexiones en un despliegue de 50 pasos, donde en el ensayo perdía. Con
una sola semilla es prometedor, no confirmado; mosca contra aleatorio y recableado necesita
los 18. Todas las cifras, también las que salieron mal, están en
[`RESULTADOS.md`](RESULTADOS.md).

<!-- RESULTADOS -->

## Lo que hemos aprendido por el camino

- **Un bug en el aumento de datos**: la traslación libre se aplicaba sobre el eje que lleva
  el forzado `sin(4y)`, así que los pares aumentados tenían un 4 % de contenido físicamente
  falso cuando la señal que hay que aprender es ~10 % del campo. Lo detectó un test contra el
  simulador; se descartó todo lo entrenado antes con aumento.
- **Puertos injustos**: dejar que las neuronas de entrada y de salida se solaparan libremente
  daba a la mosca 159 atajos entrada→salida y al aleatorio 18. Ahora el solape es 92 en todos
  los modelos.
- **El control de cero conexiones**: a escala de ensayo (N=500, 32x32, 20.000 pasos) la
  mosca le gana a `sin_nucleo` —las mismas neuronas sin ninguna conexión— por 0,0004 de
  error. Es el número que dice si hay algo que discutir, y de momento dice que no: a esta
  escala lo que predice el modelo no lo hace su recurrencia.
- **El techo del diseño**: incluso un núcleo perfecto solo bajaría el error a un paso de
  0,105 a 0,083 a 64×64, porque el codificador solo ve los modos de Fourier más bajos. Hay
  margen para distinguir cableados, pero es estrecho.

## Ejecutarlo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # en GPU, instalar la rueda de torch con CUDA
pytest -q

python -m data.download_flywire           # conectoma, 105 MB
python -m data.download_annotations       # posiciones de las neuronas para el visor, 32 MB
python -m data.simulate_ns   --config configs/ns_32_proto.yaml
python -m connectome.subgraph --config configs/conectoma_500_proto.yaml
python benchmark.py --config configs/ensayo_32.yaml          # coste medido antes de lanzar
python train.py    --config configs/ensayo_32.yaml
python evaluate.py --config configs/ensayo_32.yaml
python visor.py    --config configs/ensayo_32.yaml --models mosca aleatorio   # HTML interactivo
python video.py    --config configs/ensayo_32.yaml --models mosca              # el vídeo de arriba
```

Y la tarea de Mario:

```bash
pip install nes-py gym-super-mario-bros        # solo para el entorno de NES
python train_rl.py    --config configs/mario_proto.yaml        # entorno propio, prototipo
python evaluate_rl.py --config configs/mario.yaml --muestrea --repeticiones 6
python diagnostico_rl.py artifacts/mario_nes_demo/mosca_s0.pt   # ¿ve algo el crítico? ¿usa memoria?
python clonado_rl.py  --config configs/mario_proto.yaml        # ¿puede representar la heurística?
python visor_nes.py   --config configs/mario_nes_demo.yaml     # juego + cerebro anatómico
```

Todo se configura por YAML; las semillas y la configuración se escriben en cada salida.

El experimento real no cabe en un portátil: la receta completa para la máquina con GPU está
en [`TRASPASO.md`](TRASPASO.md).

## Datos y créditos

- Conectoma: FlyWire, materialización v783 — Dorkenwald et al., *Nature* 2024; Schlegel et
  al., *Nature* 2024. Conectividad y signos tal como los empaqueta Shiu et al., *Nature* 2024
  ([philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)).
- Posiciones de las neuronas: [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations).
- Los datos descargados no se redistribuyen en este repositorio.

Licencia MIT.
