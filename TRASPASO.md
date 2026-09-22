# Traspaso al PC con GPU

Lo que no cabe en el portátil y hay que ejecutar en la máquina con la RTX 3060 Ti. El
código detecta el dispositivo solo (`device: auto` en todos los YAML); no hay nada que
tocar en el código, solo ejecutar en orden.

Al cierre del 22-09-2026: nada corriendo, `pytest -q` en verde (96 tests), todo el trabajo
en la rama `fase-6-mario`.

## 0. Resumen de lo que hay que lanzar allí

| Tanda | Config | Entrenamientos | Se mide con |
|---|---|---|---|
| Fase 4 — fluidos | `configs/entrenamiento_64.yaml` | 6 modelos x 3 semillas = 18 | `evaluate.py` |
| Fase 6 — plataformas | `configs/mario.yaml` | 6 x 3 = 18 | `evaluate_rl.py` |
| Fase 6b — NES | `configs/mario_nes.yaml` | 6 x 3 = 18 | `evaluate_rl.py` |

`persistencia` no se entrena, es referencia. **La fase 4 es la que responde a la pregunta
del proyecto**; las otras dos son la sección 13 del diseño y van después.

## 1. Preparar

```bash
git clone https://github.com/ddaikodaiko/Mosca && cd Mosca
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # la rueda por defecto de torch ya trae CUDA
pytest -q                                # 96 tests, todos deben pasar
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Python 3.13 en el portátil; cualquier 3.11+ vale. `requirements.txt` incluye ya el entorno
de NES de las fases 6 y 6b (la ROM viene dentro del paquete `gym-super-mario-bros`; no se
descarga nada de ningún sitio) con `pyglet` fijado a 1.5.21, porque `nes-py` rompe con
pyglet 2.x.

Hasta el 22-09-2026 `requirements.txt` pedía `torch==2.14.0+cpu`, que **no existe en PyPI**:
la instalación habría fallado ahí mismo. Ya va sin la etiqueta local.

Si `torch.cuda.is_available()` sale `False`, el driver de NVIDIA se queda corto para la
rueda por defecto: `pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu124`
y probar otra vez. **No lanzar nada hasta que esa línea imprima el nombre de la 3060 Ti**,
porque `device: auto` cae a CPU sin quejarse y lo que deberían ser horas serían días.

## 2. Datos que hay que regenerar allí (no se copian)

El set real de 64x64 son ~900 MB: se regenera, no se transfiere. Es **bit a bit
reproducible con el mismo `batch_size`**, que forma parte del contrato y se registra en
`meta.json` (ver la nota de la fase 1 en RESULTADOS.md).

```bash
python -m data.simulate_ns --config configs/ns_64.yaml     # ~13 min de CPU; menos en GPU
```

Semillas: train 1000-1199, val 2000-2039, test 3000-3039. Disjuntas por construcción.

## 3. Lo que SÍ hay que copiar (menos de 3 MB)

```
artifacts/conectoma_4000/mosca.npz
artifacts/conectoma_4000/aleatorio.npz
artifacts/conectoma_4000/recableado.npz
artifacts/conectoma_4000/stats.json
artifacts/conectoma_4000/controls_stats.json
```

Se pueden regenerar con los tres comandos de abajo (la semilla está en el YAML, así que
sale el mismo subgrafo), pero copiarlos evita volver a bajar los 104 MB del conectoma:

```bash
python -m data.download_flywire
python -m connectome.subgraph --config configs/conectoma_4000.yaml
python -m connectome.controls --config configs/conectoma_4000.yaml
```

**Si se regenera el subgrafo, se rehacen los seis modelos**: el subgrafo depende de la
semilla y los seis tienen que compartir exactamente el mismo (nota de la fase 2).

## 4. Medir el coste ANTES de lanzar las tandas

Esto es obligatorio, no opcional: en esta máquina un solo entrenamiento de fluidos son
12,3 h de CPU, y la única forma de saber qué sale en la 3060 Ti es medirlo.

```bash
python benchmark.py --config configs/entrenamiento_64.yaml          # fases 4 y 5
```

Imprime, por modelo, segundos por paso de la fase A y de la fase B, **pico de memoria de
GPU** y una cota de horas por semilla, y escribe `artifacts/train_64/benchmark.json`.
Funciona sin los datos generados (usa campos aleatorios de la misma forma y lo dice).

Referencia de este portátil (CPU, 6 hilos, 64x64, N=4.000, datos sintéticos), que es
contra lo que hay que comparar lo que salga en la 3060 Ti:

| modelo | s/paso fase A | s/paso fase B (h=20) | h por semilla |
|---|---|---|---|
| mosca | 0,363 | 6,13 | 12,24 |
| aleatorio | 0,312 | 6,33 | 12,28 |
| recableado | 0,341 | 5,97 | 11,84 |
| denso | 0,164 | 3,21 | 6,25 |
| `sin_nucleo` | 0,125 | 2,62 | 5,06 |
| fno | 1,029 | 7,33 | 17,94 |

Los 18 entrenamientos son 197 h aquí, y **la fase B se lleva el 80 %**. El despliegue es
secuencial (20 pasos encadenados de cuatro subpasos cada uno), así que es lo que peor
acelera en GPU: si allí sigue dominando, la palanca es `phase_b.steps_per_stage` o recortar
el currículo, no la fase A. Esa corrida del benchmark vale además de prueba de humo: los
seis modelos construyen, propagan y retropropagan a escala real sin fallar.

Para las fases 6 y 6b, que no pasan por `benchmark.py`:

```bash
python train_rl.py --config configs/mario_nes.yaml --model mosca --seed 0 --iterations 3
```

Cronometra las primeras iteraciones y extrapola las 800. Con `--iterations` la salida va a
un subdirectorio `medicion_<n>it/`, así que no pisa los checkpoints de verdad.

**Los 8 GB de la 3060 Ti.** El núcleo con máscara es disperso y el lote es pequeño, así que
lo esperable es que sobre memoria de largo; el número que manda es el `pico MB` del
benchmark. Si algo diera OOM, el orden para recortar es: `phase_b.batch_size`, luego
`phase_a.batch_size` (y entonces hay que decirlo en el informe: el tamaño de lote es parte
del contrato de reproducibilidad). En PPO, el que pesa es `ppo.n_envs` x `rollout_steps`.

## 5. Fase 4 y fase 5 — fluidos, la pregunta del proyecto

```bash
python train.py    --config configs/entrenamiento_64.yaml
python evaluate.py --config configs/entrenamiento_64.yaml
```

### Antes de las 18, las 3 que deciden

No lances la tanda entera de primeras. Con **tres entrenamientos** (una sola semilla) ya
sabes si el experimento tiene resolución, y cuestan una fracción de las 18:

```bash
python train.py --config configs/entrenamiento_64.yaml --model mosca      --seed 0
python train.py --config configs/entrenamiento_64.yaml --model sin_nucleo --seed 0
python evaluate.py --config configs/entrenamiento_64.yaml     # persistencia sale sola
```

`evaluate.py` imprime ya los dos contrastes pareados que deciden: mosca contra
persistencia y mosca contra `sin_nucleo`, a un paso y en despliegue. Si la mosca no se
despega de las dos, las quince restantes miden un empate que ya conocemos, y lo que toca
es rediseñar el codificador, no gastar la GPU. En el ensayo de 32x32 la mosca le ganaba a
la persistencia por 0,0008 a un paso **y le perdía por 0,629 en despliegue**: eso es no
tener resolución.

Si se despega, entonces sí, la tanda entera.

### La tanda entera

Mejor aún, la tanda entera de uno en uno, con un log por entrenamiento y saltando lo que
ya esté terminado. Se puede cortar y relanzar las veces que haga falta (después de un
reinicio, por ejemplo) y no repite nada:

```bash
scripts/tanda.sh configs/entrenamiento_64.yaml               # 18 entrenamientos
tail -f artifacts/train_64/logs/mosca_s0.log                 # ver cómo va
python evaluate.py --config configs/entrenamiento_64.yaml    # lee los checkpoints sueltos
```

La marca de "terminado" es el informe `train_report_<modelo>_s<n>.json`, que se escribe al
acabar; el checkpoint no vale de marca porque en PPO se guarda también a medias. Si un
modelo falla, la tanda sigue con los demás y lo dice en el resumen final.

A mano es lo mismo, un trozo cada vez; cada uno escribe su propio informe sin pisar los
otros:

```bash
python train.py --config configs/entrenamiento_64.yaml --model mosca --seed 0
```

El YAML ya no tiene decisiones pendientes: `augment.continuous_x` se queda en `false`
porque la ablación del 22-09-2026 midió que a 32x32 no cambia nada (0,0994 a un paso con la
traslación continua y con la discreta). Está en RESULTADOS.md.

### Cómo se lee el resultado

1. **Primero, contra la persistencia.** Si el núcleo con máscara no se despega de ella (en
   el prototipo: mosca 0,0999, aleatorio 0,0998, persistencia 0,1001), el experimento **no
   tiene resolución** y el empate mosca-aleatorio **no se publica como H0 confirmada**. Hay
   que rediseñar el codificador.
2. **Después, contra `sin_nucleo`.** Si la mosca no le gana, lo que predice no sale del
   cableado. A escala de ensayo le gana por 0,0004 (0,0994 contra 0,0998), que es no
   ganarle.
3. **Solo entonces**, mosca contra aleatorio y contra recableado, con la permutación
   pareada que ya hace `evaluate.py`.
4. El error espectral, **por bandas y en log**: el 91 % de la energía está en k<=4 y en
   crudo la métrica no discrimina. La banda k>9 no discrimina por construcción (el
   codificador no la ve) y se informa aparte.

## 6. Fases 6 y 6b — Mario

```bash
scripts/tanda.sh      configs/mario.yaml     mario            # entorno propio
python evaluate_rl.py --config configs/mario.yaml --muestrea --repeticiones 6

scripts/tanda.sh      configs/mario_nes.yaml mario            # NES con emulador
python evaluate_rl.py --config configs/mario_nes.yaml --muestrea --repeticiones 6
```

Tres cosas que no se pueden saltar al leer estos números:

- **Las cuatro líneas de referencia** (azar, siempre derecha, salto constante 0,381,
  heurística 0,752) salen siempre en el informe. Un número de modelo sin ellas no se lee.
  En el prototipo ningún modelo le ganó al salto constante.
- **La evaluación muestreada necesita `--repeticiones 6` o más.** Con una partida por nivel
  el orden entre modelos se da la vuelta solo por ruido de muestreo.
- **Los niveles de NES son fijos**: `envs/nes_mario.STAGES` reparte los 32 en conjuntos
  disjuntos. Cualquier cifra medida en un nivel entrenado no vale.
  `configs/mario_nes_demo.yaml` entrena y mide en 1-1 **a propósito, solo para grabar
  vídeo**, y está marcado como tal.

**La 3060 Ti va a ayudar poco aquí, y conviene saberlo antes.** Medido el 22-09-2026 en
este portátil con la config real de NES (N=4.000, 16 entornos, 6 hilos): 44,2 s por
iteración, de los cuales 27,9 s (el 63 %) son recolección en el emulador, que es CPU pura.
La GPU solo muerde el 37 % restante. Las 800 iteraciones son ~9,8 h por modelo y semilla
aquí, ~177 h los 18. Si hay que acelerar, lo que mueve la aguja es más paralelismo de CPU
(subir `ppo.n_envs`, o lanzar varios modelos a la vez), no la tarjeta.

En el entorno propio el reparto es el contrario —la actualización se llevaba el 80 %—
porque ahí no hay emulador. Esa sí se acelera.

Las dos configs arrancan y completan una iteración entera a escala real; comprobado con
`--iterations 1`, así que lo que quede por fallar no es la construcción del modelo.

## 7. Qué traer de vuelta

```
artifacts/train_64/train_report*.json
artifacts/train_64/eval_report.json
artifacts/train_64/benchmark.json
artifacts/train_64/figs/*.png
artifacts/mario*/rl_report*.json
artifacts/mario*/rl_evaluacion.json
artifacts/*/checkpoints/*.pt      (opcional, pesan)
```

Con eso se actualizan `RESULTADOS.md` y la sección de estado del README.

## 8. Lo que sigue abierto, por orden de valor

1. **El gradiente de la fase 6 está truncado a un paso de entorno**: la h persiste durante
   el despliegue pero entra desacoplada en la actualización, así que el núcleo no puede
   construir memoria de más de un paso del juego. Es el sospechoso principal de que nada
   aprenda. Cambiarlo es una desviación del diseño: **hay que consultarlo antes**.
2. **La saturación por crecimiento de los pesos del codificador** (66 % de las sensoriales
   con |impulso| > 2,5 tras 30 iteraciones). Normalizar la entrada, que ya está hecho, no
   lo arregla; normalizar el impulso sí, y también es desviación del diseño.
3. **Qué hacer si el núcleo no se despega de la persistencia en la fase 4** (punto 5.1).
4. **Réplica con cascada de enstrofia resuelta**, solo después de tener el resultado a
   64x64: malla >=256, hiperviscosidad y ~2000 modos en el codificador. ~80x en simulación
   y ~15 GB de datos, en los seis modelos por igual.

El checkpoint de la demo de NES (`artifacts/mario_nes_demo/mosca_s0.pt`, iteración 190 de
600) es anterior a la normalización de entrada y avisa al cargarlo. No hay reanudación: si
se quiere, se relanza entero.
