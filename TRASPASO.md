# Traspaso al PC con GPU

Lo que no cabe en el portátil y hay que ejecutar en la máquina con la RTX 3060 Ti. El
código detecta el dispositivo solo (`device: auto` en todos los YAML); no hay nada que
tocar en el código, solo ejecutar en orden.

**Hashes de commit (21-10-2026):** ese dia se reescribieron los mensajes de dos commits
antiguos; el codigo no cambio, pero 25 commits cambiaron de hash. Los documentos citan
ya los nuevos. Los `.tar.gz` del bucket conservan el nombre de antes.

**Al 21-10-2026, 19:32 (hora del PC): la tanda de fluidos ha terminado.** Los 18
entrenamientos y `evaluate.py`, codigo 0. Resultado en RESULTADOS.md (seccion del
21-10-2026, fases 4 y 5 a escala real): la mosca gana a la persistencia y a `sin_nucleo` y
pierde contra el aleatorio y el recableado. Informes y figuras bajados a
`artifacts/train_64_pc/`. El `eval_report.json` del PC estaba en cp1252 (Windows escribe
asi por defecto); convertido, y `evaluate.py`, `train.py` y `benchmark.py` escriben ya en
UTF-8. **El PC no tiene nada que ejecutar.**

**Al 21-10-2026, tarde: la 6e ha terminado y esta leida** (11:55 UTC, codigo 0;
veredicto en RESULTADOS.md: el reescalado espectral queda descartado como causa). Bajada
en `artifacts/mario_proto_6e_row_gcp/`. **En el PC se lanzo la tanda de fluidos** a las
17:48 (hora del PC): habia 15 de 18 hechos, no 8; faltaban `denso s0`, `fno s1` y
`fno s2`, que reanudan de su instantanea. El lanzador
(`artifacts/train_64/tanda_21oct.cmd`, en el PC) corre `scripts/tanda.sh` y despues
`evaluate.py`; log en `artifacts/train_64/tanda_21oct.log`. Los logs y el
`eval_report.json` de antes quedan al lado como `*.antes_21oct.*`. El codigo de fluidos
del PC (2772f6d) es identico al de HEAD. Se lanzo por SSH, no como tarea programada: si
se corta la conexion, se relanza el mismo `.cmd` y sigue donde iba.

**Google Cloud vaciado el 21-10-2026:** borradas las VMs `mosca-6c` y `mosca-6c-b` (con
sus discos) y la imagen `mosca-6c-img`. Solo queda el bucket. Lo que se dice abajo de
borrar VMs e imagen ya esta hecho.

**Al 21-10-2026: la 6d ha terminado** (20-10, 17:20 UTC, codigo 0; veredicto en
RESULTADOS.md, seccion del 21-10-2026). Todo lo del bucket esta bajado en
`artifacts/mario_proto_6c_soltar_gcp/`. No hay nada corriendo ni en Google Cloud ni en el
PC. Pendiente: **borrar las dos VMs y la imagen** si no se van a reutilizar (paradas
siguen cobrando el disco).

**Al 19-10-2026: la 6c ha terminado (veredicto en RESULTADOS.md) y corre la fase 6d en
Google Cloud.** Clonado de la heuristica + PPO desde el clonado, seis modelos x semillas
0-2, entorno propio con `release_to_jump`, en la VM `mosca-6c` (8 vCPU), con el codigo de
`ea8865f` y el plan de analisis ya escrito en RESULTADOS.md. Lanzador
`scripts/gcp/run_6d.sh`; la VM se apaga sola. El PC sigue sin nada que ejecutar.

**20-10-2026: la primera tanda se colgo y se relanzo a las 16:26 UTC.** El clonado
termino a las 17:47 UTC del 19-10, pero el `wait` sin argumentos tambien esperaba al
bucle de subida y, en bash 5.2, a la sustitucion de proceso del `tee` del log. Ninguno de
los dos acaba nunca, asi que la VM se quedo 22 h encendida sin hacer nada. Ahora se
espera solo a los PID del clonado. Se relanzo desde cero con el mismo commit de codigo
(`ea8865f`); lo de la tanda colgada queda en `~/*_colgada` dentro de la VM.

```bash
B=gs://mosca-runs-317660842762/6d
gcloud storage cat $B/logs/vm.log | tail                     # por donde va
gcloud storage cp -r $B/artifacts artifacts/mario_proto_6c_soltar_gcp   # al acabar
```

Al terminar la 6d hay que **borrar las dos VMs y la imagen** (el disco parado tambien
cobra), salvo que se vayan a reutilizar.

**Al 14-10-2026 el PC con GPU NO tiene nada que ejecutar.** La unica prioridad son las
**replicas de la fase 6c**, y corren en Google Cloud. La s0 (mosca + MLP) termino el
13-10-2026 (resultado en RESULTADOS.md, seccion del 14-10-2026). Desde el 14-10-2026 a las
06:56 UTC entrenan los seis modelos del diseno con las semillas 0, 1 y 2 (faltaban 16), con
el codigo exacto de la s0 (2772f6d), en dos VMs de `europe-west1-b` que comparten una cola
en el bucket:

- `mosca-6c` (c2d-highcpu-8, ~2,7 h por modelo) y `mosca-6c-b` (c2d-highcpu-4, ~4 h),
  creada a partir de la imagen `mosca-6c-img`. Unas 28 h en total.
- Lanzador `~/run_replicas_v4.sh` en cada VM (`scripts/gcp/`). La ultima en acabar lanza
  la evaluacion final (val+test y los niveles extra, muestreada y codiciosa, con el control
  marginal) y `analiza_6c.py`, y las dos se apagan solas. El plan de analisis y su
  enmienda estan fijados en RESULTADOS.md **antes** de ver las replicas. A las 11:40 UTC
  del 14-10 iban 3 de 16; fin estimado hacia las 03:30 UTC del 15-10, mas ~40 min de
  evaluacion. El informe legible queda en `$B/logs/analisis_6c_muestreada.md`.

```bash
B=gs://mosca-runs-317660842762/6c-replicas
gcloud storage ls $B/claims/ $B/hechos/              # reclamados / terminados (16 en total)
gcloud storage cat $B/logs/mosca-6c.log $B/logs/mosca-6c-b.log
gcloud storage cat $B/logs/<modelo>_s<semilla>.log | tail
gcloud storage cat $B/logs/analisis_6c_muestreada.md      # el veredicto, al acabar
gcloud storage cp -r $B/artifacts/mario_nes_sola artifacts/   # al acabar, con los JSON finales
```

Al terminar hay que **borrar las dos VMs y la imagen** (el disco parado tambien cobra).
**No lanzar nada de la 6c en el PC** y **no lanzar fluidos (fases 4 y 5)**: siguen
aparcados hasta leer el resultado de las replicas. Todo lo que hay debajo es historico.

---

## FASE 6c — QUE LA MOSCA APRENDA A JUGAR SOLA (28-09-2026) — HISTORICO: ahora corre en Google Cloud, ver arriba

Encargo del usuario: que juegue a Mario **aprendiendo sola**, sin imitar a nadie, y que
juegue en niveles que no ha visto. Rama `main` (= `fase-6c-sola`), `pytest -q` en verde (128 tests).
Todo esta en `configs/mario_nes_sola.yaml`, que explica cada cambio. Resumen: la
observacion lleva velocidad vertical y A pulsado, la mitad de los episodios de
entrenamiento arrancan poco antes de donde mas lejos llego ella misma, la entropia baja
de 0,02 a 0,002, y los 16 emuladores corren en procesos aparte.

**Hace falta el cableado del prototipo** (96 KB, git-ignored). Se copia
`artifacts/conectoma_500_proto/` entero o se regenera:

```bash
python -m connectome.subgraph --config configs/conectoma_500_proto.yaml
python -m connectome.controls --config configs/conectoma_500_proto.yaml
```

**1. Medir el coste** (en el portatil: 10-11 s por iteracion, el 95 % emulando):

```bash
python train_rl.py --config configs/mario_nes_sola.yaml --model mosca --seed 0 --iterations 3
```

**2. La corrida.** Mosca y MLP libre con la receta identica, 1.500 iteraciones cada uno
(en el portatil serian ~4,5 h por modelo). Reanudable si se corta:

```bash
python train_rl.py --config configs/mario_nes_sola.yaml --model mosca --seed 0
python train_rl.py --config configs/mario_nes_sola.yaml --model mlp   --seed 0
```

Mirar en la salida, cada 25 iteraciones: `val progreso` (niveles NO entrenados) y `H`.
**Vigilar el KL** (en el `rl_report`): en la prueba de 30 iteraciones seguia en ~1e-5, el
mismo sintoma de "la politica no se mueve" de la fase 6. Si a las 300 iteraciones el KL
sigue ahi y el progreso de train no sube, parar y traerlo: la receta no basta.

**Medido en el PC grande al cierre del 28-09-2026, iteracion 475 de 1.500 (mosca, s0):** el
KL **si se mueve** (no es el sintoma viejo). Promedios de KL por bloques de 50 iteraciones:
6,2e-5 / 5,5e-5 / 4,9e-5 / 4,2e-5 / 3,8e-5 / 6,4e-5 / **8,2e-5** (iter 0-300, subiendo otra
vez al final) — un orden de magnitud por encima del 1e-5 fijo de la fase 6 vieja. El
progreso de train sube con ruido pero con tendencia clara: 0,165 -> 0,183 -> 0,189 -> 0,197
-> 0,198 -> 0,202 -> 0,215 (medias por bloques de 50) y sigue subiendo hasta 0,29-0,31 hacia
la iteracion 450. El val progreso tambien se despega un poco de la linea base de 0,098
("saltar siempre"): ronda 0,12-0,13 desde la iteracion ~200. Con el checkpoint de esa
iteracion, una partida suelta en 1-1 (entrenado) dio 9,7 % y en 1-3 (NO entrenado) dio 13,1
%: parecidos entre si, osea que no esta memorizando un nivel concreto, y ninguno es meta
todavia. **Se corto para apagar la maquina en la iteracion ~475-500** (sin instantanea
perdida: `resume_every: 10`). Retomar con el mismo comando de la seccion 2; sigue
`--model mlp` despues, automatico si se usa el mismo `&&` de antes.

**Aviso al apagar (28-09-2026):** al parar la 6c con `taskkill` aparecio un
`python.exe ... train.py --config configs/entrenamiento_64.yaml --model fno --seed 2`
corriendo sin que estuviera en ningun log de esta sesion — sobra de una tanda de fluidos
anterior que no se habia cerrado del todo (parar el proceso de shell con `TaskStop` no mata
siempre al `python.exe` hijo en Windows; hay que comprobarlo con `tasklist | grep python` y
matar el PID a mano). Se mato tambien. Antes de lanzar nada nuevo en esta maquina,
**comprobar `tasklist` y `nvidia-smi` primero** para no competir por GPU/CPU con un proceso
que se creia parado.

**3. Evaluar contra las referencias** y traer el JSON:

```bash
python evaluate_rl.py --config configs/mario_nes_sola.yaml --muestrea --repeticiones 6
```

Se lee asi: sabe jugar si en val le gana con claridad a `azar` y al salto constante; la
heuristica (se pasa el 1-1, muere pronto en el resto) es la referencia alta. Si solo
mejora en train, ha aprendido esos niveles, no a jugar: se dice asi.

---

## PENDIENTES PARA EL PC GRANDE (al cierre del 23-09-2026) — HISTORICO, aparcado: ver la 6c arriba

Rama `fase-6-mario`, `pytest -q` en verde (108). Nada corriendo en el portatil. Lo que el
portatil dejo hecho el 23-09 esta en RESULTADOS.md (tres secciones con esa fecha). Los
checkpoints no viajan por git: lo que haga falta se regenera con los comandos de abajo.

**1. Terminar la tanda de 18 de la fase 4 y evaluarla.** Es lo unico que responde a la
pregunta del proyecto. Sigue igual que en la seccion siguiente (8 de 18 al cierre del
23-09-2026, con `recableado s1` a medias):

```bash
bash scripts/tanda.sh configs/entrenamiento_64.yaml
python evaluate.py --config configs/entrenamiento_64.yaml
```

**2. Demo de NES: afinar con PPO la mosca clonada del 1-1.** En el portatil se clono (0,947
de progreso, bandera 4 de 12) pero afinar con PPO son ~50 min de CPU y no se lanzo. En el
PC grande el cuello sigue siendo el emulador (CPU), ~40 min. Primero regenerar la clonada
(~7 min, el checkpoint se quedo en el portatil) y comprobar que da ~0,95:

```bash
python clonado_rl.py --config configs/mario_nes_demo.yaml --modelos mosca \
    --niveles-demo 40 --niveles-val 3 --niveles-juego 4 --ruido 0.15 --ventana 8 \
    --lote 64 --pasos 4000 --guarda-pesos \
    --salida artifacts/mario_nes_demo/clonado/clonado_s0.json
python train_rl.py --config configs/mario_nes_demo.yaml --model mosca --seed 0 \
    --desde artifacts/mario_nes_demo/clonado/mosca_s0.pt
python evaluate_rl.py --config configs/mario_nes_demo.yaml --levels 12 \
    --checkpoint artifacts/mario_nes_demo/clonado/mosca_s0.pt \
    --checkpoint artifacts/mario_nes_demo/desde_clonado/mosca_s0.pt
```

Lectura: el objetivo es que pase de la escalera final (donde se atasca en 8 de 12). Ojo:
PPO entrena con la h desacoplada entre pasos y el clonado aprendio a usar memoria de 8
pasos; si PPO empeora la clonada, puede ser por eso, y se dice. Si mejora, grabar video
nuevo con `visor_nes.py --checkpoint .../desde_clonado/mosca_s0.pt --stage 1-1 --record
docs/media/...` eligiendo una semilla que llegue a la bandera. Todo esto es **demo**: mismo
nivel e imitacion.

**3. (Barato, opcional) Control de cableado en la demo.** El mismo clonado de arriba con
`--modelos mosca,aleatorio,recableado,denso,sin_nucleo,mlp` (~6x el tiempo, ~40 min). Da
una tabla con controles en vez de un video suelto. Sigue siendo el mismo nivel: no vale
como resultado de la fase 6b, solo como diagnostico.

**4. DECISION DEL USUARIO antes de tocar codigo: la fase 6 de verdad.** Medido el 23-09:
PPO desde cero no encuentra la politica buena (ni la mosca ni el MLP en el prototipo),
pero si la conserva si arranca de ella. Las tres salidas (arrancar del clonado, explorar
mas con entropia o curriculo, o cerrar la fase como sin resolucion) estan al final de la
seccion "PPO no destruye la politica buena" de RESULTADOS.md. Las tres son desviacion del
diseno. En el NES ademas la heuristica solo se pasa el 1-1, asi que "arrancar del
clonado" en niveles disjuntos exige antes una maestra que generalice.
**Decidido el 24-09-2026: la fase 6 se cierra como "sin resolucion".** Todos se quedan
pegados a la politica degenerada, y ese empate no dice nada del cableado. No se lanzan los
36 entrenamientos de las fases 6 y 6b; la GPU queda para la fase 4. La demo de NES (puntos
2 y 3) sigue valiendo como video. Si mas adelante se quiere reabrir, la via mas barata es
arrancar del clonado en el entorno propio (opcion 1 de RESULTADOS.md).

---

## Actualizacion, cierre de la primera sesion en la GPU (22-09-2026 noche)

Todo lo de abajo (puntos 1-6) ya se hizo una vez en la 3060 Ti. Quedan dos cosas abiertas
para la proxima sesion, **en esta misma maquina** (los datos y checkpoints no viajan por
git, `artifacts/` esta en `.gitignore`):

1. **Terminar la tanda de 18 de la fase 4.** Al cierre del 23-09-2026 hay **8 de 18
   terminados** (mosca x3, aleatorio x3, recableado s0, sin_nucleo s0) y `recableado s1` en
   marcha; quedan denso x3, sin_nucleo s1/s2, fno x3 y lo que falte de recableado. Cada
   entrenamiento son ~17-19 min. Se corto para apagar la maquina. Retomar con el mismo
   comando de siempre, no pierde nada (instantanea cada 500 pasos):
   ```bash
   bash scripts/tanda.sh configs/entrenamiento_64.yaml
   python evaluate.py --config configs/entrenamiento_64.yaml
   ```
   Las 3 que deciden ya se corrieron a escala real y la mosca **se despega** de persistencia
   y de `sin_nucleo` (detalle en RESULTADOS.md, seccion "22-09-2026: primera sesion en la
   GPU"): es lo que justifico lanzar la tanda completa.
2. **El MLP libre de la fase 6 ya corrio y SI aprende** (varianza explicada +0,494,
   progreso 0,36-0,56). Segun el criterio pre-registrado, el freno esta en el nucleo, no en
   la tarea. El unico sospechoso que queda —el gradiente truncado a un paso de entorno— es
   una desviacion del diseno: **pedir permiso al usuario antes de tocar codigo**, no
   proponerlo solo.
   **Superado (fusion del 24-09-2026):** este punto se escribio en el PC sin los commits
   del portatil del 23-09. El gradiente truncado ya esta descartado y el freno es de
   exploracion; ver la seccion "PENDIENTES PARA EL PC GRANDE", punto 4.
3. **Demo cableada de Mario en la GPU, a medias.** `python train_rl.py --config
   configs/mario_nes_demo_critico.yaml --model mosca --seed 0` iba por la iteracion 120 de
   300 (~8,6 s/it, cuello en el emulador de CPU). Relanzar el mismo comando la retoma
   (`resume_every: 10`; no es bit a bit, los entornos arrancan de cero). Al acabar:
   `python visor_nes.py --config configs/mario_nes_demo_critico.yaml --checkpoint
   artifacts/mario_nes_demo_critico/mosca_s0.pt --stage 1-1 --sample` abre el visor del
   cerebro (usa matplotlib; `watch_nes.py` NO funciona aqui: pyglet 1.5.21 rompe con Python
   3.14).
4. Nota menor: `torch==2.14.0` con CUDA necesita el indice `cu126` o `cu130` en esta
   maquina (Python 3.14), no `cu124` como decia este documento mas arriba — `cu124` no
   tiene build para cp314. `pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126`.

### Tareas para el portatil (sin GPU, mientras la tanda sigue en el PC)

El portatil no puede con la tanda (197 h de CPU), asi que lo suyo es lo que no necesita GPU:

1. `git pull` en `Mosca` y en `Mosca-notas`; traer las notas internas a `Mosca/`.
   **Hecho el 24-09-2026, al reves:** las notas del portatil eran las mas nuevas (tenian
   el 23-09) y son las que suben a `Mosca-notas`.
2. **Grabar el video del visor con el checkpoint bueno que ya tiene el portatil**
   (`artifacts/mario_nes_demo/mosca_s0.pt`): `python visor_nes.py --config
   configs/mario_nes_demo.yaml --checkpoint artifacts/mario_nes_demo/mosca_s0.pt --stage
   1-1 --sample --record docs/media/visor.mp4`. Lo que se ve es el mismo modelo que se midio
   (0,285 de progreso muestreado), no el reentrenado en la GPU.
   **Hecho el 24-09-2026** con `--seed 3`: 26,0 % de progreso, la mas cercana a la media
   de las cuatro semillas grabadas (0: 9,8 %, 1: 45,1 %, 2: 21,9 %, 3: 26,0 %). Se elige
   la tipica y no la mejor, para no vender una partida escogida.
3. **Preparar la decision sobre el gradiente truncado de la fase 6** (unico sospechoso que
   queda tras el MLP libre): escribir la propuesta, sin tocar `train_rl.py` ni
   `models/masked_policy.py`, y llevarsela al usuario. Es desviacion del diseno.
   **Superado:** el gradiente truncado ya no es sospechoso (RESULTADOS.md, 23-09-2026). La
   decision que queda es la de exploracion, punto 4 de "PENDIENTES PARA EL PC GRANDE".
4. Cuando el PC termine la tanda, traer al portatil `artifacts/train_64/train_report*.json`,
   `eval_report.json` y `figs/` para actualizar la seccion de estado del README.
   Las cifras se vuelcan con `python informe_readme.py artifacts/train_64/eval_report.json`
   (tabla y contrastes en los dos README, entre marcas; la lectura se escribe a mano).

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
pytest -q                                # 107 tests, todos deben pasar
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
python -m data.simulate_ns --config configs/ns_64.yaml     # ~13 min de CPU
```

Semillas: train 1000-1199, val 2000-2039, test 3000-3039. Disjuntas por construcción.

**Ojo con dar por hecho que en GPU irá más rápido.** El simulador trabaja en `float64`
(`simulation.precision`, y ahí se queda: es lo que da la deriva de energía del contrato), y
una tarjeta de consumo como la 3060 Ti tiene el `float64` capado a una fracción del
`float32`. Puede salir más lenta que la CPU. Son 13 minutos: se lanza tal cual, y si va
peor, `--device cpu` y listo. Lo que **no** se debe hacer es generar unos splits en GPU y
otros en CPU: el dispositivo cambia los bits y el `meta.json` guarda cuál se usó.

## 3. Lo que SÍ hay que copiar (menos de 3 MB)

```
artifacts/conectoma_4000/mosca.npz
artifacts/conectoma_4000/aleatorio.npz
artifacts/conectoma_4000/recableado.npz
artifacts/conectoma_4000/stats.json
artifacts/conectoma_4000/controls_stats.json
```

Son **1,7 MB en total**: caben en cualquier sitio.

Se pueden regenerar con los tres comandos de abajo (la semilla está en el YAML, así que
sale el mismo subgrafo), pero copiarlos evita volver a bajar los 104 MB del conectoma.
Comprobado el 22-09-2026: las dos descargas siguen vivas y con el tamaño exacto que el
verificador espera (100.804.642 y 3.327.347 bytes), así que regenerar desde cero funciona.
El descargador ya prueba primero el `raw` de GitHub, que es el que sirve; el de LFS
devuelve 404 desde hace tiempo y por eso va el segundo.

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

**Y un entrenamiento cortado no se pierde.** Cada 500 pasos (`resume_every` en el YAML)
`train.py` guarda una instantánea en `checkpoints/<modelo>_s<n>.reanudar.pt` con pesos,
optimizador, planificador, parada temprana y el estado del generador. Si el PC se reinicia
o le das a Ctrl-C, vuelves a lanzar el mismo comando y sigue donde estaba, con los mismos
pesos que si no hubiera parado (hay un test que lo compara bit a bit). Al terminar bien, la
instantánea se borra sola. Así que el PC puede reiniciarse a mitad de la tanda sin drama.

**`train_rl.py` (Mario) también se reanuda**, cada `resume_every` iteraciones, pero con una
promesa más débil y conviene saberlo: los entornos no se pueden serializar —el emulador
lleva su propia RAM— así que al reanudar arrancan de cero y la recolección ya no es la que
habría salido de un tirón. Lo aprendido sobrevive entero (pesos, optimizador, planificador,
contador e historial), pero **un entrenamiento reanudado no se reproduce desde la semilla**,
y por eso el informe anota `reanudaciones`. Si un número tiene que ser reproducible, se
relanza de cero y se deja terminar.

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

### Antes de los 18: mirar si el crítico ve algo

```bash
python train_rl.py      --config configs/mario_nes.yaml --model mosca --seed 0
python diagnostico_rl.py artifacts/mario_nes/mosca_s0.pt --config configs/mario_nes.yaml
```

Imprime la **varianza explicada** del crítico. Si sale por debajo de 0,3, las ventajas son
ruido y los 18 entrenamientos medirían el ruido de seis cableados, no los cableados. En la
demo salía **0,10**, con el crítico prediciendo 27,7 para cualquier estado.

**Medido el 22-09-2026: el arreglo funciona y no desbloquea nada.** Con
`ppo.normalise_returns` la varianza explicada sube a 0,36-0,40, la pérdida de valor cae de
41,4 a 0,27 y la saturación de las motoras baja del 7,8 % al 3,9 %, pero el KL por
actualización sigue en 1e-5 y el progreso no se mueve. Así que la puerta antes de los 18 **no
es solo la varianza explicada**: es que el KL por actualización se despegue de 1e-5. Si el
crítico ve y la política sigue sin moverse, los 18 entrenamientos medirían seis cableados
inmóviles. `normalise_returns` se activa igualmente, y **en los seis modelos por igual**,
porque el crítico ciego era un defecto real.

Tres cosas que no se pueden saltar al leer estos números:

- **Las cuatro líneas de referencia** (azar, siempre derecha, salto constante 0,381,
  heurística 0,752) salen siempre en el informe. Un número de modelo sin ellas no se lee.
  En el prototipo ningún modelo le ganó al salto constante.
- **La evaluación muestreada necesita `--repeticiones 6` o más.** Con una partida por nivel
  el orden entre modelos se da la vuelta solo por ruido de muestreo. Desde el 22-09-2026 ese
  ruido va además acoplado entre modelos (la semilla se refija antes de cada repetición, con
  el mismo valor para todos), así que la comparación es más fina; las cifras de fase 6
  anteriores a esa fecha no son comparables una a una con las nuevas.
- **Para comparar versiones de un mismo entrenamiento** (por ejemplo el mismo modelo a 300 y
  a 900 iteraciones) está `--checkpoint`, repetible: cada fichero entra como una fila contra
  las mismas cuatro líneas de referencia, y el informe va a `rl_evaluacion_checkpoints.json`
  sin pisar el normal.
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

**Lo que ya esta descartado con medida** (tabla completa al final de RESULTADOS.md), para no
volver a gastar GPU en ello: las motoras no estan saturadas; el critico ciego era real, esta
arreglado con `ppo.normalise_returns` y **no era el freno**; el bonus de entropia no aplasta
a la politica (con 10x menos, el KL baja todavia mas, a 1e-6); y la politica **si** depende
de lo que ve (distancia de variacion total al promedio: mediana 0,17, maxima 0,50).

1. **Terminar el control que decide la fase 6: el MLP libre con la receta identica.** Es lo
   que quedo corriendo y cortado en el portatil, y no hay nada guardado. Mismo config de la
   demo, `model: mlp`, mismas iteraciones:

   ```bash
   python train_rl.py       --config configs/mario_nes_demo_critico.yaml --model mlp --seed 0
   python diagnostico_rl.py artifacts/mario_nes_demo_critico/mlp_s0.pt --config configs/mario_nes_demo_critico.yaml
   ```

   Lectura: **si el MLP libre tampoco aprende**, el freno esta en la tarea o en como esta
   planteado PPO, y la fase 6 no tiene resolucion por construccion — no se lanzan los 18.
   **Si el MLP si aprende**, el freno esta en el nucleo, y el sospechoso que queda es el
   punto 2.
2. **El gradiente de la fase 6 está truncado a un paso de entorno**: la h persiste durante
   el despliegue pero entra desacoplada en la actualización, así que el núcleo no puede
   construir memoria de más de un paso del juego. Es el único sospechoso que sigue en pie.
   Cambiarlo es una desviación del diseño: **hay que consultarlo antes**.
3. **La saturación por crecimiento de los pesos del codificador** (66 % de las sensoriales
   con |impulso| > 2,5 tras 30 iteraciones). Normalizar la entrada, que ya está hecho, no
   lo arregla; normalizar el impulso sí, y también es desviación del diseño. Con
   `normalise_returns` la saturacion de las motoras ya baja al 3,9 %, asi que esto pesa
   menos que antes.
4. **Qué hacer si el núcleo no se despega de la persistencia en la fase 4** (punto 5.1).
5. **Réplica con cascada de enstrofia resuelta**, solo después de tener el resultado a
   64x64: malla >=256, hiperviscosidad y ~2000 modos en el codificador. ~80x en simulación
   y ~15 GB de datos, en los seis modelos por igual.

(Aparcado el 28-09-2026: primero la 6c.) Nada de esto bloquea la **fase 4**, que es la que responde a la pregunta del proyecto: esa
se podria lanzar (punto 5) en paralelo a resolver el punto 1.

La demo de NES (`artifacts/mario_nes_demo/mosca_s0.pt`) está reentrenada entera el
22-09-2026: 300 iteraciones con la normalización de entrada puesta, 52 min de portátil.
Muestreada saca 0,285 de progreso y un 2,1 % de metas, le gana al salto constante
(p = 0,008) y **sigue sin distinguirse del azar** (p = 0,25). El checkpoint viejo, cortado
en la iteración 190 y sin normalizar, se queda al lado como `mosca_s0_190it_sin_normalizar.pt`.

## Actualizacion del 23-09-2026 (portatil)

El punto 2 de arriba ya no aplica: **el gradiente truncado no es el freno** de la fase 6
(el nucleo no usa memoria y, por clonado de la heuristica, si puede representar una buena
politica; detalle en RESULTADOS.md, seccion del 23-09-2026). No hay que pedir permiso para
quitarlo porque no hay que quitarlo. El freno esta en como PPO entrena el nucleo.

Y la misma tarde, medido: arrancando PPO **desde la politica clonada** (mosca y MLP, 3
semillas, prototipo propio) la politica buena se **mantiene** (mosca 0,908 -> 0,924, MLP
0,926 -> 0,906, frente a 0,824 / 0,830 desde cero y 0,793 del salto constante). PPO no la
destruye; desde cero no la encuentra, y en el prototipo **tampoco el MLP**. El freno es de
exploracion. Las tres salidas posibles estan al final de RESULTADOS.md y las tres son
desviacion del diseno: **decision del usuario antes de tocar nada**.
