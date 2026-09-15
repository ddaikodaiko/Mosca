# Resultados por fase

Metricas medidas, no estimadas. Si algo no cumple su criterio, se dice.

---

## Fase 1 — Simulador de Navier-Stokes

Rama `fase-1-simulador`. Navier-Stokes 2D incompresible en forma de vorticidad,
pseudoespectral con RK4 y desaliasing 2/3, forzado de Kolmogorov `f = sin(4y)`,
`nu = 1e-3`, `alpha = 0.1`, dominio `[0, 2pi]^2` periodico.

### Parametros fijados tras medir

`dt = 0.01`. El plan inicial proponia `1e-3`, dos ordenes por debajo de lo necesario:
el CFL medido con `dt = 0.01` es **0,204** a 64x64 y **0,106** a 32x32, ambos muy por
debajo del limite de 0,5 que hace abortar el script. Bajar `dt` solo habria multiplicado
el coste por diez sin ganar precision.

Transitorio de 2.000 pasos (`t = 20`, dos tiempos de arrastre, `1/alpha = 10`), luego
200 instantaneas cada 10 pasos (`dt_guardado = 0.1`). Semillas disjuntas por
construccion: train 1000-1199, val 2000-2039, test 3000-3039.

### Criterios de aceptacion

| Criterio | 64x64 (validacion, 4+2+2 tray.) | 32x32 (set completo, 200+40+40) | Limite |
|---|---|---|---|
| 1. Deriva de energia, 2a mitad | **+0,468 %/t** | **-0,056 %/t** | < 5 % |
| 2. Pendiente de E(k) | **-4,67** (banda k=5-20) | **-3,69** (banda k=3-9) | sin objetivo |
| 3. Reproducible bit a bit | **True** | **True** | — |
| 4. CFL maximo | **0,204** | **0,106** | < 0,5 |
| 5. Figuras | generadas | generadas | — |

Coste medido: 64x64 unos 3,4 s por trayectoria en lotes de 4; 32x32, 2 min 35 s para
las 280 trayectorias en lotes de 40 (AMD Ryzen 5 5625U, 12 hilos, solo CPU).

### Criterio 2: NO se cumple, y no se puede cumplir a esta resolucion

La pendiente de `E(k)` a 64x64 sale **-4,67**, cuando la cascada de enstrofia 2D pediria
-3. Medida en varias bandas para descartar que sea un artefacto del ajuste:

```
k= 5-20: -4,67    k=5-12: -4,85    k=5-10: -4,75    k=6-14: -4,99    k=10-20: -4,39
```

Es geometrico, no es un error de implementacion: con el forzado en k=4 y el corte de
desaliasing en k=21 hay **0,7 decadas** entre inyeccion y disipacion. No caben un rango
inercial y uno disipativo en ese hueco, asi que lo que se mide es esencialmente el rango
disipativo. Con la malla 64x64 y `nu = 1e-3` que fija el diseno, esto es lo que hay.

**Decision (usuario, 12-09-2026):** aceptar y documentar. No afecta a la pregunta del
proyecto, porque los seis modelos ven exactamente los mismos datos y la comparacion entre
ellos es internamente valida con rango inercial o sin el. Si limita lo que podremos
afirmar: **"flujo 2D caotico, forzado y disipativo, en regimen estacionario"**, nunca
"turbulencia con cascada de enstrofia".

Dos consecuencias apuntadas para mas adelante:

- **Fase 5:** el 91 % de la energia esta en `k <= 4` (`E(k<=4) = 0,158` de `0,174`). La
  metrica de error espectral hay que calcularla por bandas y en escala logaritmica; en
  crudo la dominarian cuatro modos y no distinguiria entre modelos.
- **No subir a 128x128:** el codificador guarda 256 numeros reales, unos 128 modos
  complejos, un disco de radio `k ~ 6`. El rango inercial que ganariamos caeria entero
  fuera de lo que la red puede representar.

### Reproducibilidad: depende del tamano de lote

La primera comprobacion dio `False`. La causa no era el sembrado: el primer fotograma
guardado era identico bit a bit y la desviacion maxima en toda la trayectoria era
`2,384e-07`, exactamente **1 ULP de float32** sobre campos de rms ~2.

`torch.fft` no produce bits identicos con distinto tamano de lote, y Navier-Stokes es
caotico: una diferencia de ~1e-16 en float64 crece hasta ~1e-7 en `t = 20`. La primera
version del chequeo comparaba un lote de 4 contra un lote de 1.

Corregido: la reproducibilidad se verifica repitiendo **la misma forma de ejecucion**, y
`batch_size` se registra en `meta.json` como parte del contrato. La desviacion entre
lotes distintos se sigue midiendo y se informa aparte. A la precision en que se guardan
los datos (float32) las trayectorias son identicas de todos modos.

### Salidas

```
artifacts/ns32/              set completo 32x32 para el prototipo de la fase 4  (230 MB)
artifacts/ns64_validacion/   4+2+2 trayectorias 64x64 de validacion fisica       (26 MB)
```

El set real de 64x64 (200+40+40 trayectorias, ~900 MB) **no se genera en el portatil**:
serian 13 min de CPU sostenida. Se genera en el PC con GPU con el mismo script, el mismo
YAML y las mismas semillas. Comando en `TRASPASO.md`.

---

## Fase 2 — Conectoma y subgrafo

Rama `fase-2-conectoma`. Conectoma FlyWire v783 del repo `philshiu/Drosophila_brain_model`
(Shiu et al., 2024): **138.639 neuronas y 15.091.983 aristas**.

### Descarga

`data/download_flywire.py` baja los dos ficheros sueltos en vez de clonar el repo, que
arrastraria ademas la version 630 (~90 MB que no usamos). Cache en `data/raw/`, verificada
por tamano exacto.

```
Connectivity_783.parquet  100,8 MB  sha256 efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347
Completeness_783.csv        3,3 MB  sha256 bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311
```

### El signo ya viene dado, y ya cumple la ley de Dale

El parquet trae una columna `Excitatory` con valores `+1` / `-1`. Comprobado sobre las
15,1 millones de aristas: **el signo es constante para cada neurona presinaptica, sin una
sola excepcion**. No hay que derivarlo del neurotransmisor a mano; la convencion del
diseno (acetilcolina +1, GABA y glutamato -1) es la que Shiu et al. ya aplicaron. En el
conectoma completo el 29,95 % de las neuronas son inhibidoras.

`Completeness_783.csv` no filtra nada: sus 138.639 filas estan todas en `Completed = True`.
Sirve como mapa indice -> ID de FlyWire, en el mismo orden que los indices del parquet.

### Subgrafo de 4.000 neuronas (`configs/conectoma_4000.yaml`, semilla 20260915)

| Metrica | Valor |
|---|---|
| Neuronas | **4.000** (de 4.000 crecidas: no se poda ninguna) |
| Aristas | **175.744** |
| Densidad | **0,01099** |
| Fraccion inhibidora (aristas) | **0,4318** |
| Fraccion inhibidora (neuronas) | **0,4123** |
| Grado medio / mediano | **87,9 / 68,0** |
| Grado maximo | **710** |
| Componentes debiles | **1** |
| Neuronas con grado interno 0 | **0** |
| En la componente principal | **100 %** (criterio: >= 90 %) |

Coste: 1 s de carga del parquet, 2,5 s de construccion. Nada pesado.

**Criterio de aceptacion superado con holgura**, y no es suerte de la semilla: con las
semillas 1, 2 y 3 sale tambien **100 %**. Lo que si cambia con la semilla es *que*
subgrafo toca: aristas 160.095-184.637, densidad 0,0100-0,0115 y fraccion inhibidora
0,362-0,499. **Consecuencia para la fase 4:** las 3 semillas por modelo son semillas de
*entrenamiento*; el subgrafo tiene que ser el mismo para los seis modelos o la comparacion
deja de ser justa.

Una sola neurona de las 4.000 tiene signo indefinido (`sign = 0`): no tiene ninguna salida
en el conectoma completo, asi que nunca aparece como presinaptica. Su columna de `W` esta
enmascarada a cero, o sea que es inocua, pero se cuenta en `stats.json` (`neuronas_sin_signo`)
para no esconderla.

### Controles: iguales en todo menos en el cableado

| | densidad | desviacion | frac. inhib. aristas | desviacion | grado mediano |
|---|---|---|---|---|---|
| mosca | 0,010987 | — | 0,43184 | — | 68 |
| aleatorio | 0,010987 | **0,000 %** | 0,43190 | **0,013 %** | 88 |
| recableado | 0,010987 | **0,000 %** | 0,43184 | **0,000 %** | 68 |

Tolerancia exigida: 1 %. El **aleatorio** es Erdos-Renyi con el mismo numero exacto de
aristas (muestreo sin reemplazo sobre el indice lineal sin diagonal, asi que no hay
duplicados ni autoconexiones) y las neuronas se marcan inhibidoras en orden aleatorio
hasta alcanzar la fraccion de aristas objetivo — el signo es de la neurona, no de la
arista, para que la ley de Dale siga teniendo sentido. Se nota en el grado maximo: 123 en
el aleatorio frente a 710 en la mosca; ahi esta la cola larga que el ER no reproduce.

El **recableado** hace 1.757.440 intentos de intercambio doble (10 por arista), de los que
acepta **1.638.448 (93,2 %)** y mueve el **99,95 %** de las aristas. Conservacion de grados
verificada elemento a elemento: `grado_salida_conservado = True`, `grado_entrada_conservado
= True`. Como el signo es de la presinaptica y su grado de salida no cambia, la fraccion
inhibidora se conserva **exactamente**, no dentro de una tolerancia.

### Tests

`pytest -q`: **27 pasan**. Los de la fase 2 (`tests/test_conectoma.py`) corren sobre grafos
sinteticos pequenos, sin tocar `data/raw/`; el unico que usa el subgrafo real se salta si
no se ha generado. Cubren: reparto del presupuesto entre manchas, determinismo por semilla,
poda por componente principal, reindexado del subgrafo inducido, conservacion exacta de
grados en el recableado, ausencia de duplicados y autoconexiones, y equivalencia de
densidad y fraccion inhibidora en el aleatorio.

### Salidas

```
artifacts/conectoma_4000/mosca.npz        nodes, rows, cols, synapses, sign
artifacts/conectoma_4000/aleatorio.npz    misma estructura, cableado ER
artifacts/conectoma_4000/recableado.npz   misma estructura, grados conservados
artifacts/conectoma_4000/stats.json       metricas del subgrafo + config completa
artifacts/conectoma_4000/controls_stats.json  comparacion de los tres cableados
```

Los tres `.npz` juntos ocupan menos de 3 MB: se copian al PC con GPU sin problema.
