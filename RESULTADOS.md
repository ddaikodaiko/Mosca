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

---

## Fase 3 — Modelo (RNN con mascara + FNO)

Rama `fase-3-modelo`. `models/masked_rnn.py` y `models/fno.py`, configurados por
`configs/modelo_4000.yaml`.

### Como se garantiza la mascara: no existiendo

Las entradas fuera de la mascara **no son un parametro puesto a cero**: no existen. Los
pesos viven en un vector `theta` de tamano `n_aristas` (175.744) y la matriz densa 4000x4000
no se materializa nunca en el camino del gradiente; se usa una matriz dispersa de 175.744
valores. Por construccion es imposible que una sinapsis inexistente se vuelva no nula.

La ley de Dale es `W = signo * softplus(theta)`, con el signo tomado de la neurona
**presinaptica** y guardado como buffer no entrenable. `softplus > 0` siempre, asi que el
signo no puede cambiar por mucho que empuje el gradiente.

### Test obligatorio de la fase: pasa

`tests/test_masked_rnn.py`, 14 tests. Los dos que exige el encargo, tras **100 pasos de
Adam**:

- entradas fuera de la mascara **exactamente** `0.0` (comparacion exacta, no `allclose`);
- **ninguna** neurona cambia de signo, con `lr = 1e-1` a proposito para forzarlo.

Y un tercero que hace de contrapeso: **los pesos se mueven de verdad**. Sin el, los dos
anteriores pasarian igual con el gradiente a cero, que es la forma tonta de "cumplir".

`pytest -q`: **41 pasan** (fases 1, 2 y 3).

### Un fallo real encontrado por el test del codificador

La primera version elegia los 256 modos de Fourier de menor `|k|` sin mirar la simetria
hermitica. El campo es real, asi que en las columnas `ky = 0` y `ky = N/2` el coeficiente
de `-kx` es el conjugado del de `+kx`. Al guardar uno solo y reconstruir, `irfft2`
simetriza y **se pierde la mitad de la amplitud de esos modos**. Se detecto porque
`decode(encode(x))` no era idempotente: diferencia 0,018 sobre un campo de rms 0,238.

Corregido: se eligen solo los modos **independientes** y el decodificador rellena los
espejos conjugados. En la malla de 64 hay 9 espejos que rellenar de los 128 modos.

### Lo que la red puede ver, medido

Con 128 modos complejos independientes a 64x64, el disco llega a **|k| = 9,1** (no ~6 como
decia una nota previa, que suponia el disco completo en vez de la mitad
independiente). Medido sobre trayectorias reales de `artifacts/ns64_validacion`:

```
varianza del campo retenida por el codificador   0,8846
error L2 relativo del filtro paso bajo           0,3397
```

Es decir: la red **ve** el 88,5 % de la varianza y **solo puede corregir** por debajo de
`k = 9,1`. No es un suelo del error de prediccion, porque la prediccion es residual
(`omega + delta`) y el campo completo pasa de largo: lo que queda por encima de `k = 9,1`
evoluciona como persistencia pura. **Consecuencia para la fase 5:** al medir el error
espectral por bandas hay que separar `k <= 9` de `k > 9`; en la banda alta, todos los
cableados son persistencia por construccion y no discriminan nada.

### Los seis modelos comparten todo menos la mascara

| Modelo | Neuronas | Parametros entrenables |
|---|---|---|
| mosca | 4.000 | **311.328** |
| aleatorio | 4.000 | **311.328** |
| recableado | 4.000 | **311.328** |
| denso | 420 | 176.400 sinapsis (0,4 % del presupuesto de la mosca) |
| fno | — | 1.188.353 |
| persistencia | — | 0 |

Los tres cableados dan el mismo numero de parametros al bit, verificado por test: el
codificador (256->256), el decodificador (256->256) y el sesgo son identicos, y las
175.744 sinapsis tambien, porque los controles se construyeron con ese numero exacto.
El `denso` usa 420 neuronas, que es el N que iguala `N(N-1)` a las aristas de la mosca.

El FNO tiene casi 4x los parametros. Es deliberado: es el techo de rendimiento, no un
control emparejado.

### Coste medido en el portatil

Prueba corta de entrenamiento (mosca, malla 32, lote 16, 150 pasos de Adam):

```
persistencia (suelo)   L2 rel 0,0966
tras 150 pasos         L2 rel 0,0953
coste                  0,30 s/paso  (45 s las 150)
```

El gradiente fluye por el nucleo disperso y la perdida baja del suelo de persistencia,
pero **150 pasos no son un entrenamiento**: esto solo demuestra que la maquinaria funciona.
A 0,30 s/paso, un entrenamiento de 20.000 pasos son ~100 minutos de CPU al 100 %, o sea
que la fase 4 completa **no se hace en el portatil** (regla 6). Ojo tambien al suelo: con
`dt_guardado = 0,1` la persistencia ya acierta con un 9,7 % de error relativo, asi que el
margen de mejora a un paso es estrecho y la metrica que de verdad separa modelos sera el
despliegue largo.

---

## Fase 4 — Entrenamiento

Rama `fase-4-entrenamiento`. `train.py` con las dos fases del encargo: fase A a un paso
con Adam y decaimiento coseno, fase B con currículo de despliegue y retropropagacion a
traves del tiempo. Recorte de gradiente a 1,0 y parada temprana por validacion en ambas.

**El prototipo del portatil no es un resultado**: N=500, malla 32x32, 64 modos, 4.000
pasos, una sola semilla. Sirve para que la maquinaria quede probada y para lo que sigue,
que es lo importante de esta fase.

### Dos fallos reales encontrados, uno arreglado

**1. La entrada saturaba tanh (arreglado).** Los coeficientes de Fourier de un campo de
rms 2,2 llegan a valer 22. Sin normalizar, el impulso de entrada llegaba al nucleo con
rms 2,2 y maximo 6,5: el **24,9 %** de las neuronas sensoriales quedaban clavadas en
`|h| > 0,99`, donde la derivada de tanh es cero, y el gradiente se moria (`|grad|` caia a
0,0000 y el modelo no podia sobreajustar ni 32 pares fijos). Corregido midiendo media y
desviacion de los modos **solo sobre entrenamiento**, guardadas como buffers para que
viajen en el checkpoint. Tras la correccion: impulso de rms 0,59, saturacion 0,000.

**2. El radio espectral del diseno no propaga (medido, no arreglado).** Fijar el autovalor
mayor a 0,95 deja el **grueso** del espectro en 0,071 (N=4.000) y 0,188 (N=500): un impulso
generico se atenua ~14 veces por salto. Medido en el nucleo: `|h|` medio 0,762 en las
sensoriales, 0,012 en el resto, 0,051 en las motoras. Se probo una normalizacion por filas
que sube el grueso a 0,555 sin tocar el maximo, **y no cambia el resultado final**
(0,0998 contra 0,0999). Queda como opcion de config (`core.normalisation: max_eig | row`),
con el valor del diseno por defecto.

### Por que no aprendia: no era el conectoma

Cadena de diagnostico, cada paso medido:

| Prueba | Resultado | Que descarta |
|---|---|---|
| Sobreajustar 32 pares fijos | se estanca en 0,0882 | no es falta de datos |
| Tres normalizaciones del nucleo | **las tres en 0,0882** | no es la propagacion |
| Mejor mapa **lineal**, solucion cerrada | 0,1001 = persistencia | nada, la dinamica es cuadratica |
| **FNO** con los mismos datos | 0,1001 -> **0,0369** | la tarea SI es aprendible, los datos estan bien |
| **MLP** que ve solo los modos bajos | train 0,087 / val 0,105 | el cuello es el codificador, no la mascara |
| MLP con 64 / 128 / 256 modos | val 0,105 / 0,110 / **0,117** | no es falta de modos: mas modos, peor |

El codificador del diseno (FFT -> capa densa -> nucleo -> capa densa -> iFFT) no tiene
sesgo inductivo ninguno: memoriza y no generaliza. El FNO gana porque comparte pesos en el
espacio, no porque tenga mas capacidad.

### El arreglo: aumento de datos por simetrias exactas

El dominio es `[0,2pi]^2` periodico, asi que **trasladar en x es una simetria exacta** de
Navier-Stokes. En y lo son las traslaciones de un periodo del forzado `sin(4y)`, o sea
multiplos de `malla/4` casillas. No es regularizacion: son transiciones que el simulador
podria haber producido y que no estabamos usando.

```
MLP sin aumento    train 0,0871   val 0,1046   (peor que persistencia)
MLP con aumento    train 0,0923   val 0,0953   (mejor que persistencia)
```

Se aplica **identico a los seis modelos**, asi que la comparacion sigue siendo justa.
Es una **desviacion de la receta de entrenamiento del diseno** y esta pendiente del visto
bueno del usuario (`augment.enabled` en el YAML, se apaga con un flag).

### Resultado del prototipo, y el riesgo que destapa

Malla 32x32, N=500, 64 modos, 4.000 pasos, una semilla, con aumento:

| Modelo | 1 paso | despliegue 10 | despliegue 50 |
|---|---|---|---|
| persistencia | 0,1001 | 0,4496 | 0,9343 |
| **mosca** | **0,0999** | 0,4481 | **0,9300** |
| **aleatorio** | **0,0998** | 0,4481 | 0,9359 |
| fno | **0,0395** | **0,1679** | **0,4885** |

**Mosca y aleatorio se diferencian en un 0,1 %, y ninguno se despega de la persistencia.**
El FNO gana por un factor 2,5 a un paso y 1,9 en despliegue de 50.

Esto es exactamente la prediccion registrada (H0/H1: la mosca empata con los controles y el
FNO gana a todos), pero **hay que decir por que empatan**: en esta configuracion empatan
porque **ninguno de los dos aprende nada**, no porque el cableado de la mosca no aporte
sobre el aleatorio. Un empate entre dos modelos que no aprenden no responde a la pregunta
del proyecto. Con el MLP llegando a 0,0953 con el mismo codificador y los mismos datos,
sabemos que ese techo no es del codificador: es del nucleo con mascara tal y como esta
especificado.

**Lo que NO se puede concluir todavia:** el prototipo tiene 500 neuronas y 9.217 sinapsis,
contra las 4.000 y 175.744 de la configuracion real, la mitad de modos y la quinta parte de
pasos. Puede que a escala real si se despegue. Es una pregunta abierta, no un resultado.

### Coste medido

```
mosca / aleatorio (N=500, 4.000 pasos + fase B)   ~80 s
fno (modes 8, width 16)                            241 s
```

La configuracion real (`configs/entrenamiento_64.yaml`, 6 modelos x 3 semillas, 20.000
pasos, malla 64x64, N=4.000) **no se lanza en el portatil**: regla 6.
