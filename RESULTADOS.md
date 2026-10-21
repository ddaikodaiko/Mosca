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
YAML y las mismas semillas (ver "Ejecutarlo" en el README).

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

---

## Fase 5 — Evaluacion

Rama `fase-5-evaluacion`. `evaluate.py`, con las cuatro metricas del encargo sobre el
conjunto de test.

### Metricas

1. **Error L2 relativo a un paso.**
2. **Error L2 relativo en despliegue** (50 pasos por defecto, configurable).
3. **Tiempo hasta que la correlacion espacial cae bajo 0,8.** Si un modelo nunca baja del
   umbral se devuelve la duracion completa del despliegue, y eso se lee como "al menos
   esto": es un dato **censurado**, no el valor real. Hay un test que lo fija.
4. **Error del espectro de energia POR BANDAS y en logaritmo**, `|log10(E_pred/E_real)|`.

La cuarta va por bandas por lo medido en la fase 1: el 91 % de la energia esta en
`k <= 4`, asi que en crudo la metrica la dominarian cuatro modos y no distinguiria nada.
Las tres bandas son **inyeccion** (k 1-4), **alcanzable** (k 5 hasta donde llega el
codificador) y **fuera del codificador** (por encima, hasta el corte de desaliasing).
La tercera existe porque en ella **todos los cableados son persistencia por
construccion**: el decodificador no escribe ahi. Informarla junta con el resto seria
diluir la comparacion con una banda en la que ningun modelo puede hacer nada.

### Prueba de permutacion

Pareada por trayectoria y con sorteo de signo, 10.000 permutaciones. Pareada porque las
dos columnas son el **mismo** conjunto de test evaluado por dos modelos: la variabilidad
entre trayectorias es enorme comparada con la diferencia entre modelos, y emparejar la
elimina. Se informa siempre junto a la **desviacion tipica entre semillas**, y el veredicto
dice explicitamente si la diferencia cae dentro de esa variabilidad.

Con menos de dos semillas el veredicto **no se emite**: la desviacion sale 0 por
construccion y cualquier diferencia pareceria significativa. En ese caso imprime
"no juzgable: 1 semilla".

Limite que conviene tener presente: una prueba de permutacion a nivel de **semilla**, con
3 por modelo, tiene como minimo p = 1/20 (hay C(6,3) = 20 reparticiones posibles). Por eso
la prueba principal es a nivel de trayectoria y la dispersion entre semillas se informa al
lado en vez de fundirlas en un solo numero.

### Figuras

Tres, en `<output_dir>/figs/`: barras de las tres metricas escalares con barras de error
entre semillas y la persistencia como linea de referencia; decaimiento de la correlacion
con banda de +-1 desviacion; y error espectral por bandas. Paleta categorica en orden fijo
validada para daltonismo (peor par adyacente Delta E 9,1 en protanopia), con la
persistencia en gris porque no es una serie sino la referencia, y con etiquetas numericas
directas sobre cada barra.

### Comprobacion sobre el prototipo

Ejecutado sobre `artifacts/train_32_proto` (32x32, N=500, **una sola semilla**, asi que
no son resultados del proyecto, solo la comprobacion de que la fase corre):

| modelo | L2 1 paso | L2 despliegue 50 | t hasta corr<0,8 |
|---|---|---|---|
| persistencia | 0,1007 | 0,9033 | 0,99 |
| mosca | 0,1004 | 0,9003 | 0,99 |
| aleatorio | 0,1004 | 0,9057 | 0,99 |
| fno | 0,0394 | 0,4715 | 3,70 |

Un detalle que sale de la metrica espectral y que merece anotarse: en la banda **fuera del
codificador** el FNO es el **peor** (0,11 contra 0,05 de los cableados). No es un fallo del
FNO: es que el FNO escribe contenido de alta frecuencia y se equivoca, mientras que los
cableados dejan esa banda intacta y por tanto heredan el acierto de la persistencia. Al
leer la tabla final hay que tenerlo en cuenta: en esa banda, "mejor" significa "no ha
tocado nada".

### Tests

`tests/test_evaluate.py`, 12 tests sobre campos sinteticos con la respuesta conocida:
correlacion de un campo consigo mismo y con su opuesto, invarianza a desplazamiento
constante, primer cruce del umbral y censura cuando no cruza, error espectral cero para
campos identicos y exactamente 2,0 al escalar el campo por 10, permutacion con muestras
identicas (p = 1), con diferencia grande (p < 0,01) y con solo ruido (p > 0,05), y que las
bandas cubren el rango sin solaparse.

`pytest -q`: **58 pasan**.


---

## Pendientes al cierre del 15-09-2026

Codigo de las cinco fases completo, `pytest -q`: 58 pasan. Nada fusionado a `main`
todavia: las cinco fases cuelgan de cinco ramas encadenadas.

### Bloqueante

- **`denso` no arranca.** Detectado al probar los seis modelos uno a uno, que era lo unico
  que quedaba por hacer en el portatil antes de encender la GPU. Iguala el presupuesto de
  parametros con 420 neuronas y el diseno pide 256 + 256 puertos disjuntos. Habria reventado
  el entrenamiento 10 de 15 en la GPU. Tres salidas posibles, todas con coste (resuelto el 16-09-2026, ver mas abajo).
  Los otros cinco modelos construyen, propagan gradiente y despliegan sin problema.

### Decisiones del usuario, abiertas

1. Visto bueno al **aumento de datos** por simetrias exactas (desviacion de la receta del
   diseno). Sin el, ningun modelo con mascara supera a la persistencia.
2. Que hacer si el nucleo con mascara **no se despega de la persistencia** a escala real.
   El empate del prototipo no responde a la pregunta del proyecto.
3. Como se arregla `denso`.

### Trabajo util que aun cabe en el portatil

- **Ensayo general a 32x32 con las 3 semillas.** Es lo unico que puede decir, antes de
  gastar GPU, si la comparacion mosca contra controles tiene alguna resolucion o si todos
  empatan en la persistencia. Coste estimado ~30 min de CPU al 100 %: cruza la regla 6, hay
  que trocearlo y pedir permiso.
- **Fusionar las ramas a `main`.**
- **La mitad barata del visualizador**, que ahora sirve de diagnostico y no solo de video:
  el nucleo apenas se activa (`|h|` medio 0,012 fuera de las sensoriales).

---

## 16-09-2026: puertos con solape fijo

Decisiones del usuario: `denso` se arregla con solape de puertos en los seis modelos, y el
aumento de datos queda aprobado.

El solape libre (cada puerto elegido solo por su grado) daba estas neuronas compartidas
entre sensoriales y motoras, con 256 de cada:

| modelo | N | compartidas |
|---|---|---|
| mosca | 4.000 | 159 |
| recableado | 4.000 | 159 |
| aleatorio | 4.000 | 18 |
| denso | 420 | 256 (grados identicos: desempate por indice) |

Cada compartida es un atajo entrada -> salida que no pasa por la recurrencia, y la mosca
habria tenido nueve veces mas que el aleatorio. Se fija `n_shared = 92` (256 + 256 - 420)
en todos; medido tras el cambio: 92/256 en los cuatro. `denso` construye, propaga,
retropropaga con gradiente finito y despliega a 64x64. `pytest -q`: **59 pasan**.

Los resultados del prototipo de la fase 4 usaban puertos disjuntos: quedan obsoletos.

---

## 16-09-2026: por que el nucleo no aprendia

Todas las sondas son baratas (minutos de CPU), en espacio de caracteristicas o con el
simulador exacto. Datos: `artifacts/ns32` (32x32, 64 modos) y `artifacts/ns64_validacion`.

### 1. La senal tiene camino

Distancia en saltos de la sensorial mas cercana a cada motora: **mediana 1, maximo 1** en
mosca, aleatorio y recableado, a N=500 y a N=4.000 (maximo 2 en el aleatorio de 4.000).
Cuatro subpasos internos sobran. Hipotesis de "la senal no cruza el grafo": descartada.

### 2. Actividad por papel (visor, mosca s0, antes del arreglo del aumento)

|h| medio: sensoriales 0,59, compartidas 0,11, motoras **0,013**, interior 0,032
(aleatorio: 0,52 / 0,10 / 0,032 / 0,094). La lectura esta casi apagada.

### 3. La tarea es resoluble con lo que ve el codificador

Fisica exacta integrando solo la proyeccion del campo a los modos del codificador, y
proyectando el incremento resultante, frente al incremento real proyectado:

| malla | modos | energia de Delta-omega en esos modos | R2 del incremento (fisica) | error 1 paso: persistencia / techo / fisica |
|---|---|---|---|---|
| 32 | 64 | 26,0 % | 0,624 | 0,1001 / 0,0861 / 0,0918 |
| 32 | 128 | 55,9 % | 0,742 | 0,1001 / 0,0663 / 0,0766 |
| 64 | 128 | 37,3 % | 0,823 | 0,1054 / 0,0833 / 0,0877 |
| 64 | 256 | 54,9 % | 0,891 | 0,1054 / 0,0706 / 0,0752 |

"Techo" = incremento real perfecto en esos modos. **A la escala del diseno (64x64, 128
modos) un nucleo perfecto solo baja el error a 1 paso de 0,105 a 0,083.**

### 4. Sin simetrias, todo memoriza

Sonda: predecir el incremento normalizado de los 64 modos desde los modos actuales.

| modelo | R2 train | R2 val |
|---|---|---|
| lineal (minimos cuadrados) | - | 0,024 |
| MLP 128-512-128 | 0,203 | 0,005 |
| nucleo mosca, lr 3e-3 | 0,092 | 0,017 |

Con aumento por traslaciones (hecho en fase, correcto) el MLP de 1024 pasa a R2 val
**0,23-0,33**; sin aumento, -0,56 a -0,97. La palanca es la simetria, no el tamano de datos.

### 5. BUG: el aumento tenia los ejes cambiados

El forzado `sin(4y)` varia en la dimension -1 del array; `Augmenter` aplicaba ahi la
traslacion libre. Con el simulador: trasladar 1 casilla en dim -2 da error de simetria
3e-16; en dim -1, 2,5e-2; el `Augmenter` antiguo, **4,0e-2** sobre pares reales. El
incremento que se aprende es ~10 % del campo. El test antiguo tenia el mismo error de ejes.
Corregido con un test contra el simulador. Tras el arreglo, la sonda del nucleo con aumento
correcto: **mosca R2 val 0,095, aleatorio 0,098** (antes 0,017).

Todo lo entrenado con aumento antes del arreglo se descarta (archivado en
`artifacts/ensayo_32_aumento_roto`).

### 6. Limite de predictibilidad (correlacion con el campo real, test 32x32, 10 trayectorias)

| predictor | t=1 | t=2 | t=4 |
|---|---|---|---|
| persistencia | 75,0 % | 53,1 % | 30,3 % |
| fisica exacta, error inicial 1 % | 100 % | 100 % | 100 % |
| fisica exacta viendo 64 modos | 91,7 % | 91,7 % | 84,3 % |
| fisica exacta viendo 128 modos | 96,7 % | 96,7 % | 93,1 % |

En este horizonte el flujo no es lo bastante caotico como para impedir el 100 %: el limite
lo pone el codificador (~90 %) y, sobre todo, lo que el nucleo consigue aprender.

### 7. Control nuevo y cambios

- `sin_nucleo`: mismas neuronas, signos y puertos que la mosca, cero conexiones. Si la
  mosca no le gana, lo que predice no sale de su cableado.
- `augment.continuous_x`: traslacion en fracciones de casilla por fase de Fourier (test
  contra el simulador < 1e-10; fase en float64).

## Fase 6 — Mario (seccion 13 del diseno), 21-09-2026

Encargo explicito del usuario, adelantado antes de tener la fase 5 real. Lo que hay es
**infraestructura medida, no un resultado sobre el cableado**: los 18 entrenamientos
(6 modelos x 3 semillas) no caben en esta maquina.

### 1. El entorno

`envs/platformer.py`: motor de plataformas propio, sin emulador ni ROM, con estado
simbolico (lo que pide la nota de viabilidad del diseno). Nivel procedural por semilla,
observacion de 302 numeros (ventana de 11x9 casillas en tres canales — solido, enemigo,
vacio — mas cinco escalares) y siete acciones que cubren los cuatro botones.

Dos correcciones que hicieron falta para que los niveles tuvieran solucion:

| Problema | Sintoma | Arreglo |
|---|---|---|
| Dos huecos seguidos | vanos de hasta 6 casillas; el salto cruza 5,4 | nunca dos huecos seguidos, y el aterrizaje mantiene la altura |
| `JUMP_VELOCITY = 0,50` | el salto subia 1,84 casillas y los escalones de 2 eran muros | 0,58, que sube 2,52 (medido, no calculado) |

Con eso, la heuristica escrita a mano pasa del 7,5 % de niveles terminados al 55 %.

### 2. Las cuatro lineas de referencia (40 niveles de test, 160 casillas)

| Politica | Progreso | Metas |
|---|---|---|
| quieto | 0,016 | 0 % |
| siempre derecha + carrera | 0,075 | 0 % |
| **salto constante a la derecha** | **0,381** | **8 %** |
| heuristica (mira huecos, muros y enemigos) | 0,752 | 55 % |

El salto constante es la linea que importa: **PPO cae en ella sola**. En el prototipo
(N=500, niveles de 60 casillas, 400 iteraciones) tanto la mosca como el MLP terminan con
la politica codiciosa emitiendo siempre la misma accion, y el valor de validacion queda
congelado (mosca: 0,585 en las ocho evaluaciones; MLP: 0,583 -> 0,624, que si se mueve).
Todo numero de la fase 6 se lee contra estas cuatro lineas o no se lee.

### 3. El modelo

`models/masked_policy.py` reutiliza `MaskedRecurrentCore` sin tocarlo: misma mascara,
misma ley de Dale, mismo `n_shared` fijo en todos los modelos. Cambian solo el
codificador (la ventana de casillas) y el decodificador (logits de botones y valor).
`tests/test_masked_policy.py` repite el test obligatorio contra el bucle de PPO: tras
entrenar, ninguna sinapsis fuera de la mascara es distinta de cero y ninguna neurona
cambia de signo.

**El gradiente esta truncado a un paso de entorno.** La h persiste durante el despliegue
pero entra desacoplada en la actualizacion, asi que el nucleo no puede construir memoria
de mas de un paso del juego. Es la unica variante que cabe en CPU y condiciona lo que se
puede afirmar.

### 4. Coste

| | CPU de esta maquina |
|---|---|
| recoleccion (32 entornos x 128 pasos, N=4.000) | 11 s |
| actualizacion PPO (3 epocas x 8 minilotes) | 40 s |
| iteracion completa | 51 s |
| 800 iteraciones (3,3 M pasos de entorno) | 11 h por modelo y semilla |
| los 18 entrenamientos | 8 dias |

No cabe: la fase 6 real va a GPU, igual que la 4 y la 5. La actualizacion se lleva el
80 % del tiempo, asi que es ahi donde hay que recortar si hace falta.

### 5. Lo que falta

- Medir una iteracion en GPU y ajustar `iterations` antes de lanzar los 18.
- Que el margen entre el salto constante (0,38) y la heuristica (0,75) sea suficiente
  para separar cableados esta **por comprobar**. Si los seis modelos se quedan pegados a
  la politica degenerada, la fase 6 tampoco tendra resolucion para responder a la
  pregunta, igual que pasa con el empate de la fase 4.

## 22-09-2026: coste medido antes de entrenar, no estimado despues

`benchmark.py` mide por modelo el segundo por paso de la fase A, el de la fase B al
horizonte mas largo del curriculo y el pico de memoria de GPU, y extrapola las horas por
semilla. Corre sin datos generados (campos aleatorios de la misma forma, marcado como tal
en el informe) porque lo que fija el tiempo por paso son las formas, no el contenido.

Referencia de este portatil (CPU, 64x64, N=4.000, `configs/entrenamiento_64.yaml`, datos
sinteticos, 2 pasos cronometrados tras 3 de calentamiento):

| modelo | parametros | s/paso fase A | s/paso fase B (h=20) | horas por semilla |
|---|---|---|---|---|
| mosca | 311.328 | 0,351 | 6,23 | 12,33 |

Es la cota optimista: no cuenta evaluaciones periodicas ni carga de datos, y la parada
temprana puede recortarla. Confirma lo que ya se sabia: los 18 entrenamientos de la
fase 4 no caben en este portatil ni de lejos.

`train_rl.py` acepta ahora `--iterations` (escribe en `medicion_<n>it/`, sin pisar los
checkpoints) y cronometra las primeras iteraciones separando recoleccion y actualizacion.
Con `eval_every: 50`, antes la primera senal de coste tardaba una hora en aparecer.

## 22-09-2026: el cribado largo, leido (y la ablacion de continuous_x)

El ensayo largo de `configs/ensayo_32_largo.yaml` se habia corrido el 16-09 y sus cinco
informes llevaban desde entonces sin leerse. Son 20.000 pasos de fase A (cinco veces el
ensayo corto), una semilla, 32x32 y 64 modos.

| modelo | error a 1 paso | despliegue 10 | despliegue 50 |
|---|---|---|---|
| persistencia | 0,1001 | 0,4496 | 0,934 |
| `sin_nucleo` | 0,0998 | 0,4475 | 0,920 |
| mosca | **0,0994** | 0,4725 | 1,618 |
| aleatorio | **0,0994** | 0,4584 | 1,434 |
| recableado | **0,0993** | 0,4640 | 1,472 |
| fno (del ensayo corto, 4.000 pasos) | 0,035-0,038 | 0,15-0,16 | 0,42-0,47 |

Tres lecturas, ninguna buena para el nucleo:

1. **La mosca le gana al control `sin_nucleo` por 0,0004.** Cuatro diezmilesimas entre
   "tiene el cableado de una mosca" y "no tiene ninguna conexion". A esta escala, lo que
   predice el modelo no lo hace su recurrencia.
2. **Entrenar cinco veces mas compra 0,0004** (0,0998 -> 0,0994). La curva de la fase A no
   llega a estancarse ni dispara la parada temprana — baja monotona de 0,1002 en el paso
   1.000 a 0,0990 en el 20.000 — pero al ritmo de 0,0001 cada 2.000 pasos. El techo de la
   fisica viendo esos mismos 64 modos esta en 0,0861.
3. **En despliegue largo el nucleo empeora con mas entrenamiento.** Los tres cableados se
   van a 1,43-1,62 a 50 pasos, peor que no hacer nada (0,934). El aviso hay que ponerlo
   entero: la fase B del prototipo son **60 pasos a horizonte 5**, un muñon; la config real
   entrena 2.000 por etapa hasta horizonte 20. Esta divergencia no predice la de la GPU,
   pero si dice que la estabilidad del despliegue hay que mirarla, y que a un paso el
   modelo esta aprendiendo algo que a 50 pasos se realimenta y explota.

### Ablacion: `augment.continuous_x` no aporta nada

El cribado cambiaba dos cosas a la vez (4.000 -> 20.000 pasos y la traslacion en x por
fracciones de casilla), asi que no atribuia. `configs/ensayo_32_largo_x_discreta.yaml` es
identico al largo salvo la bandera:

| traslacion en x | mejor val de la fase A | error a 1 paso | despliegue 50 |
|---|---|---|---|
| continua (fase de Fourier) | 0,0990 | 0,0994 | 1,618 |
| por casillas enteras | 0,0991 | 0,0994 | 1,558 |

Identicas. La mejora del cribado venia de entrenar mas, no de la bandera. **Decision: la
fase 4 real va con `continuous_x: false`**, que es lo ya probado y lo mas simple; la
bandera se queda en el codigo, con su test contra el simulador, por si a 64x64 cambia algo.

Con esto el YAML de la fase 4 ya no tiene ningun PENDIENTE.

### `evaluate.py` contrasta ahora tambien a un paso y contra la persistencia

Faltaban las dos comparaciones que mas importan. La prueba de permutacion solo corria
sobre el despliegue y el tiempo de descorrelacion, y la lista de controles no incluia a la
persistencia: el resultado del prototipo vive a un paso y contra no hacer nada. El error a
un paso se guarda ahora **par a par** (mismo orden de pares en todos los modelos, asi que
estan pareados por construccion) y `persistencia` va la primera de la lista. Es la misma
leccion que ya estaba aprendida en `evaluate_rl.py` con la politica al azar: sin el
contraste escrito al lado, empatar con el control se lee como un resultado.

Aplicado al cribado largo (split de **test**, 40 trayectorias, 50 pasos de despliegue,
una semilla, el codificador llega a k=6,3):

| modelo | L2 1 paso | L2 despliegue | t hasta corr<0,8 | inyeccion (k<=4) | alcanzable | fuera del codificador |
|---|---|---|---|---|---|---|
| mosca | 0,0999 | 1,533 | 0,943 | 0,857 | 1,061 | 0,046 |
| aleatorio | 0,0999 | 1,351 | 0,928 | 0,717 | 0,868 | 0,046 |
| recableado | 0,0999 | 1,358 | 0,925 | 0,827 | 0,855 | 0,046 |
| `sin_nucleo` | 0,1003 | 0,887 | 0,995 | 0,050 | 0,054 | 0,046 |
| persistencia | 0,1007 | 0,903 | 0,985 | 0,033 | 0,053 | 0,046 |

Diferencias pareadas de la mosca (p de permutacion con 10.000 remuestreos; con **una sola
semilla** el veredicto entre semillas no es juzgable y asi se imprime):

| contraste | a 1 paso | en despliegue | en t de correlacion |
|---|---|---|---|
| vs persistencia | **-0,0008** (p<1e-4) | **+0,629** (p<1e-4) | -0,043 (p=0,02) |
| vs `sin_nucleo` | **-0,0004** (p<1e-4) | **+0,645** (p<1e-4) | -0,053 (p=0,003) |
| vs aleatorio | +0,0000 (p=0,76) | +0,182 (p<1e-4) | +0,015 (p=0,42) |
| vs recableado | +0,0001 (p<1e-4) | +0,175 (p<1e-4) | +0,018 (p=0,22) |

El cuadro completo, dicho sin adornos: el nucleo cableado gana un 0,8 % a un paso y a
cambio **empeora un 70 % el despliegue** y pierde tiempo de correlacion. El error espectral
lo explica: en las bandas que el codificador si ve, la mosca esta en 0,86-1,06 y la
persistencia en 0,03-0,05, o sea que el nucleo esta metiendo energia donde no toca y el
despliegue autoregresivo la realimenta. La banda de fuera del codificador sale 0,046 en los
cinco modelos, identica, que es exactamente lo que predice la nota de la fase 3.

Otra vez el aviso: la fase B de este prototipo son 60 pasos a horizonte 5. La config real
entrena 2.000 por etapa hasta horizonte 20, y es justo lo que deberia atacar esta
divergencia. Con lo medido no se puede decir si lo consigue.

### Los seis modelos, medidos a escala real (22-09-2026, CPU con 6 hilos)

`benchmark.py` sobre `configs/entrenamiento_64.yaml` sirve de dos cosas a la vez: mide, y
de paso comprueba que los seis modelos **construyen, propagan y retropropagan a la escala
real** (N=4.000, malla 64, 128 modos complejos, despliegue de horizonte 20). Ninguno falla.
Datos sinteticos de la misma forma que los reales, que es lo que fija el tiempo por paso.

| modelo | parametros | s/paso fase A | s/paso fase B (h=20) | h fase A | h fase B | h por semilla |
|---|---|---|---|---|---|---|
| mosca | 311.328 | 0,363 | 6,13 | 2,02 | 10,22 | 12,24 |
| aleatorio | 311.328 | 0,312 | 6,33 | 1,73 | 10,54 | 12,28 |
| recableado | 311.328 | 0,341 | 5,97 | 1,89 | 9,95 | 11,84 |
| denso | 307.984 | 0,164 | 3,21 | 0,91 | 5,34 | 6,25 |
| `sin_nucleo` | 135.584 | 0,125 | 2,62 | 0,69 | 4,37 | 5,06 |
| fno | 1.188.353 | 1,029 | 7,33 | 5,72 | 12,22 | 17,94 |

Los 18 entrenamientos son **197 h de esta CPU**, ocho dias largos. Y el reparto importa mas
que el total: **la fase B se lleva el 80 %** (158 h de 197). El despliegue es secuencial —
20 pasos encadenados, cada uno con cuatro subpasos del nucleo — asi que es la parte que
peor acelera en GPU. Si al medir en la 3060 Ti la fase B sigue dominando, la palanca es
`phase_b.steps_per_stage` o recortar el curriculo, no tocar la fase A.

Nota de honestidad sobre estas cifras: son con 6 de los 12 hilos (el portatil tenia que
seguir usable) y con datos sinteticos. Valen como reparto entre fases y como orden de
magnitud, no como record de la maquina.

### La fase 6b a escala real: el emulador manda, no la GPU (22-09-2026)

`configs/mario_nes.yaml` (N=4.000, 16 entornos, despliegue de 128 pasos) no se habia
ejecutado nunca. Con `--iterations 1` arranca y completa una iteracion: **44,2 s**, de los
cuales **27,9 s son recoleccion** (63 %) y 16,3 s la actualizacion de PPO. Son 6 hilos.

El reparto se da la vuelta respecto al entorno propio, donde la actualizacion se llevaba el
80 %: aqui el cuello es el emulador de NES, que corre en CPU y no se acelera con la 3060 Ti.
Las 800 iteraciones son ~9,8 h por modelo y semilla, ~177 h los 18 entrenamientos, y la GPU
solo puede morder ese 37 % de la actualizacion. Si hay que acelerar la fase 6b, lo que
mueve la aguja es mas paralelismo de CPU (mas entornos, o varios procesos), no la tarjeta.

Vale tambien de prueba de humo: la politica cableada con las 4.000 neuronas del conectoma
corre sobre el Super Mario Bros de verdad, con la observacion simbolica de la RAM, y
completa recoleccion, GAE, actualizacion, evaluacion y checkpoint.

### Las figuras de la fase 5, legibles (22-09-2026)

El panel del error a un paso se dibujaba con barras desde cero: 0,0999 y 0,1003 salian
identicos a ojo, y encima la etiqueta de la persistencia se solapaba con los valores. Es
decir, la figura de la metrica que decide era la unica que no se podia leer. Ahora:

- **puntos con barra de error y tallo hasta la persistencia**, no barras. Un punto codifica
  posicion, asi que el eje no tiene que empezar en cero para ser honesto, y se dice en el
  propio pie de la figura. El tallo mide la distancia a "no hacer nada": hacia arriba, peor.
- **los decimales salen del rango, no de la magnitud.** Con la regla anterior los cuatro
  cableados imprimian "0.0999" y la diferencia tampoco se veia en el texto. Ahora: mosca
  0,09991, aleatorio 0,09991, recableado 0,09985, `sin_nucleo` 0,10031, persistencia 0,10072.
- **el espectro por bandas lleva la persistencia como tramo discontinuo en cada banda.** Sin
  ella el panel se leia como "unos cableados distorsionan mas que otros"; con ella dice lo
  que hay: los tres distorsionan un orden de magnitud mas que no hacer nada (0,86-1,06
  contra 0,03-0,05), y en la banda de fuera del codificador los cuatro coinciden en 0,046.

### La fase 5 a 64x64, probada de verdad (22-09-2026)

`evaluate.py` no se habia ejecutado nunca a la malla real. Con la config de la fase 4
apuntando al set pequeño de `artifacts/ns64_validacion` corre entera: bandas espectrales,
tabla, pruebas y figuras. El codificador llega a **k = 9,1**, exactamente el numero
documentado en la fase 3, asi que las bandas se cortan donde deben. La persistencia a
64x64 da 0,0974 a un paso y 0,900 en despliegue de 50 pasos.

De paso, una aspereza del flujo nuevo: con la puerta del traspaso (entrenar solo mosca y
`sin_nucleo` antes de decidir) es facil llamar a `evaluate.py` cuando todavia no hay ningun
checkpoint. Antes dibujaba tres paneles vacios con una linea de persistencia y ni una
palabra. Ahora lo dice y no dibuja nada.

### `train.py` se puede reanudar (22-09-2026)

Un entrenamiento de la fase 4 son horas; un reinicio, una actualizacion del sistema o un
Ctrl-C los tiraba enteros, y `scripts/tanda.sh` solo sabia saltar los **terminados**. Ahora
cada `resume_every` pasos (500 en la config real) se guarda una instantanea en
`checkpoints/<modelo>_s<semilla>.reanudar.pt` con pesos, optimizador, planificador, parada
temprana, registro y **el estado del generador aleatorio**. Se escribe en un temporal y se
renombra, asi que un corte durante el guardado no deja un fichero a medias, y al terminar
bien se borra.

Lo del generador no es un detalle de implementacion: sin el, al reanudar cambia el orden de
los lotes y el entrenamiento deja de ser el que la semilla dice que es. Con el, **reanudar
da los mismos pesos, bit a bit, que no haber parado**, y hay un test que lo comprueba —
comprobado ademas que el test falla si se quita la linea que restaura el generador.

Cubre las dos fases, incluidas las etapas del curriculo de la fase B, que es donde estan el
80 % de las horas. Probado tambien de verdad: matando el proceso con SIGKILL a mitad y
relanzando, retoma en el paso donde estaba.

### `train_rl.py` tambien se reanuda, con una promesa mas debil (22-09-2026)

Misma idea que en fluidos —instantanea cada `resume_every` iteraciones con pesos,
optimizador, planificador, generador, historial y contador— pero aqui **no es bit a bit**,
y conviene decirlo antes de que alguien lo suponga: los entornos no se pueden serializar
(el emulador lleva su propia RAM), asi que al reanudar arrancan de cero y la recoleccion ya
no es la que habria salido de un tiron. Lo aprendido sobrevive entero; la reproducibilidad
desde la semilla, no. Por eso el informe anota `reanudaciones`: si un numero tiene que ser
reproducible, se relanza y se deja terminar.

El test corta la tercera iteracion, reanuda y comprueba que la historia sale [0,1,2,3] sin
repetidos, que `reanudaciones` vale 1 y que la instantanea se borra al terminar bien.
Probado ademas matando el proceso con SIGKILL: retoma en la iteracion que tocaba.

## 22-09-2026: la demo de NES, entrenada entera (300 iteraciones)

El checkpoint que habia se corto en la iteracion 190 de 600 y era anterior a la
normalizacion de entrada. Reentrenada de cero con la normalizacion puesta, 300 iteraciones
(**52 min** en el portatil con 6 hilos, 8,6 s por iteracion, 0 reanudaciones).

**Recordatorio de que esto no es el experimento:** entrena y mide en 1-1, el mismo nivel,
que es justo lo que el diseno prohibe. Sirve para grabar video. La comparacion entre
cableados va en `configs/mario_nes.yaml`, con niveles disjuntos, y en GPU.

Evaluacion muestreada, 8 niveles x 6 partidas, semilla de muestreo fijada:

| politica | progreso | metas |
|---|---|---|
| azar (botones al azar) | 0,233 | 0 % |
| siempre a la derecha | 0,094 | 0 % |
| salto constante | 0,098 | 0 % |
| **mosca (300 it)** | **0,285** | **2,1 %** |

- **vs salto constante: +0,187, p = 0,008.** Le gana.
- **vs azar: +0,052, p = 0,25.** Sigue **sin distinguirse de apretar botones al azar**.

Contra la version anterior (190 iteraciones, sin normalizar): 0,244 -> 0,285 de progreso,
y la tasa de metas pasa de 0 % a 2,1 % (alguna partida suelta llega a la bandera). El
contraste contra el azar mejora de p = 0,68 a p = 0,25, que **sigue sin ser nada**.

Con la politica codiciosa saca 0,136, frente a los 0,098 de antes, que eran exactamente
los de "saltar siempre". O sea que ha dejado de ser una accion constante, pero poco mas.

En el visor se nota ademas el efecto de la normalizacion: la tira de sensoriales, que antes
era una franja amarilla uniforme (saturada), ahora tiene rango. Videos en `artifacts/`
(git-ignored) y un recorte en `docs/media/mosca_juega_nes.gif`. En la partida grabada
(muestreada, semilla 3) llega al **77,6 %** del nivel; con la codiciosa, al 13,6 %. Una
partida no es una medida: la medida son los 0,285 de la tabla.

### El ruido del muestreo, acoplado entre modelos (22-09-2026)

`evaluate_rl.py` fijaba la semilla **una vez** al empezar. Como los modelos se evaluan en
fila, cada uno consumia un tramo distinto del generador: la mosca sorteaba sus acciones con
unos numeros y el aleatorio con otros. No esta sesgado —los dos son estimadores validos—
pero desperdicia precision justo donde menos sobra: ya hubo una vez en que el ruido del
muestreo dio la vuelta al orden entre modelos.

Ahora la semilla se refija **antes de cada repeticion y con el mismo valor para todos los
modelos** (numeros aleatorios comunes). No hace que jueguen igual, porque cada politica
sortea de su propia distribucion, pero acopla el ruido.

Comprobado de la forma mas directa: evaluar el mismo checkpoint dos veces con nombres
distintos. Con el acoplamiento, las dos filas salen identicas y el contraste pareado da
**diferencia +0,0000 y p = 1,0000**. Sin el, esa misma pareja habria dado una diferencia
espuria y un p cualquiera.

Afecta al experimento de la fase 6 en GPU: los 18 entrenamientos se comparan con esta
funcion. Los numeros absolutos se mueven un poco respecto a evaluaciones anteriores (cambia
que sorteo le toca a cada partida), asi que las cifras de la fase 6 anteriores al
22-09-2026 no son comparables una a una con las nuevas; las conclusiones no cambian.

## 22-09-2026: el critico estaba ciego, y eso explica el estancamiento

Antes de echarle mas iteraciones a la demo, se miro por dentro el entrenamiento de 300.
PPO mejora la politica siguiendo la **ventaja** = retorno menos valor predicho, asi que si
el critico no explica el retorno, la ventaja es ruido y da igual cuanto se entrene.

`diagnostico_rl.py` juega episodios completos y compara el valor predicho en cada paso con
el retorno descontado real desde ese paso:

| semilla | pasos | retorno real | valor predicho |
|---|---|---|---|
| 101 | 69 | 42,3 | **27,7** |
| 102 | 29 | 17,7 | **27,7** |
| 103 | 257 | 68,3 | **27,7** |
| 104 | 31 | 18,0 | **27,7** |

**Varianza explicada +0,10** (y +0,14 / +0,18 en otras tiradas; +0,08 en el checkpoint de
190). Desviacion del valor 5,3 contra 22,6 del retorno. El critico predice practicamente
una constante.

Concuerda con lo que ya decian las curvas del entrenamiento: KL por actualizacion de 2e-5
a 9e-5 (la politica casi no se mueve), perdida de politica de 1e-4, y perdida de valor
estancada en 42 con recompensas medias de 80. El progreso se aplana en la iteracion 150.

### No es que la lectura este muerta

Primera hipotesis descartada: que las motoras esten saturadas y no lleven informacion. Se
midio la actividad de las 56 que leen el actor y el critico a lo largo de 298 pasos:
desviacion mediana entre pasos **0,079**, maxima 0,872, solo **6 de 56** practicamente
constantes, y **7,8 %** de lecturas saturadas. Hay senal; el critico no la usa.

### Es la escala del objetivo

El critico es un `Linear` sobre neuronas acotadas por tanh que tiene que producir valores
de 30-60. Para eso necesita pesos enormes, y con lr 3e-4 y 3.600 actualizaciones no llega.
`NormalizadorDeRetorno` (en `models/masked_policy.py`) hace que aprenda en escala
normalizada y desnormaliza donde hacen falta unidades de recompensa, que es el calculo de
GAE. Medias moviles, viajan en el checkpoint, y con `ppo.normalise_returns: false` —el
valor por defecto— es la identidad exacta: nada de lo entrenado hasta ahora cambia.

Bajar `reward_scale` no servia de atajo: `death_penalty` se resta **despues** de escalar,
asi que habria cambiado la tarea y no solo la escala.

### Por que importa mas alla de la demo

Si el critico esta ciego, lo esta en los seis modelos por igual, y entonces **la fase 6 no
tiene resolucion por construccion**: se estaria comparando el ruido de seis cableados. Es
el mismo cuadro que en fluidos —donde el empate venia de que ninguno aprende— pero por una
causa distinta y, esta si, arreglable. Por eso la varianza explicada del critico pasa a ser
una **puerta** antes de lanzar los 18 entrenamientos en GPU.

### A/B del critico: arreglado, y no basta (22-09-2026)

`configs/mario_nes_demo_critico.yaml`, identico a la demo de 300 iteraciones salvo
`ppo.normalise_returns` (verificado comparando las dos configs clave a clave: solo difieren
en esa linea y en el directorio de salida).

| | referencia | con normalizacion |
|---|---|---|
| varianza explicada del critico | +0,10 | **+0,40** |
| perdida de valor (iter 250-300) | 42, y subiendo | **0,27** |
| desviacion del valor / del retorno | 5,3 / 22,6 | 9,1 / 18,2 |
| saturacion de las motoras | 7,8 % | **3,9 %** |
| \|h\| medio de las motoras | 0,727 | 0,283 |
| progreso de entrenamiento (250-300) | 0,285 | 0,291 |
| entropia final | 1,353 | 1,693 |
| **KL por actualizacion** | 1e-5 | **1e-5** |
| test muestreado | 0,251 | 0,221 |
| test codicioso | 0,136 | 0,136 |

El critico pasa de predecir 27,7 para cualquier estado a explicar el 40 % de la varianza, y
de regalo la saturacion de las motoras baja a la mitad (al no dominar el gradiente del
valor, los pesos del codificador crecen menos: es el mismo mecanismo que ya se habia
medido). **La politica no se mueve.** El KL por actualizacion sigue en 1e-5 y el progreso
esta dentro del ruido.

O sea: el critico ciego era un problema real y esta arreglado, pero **no era el que frenaba
a la politica**.

### El siguiente sospechoso, con numero

Comparando las magnitudes de la perdida en las dos corridas (media de las iteraciones
250-300):

| | \|perdida de politica\| | termino de entropia (coef x H) | razon |
|---|---|---|---|
| referencia | 0,00005 | 0,01353 | **250x** |
| normalizado | 0,00005 | 0,01693 | **320x** |

El premio por explorar es **dos ordenes y medio de magnitud** mayor que la senal de mejorar.
La entropia se queda en 1,69 de un maximo de ln(7) = 1,95: una politica casi uniforme, que
es exactamente lo que se ve al jugar (da botonazos razonables pero no se compromete). El
A/B siguiente baja `entropy_coefficient` de 0,01 a 0,001, una linea otra vez.

### Contexto: no somos los primeros en poner al conectoma a jugar

Buscado el 22-09-2026. Tras el mapeo del MaleCNS v1.0 (166.700 neuronas, septiembre de
2026) aparecieron en dias varios proyectos: `fly-games` (Mario, Kung Fu y Doom), `nfly`
(cualquier entorno de Gymnasium; con el cableado congelado y una lectura supervisada juega
al Pong al nivel de una CNN), `FlyBrain-Worldle` (REINFORCE sobre ganancias sinapticas) y
`drosophilos` (un motor tipo Doom dentro del sustrato). La critica publica de esa ola
(P. Mineault, neuroai.science) es que son redes **congeladas** con un decodificador
entrenado encima, que la vision es casi decorativa, y que lo que se ve es ruido recurrente:
"random button mashing in a video game environment".

Lo que distingue a este repo no es llegar primero, que no llegamos: es que aqui **se
entrena el nucleo con mascara** (no solo una lectura), hay **controles emparejados**
(aleatorio con misma densidad, recableado con grados exactos, denso con mismos parametros,
y cero conexiones), los niveles de test son **disjuntos** en el experimento real, las
hipotesis estaban **registradas de antemano**, y el contraste contra apretar botones al
azar se informa siempre. Esa ultima linea es justo la que la critica les reprocha a ellos.

### A/B de la entropia: tampoco era eso (22-09-2026)

`configs/mario_nes_demo_entropia.yaml`, identico al del critico salvo
`entropy_coefficient` de 0,01 a 0,001 (verificado clave a clave).

| corrida | progreso 250-300 | entropia | KL | perdida de valor | test muestreado | test codicioso |
|---|---|---|---|---|---|---|
| referencia | 0,285 | 1,353 | 1,5e-5 | 41,4 | 0,251 | 0,136 |
| + critico normalizado | 0,291 | 1,693 | 0,9e-5 | 0,27 | 0,221 | 0,136 |
| + entropia 10x menor | 0,291 | 1,640 | **0,1e-5** | 0,26 | 0,216 | 0,136 |

Con el premio por explorar diez veces mas pequeno, la entropia baja de 1,69 a 1,64 —casi
nada— y el **KL cae todavia mas, a 1e-6**. Eso se lee al reves de lo que parece: si quitar
el bonus de entropia hace que la politica se mueva MENOS, es que casi todo su movimiento
venia del bonus, no de la ventaja. **El gradiente de politica que viene de la ventaja es
practicamente nulo.**

### Lo que queda descartado, con numero

| hipotesis | medida | veredicto |
|---|---|---|
| las motoras estan saturadas y no llevan senal | desviacion mediana 0,066-0,079 entre pasos; 2-6 de 56 constantes; 3,7-7,8 % saturadas | **descartada** |
| el critico esta ciego | varianza explicada 0,10 -> 0,36-0,40 tras normalizar | **arreglada, y no era el freno** |
| el bonus de entropia aplasta a la politica | con 10x menos, KL baja a 1e-6 | **descartada** |
| la politica no depende de lo que ve | distancia al promedio: mediana 0,17, p90 0,34, maxima 0,50; 4 acciones distintas salen como maximo | **descartada** |

Queda en pie la sospecha estructural: **el gradiente esta truncado a un paso de entorno**.
El estado oculto persiste durante el despliegue pero entra desacoplado en la actualizacion,
asi que el nucleo no puede asignar credito a nada que pase a mas de un paso de distancia.
La comparacion que lo decide es el control **MLP libre** con la receta identica: si tampoco
aprende, el freno esta en la tarea o en el planteamiento de PPO; si aprende, esta en el
nucleo.

---

## 22-09-2026: primera sesion en la GPU (RTX 3060 Ti)

Traspaso hecho: clon nuevo del repo en la maquina con GPU, entorno preparado desde cero.

### Entorno: una diferencia real con lo que decia el traspaso

Python instalado en esta maquina es **3.14**, no 3.11-3.13 como en el portatil. La rueda
`torch==2.14.0` de PyPI (CPU) instala sin problema, pero la variante CUDA que recomendaba
`TRASPASO.md` (`--index-url .../whl/cu124`) **no tiene build para cp314**: solo llega hasta
cp313. Los indices `cu126` y `cu130` si tienen cp314. Instalado con `cu126`:

```
torch 2.14.0+cu126, torch.cuda.is_available() == True, "NVIDIA GeForce RTX 3060 Ti"
```

`pytest -q`: **107 pasan** (antes de regenerar el subgrafo real salian 103 + 4 skip; con
`conectoma_4000` generado, los 4 que dependian de el corren tambien).

### Coste real medido en la 3060 Ti, no estimado

`benchmark.py --config configs/entrenamiento_64.yaml`, con datos sinteticos:

| modelo | parametros | s/paso fase A | s/paso fase B (h=20) | h/semilla |
|---|---|---|---|---|
| mosca | 311.328 | 0,015 | 0,276 | 0,54 |
| aleatorio | 311.328 | 0,015 | 0,266 | 0,53 |
| recableado | 311.328 | 0,015 | 0,260 | 0,51 |
| denso | 307.984 | 0,018 | 0,362 | 0,70 |
| sin_nucleo | 135.584 | 0,008 | 0,107 | 0,22 |
| fno | 1.188.353 | 0,023 | 0,209 | 0,48 |

**18 entrenamientos: 8,9 h** (contra las 197 h de la CPU del portatil). Pico de memoria del
fno, el mas pesado: 2,9 GB de los 8 GB de la tarjeta — sobra margen de sobra. La fase B
sigue llevandose la mayoria (83 %), igual que en CPU.

### Datos reales de fluidos 64x64, generados en la GPU

`python -m data.simulate_ns --config configs/ns_64.yaml`: **~80 s** (contra los ~13 min
estimados de CPU; el simulador trabaja en float64, y aun asi la GPU lo saca mucho mas
rapido que 6 hilos de CPU aqui). Los cinco criterios de aceptacion de la fase 1 se repiten
sobre el set completo (200+40+40 trayectorias, no la validacion pequena):

| Criterio | Valor | Limite |
|---|---|---|
| Deriva de energia | +0,025 %/t (train) | < 5 % |
| CFL maximo | 0,213 | < 0,5 |
| Reproducible bit a bit | True, desviacion 0,000e+00 | — |

### Las 3 que deciden, a escala real: la mosca SI se despega (giro respecto al prototipo)

`train.py --config configs/entrenamiento_64.yaml --model {mosca,sin_nucleo} --seed 0`,
seguido de `evaluate.py`. N=4.000, 128 modos, malla 64x64, 20.000 pasos fase A + curriculo
5/10/20 completo en fase B (no el munon de 60 pasos del cribado de 32x32).

| modelo | L2 1 paso | L2 despliegue (50) | t hasta corr<0,8 |
|---|---|---|---|
| persistencia | 0,1010 | 0,8866 | 1,08 |
| **mosca** | **0,1004** | **0,7884** | **1,15** |
| sin_nucleo | 0,1018 | 0,8210 | 1,07 |

- **vs persistencia:** -0,0006 a un paso, **-0,0982 en despliegue** (11 % relativo mejor),
  +0,07 en tiempo de correlacion.
- **vs sin_nucleo:** -0,0014 a un paso, **-0,0326 en despliegue** (4 % relativo mejor),
  +0,0775 en tiempo de correlacion.

Es la lectura opuesta al cribado de 32x32 (RESULTADOS.md, seccion 22-09-2026 anterior),
donde la mosca **perdia** contra la persistencia en despliegue por 0,629: aqui gana. La
diferencia estructural es la fase B completa (curriculo real hasta horizonte 20, no 60
pasos a horizonte 5) y la escala real (128 modos, N=4.000, no 64 modos y N=500).

**Aviso obligatorio:** los p-valores de permutacion por trayectoria salen significativos
(p < 0,0001 en casi todos los contrastes), pero **con una sola semilla el veredicto entre
semillas no es juzgable** — es la advertencia que ya lleva incorporada `evaluate.py`. La
lectura es prometedora, no confirmada; para eso estan las otras dos semillas por modelo.

**Decision (segun el criterio pre-registrado en TRASPASO.md 5):** el nucleo se despega,
asi que se lanza la tanda completa. `mosca_s0` ya contaba (mas arriba); mosca_s1 tardo
tambien **19 min real** (contra los 32 min que estimaba el benchmark: la parada temprana o
el patience recortan algo el tiempo real).

### La tanda completa de 18, lanzada y en marcha

`scripts/tanda.sh configs/entrenamiento_64.yaml`, uno a uno, saltando lo ya hecho. Al cierre
de esta sesion: **mosca s0 y s1 terminados, mosca s2 en marcha**, 15 pendientes. Se
interrumpio para apagar la maquina (sin drama: instantanea cada 500 pasos, se retoma con el
mismo comando). Pendiente para la proxima sesion: dejarla terminar (~8 h que quedan mas o
menos, contando lo ya hecho) y correr `evaluate.py` sobre las 18 completas.

### Fase 6: el MLP libre SI aprende — el freno esta en el nucleo

Pendiente que habia quedado a medias en el portatil (ver seccion anterior). Corrido en la
GPU: `configs/mario_nes_demo_critico.yaml --model mlp --seed 0`, 300 iteraciones, **~40
min** (el cuello es el emulador de NES en CPU, la GPU no acelera la recoleccion).

```
python diagnostico_rl.py artifacts/mario_nes_demo_critico/mlp_s0.pt --config configs/mario_nes_demo_critico.yaml
```

| | nucleo con mascara (referencia) | **MLP libre** |
|---|---|---|
| varianza explicada del critico | 0,10 -> 0,36-0,40 (normalizado) | **+0,494** |
| progreso muestreado | 0,098 (constante: "saltar siempre") | **0,36-0,56** segun nivel |
| entropia de la politica | se aplana | **1,376** de un maximo de 1,946, viva |
| acciones distintas como maximo probable | 1 (degenerada) | **5** |

El MLP **si** aprende, y con margen: supera el salto constante (0,381) en 3 de 4 niveles de
validacion. Segun el criterio pre-registrado (RESULTADOS.md, seccion del critico ciego): si
el MLP aprende, el freno esta en el nucleo, no en la tarea ni en el planteamiento de PPO. El
unico sospechoso que queda en pie es el gradiente truncado a un paso de entorno — cambiarlo
es una desviacion del diseno y esta pendiente de consultar con el usuario antes de tocar
codigo (no se ha tocado).

`diagnostico_rl.py` no soportaba politicas sin roles anatomicos (`MLPPolicy` no tiene
`.motor`/`.sensory`: no es recurrente, no hay "neuronas motoras" que leer). Se le anadio un
guard (`hasattr(politica, "motor")`) para que la varianza explicada se calcule igual y solo
se omita la seccion de diagnostico de motoras cuando no aplica. Es un arreglo de
herramienta, no un cambio de diseno del experimento.

### Pendiente para la proxima sesion, por orden

1. Dejar terminar la tanda de 18 (en marcha, interrumpida para apagar) y correr
   `evaluate.py` sobre el resultado completo — es la que responde a la pregunta del
   proyecto con las 3 semillas.
2. Decision del usuario: **el gradiente truncado a un paso de entorno** en la fase 6 (unico
   sospechoso que queda, cambiarlo es desviacion del diseno).
3. El checkpoint de la demo de NES entrenada (`artifacts/mario_nes_demo/mosca_s0.pt`, las
   300 iteraciones del 22-09-2026) vive solo en el disco del portatil — `artifacts/` esta
   en `.gitignore` y no viaja con el clon. Para verlo jugar con el visor anatomico
   (`visor_nes.py`) en esta maquina hay que traerlo a mano o reentrenarlo aqui (~40 min,
   limitado por el emulador en CPU, no por la GPU).

---

## 23-09-2026: el gradiente truncado NO es el freno de la fase 6 (portatil, sin GPU)

Revision del veredicto de la seccion anterior, con tres medidas baratas y ningun cambio
del diseno.

### 1. El argumento no se sostenia: el MLP que aprende no tiene memoria

`MLPPolicy.forward` devuelve la `h` sin tocarla: es una red **sin memoria**. Si aprende a
jugar, jugar bien no necesita memoria. Alargar el gradiente mas alla de un paso de entorno
solo sirve para que el nucleo aprenda memoria, asi que no puede ser lo que le impide
aprender lo que el MLP aprende sin ella.

### 2. Medido: el nucleo entrenado no usa la h arrastrada

`diagnostico_rl.py` (seccion nueva) sobre `artifacts/mario_nes_demo/mosca_s0.pt`, 4
episodios, semillas 101-104: poner `h = 0` en cada paso cambia la politica **0,006** de
variacion total (mediana; p90 0,008) y la accion mas probable coincide en el **100 %** de
los pasos. Cambiar la observacion por la de otro momento la mueve 0,018 (p90 0,139). El
nucleo ya se comporta como una red sin memoria.

### 3. Medido: el nucleo SI puede representar una buena politica (clonado de conducta)

`clonado_rl.py --config configs/mario_proto.yaml --semilla {0,1,2}`: entorno propio,
N=500, 56+56 puertos, `n_shared` 15 (la misma escala que la demo de NES). Entropia cruzada
contra la heuristica en 300 niveles de entrenamiento (56.759 pasos), 1.500 pasos de Adam,
lr 1e-3, lote 256, sin memoria (`h = 0`). Progreso jugando en codicioso sobre 40 niveles
de validacion. Las semillas cambian la inicializacion y el orden de los lotes; los niveles
son los mismos.

| modelo | s0 | s1 | s2 | media |
|---|---|---|---|---|
| heuristica (maestra) | | | | 0,946 |
| mlp | 0,861 | 0,844 | 0,865 | 0,857 |
| denso | 0,852 | 0,859 | 0,865 | 0,859 |
| recableado | 0,845 | 0,835 | 0,828 | 0,836 |
| aleatorio | 0,828 | 0,828 | 0,828 | 0,828 |
| **mosca** | 0,806 | 0,835 | 0,828 | **0,823** |
| sin_nucleo | 0,710 | 0,754 | 0,798 | 0,754 |
| saltar siempre | | | | 0,747 |
| siempre derecha | | | | 0,229 |

El acierto por acciones no sirve para leer esto (la clase mayoritaria da el 91,4 % y todos
sacan el 99,5 %); lo que separa es el recall del salto (0,94-0,97) y el progreso al jugar.

Lectura:
- Los tres cableados con nucleo aprenden **por supervision** una politica que le gana al
  salto constante por ~0,08, en 1.500 pasos. Con PPO, el prototipo no se lo ganaba nadie.
- El nucleo aporta frente a no tenerlo: `sin_nucleo` se queda en el salto constante.
- Mosca, aleatorio y recableado empatan (0,82-0,84, dentro de la dispersion entre
  semillas). No es una comparacion del experimento: es un diagnostico de capacidad.

### Veredicto revisado

El freno de la fase 6 **no es** de representacion ni de memoria: el nucleo puede expresar
una politica buena y sin memoria, y la aprende enseguida cuando la senal es supervisada.
Lo que falla es **llevarlo hasta ella con PPO**. Quitar el truncado seria una desviacion
del diseno cara (BPTT por el despliegue) que, segun estas medidas, no ataca el problema.

Pista medida pero **no decisiva**: en el checkpoint entrenado, el gradiente del valor
(norma 6,07) es ~90 veces el de la politica (0,066), y `clip_grad_norm_` a 1,0 sobre todos
los parametros multiplica la actualizacion entera por 0,165, asi que la politica aprende de
hecho a ~5e-5 en vez de 3e-4. Al inicializar el factor es parecido en el MLP (0,42) y en la
mosca (0,51), y con el critico normalizado el KL siguio en 1e-5, asi que esto solo no lo
explica. Al inicializar, el gradiente que llega al codificador desde la politica es ~1.000
veces menor que el que llega desde el valor en todos los modelos, MLP incluido: el tronco
compartido lo esculpe el critico.

### Correccion a la tabla del MLP de la seccion anterior

El "0,098" del nucleo es su progreso **codicioso**; el muestreado de la demo reentrenada es
**0,285**. La fila "progreso muestreado" comparaba cosas distintas. El MLP (0,36-0,56) sigue
por encima, pero la distancia es menor de lo que decia la tabla, y falta su contraste
contra `azar` (esta en la lista fija de `evaluate_rl.py`).

---

## 23-09-2026 (tarde): PPO no destruye la politica buena, no la encuentra (portatil)

Siguiente paso de la seccion anterior: si el nucleo puede representar la politica y PPO no
lo lleva hasta ella, ¿es que el gradiente de PPO la empuja lejos, o que desde cero no la
alcanza? Se arranca PPO **desde la politica clonada** y se compara con arrancar de cero.

Herramienta, sin cambio de diseno: `clonado_rl.py --guarda-pesos` guarda cada politica
clonada en `<output_dir>/clonado/`, y `train_rl.py --desde <checkpoint>` arranca de esos
pesos (la normalizacion de la entrada viaja en el checkpoint y no se vuelve a medir) y
escribe en `<output_dir>/desde_<carpeta>/`. Receta PPO identica a `configs/mario_proto.yaml`
(400 iteraciones, sin `normalise_returns`). Las lineas "desde cero" de las semillas 1 y 2
son el mismo YAML con `output_dir: artifacts/mario_proto/desde_cero` (el prototipo original
solo tenia la semilla 0). Mosca y MLP, semillas 0-2, 18 checkpoints evaluados en
codicioso sobre los **mismos 40 niveles de test** con `evaluate_rl.py --checkpoint`.

| modelo | desde cero | solo clonado | clonado + 400 it de PPO |
|---|---|---|---|
| mosca s0 / s1 / s2 | 0,823 / 0,824 / 0,824 | 0,904 / 0,910 / 0,910 | 0,947 / 0,902 / 0,924 |
| **mosca, media** | **0,824** | **0,908** | **0,924** |
| mlp s0 / s1 / s2 | 0,841 / 0,823 / 0,827 | 0,923 / 0,923 / 0,932 | 0,869 / 0,960 / 0,890 |
| **mlp, media** | **0,830** | **0,926** | **0,906** |

Referencias en esos niveles: azar 0,521, siempre derecha 0,255, **salto constante 0,793**,
heuristica 0,978. Contra el salto constante (permutacion pareada por nivel): **ninguna** de
las seis de cero se distingue (+0,03 a +0,05, p 0,42-0,60); las seis clonadas y las seis
clonadas + PPO le ganan (+0,08 a +0,17; p < 0,05 en 10 de 12, 0,054 y 0,096 las otras
dos). Salida completa en `artifacts/mario_proto/desde_clonado/rl_evaluacion_cero_clon_clonppo.*`.

Lectura:
- **PPO no destruye la politica buena.** Desde el clonado la mantiene: mosca +0,016 de
  media, MLP -0,020, con una dispersion entre semillas (±0,04) mayor que el efecto. No hay
  un gradiente que empuje de vuelta al salto constante.
- **Desde cero no la encuentra**, ni el nucleo ni el MLP libre: los seis se quedan pegados
  al salto constante (la mosca da 0,823-0,824 en las tres semillas, la misma politica
  deterministica). En este entorno el MLP tampoco aprende, a diferencia del NES.
- O sea: la politica buena es **otra cuenca**, y PPO desde una inicializacion aleatoria cae
  en la de la politica degenerada y de ahi no sale. El freno es de **exploracion**, no de
  representacion (seccion anterior), ni de memoria, ni de que el gradiente de PPO sea
  toxico. Y en el prototipo no es especifico del nucleo.

Sobre la pista del recorte global de la seccion anterior: Adam es invariante a multiplicar
todo el gradiente por una constante, asi que un factor de recorte de 0,165 **no** convierte
por si solo el lr de 3e-4 en 5e-5. Lo que si puede pesar es que en los parametros
compartidos (codificador y nucleo) la suma ponderada la domina el valor (~90x): ahi la
politica apenas deja huella en los momentos de Adam. Coherente con lo de arriba, pero no
medido aparte.

### Lo que esto deja para decidir (todo es desviacion del diseno: se consulta antes)

1. **Arrancar las seis politicas del clonado de la heuristica** y afinar con PPO. Funciona
   (tabla), pero el experimento pasa a medir "cuanto mejora PPO una politica ya buena" y la
   maestra fija el techo; ademas solo vale en el entorno propio (la heuristica lee el estado
   interno del nivel, en el NES no hay heuristica).
2. **Explorar mas desde cero**: entropia al principio mas alta con decaimiento, o un
   curriculo de niveles (sin enemigos ni tubos al principio). Mantiene el experimento como
   esta; no garantiza salir de la cuenca.
3. **Dejar la fase 6 como esta** y darla por "sin resolucion": un empate con todos pegados a
   la politica degenerada no dice nada del cableado.

---

## 23-09-2026: la mosca se pasa el 1-1 de NES (demo, por imitacion)

Por encargo del usuario: que juegue al Mario de verdad. Es la receta que funciono en el
prototipo (seccion anterior), llevada al NES. **Es demo, no experimento**: entrena y mide
en el 1-1, y lo que aprende lo copia de una heuristica escrita a mano, no lo descubre.

### Una maestra para el NES

`envs/nes_mario.JugadorHeuristico`, leyendo la RAM: corre con B, salta con A si hay muro,
hoyo o enemigo delante, mantiene A mientras sube y la suelta al caer. Dos trampas medidas
por el camino: (1) soltar A al aterrizar en vez de al caer perdia un paso y un goomba lo
alcanzaba (moria en x = 693); (2) mirar solo la casilla de debajo tomaba el borde de la
tuberia alta por un hoyo, saltaba desde ella y caia en el hoyo de verdad (x = 1.130). Con
"hoyo = columna vacia hasta el fondo" **se pasa el 1-1 entero** en 315 pasos. En el resto
de niveles no vale (progreso 0,06-0,55 en 1-2 ... 8-1): no sabe de enemigos voladores,
plataformas ni castillos. Test: `test_la_heuristica_se_pasa_el_1_1`.

### Clonado con memoria

Sin memoria (`h = 0` en cada paso, como en el prototipo) acierta el 92,7 % de las
acciones y solo recorre el **13,6 %**: la maestra decide cosas que no estan en la
observacion (si sube o baja, si acaba de aterrizar con A pulsada), y si la red se
equivoca ahi se queda con A apretada contra la primera tuberia. `clonado_rl.py --ventana
K` entrena sobre tramos de K pasos seguidos arrastrando la h y juega con ella: el nucleo
recurrente lo recuerda. Tambien `--ruido p`: se ejecuta una accion al azar con
probabilidad p pero la etiqueta es la de la maestra, porque sin ruido las 20 partidas del
1-1 son identicas.

| clonado (mosca, N = 500, 1-1) | acierto val | progreso |
|---|---|---|
| sin memoria, 20 partidas, ruido 0,10, 1.500 pasos | 0,927 | 0,136 |
| ventana 8, 20 partidas, ruido 0,10, 1.500 pasos | 0,966 | 0,777 |
| ventana 8, 40 partidas, ruido 0,15, 3.000 pasos | 0,977 | 0,942 |
| **ventana 8, 40 partidas, ruido 0,15, 4.000 pasos** | **0,980** | **0,971** |

`evaluate_rl.py --checkpoint`, codicioso, 12 partidas del 1-1:

| | progreso | bandera |
|---|---|---|
| **mosca clonada** | **0,947** | **4 de 12** |
| mosca con PPO (300 it, la demo anterior) | 0,098 | 0 |
| azar / saltar siempre | 0,222 / 0,098 | 0 |

Contra el azar +0,725, p = 0,0004. Cuando no llega a la bandera se queda atascada contra
la escalera final (x = 2.996, 94 %) hasta que salta el corte por atasco; una vez de doce
en x = 2.470. Video: `docs/media/mosca_clonada_1-1.mp4` (semilla 5, bandera).
Checkpoint: `artifacts/mario_nes_demo/clonado/mosca_s0.pt` (no viaja con el repo).

Encuadre si se publica: "una red cableada como el cerebro de una mosca aprende a imitar a
un jugador escrito a mano y se pasa el 1-1". No dice nada del cableado (no hay control
aleatorio en esta demo) ni de aprendizaje por refuerzo.

## 24-09-2026: controles del clonado del 1-1 (demo, portatil)

La misma receta del clonado con memoria de la seccion anterior (`--ventana 8 --ruido 0.15`,
40 partidas de demostracion, 4.000 pasos, lote 64, semilla 0), con los seis modelos en vez
de solo la mosca. Mismo nivel e imitacion: **diagnostico, no resultado de la fase 6b.**
Salida en `artifacts/mario_nes_demo/clonado/clonado_controles_s0.{json,log}`.

| modelo | acierto val | progreso (4 partidas) |
|---|---|---|
| mosca | 0,976 | 0,942 |
| aleatorio | 0,990 | 0,968 |
| recableado | 0,987 | 1,000 |
| denso | 0,984 | 0,971 |
| `sin_nucleo` | 0,928 | 0,136 |
| MLP | 0,931 | 0,136 |

Referencias: heuristica 1,000, siempre derecha 0,093, saltar siempre 0,098. La clase
mayoritaria acierta 0,542.

Lectura:
- **Lo que separa es la memoria, no el cableado.** Los cuatro nucleos recurrentes se pasan
  el nivel o casi (0,94-1,00); los dos modelos sin recurrencia (`sin_nucleo` y el MLP, que
  devuelve la h sin tocarla) se quedan en 0,136, exactamente lo mismo que la mosca clonada
  sin memoria de la seccion anterior. Coincide con que la maestra decide con informacion
  que no esta en la observacion (si sube o baja, si A esta pulsada).
- **La mosca es la peor de los cuatro recurrentes**, pero con una semilla y 4 partidas la
  diferencia (0,03-0,06) no se puede leer. Entre cableados, empate.
- Para el video: el 0,947 de la mosca **no es merito del conectoma**; un grafo aleatorio o
  denso con la misma memoria clona igual o mejor.

### Tres semillas (24-09-2026, mismo dia)

Semillas 1 y 2 con la misma receta (`clonado_controles_s{1,2}.{json,log}`). Las
demostraciones son las mismas en las tres (6.855 pasos, identicas clases): la semilla
cambia la inicializacion, el entrenamiento y las partidas de juego, no los datos.

| modelo | s0 | s1 | s2 | media | acierto val medio |
|---|---|---|---|---|---|
| mosca | 0,942 | 1,000 | 1,000 | 0,981 | 0,981 |
| aleatorio | 0,968 | 0,968 | 1,000 | 0,979 | 0,987 |
| recableado | 1,000 | 1,000 | 1,000 | 1,000 | 0,989 |
| denso | 0,971 | 0,971 | 0,777 | 0,906 | 0,983 |
| `sin_nucleo` | 0,136 | 0,136 | 0,700 | 0,324 | 0,930 |
| MLP | 0,136 | 0,136 | 0,685 | 0,319 | 0,941 |

- **Correccion a la lectura de s0:** 0,136 no es un muro. En s2 los dos modelos sin
  recurrencia pasan la primera tuberia y llegan a 0,70. Sin memoria el resultado es
  bimodal (se atasca en la tuberia o la pasa); con memoria se pasan el nivel casi siempre.
  El acierto en validacion lo confirma sin depender de 4 partidas: 0,93-0,94 sin memoria
  contra 0,98-0,99 con ella, en las tres semillas.
- **Entre cableados, empate.** Mosca 0,981, aleatorio 0,979, recableado 1,000; el denso
  baja por una sola partida mala en s2. En acierto la mosca es la mas baja de los cuatro
  (0,981 contra 0,983-0,989), diferencia del tamano del ruido entre semillas.
- Sigue siendo el mismo nivel e imitacion: no dice nada de la fase 6b.


## 28-09-2026: fase 6c, que aprenda sola — preparada, sin resultado todavia (portatil)

Encargo del usuario: que juegue a Mario **aprendiendo sola** y en niveles que no ha visto.
Se descarta imitar (a la heuristica o a una persona). Cambios, todos con su mando en el
YAML y apagados por defecto (sin tocarlos, todo lo anterior es identico, con tests):

- **Observacion con velocidad vertical y A pulsado** (`nes.motion_features`). RAM 0x009F
  (con signo, -5 subiendo a +5 cayendo) y bit 0x80 de 0x06FC, calibrados saltando en el
  1-1. Es la informacion que la heuristica usa y la red no veia: sin memoria no se podia
  clonar (0,93 de acierto contra 0,98 con memoria, seccion del 24-09).
- **Arranques en la frontera** (`nes.frontier_probability`, estilo Go-Explore). Un
  episodio de entrenamiento puede empezar `frontier_rewind` pasos antes del punto mas
  lejano al que la propia red llego en ese nivel. Medido: el NES es determinista, rejugar
  espera y botones da la RAM identica byte a byte; el estado se fija con la copia de
  seguridad del emulador. Val y test arrancan siempre en la salida (forzado y con test), y
  el progreso de entrenamiento se informa solo con los episodios que empiezan en la salida.
- **Entropia decreciente** (`ppo.entropy_final`), de 0,02 a 0,002.
- **Emuladores en procesos** (`parallel_envs`). Medido con 16 entornos: 159 ms por paso
  vectorial en serie, 48 ms en paralelo (x3,3 con 6 nucleos). Mismas semillas y mismas
  observaciones que en serie (test).

Prueba de 30 iteraciones de la mosca (`configs/mario_nes_sola.yaml`, N=500): 10-11 s por
iteracion (recoleccion 9-11 s, actualizacion 0,4 s). Entre el 17 % y el 68 % de los
episodios arrancan en la frontera. Progreso de val 0,101 (muestreado 0,103), entropia
1,946 -> 1,941, **KL por actualizacion ~1e-5**. Con 30 de 1.500 iteraciones no dice nada
del aprendizaje; el KL es lo que hay que vigilar en la corrida larga, porque es el sintoma
de la fase 6. La corrida (mosca y MLP, 1.500 iteraciones) va a la GPU: TRASPASO.md.

## 14-10-2026: fase 6c, primera semilla (Google Cloud) y plan de las replicas

### Corrida s0 (13-10-2026)

VM `mosca-6c` (c2d-highcpu-8, solo CPU), codigo 44b5b40, de 16:21 a 20:38 UTC, salida 0,
sin reanudaciones. Mosca 2,22 h, MLP 2,02 h. Parametros: mosca 27.253 (500 neuronas),
MLP 27.343 (71 neuronas). Artefactos en `gs://mosca-runs-317660842762/6c/` y
`artifacts/mario_nes_sola/` (no viajan con el repo).

Entrenamiento, medias por bloques de 250 iteraciones (progreso de train: solo episodios
que arrancan en la salida):

| | 0-250 | 250-500 | 500-750 | 750-1000 | 1000-1250 | 1250-1500 |
|---|---|---|---|---|---|---|
| KL mosca | 7,4e-5 | 1,3e-4 | 1,5e-4 | 1,4e-4 | 7,5e-5 | 1,5e-5 |
| KL MLP | 9,8e-4 | 1,1e-3 | 9,7e-4 | 7,1e-4 | 4,0e-4 | 7,9e-5 |
| progreso train mosca | 0,190 | 0,248 | 0,324 | 0,379 | 0,410 | 0,448 |
| progreso train MLP | 0,196 | 0,324 | 0,456 | 0,558 | 0,626 | 0,669 |

- **El sintoma de la fase 6 no aparece.** El KL de la mosca esta un orden de magnitud por
  encima del 1e-5 fijo de entonces; el del MLP, dos. Entropia de la mosca 1,90 -> 1,05,
  del MLP -> 0,81.
- **Val se estanca desde la iteracion ~750** en 0,17-0,20 (codicioso, 12 partidas en
  1-3/3-3/5-3) en los dos modelos, mientras train sigue subiendo: lo que aprenden en los 8
  niveles de entrenamiento se transfiere poco a los x-3. El MLP es el que mas se separa
  (0,67 en train). Mas iteraciones no parece que vayan a cambiar val. El "mejor val", que
  solo se registra y no elige checkpoint, fue 0,204 (mosca, it 924) y 0,197 (MLP, it 724).

Evaluacion de cierre (`evaluate_rl.py --muestrea --repeticiones 6`, semillas de test 0..15):

| modelo | progreso | bandera |
|---|---|---|
| azar | 0,135 | 0 |
| siempre derecha | 0,078 | 0 |
| salto constante | 0,081 | 0 |
| **mosca** | **0,265** | 0 |
| **MLP** | **0,333** | 0 |

Mosca contra azar +0,131 (p < 0,0001), contra salto constante +0,184 (p < 0,0001), contra
el MLP -0,068 (p = 0,0019). **Ningun modelo llega a la bandera en ningun nivel no visto.**

### Tres problemas de esa evaluacion (14-10-2026)

1. **Los "16 niveles" de test son 4.** En el NES la semilla elige uno de los cuatro niveles
   de test (2-3, 4-3, 6-3, 8-1) y una espera de 0 a 20 fotogramas. Las semillas 0..15 caen
   7/4/3/2 veces (8-1 / 6-3 / 4-3 / 2-3): el 8-1 pesa el 44 % de la media, y la prueba de
   permutacion trata como independientes partidas del mismo nivel. **Las p de arriba estan
   infladas.** Con el nivel como unidad y cuatro niveles, la p bilateral exacta no puede
   bajar de 2/16 = 0,125: con el test solo, ninguna diferencia entre modelos puede salir
   significativa.
2. **Una sola semilla de entrenamiento.** La p = 0,0019 dice que *este* MLP juega mejor que
   *esta* mosca en esos niveles, no que una arquitectura sea mejor que la otra. Para eso
   la unidad tiene que ser el entrenamiento.
3. **El ruido del muestreo da la vuelta al orden.** El test muestreado de una partida que
   hace `train_rl.py` al acabar da mosca 0,276 y MLP 0,216; el de `evaluate_rl.py` con seis
   repeticiones, 0,265 y 0,333. Codicioso: 0,195 y 0,289.

Una evaluacion parcial con el protocolo nuevo (abajo) dio **0,198** para la mosca s0 en los
siete niveles equilibrados, contra el 0,265 del test desequilibrado. Se corto a los 10 min
(regla 6) antes de llegar al MLP. La completa va en la nube.

### Cambios en `evaluate_rl.py` (por defecto da exactamente lo mismo que antes)

- `--split val+test`. Val (1-3, 3-3, 5-3) es tan limpio como test: el checkpoint que se
  evalua es el de la ultima iteracion, y el mejor en val solo se registra. Son 7 niveles
  no vistos en vez de 4, y la p minima por nivel baja a 2/128 = 0,016.
- `--por-nivel K`: las primeras K semillas de cada nivel, asi cada nivel pesa igual. En
  cada partida se comprueba que el nivel jugado es el previsto (`nivel_previsto`, con un
  test contra `NesMarioEnv.reset`).
- Tres pruebas por pareja: **por partida** (la de siempre), **por nivel** (exacta, los 2^n
  signos sobre las medias por nivel) y **por semilla** (exacta, todas las particiones de
  las medias de cada entrenamiento). Cada una informa de la p minima que puede dar.
- El JSON guarda cada partida y cada semilla. `--modelos`, `--semillas` y `--salida` para
  evaluar las replicas sin tocar el YAML con el que se entrena.
- 12 tests nuevos (`tests/test_evaluate_rl.py` y uno en `tests/test_nes_mario.py`); 140 en
  verde.

### Replicas en marcha (desde el 14-10-2026, 06:56 UTC)

Los seis modelos del diseno con las semillas 0, 1 y 2, con la receta y el codigo exactos
de la s0: 44b5b40, comprobado por sha256 de los 64 `.py`/`.yaml` y de los tres cableados
contra el repo. Faltan 16 entrenamientos. Van en dos VMs, porque la cuenta tiene un limite
de 12 vCPU: `mosca-6c` (8 vCPU, ~2,7 h por modelo) y `mosca-6c-b` (4 vCPU, ~4 h). Las dos
comparten una cola en `gs://mosca-runs-317660842762/6c-replicas/`, donde cada trabajo se
reclama de forma atomica creando un objeto. Orden: aleatorio, recableado y `sin_nucleo`
en s0; luego los cinco en s1; luego en s2; el denso al final. Unas 28 h. La ultima VM en
acabar lanza la evaluacion final y se apaga.

### Plan de analisis (fijado antes de ver ninguna replica)

- **Metrica principal:** progreso con politica muestreada en los 7 niveles no vistos
  (val+test), 8 partidas por nivel, 6 repeticiones, semilla de muestreo 0
  (`rl_evaluacion_final_muestreada.json`). Secundarias: politica codiciosa
  (`rl_evaluacion_final_codiciosa.json`) y bandera.
- **La pregunta del proyecto:** mosca contra aleatorio y contra recableado. La prediccion
  registrada en el diseno es H0: empate.
- **Regla de decision.** Se dice que el cableado de la mosca gana (o pierde) contra un
  control solo si se cumplen las dos condiciones:
  (a) p por nivel < 0,05 con correccion de Holm sobre los dos contrastes principales. Con 7
  niveles, eso exige que la diferencia vaya en el mismo sentido en los 7 para el primer
  contraste, o en 6 de 7 para el segundo;
  (b) las tres semillas de la mosca quedan por encima (o por debajo) de las tres del
  control (p por semilla = 0,1, el minimo posible con 3 + 3).
  Si falla cualquiera de las dos, el resultado es **no resuelto a esta escala**. Solo se
  dice "empate" si ademas la diferencia es menor que la dispersion entre semillas de un
  mismo modelo.
- **Secundario, con las mismas reglas y sin Holm:** mosca contra `sin_nucleo` (si no le
  gana, el nucleo no aporta nada en esta tarea y la pregunta del cableado no tiene
  resolucion), contra el denso y contra el MLP; cada modelo contra azar y salto constante.
- **"Juega"**, que es el criterio para publicar: bandera en algun nivel no visto. La s0 no
  lo cumple.
- Lo que no cambia: N = 500 es el prototipo. Escalar a 4.000 se decide despues, con este
  resultado delante.

### Enmienda al plan (14-10-2026, 11:40 UTC)

Escrita sin haber mirado ninguna cifra de las replicas: de las tres que habian terminado
(aleatorio, recableado y `sin_nucleo` en s0) solo se ha mirado el codigo de salida (0) y
que llegaron a la iteracion 1.500. La regla de decision de arriba no cambia. Se anaden dos
controles y un analisis secundario, y la regla pasa a estar escrita como codigo.

**1. Que significa "progreso" en el NES.** Es `max_x / 3180`, la distancia recorrida
dividida por la longitud aproximada del 1-1, no la fraccion del nivel: el 8-1, por
ejemplo, es bastante mas largo. Las comparaciones entre modelos dentro de un nivel no
cambian; "0,265" se lee "unos 840 px", no "el 26 % del nivel".

**2. Auditoria de igualdad de condiciones** (los seis modelos construidos con
`mario_nes_sola.yaml`). Mosca, aleatorio y recableado: 27.253 parametros, 500 neuronas, 56
sensoriales, 56 motoras y 15 compartidas en los tres. Denso: 26.945 parametros (97
neuronas, mismos puertos). MLP: 27.343. `sin_nucleo`: 18.036, a proposito (los puertos de
la mosca sin pesos recurrentes).

**3. Control marginal (`evaluate_rl.py --marginal`).** Cada modelo contra una politica
que pulsa los botones con **sus mismas frecuencias, sorteados sin mirar la pantalla**.
Ganarle al azar uniforme no demuestra que la red reaccione a lo que ve: una buena
proporcion de botones (mucho "derecha" y salto, pero no siempre) ya recorre mas que el
azar y que cualquier accion constante. El JSON guarda tambien las frecuencias de accion de
cada checkpoint. Piloto con la s0 y un protocolo reducido (4 partidas por nivel y 4
repeticiones; orientativo, en el portatil):

| nivel | mosca | su marginal | MLP | su marginal | azar |
|---|---|---|---|---|---|
| 8-1 (exterior) | **0,391** | 0,139 | **0,475** | 0,101 | 0,111 |
| 2-3 | 0,318 | 0,322 | 0,412 | 0,362 | 0,273 |
| otros cinco x-3 | 0,131-0,186 | 0,105-0,128 | 0,127-0,235 | 0,091-0,123 | 0,087-0,110 |
| media de los 7 | 0,208 | 0,148 | 0,244 | 0,142 | 0,128 |

Casi toda la ventaja de la mosca sobre sus propias frecuencias sale del 8-1, el unico
nivel no visto del mismo tipo que los de entrenamiento. En los atleticos le saca de 0 a
0,07, y en el 2-3 nada.

**4. Niveles extra y analisis por tipo de nivel.** Entrenamiento son siete exteriores y
un subterraneo (1-2). De los 7 niveles de val+test, seis son **atleticos** (x-3: copas de
arbol, setas, puentes), un tipo que la red no ha visto nunca, y uno (8-1) es **exterior**.
La evaluacion principal mezcla "nivel nuevo" con "tipo de nivel nuevo". Se anaden, solo
para evaluar, los niveles que no estan en ninguna particion: los exteriores 3-2, 5-2, 6-2,
8-2 y 8-3, y el atletico 7-3 (`evaluate_rl.NIVELES_EXTRA`, `--split extra`). El tipo esta
comprobado con el emulador: fotograma inicial de los 32 niveles y tipo de zona en la RAM
(0x074E: agua 0, exterior 1, subterraneo 2, castillo 3); los atleticos tambien dan 1 en
la RAM, y se separan por la lista conocida de SMB (`evaluate_rl.TIPO_DE_NIVEL`, con test).
No se toca el codigo del entorno: el entrenamiento sigue siendo byte a byte el de la s0.
Asi quedan 6 exteriores no vistos (p minima por nivel 0,031) y 7 atleticos (0,016).

**5. El analisis, escrito como codigo antes de los datos (`analiza_6c.py`, con tests).**
Lee los cuatro JSON de la evaluacion final y aplica la regla. Concreta lo que el texto
dejaba abierto:
- Diferencia = media sobre niveles de (media de semillas de A - media de semillas de B).
- "Semillas separadas" contra una referencia (azar, salto constante), que no tiene
  semillas: todas las semillas del modelo del mismo lado.
- "Dispersion entre semillas" = media de las desviaciones tipicas entre semillas de las
  dos filas (solo las que tienen mas de una).
- Principal: 7 niveles de val+test, Holm sobre mosca-aleatorio y mosca-recableado.
- Secundarios, umbral 0,05: mosca contra `sin_nucleo`, denso y MLP; cada modelo contra
  azar, salto constante y **su marginal**; y todo lo anterior por separado en los 6
  exteriores y en los 7 atleticos (8-1 sale de la principal; el resto, de la extra).
- "Juega": bandera en algun nivel no visto (los 13), con cualquier semilla.

Salidas de la VM: `rl_evaluacion_final_{muestreada,codiciosa}.json`,
`rl_evaluacion_final_extra_{muestreada,codiciosa}.json` y
`analisis_6c_{muestreada,codiciosa}.json`, con el informe legible en
`logs/analisis_6c_{muestreada,codiciosa}.md` del bucket. Lanzador `scripts/gcp/run_replicas_v4.sh`
(cambiado en caliente con `scripts/gcp/cambia_lanzador.sh`, sin tocar los entrenamientos).
`pytest -q`: 153 en verde.

## 19-10-2026: veredicto de la fase 6c (replicas leidas)

Las 16 replicas y la evaluacion final terminaron el 15-10-2026 a las 05:07 UTC; se leyeron
el 19-10. Informe completo en `logs/analisis_6c_{muestreada,codiciosa}.md` del bucket y en
`artifacts/mario_nes_sola/`. Principal: 7 niveles no vistos, muestreada.

| fila | progreso | por semilla |
|---|---|---|
| mosca | 0,195 | 0,198 0,192 0,195 |
| aleatorio | 0,198 | 0,195 0,187 0,211 |
| recableado | 0,203 | 0,206 0,209 0,193 |
| denso | 0,202 | 0,222 0,184 0,200 |
| sin_nucleo | 0,190 | 0,184 0,191 0,194 |
| mlp | 0,213 | 0,227 0,195 0,218 |
| azar | 0,135 | |
| salto constante | 0,090 | |

- **Pregunta del proyecto:** mosca contra aleatorio, **empate** (-0,003, p por nivel
  0,75); contra recableado, **no resuelto** (-0,008, a favor del recableado en 5 de 7).
  La codiciosa dice lo mismo.
- Los seis le ganan al azar, al salto constante y **a su propia marginal** en los 7
  niveles: miran la pantalla. Ninguno llega a la bandera en val+test salvo `sin_nucleo`
  en el 2-3 (2 %); en el 3-2 (extra) la cogen todos, el MLP el 31 %.
- **Limite:** `sin_nucleo` empata con la mosca (+0,005, no resuelto). La tarea apenas usa
  la recurrencia, y entonces el nulo dice poco sobre el cableado.
- **Por que no juega mejor:** en entrenamiento el progreso sube hasta el final (mosca s1
  0,39-0,58, MLP s1 0,62-0,69) y en validacion se congela en ~0,16 desde la iteracion
  ~450. Con 8 niveles de entrenamiento se los aprende de memoria. El NES no tiene niveles
  suficientes para arreglarlo.

## 19-10-2026: pilotos en el entorno procedural (portatil, N=500, una semilla)

Idea: el entorno propio tiene niveles infinitos, que es lo que le falto a la 6c. Receta
de PPO de la 6c (16 x 128, entropia 0,02 -> 0,002, critico normalizado), 300 iteraciones,
mosca y MLP, evaluacion codiciosa en 200 niveles de test.

| piloto | mosca | MLP | mejor constante | heuristica |
|---|---|---|---|---|
| 1. niveles de 60 | 0,845 / 69,5 % | 0,847 / 69,5 % | 0,809 / 67 % | 0,968 / 92,5 % |
| 2. 160, huecos 0,35 | 0,460 / 13 % | 0,465 / 14,5 % | 0,383 / 10 % | 0,780 / 56 % |
| 3. 160 + soltar para saltar | 0,085 / 0 % | 0,130 / 0 % | 0,106 / 0 % | 0,758 / 54,5 % |

(progreso / metas; en el 3 el azar saca 0,230.)

- En 1 y 2 la mosca y el MLP pulsan "derecha + salto + carrera" en el 93 % de los pasos.
  En este motor mantener el salto rebota en cada aterrizaje, asi que esa constante es casi
  buena sin mirar nada. **No estaba en las referencias**: `salto_corriendo` anadida a
  `evaluate_rl.py`. La mosca de la fase 6 (0,808) era exactamente esa constante (0,809).
- `release_to_jump` (regla del NES: hay que soltar el salto para repetirlo, y el boton
  entra en la observacion) hunde todas las constantes. Apagada, el motor da trayectorias
  identicas bit a bit (huella sha256 antes y despues), y tiene tests.
- Con la regla, desde cero no aprende nada en 300 iteraciones. La 6c del NES estaba igual
  en la iteracion 300.
- **Clonado de la heuristica + PPO desde ahi** (`clonado_rl.py --ruido 0.1 --guarda-pesos`,
  `train_rl.py --desde`): mosca clonada 0,294 / 1,5 %, **afinada 0,431 / 10 %** (+0,137
  por PPO, p < 1e-4); MLP clonado 0,570 / 23 %, afinado 0,573 / 27 %. Primera vez que la
  mosca juega niveles no vistos por encima de todas las referencias salvo la maestra. Una
  semilla y sin controles de cableado: piloto.

## 19-10-2026: fase 6d, plan de analisis registrado antes de los datos

Tarea: entorno propio, niveles de `mario.yaml` (160 casillas), `release_to_jump: true`,
N = 500 (`configs/mario_proto_6c_soltar.yaml`). Seis modelos x semillas 0, 1, 2. Por
semilla: clonado de la heuristica (`--ruido 0.1`, 1.500 pasos supervisados, 300 niveles de
demostracion) y PPO desde el clonado, 300 iteraciones. Mismo codigo, mismas semillas y
mismo presupuesto para los seis. Las demostraciones son las mismas en las tres
semillas (niveles de train 0-299, mismo ruido): la semilla cambia la inicializacion,
el orden de los lotes y todo el PPO.

- **Metrica y niveles:** progreso en 200 niveles de test (semillas 3.000.000-3.000.199,
  disjuntas de train y val), politica **codiciosa** (determinista: sin ruido de muestreo).
  Secundaria: muestreada con 3 repeticiones por nivel.
- **Principal:** modelos *afinados* (clonado + PPO). Mosca contra aleatorio y mosca contra
  recableado, con Holm sobre los dos.
- **Regla de decision** (la de la 6c, adaptada a 200 niveles independientes): gana o
  pierde solo si (a) p por nivel < 0,05 con Holm, y (b) las tres semillas de la mosca
  quedan del mismo lado que las tres del control (p por semilla 0,1). Si falla cualquiera,
  **no resuelto**. "Empate" solo si ademas la diferencia es menor que la dispersion entre
  semillas.
- **Secundarios** (umbral 0,05, sin Holm): mosca contra `sin_nucleo`, denso y MLP; cada
  modelo contra azar y `salto_corriendo`; las mismas comparaciones en los modelos solo
  clonados, para separar "imita mejor" de "aprende mejor"; y la ganancia por PPO
  (afinado - clonado) de cada modelo.
- **"Juega":** meta en algun nivel de test, y ganarle al azar y a la mejor constante.
- Lo que este plan **no** puede decir: nada sobre aprender sin maestra (hay imitacion
  delante), ni nada a N = 4.000.
- La regla se aplica a mano sobre `contrastes_por_nivel` y `contrastes_por_semilla` de
  `evaluate_rl.py`. Si hay que cambiar algo, va como enmienda fechada aparte.

### Enmienda al plan de la 6d (20-10-2026, 16:40 UTC, antes de tener ningun dato de evaluacion)

Escrita cuando el afinado acababa de empezar en la VM y no habia ningun JSON de
evaluacion en el bucket. Corrige el plan en lo que no se podia aplicar tal como estaba
escrito; no cambia ninguna comparacion ni ningun umbral.

1. **Donde se lee la p por nivel.** El plan decia `contrastes_por_nivel`, pero
   `evaluate_rl.py` solo lo rellena con 20 niveles o menos, y aqui hay 200: saldra `{}`.
   En el entorno propio cada partida es un nivel distinto, y `contrastes` ya es la prueba
   por nivel: permutacion pareada de signos sobre los 200 niveles, con cada valor
   promediado sobre las 3 semillas (y sobre las repeticiones en la muestreada), 10.000
   permutaciones, semilla 0. El criterio (a) se lee de `contrastes[...]["progreso"]["p"]`,
   con Holm sobre las dos comparaciones principales. El (b), de `contrastes_por_semilla`,
   como estaba.
2. **Comparaciones contra las referencias que el codigo no hace.** `evaluate_rl.py` solo
   enfrenta a mosca y MLP con `azar` y `salto_corriendo`. Para aleatorio, recableado,
   `sin_nucleo` y denso se calculan despues, con la misma funcion
   (`evaluate.prueba_de_permutacion`) y los mismos parametros, a partir de las series que
   guarda el JSON (`series_por_semilla`, promediada por nivel sobre las semillas, contra
   `series_referencias`). Igual en los modelos solo clonados.
3. **Ganancia por PPO.** El plan no fijaba la prueba: es la misma permutacion pareada por
   nivel, afinado contra clonado del mismo modelo, con las series de
   `eval_afinado_codiciosa.json` y `eval_clonado_codiciosa.json`. Umbral 0,05 sin Holm,
   como el resto de secundarios.
4. **Comprobacion previa.** El lanzador no comprueba el codigo de salida de cada trabajo,
   y `evaluate_rl.py` se salta en silencio los checkpoints que falten. Antes de leer nada
   se comprueba que hay 18 checkpoints afinados y que cada modelo tiene 3 semillas en
   `series_por_semilla` de los tres JSON. Si falta alguna, el analisis se para y se avisa.

## 21-10-2026: veredicto de la fase 6d (leido con el analisis registrado)

La tanda relanzada el 20-10 a las 16:26 UTC termino a las 17:20 UTC con codigo 0: 18
clonados, 18 afinados hasta la iteracion 300 y las tres evaluaciones. `analiza_6d.py`
paso la comprobacion previa (seis modelos x tres semillas x los mismos 200 niveles en los
tres informes) y sus 27 p coinciden con las de `contrastes`. Artefactos en
`artifacts/mario_proto_6c_soltar_gcp/` (con `logs_gcp/` y `analisis_6d.{json,txt}`).

Principal: afinados (clonado + PPO), politica codiciosa, 200 niveles de test.

| fila | progreso | metas | por semilla |
|---|---|---|---|
| mosca | 0,462 | 10,8 % | 0,431 0,482 0,474 |
| aleatorio | 0,568 | 21,7 % | 0,598 0,574 0,534 |
| recableado | 0,527 | 19,0 % | 0,633 0,447 0,501 |
| sin_nucleo | 0,477 | 10,8 % | 0,513 0,474 0,444 |
| denso | 0,496 | 9,8 % | 0,468 0,494 0,525 |
| mlp | 0,562 | 24,3 % | 0,657 0,596 0,434 |
| azar | 0,230 | 0 % | |
| salto_corriendo (mejor constante) | 0,106 | 0 % | |
| heuristica (la maestra) | 0,758 | 54,5 % | |

- **Pregunta del proyecto:** mosca contra aleatorio, **pierde** (-0,106; la mosca va
  mejor en 36 de 200 niveles; p por nivel < 1e-4 con umbral de Holm 0,025; sus tres
  semillas quedan por debajo de las tres del aleatorio). Contra recableado, **no
  resuelto** (-0,064, p < 1e-4, pero las semillas se solapan: recableado s1 saca 0,447).
  Es la primera vez que la regla registrada da algo distinto de empate o no resuelto, y
  sale **en contra** de la mosca. H0 predecia empate.
- La muestreada (secundaria) apunta al mismo lado con otro reparto: contra aleatorio
  **empate** (-0,030), contra recableado **pierde** (-0,044, semillas separadas). En
  ninguna de las dos evaluaciones queda la mosca por encima de un cableado de control.

| muestreada | mosca | aleatorio | recableado | sin_nucleo | denso | mlp |
|---|---|---|---|---|---|---|
| progreso | 0,587 | 0,617 | 0,631 | 0,557 | 0,578 | 0,676 |
| metas | 28,4 % | 29,6 % | 31,8 % | 22,8 % | 26,8 % | 40,4 % |

- **Tras PPO el nucleo sigue sin aportar:** mosca contra `sin_nucleo`, **empate** (-0,015,
  p 0,25). Antes de PPO si aporta: clonada, la mosca le gana a `sin_nucleo` (0,299 contra
  0,222), pero ya queda por debajo del aleatorio (0,375) y del recableado (0,371).
- **PPO sube a los cinco nucleos y al MLP no:** +0,163 mosca, +0,193 aleatorio, +0,156
  recableado, +0,255 `sin_nucleo`, +0,387 denso, todos significativos; MLP -0,001. El MLP
  ya llega a 0,563 solo imitando.
- **Los seis juegan:** llegan a la meta en niveles no vistos (codiciosa: mosca 37 de 200,
  aleatorio 80, recableado 72, `sin_nucleo` 34, denso 49, MLP 101) y le ganan al azar y a
  la mejor constante. Lo que la 6d si demuestra es que la receta clonado + PPO funciona
  con cualquier cableado.
- **En la practica la tarea es un bit por paso.** Los 18 modelos codiciosos usan casi solo
  "derecha + carrera" y "derecha + salto + carrera": deciden cuando saltar. La mosca salta
  el 64 % de los pasos en las tres semillas, el aleatorio el 46-52 %. La mosca ya salta asi
  recien clonada (65-68 %) y PPO no lo cambia.
- **Limites:** N = 500, no el subgrafo de 4.000. La codiciosa es una foto de la ultima
  iteracion, y el progreso de validacion oscila 0,30-0,49 entre evaluaciones de una misma
  corrida. Con 3 semillas la p por semilla no baja de 0,1: la separacion es criterio de
  consistencia, no prueba. Las semillas cambian el entrenamiento, no la mascara: hay una
  sola instancia de aleatorio y una de recableado (`artifacts/conectoma_500_proto/`), asi
  que el "pierde" es contra un sorteo de Erdos-Renyi, no contra la familia. El piloto de
  una semilla se reproduce exacto (mosca s0 afinada 0,431).
- **Lo que no se puede decir:** que el conectoma sea peor en general. Es un fragmento de
  500 neuronas elegido por grado, con los pesos entrenables, en una tarea ajena a la
  mosca. Para el signo hay hipotesis y ninguna medida: los centros de la cola de grados,
  que el aleatorio no tiene, y la reciprocidad (fraccion de aristas con su reciproca:
  0,346 en la mosca, 0,092 en el recableado, 0,035 en el aleatorio, medido el
  20-10-2026). El orden de las medias (mosca 0,462 < recableado 0,527 < aleatorio 0,568)
  encaja con las dos, pero recableado contra aleatorio no estaba en el plan y no se
  contrasta aqui.

## 21-10-2026: el reescalado espectral apaga la recurrencia de la mosca (diagnostico, exploratorio)

`diagnostico_cableado.py`, sin entrenar nada, sobre las mascaras de la 6d (N = 500) y las
del subgrafo de 4.000, con la inicializacion exacta del nucleo (`max_eig`, radio 0,95) y
los puertos exactos de `select_ports`. Datos en
`artifacts/mario_proto_6c_soltar_gcp/diagnostico_cableado.json` y
`artifacts/conectoma_4000/diagnostico_cableado.json`.

| | mosca | recableado | aleatorio |
|---|---|---|---|
| radio espectral bruto (conteos con signo, media 1), N = 500 | 74,5 | 16,4 | 10,4 |
| grueso del espectro tras reescalar (mediana de \|autovalor\|), N = 500 | **0,013** | 0,102 | 0,356 |
| autovalores con \|autovalor\| > 0,1, de 500 | 20 | 250 | 460 |
| peso medio por arista, N = 500 | 0,013 | 0,058 | 0,091 |
| grueso del espectro, N = 4.000 | **0,004** | 0,075 | 0,284 |
| reciprocidad, N = 500 | 0,346 | 0,092 | 0,035 |
| conteo medio en aristas reciprocas / en el resto, N = 500 | 5,97 / 3,32 | 5,47 / 4,11 | 3,28 / 4,27 |
| cuota de \|W\| que sale del 5 % de mas grado, N = 500 | 0,23 | 0,19 | 0,07 |
| progreso en la 6d (afinados, codiciosa) | 0,462 | 0,527 | 0,568 |

- **Lo que pasa:** la mosca tiene pares reciprocos con muchas sinapsis entre sus centros, y
  eso le da un autovalor dominante de 74,5. Al llevarlo a 0,95, todo lo demas se divide
  por 78: una senal generica se atenua unas 75 veces por salto. En el aleatorio, por 11.
  Es el mismo fenomeno que la fase 4 midio solo en la mosca; lo nuevo es la comparacion:
  **igualar el radio espectral no iguala la ganancia efectiva**, y castiga a la mosca 27
  veces mas que al aleatorio a N = 500 y 73 veces mas a N = 4.000. Afecta a todas las
  fases, porque todas usan `max_eig`.
- El orden del grueso del espectro (mosca < recableado < aleatorio) es el orden del
  progreso en la 6d, a N = 500 y a N = 4.000. Son tres puntos: encaja, no prueba.
- **El entrenamiento no lo corrige.** Tras clonado + PPO, el grueso de la mosca sigue en
  0,013-0,014 y el del aleatorio en 0,37; los pesos de la mosca se mueven un 20 % (los del
  aleatorio un 7 %), sin salir del regimen.
- **En la 6d la recurrencia si sirve cuando esta viva** (exploratorio, la misma prueba por
  nivel sobre los datos de la 6d): aleatorio contra `sin_nucleo` +0,091 (p < 1e-4),
  recableado contra `sin_nucleo` +0,049 (p < 1e-4); en la muestreada +0,060 y +0,074. El
  denso, +0,019 (p 0,22). La lectura de la 6c ("la tarea apenas usa la recurrencia") no
  vale aqui: el que no la usa es la mosca, y empata con no tener nucleo.
- **H3, atajos entre puertos:** en los tres cableados todas las motoras exclusivas estan a
  un salto de alguna sensorial. Aristas directas sensorial -> motora: 961 en la mosca, 862
  en el recableado y 236 en el aleatorio. La mosca tiene cuatro veces mas atajos y pierde,
  asi que los atajos no son lo que la frena.
- **H2:** la composicion de signos de los pares reciprocos es parecida (EE 0,20 / 0,25 /
  0,25; EI 0,56 / 0,48 / 0,56). No apunta a nada.
- **H4:** entropia de los valores singulares 0,78 / 0,83 / 0,88, el mismo orden, pero es
  consecuencia del mismo autovalor dominante, no una causa aparte.
- **Discrepancia con la fase 4:** alli se anoto un grueso de 0,188 (N = 500) y 0,071
  (N = 4.000) para la mosca; con esta medida sale 0,013 y 0,004. No consta como se midio
  entonces. Las cifras de esta tabla salen de `MaskedRecurrentCore.dense_matrix()` y se
  reproducen con el script. La normalizacion por filas se probo en la fase 4, solo en la
  mosca y en el prototipo de fluidos, donde ningun modelo aprendia ("no cambia el
  resultado final", 0,0998 contra 0,0999): aquel ensayo no podia ver este efecto.

## 21-10-2026: fase 6e, plan registrado antes de los datos (EXPLORATORIA)

Fuera del diseno: la normalizacion del diseno es `max_eig`. **No sustituye al veredicto
registrado de la 6d**, que sigue siendo "pierde".

- **Que:** la 6d repetida entera (`configs/mario_proto_6e_row.yaml`, `scripts/gcp/run_6e.sh`:
  clonado de la heuristica con ruido 0,1, 300 iteraciones de PPO, semillas 0-2, los mismos
  200 niveles de test, codiciosa y muestreada con 3 repeticiones) cambiando solo
  `core.normalisation` a `row`: cada neurona recibe la misma corriente total antes de
  llevar el autovalor mayor a 0,95. Solo mosca, aleatorio y recableado: `sin_nucleo` y el
  MLP no tienen aristas, y en el denso todas las neuronas reciben lo mismo, asi que `row`
  le da los mismos pesos. Esos tres se toman de la 6d.
- Con `row` el grueso del espectro queda en 0,146 / 0,498 / 0,595 (mosca / recableado /
  aleatorio): la distancia baja de 27 veces a 4, pero no desaparece.
- **Analisis:** `analiza_6e.py`, con la regla en su cabecera y tests en
  `tests/test_analiza_6e.py`. Principal, con Holm sobre los dos: la interaccion por nivel
  d = (mosca_row - X_row) - (mosca_max - X_max), X = aleatorio y recableado, permutacion
  pareada sobre los 200 niveles, con las tres semillas del mismo lado como criterio de
  consistencia. Secundarios con la regla de la 6d.
- **Prediccion de H1 (el deficit viene del reescalado):** la interaccion "reduce el
  deficit" contra los dos controles, la mosca con `row` le gana a la mosca con `max_eig`, y
  la mosca con `row` le gana a `sin_nucleo`. H1 no predice que la mosca alcance al
  aleatorio, porque el grueso sigue siendo cuatro veces menor.
- **Lectura:** si reduce el deficit, al menos parte del "pierde" de la 6d (y del empate con
  `sin_nucleo` de todas las fases) es un efecto del diseno de la inicializacion, no del
  cableado, y la comparacion justa necesita igualar la ganancia efectiva. Si sale "sin
  efecto" contra los dos, el reescalado queda descartado como causa y quedan H5 (donde caen
  los conteos) y H6 (fragmento de 500 neuronas).
- **Coste:** 9 entrenamientos y 3 evaluaciones en `mosca-6c`, alrededor de una hora.

## 21-10-2026: veredicto de la fase 6e (EXPLORATORIA, leido con el analisis registrado)

La tanda de `mosca-6c` termino el 21-10 a las 11:55 UTC con codigo 0: 9 clonados, 9
afinados y las tres evaluaciones. El codigo (`codigo_a7db8eb.tar.gz` en el bucket, nombre
anterior a la reescritura) es identico a c7d16de en todo el Python y los YAML; solo
cambian documentos y comentarios. `analiza_6e.py` paso la
comprobacion previa (los mismos 200 niveles y las mismas referencias en la 6e y en la
6d). Artefactos en `artifacts/mario_proto_6e_row_gcp/` (con `logs_gcp/` y
`analisis_6e.{json,txt}`).

Principal: afinados, codiciosa, 200 niveles de test.

| fila | progreso | metas | por semilla |
|---|---|---|---|
| mosca, `row` | 0,503 | 14,7 % | 0,511 0,493 0,505 |
| aleatorio, `row` | 0,653 | 33,8 % | 0,739 0,578 0,641 |
| recableado, `row` | 0,432 | 10,2 % | 0,446 0,530 0,319 |
| mosca, `max_eig` (6d) | 0,462 | 10,8 % | 0,431 0,482 0,474 |
| aleatorio, `max_eig` (6d) | 0,568 | 21,7 % | 0,598 0,574 0,534 |
| recableado, `max_eig` (6d) | 0,527 | 19,0 % | 0,633 0,447 0,501 |
| sin_nucleo (6d) | 0,477 | 10,8 % | 0,513 0,474 0,444 |

| interaccion d = (mosca_row - X_row) - (mosca_max - X_max) | deficit `max_eig` | deficit `row` | d | p por nivel (Holm) | D_s por semilla | veredicto |
|---|---|---|---|---|---|---|
| X = aleatorio | -0,106 | -0,150 | -0,044 | 0,008 (0,05) | -0,062 +0,007 -0,077 | **sin efecto** |
| X = recableado | -0,064 | +0,071 | +0,135 | < 1e-4 (0,025) | +0,266 -0,073 +0,212 | **sin efecto** |

- **Veredicto registrado: sin efecto contra los dos controles.** Por la regla escrita
  antes de los datos, **el reescalado espectral queda descartado como causa** del deficit
  de la mosca en la 6d. La prediccion principal de H1 no se cumple.
- **Que quiere decir "sin efecto" aqui.** La regla lo da cuando la media no supera la
  desviacion entre semillas, y eso es lo que pasa, pero no con una interaccion nula:
  contra el aleatorio la interaccion va **en contra** de H1 (el deficit crece de -0,106 a
  -0,150, dos semillas de tres en ese sentido); contra el recableado va a favor
  (+0,135), pero una semilla de tres dice lo contrario y la media la arrastran el
  recableado `row` s2 (0,319) y s0 (0,446). Ninguna de las dos lecturas apoya que igualar
  la corriente de entrada saque a la mosca del sotano.
- **`row` le sienta bien a la mosca y mejor al aleatorio.** Mosca `row` contra mosca
  `max_eig`: **gana** (+0,040, las tres semillas por encima). Aleatorio `row` contra
  aleatorio `max_eig`: +0,084, no resuelto (semillas solapadas). Recableado: -0,095,
  empate. La mosca con `row` sigue **perdiendo** contra el aleatorio con `row` (-0,150,
  39 de 200 niveles, semillas separadas) y contra el recableado con `row` sale no
  resuelto (+0,071, semillas solapadas).
- **Contra `sin_nucleo`, todavia sin despegarse.** Codiciosa +0,026 (p 0,058, no
  resuelto); en la muestreada +0,045, **gana**. Las otras dos predicciones de H1 salen
  una si (mosca_row > mosca_max) y otra a medias.
- **Muestreada (secundaria):** mismo dibujo. Interaccion contra aleatorio -0,018 (no
  resuelto), contra recableado +0,046 (sin efecto). Mosca `row` contra aleatorio `row`
  **pierde** (-0,048, semillas separadas); contra recableado `row`, empate (+0,002).
- **Solo clonados (secundario):** aqui si sale "reduce el deficit" contra el aleatorio
  (+0,114, las tres semillas a favor): recien clonada, la mosca con `row` empata con el
  aleatorio con `row` (0,352 contra 0,314) y le gana a `sin_nucleo` (+0,130, semillas
  separadas). Pero parte de eso es que el aleatorio clonado empeora con `row` (0,375 ->
  0,314), y PPO lo deshace: tras afinar, el aleatorio `row` sube +0,339 y la mosca
  `row` +0,151.
- **Lo que queda en pie del diagnostico.** El hecho medido sigue siendo cierto: con
  `max_eig` la recurrencia de la mosca queda casi apagada (grueso del espectro 0,013
  contra 0,356). Lo que la 6e dice es que **no es eso lo que la hace perder**: con la
  recurrencia mucho mas viva (0,146) el deficit contra el aleatorio no baja. La frase del
  README "hemos encontrado un mecanismo que lo explicaria" no se sostiene y se retira.
  Quedan H5 (donde caen los conteos de sinapsis) y H6 (fragmento de 500 neuronas elegido
  por grado), sin medir.
- **Limites:** los mismos de la 6d (N = 500, una instancia de cada control, 3 semillas,
  foto de la ultima iteracion), y uno mas: la mitad `max_eig` de cada interaccion sale de
  otra tanda (la 6d), con el mismo codigo salvo la normalizacion y los mismos niveles.
  La varianza entre semillas de los controles con `row` (recableado 0,319-0,530) es mayor
  que la de la 6d, y con tres semillas no se puede saber si es real.

## 21-10-2026: fases 4 y 5 a escala real, los 18 entrenamientos (RTX 3060 Ti)

Es el experimento original del proyecto: N = 4.000, 128 modos, malla 64x64, 20.000 pasos
de fase A y curriculo 5/10/20 en la fase B, seis modelos x semillas 0-2 sobre un unico
subgrafo. Habia 15 de 18 hechos en el PC (22-09, 23-09 y 28-09); los tres que faltaban
(`denso s0`, `fno s1`, `fno s2`) se reanudaron hoy de su instantanea (46, 37 y 18 min).
Los 18 informes llevan la misma config, y el codigo de fluidos no cambia desde el 22-09 a
las 09:40, antes del primer entrenamiento (y es identico al de HEAD). `evaluate.py` sobre
las 40 trayectorias de test, despliegue de 50 pasos. Informes y figuras en
`artifacts/train_64_pc/` (en el PC, `artifacts/train_64/`; el `eval_report.json` del 22-09,
con tres modelos, queda como `eval_report.antes_21oct.json`).

| modelo | L2 1 paso | L2 despliegue (50) | t hasta corr < 0,8 | E(k) inyeccion | E(k) alcanzable | E(k) fuera |
|---|---|---|---|---|---|---|
| mosca | 0,1003 ± 0,0001 | 0,7871 ± 0,0030 | 1,161 ± 0,015 | 0,088 | 0,161 | 0,040 |
| aleatorio | 0,1001 ± 0,0001 | 0,7701 ± 0,0045 | 1,198 ± 0,013 | 0,094 | 0,219 | 0,040 |
| recableado | 0,0999 ± 0,0000 | 0,7663 ± 0,0028 | 1,221 ± 0,010 | 0,092 | 0,180 | 0,040 |
| denso | 0,1008 ± 0,0001 | 0,7867 ± 0,0023 | 1,148 ± 0,006 | 0,077 | 0,246 | 0,040 |
| sin_nucleo | 0,1018 ± 0,0000 | 0,8216 ± 0,0006 | 1,074 ± 0,003 | 0,096 | 0,146 | 0,040 |
| fno | 0,0363 ± 0,0041 | 0,1124 ± 0,0029 | >= 5,0 (censurado) | 0,003 | 0,006 | 0,068 |
| persistencia | 0,1010 | 0,8866 | 1,08 | 0,028 | 0,019 | 0,040 |

(media ± desviacion entre las tres semillas de entrenamiento; E(k) es |log10(E_pred/E_real)|
medio por banda: inyeccion k = 1-4, alcanzable 5-9, fuera del codificador 10-21.)

Lectura en el orden fijado en TRASPASO.md (seccion 5, "Como se lee el resultado"):

1. **Contra la persistencia: el nucleo se despega.** Mosca -0,0995 en despliegue (11 %
   menos de error), con sus tres semillas entre 0,784 y 0,789 contra 0,887. A un paso,
   -0,0007. `evaluate.py` lo marca "no juzgable" solo porque la persistencia no tiene
   semillas. El experimento **tiene resolucion**, que en el prototipo de 32x32 no tenia.
2. **Contra `sin_nucleo`: la mosca gana.** -0,0345 en despliegue (4 %), -0,0015 a un paso,
   +0,087 de tiempo de correlacion; p < 1e-4 por trayectoria y por encima de la
   variabilidad entre semillas en las tres metricas. El cableado de la mosca hace algo.
3. **Mosca contra aleatorio: pierde.** +0,0170 de error en despliegue (2,2 % peor), +0,0002
   a un paso, -0,0375 de tiempo de correlacion; p < 1e-4 por trayectoria y por encima de la
   variabilidad entre semillas en las tres. Sus tres semillas quedan peor que las tres del
   aleatorio en despliegue (0,7836-0,7893 contra 0,7659-0,7749) y en tiempo de correlacion.
   **Contra recableado, igual:** +0,0208 en despliegue (2,7 % peor), +0,0004 a un paso,
   -0,060 de tiempo de correlacion, las mismas p y las semillas separadas. **Contra el
   denso, empate** en despliegue (+0,0004, p 0,87) y en tiempo de correlacion (dentro de la
   variabilidad entre semillas); a un paso la mosca va 0,0005 mejor.
4. **Espectro (descriptivo, `evaluate.py` no lo contrasta):** en la banda que el
   codificador alcanza, todos los nucleos deforman E(k) mucho mas que la persistencia
   (0,15-0,25 contra 0,02). La mosca la deforma **menos** que el aleatorio (0,161 contra
   0,219) y el recableado (0,180), y algo mas que `sin_nucleo` (0,146). La banda de fuera
   del codificador sale igual en los cinco nucleos (0,040, la de la persistencia), como
   tenia que salir por construccion.

- **Prediccion registrada (H0/H1: mosca empata con aleatorio y recableado; el FNO gana a
  todos).** La mitad del FNO se cumple de sobra: 7 veces menos error en despliegue que
  cualquier nucleo, y no se descorrelaciona en los 50 pasos. La otra mitad **no**: la
  mosca no empata, queda por debajo de los dos controles, con poco margen (2-3 %) y con
  las tres semillas del mismo lado. H2 (ventaja por estructura especifica) queda
  descartada a esta escala; H1 no aplica porque no hay ventaja que explicar.
- **El mismo signo que la 6d, en otra tarea, a otra escala y con supervision directa.** En
  plataformas (N = 500, clonado + PPO) la mosca perdia contra el aleatorio y quedaba en
  "no resuelto" contra el recableado; aqui pierde contra los dos. Y en las dos tareas la
  mosca queda entre `sin_nucleo` y los controles: en despliegue sin_nucleo 0,822 > mosca
  0,787 > aleatorio 0,770 > recableado 0,766. Encaja con una recurrencia mas debil (con
  `max_eig`, grueso del espectro 0,004 en la mosca contra 0,284 en el aleatorio a
  N = 4.000), pero la 6e descarto el reescalado como causa en plataformas, y aqui no se ha
  probado: es una correlacion, no una explicacion.
- **El margen es estrecho.** A un paso todos los nucleos estan en 0,0999-0,1008, contra
  0,1010 de la persistencia: como mucho un 1 % menos de error. Con el incremento perfecto
  en los 128 modos, la fase 4 midio que se podia bajar un 21 % (de 0,1054 a 0,0833, en
  aquel conjunto), asi que los nucleos recorren una vigesima parte de la ventana. Lo que
  separa a los modelos es el despliegue, y ahi las diferencias entre cableados son de
  0,02 sobre 0,77.
- **Limites:** una sola instancia de aleatorio y de recableado (las semillas son de
  entrenamiento, sobre un unico subgrafo compartido, como fija la fase 2), asi que
  "pierde" es contra un sorteo de Erdos-Renyi y un recableado concretos, no contra la
  familia. 40 trayectorias de test. Tres de los 18 entrenamientos se reanudaron de
  instantanea, que en `train.py` da los mismos pesos bit a bit (test que lo fija). El
  tiempo de correlacion del FNO esta censurado en 5,0.
