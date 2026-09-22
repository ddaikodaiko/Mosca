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
