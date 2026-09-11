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
