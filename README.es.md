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

## Los seis modelos

Entre los modelos cableados solo cambia la máscara de conectividad; codificador,
decodificador, receta de entrenamiento y presupuesto de parámetros son comunes.

| Modelo | Núcleo recurrente |
|---|---|
| `mosca` | Subgrafo de FlyWire, signos reales |
| `aleatorio` | Erdős–Rényi, misma densidad y fracción inhibidora |
| `recableado` | Grafo de la mosca con aristas barajadas, **grados conservados exactamente** |
| `denso` | Todas contra todas, menos neuronas, mismo número de pesos entrenables |
| `fno` | Fourier Neural Operator (referencia superior) |
| `persistencia` | Predice "no cambia nada" (referencia inferior) |

Recorrido: campo de vorticidad → FFT → modos de Fourier más bajos → neuronas *sensoriales*
(mayor grado de salida) → núcleo recurrente con máscara (4 subpasos internos) → neuronas
*motoras* (mayor grado de entrada) → modos → FFT inversa → **incremento** que se suma al campo.

## Estado

**En curso. Todavía no hay resultado final.** El experimento real (N = 4.000 neuronas, malla
64×64, 3 semillas por modelo) va en GPU. Lo que hay hasta ahora es un ensayo en portátil
(N = 500, 32×32). Todas las cifras, también las que salieron mal, están en
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
python train.py    --config configs/ensayo_32.yaml
python evaluate.py --config configs/ensayo_32.yaml
python visor.py    --config configs/ensayo_32.yaml --models mosca aleatorio   # HTML interactivo
python video.py    --config configs/ensayo_32.yaml --models mosca              # el vídeo de arriba
```

Todo se configura por YAML; las semillas y la configuración se escriben en cada salida.

## Datos y créditos

- Conectoma: FlyWire, materialización v783 — Dorkenwald et al., *Nature* 2024; Schlegel et
  al., *Nature* 2024. Conectividad y signos tal como los empaqueta Shiu et al., *Nature* 2024
  ([philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)).
- Posiciones de las neuronas: [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations).
- Los datos descargados no se redistribuyen en este repositorio.

Licencia MIT.
