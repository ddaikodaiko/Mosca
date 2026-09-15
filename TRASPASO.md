# Traspaso al PC con GPU

Lo que no cabe en el portatil y hay que ejecutar en la maquina con GPU. El codigo
detecta el dispositivo solo (`device: auto` en todos los YAML); no hay nada que tocar.

## 1. Preparar

```bash
git clone <este repo> && cd Mosca
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch con CUDA segun la maquina
pytest -q                                # 58 tests, todos deben pasar
```

## 2. Datos que hay que regenerar alli (no se copian)

El set real de 64x64 son ~900 MB: se regenera, no se transfiere. Es **bit a bit
reproducible con el mismo `batch_size`**, que forma parte del contrato y se registra en
`meta.json` (ver la nota de la fase 1 en RESULTADOS.md).

```bash
python -m data.simulate_ns --config configs/ns_64.yaml     # ~13 min de CPU; menos en GPU
```

Semillas: train 1000-1199, val 2000-2039, test 3000-3039. Disjuntas por construccion.

## 3. Lo que SI hay que copiar (menos de 3 MB)

```
artifacts/conectoma_4000/mosca.npz
artifacts/conectoma_4000/aleatorio.npz
artifacts/conectoma_4000/recableado.npz
artifacts/conectoma_4000/stats.json
artifacts/conectoma_4000/controls_stats.json
```

Se pueden regenerar con los dos comandos de abajo (la semilla esta en el YAML, asi que
sale el mismo subgrafo), pero copiarlos evita volver a bajar los 104 MB del conectoma:

```bash
python -m data.download_flywire
python -m connectome.subgraph --config configs/conectoma_4000.yaml
python -m connectome.controls --config configs/conectoma_4000.yaml
```

## 4. Entrenamiento y evaluacion

```bash
python train.py    --config configs/entrenamiento_64.yaml
python evaluate.py --config configs/entrenamiento_64.yaml
```

Son **6 modelos x 3 semillas** (persistencia no se entrena): 15 entrenamientos. Se puede
trocear por modelo y por semilla, que los checkpoints son independientes:

```bash
python train.py --config configs/entrenamiento_64.yaml --model mosca --seed 0
```

## 5. Antes de lanzar, dos decisiones abiertas

1. **`augment.enabled: true`** en el YAML. Es aumento de datos por simetrias exactas del
   sistema (traslacion en x, y en y por periodos del forzado). Sin el, ningun modelo con
   mascara supera a la persistencia. Es una **desviacion de la receta del diseno** y
   sigue pendiente del visto bueno del usuario. Se apaga poniendolo a `false`.

2. **Que hacer si el nucleo con mascara no se despega de la persistencia.** En el
   prototipo (32x32, N=500) mosca dio 0,0999 y aleatorio 0,0998, con la persistencia en
   0,1001: empatan porque ninguno aprende. Si eso se repite a escala real, el experimento
   no tiene resolucion para responder a la pregunta del proyecto y **el empate no se debe
   publicar como H0 confirmada**. Ver la seccion de la fase 4 en RESULTADOS.md.

## 6. Que traer de vuelta

```
artifacts/train_64/train_report.json
artifacts/train_64/eval_report.json
artifacts/train_64/figs/*.png
artifacts/train_64/checkpoints/*.pt      (opcional, pesan)
```
