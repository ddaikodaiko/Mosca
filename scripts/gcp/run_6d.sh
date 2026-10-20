#!/bin/bash
# Fase 6d (19-10-2026): clonado de la heuristica + PPO desde el clonado, seis modelos x
# tres semillas, entorno propio con release_to_jump (configs/mario_proto_6c_soltar.yaml).
# Plan de analisis en RESULTADOS.md, escrito antes de lanzar esto.
#
#   bash run_6d.sh <commit>
#
# Una sola VM (mosca-6c). El codigo es el `git archive` de <commit>, subido al bucket
# como codigo_<commit>.tar.gz; se usa el .venv y el cableado del prototipo que la VM ya
# tiene de la 6c. Un hilo por proceso y tantos procesos como vCPU. La VM se apaga sola.
set -u
COMMIT=$1
B=gs://mosca-runs-317660842762/6d
DIR=~/Mosca6d
LOG=~/6d.log
PY=".venv/bin/python -u"
CFG=configs/mario_proto_6c_soltar.yaml
D=artifacts/mario_proto_6c_soltar
MODELOS="mosca aleatorio recableado sin_nucleo denso mlp"
N=$(nproc)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec > >(tee -a "$LOG") 2>&1

sube() {
    gcloud storage rsync -r "$DIR/$D" "$B/artifacts" --quiet
    gcloud storage rsync -r ~/logs6d "$B/logs" --quiet
    gcloud storage cp "$LOG" "$B/logs/vm.log" --quiet
}
fin() {
    echo "== fin $(date -Is), codigo $1 =="
    sube
    sudo poweroff
}
trap 'fin $?' EXIT

echo "== 6d inicio $(date -Is) en $(hostname) ($N vCPU), commit $COMMIT =="
mkdir -p "$DIR" ~/logs6d && cd "$DIR" || exit 1
gcloud storage cp "$B/codigo_$COMMIT.tar.gz" /tmp/codigo.tar.gz --quiet || exit 1
tar xzf /tmp/codigo.tar.gz || exit 1
ln -sfn ~/Mosca/.venv .venv
mkdir -p artifacts && cp -r ~/Mosca/artifacts/conectoma_500_proto artifacts/ || exit 1
$PY -m pytest -q tests/test_platformer.py || exit 1

( while sleep 600; do sube >/dev/null 2>&1; done ) &

# 1. Clonado: un proceso por semilla, los seis modelos en cada uno.
# `wait` con los PID: sin ellos espera tambien al bucle de subida, que no acaba nunca
# (asi se quedo colgada la primera tanda el 19-10-2026, 22 h con la VM parada).
echo "== clonado $(date -Is) =="
CLONADOS=()
for s in 0 1 2; do
    $PY clonado_rl.py --config $CFG --modelos "${MODELOS// /,}" --ruido 0.1 --guarda-pesos \
        --semilla $s --salida $D/clonado_s$s.json > ~/logs6d/clonado_s$s.log 2>&1 &
    CLONADOS+=($!)
done
wait "${CLONADOS[@]}"
ls $D/clonado/*.pt | wc -l
sube >/dev/null 2>&1

# 2. PPO desde el clonado: 18 trabajos, $N a la vez. Salida en $D/desde_clonado/.
echo "== afinado $(date -Is) =="
for s in 0 1 2; do for m in $MODELOS; do echo "$m $s"; done; done |
    xargs -P "$N" -L 1 bash -c '.venv/bin/python -u train_rl.py --config '"$CFG"' --model $0 --seed $1 --desde '"$D"'/clonado/$0_s$1.pt > ~/logs6d/afinado_$0_s$1.log 2>&1; echo "   $0 s$1 codigo $?"'
ls $D/desde_clonado/*_s?.pt | wc -l
sube >/dev/null 2>&1

# 3. Evaluacion: 200 niveles de test, mismos niveles para todos.
echo "== evaluacion $(date -Is) =="
sed "s|^output_dir: .*|output_dir: $D/desde_clonado|" $CFG > /tmp/eval_afinado.yaml
sed "s|^output_dir: .*|output_dir: $D/clonado|" $CFG > /tmp/eval_clonado.yaml
COMUN="--modelos $MODELOS --semillas 0 1 2 --levels 200 --procesos $N"
$PY evaluate_rl.py --config /tmp/eval_afinado.yaml $COMUN --salida eval_afinado_codiciosa.json > ~/logs6d/eval_afinado_codiciosa.log 2>&1
$PY evaluate_rl.py --config /tmp/eval_clonado.yaml $COMUN --salida eval_clonado_codiciosa.json > ~/logs6d/eval_clonado_codiciosa.log 2>&1
sube >/dev/null 2>&1
$PY evaluate_rl.py --config /tmp/eval_afinado.yaml $COMUN --muestrea --repeticiones 3 --semilla 0 --salida eval_afinado_muestreada.json > ~/logs6d/eval_afinado_muestreada.log 2>&1
echo "== evaluacion termina $(date -Is) =="
