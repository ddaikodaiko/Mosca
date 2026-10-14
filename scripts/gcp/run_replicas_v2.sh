#!/bin/bash
# Fase 6c, replicas (v2). Igual que la v1 —cola compartida en el bucket, reclamo atomico
# con --if-generation-match=0, la VM se apaga sola— mas dos cosas:
#   - adopta el entrenamiento que dejo en marcha la v1:  run_replicas_v2.sh <pid> <id>
#     (espera a que acabe y lo da por bueno si escribio su rl_report).
#   - cuando no queda nada que reclamar y estan los 16 trabajos hechos, esta VM es la
#     ultima: reclama "evaluacion" y corre la evaluacion final antes de apagarse.
# Codigo de entrenamiento = 2772f6d (sha256 comprobado el 14-10-2026). evaluate_rl.py es
# la version con --split/--por-nivel/--modelos/--semillas.
cd ~/Mosca
B=gs://mosca-runs-317660842762/6c-replicas
CFG=configs/mario_nes_sola.yaml
PY=".venv/bin/python -u"
VM=$(hostname)
LOG=~/replicas.log
mkdir -p ~/logs
COLA="aleatorio:0 recableado:0 sin_nucleo:0
      mosca:1 mlp:1 aleatorio:1 recableado:1 sin_nucleo:1
      mosca:2 mlp:2 aleatorio:2 recableado:2 sin_nucleo:2
      denso:0 denso:1 denso:2"
N_TRABAJOS=$(echo $COLA | wc -w)

sube() {
    gcloud storage rsync -r artifacts/mario_nes_sola "$B/artifacts/mario_nes_sola" --quiet
    gcloud storage rsync -r ~/logs "$B/logs" --quiet
    gcloud storage cp "$LOG" "$B/logs/$VM.log" --quiet
}
fin() {
    echo "== fin $(date -Is) en $VM, codigo $1 =="
    sube
    sudo poweroff
}
trap 'fin $?' EXIT

reclama() {
    echo "$VM $(date -Is)" > /tmp/claim
    for intento in 1 2 3; do
        gcloud storage cp /tmp/claim "$B/claims/$1" --if-generation-match=0 --quiet 2>/dev/null && return 0
        gcloud storage ls "$B/claims/$1" >/dev/null 2>&1 && return 1   # es de otra VM
        sleep 10
    done
    return 1
}
hecho() {   # $1 = id, $2 = codigo
    echo "== $1 termina $(date -Is), codigo $2 =="
    echo "$VM $(date -Is) codigo $2" > /tmp/hecho
    gcloud storage cp /tmp/hecho "$B/hechos/$1" --quiet
    sube >/dev/null 2>&1
}

( while sleep 900; do sube >/dev/null 2>&1; done ) &

echo "== v2 inicio $(date -Is) en $VM ($(nproc) vCPU) =="
if [ -n "${1:-}" ]; then
    echo "== adopta $2 (pid $1) =="
    while kill -0 "$1" 2>/dev/null; do sleep 60; done
    if [ -f "artifacts/mario_nes_sola/rl_report_$2.json" ]; then hecho "$2" 0; else hecho "$2" 1; fi
fi

for trabajo in $COLA; do
    m=${trabajo%:*}; s=${trabajo#*:}; id="${m}_s${s}"
    reclama "$id" || continue
    echo "== $id empieza $(date -Is) =="
    $PY train_rl.py --config $CFG --model "$m" --seed "$s" > ~/logs/$id.log 2>&1
    hecho "$id" $?
done

hechos=$(gcloud storage ls "$B/hechos/" 2>/dev/null | wc -l)
echo "== cola vacia en $VM: $hechos de $N_TRABAJOS hechos =="
if [ "$hechos" -ge "$N_TRABAJOS" ] && reclama evaluacion; then
    echo "== evaluacion final empieza $(date -Is) =="
    gcloud storage cp "$B/artifacts/mario_nes_sola/*.pt" artifacts/mario_nes_sola/ --quiet
    cp ~/s0_original/mosca_s0.pt ~/s0_original/mlp_s0.pt artifacts/mario_nes_sola/
    ls artifacts/mario_nes_sola/*.pt
    COMUN="--config $CFG --modelos mosca aleatorio recableado sin_nucleo denso mlp --semillas 0 1 2 --split val+test --por-nivel 8"
    $PY evaluate_rl.py $COMUN --muestrea --repeticiones 6 --salida rl_evaluacion_final_muestreada.json > ~/logs/evaluacion_muestreada.log 2>&1 &
    muestreada=$!
    $PY evaluate_rl.py $COMUN --salida rl_evaluacion_final_codiciosa.json > ~/logs/evaluacion_codiciosa.log 2>&1 &
    codiciosa=$!
    # Solo esas dos: un `wait` a secas esperaria tambien al bucle de subidas, que no acaba.
    wait $muestreada $codiciosa
    echo "== evaluacion final termina $(date -Is) =="
    echo "$VM $(date -Is)" > /tmp/hecho; gcloud storage cp /tmp/hecho "$B/hechos_evaluacion" --quiet
fi
