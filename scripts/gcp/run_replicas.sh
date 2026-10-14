#!/bin/bash
# Fase 6c, replicas: los seis modelos del diseno x 3 semillas con la receta de la s0.
# Cola compartida entre VMs en el bucket: cada VM reclama el siguiente trabajo libre
# creando claims/<modelo>_s<semilla> con --if-generation-match=0, que falla si otra VM
# ya lo creo (reclamo atomico). El orden de la cola es la prioridad: primero los
# controles de cableado en s0, luego las semillas 1 y 2, el denso al final.
# Codigo = 2772f6d (git archive), comprobado por sha256 contra el repo el 14-10-2026.
# Al vaciarse la cola sube todo y apaga la VM, tambien si algo falla.
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

( while sleep 900; do sube >/dev/null 2>&1; done ) &

echo "== inicio $(date -Is) en $VM ($(nproc) vCPU) =="
for trabajo in $COLA; do
    m=${trabajo%:*}; s=${trabajo#*:}; id="${m}_s${s}"
    reclama "$id" || continue
    echo "== $id empieza $(date -Is) =="
    $PY train_rl.py --config $CFG --model "$m" --seed "$s" > ~/logs/$id.log 2>&1
    codigo=$?
    echo "== $id termina $(date -Is), codigo $codigo =="
    echo "$VM $(date -Is) codigo $codigo" > /tmp/hecho
    gcloud storage cp /tmp/hecho "$B/hechos/$id" --quiet
    sube >/dev/null 2>&1
done
