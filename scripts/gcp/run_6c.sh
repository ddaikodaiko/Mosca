#!/bin/bash
# Fase 6c en GCP: mosca y mlp seguidos, evaluacion, y la VM se apaga sola.
# Si algo falla tambien sube lo que haya y apaga, para no quemar creditos parada.
cd ~/Mosca
B=gs://mosca-runs-317660842762/6c
CFG=configs/mario_nes_sola.yaml
PY=".venv/bin/python -u"

sube() {
    gcloud storage rsync -r artifacts/mario_nes_sola "$B/artifacts/mario_nes_sola" --quiet
    gcloud storage cp ~/6c.log "$B/6c.log" --quiet
}
fin() {
    echo "== fin $(date -Is), codigo $1 =="
    sube
    sudo poweroff
}
trap 'fin $?' EXIT

( while sleep 900; do sube >/dev/null 2>&1; done ) &

set -e
echo "== inicio $(date -Is) en $(hostname) =="
$PY train_rl.py --config $CFG --model mosca --seed 0
$PY train_rl.py --config $CFG --model mlp --seed 0
$PY evaluate_rl.py --config $CFG --muestrea --repeticiones 6
