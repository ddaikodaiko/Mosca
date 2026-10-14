#!/bin/bash
# Cambia de lanzador sin tocar el entrenamiento en marcha:
#   cambia_lanzador.sh <patron del viejo, p. ej. run_replicas_v2[.]sh> <script nuevo>
# El patron va con [.] para que pgrep no coincida con la propia linea de este comando.
set -u
VIEJO=$1
NUEVO=$2
PYPID=$(pgrep -f "train_rl.py --config" | head -1)
ID=$(ps -o args= -p "$PYPID" | sed -E 's/.*--model ([a-z_]+) --seed ([0-9]+).*/\1_s\2/')
V1=$(pgrep -f "$VIEJO" | tr '\n' ' ')
echo "python $PYPID ($ID); lanzador viejo: $V1"
[ -n "$PYPID" ] && [ -n "$ID" ] && [ -n "$V1" ] || { echo "falta algo, no toco nada"; exit 1; }
kill -9 $V1
sleep 2
kill -0 "$PYPID" && echo "python sigue vivo" || { echo "PYTHON MUERTO"; exit 1; }
chmod +x "$NUEVO"
(setsid nohup "$NUEVO" "$PYPID" "$ID" >> ~/replicas.log 2>&1 < /dev/null &)
sleep 5
tail -3 ~/replicas.log
pgrep -af "run_replicas|train_rl"
