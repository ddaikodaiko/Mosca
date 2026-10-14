#!/bin/bash
# Cambia el lanzador v1 por el v2 sin tocar el entrenamiento en marcha.
set -u
PYPID=$(pgrep -f "train_rl.py --config" | head -1)
ID=$(ps -o args= -p "$PYPID" | sed -E 's/.*--model ([a-z_]+) --seed ([0-9]+).*/\1_s\2/')
V1=$(pgrep -f "run_replicas[.]sh" | tr '\n' ' ')
echo "python $PYPID ($ID); lanzador v1: $V1"
[ -n "$PYPID" ] && [ -n "$ID" ] && [ -n "$V1" ] || { echo "falta algo, no toco nada"; exit 1; }
kill -9 $V1
sleep 2
kill -0 "$PYPID" && echo "python sigue vivo" || { echo "PYTHON MUERTO"; exit 1; }
chmod +x ~/run_replicas_v2.sh
(setsid nohup ~/run_replicas_v2.sh "$PYPID" "$ID" >> ~/replicas.log 2>&1 < /dev/null &)
sleep 5
tail -3 ~/replicas.log
pgrep -af "run_replicas|train_rl"
