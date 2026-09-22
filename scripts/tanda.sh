#!/usr/bin/env bash
# Lanza una tanda entera (todos los modelos x todas las semillas del YAML) de uno en uno,
# con un log por entrenamiento, y SALTA los que ya estan terminados. Sirve para dejarlo
# corriendo y volver, y para retomarlo despues de un corte o un reinicio sin repetir nada.
#
#   scripts/tanda.sh configs/entrenamiento_64.yaml            # fase 4, fluidos
#   scripts/tanda.sh configs/mario_nes.yaml     mario         # fases 6 y 6b
#
# "Terminado" = existe el informe JSON del trozo, que se escribe al acabar. El checkpoint
# no sirve de marca: en PPO se guarda en cada evaluacion, o sea tambien a medias.
#
# Si un entrenamiento falla, la tanda sigue con el siguiente y lo dice en el resumen: un
# modelo roto no debe llevarse por delante los otros cinco.
set -uo pipefail

config="${1:-}"
tarea="${2:-fluidos}"
if [ -z "$config" ] || [ ! -f "$config" ]; then
  echo "uso: scripts/tanda.sh <config.yaml> [fluidos|mario]" >&2
  exit 2
fi

case "$tarea" in
  fluidos) entrenador="train.py" ;;
  mario)   entrenador="train_rl.py" ;;
  *) echo "tarea desconocida: $tarea (fluidos|mario)" >&2; exit 2 ;;
esac

leer_yaml() {  # $1 = fichero YAML, $2 = expresion python sobre `cfg`
  python -c "import sys, yaml; cfg = yaml.safe_load(open(sys.argv[1])); print($2)" "$1"
}

salida=$(leer_yaml "$config" 'cfg["output_dir"]')
modelos=$(leer_yaml "$config" '" ".join(m for m in cfg["models"] if m != "persistencia")')
semillas=$(leer_yaml "$config" '" ".join(str(s) for s in cfg["seeds"])')
logs="$salida/logs"
mkdir -p "$logs"

echo "config    $config"
echo "salida    $salida"
echo "modelos   $modelos"
echo "semillas  $semillas"
echo

hechos=0; fallados=0; saltados=0
comenzada=$(date +%s)
for modelo in $modelos; do
  for semilla in $semillas; do
    informe="$salida/$( [ "$tarea" = fluidos ] && echo train_report || echo rl_report )_${modelo}_s${semilla}.json"
    if [ -f "$informe" ]; then
      echo "[saltado] $modelo s$semilla — ya tiene informe"
      saltados=$((saltados + 1))
      continue
    fi
    log="$logs/${modelo}_s${semilla}.log"
    echo "[corriendo] $modelo s$semilla — log en $log"
    inicio=$(date +%s)
    # -u: sin el, python bufferea stdout al redirigir a fichero y el log se queda
    # vacio horas, que es justo cuando uno quiere mirar como va.
    if python -u "$entrenador" --config "$config" --model "$modelo" --seed "$semilla" > "$log" 2>&1; then
      echo "[hecho]   $modelo s$semilla — $(( ($(date +%s) - inicio) / 60 )) min"
      hechos=$((hechos + 1))
    else
      echo "[FALLO]   $modelo s$semilla — ultimas lineas:"
      tail -5 "$log" | sed 's/^/           /'
      fallados=$((fallados + 1))
    fi
  done
done

echo
echo "resumen: $hechos hechos, $saltados saltados, $fallados fallados"
echo "tiempo total: $(( ($(date +%s) - comenzada) / 60 )) min"
[ "$fallados" -eq 0 ] || exit 1
