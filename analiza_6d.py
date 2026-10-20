"""Analisis final de la fase 6d. Escrito el 20-10-2026, ANTES de tener ningun dato de evaluacion.

Aplica el plan registrado en RESULTADOS.md (seccion del 19-10-2026 y su enmienda del
20-10-2026) a los tres informes de `evaluate_rl.py` que deja `scripts/gcp/run_6d.sh`:

    python analiza_6d.py --dir artifacts/mario_proto_6c_soltar_gcp/desde_clonado \\
                         --clonado artifacts/mario_proto_6c_soltar_gcp/clonado \\
                         --salida artifacts/mario_proto_6c_soltar_gcp/analisis_6d.json

  eval_afinado_codiciosa.json   decide (clonado + PPO, politica codiciosa, 200 niveles)
  eval_afinado_muestreada.json  secundario (3 repeticiones por nivel)
  eval_clonado_codiciosa.json   secundario (solo clonado: separa "imita" de "aprende")

Regla, la de la 6c adaptada a 200 niveles independientes:
  - Diferencia = media sobre niveles de (media de semillas de A - media de semillas de B).
  - Prueba por nivel: permutacion pareada de signos sobre los 200 niveles, con las
    permutaciones del YAML (10.000) y semilla 0. Es `evaluate.prueba_de_permutacion`, la
    misma que da `contrastes` en el informe, y se comprueba que las p coinciden.
  - "Gana" o "pierde" solo si p < umbral (Holm sobre mosca-aleatorio y mosca-recableado en
    el informe que decide; 0,05 en todo lo demas) y las tres semillas de A quedan todas a
    un lado de las de B. "Empate" solo si |diferencia| < dispersion entre semillas. Si
    no, "no resuelto".
  - Ganancia por PPO: la misma prueba, afinado contra clonado del mismo modelo, 0,05.
  - "Juega": meta en algun nivel de test con alguna semilla, y "gana" contra el azar y
    contra la mejor constante (la de mas progreso medio entre siempre_dcha, salto_cte y
    salto_corriendo en el mismo informe).

Antes de nada comprueba que los tres informes tienen los seis modelos con las semillas
0, 1 y 2 y los mismos 200 niveles. Si no, para: el lanzador de esta tanda no comprobaba
los codigos de salida y `evaluate_rl.py` se salta en silencio los checkpoints que faltan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analiza_6c import (
    MODELOS,
    PRINCIPALES,
    SECUNDARIOS_MOSCA,
    Fila,
    decide,
    dispersion_entre_semillas,
    filas_de_informe,
    tabla,
    umbrales_holm,
)
from evaluate import prueba_de_permutacion
from evaluate_rl import permutacion_exacta_entre_grupos

SEMILLAS = (0, 1, 2)
N_NIVELES = 200
CONSTANTES = ("siempre_dcha", "salto_cte", "salto_corriendo")
INFORMES = {
    "afinado": ("eval_afinado_codiciosa.json", False),
    "muestreada": ("eval_afinado_muestreada.json", True),
    "clonado": ("eval_clonado_codiciosa.json", False),
}


def comprueba(informes: dict[str, dict[str, Any]]) -> None:
    """Para si falta una semilla, un modelo o un nivel, o si los niveles no son los mismos."""
    partidas = None
    for nombre, informe in informes.items():
        muestreada = INFORMES[nombre][1]
        if bool(informe["muestreada"]) != muestreada:
            raise SystemExit(f"{nombre}: muestreada = {informe['muestreada']}, se esperaba {muestreada}")
        for modelo in MODELOS:
            esperados = [f"{modelo}_s{s}" for s in SEMILLAS]
            tiene = [s["checkpoint"] for s in informe["series_por_semilla"].get(modelo, [])]
            if tiene != esperados:
                raise SystemExit(f"{nombre}: {modelo} tiene {tiene}, se esperaban {esperados}")
        niveles = [p["nivel"] for p in informe["partidas"]]
        if len(niveles) != N_NIVELES or len(set(niveles)) != N_NIVELES:
            raise SystemExit(f"{nombre}: {len(niveles)} partidas y {len(set(niveles))} niveles,"
                             f" se esperaban {N_NIVELES} distintos")
        if partidas is None:
            partidas = informe["partidas"]
        elif informe["partidas"] != partidas:
            raise SystemExit(f"{nombre}: no juega los mismos niveles que los demas informes")


def niveles_en_orden(informe: dict[str, Any]) -> list[str]:
    """En el orden de las partidas, que es el que usa `evaluate_rl.py`: la permutacion de
    Monte Carlo asigna los signos por posicion y otro orden daria otra p."""
    return [p["nivel"] for p in informe["partidas"]]


def contraste(a: Fila, b: Fila, niveles: list[str], permutaciones: int) -> dict[str, Any]:
    columnas_a, columnas_b = a.columnas(niveles), b.columnas(niveles)
    media_a, media_b = columnas_a.mean(axis=0), columnas_b.mean(axis=0)
    por_nivel = media_a - media_b
    prueba = prueba_de_permutacion(media_a, media_b, permutaciones, semilla=0)
    medias_a, medias_b = columnas_a.mean(axis=1), columnas_b.mean(axis=1)
    resultado: dict[str, Any] = {
        "a": a.nombre, "b": b.nombre, "niveles": niveles,
        "diferencia": float(por_nivel.mean()),
        "niveles_a_favor": int((por_nivel > 0).sum()),
        "niveles_en_contra": int((por_nivel < 0).sum()),
        "p_nivel": prueba["p"], "n_permutaciones": permutaciones,
        "semillas_a": medias_a.tolist(), "semillas_b": medias_b.tolist(),
        "separadas": bool(medias_a.min() > medias_b.max() or medias_a.max() < medias_b.min()),
        "dispersion": dispersion_entre_semillas(
            Fila(a.nombre, niveles, columnas_a, a.checkpoints),
            Fila(b.nombre, niveles, columnas_b, b.checkpoints)),
    }
    if medias_a.size >= 2 and medias_b.size >= 2:
        resultado["p_semilla"] = permutacion_exacta_entre_grupos(medias_a, medias_b)["p"]
    return resultado


def coincide_con_el_informe(contrastes: list[dict[str, Any]], informe: dict[str, Any]) -> int:
    """Las p de aqui tienen que ser las de `contrastes` del informe, donde las haya."""
    revisados = 0
    for c in contrastes:
        suyo = informe["contrastes"].get(f"{c['a']}_vs_{c['b']}")
        if suyo is None:
            continue
        if suyo["progreso"]["p"] != c["p_nivel"] or not np.isclose(
                suyo["progreso"]["diferencia_media"], c["diferencia"], rtol=0, atol=1e-12):
            raise SystemExit(f"{c['a']} vs {c['b']}: p {c['p_nivel']} / dif {c['diferencia']}"
                             f" aqui, p {suyo['progreso']['p']} / dif"
                             f" {suyo['progreso']['diferencia_media']} en el informe")
        revisados += 1
    return revisados


def mejor_constante(filas: dict[str, Fila]) -> str:
    return max((c for c in CONSTANTES if c in filas), key=lambda c: float(filas[c].valores.mean()))


def bateria(filas: dict[str, Fila], niveles: list[str], permutaciones: int,
            con_holm: bool) -> dict[str, Any]:
    """Los contrastes registrados sobre un informe."""
    principales = [contraste(filas[a], filas[b], niveles, permutaciones) for a, b in PRINCIPALES]
    umbrales = (umbrales_holm([c["p_nivel"] for c in principales]) if con_holm
                else [0.05] * len(principales))
    mejor = mejor_constante(filas)
    referencias = ["azar", "salto_corriendo"] + ([mejor] if mejor != "salto_corriendo" else [])
    parejas = list(SECUNDARIOS_MOSCA) + [(m, r) for m in MODELOS for r in referencias]
    secundarios = [contraste(filas[a], filas[b], niveles, permutaciones) for a, b in parejas]
    return {"principales": decide(principales, umbrales),
            "secundarios": decide(secundarios, [0.05] * len(secundarios)),
            "mejor_constante": mejor}


def ganancia_por_ppo(afinado: dict[str, Fila], clonado: dict[str, Fila], niveles: list[str],
                     permutaciones: int) -> list[dict[str, Any]]:
    contrastes = []
    for m in MODELOS:
        if afinado[m].checkpoints != clonado[m].checkpoints:
            raise SystemExit(f"{m}: semillas distintas en afinado y clonado")
        a = Fila(f"{m}_afinado", afinado[m].niveles, afinado[m].valores, afinado[m].checkpoints)
        b = Fila(f"{m}_clonado", clonado[m].niveles, clonado[m].valores, clonado[m].checkpoints)
        contrastes.append(contraste(a, b, niveles, permutaciones))
    return decide(contrastes, [0.05] * len(contrastes))


def juega(metas: dict[str, Fila], resultado: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Meta en algun nivel con alguna semilla, y gana al azar y a la mejor constante."""
    veredictos = {(c["a"], c["b"]): c["veredicto"] for c in resultado["secundarios"]}
    mejor = resultado["mejor_constante"]
    filas = {}
    for m in MODELOS:
        con_meta = int((metas[m].valores.max(axis=0) > 0).sum())
        al_azar, a_la_constante = veredictos[(m, "azar")], veredictos[(m, mejor)]
        filas[m] = {"niveles_con_meta": con_meta, "contra_azar": al_azar,
                    f"contra_{mejor}": a_la_constante,
                    "juega": con_meta > 0 and al_azar == "gana" and a_la_constante == "gana"}
    return filas


def resumen(filas: dict[str, Fila]) -> str:
    lineas = ["| fila | media | por semilla |", "|---|---|---|"]
    for nombre, fila in filas.items():
        lineas.append(f"| {nombre} | {fila.valores.mean():.3f} | "
                      + " ".join(f"{v:.3f}" for v in fila.valores.mean(axis=1)) + " |")
    return "\n".join(lineas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, type=Path,
                        help="carpeta de los afinados (eval_afinado_*.json)")
    parser.add_argument("--clonado", required=True, type=Path,
                        help="carpeta de los clonados (eval_clonado_codiciosa.json)")
    parser.add_argument("--salida", type=Path, default=None)
    args = parser.parse_args()

    carpetas = {"afinado": args.dir, "muestreada": args.dir, "clonado": args.clonado}
    informes = {nombre: json.loads((carpetas[nombre] / fichero).read_text())
                for nombre, (fichero, _) in INFORMES.items()}
    comprueba(informes)
    niveles = niveles_en_orden(informes["afinado"])
    analisis: dict[str, Any] = {"informes": {n: str(carpetas[n] / f) for n, (f, _) in INFORMES.items()}}

    print("# Fase 6d: analisis registrado")
    print(f"\n{N_NIVELES} niveles de test, seis modelos x semillas {SEMILLAS}, progreso."
          " Comprobacion previa superada.")
    progreso: dict[str, dict[str, Fila]] = {}
    revisados = 0
    for nombre, informe in informes.items():
        permutaciones = int(informe["config"].get("permutations", 10000))
        filas = filas_de_informe(informe, "progreso")
        progreso[nombre] = filas
        decide_aqui = nombre == "afinado"
        resultado = bateria(filas, niveles, permutaciones, con_holm=decide_aqui)
        revisados += coincide_con_el_informe(
            resultado["principales"] + resultado["secundarios"], informe)
        resultado["juega"] = juega(filas_de_informe(informe, "meta"), resultado)
        resultado["tabla"] = {n: f.valores.mean(axis=1).tolist() for n, f in filas.items()}
        analisis[nombre] = resultado
        titulo = {"afinado": "Afinados, codiciosa (DECIDE)",
                  "muestreada": "Afinados, muestreada (secundario)",
                  "clonado": "Solo clonados, codiciosa (secundario)"}[nombre]
        print(f"\n## {titulo}\n\n{resumen(filas)}")
        print(tabla("Pregunta del proyecto" + (" (Holm sobre los dos)" if decide_aqui else
                                               " (sin Holm)"), resultado["principales"]))
        print(tabla(f"Secundarios (umbral 0,05; mejor constante: {resultado['mejor_constante']})",
                    resultado["secundarios"]))
        print("\n**Juega** (meta en algun nivel y gana al azar y a la mejor constante)\n")
        for m, j in resultado["juega"].items():
            print(f"- {m}: {'SI' if j['juega'] else 'no'} (niveles con meta {j['niveles_con_meta']},"
                  f" contra azar {j['contra_azar']},"
                  f" contra {resultado['mejor_constante']} {j['contra_' + resultado['mejor_constante']]})")

    permutaciones = int(informes["afinado"]["config"].get("permutations", 10000))
    analisis["ganancia_por_ppo"] = ganancia_por_ppo(progreso["afinado"], progreso["clonado"],
                                                    niveles, permutaciones)
    print(tabla("Ganancia por PPO, afinado contra clonado (umbral 0,05)",
                analisis["ganancia_por_ppo"]))
    analisis["p_comprobadas_contra_el_informe"] = revisados
    print(f"\n{revisados} p coinciden con las de `contrastes` de los informes.")

    if args.salida:
        args.salida.write_text(json.dumps(analisis, indent=2, ensure_ascii=False))
        print(f"\nanalisis en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
