"""Analisis de la fase 6e (EXPLORATORIA). Escrito el 21-10-2026, ANTES de tener ningun dato.

Pregunta: el deficit de la mosca en la 6d (0,462 contra 0,568 del aleatorio y 0,527 del
recableado), ¿viene del reescalado espectral? Con la normalizacion del diseno (`max_eig`)
el grueso del espectro de la mosca queda en 0,013, contra 0,356 del aleatorio y 0,102 del
recableado: su recurrencia esta casi apagada. La 6e repite la 6d entera (clonado + 300
iteraciones de PPO, semillas 0-2, los mismos 200 niveles de test) cambiando solo
`core.normalisation` a `row`, en los tres modelos a los que afecta. Plan en RESULTADOS.md,
seccion del 21-10-2026.

    python analiza_6e.py --row artifacts/mario_proto_6e_row_gcp \\
                         --max artifacts/mario_proto_6c_soltar_gcp \\
                         --salida artifacts/mario_proto_6e_row_gcp/analisis_6e.json

De cada carpeta lee desde_clonado/eval_afinado_codiciosa.json (decide),
desde_clonado/eval_afinado_muestreada.json y clonado/eval_clonado_codiciosa.json.

Regla:
  - Principal, con Holm sobre los dos: la interaccion por nivel
        d = (mosca_row - X_row) - (mosca_max - X_max),   X en {aleatorio, recableado},
    cada termino como media de las tres semillas en ese nivel. Permutacion pareada de
    signos sobre los 200 niveles (`evaluate.prueba_de_permutacion`, 10.000, semilla 0).
    Por semilla, D_s con la misma semilla en las cuatro filas.
      "reduce el deficit"  p < umbral, media > 0 y las tres D_s > 0
      "lo agranda"         p < umbral, media < 0 y las tres D_s < 0
      "sin efecto"         si no, y |media| <= desviacion tipica de las D_s
      "no resuelto"        en cualquier otro caso
  - Secundarios, umbral 0,05 y la regla de la 6d (`analiza_6c.veredicto`): mosca_row
    contra aleatorio_row y recableado_row; cada modelo con `row` contra si mismo con
    `max_eig`; mosca_row contra `sin_nucleo` (el de la 6d: sin aristas, la normalizacion
    no le cambia nada); cada modelo con `row` contra azar y la mejor constante.
  - Lo mismo, sin Holm, en la muestreada y en los solo clonados.

Prediccion de H1, registrada antes de los datos: la interaccion "reduce el deficit"
contra los dos controles, mosca_row gana a mosca_max y mosca_row gana a `sin_nucleo`. Si
la interaccion sale "sin efecto" contra los dos, el reescalado queda descartado como causa.
`row` no iguala el grueso (0,146 contra 0,595), asi que H1 no predice que la mosca alcance
al aleatorio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analiza_6c import Fila, decide, filas_de_informe, tabla, umbrales_holm
from analiza_6d import CONSTANTES, contraste, niveles_en_orden
from evaluate import prueba_de_permutacion

CABLEADOS = ("mosca", "aleatorio", "recableado")
SEMILLAS = (0, 1, 2)
N_NIVELES = 200
INFORMES = {
    "afinado": ("desde_clonado/eval_afinado_codiciosa.json", False),
    "muestreada": ("desde_clonado/eval_afinado_muestreada.json", True),
    "clonado": ("clonado/eval_clonado_codiciosa.json", False),
}


def carga(carpeta: Path) -> dict[str, dict[str, Any]]:
    return {nombre: json.loads((carpeta / fichero).read_text())
            for nombre, (fichero, _) in INFORMES.items()}


def comprueba(row: dict[str, dict[str, Any]], maximo: dict[str, dict[str, Any]]) -> None:
    """Para si falta una semilla o un modelo, o si los niveles no son los mismos."""
    partidas = row["afinado"]["partidas"]
    niveles = [p["nivel"] for p in partidas]
    if len(niveles) != N_NIVELES or len(set(niveles)) != N_NIVELES:
        raise SystemExit(f"6e: {len(niveles)} partidas y {len(set(niveles))} niveles")
    for etiqueta, informes, modelos in (("6e", row, CABLEADOS),
                                        ("6d", maximo, CABLEADOS + ("sin_nucleo",))):
        for nombre, informe in informes.items():
            if bool(informe["muestreada"]) != INFORMES[nombre][1]:
                raise SystemExit(f"{etiqueta} {nombre}: muestreada = {informe['muestreada']}")
            if informe["partidas"] != partidas:
                raise SystemExit(f"{etiqueta} {nombre}: no juega los mismos niveles")
            for modelo in modelos:
                esperados = [f"{modelo}_s{s}" for s in SEMILLAS]
                tiene = [s["checkpoint"] for s in informe["series_por_semilla"].get(modelo, [])]
                if tiene != esperados:
                    raise SystemExit(f"{etiqueta} {nombre}: {modelo} tiene {tiene}")
        if etiqueta == "6e":
            sobran = set(row["afinado"]["series_por_semilla"]) - set(CABLEADOS)
            if sobran:
                raise SystemExit(f"6e: modelos que no tocaban: {sorted(sobran)}")


def referencias_iguales(row: dict[str, Any], maximo: dict[str, Any]) -> bool:
    """Las politicas de referencia no dependen de la normalizacion: si no salen iguales,
    algo distinto de los niveles ha cambiado entre las dos tandas. Se informa, no para."""
    a, b = row["series_referencias"], maximo["series_referencias"]
    return a.keys() == b.keys() and all(
        np.array_equal(a[k]["progreso"], b[k]["progreso"]) for k in a)


def renombra(filas: dict[str, Fila], sufijo: str, modelos: tuple[str, ...]) -> dict[str, Fila]:
    return {f"{m}_{sufijo}": Fila(f"{m}_{sufijo}", filas[m].niveles, filas[m].valores,
                                  filas[m].checkpoints) for m in modelos}


def interaccion(m_row: Fila, x_row: Fila, m_max: Fila, x_max: Fila, niveles: list[str],
                permutaciones: int, umbral: float) -> dict[str, Any]:
    c = {n: f.columnas(niveles) for n, f in
         (("m_row", m_row), ("x_row", x_row), ("m_max", m_max), ("x_max", x_max))}
    deficit_row = c["m_row"].mean(axis=0) - c["x_row"].mean(axis=0)
    deficit_max = c["m_max"].mean(axis=0) - c["x_max"].mean(axis=0)
    prueba = prueba_de_permutacion(deficit_row, deficit_max, permutaciones, semilla=0)
    d = deficit_row - deficit_max
    por_semilla = ((c["m_row"].mean(axis=1) - c["x_row"].mean(axis=1))
                   - (c["m_max"].mean(axis=1) - c["x_max"].mean(axis=1)))
    media, desviacion = float(d.mean()), float(por_semilla.std(ddof=1))
    if prueba["p"] < umbral and media > 0 and (por_semilla > 0).all():
        veredicto = "reduce el deficit"
    elif prueba["p"] < umbral and media < 0 and (por_semilla < 0).all():
        veredicto = "lo agranda"
    elif abs(media) <= desviacion:
        veredicto = "sin efecto"
    else:
        veredicto = "no resuelto"
    return {"control": x_row.nombre.removesuffix("_row"),
            "deficit_max": float(deficit_max.mean()), "deficit_row": float(deficit_row.mean()),
            "interaccion": media, "niveles_a_favor": int((d > 0).sum()),
            "p_nivel": prueba["p"], "umbral": umbral, "por_semilla": por_semilla.tolist(),
            "desviacion_por_semilla": desviacion, "veredicto": veredicto}


def bateria(row: dict[str, Fila], maximo: dict[str, Fila], niveles: list[str],
            permutaciones: int, con_holm: bool) -> dict[str, Any]:
    principales_sin_umbral = [
        prueba_de_permutacion(
            row["mosca_row"].columnas(niveles).mean(0) - row[f"{x}_row"].columnas(niveles).mean(0),
            maximo["mosca_max"].columnas(niveles).mean(0) - maximo[f"{x}_max"].columnas(niveles).mean(0),
            permutaciones, semilla=0)["p"]
        for x in ("aleatorio", "recableado")]
    umbrales = umbrales_holm(principales_sin_umbral) if con_holm else [0.05, 0.05]
    principales = [
        interaccion(row["mosca_row"], row[f"{x}_row"], maximo["mosca_max"], maximo[f"{x}_max"],
                    niveles, permutaciones, u)
        for x, u in zip(("aleatorio", "recableado"), umbrales)]

    todas = {**row, **maximo}
    mejor = max((k for k in CONSTANTES if k in todas),
                key=lambda k: float(todas[k].valores.mean()))
    parejas = [("mosca_row", "aleatorio_row"), ("mosca_row", "recableado_row")]
    parejas += [(f"{m}_row", f"{m}_max") for m in CABLEADOS]
    parejas += [("mosca_row", "sin_nucleo")]
    parejas += [(f"{m}_row", r) for m in CABLEADOS for r in ("azar", mejor)]
    secundarios = [contraste(todas[a], todas[b], niveles, permutaciones) for a, b in parejas]
    return {"principales": principales,
            "secundarios": decide(secundarios, [0.05] * len(secundarios)),
            "mejor_constante": mejor}


def tabla_interaccion(titulo: str, filas: list[dict[str, Any]]) -> str:
    lineas = [f"\n**{titulo}**\n",
              "| control | deficit max_eig | deficit row | interaccion | niveles a favor | p por nivel"
              " | por semilla | veredicto |", "|---|---|---|---|---|---|---|---|"]
    for f in filas:
        lineas.append(
            f"| {f['control']} | {f['deficit_max']:+.3f} | {f['deficit_row']:+.3f} |"
            f" {f['interaccion']:+.3f} | {f['niveles_a_favor']}/{N_NIVELES} |"
            f" {f['p_nivel']:.4f} (umbral {f['umbral']:.3f}) |"
            f" {' '.join(f'{v:+.3f}' for v in f['por_semilla'])} | **{f['veredicto']}** |")
    return "\n".join(lineas)


def resumen(filas: dict[str, Fila]) -> str:
    lineas = ["| fila | media | por semilla |", "|---|---|---|"]
    for nombre, fila in filas.items():
        lineas.append(f"| {nombre} | {fila.valores.mean():.3f} | "
                      + " ".join(f"{v:.3f}" for v in fila.valores.mean(axis=1)) + " |")
    return "\n".join(lineas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--row", required=True, type=Path, help="carpeta de la 6e")
    parser.add_argument("--max", required=True, type=Path, help="carpeta de la 6d")
    parser.add_argument("--salida", type=Path, default=None)
    args = parser.parse_args()

    row_informes, max_informes = carga(args.row), carga(args.max)
    comprueba(row_informes, max_informes)
    niveles = niveles_en_orden(row_informes["afinado"])
    analisis: dict[str, Any] = {"row": str(args.row), "max": str(args.max)}

    print("# Fase 6e (exploratoria): analisis escrito antes de los datos")
    print(f"\n{N_NIVELES} niveles de test, mosca/aleatorio/recableado x semillas {SEMILLAS},"
          " progreso. Comprobacion previa superada.")
    for nombre in INFORMES:
        permutaciones = int(row_informes[nombre]["config"].get("permutations", 10000))
        filas_row = filas_de_informe(row_informes[nombre], "progreso")
        filas_max = filas_de_informe(max_informes[nombre], "progreso")
        row = renombra(filas_row, "row", CABLEADOS)
        maximo = renombra(filas_max, "max", CABLEADOS)
        maximo["sin_nucleo"] = filas_max["sin_nucleo"]
        for ref in ("azar",) + CONSTANTES + ("heuristica",):
            if ref in filas_row:
                maximo[ref] = filas_row[ref]
        decide_aqui = nombre == "afinado"
        resultado = bateria(row, maximo, niveles, permutaciones, con_holm=decide_aqui)
        resultado["referencias_iguales_en_las_dos_tandas"] = referencias_iguales(
            row_informes[nombre], max_informes[nombre])
        resultado["tabla"] = {n: f.valores.mean(axis=1).tolist() for n, f in {**row, **maximo}.items()}
        analisis[nombre] = resultado
        titulo = {"afinado": "Afinados, codiciosa (DECIDE)",
                  "muestreada": "Afinados, muestreada (secundario)",
                  "clonado": "Solo clonados, codiciosa (secundario)"}[nombre]
        print(f"\n## {titulo}\n\n{resumen({**row, **maximo})}")
        print(f"\nReferencias identicas en las dos tandas: "
              f"{resultado['referencias_iguales_en_las_dos_tandas']}")
        print(tabla_interaccion("Interaccion normalizacion x cableado"
                                + (" (Holm sobre los dos)" if decide_aqui else " (sin Holm)"),
                                resultado["principales"]))
        print(tabla(f"Secundarios (umbral 0,05; mejor constante: {resultado['mejor_constante']})",
                    resultado["secundarios"]))

    if args.salida:
        args.salida.write_text(json.dumps(analisis, indent=2, ensure_ascii=False))
        print(f"\nanalisis en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
