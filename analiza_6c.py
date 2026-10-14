"""Analisis final de la fase 6c. Escrito el 14-10-2026, ANTES de ver ninguna replica.

Aplica la regla de decision registrada en RESULTADOS.md (seccion del 14-10-2026) a los
informes de `evaluate_rl.py`:

    python analiza_6c.py --principal artifacts/mario_nes_sola/rl_evaluacion_final_muestreada.json \\
                         --extra artifacts/mario_nes_sola/rl_evaluacion_final_extra_muestreada.json \\
                         --salida artifacts/mario_nes_sola/analisis_6c_muestreada.json

El informe principal (val+test, 7 niveles) decide. El extra (6 niveles que no estan en
ninguna particion) solo entra en el analisis por tipo de nivel, que es secundario.

Regla de decision, para que esta en el codigo y no en la cabeza de quien lea los numeros:
  - Diferencia = media sobre niveles de (media de semillas de A - media de semillas de B).
  - "Gana" o "pierde" solo si (a) la p exacta por nivel queda bajo el umbral (Holm sobre
    los dos contrastes principales; 0,05 en los secundarios) y (b) las semillas de A
    quedan todas por encima, o todas por debajo, de las de B.
  - Si no, "empate" solo si |diferencia| < dispersion entre semillas (media de las
    desviaciones tipicas de semilla de los modelos con mas de una). Si no, "no resuelto".
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_rl import TIPO_DE_NIVEL, permutacion_exacta_entre_grupos, permutacion_exacta_pareada

MODELOS = ("mosca", "aleatorio", "recableado", "sin_nucleo", "denso", "mlp")
PRINCIPALES = (("mosca", "aleatorio"), ("mosca", "recableado"))
SECUNDARIOS_MOSCA = (("mosca", "sin_nucleo"), ("mosca", "denso"), ("mosca", "mlp"))
REFERENCIAS = ("azar", "salto_cte")


@dataclass
class Fila:
    """Un modelo (una fila por semilla) o una referencia (una sola fila), por nivel."""

    nombre: str
    niveles: list[str]
    valores: np.ndarray            # (semillas, niveles)
    checkpoints: list[str]

    def columnas(self, niveles: list[str]) -> np.ndarray:
        return self.valores[:, [self.niveles.index(n) for n in niveles]]


def filas_de_informe(informe: dict[str, Any], metrica: str) -> dict[str, Fila]:
    """Medias por semilla y nivel a partir de las series por partida del informe."""
    niveles_partida = np.array([p["nivel"] for p in informe["partidas"]])
    niveles = sorted(set(niveles_partida.tolist()))

    def por_nivel(serie: list[float]) -> np.ndarray:
        valores = np.asarray(serie, dtype=float)
        return np.array([valores[niveles_partida == n].mean() for n in niveles])

    filas: dict[str, Fila] = {}
    for nombre, semillas in informe["series_por_semilla"].items():
        filas[nombre] = Fila(nombre, niveles, np.stack([por_nivel(s[metrica]) for s in semillas]),
                             [s["checkpoint"] for s in semillas])
    for nombre, series in informe.get("series_referencias", {}).items():
        filas[nombre] = Fila(nombre, niveles, por_nivel(series[metrica])[None, :], [nombre])
    return filas


def une(a: dict[str, Fila], b: dict[str, Fila]) -> dict[str, Fila]:
    """Junta dos informes con niveles distintos. Solo quedan las filas que estan en los dos,
    y tienen que ser los mismos checkpoints en el mismo orden."""
    unidas: dict[str, Fila] = {}
    for nombre in a.keys() & b.keys():
        fa, fb = a[nombre], b[nombre]
        if set(fa.niveles) & set(fb.niveles):
            raise ValueError(f"{nombre}: los dos informes comparten niveles")
        if fa.checkpoints != fb.checkpoints:
            raise ValueError(f"{nombre}: checkpoints distintos {fa.checkpoints} / {fb.checkpoints}")
        unidas[nombre] = Fila(nombre, fa.niveles + fb.niveles,
                              np.concatenate([fa.valores, fb.valores], axis=1), fa.checkpoints)
    return unidas


def dispersion_entre_semillas(*filas: Fila) -> float:
    """Media de las desviaciones tipicas entre semillas (de la media sobre niveles)."""
    desviaciones = [float(f.valores.mean(axis=1).std(ddof=1)) for f in filas if f.valores.shape[0] >= 2]
    return float(np.mean(desviaciones)) if desviaciones else float("nan")


def contraste(a: Fila, b: Fila, niveles: list[str]) -> dict[str, Any]:
    columnas_a, columnas_b = a.columnas(niveles), b.columnas(niveles)
    por_nivel = columnas_a.mean(axis=0) - columnas_b.mean(axis=0)
    prueba = permutacion_exacta_pareada(por_nivel)
    medias_a, medias_b = columnas_a.mean(axis=1), columnas_b.mean(axis=1)
    resultado: dict[str, Any] = {
        "a": a.nombre, "b": b.nombre, "niveles": niveles,
        "diferencia": float(por_nivel.mean()),
        "diferencia_por_nivel": dict(zip(niveles, por_nivel.tolist())),
        "niveles_a_favor": int((por_nivel > 0).sum()),
        "p_nivel": prueba["p"], "p_nivel_minima": prueba["p_minima_posible"],
        "semillas_a": medias_a.tolist(), "semillas_b": medias_b.tolist(),
        "separadas": bool(medias_a.min() > medias_b.max() or medias_a.max() < medias_b.min()),
        "dispersion": dispersion_entre_semillas(
            Fila(a.nombre, niveles, columnas_a, a.checkpoints),
            Fila(b.nombre, niveles, columnas_b, b.checkpoints)),
    }
    if medias_a.size >= 2 and medias_b.size >= 2:
        resultado["p_semilla"] = permutacion_exacta_entre_grupos(medias_a, medias_b)["p"]
    return resultado


def veredicto(p_nivel: float, umbral: float, separadas: bool, diferencia: float,
              dispersion: float) -> str:
    if p_nivel < umbral and separadas:
        return "gana" if diferencia > 0 else "pierde"
    if np.isfinite(dispersion) and abs(diferencia) < dispersion:
        return "empate"
    return "no resuelto"


def umbrales_holm(ps: list[float], alfa: float = 0.05) -> list[float]:
    """Umbral de cada prueba con Holm. En cuanto una no pasa, las siguientes tampoco
    pueden (umbral 0)."""
    umbrales = [0.0] * len(ps)
    sigue = True
    for k, i in enumerate(np.argsort(ps, kind="stable")):
        umbral = alfa / (len(ps) - k)
        umbrales[int(i)] = umbral if sigue else 0.0
        sigue = sigue and ps[int(i)] < umbral
    return umbrales


def decide(contrastes: list[dict[str, Any]], umbrales: list[float]) -> list[dict[str, Any]]:
    for c, umbral in zip(contrastes, umbrales):
        c["umbral"] = umbral
        c["veredicto"] = veredicto(c["p_nivel"], umbral, c["separadas"], c["diferencia"],
                                   c["dispersion"])
    return contrastes


def bateria(filas: dict[str, Fila], niveles: list[str], con_holm: bool) -> dict[str, list[dict[str, Any]]]:
    """Los contrastes registrados sobre un conjunto de niveles."""
    def hay(a: str, b: str) -> bool:
        return a in filas and b in filas

    principales = [contraste(filas[a], filas[b], niveles) for a, b in PRINCIPALES if hay(a, b)]
    umbrales = (umbrales_holm([c["p_nivel"] for c in principales]) if con_holm
                else [0.05] * len(principales))
    parejas = list(SECUNDARIOS_MOSCA)
    parejas += [(m, r) for m in MODELOS for r in REFERENCIAS]
    parejas += [(m, f"marginal_{m}") for m in MODELOS]
    secundarios = [contraste(filas[a], filas[b], niveles) for a, b in parejas if hay(a, b)]
    return {"principales": decide(principales, umbrales),
            "secundarios": decide(secundarios, [0.05] * len(secundarios))}


def linea(c: dict[str, Any]) -> str:
    semillas = " ".join(f"{v:.3f}" for v in c["semillas_a"])
    p_semilla = f"  p semilla {c['p_semilla']:.2f}" if "p_semilla" in c else ""
    return (f"| {c['a']} vs {c['b']} | {c['diferencia']:+.3f} | {c['niveles_a_favor']}/"
            f"{len(c['niveles'])} | {c['p_nivel']:.4f} (umbral {c['umbral']:.3f}){p_semilla} |"
            f" {'si' if c['separadas'] else 'no'} | {c['dispersion']:.3f} | {semillas} |"
            f" **{c['veredicto']}** |")


def tabla(titulo: str, contrastes: list[dict[str, Any]]) -> str:
    cabecera = ("| contraste | dif | niveles a favor | p por nivel | semillas separadas |"
                " dispersion | semillas de A | veredicto |\n|---|---|---|---|---|---|---|---|")
    return f"\n**{titulo}**\n\n{cabecera}\n" + "\n".join(linea(c) for c in contrastes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--principal", required=True, type=Path)
    parser.add_argument("--extra", type=Path, default=None)
    parser.add_argument("--salida", type=Path, default=None)
    args = parser.parse_args()

    principal = json.loads(args.principal.read_text())
    extra = json.loads(args.extra.read_text()) if args.extra else None
    analisis: dict[str, Any] = {"principal": str(args.principal),
                                "extra": str(args.extra) if args.extra else None,
                                "muestreada": principal["muestreada"]}
    filas = filas_de_informe(principal, "progreso")
    niveles = filas["azar"].niveles
    print(f"# Fase 6c: analisis registrado ({'muestreada' if principal['muestreada'] else 'codiciosa'})")
    print(f"\nPrincipal: {len(niveles)} niveles ({', '.join(niveles)}), progreso.")
    print("\n| fila | media | por semilla | " + " | ".join(niveles) + " |")
    print("|---|---|---|" + "---|" * len(niveles))
    for nombre, fila in filas.items():
        print(f"| {nombre} | {fila.valores.mean():.3f} | "
              + " ".join(f"{v:.3f}" for v in fila.valores.mean(axis=1)) + " | "
              + " | ".join(f"{v:.3f}" for v in fila.valores.mean(axis=0)) + " |")
    analisis["tabla"] = {n: {"semillas": f.valores.mean(axis=1).tolist(),
                             "niveles": dict(zip(f.niveles, f.valores.mean(axis=0).tolist()))}
                         for n, f in filas.items()}
    analisis["principal_7_niveles"] = bateria(filas, niveles, con_holm=True)
    print(tabla("Pregunta del proyecto (Holm sobre los dos)", analisis["principal_7_niveles"]["principales"]))
    print(tabla("Secundarios (umbral 0,05)", analisis["principal_7_niveles"]["secundarios"]))

    if extra is not None:
        todas = une(filas, filas_de_informe(extra, "progreso"))
        analisis["por_tipo"] = {}
        for tipo in ("exterior", "atletico"):
            del_tipo = [n for n in todas["azar"].niveles if TIPO_DE_NIVEL.get(n) == tipo]
            analisis["por_tipo"][tipo] = bateria(todas, del_tipo, con_holm=False)
            plural = {"exterior": "exteriores", "atletico": "atleticos"}[tipo]
            print(f"\n## Niveles {plural} ({', '.join(del_tipo)}), secundario")
            print(tabla("Pregunta del proyecto (sin Holm)", analisis["por_tipo"][tipo]["principales"]))
            print(tabla("Secundarios", analisis["por_tipo"][tipo]["secundarios"]))

    # "Juega" = bandera en algun nivel no visto, con cualquier semilla.
    metas = filas_de_informe(principal, "meta")
    if extra is not None:
        metas = une(metas, filas_de_informe(extra, "meta"))
    analisis["banderas"] = {
        nombre: {n: float(v) for n, v in zip(f.niveles, f.valores.max(axis=0)) if v > 0}
        for nombre, f in metas.items()
    }
    con_bandera = {n: b for n, b in analisis["banderas"].items() if b}
    print("\n## Bandera en niveles no vistos\n")
    print("\n".join(f"- {n}: {b}" for n, b in con_bandera.items()) if con_bandera
          else "Ninguna fila llega a la bandera en ningun nivel no visto.")

    if args.salida:
        args.salida.write_text(json.dumps(analisis, indent=2, ensure_ascii=False))
        print(f"\nanalisis en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
