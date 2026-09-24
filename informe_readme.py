"""Vuelca las cifras de `eval_report.json` en la seccion de estado de los dos README.

Uso:
    python informe_readme.py artifacts/train_64/eval_report.json

Escribe entre `<!-- RESULTS -->` y `<!-- /RESULTS -->` (README.md) y entre
`<!-- RESULTADOS -->` y `<!-- /RESULTADOS -->` (README.es.md). Si falta la marca de
cierre, la pone. Relanzarlo sustituye el bloque, no lo duplica.

Solo copia numeros: la tabla de `evaluate.py` y los contrastes de la mosca con su
p-valor. La lectura se escribe a mano, fuera del bloque. Con menos de dos semillas lo
dice dentro del propio bloque, porque sin variabilidad entre semillas ningun contraste
se puede juzgar (es el aviso que ya da `evaluate.py`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MARCAS = {
    "README.md": ("<!-- RESULTS -->", "<!-- /RESULTS -->"),
    "README.es.md": ("<!-- RESULTADOS -->", "<!-- /RESULTADOS -->"),
}

# Cabeceras de la tabla de evaluate.py, que salen en espanol.
TRADUCCION = {
    "modelo": "model",
    "L2 1 paso": "L2 one step",
    "L2 despliegue": "L2 rollout",
    "t hasta corr<0,8": "t until corr<0.8",
    "inyeccion": "injection",
    "alcanzable": "reachable",
    "fuera del codificador": "outside the encoder",
}

METRICAS = ("l2_1paso", "l2_despliegue", "tiempo_corr")
NOMBRE_METRICA = {
    "es": {"l2_1paso": "L2 a un paso", "l2_despliegue": "L2 en despliegue",
           "tiempo_corr": "tiempo hasta corr<0,8"},
    "en": {"l2_1paso": "L2 one step", "l2_despliegue": "L2 rollout",
           "tiempo_corr": "time until corr<0.8"},
}
CONTROLES = ("persistencia", "sin_nucleo", "aleatorio", "recableado")


def n_semillas(informe: dict[str, Any]) -> int:
    return int(informe["resumen"]["mosca"]["l2_1paso"]["n_semillas"])


def tabla_en_ingles(tabla: str) -> str:
    cabecera, resto = tabla.split("\n", 1)
    celdas = [c.strip() for c in cabecera.strip("|").split("|")]
    celdas = [TRADUCCION.get(c, c) for c in celdas]
    return "| " + " | ".join(celdas) + " |\n" + resto


def formato_p(prueba: dict[str, Any]) -> str:
    # Con n permutaciones, un p de 0 solo dice que es menor que 1/n.
    if prueba["p"] == 0 and "n_permutaciones" in prueba:
        return f"< {1 / prueba['n_permutaciones']:.4f}"
    return f"{prueba['p']:.4f}"


def tabla_de_contrastes(informe: dict[str, Any], idioma: str) -> str:
    pruebas = informe["pruebas_de_permutacion"]
    if idioma == "es":
        lineas = ["| contraste | diferencia (mosca - control) | p |", "|---|---|---|"]
    else:
        lineas = ["| contrast | difference (fly - control) | p |", "|---|---|---|"]
    for metrica in METRICAS:
        for control in CONTROLES:
            clave = f"{metrica}: mosca vs {control}"
            if clave not in pruebas:
                continue
            prueba = pruebas[clave]
            etiqueta = f"{NOMBRE_METRICA[idioma][metrica]}: mosca vs {control}"
            if idioma == "en":
                etiqueta = etiqueta.replace("mosca vs", "fly vs")
            dentro = prueba.get("dentro_de_la_variabilidad_entre_semillas")
            if dentro:
                etiqueta += " *"
            lineas.append(f"| {etiqueta} | {prueba['diferencia_media']:+.4f} | "
                          f"{formato_p(prueba)} |")
    return "\n".join(lineas)


def bloque(informe: dict[str, Any], idioma: str) -> str:
    semillas = n_semillas(informe)
    tabla = informe["tabla_markdown"]
    if idioma == "es":
        cabecera = (f"Cifras de `{informe['config']}`, conjunto `{informe['split']}`, "
                    f"{semillas} semilla(s) por modelo, despliegue de "
                    f"{informe['pasos_despliegue']} pasos. Media ± desviacion entre "
                    "semillas; error espectral por bandas y en log.")
        pie = ("\\* la diferencia cae dentro de la variabilidad entre semillas.\n"
               "En L2 y en espectro, negativo es mejor para la mosca; en tiempo, positivo.")
        aviso = ("**Con una sola semilla ningun contraste se puede juzgar:** los p-valores "
                 "son entre trayectorias, no entre entrenamientos.")
    else:
        tabla = tabla_en_ingles(tabla)
        cabecera = (f"Numbers from `{informe['config']}`, `{informe['split']}` split, "
                    f"{semillas} seed(s) per model, {informe['pasos_despliegue']}-step "
                    "rollout. Mean ± std across seeds; spectral error by band, in log.")
        pie = ("\\* the difference falls within the variability across seeds.\n"
               "For L2 and spectrum, negative favours the fly; for time, positive does.")
        aviso = ("**With a single seed no contrast can be judged:** the p-values are "
                 "across trajectories, not across training runs.")
    partes = [cabecera, "", tabla, "", tabla_de_contrastes(informe, idioma), "", pie]
    if semillas < 2:
        partes += ["", aviso]
    return "\n".join(partes)


def sustituye(texto: str, apertura: str, cierre: str, contenido: str) -> str:
    if apertura not in texto:
        raise ValueError(f"no esta la marca {apertura}")
    antes, despues = texto.split(apertura, 1)
    if cierre in despues:
        despues = despues.split(cierre, 1)[1]
    despues = despues.lstrip("\n")
    return f"{antes}{apertura}\n{contenido}\n{cierre}\n\n{despues}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("informe", type=Path)
    parser.add_argument("--raiz", type=Path, default=Path("."))
    args = parser.parse_args()

    informe = json.loads(args.informe.read_text())
    for nombre, (apertura, cierre) in MARCAS.items():
        idioma = "es" if nombre.endswith(".es.md") else "en"
        ruta = args.raiz / nombre
        ruta.write_text(sustituye(ruta.read_text(), apertura, cierre,
                                  bloque(informe, idioma)))
        print(f"{ruta}: {n_semillas(informe)} semilla(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
