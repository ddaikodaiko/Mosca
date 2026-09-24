"""Tests del volcado de eval_report.json a los README.

Lo que importa es que relanzarlo no duplique el bloque ni toque lo que hay fuera de
las marcas: la lectura escrita a mano vive justo al lado.
"""

from __future__ import annotations

from typing import Any

from informe_readme import bloque, sustituye


def _informe(semillas: int) -> dict[str, Any]:
    dato = {"media": 0.1, "desviacion": 0.0, "n_semillas": semillas}
    return {
        "config": "configs/x.yaml",
        "split": "test",
        "pasos_despliegue": 50,
        "resumen": {"mosca": {"l2_1paso": dato}},
        "tabla_markdown": "| modelo | L2 1 paso |\n|---|---|\n| mosca | 0.1000 ± 0.0000 |",
        "pruebas_de_permutacion": {
            "l2_1paso: mosca vs persistencia": {
                "diferencia_media": -0.0006, "p": 0.0, "n_permutaciones": 10000,
                "dentro_de_la_variabilidad_entre_semillas": None},
            "l2_despliegue: mosca vs aleatorio": {
                "diferencia_media": 0.002, "p": 0.4,
                "dentro_de_la_variabilidad_entre_semillas": True},
        },
    }


def test_relanzar_sustituye_el_bloque_sin_duplicarlo() -> None:
    texto = "## Estado\n\nantes\n\n<!-- R -->\n\n## Lo que hemos aprendido\n\nfuera\n"
    una = sustituye(texto, "<!-- R -->", "<!-- /R -->", "cifras 1")
    dos = sustituye(una, "<!-- R -->", "<!-- /R -->", "cifras 2")
    assert dos.count("<!-- R -->") == 1 and dos.count("<!-- /R -->") == 1
    assert "cifras 1" not in dos and "cifras 2" in dos
    assert dos.startswith("## Estado\n\nantes\n\n")
    assert dos.endswith("<!-- /R -->\n\n## Lo que hemos aprendido\n\nfuera\n")


def test_con_una_semilla_avisa_de_que_no_se_puede_juzgar() -> None:
    assert "sola semilla" in bloque(_informe(1), "es")
    assert "single seed" in bloque(_informe(1), "en")
    assert "sola semilla" not in bloque(_informe(3), "es")


def test_marca_los_contrastes_dentro_de_la_variabilidad() -> None:
    texto = bloque(_informe(3), "en")
    assert "L2 rollout: fly vs aleatorio * | +0.0020 | 0.4000 |" in texto
    assert "L2 one step: fly vs persistencia | -0.0006 | < 0.0001 |" in texto
    assert "| model | L2 one step |" in texto
