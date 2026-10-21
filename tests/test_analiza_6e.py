"""Tests del analisis de la fase 6e (`analiza_6e.py`), escritos antes de tener datos.

Informes sinteticos con la forma de los de `evaluate_rl.py`: progreso por nivel = media
del modelo y semilla + dificultad del nivel (comun a todos) + ruido.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from analiza_6c import filas_de_informe
from analiza_6e import CABLEADOS, INFORMES, N_NIVELES, comprueba, interaccion, renombra

REFERENCIAS = {"azar": 0.20, "siempre_dcha": 0.08, "salto_cte": 0.10, "salto_corriendo": 0.12}
NIVELES = [str(3_000_000 + i) for i in range(N_NIVELES)]


def _informe(medias: dict[str, list[float]], muestreada: bool, semilla: int = 0) -> dict[str, Any]:
    generador = np.random.default_rng(semilla)
    dificultad = generador.normal(0, 0.1, N_NIVELES)

    def serie(media: float) -> list[float]:
        return (media + dificultad + generador.normal(0, 0.05, N_NIVELES)).tolist()

    return {
        "config": {"permutations": 2000}, "muestreada": muestreada,
        "partidas": [{"particion": "test", "semilla": int(n), "nivel": n} for n in NIVELES],
        "series_por_semilla": {
            m: [{"checkpoint": f"{m}_s{s}", "progreso": serie(v), "meta": [0.0] * N_NIVELES}
                for s, v in enumerate(vs)] for m, vs in medias.items()},
        "series_referencias": {r: {"progreso": serie(v), "meta": [0.0] * N_NIVELES}
                               for r, v in REFERENCIAS.items()},
    }


def _tanda(medias: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    return {n: _informe(medias, m, semilla=i) for i, (n, (_, m)) in enumerate(INFORMES.items())}


MAX = {"mosca": [0.43, 0.48, 0.47], "aleatorio": [0.60, 0.57, 0.53],
       "recableado": [0.63, 0.45, 0.50], "sin_nucleo": [0.51, 0.47, 0.44]}
ROW = {m: MAX[m] for m in CABLEADOS}


def test_comprueba_acepta_las_dos_tandas_completas() -> None:
    comprueba(_tanda(ROW), _tanda(MAX))


def test_comprueba_para_si_falta_una_semilla() -> None:
    row = _tanda(ROW)
    row["afinado"]["series_por_semilla"]["mosca"].pop()
    with pytest.raises(SystemExit, match="mosca"):
        comprueba(row, _tanda(MAX))


def test_comprueba_para_si_los_niveles_no_son_los_mismos() -> None:
    maximo = _tanda(MAX)
    maximo["clonado"]["partidas"][0] = {"particion": "test", "semilla": 1, "nivel": "1"}
    with pytest.raises(SystemExit, match="mismos niveles"):
        comprueba(_tanda(ROW), maximo)


def test_comprueba_para_si_la_6e_trae_modelos_que_no_tocaban() -> None:
    row = _tanda({**ROW, "mlp": [0.5, 0.5, 0.5]})
    with pytest.raises(SystemExit, match="no tocaban"):
        comprueba(row, _tanda(MAX))


def _filas(informe: dict[str, Any], sufijo: str) -> dict[str, Any]:
    return renombra(filas_de_informe(informe, "progreso"), sufijo, CABLEADOS)


@pytest.mark.parametrize("desplazamiento, esperado", [
    (+0.08, "reduce el deficit"), (-0.08, "lo agranda"), (0.0, "sin efecto")])
def test_interaccion(desplazamiento: float, esperado: str) -> None:
    maximo = _informe(MAX, False)
    row = copy.deepcopy(maximo)
    for s in row["series_por_semilla"]["mosca"]:
        s["progreso"] = [v + desplazamiento for v in s["progreso"]]
    fr, fm = _filas(row, "row"), _filas(maximo, "max")
    r = interaccion(fr["mosca_row"], fr["aleatorio_row"], fm["mosca_max"], fm["aleatorio_max"],
                    NIVELES, 2000, 0.025)
    assert r["veredicto"] == esperado
    assert r["interaccion"] == pytest.approx(desplazamiento)
    assert r["deficit_row"] - r["deficit_max"] == pytest.approx(desplazamiento)
