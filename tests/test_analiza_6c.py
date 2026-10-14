"""Tests del analisis registrado de la fase 6c (`analiza_6c.py`).

La regla de decision se escribio antes de ver las replicas; estos tests fijan que el codigo
hace lo que dice RESULTADOS.md, con casos calculados a mano.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from analiza_6c import (
    Fila,
    bateria,
    contraste,
    filas_de_informe,
    umbrales_holm,
    une,
    veredicto,
)


def test_veredicto() -> None:
    assert veredicto(0.01, 0.025, True, +0.05, 0.02) == "gana"
    assert veredicto(0.01, 0.025, True, -0.05, 0.02) == "pierde"
    # p baja pero semillas mezcladas: no basta
    assert veredicto(0.01, 0.025, False, +0.05, 0.02) == "no resuelto"
    # semillas separadas pero p sobre el umbral de Holm
    assert veredicto(0.03, 0.025, True, +0.05, 0.02) == "no resuelto"
    # ni gana ni pierde, y la diferencia cabe en la dispersion entre semillas
    assert veredicto(0.5, 0.05, False, +0.01, 0.02) == "empate"
    # sin semillas no hay dispersion con que llamarlo empate
    assert veredicto(0.5, 0.05, False, +0.01, float("nan")) == "no resuelto"


@pytest.mark.parametrize("ps, esperados", [
    ([0.01, 0.04], [0.025, 0.05]),
    ([0.03, 0.01], [0.05, 0.025]),
    ([0.03, 0.04], [0.025, 0.0]),     # la menor no pasa: la otra ya no puede
])
def test_holm(ps: list[float], esperados: list[float]) -> None:
    assert umbrales_holm(ps) == pytest.approx(esperados)


def _fila(nombre: str, valores: list[list[float]], niveles: list[str] | None = None) -> Fila:
    array = np.asarray(valores, dtype=float)
    niveles = niveles or [f"{i}-3" for i in range(1, array.shape[1] + 1)]
    return Fila(nombre, niveles, array, [f"{nombre}_s{i}" for i in range(array.shape[0])])


def test_contraste_a_mano() -> None:
    a = _fila("mosca", [[0.3, 0.4, 0.5, 0.6], [0.4, 0.5, 0.6, 0.7], [0.5, 0.6, 0.7, 0.8]])
    b = _fila("aleatorio", [[0.1, 0.1, 0.1, 0.1]] * 3)
    c = contraste(a, b, a.niveles)
    assert c["diferencia"] == pytest.approx(0.45)      # medias por nivel 0,4..0,7 - 0,1
    assert c["niveles_a_favor"] == 4
    assert c["p_nivel"] == pytest.approx(2 / 16)          # todos a favor, 4 niveles
    assert c["separadas"]
    assert c["p_semilla"] == pytest.approx(0.1)            # 3 contra 3, separadas
    # dispersion: desviacion de [0.45, 0.55, 0.65] = 0.1; la del aleatorio, 0
    assert c["dispersion"] == pytest.approx(0.05)


def test_contra_una_referencia_separadas_es_estar_todas_del_mismo_lado() -> None:
    a = _fila("mosca", [[0.2, 0.2], [0.3, 0.3], [0.05, 0.05]])
    azar = _fila("azar", [[0.1, 0.1]])
    c = contraste(a, azar, a.niveles)
    assert not c["separadas"]
    assert "p_semilla" not in c


def test_la_bateria_aplica_holm_solo_a_los_principales() -> None:
    niveles = [f"{i}-3" for i in range(1, 8)]
    alto = [[0.5] * 7, [0.6] * 7, [0.7] * 7]
    bajo = [[0.1] * 7, [0.2] * 7, [0.3] * 7]
    filas = {n: _fila(n, v, niveles) for n, v in
             [("mosca", alto), ("aleatorio", bajo), ("recableado", bajo), ("mlp", bajo)]}
    filas["azar"] = _fila("azar", [[0.1] * 7], niveles)
    resultado = bateria(filas, niveles, con_holm=True)
    umbrales = sorted(c["umbral"] for c in resultado["principales"])
    assert umbrales == pytest.approx([0.025, 0.05])
    assert all(c["veredicto"] == "gana" for c in resultado["principales"])
    assert all(c["umbral"] == 0.05 for c in resultado["secundarios"])


def _informe(niveles_partida: list[str], semillas: dict[str, list[list[float]]],
             referencias: dict[str, list[float]]) -> dict:
    return {
        "muestreada": True,
        "partidas": [{"particion": "x", "semilla": i, "nivel": n} for i, n in enumerate(niveles_partida)],
        "series_por_semilla": {
            nombre: [{"checkpoint": f"{nombre}_s{i}", "progreso": s, "meta": [0.0] * len(s)}
                     for i, s in enumerate(series)]
            for nombre, series in semillas.items()},
        "series_referencias": {n: {"progreso": s, "meta": [0.0] * len(s)} for n, s in referencias.items()},
    }


def test_de_partidas_a_medias_por_nivel_y_union_de_informes(tmp_path) -> None:
    principal = _informe(["1-3", "1-3", "8-1", "8-1"],
                         {"mosca": [[0.1, 0.3, 0.5, 0.7], [0.2, 0.2, 0.4, 0.4]]},
                         {"azar": [0.1, 0.1, 0.2, 0.2]})
    extra = _informe(["3-2", "3-2"], {"mosca": [[0.6, 0.8], [0.5, 0.5]]}, {"azar": [0.3, 0.3]})
    filas = filas_de_informe(principal, "progreso")
    assert filas["mosca"].niveles == ["1-3", "8-1"]
    assert np.allclose(filas["mosca"].valores, [[0.2, 0.6], [0.2, 0.4]])
    assert filas["azar"].valores.shape == (1, 2)
    todas = une(filas, filas_de_informe(extra, "progreso"))
    assert todas["mosca"].niveles == ["1-3", "8-1", "3-2"]
    assert todas["mosca"].valores[:, 2].tolist() == pytest.approx([0.7, 0.5])
    with pytest.raises(ValueError):
        une(filas, filas)                               # mismos niveles dos veces
    (tmp_path / "p.json").write_text(json.dumps(principal))   # serializable tal cual
