"""Tests de la estadistica de la fase 6c: pruebas exactas por nivel y por semilla.

En el NES el test son cuatro niveles. Si la unidad del contraste no es el nivel, la p sale
inflada; y si una prueba no puede bajar de cierta p, hay que saberlo antes de leer un
"no significativo" como empate. Todo con casos que se calculan a mano.
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pytest

from envs.nes_mario import STAGES
from evaluate_rl import (
    media_por_nivel,
    nivel_previsto,
    partidas_equilibradas,
    permutacion_exacta_entre_grupos,
    permutacion_exacta_pareada,
)


def test_todas_las_diferencias_del_mismo_signo_dan_la_p_minima() -> None:
    """Con 4 niveles, el mejor caso posible es 2 de 16 asignaciones: p = 0,125."""
    prueba = permutacion_exacta_pareada(np.array([0.1, 0.2, 0.05, 0.3]))
    assert prueba["p"] == pytest.approx(0.125)
    assert prueba["p_minima_posible"] == pytest.approx(0.125)
    assert prueba["diferencia_media"] == pytest.approx(0.1625)


def test_sin_diferencias_la_p_es_uno() -> None:
    assert permutacion_exacta_pareada(np.zeros(5))["p"] == pytest.approx(1.0)


def test_la_pareada_exacta_coincide_con_contarlas_a_mano() -> None:
    diferencias = np.array([0.3, -0.1, 0.2, 0.05, -0.02, 0.15])
    observado = abs(diferencias.mean())
    extremas = sum(
        abs((np.array(signos) * diferencias).mean()) >= observado - 1e-12
        for signos in product((-1.0, 1.0), repeat=diferencias.size)
    )
    assert permutacion_exacta_pareada(diferencias)["p"] == pytest.approx(extremas / 64)


def test_la_pareada_exacta_no_acepta_demasiadas_unidades() -> None:
    with pytest.raises(ValueError):
        permutacion_exacta_pareada(np.ones(21))


def test_tres_contra_tres_separados_dan_la_p_minima() -> None:
    """Las 20 particiones de 6 en 3+3: solo 2 son tan extremas. p = 0,1."""
    prueba = permutacion_exacta_entre_grupos(np.array([0.5, 0.6, 0.7]), np.array([0.1, 0.2, 0.3]))
    assert prueba["p"] == pytest.approx(0.1)
    assert prueba["p_minima_posible"] == pytest.approx(0.1)
    assert prueba["diferencia_media"] == pytest.approx(0.4)


def test_grupos_iguales_dan_p_uno() -> None:
    valores = np.array([0.2, 0.3, 0.4])
    assert permutacion_exacta_entre_grupos(valores, valores.copy())["p"] == pytest.approx(1.0)


def test_entre_grupos_coincide_con_contarlas_a_mano() -> None:
    a, b = np.array([0.31, 0.22, 0.40]), np.array([0.25, 0.18, 0.35, 0.27])
    todos = np.concatenate([a, b])
    observado = abs(a.mean() - b.mean())
    extremas = 0
    particiones = list(combinations(range(7), 3))
    for indices in particiones:
        dentro = np.isin(np.arange(7), indices)
        extremas += abs(todos[dentro].mean() - todos[~dentro].mean()) >= observado - 1e-12
    prueba = permutacion_exacta_entre_grupos(a, b)
    assert prueba["p"] == pytest.approx(extremas / len(particiones))
    assert prueba["p_minima_posible"] == pytest.approx(1 / 35)


def test_media_por_nivel() -> None:
    valores = np.array([1.0, 3.0, 10.0, 0.0, 2.0])
    niveles = np.array(["8-1", "8-1", "2-3", "4-3", "4-3"])
    etiquetas, medias = media_por_nivel(valores, niveles)
    assert etiquetas == ["2-3", "4-3", "8-1"]
    assert medias.tolist() == [10.0, 1.0, 2.0]


@pytest.mark.parametrize("splits", [["test"], ["val", "test"]])
def test_las_partidas_equilibradas_reparten_igual_entre_niveles(splits: list[str]) -> None:
    partidas = partidas_equilibradas(splits, por_nivel=5)
    niveles = [nivel_previsto(split, semilla) for split, semilla in partidas]
    esperados = {f"{m}-{n}" for split in splits for m, n in STAGES[split]}
    assert set(niveles) == esperados
    assert all(niveles.count(nivel) == 5 for nivel in esperados)
    assert len(set(partidas)) == len(partidas)


def test_las_semillas_de_siempre_no_estan_equilibradas() -> None:
    """Lo que motiva --por-nivel: las semillas 0..15 de test dan 7/4/3/2 partidas."""
    cuenta: dict[str, int] = {}
    for semilla in range(16):
        nivel = nivel_previsto("test", semilla)
        cuenta[nivel] = cuenta.get(nivel, 0) + 1
    assert sorted(cuenta.values()) == [2, 3, 4, 7]
