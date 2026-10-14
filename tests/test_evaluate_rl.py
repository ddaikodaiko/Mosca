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
    NIVELES_EXTRA,
    TIPO_DE_NIVEL,
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


def test_los_niveles_extra_no_estan_en_ninguna_particion() -> None:
    usados = {nivel for niveles in STAGES.values() for nivel in niveles}
    assert not set(NIVELES_EXTRA) & usados
    assert len(set(NIVELES_EXTRA)) == len(NIVELES_EXTRA)


def test_todo_nivel_evaluado_tiene_tipo_y_train_es_exterior() -> None:
    """El analisis por tipo cuenta con esto: los no vistos se reparten en exteriores
    (como los de entrenamiento) y atleticos (un tipo que no se entrena)."""
    evaluados = [*STAGES["val"], *STAGES["test"], *NIVELES_EXTRA]
    tipos = [TIPO_DE_NIVEL[f"{m}-{n}"] for m, n in evaluados]
    assert tipos.count("exterior") == 6 and tipos.count("atletico") == 7
    entrenados = {TIPO_DE_NIVEL.get(f"{m}-{n}", "otro") for m, n in STAGES["train"]}
    assert entrenados == {"exterior", "otro"}   # el otro es 1-2, subterraneo


def test_las_partidas_extra_tambien_se_equilibran() -> None:
    partidas = partidas_equilibradas(["extra"], por_nivel=3)
    niveles = [nivel_previsto(split, semilla) for split, semilla in partidas]
    assert sorted(set(niveles)) == sorted(f"{m}-{n}" for m, n in NIVELES_EXTRA)
    assert all(niveles.count(nivel) == 3 for nivel in set(niveles))


def test_las_semillas_de_siempre_no_estan_equilibradas() -> None:
    """Lo que motiva --por-nivel: las semillas 0..15 de test dan 7/4/3/2 partidas."""
    cuenta: dict[str, int] = {}
    for semilla in range(16):
        nivel = nivel_previsto("test", semilla)
        cuenta[nivel] = cuenta.get(nivel, 0) + 1
    assert sorted(cuenta.values()) == [2, 3, 4, 7]


# --- La politica marginal y el reparto entre procesos, en el entorno propio (sin emulador).

def _checkpoint_de_juguete(tmp_path, nombre: str = "mosca"):
    """Un cableado de 40 neuronas guardado como los de la fase 2 y una politica sin entrenar."""
    import torch
    import yaml
    from pathlib import Path

    from envs.platformer import N_ACTIONS
    from models.masked_policy import build_policy
    from models.masked_rnn import Wiring
    from tests.test_masked_rnn import toy_wiring

    cableado = toy_wiring(n_neurons=40, seed=0)
    for control in ("mosca", "aleatorio", "recableado"):
        np.savez(tmp_path / f"{control}.npz", rows=cableado.rows, cols=cableado.cols,
                 synapses=cableado.synapses, sign=cableado.sign)
    cfg = yaml.safe_load(Path("configs/mario_proto.yaml").read_text())
    cfg["wiring_dir"] = str(tmp_path)
    cfg["output_dir"] = str(tmp_path)
    cfg["encoder"] = {"n_sensory": 12, "n_motor": 12, "n_shared": 4}
    cfg["max_episode_steps"] = 60
    wirings = {c: Wiring.load(tmp_path / f"{c}.npz") for c in ("mosca", "aleatorio", "recableado")}
    from evaluate_rl import construye_entorno

    politica = build_policy(nombre, cfg, construye_entorno(cfg).n_features, N_ACTIONS, 0, wirings)
    ruta = tmp_path / f"{nombre}_s0.pt"
    torch.save({"modelo": nombre, "semilla": 0, "config": cfg,
                "state_dict": politica.state_dict()}, ruta)
    return cfg, ruta


def _peticion(cfg, ruta, **cambios):
    from evaluate_rl import niveles_de_test

    peticion = {"cfg": cfg, "device": "cpu", "ruta": ruta,
                "semillas": [("test", s) for s in niveles_de_test(cfg, 3)],
                "repeticiones": 2, "semilla": 0, "muestrea": True, "marginal": True,
                "en_proceso": False}
    peticion.update(cambios)
    return peticion


def test_la_marginal_juega_con_las_frecuencias_del_modelo(tmp_path) -> None:
    from evaluate_rl import evalua_checkpoint

    cfg, ruta = _checkpoint_de_juguete(tmp_path)
    resultado = evalua_checkpoint(_peticion(cfg, ruta))
    frecuencias = np.asarray(resultado["frecuencias"])
    assert frecuencias.sum() == pytest.approx(1.0)
    assert (frecuencias >= 0).all()
    for clave in ("series", "marginal"):
        assert resultado[clave]["progreso"].shape == (3,)
    # Sin --marginal no se juega
    assert "marginal" not in evalua_checkpoint(_peticion(cfg, ruta, marginal=False))


def test_en_procesos_da_lo_mismo_que_en_serie(tmp_path) -> None:
    """La semilla se refija en cada repeticion: el proceso en que cae no cambia nada."""
    import multiprocessing

    from evaluate_rl import evalua_checkpoint

    cfg, ruta = _checkpoint_de_juguete(tmp_path)
    en_serie = evalua_checkpoint(_peticion(cfg, ruta))
    with multiprocessing.get_context("spawn").Pool(2) as pool:
        en_procesos = pool.map(evalua_checkpoint, [_peticion(cfg, ruta, en_proceso=True)] * 2)
    for resultado in en_procesos:
        assert resultado["frecuencias"] == en_serie["frecuencias"]
        for clave in ("series", "marginal"):
            for metrica, valores in en_serie[clave].items():
                assert np.array_equal(resultado[clave][metrica], valores)
