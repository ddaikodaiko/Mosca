"""Tests de la grabacion de partidas humanas (fase 6c).

Vigilan lo que estropearia en silencio el aprendizaje por imitacion: que un boton se
traduzca a la accion equivocada, y que lo que se guarda no sea exactamente lo que la red
ve al jugar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gym_super_mario_bros", reason="falta gym-super-mario-bros")

from envs.nes_mario import NesConfig, NesMarioEnv  # noqa: E402
from envs.platformer import ObservationConfig  # noqa: E402
from grabar_humano import accion_de_teclas, carga_demos, graba_partida  # noqa: E402


@pytest.mark.parametrize(
    ("izquierda", "derecha", "a", "b", "esperada"),
    [
        (False, False, False, False, 0),
        (False, True, False, False, 1),
        (False, True, True, False, 2),
        (False, True, False, True, 3),
        (False, True, True, True, 4),
        (False, False, True, False, 5),
        (True, False, False, False, 6),
        (True, False, True, False, 5),   # izquierda+A no existe: pesa el salto
        (True, True, False, False, 6),   # las dos flechas: no avanza
        (False, False, False, True, 0),  # B sola no hace nada en SMB
    ],
)
def test_botones_a_simple_movement(izquierda: bool, derecha: bool, a: bool, b: bool,
                                   esperada: int) -> None:
    assert accion_de_teclas(izquierda, derecha, a, b) == esperada


def test_lo_grabado_es_lo_que_ve_la_red(tmp_path: Path) -> None:
    """Rejugar las acciones grabadas con la misma semilla da las mismas observaciones."""
    cfg = NesConfig(max_steps=30, random_noop=5, stage=(1, 1))
    env = NesMarioEnv("train", ObservationConfig(), cfg)
    secuencia = iter([1, 3, 4, 4, 0, 2, 5, 6] * 10)
    resumen = graba_partida(env, 7, lambda: next(secuencia), tmp_path / "p.npz", {})
    [partida] = carga_demos(tmp_path)
    assert partida["resumen"] == resumen
    assert partida["obs"].shape == (resumen["pasos"], ObservationConfig().n_features)

    observacion = env.reset(7)
    for paso, accion in enumerate(partida["acciones"]):
        np.testing.assert_array_equal(observacion, partida["obs"][paso])
        observacion, _, _, _ = env.step(int(accion))
    env.close()


def test_carga_filtra_por_nivel_y_progreso(tmp_path: Path) -> None:
    env = NesMarioEnv("train", ObservationConfig(), NesConfig(max_steps=10, random_noop=0,
                                                               stage=(1, 1)))
    graba_partida(env, 0, lambda: 1, tmp_path / "a.npz", {})
    env.close()
    assert len(carga_demos(tmp_path, {"1-1"})) == 1
    assert carga_demos(tmp_path, {"2-1"}) == []
    assert carga_demos(tmp_path, progreso_minimo=0.9) == []
