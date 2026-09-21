"""Tests del entorno de NES (fase 6b).

Se saltan enteros si no esta instalado `gym-super-mario-bros`, porque el resto del repo
no depende de el. Lo que vigilan es lo que invalidaria el experimento: que la
observacion sea intercambiable con la del entorno propio, que el mapa de casillas de la
RAM este bien calibrado, y que los conjuntos de niveles no se solapen.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.platformer import N_ACTIONS, ObservationConfig, PlatformerEnv, LevelConfig

gym_super_mario_bros = pytest.importorskip(
    "gym_super_mario_bros", reason="falta gym-super-mario-bros (pip install gym-super-mario-bros)"
)

from envs.nes_mario import (  # noqa: E402
    N_NES_ACTIONS,
    STAGES,
    NesConfig,
    NesMarioEnv,
    VectorNesMario,
)

RAPIDO = NesConfig(max_steps=40, random_noop=0)


def test_la_observacion_es_intercambiable_con_la_del_entorno_propio() -> None:
    """La politica del conectoma tiene que servir en los dos entornos sin tocar nada."""
    obs_cfg = ObservationConfig()
    nes = NesMarioEnv("train", obs_cfg, RAPIDO)
    propio = PlatformerEnv(LevelConfig(width=60, height=12), obs_cfg, max_steps=40)
    a = nes.reset(0)
    b = propio.reset(0)
    assert a.shape == b.shape == (obs_cfg.n_features,)
    assert a.dtype == b.dtype == np.float32
    assert N_NES_ACTIONS == N_ACTIONS
    nes.close()


def test_el_mapa_de_casillas_esta_calibrado_contra_la_ram() -> None:
    """De pie: aire donde esta Mario y suelo justo debajo. Si esto falla, el modelo ve
    el nivel desplazado y no hay forma de notarlo mirando el video."""
    env = NesMarioEnv("train", ObservationConfig(), RAPIDO)
    env.stages = ((1, 1),)
    env.reset(0)
    for _ in range(6):
        env.step(1)
    assert env.on_ground, "deberia estar en el suelo al empezar 1-1"
    assert env._tile(env.x, env.y) == 0, "Mario esta dentro de una casilla solida"
    assert env._tile(env.x, env.y + 16) != 0, "no hay suelo bajo sus pies"
    env.close()


def test_la_ventana_situa_el_suelo_justo_debajo_del_centro() -> None:
    obs_cfg = ObservationConfig(window_width=11, window_height=9)
    env = NesMarioEnv("train", obs_cfg, RAPIDO)
    env.stages = ((1, 1),)
    obs = env.reset(0)
    solido = obs[: obs_cfg.window_width * obs_cfg.window_height].reshape(
        obs_cfg.window_height, obs_cfg.window_width)
    fila, columna = obs_cfg.window_height // 2, obs_cfg.window_width // 2
    assert solido[fila + 1, columna] == 1.0
    assert solido[fila, columna] == 0.0
    env.close()


def test_los_conjuntos_de_niveles_son_disjuntos() -> None:
    """SMB tiene 32 niveles fijos: sin esto, se mide sobre lo que se ha entrenado."""
    train, val, test = (set(STAGES[k]) for k in ("train", "val", "test"))
    assert not train & val and not train & test and not val & test


def test_la_semilla_fija_el_nivel_y_la_espera_inicial() -> None:
    cfg = NesConfig(max_steps=20, random_noop=20)
    a = NesMarioEnv("train", ObservationConfig(), cfg)
    b = NesMarioEnv("train", ObservationConfig(), cfg)
    assert np.array_equal(a.reset(7), b.reset(7))
    assert a.stage == b.stage
    distinto = a.reset(8)
    assert not np.array_equal(distinto, b.reset(7)) or a.stage != b.stage
    a.close(); b.close()


def test_morir_termina_el_episodio_y_castiga() -> None:
    env = NesMarioEnv("train", ObservationConfig(), NesConfig(max_steps=400, random_noop=0))
    env.stages = ((1, 1),)
    env.reset(0)
    recompensa_final = 0.0
    for _ in range(400):
        _, recompensa_final, done, info = env.step(1)   # derecha: muere con el primer Goomba
        if done:
            break
    assert done and info["muerte"] and recompensa_final < 0
    env.close()


def test_el_corte_por_atasco_termina_el_episodio() -> None:
    env = NesMarioEnv("train", ObservationConfig(),
                      NesConfig(max_steps=400, random_noop=0, stuck_steps=10))
    env.stages = ((1, 1),)
    env.reset(0)
    for paso in range(200):
        _, _, done, info = env.step(0)    # quieto: no avanza nunca
        if done:
            break
    assert info["atascado"] and paso < 30, "no ha cortado por atasco"
    env.close()


def test_el_vector_reinicia_solo_y_no_repite_semilla() -> None:
    envs = VectorNesMario(2, "train", ObservationConfig(), NesConfig(max_steps=8, random_noop=0))
    obs = envs.reset()
    assert obs.shape == (2, envs.n_features)
    terminados = np.zeros(2, dtype=bool)
    for _ in range(12):
        obs, recompensa, done, infos = envs.step(np.ones(2, dtype=np.int64))
        terminados |= done
    assert terminados.all()
    assert envs.counters.tolist() != [0, 1], "no ha consumido semillas nuevas"
    for env in envs.envs:
        env.close()
