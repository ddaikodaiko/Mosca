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
    JugadorHeuristico,
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


def test_la_heuristica_se_pasa_el_1_1() -> None:
    """Es la maestra del clonado de la demo: si deja de pasarse el 1-1, la mosca clonada
    aprende de una maestra que no sabe jugar. Fija tambien las dos trampas que costaron
    encontrar: soltar A al caer y no tomar el borde de una tuberia por un hoyo."""
    env = NesMarioEnv("train", ObservationConfig(), NesConfig(stage=[1, 1], random_noop=0))
    env.reset(0)
    jugador, done = JugadorHeuristico(), False
    while not done:
        _, _, done, info = env.step(jugador(env))
    env.close()
    assert info["meta"], info



# --- Fase 6c: que aprenda sola ------------------------------------------------------


def test_sin_tocar_nada_la_observacion_no_cambia() -> None:
    """Las opciones de la fase 6c vienen apagadas: 302 numeros, como antes."""
    env = NesMarioEnv("train", ObservationConfig(), RAPIDO)
    assert env.reset(0).shape == (ObservationConfig().n_features,)
    env.close()


def test_velocidad_vertical_y_boton_a() -> None:
    cfg = NesConfig(max_steps=40, random_noop=0, stage=(1, 1), motion_features=True)
    env = NesMarioEnv("train", ObservationConfig(), cfg)
    obs = env.reset(0)
    assert obs.shape == (env.n_features,) == (ObservationConfig().n_features + 2,)
    assert obs[-2] == 0.0 and obs[-1] == 0.0            # quieto en el suelo, A suelto
    obs, *_ = env.step(2)                               # derecha + A: salta
    assert obs[-2] < 0.0 and obs[-1] == 1.0             # subiendo, A pulsado
    for _ in range(8):
        obs, *_ = env.step(1)                           # suelta A
    assert obs[-1] == 0.0
    env.close()


def _partida_hasta_morir(env: NesMarioEnv, semilla: int) -> None:
    env.reset(semilla)
    done = False
    while not done:
        _, _, done, _ = env.step(3)                      # correr sin saltar: muere pronto


def test_la_frontera_arranca_en_el_estado_exacto_de_la_mejor_partida() -> None:
    cfg = NesConfig(max_steps=300, random_noop=5, stage=(1, 1), frontier_probability=1.0,
                    frontier_rewind=5)
    env = NesMarioEnv("train", ObservationConfig(), cfg)
    _partida_hasta_morir(env, 0)
    mejor = env._mejor[(1, 1)]
    obs = env.reset(1)
    assert env.desde_frontera
    inicio = env.steps
    assert inicio == int(np.argmax(mejor["xs"])) + 1 - 5

    # La misma partida rejugada a mano desde la salida tiene que dar el mismo estado.
    testigo = NesMarioEnv("train", ObservationConfig(), NesConfig(
        max_steps=300, random_noop=0, stage=(1, 1)))
    testigo.reset(0)
    for _ in range(mejor["espera"]):
        testigo.env.step(0)
    for accion in mejor["acciones"][:inicio]:
        testigo.step(accion)
    np.testing.assert_array_equal(env.env.unwrapped.ram, testigo.env.unwrapped.ram)
    assert env.max_x == max(mejor["xs"][:inicio])
    np.testing.assert_array_equal(obs[:-5], testigo._observe()[:-5])  # escalares: max_x/pasos
    env.close()
    testigo.close()


def test_la_frontera_nunca_se_usa_fuera_de_entrenamiento() -> None:
    cfg = NesConfig(max_steps=300, random_noop=0, frontier_probability=1.0)
    for particion in ("val", "test"):
        env = NesMarioEnv(particion, ObservationConfig(), cfg)
        _partida_hasta_morir(env, 0)
        env.reset(1)
        assert not env.desde_frontera and env.steps == 0 and not env._mejor
        env.close()


def test_en_paralelo_juega_lo_mismo_que_en_serie() -> None:
    """Mismas semillas -> mismos niveles y mismas observaciones, con o sin procesos."""
    from envs.nes_mario import VectorNesMarioParalelo

    cfg = NesConfig(max_steps=15, random_noop=3)
    serie = VectorNesMario(3, "train", ObservationConfig(), cfg)
    paralelo = VectorNesMarioParalelo(3, "train", ObservationConfig(), cfg)
    np.testing.assert_array_equal(serie.reset(), paralelo.reset())
    rng = np.random.default_rng(0)
    for _ in range(40):                                  # cruza varios reinicios
        acciones = rng.integers(0, N_NES_ACTIONS, size=3)
        a, b = serie.step(acciones), paralelo.step(acciones)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[2], b[2])
        assert [i["nivel"] for i in a[3]] == [i["nivel"] for i in b[3]]
    paralelo.close()
