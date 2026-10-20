"""Tests del entorno de plataformas (fase 6).

Lo que se comprueba aqui es lo que, si falla, invalida el experimento entero: que el
nivel dependa solo de la semilla, que los rangos de train/val/test sean disjuntos, y que
la fisica no deje al jugador atravesar el suelo ni saltar mas de lo que dice el calculo.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.platformer import (
    ACTIONS,
    GRAVITY,
    JUMP_VELOCITY,
    N_ACTIONS,
    LevelConfig,
    ObservationConfig,
    PlatformerEnv,
    SeedRanges,
    VectorPlatformer,
    generate_level,
)

SMALL = LevelConfig(width=60, height=12)


def _fraccion_de_metas(accion, release_to_jump: bool = False) -> float:
    """Fraccion de metas de una politica en 20 niveles de test de 160 casillas.

    Los dos tests de margen la comparten: si cada uno tuviera su copia, cambiar los
    niveles o el limite de pasos en uno solo los haria medir cosas distintas sin avisar.
    """
    cfg = LevelConfig(width=160, height=14)
    metas = []
    for seed in range(3_000_000, 3_000_020):
        env = PlatformerEnv(cfg, max_steps=1200, release_to_jump=release_to_jump)
        env.reset(seed)
        done = False
        while not done:
            _, _, done, info = env.step(accion(env))
        metas.append(info["meta"])
    return float(np.mean(metas))


def test_nivel_reproducible_y_distinto_por_semilla() -> None:
    a = generate_level(7, SMALL)
    b = generate_level(7, SMALL)
    c = generate_level(8, SMALL)
    assert np.array_equal(a.tiles, b.tiles)
    assert np.array_equal(a.enemies, b.enemies)
    assert not np.array_equal(a.tiles, c.tiles)


def test_el_nivel_tiene_huecos_y_suelo_en_el_arranque() -> None:
    levels = [generate_level(s, SMALL) for s in range(20)]
    assert any((level.floor >= level.height).any() for level in levels), "ningun hueco"
    for level in levels:
        # El jugador tiene que nacer sobre suelo firme, no sobre un hueco.
        assert level.floor[int(level.start[0])] < level.height


def test_rangos_de_semillas_disjuntos() -> None:
    ranges = SeedRanges()
    spans = [set(range(low, min(high, low + 50))) for low, high in
             (ranges.train, ranges.val, ranges.test)]
    assert ranges.train[1] <= ranges.val[0] and ranges.val[1] <= ranges.test[0]
    assert not spans[0] & spans[1] and not spans[1] & spans[2]


def test_la_gravedad_tira_hacia_abajo_y_el_suelo_para() -> None:
    env = PlatformerEnv(SMALL, max_steps=200)
    env.reset(0)
    floor = env.y
    env.y -= 3.0                      # se le deja caer desde tres casillas
    env.on_ground = False
    for _ in range(60):
        env.step(0)
    assert env.y == pytest.approx(floor, abs=1e-3)
    assert env.on_ground


def test_el_salto_sube_lo_suficiente_y_no_mas_de_lo_teorico() -> None:
    env = PlatformerEnv(SMALL, max_steps=200)
    env.reset(0)
    floor = env.y
    highest = floor
    for step in range(40):
        env.step(6 if step < 25 else 0)   # salta, y luego se le suelta el boton
        highest = min(highest, env.y)
    altura = floor - highest
    teorica = JUMP_VELOCITY ** 2 / (2 * GRAVITY)
    # El generador pone escalones y tuberias de hasta 2 casillas: si el salto no las
    # sube, hay niveles sin solucion. Y no puede pasar de lo que permite la fisica.
    assert 2.2 <= altura <= teorica
    assert env.y == pytest.approx(floor, abs=1e-3), "no vuelve al suelo"


def test_no_se_atraviesa_una_pared_solida() -> None:
    env = PlatformerEnv(SMALL, max_steps=500)
    env.reset(3)
    env.level.tiles[:, 5] = 1         # muro de suelo a techo dentro del llano inicial
    for _ in range(200):
        _, _, done, _ = env.step(2)   # correr a la derecha contra el muro
        assert env.x < 5.0, "ha atravesado el muro"
        assert not done


def test_caer_al_vacio_termina_el_episodio_con_castigo() -> None:
    env = PlatformerEnv(SMALL, max_steps=500)
    env.reset(0)
    env.y = env.level.height + 1      # ya por debajo del nivel
    _, reward, done, info = env.step(0)
    assert done and info["muerte"] and reward < -4


def test_la_observacion_tiene_la_forma_declarada_y_esta_acotada() -> None:
    obs_cfg = ObservationConfig(window_width=11, window_height=9)
    env = PlatformerEnv(SMALL, obs_cfg, max_steps=100)
    obs = env.reset(1)
    assert obs.shape == (obs_cfg.n_features,)
    assert obs.dtype == np.float32
    for _ in range(50):
        obs, *_ = env.step(np.random.randint(N_ACTIONS))
        assert np.isfinite(obs).all() and np.abs(obs).max() <= 2.0


def test_hay_suelo_bajo_los_pies_en_la_observacion_al_empezar() -> None:
    obs_cfg = ObservationConfig()
    env = PlatformerEnv(SMALL, obs_cfg, max_steps=100)
    obs = env.reset(0)
    solid = obs[: obs_cfg.window_width * obs_cfg.window_height].reshape(
        obs_cfg.window_height, obs_cfg.window_width)
    centre_r, centre_c = obs_cfg.window_height // 2, obs_cfg.window_width // 2
    assert solid[centre_r + 1, centre_c] == 1.0, "no ve el suelo que pisa"
    assert solid[centre_r, centre_c] == 0.0, "se ve a si mismo dentro de una pared"


def test_avanzar_a_la_derecha_da_mas_recompensa_que_quedarse_quieto() -> None:
    quieto, corriendo = [], []
    for seed in range(4):
        for action, bucket in ((0, quieto), (2, corriendo)):
            env = PlatformerEnv(SMALL, max_steps=60)
            env.reset(seed)
            total = 0.0
            for _ in range(60):
                _, reward, done, _ = env.step(action)
                total += reward
                if done:
                    break
            bucket.append(total)
    assert np.mean(corriendo) > np.mean(quieto)


def test_el_vector_reinicia_solo_y_no_repite_nivel_entre_entornos() -> None:
    envs = VectorPlatformer(4, "train", SeedRanges(), SMALL, max_steps=20)
    obs = envs.reset()
    assert obs.shape == (4, envs.n_features)
    semillas = {env.level.seed for env in envs.envs}
    assert len(semillas) == 4
    terminados = np.zeros(4, dtype=bool)
    for _ in range(25):
        obs, reward, done, infos = envs.step(np.zeros(4, dtype=np.int64))
        terminados |= done
    assert terminados.all(), "con max_steps=20 tendrian que haber terminado"
    assert {env.level.seed for env in envs.envs} != semillas, "no ha reiniciado con nivel nuevo"


def test_los_niveles_generados_tienen_solucion() -> None:
    """Sin esto, medir aprendizaje no significa nada: puede que el nivel sea imposible."""
    from envs.platformer import scripted_action

    cfg = LevelConfig(width=160, height=14)
    metas = []
    for seed in range(20):
        env = PlatformerEnv(cfg, max_steps=1200)
        env.reset(seed)
        done = False
        while not done:
            _, _, done, info = env.step(scripted_action(env))
        metas.append(info["meta"])
    assert np.mean(metas) >= 0.4, f"la heuristica solo termina el {np.mean(metas):.0%}"


def test_no_hay_huecos_mas_anchos_de_lo_que_cruza_un_salto() -> None:
    cfg = LevelConfig(width=160, height=14)
    for seed in range(30):
        level = generate_level(seed, cfg)
        hueco = level.floor >= level.height
        longitudes = np.diff(np.flatnonzero(np.diff(np.r_[0, hueco.astype(int), 0])))[::2]
        if longitudes.size:
            assert longitudes.max() <= cfg.max_gap


def test_la_tarea_tiene_margen_sobre_la_politica_degenerada() -> None:
    """Si "saltar siempre a la derecha" pasara los niveles, la fase 6 no mediria nada.

    PPO cae solo en esa politica (medido en el prototipo: los seis modelos convergen a la
    accion 3 o la 4 y ahi se quedan). Lo que hace util la comparacion entre cableados es
    la distancia entre esa politica y la heuristica que si mira el nivel.
    """
    from envs.platformer import scripted_action

    corre = _fraccion_de_metas

    degenerada = corre(lambda env: 3)
    heuristica = corre(scripted_action)
    assert degenerada < 0.25, f"saltar sin mirar ya termina el {degenerada:.0%} de los niveles"
    assert heuristica > degenerada + 0.2, "no queda margen que aprender"


def test_las_acciones_cubren_los_cuatro_botones_del_diseno() -> None:
    direcciones = {d for d, _, _ in ACTIONS}
    assert direcciones == {-1, 0, 1}
    assert any(j for _, j, _ in ACTIONS) and any(r for _, _, r in ACTIONS)


def _despegues(env: PlatformerEnv, acciones: list[int]) -> int:
    """Cuantas veces pasa de estar en el suelo a subir."""
    despegues = 0
    for accion in acciones:
        en_suelo = env.on_ground
        env.step(accion)
        despegues += int(en_suelo and env.vy < 0)
    return despegues


def test_soltar_para_saltar_mantener_el_boton_salta_una_sola_vez() -> None:
    """La regla del NES: con el boton mantenido no se rebota en cada aterrizaje."""
    salto_en_el_sitio = 6
    assert ACTIONS[salto_en_el_sitio] == (0, 1, 0)
    sin_regla = PlatformerEnv(SMALL, max_steps=200)
    sin_regla.reset(0)
    con_regla = PlatformerEnv(SMALL, max_steps=200, release_to_jump=True)
    con_regla.reset(0)
    assert _despegues(sin_regla, [salto_en_el_sitio] * 150) > 3
    assert _despegues(con_regla, [salto_en_el_sitio] * 150) == 1


def test_soltar_para_saltar_soltando_un_paso_se_vuelve_a_saltar() -> None:
    env = PlatformerEnv(SMALL, max_steps=200, release_to_jump=True)
    env.reset(0)
    assert _despegues(env, [6] * 40) == 1        # ya ha aterrizado y sigue pulsado
    assert env.on_ground
    assert _despegues(env, [0] + [6] * 5) == 1   # suelta un paso y salta otra vez


def test_soltar_para_saltar_pone_el_boton_en_la_observacion() -> None:
    sin_regla = PlatformerEnv(SMALL, max_steps=50)
    env = PlatformerEnv(SMALL, max_steps=50, release_to_jump=True)
    assert env.n_features == sin_regla.n_features + 1
    assert env.reset(0).shape == (env.n_features,) and env.reset(0)[-1] == 0.0
    assert env.step(6)[0][-1] == 1.0
    assert env.step(0)[0][-1] == 0.0
    vector = VectorPlatformer(2, "train", SeedRanges(), SMALL, max_steps=20, release_to_jump=True)
    assert vector.reset().shape == (2, env.n_features)


def test_soltar_para_saltar_deja_sin_valor_a_la_constante_y_no_a_la_heuristica() -> None:
    """Por eso existe la regla: medido en 100 niveles de 160 casillas, la mejor constante
    pasa de 17 % de metas a 0 % y la heuristica se queda en 53 %."""
    from envs.platformer import scripted_action

    def corre(accion) -> float:
        return _fraccion_de_metas(accion, release_to_jump=True)

    constante = max(corre(lambda env: 3), corre(lambda env: 4))
    heuristica = corre(scripted_action)
    assert constante <= 0.05, f"saltar sin mirar aun termina el {constante:.0%}"
    assert heuristica >= 0.3, f"la heuristica solo termina el {heuristica:.0%}"


def test_from_config_lleva_las_mismas_opciones_al_entorno_y_al_vector() -> None:
    """Clonado, entrenamiento y evaluacion construyen el entorno desde el YAML con
    `from_config`; si una opcion no llegara a uno de los dos, jugarian a juegos distintos."""
    cfg = {"level": {"width": 60, "height": 12}, "max_episode_steps": 77,
           "release_to_jump": True, "level_seeds": {"train": [10, 20]}}
    env = PlatformerEnv.from_config(cfg)
    vector = VectorPlatformer.from_config(cfg, 2, "train")
    for e in (env, *vector.envs):
        assert e.level_cfg == SMALL and e.max_steps == 77 and e.release_to_jump
    assert (vector.low, vector.high) == (10, 20)
    assert vector.n_features == env.n_features
    assert PlatformerEnv.from_config({}).release_to_jump is False
