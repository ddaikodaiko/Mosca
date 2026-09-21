"""Plataformas procedurales: el entorno de la tarea "Mario" (seccion 13 del diseno).

No es un emulador ni una ROM: es un motor de plataformas propio de ~300 lineas, con
estado simbolico en vez de pixeles, como exige la nota de viabilidad del diseno. Las
razones estan ahi: aprender desde imagen multiplica el coste por ordenes de magnitud, y
usar una ROM comercial es un problema de derechos.

Tres cosas que el diseno impone y aqui se cumplen:

1. **Niveles generados aleatoriamente.** Cada episodio saca su nivel de una semilla. Con
   un nivel fijo la red memoriza la secuencia de saltos y la metrica deja de medir nada.
2. **Semillas de nivel en rangos disjuntos** para train / val / test (regla 2 del repo):
   quien las reparte es `SeedRanges`.
3. **Observacion simbolica**: ventana de casillas alrededor del jugador (solido, enemigo,
   vacio) mas cinco escalares. Ni un pixel.

Unidades: casillas y fotogramas. `y` crece hacia ABAJO, como las filas de la matriz de
casillas. El jugador es una caja de 0,8 x 0,9 casillas con `x` en su centro e `y` en sus
pies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Fisica. Calibrada midiendo, no en el papel: el salto sube 2,52 casillas y en el aire
# recorre 5,4 a la carrera, asi que pasa huecos de hasta 3 y sube escalones y tuberias de
# 2 (que es el maximo que genera el nivel). Con JUMP_VELOCITY = 0.50 subia 1,84 y los
# escalones de 2 eran muros infranqueables: media generacion de niveles sin solucion.
GRAVITY = 0.06
JUMP_VELOCITY = 0.58
MAX_FALL = 0.50
WALK_SPEED = 0.18
RUN_SPEED = 0.28
ACCELERATION = 0.045
FRICTION = 0.025
ENEMY_SPEED = 0.045
STOMP_BOUNCE = 0.30

PLAYER_HALF_WIDTH = 0.4
PLAYER_HEIGHT = 0.9
EPS = 1e-4

# (direccion, salto, carrera). Los cuatro botones del diseno, en las 7 combinaciones que
# tienen sentido: quieto, andar/correr a la derecha con y sin salto, izquierda, salto seco.
ACTIONS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
    (-1, 0, 0),
    (0, 1, 0),
)
N_ACTIONS = len(ACTIONS)


@dataclass(frozen=True)
class LevelConfig:
    width: int = 160
    height: int = 14
    gap_probability: float = 0.25
    step_probability: float = 0.30
    pipe_probability: float = 0.12
    ceiling_probability: float = 0.0
    enemy_probability: float = 0.05
    max_gap: int = 3


@dataclass(frozen=True)
class Level:
    """Un nivel ya generado. `tiles[r, c] = 1` es solido."""

    tiles: np.ndarray          # uint8 (height, width)
    floor: np.ndarray          # int32 (width,) fila superior del suelo; height si es hueco
    enemies: np.ndarray        # float32 (n_enemies, 2) posicion inicial (x, y_pies)
    start: tuple[float, float]
    goal_x: float
    seed: int

    @property
    def height(self) -> int:
        return int(self.tiles.shape[0])

    @property
    def width(self) -> int:
        return int(self.tiles.shape[1])


def generate_level(seed: int, cfg: LevelConfig = LevelConfig()) -> Level:
    """Nivel procedural reproducible: misma semilla, mismas casillas y mismos enemigos."""
    rng = np.random.default_rng(seed)
    width, height = cfg.width, cfg.height
    floor = np.full(width, height, dtype=np.int32)   # height = hueco sin suelo
    tiles = np.zeros((height, width), dtype=np.uint8)

    base = height - 3
    level_floor = base
    column = 0
    safe_start, safe_end = 6, 8   # arranque y meta siempre en llano
    floor[:safe_start] = level_floor
    column = safe_start

    after_gap = False
    while column < width - safe_end:
        roll = rng.random()
        remaining = width - safe_end - column
        # Dos huecos seguidos darian un vano de hasta 6 casillas, y el salto solo cruza
        # 5,4 al maximo: seria un nivel sin solucion. Tras un hueco va siempre llano.
        if roll < cfg.gap_probability and remaining > cfg.max_gap + 4 and not after_gap:
            length = int(rng.integers(1, cfg.max_gap + 1))
            column += length                      # floor se queda en `height`: hueco
            after_gap = True
            continue
        length = int(min(rng.integers(4, 10), remaining))
        # Y el aterrizaje tras el hueco mantiene la altura: cruzar un vano de 3 y ademas
        # subir 2 casillas no cabe en un solo salto.
        if roll < cfg.gap_probability + cfg.step_probability and not after_gap:
            level_floor = int(np.clip(level_floor + rng.choice([-2, -1, 1, 2]), 4, height - 2))
        after_gap = False
        floor[column:column + length] = level_floor
        if rng.random() < cfg.pipe_probability and length >= 5:
            pipe_column = column + int(rng.integers(2, length - 1))
            pipe_height = int(rng.integers(1, 3))
            tiles[level_floor - pipe_height:level_floor, pipe_column] = 1
            floor[pipe_column] = level_floor - pipe_height
        elif rng.random() < cfg.ceiling_probability and length >= 6 and level_floor >= 6:
            # Techo bajo: se pasa corriendo por debajo, pero si vas saltando te das en
            # la cabeza y pierdes el impulso. Se anadio para castigar la politica
            # degenerada "saltar siempre a la derecha", pero MEDIDO no la castiga: con
            # 30 % de techos pasa de 0,381 a 0,420 de progreso y la heuristica baja de
            # 0,752 a 0,720. Por eso viene apagado por defecto. Queda el mando porque el
            # nivel gana variedad, no porque arregle nada.
            ceiling = level_floor - 3
            tiles[ceiling, column + 1:column + length - 1] = 1
        column += length

    floor[width - safe_end:] = level_floor
    for c in range(width):
        if floor[c] < height:
            tiles[floor[c]:, c] = 1

    # Enemigos: solo en llano, lejos del arranque y con suelo a los lados para patrullar.
    positions = []
    for c in range(safe_start + 6, width - safe_end):
        if rng.random() >= cfg.enemy_probability:
            continue
        if floor[c] >= height or floor[c - 1] != floor[c] or floor[c + 1] != floor[c]:
            continue
        positions.append((c + 0.5, float(floor[c])))
    enemies = np.asarray(positions, dtype=np.float32).reshape(-1, 2)

    start = (2.5, float(floor[2]))
    return Level(tiles, floor, enemies, start, float(width - 3), int(seed))


@dataclass
class ObservationConfig:
    """Ventana de casillas alrededor del jugador. Tres canales: solido, enemigo, vacio."""

    window_width: int = 11
    window_height: int = 9

    @property
    def n_features(self) -> int:
        return 3 * self.window_width * self.window_height + 5


@dataclass
class RewardConfig:
    progress: float = 1.0      # por casilla de avance (solo el maximo alcanzado)
    step_cost: float = 0.02
    death: float = 5.0
    goal: float = 10.0
    stomp: float = 1.0


class PlatformerEnv:
    """Un episodio de plataformas. Sin dependencias de gym: `reset` / `step` y ya."""

    def __init__(
        self,
        level_cfg: LevelConfig = LevelConfig(),
        obs_cfg: ObservationConfig = ObservationConfig(),
        reward_cfg: RewardConfig = RewardConfig(),
        max_steps: int = 1200,
    ) -> None:
        self.level_cfg = level_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.max_steps = max_steps
        self.level: Level | None = None

    @property
    def n_features(self) -> int:
        return self.obs_cfg.n_features

    def reset(self, seed: int) -> np.ndarray:
        self.level = generate_level(seed, self.level_cfg)
        self.x, self.y = self.level.start
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = True
        self.steps = 0
        self.max_x = self.x
        self.enemy_pos = self.level.enemies.copy()
        self.enemy_dir = np.where(
            np.arange(self.enemy_pos.shape[0]) % 2 == 0, -1.0, 1.0
        ).astype(np.float32)
        self.enemy_alive = np.ones(self.enemy_pos.shape[0], dtype=bool)
        return self._observe()

    # -- fisica ----------------------------------------------------------------

    def _solid(self, row: int, column: int) -> bool:
        level = self.level
        assert level is not None
        if column < 0 or column >= level.width:
            return column < 0           # pared invisible a la izquierda; la meta, abierta
        if row < 0:
            return False                # se puede salir por arriba
        if row >= level.height:
            return False                # el vacio no frena: se cae y se muere
        return bool(level.tiles[row, column])

    def _overlapping(self, x: float, y: float) -> list[tuple[int, int]]:
        left = int(np.floor(x - PLAYER_HALF_WIDTH + EPS))
        right = int(np.floor(x + PLAYER_HALF_WIDTH - EPS))
        top = int(np.floor(y - PLAYER_HEIGHT + EPS))
        bottom = int(np.floor(y - EPS))
        return [
            (r, c)
            for c in range(left, right + 1)
            for r in range(top, bottom + 1)
            if self._solid(r, c)
        ]

    def _move(self, direction: int, jump: bool, run: bool) -> None:
        target = (RUN_SPEED if run else WALK_SPEED) * direction
        if direction == 0:
            self.vx -= np.sign(self.vx) * min(FRICTION, abs(self.vx))
        else:
            self.vx += np.clip(target - self.vx, -ACCELERATION, ACCELERATION)
        if jump and self.on_ground:
            self.vy = -JUMP_VELOCITY
            self.on_ground = False
        elif not jump and self.vy < 0:
            self.vy += GRAVITY          # salto variable: soltar el boton corta la subida
        self.vy = min(self.vy + GRAVITY, MAX_FALL)

        self.x += self.vx
        for _, column in self._overlapping(self.x, self.y):
            self.x = (column - PLAYER_HALF_WIDTH - EPS) if self.vx > 0 else (
                column + 1 + PLAYER_HALF_WIDTH + EPS)
            self.vx = 0.0
            break

        self.y += self.vy
        self.on_ground = False
        hits = self._overlapping(self.x, self.y)
        if hits:
            rows = [row for row, _ in hits]
            if self.vy > 0:
                self.y = float(min(rows))
                self.on_ground = True
            else:
                self.y = float(max(rows) + 1 + PLAYER_HEIGHT + EPS)
            self.vy = 0.0

    def _move_enemies(self) -> None:
        level = self.level
        assert level is not None
        for i in np.flatnonzero(self.enemy_alive):
            x = self.enemy_pos[i, 0] + ENEMY_SPEED * self.enemy_dir[i]
            column = int(np.floor(x))
            ahead = int(np.floor(x + 0.5 * self.enemy_dir[i]))
            blocked = self._solid(int(self.enemy_pos[i, 1]) - 1, ahead)
            ledge = not self._solid(int(self.enemy_pos[i, 1]), ahead)
            if blocked or ledge or column < 0 or column >= level.width:
                self.enemy_dir[i] *= -1.0
            else:
                self.enemy_pos[i, 0] = x

    def _collide_enemies(self) -> tuple[int, bool]:
        """Devuelve (enemigos pisados, muerte por contacto)."""
        stomped = 0
        for i in np.flatnonzero(self.enemy_alive):
            ex, ey = self.enemy_pos[i]
            if abs(ex - self.x) > PLAYER_HALF_WIDTH + 0.4:
                continue
            if not (ey - 0.8 < self.y and self.y - PLAYER_HEIGHT < ey):
                continue
            if self.vy > 0 and self.y - ey < 0.45:
                self.enemy_alive[i] = False
                self.vy = -STOMP_BOUNCE
                self.on_ground = False
                stomped += 1
            else:
                return stomped, True
        return stomped, False

    # -- API -------------------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        level = self.level
        assert level is not None, "hay que llamar a reset antes que a step"
        direction, jump, run = ACTIONS[int(action)]
        self._move(direction, bool(jump), bool(run))
        self._move_enemies()
        stomped, touched = self._collide_enemies()
        self.steps += 1

        reward = -self.reward_cfg.step_cost + self.reward_cfg.stomp * stomped
        if self.x > self.max_x:
            reward += self.reward_cfg.progress * (self.x - self.max_x)
            self.max_x = self.x

        fell = self.y > level.height
        reached = self.x >= level.goal_x
        done = bool(fell or touched or reached or self.steps >= self.max_steps)
        if fell or touched:
            reward -= self.reward_cfg.death
        if reached:
            reward += self.reward_cfg.goal

        info = {
            "progreso": float(self.max_x / level.goal_x),
            "meta": bool(reached),
            "muerte": bool(fell or touched),
            "pasos": self.steps,
            "pisados": stomped,
        }
        return self._observe(), float(reward), done, info

    def _observe(self) -> np.ndarray:
        level = self.level
        assert level is not None
        cfg = self.obs_cfg
        half_w, half_h = cfg.window_width // 2, cfg.window_height // 2
        column = int(np.floor(self.x))
        row = int(np.floor(self.y - EPS))

        columns = np.arange(column - half_w, column + half_w + 1)
        rows = np.arange(row - half_h, row + half_h + 1)
        inside = (
            (columns[None, :] >= 0) & (columns[None, :] < level.width)
            & (rows[:, None] >= 0) & (rows[:, None] < level.height)
        )
        safe_c = np.clip(columns, 0, level.width - 1)
        safe_r = np.clip(rows, 0, level.height - 1)
        solid = level.tiles[np.ix_(safe_r, safe_c)].astype(np.float32) * inside
        # Fuera del nivel por los lados es pared; por abajo y por arriba, vacio.
        solid = np.where(
            (columns[None, :] < 0) & np.ones_like(solid, dtype=bool), 1.0, solid
        ).astype(np.float32)

        enemies = np.zeros_like(solid)
        for i in np.flatnonzero(self.enemy_alive):
            ex, ey = self.enemy_pos[i]
            dc = int(np.floor(ex)) - columns[0]
            dr = int(np.floor(ey - EPS)) - rows[0]
            if 0 <= dr < cfg.window_height and 0 <= dc < cfg.window_width:
                enemies[dr, dc] = 1.0

        void = (1.0 - inside.astype(np.float32))
        scalars = np.array(
            [
                self.vx / RUN_SPEED,
                self.vy / MAX_FALL,
                1.0 if self.on_ground else 0.0,
                (level.goal_x - self.x) / level.goal_x,
                1.0 - self.steps / self.max_steps,
            ],
            dtype=np.float32,
        )
        return np.concatenate([solid.ravel(), enemies.ravel(), void.ravel(), scalars])


@dataclass
class SeedRanges:
    """Rangos DISJUNTOS de semillas de nivel (regla 2). Nunca se solapan train/val/test."""

    train: tuple[int, int] = (0, 1_000_000)
    val: tuple[int, int] = (2_000_000, 2_010_000)
    test: tuple[int, int] = (3_000_000, 3_010_000)

    def range(self, split: str) -> tuple[int, int]:
        return {"train": self.train, "val": self.val, "test": self.test}[split]

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SeedRanges":
        section = cfg.get("level_seeds", {})
        return cls(**{k: tuple(v) for k, v in section.items()})


class VectorPlatformer:
    """`n_envs` episodios en paralelo, con reinicio automatico y semillas sin repetir.

    Cada entorno consume semillas del rango de su particion en progresion aritmetica de
    paso `n_envs`, asi que dos entornos del mismo vector nunca juegan el mismo nivel.
    """

    def __init__(
        self,
        n_envs: int,
        split: str,
        seeds: SeedRanges,
        level_cfg: LevelConfig = LevelConfig(),
        obs_cfg: ObservationConfig = ObservationConfig(),
        reward_cfg: RewardConfig = RewardConfig(),
        max_steps: int = 1200,
    ) -> None:
        low, high = seeds.range(split)
        self.n_envs = n_envs
        self.split = split
        self.low, self.high = low, high
        self.envs = [
            PlatformerEnv(level_cfg, obs_cfg, reward_cfg, max_steps) for _ in range(n_envs)
        ]
        self.counters = np.arange(n_envs, dtype=np.int64)

    @property
    def n_features(self) -> int:
        return self.envs[0].n_features

    def _next_seed(self, index: int) -> int:
        seed = self.low + int(self.counters[index] % (self.high - self.low))
        self.counters[index] += self.n_envs
        return seed

    def reset(self) -> np.ndarray:
        return np.stack([env.reset(self._next_seed(i)) for i, env in enumerate(self.envs)])

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        observations, rewards, dones, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            obs, reward, done, info = env.step(int(actions[i]))
            if done:
                obs = env.reset(self._next_seed(i))
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return (
            np.stack(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )


def scripted_action(env: PlatformerEnv) -> int:
    """Politica de referencia escrita a mano: correr y saltar lo que haya delante.

    No entrena nada ni entra en la comparacion entre cableados. Esta por dos razones:
    demuestra que los niveles generados tienen solucion (lo comprueba un test), y da una
    linea de referencia contra la que leer lo que consiga PPO, igual que el FNO en fluidos.
    Medido sobre 40 niveles de test de 160 casillas: llega a la meta en el 55 %, con un
    progreso medio de 0,75. Lo que la mata casi siempre es un enemigo, no un hueco; no es
    un limite superior, es solo una heuristica razonable. La linea baja, la que de verdad
    hay que superar, es "saltar siempre a la derecha": 0,08 de meta y 0,38 de progreso.
    """
    level = env.level
    assert level is not None
    column = int(np.floor(env.x))
    feet = int(np.floor(env.y - EPS))

    # Saltar en el borde justo, no tres casillas antes: adelantarse es lo que hacia
    # caer al hueco, porque el salto empieza a bajar antes de cruzarlo.
    gap_ahead = column + 1 < level.width and level.floor[column + 1] >= level.height
    wall_ahead = any(
        env._solid(feet - row, c)
        for row in (0, 1)
        for c in (column + 1, column + 2)
        if c < level.width
    )
    enemy_ahead = any(
        0 < env.enemy_pos[i, 0] - env.x < 2.0 and abs(env.enemy_pos[i, 1] - env.y) < 1.5
        for i in np.flatnonzero(env.enemy_alive)
    )
    if (gap_ahead or wall_ahead or enemy_ahead) and env.on_ground:
        return 4          # derecha + salto + carrera
    if not env.on_ground and (gap_ahead or wall_ahead):
        return 4          # mantener el boton mientras sube
    return 2              # derecha + carrera
