"""Super Mario Bros de NES de verdad, con emulador (fase 6b).

El diseno descartaba los emuladores de ROMs comerciales. Se hace igualmente por encargo
explicito del usuario (21-09-2026). La ROM no se descarga de ningun sitio: viene dentro
del paquete `gym-super-mario-bros`, que es el que usa la literatura de RL.

**La observacion sigue siendo simbolica, no pixeles.** Se saca del mapa de casillas de la
RAM del NES (0x0500-0x069F, dos pantallas de 13x16) y de las posiciones de los enemigos,
y se entrega con exactamente la misma forma que `envs/platformer.py`: ventana de casillas
en tres canales mas cinco escalares. Asi la politica del conectoma no cambia ni una linea
y los dos entornos son intercambiables. El motivo es el del diseno: aprender desde imagen
multiplica el coste por ordenes de magnitud y aqui no hay GPU.

**Lo que se pierde respecto al entorno propio.** El diseno exige niveles generados
aleatoriamente, y Super Mario Bros tiene 32 niveles fijos: la red puede memorizar la
secuencia de saltos. La unica defensa honesta es repartir los niveles en conjuntos
disjuntos (`STAGES`) y medir SIEMPRE en niveles que no se han entrenado. Un numero
medido en el nivel de entrenamiento no dice nada y no debe publicarse.

Direcciones de RAM usadas (mapa estandar de SMB):
    0x006D/0x0086  pagina y desplazamiento de la x del jugador
    0x03B8         y del jugador en pantalla
    0x001D         estado: 0 en el suelo, 1-3 en el aire o trepando
    0x0500-0x069F  mapa de casillas de las dos pantallas en memoria
    0x000F-0x0013  enemigo vivo;  0x006E-0x0072 / 0x0087-0x008B  su x;  0x00CF-0x00D3  su y
    0x009F         velocidad vertical (con signo, -5 subiendo .. +5 cayendo)   [fase 6c]
    0x06FC         mando: bit 0x80 = A pulsado                                  [fase 6c]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from envs.platformer import ObservationConfig

# SIMPLE_MOVEMENT de gym-super-mario-bros son 7 acciones, las mismas que el entorno
# propio: nada, derecha, derecha+A, derecha+B, derecha+A+B, A, izquierda.
N_NES_ACTIONS = 7

TILE_MAP = 0x0500
SCREEN_TILES = 208           # 13 filas x 16 columnas por pantalla
HUD_ROWS = 1                 # calibrado contra la RAM, no supuesto: con 1, la casilla
                             # bajo los pies de Mario sale solida (84) y la suya, aire.
                             # Con 2 el suelo aparecia dos filas por debajo de el.
TILE = 16                    # pixeles por casilla

# Reparto de niveles. Disjunto por construccion, como los rangos de semilla del entorno
# propio. Se entrena en 1-1..7-1 y se mide en niveles que la red no ha visto nunca.
STAGES: dict[str, tuple[tuple[int, int], ...]] = {
    "train": ((1, 1), (1, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)),
    "val": ((1, 3), (3, 3), (5, 3)),
    "test": ((2, 3), (4, 3), (6, 3), (8, 1)),
}


@dataclass
class NesConfig:
    frame_skip: int = 4          # una accion del agente = 4 fotogramas de NES
    max_steps: int = 2000        # pasos del agente, no fotogramas
    random_noop: int = 20        # fotogramas de espera aleatorios al empezar
    reward_scale: float = 0.1    # la recompensa del paquete llega a 15 por paso
    death_penalty: float = 5.0
    stuck_steps: int = 80        # pasos sin avanzar antes de cortar el episodio
    stage: tuple[int, int] | None = None   # forzar un unico nivel (demo, no experimento)
    # --- Fase 6c: que aprenda sola. Apagado por defecto: sin tocarlo, todo es como antes.
    # Dos numeros mas en la observacion: velocidad vertical y si A esta pulsado. En SMB un
    # salto nuevo exige SOLTAR A antes; sin esto la red no puede saber si pulsar A ahora la
    # hara saltar (la heuristica lo sabe, y por eso sin memoria no se la podia clonar).
    motion_features: bool = False
    # Arranques en la frontera: con esta probabilidad, un episodio de ENTRENAMIENTO empieza
    # `frontier_rewind` pasos antes del punto mas lejano al que la propia red ha llegado en
    # ese nivel, en vez de al principio. Nadie le ensena nada: solo practica mas donde se
    # atasca. Validacion y prueba empiezan siempre al principio (se fuerza a 0 fuera de train).
    frontier_probability: float = 0.0
    frontier_rewind: int = 20
    frontier_refresh: int = 32   # pixeles de mejora antes de rehacer el punto de arranque


class NesMarioEnv:
    """Misma API que `PlatformerEnv`: `reset(seed) -> obs` y `step(accion)`."""

    def __init__(
        self,
        split: str = "train",
        obs_cfg: ObservationConfig = ObservationConfig(),
        cfg: NesConfig = NesConfig(),
        render_mode: str | None = None,
    ) -> None:
        self.split = split
        self.obs_cfg = obs_cfg
        self.cfg = cfg
        # Sin render_mode="human" desde la construccion, `render()` sale sin hacer nada y
        # NUNCA se abre ventana. Entrenando va a None, que es lo que hay que usar.
        self.render_mode = render_mode
        # El YAML entrega listas, y el nivel se usa como clave de diccionario.
        self.stages = (tuple(cfg.stage),) if cfg.stage is not None else STAGES[split]
        self._envs: dict[tuple[int, int], Any] = {}
        self.env: Any = None
        self.frontera = cfg.frontier_probability if split == "train" else 0.0
        # Por nivel: la partida que mas lejos llego (espera inicial, acciones, x por paso)
        # y un emulador aparte cuyo "reset" vuelve a su punto de arranque.
        self._mejor: dict[tuple[int, int], dict[str, Any]] = {}
        self._arranques: dict[tuple[int, int], dict[str, Any]] = {}
        self.desde_frontera = False
        # Se llama con cada fotograma de NES dentro de `step`, no solo con el ultimo de los
        # `frame_skip`. Lo usa `grabar_humano.py` para pintar a 60 fps mientras juega una
        # persona; entrenando es None y no cuesta nada.
        self.al_fotograma: Callable[[np.ndarray], None] | None = None

    @property
    def n_features(self) -> int:
        return self.obs_cfg.n_features + (2 if self.cfg.motion_features else 0)

    def _nuevo_emulador(self, stage: tuple[int, int]) -> Any:
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace

        world, level = stage
        raw = gym_super_mario_bros.make(
            f"SuperMarioBros-{world}-{level}-v0", render_mode=self.render_mode
        )
        return JoypadSpace(raw, SIMPLE_MOVEMENT)

    def _emulator(self, stage: tuple[int, int]) -> Any:
        """Un emulador por nivel, reutilizado: construirlo cuesta ~0,3 s."""
        if stage not in self._envs:
            self._envs[stage] = self._nuevo_emulador(stage)
        return self._envs[stage]

    def _arranque_en_frontera(self, stage: tuple[int, int]) -> dict[str, Any] | None:
        """Emulador cuyo `reset()` deja la partida `frontier_rewind` pasos antes del punto
        mas lejano alcanzado en `stage`.

        El NES es determinista (comprobado: repetir espera y botones da la RAM identica
        byte a byte), asi que el estado se reconstruye rejugando la mejor partida y se fija
        con la copia de seguridad del propio emulador, que es a donde vuelve `reset()`.
        Rejugar cuesta lo mismo que jugar (~7 ms por paso), por eso se rehace solo cuando la
        mejor marca ha avanzado `frontier_refresh` pixeles.
        """
        mejor = self._mejor.get(stage)
        if mejor is None:
            return None
        actual = self._arranques.get(stage)
        if actual is not None and mejor["max_x"] - actual["max_x_ref"] < self.cfg.frontier_refresh:
            return actual
        xs = mejor["xs"]
        cima = int(np.argmax(xs))                     # primer paso en el que llego a su maximo
        inicio = max(0, cima + 1 - self.cfg.frontier_rewind)
        if inicio == 0:
            return None                               # la frontera esta en la salida
        emulador = self._nuevo_emulador(stage)
        emulador.reset()
        for _ in range(mejor["espera"]):
            emulador.step(0)
        for accion in mejor["acciones"][:inicio]:
            for _ in range(self.cfg.frame_skip):
                _, _, terminal, truncado, _ = emulador.step(int(accion))
                if terminal or truncado:              # no deberia pasar antes de la cima
                    emulador.close()
                    return None
        emulador.unwrapped._backup()
        if actual is not None:
            actual["emulador"].close()
        self._arranques[stage] = {
            "emulador": emulador, "inicio": inicio, "max_x_ref": mejor["max_x"],
            "max_x": int(max(xs[:inicio])), "espera": mejor["espera"],
            "acciones": list(mejor["acciones"][:inicio]), "xs": list(xs[:inicio]),
        }
        return self._arranques[stage]

    def reset(self, seed: int) -> np.ndarray:
        """La semilla elige el nivel dentro de la particion y la espera inicial."""
        rng = np.random.default_rng(seed)
        self.stage = self.stages[int(rng.integers(len(self.stages)))]
        self.env = self._emulator(self.stage)
        self.env.reset()
        # Arranque con ruido: sin esto, un nivel fijo se puede resolver con una secuencia
        # memorizada de botones que ni siquiera mira la pantalla.
        self._espera = int(rng.integers(self.cfg.random_noop + 1))
        for _ in range(self._espera):
            self.env.step(0)
        self.steps = 0
        self.max_x = 0
        self._acciones: list[int] = []
        self._xs: list[int] = []
        self.desde_frontera = False
        # Solo se tira el dado si la frontera esta activa: con 0 la secuencia de la semilla
        # es exactamente la de antes.
        if self.frontera > 0 and rng.random() < self.frontera:
            arranque = self._arranque_en_frontera(self.stage)
            if arranque is not None:
                self.env = arranque["emulador"]
                self.env.reset()
                self.desde_frontera = True
                self.steps = arranque["inicio"]
                self.max_x = arranque["max_x"]
                self._espera = arranque["espera"]
                self._acciones = list(arranque["acciones"])
                self._xs = list(arranque["xs"])
        # Sin esto, un agente que se queda atascado contra una tuberia gasta los 600
        # pasos del episodio sin aprender nada. Cortar pronto multiplica la experiencia
        # util por paso de CPU, que es el recurso escaso aqui.
        self.stuck = 0
        self.info: dict[str, Any] = {"x_pos": 0, "flag_get": False, "life": 2}
        # Los pixeles no entran en la observacion; se guardan solo para grabar video.
        self.frame = np.zeros((240, 256, 3), dtype=np.uint8)
        self._read_state()
        return self._observe()

    # -- lectura de la RAM ------------------------------------------------------

    def _read_state(self) -> None:
        ram = np.asarray(self.env.unwrapped.ram, dtype=np.int32)
        self.ram = ram
        self.x = int(ram[0x006D]) * 256 + int(ram[0x0086])
        self.y = int(ram[0x03B8])
        self.on_ground = int(ram[0x001D]) == 0
        self.tiles = ram[TILE_MAP:TILE_MAP + 2 * SCREEN_TILES].reshape(2, 13, 16)
        alive = ram[0x000F:0x0014] != 0
        enemy_x = ram[0x006E:0x0073] * 256 + ram[0x0087:0x008C]
        enemy_y = ram[0x00CF:0x00D4]
        self.enemies = np.stack([enemy_x[alive], enemy_y[alive]], axis=1)

    def _tile(self, x_pixel: int, y_pixel: int) -> int:
        """Casilla del mundo en coordenadas de pixel. 0 es aire."""
        row = (y_pixel - HUD_ROWS * TILE) // TILE
        if not 0 <= row < 13:
            return 0
        page = (x_pixel // 256) % 2
        column = (x_pixel % 256) // TILE
        return int(self.tiles[page, row, column])

    # -- API --------------------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        recompensa = 0.0
        terminado = False
        for _ in range(self.cfg.frame_skip):
            self.frame, r, terminal, truncado, info = self.env.step(int(action))
            if self.al_fotograma is not None:
                self.al_fotograma(self.frame)
            recompensa += float(r)
            terminado = bool(terminal or truncado)
            if terminado:
                break
        self.info = info
        self._read_state()
        self.steps += 1
        if self.x > self.max_x:
            self.max_x = self.x
            self.stuck = 0
        else:
            self.stuck += 1
        self._acciones.append(int(action))
        self._xs.append(self.x)

        murio = bool(info.get("is_dead") or info.get("is_game_over") or terminado and not info.get("flag_get"))
        meta = bool(info.get("flag_get"))
        recompensa *= self.cfg.reward_scale
        if murio:
            recompensa -= self.cfg.death_penalty
        atascado = self.stuck >= self.cfg.stuck_steps
        done = bool(murio or meta or atascado or self.steps >= self.cfg.max_steps)
        if done and self.frontera > 0 and meta:
            # Con bandera el nivel ya no tiene frontera: vuelve a empezar siempre al principio
            # hasta que otra partida se quede corta y marque una nueva.
            self._mejor.pop(self.stage, None)
        elif done and self.frontera > 0:
            previo = self._mejor.get(self.stage)
            if previo is None or self.max_x > previo["max_x"]:
                self._mejor[self.stage] = {"max_x": self.max_x, "espera": self._espera,
                                           "acciones": self._acciones, "xs": self._xs}

        # El final del nivel esta en x ~3.180 en casi todos los niveles de SMB; se usa el
        # maximo visto como escala para que "progreso" signifique lo mismo que en el
        # entorno propio: fraccion de nivel recorrida.
        progreso = min(self.max_x / 3180.0, 1.0) if not meta else 1.0
        return self._observe(), recompensa, done, {
            "progreso": float(progreso),
            "meta": meta,
            "muerte": murio,
            "pasos": self.steps,
            "pisados": 0,
            "atascado": bool(atascado),
            "nivel": f"{self.stage[0]}-{self.stage[1]}",
            "desde_frontera": self.desde_frontera,
        }

    def _observe(self) -> np.ndarray:
        cfg = self.obs_cfg
        half_w, half_h = cfg.window_width // 2, cfg.window_height // 2
        solid = np.zeros((cfg.window_height, cfg.window_width), dtype=np.float32)
        void = np.zeros_like(solid)
        enemies = np.zeros_like(solid)

        for dr in range(-half_h, half_h + 1):
            for dc in range(-half_w, half_w + 1):
                x_pixel = self.x + dc * TILE
                y_pixel = self.y + dr * TILE
                fila, columna = dr + half_h, dc + half_w
                if x_pixel < 0 or not HUD_ROWS * TILE <= y_pixel < 240:
                    void[fila, columna] = 1.0
                    continue
                solid[fila, columna] = 1.0 if self._tile(x_pixel, y_pixel) else 0.0

        for ex, ey in self.enemies:
            dc = int(round((int(ex) - self.x) / TILE)) + half_w
            dr = int(round((int(ey) - self.y) / TILE)) + half_h
            if 0 <= dr < cfg.window_height and 0 <= dc < cfg.window_width:
                enemies[dr, dc] = 1.0

        scalars = np.array(
            [
                np.clip((self.x - self.max_x) / TILE, -1.0, 1.0),
                np.clip((self.y - 176) / 64.0, -2.0, 2.0),
                1.0 if self.on_ground else 0.0,
                1.0 - min(self.max_x / 3180.0, 1.0),
                1.0 - self.steps / self.cfg.max_steps,
            ],
            dtype=np.float32,
        )
        partes = [solid.ravel(), enemies.ravel(), void.ravel(), scalars]
        if self.cfg.motion_features:
            velocidad = float(np.int8(np.uint8(self.ram[0x009F]))) / 5.0
            a_pulsado = 1.0 if int(self.ram[0x06FC]) & 0x80 else 0.0
            partes.append(np.array([np.clip(velocidad, -1.5, 1.5), a_pulsado], dtype=np.float32))
        return np.concatenate(partes)

    def close(self) -> None:
        for env in self._envs.values():
            env.close()
        for arranque in self._arranques.values():
            arranque["emulador"].close()
        self._envs.clear()
        self._arranques.clear()


class VectorNesMario:
    """`n_envs` emuladores en paralelo con reinicio automatico, como VectorPlatformer."""

    def __init__(
        self,
        n_envs: int,
        split: str,
        obs_cfg: ObservationConfig = ObservationConfig(),
        cfg: NesConfig = NesConfig(),
        seed_offset: int = 0,
    ) -> None:
        self.n_envs = n_envs
        self.envs = [NesMarioEnv(split, obs_cfg, cfg) for _ in range(n_envs)]
        self.counters = np.arange(n_envs, dtype=np.int64) + seed_offset

    @property
    def n_features(self) -> int:
        return self.envs[0].n_features

    def _next_seed(self, index: int) -> int:
        seed = int(self.counters[index])
        self.counters[index] += self.n_envs
        return seed

    def reset(self) -> np.ndarray:
        return np.stack([env.reset(self._next_seed(i)) for i, env in enumerate(self.envs)])

    def step(self, actions: np.ndarray):
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


def _trabajador(conexion: Any, indice: int, n_envs: int, seed_offset: int, split: str,
                obs_cfg: ObservationConfig, cfg: NesConfig) -> None:
    """Un emulador en su propio proceso. Misma regla de semillas que `VectorNesMario`."""
    env = NesMarioEnv(split, obs_cfg, cfg)
    contador = indice + seed_offset

    def siguiente() -> int:
        nonlocal contador
        semilla, contador = contador, contador + n_envs
        return semilla

    try:
        while True:
            orden, dato = conexion.recv()
            if orden == "reset":
                conexion.send(env.reset(siguiente()))
            elif orden == "step":
                obs, reward, done, info = env.step(int(dato))
                if done:
                    obs = env.reset(siguiente())
                conexion.send((obs, reward, done, info))
            else:
                break
    finally:
        env.close()
        conexion.close()


class VectorNesMarioParalelo:
    """Como `VectorNesMario`, pero cada emulador en un proceso (fase 6c).

    El emulador se lleva ~7,6 ms por paso y es el 90 % de la recoleccion; en serie, 16
    entornos x 128 pasos son 15 s por iteracion solo de emular. Las semillas son las mismas
    que en serie, asi que los niveles jugados no cambian. Se arranca con "spawn": hacer fork
    de un proceso que ya ha tocado CUDA no es seguro.
    """

    def __init__(
        self,
        n_envs: int,
        split: str,
        obs_cfg: ObservationConfig = ObservationConfig(),
        cfg: NesConfig = NesConfig(),
        seed_offset: int = 0,
    ) -> None:
        import multiprocessing as mp

        contexto = mp.get_context("spawn")
        self.n_envs = n_envs
        self._n_features = NesMarioEnv(split, obs_cfg, cfg).n_features
        self.conexiones, self.procesos = [], []
        for indice in range(n_envs):
            nuestra, suya = contexto.Pipe()
            proceso = contexto.Process(
                target=_trabajador, args=(suya, indice, n_envs, seed_offset, split, obs_cfg, cfg),
                daemon=True,
            )
            proceso.start()
            suya.close()
            self.conexiones.append(nuestra)
            self.procesos.append(proceso)

    @property
    def n_features(self) -> int:
        return self._n_features

    def reset(self) -> np.ndarray:
        for conexion in self.conexiones:
            conexion.send(("reset", None))
        return np.stack([conexion.recv() for conexion in self.conexiones])

    def step(self, actions: np.ndarray):
        for conexion, accion in zip(self.conexiones, actions):
            conexion.send(("step", int(accion)))
        observations, rewards, dones, infos = zip(*(c.recv() for c in self.conexiones))
        return (
            np.stack(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            list(infos),
        )

    def close(self) -> None:
        for conexion in self.conexiones:
            try:
                conexion.send(("cerrar", None))
            except (BrokenPipeError, OSError):
                pass
        for proceso in self.procesos:
            proceso.join(timeout=5)
        self.conexiones, self.procesos = [], []

    def __del__(self) -> None:
        if getattr(self, "procesos", None):
            self.close()


class JugadorHeuristico:
    """Heuristica escrita a mano para el NES, leyendo la RAM. Una instancia por partida.

    Corre con B siempre; salta con A si delante hay un muro, un hoyo o un enemigo, mantiene
    A mientras sube (salto alto) y la suelta al caer, para poder volver a saltar nada mas
    tocar el suelo. Un hoyo es una columna vacia hasta el fondo de la pantalla: con solo
    mirar la casilla de debajo, al aterrizar sobre una tuberia alta tomaba el borde por un
    hoyo, saltaba y caia en el hoyo de verdad.

    Alcance medido (23-09-2026, semilla 0): **se pasa el 1-1 entero** (315 pasos) y en los
    demas niveles muere pronto (progreso 0,06-0,55: no sabe de enemigos voladores,
    plataformas ni castillos). Sirve de maestra para la demo del 1-1, no de referencia
    superior del experimento con niveles disjuntos.
    """

    CORRER, CORRER_SALTO = 3, 4

    def __init__(self, muro: int = 40, hueco: int = 20, enemigo: int = 72) -> None:
        self.muro, self.hueco, self.enemigo = muro, hueco, enemigo
        self.anterior = self.CORRER
        self.y_anterior: int | None = None

    def __call__(self, env: NesMarioEnv) -> int:
        x, y = env.x, env.y
        subiendo = self.y_anterior is not None and y < self.y_anterior
        self.y_anterior = y
        if not env.on_ground:
            accion = self.CORRER_SALTO if subiendo else self.CORRER
        elif self.anterior == self.CORRER_SALTO:
            accion = self.CORRER  # soltar A: sin esto no hay segundo salto
        else:
            muro = any(env._tile(x + d, y + dy)
                       for d in range(16, 16 + self.muro, 8) for dy in (0, -8))
            hueco = any(not any(env._tile(x + d, yy) for yy in range(y + 16, 240, TILE))
                        for d in range(16, 16 + self.hueco, 4))
            enemigo = any(0 < ex - x < self.enemigo and -40 < int(ey) - y < 72
                          for ex, ey in env.enemies)
            accion = self.CORRER_SALTO if (muro or hueco or enemigo) else self.CORRER
        self.anterior = accion
        return accion
