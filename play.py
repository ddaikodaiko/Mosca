"""Graba una partida en GIF: el nivel, el jugador y quien lo esta jugando (fase 6).

Uso:
    python play.py --config configs/mario_proto.yaml --policy heuristica --level 3000000
    python play.py --config configs/mario_proto.yaml --policy salto,heuristica --out x.mp4
    python play.py --config configs/mario_proto.yaml --policy mosca --checkpoint a.pt

Con varias politicas separadas por comas se apilan en el mismo GIF, **jugando el mismo
nivel**, que es la unica forma honesta de compararlas de un vistazo. La camara sigue al
jugador. El marcador de cada franja dice cuanto nivel lleva recorrido.

Esto no mide nada: para numeros esta `evaluate_rl.py`. Es para ver el juego.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from data.spectral import resolve_device
from envs.platformer import N_ACTIONS, PlatformerEnv, scripted_action
from evaluate_rl import carga_politica, construye_entorno
from train_rl import load_wirings

VENTANA = 24.0   # casillas visibles a lo ancho

COLORES = {
    "cielo": "#0e1726",
    "suelo": "#5b7c99",
    "jugador": "#f2b632",
    "enemigo": "#d7504b",
    "meta": "#7fd17f",
}


def politica_constante(accion: int) -> Callable[[PlatformerEnv, np.ndarray], int]:
    return lambda env, obs: accion


def politica_de_checkpoint(
    ruta: Path, cfg: dict[str, Any], device: torch.device
) -> tuple[Callable[[PlatformerEnv, np.ndarray], int], Callable[[], None]]:
    modelo = carga_politica(ruta, cfg, load_wirings(cfg), device)
    estado = {"h": modelo.initial_state(1, device)}

    @torch.no_grad()
    def accion(env: PlatformerEnv, obs: np.ndarray) -> int:
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)[None]
        logits, _, estado["h"] = modelo(tensor, estado["h"])
        return int(logits.argmax(dim=1).item())

    return accion, lambda: estado.update(h=modelo.initial_state(1, device))


def graba_partida(
    cfg: dict[str, Any], semilla: int, accion: Callable[[PlatformerEnv, np.ndarray], int],
    reinicia: Callable[[], None] | None, max_frames: int,
) -> tuple[PlatformerEnv, list[dict[str, Any]]]:
    env = construye_entorno(cfg)
    obs = env.reset(semilla)
    if reinicia is not None:
        reinicia()
    cuadros = []
    done = False
    while not done and len(cuadros) < max_frames:
        cuadros.append({
            "x": env.x, "y": env.y,
            "enemigos": env.enemy_pos[env.enemy_alive].copy(),
            "progreso": env.max_x / env.level.goal_x,
        })
        obs, _, done, info = env.step(accion(env, obs))
    cuadros.append({
        "x": env.x, "y": env.y,
        "enemigos": env.enemy_pos[env.enemy_alive].copy(),
        "progreso": env.max_x / env.level.goal_x,
        "final": "META" if info["meta"] else ("MUERTO" if info["muerte"] else "TIEMPO"),
    })
    return env, cuadros


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--policy", default="heuristica",
                        help="coma: heuristica, salto, derecha, azar, o el nombre de un modelo")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--level", type=int, default=3_000_000)
    parser.add_argument("--out", type=Path, default=Path("docs/media/mario.mp4"),
                        help="la extension elige el formato: .mp4 o .gif")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=900)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter
    from matplotlib.patches import Rectangle

    cfg = yaml.safe_load(args.config.read_text())
    device = resolve_device(args.device or cfg.get("device", "auto"))
    checkpoints = args.checkpoint_dir or Path(cfg["output_dir"])

    nombres = [p.strip() for p in args.policy.split(",") if p.strip()]
    partidas = []
    for nombre in nombres:
        reinicia = None
        if nombre == "heuristica":
            accion = lambda env, obs: scripted_action(env)
        elif nombre == "salto":
            accion = politica_constante(3)
        elif nombre == "derecha":
            accion = politica_constante(2)
        elif nombre == "azar":
            generador = np.random.default_rng(0)
            accion = lambda env, obs: int(generador.integers(N_ACTIONS))
        else:
            ruta = checkpoints / f"{nombre}_s0.pt"
            if not ruta.exists():
                raise SystemExit(f"no hay checkpoint en {ruta}")
            accion, reinicia = politica_de_checkpoint(ruta, cfg, device)
        env, cuadros = graba_partida(cfg, args.level, accion, reinicia, args.max_frames)
        partidas.append((nombre, env, cuadros))
        print(f"  {nombre}: {len(cuadros)} cuadros, {cuadros[-1].get('final', '?')},"
              f" progreso {cuadros[-1]['progreso']:.2f}")

    nivel = partidas[0][1].level
    n = len(partidas)
    figura, ejes = plt.subplots(n, 1, figsize=(10, 2.2 * n), squeeze=False)
    figura.patch.set_facecolor(COLORES["cielo"])
    total = max(len(cuadros) for _, _, cuadros in partidas)

    # El suelo va con imshow y no con puntos: como marcadores sueltos las casillas
    # salen separadas y el suelo parece agujereado justo donde no hay agujeros.
    from matplotlib.colors import ListedColormap

    mapa = ListedColormap([COLORES["cielo"], COLORES["suelo"]])
    alto_visible = (
        float(np.min(nivel.floor[nivel.floor < nivel.height])) - 4.0,
        float(np.max(nivel.floor[nivel.floor < nivel.height])) + 2.0,
    )
    artistas = []
    for indice, (nombre, _, _) in enumerate(partidas):
        eje = ejes[indice][0]
        eje.set_facecolor(COLORES["cielo"])
        eje.imshow(nivel.tiles, cmap=mapa, vmin=0, vmax=1, interpolation="nearest",
                   extent=(0, nivel.width, nivel.height, 0), aspect="auto", zorder=0)
        eje.axvline(nivel.goal_x, color=COLORES["meta"], lw=2)
        jugador = Rectangle((0, 0), 0.8, 0.9, color=COLORES["jugador"], zorder=3)
        eje.add_patch(jugador)
        enemigos = eje.scatter([], [], marker="o", s=60, color=COLORES["enemigo"], zorder=2)
        texto = eje.text(0.01, 0.94, "", transform=eje.transAxes, color="white",
                         fontsize=10, va="top", family="monospace")
        eje.set_ylim(alto_visible[1], alto_visible[0])
        eje.set_xticks([])
        eje.set_yticks([])
        for lado in eje.spines.values():
            lado.set_visible(False)
        artistas.append((jugador, enemigos, texto, eje, nombre))

    figura.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # El formato lo decide la extension. El GIF de 199 cuadros pesa 1,3 MB y hay visores
    # que se atragantan; el mp4 con el mismo contenido baja de 200 kB.
    if args.out.suffix.lower() in (".mp4", ".m4v"):
        escritor = FFMpegWriter(fps=args.fps, codec="libopenh264", bitrate=4000)
    else:
        escritor = PillowWriter(fps=args.fps)
    with escritor.saving(figura, str(args.out), dpi=100):
        for k in range(total):
            for (jugador, enemigos, texto, eje, nombre), (_, _, cuadros) in zip(artistas, partidas):
                cuadro = cuadros[min(k, len(cuadros) - 1)]
                jugador.set_xy((cuadro["x"] - 0.4, cuadro["y"] - 0.9))
                puntos = cuadro["enemigos"]
                enemigos.set_offsets(
                    np.c_[puntos[:, 0], puntos[:, 1] - 0.4] if puntos.size else np.empty((0, 2))
                )
                centro = np.clip(cuadro["x"], VENTANA / 2, nivel.width - VENTANA / 2)
                eje.set_xlim(centro - VENTANA / 2, centro + VENTANA / 2)
                final = cuadro.get("final", "")
                texto.set_text(f"{nombre}   {cuadro['progreso'] * 100:5.1f} %   {final}")
            escritor.grab_frame()
    print(f"\nvideo en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
