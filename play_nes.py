"""Graba en video una partida de Super Mario Bros de NES de verdad (fase 6b).

Uso:
    python play_nes.py --stage 1-1 --policy derecha_salto --out docs/media/nes_1-1.mp4
    python play_nes.py --stage 1-1 --policy mosca --checkpoint artifacts/mario_nes/mosca_s0.pt

Lo que se graba son los pixeles del emulador, que es lo que se ve. Lo que la red MIRA no
son esos pixeles sino la ventana de casillas que `envs/nes_mario.py` saca de la RAM; el
video es para ver la partida, no para representar la entrada del modelo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from data.spectral import resolve_device
from envs.nes_mario import NesConfig, NesMarioEnv
from envs.platformer import ObservationConfig

# Indices de SIMPLE_MOVEMENT: 0 nada, 1 derecha, 2 derecha+A, 3 derecha+B, 4 derecha+A+B,
# 5 A, 6 izquierda.
CONSTANTES = {"derecha": 1, "derecha_salto": 2, "derecha_corre": 3, "corre_salta": 4, "quieto": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="1-1")
    parser.add_argument("--policy", default="corre_salta")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None, help="necesario con --checkpoint")
    parser.add_argument("--out", type=Path, default=Path("docs/media/nes.mp4"))
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter

    world, level = (int(part) for part in args.stage.split("-"))
    env = NesMarioEnv("train", ObservationConfig(), NesConfig(max_steps=args.max_steps))
    env.stages = ((world, level),)

    accion: Callable[[np.ndarray], int]
    etiqueta = args.policy
    if args.policy in CONSTANTES:
        constante = CONSTANTES[args.policy]
        accion = lambda obs: constante
    else:
        if args.config is None or args.checkpoint is None:
            raise SystemExit("una politica entrenada necesita --config y --checkpoint")
        from evaluate_rl import carga_politica
        from train_rl import load_wirings

        cfg = yaml.safe_load(args.config.read_text())
        device = resolve_device(cfg.get("device", "auto"))
        modelo = carga_politica(args.checkpoint, cfg, load_wirings(cfg), device)
        estado = {"h": modelo.initial_state(1, device)}

        @torch.no_grad()
        def accion(obs: np.ndarray) -> int:
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)[None]
            logits, _, estado["h"] = modelo(tensor, estado["h"])
            return int(logits.argmax(dim=1).item())

    obs = env.reset(args.seed)
    cuadros: list[np.ndarray] = []
    marcas: list[str] = []
    done = False
    while not done and len(cuadros) < args.max_steps:
        obs, _, done, info = env.step(accion(obs))
        cuadros.append(env.frame.copy())
        marcas.append(f"{info['nivel']}   x {env.x:4d}   {info['progreso'] * 100:5.1f} %")
    final = "META" if info["meta"] else ("MUERTO" if info["muerte"] else "TIEMPO")
    print(f"  {etiqueta} en {args.stage}: {len(cuadros)} cuadros, {final},"
          f" progreso {info['progreso']:.3f}")
    env.close()

    figura, eje = plt.subplots(figsize=(256 / 60, 240 / 60), dpi=60)
    figura.subplots_adjust(0, 0, 1, 1)
    eje.axis("off")
    imagen = eje.imshow(cuadros[0])
    texto = eje.text(4, 232, "", color="white", fontsize=6, family="monospace")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    escritor = (
        FFMpegWriter(fps=args.fps, codec="libopenh264", bitrate=4000)
        if args.out.suffix.lower() in (".mp4", ".m4v") else PillowWriter(fps=args.fps)
    )
    with escritor.saving(figura, str(args.out), dpi=60):
        for cuadro, marca in zip(cuadros, marcas):
            imagen.set_data(cuadro)
            texto.set_text(f"{etiqueta}   {marca}   {final if cuadro is cuadros[-1] else ''}")
            escritor.grab_frame()
    print(f"video en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
