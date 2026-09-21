"""Ver jugar en directo, en una ventana, a la politica que se le diga (fase 6b).

Uso:
    python watch_nes.py --stage 1-1 --policy corre_salta
    python watch_nes.py --stage 1-1 --policy mosca \
        --config configs/mario_nes_proto.yaml --checkpoint artifacts/mario_nes_proto/mosca_s0.pt
    python watch_nes.py --stage 1-1 --policy mosca --config ... --checkpoint ... --loop

Abre la ventana del emulador y juega a velocidad de NES (60 fotogramas por segundo).
`--speed` la cambia: 0.5 va a la mitad, 2 al doble, 0 a todo lo que de la maquina.

La red NO ve estos pixeles: ve la ventana de casillas que sale de la RAM. Esto es para
mirar, no es la entrada del modelo.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml

from data.spectral import resolve_device
from envs.nes_mario import NesConfig, NesMarioEnv
from envs.platformer import ObservationConfig
from play_nes import CONSTANTES

FRAME_SECONDS = 1.0 / 60.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="1-1")
    parser.add_argument("--policy", default="corre_salta",
                        help=f"{', '.join(CONSTANTES)} o el nombre de un modelo entrenado")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--speed", type=float, default=1.0, help="1 = velocidad de NES; 0 = sin freno")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loop", action="store_true", help="volver a empezar al morir")
    parser.add_argument("--sample", action="store_true",
                        help="muestrear la politica en vez de coger el maximo; la codiciosa"
                             " se queda atascada contra las paredes")
    parser.add_argument("--stage-only", action="store_true",
                        help="ignorar el reparto de niveles y jugar solo --stage")
    args = parser.parse_args()

    world, level = (int(parte) for parte in args.stage.split("-"))
    env = NesMarioEnv(
        "train", ObservationConfig(), NesConfig(max_steps=args.max_steps), render_mode="human"
    )
    env.stages = ((world, level),)

    reinicia: Callable[[], None] | None = None
    if args.policy in CONSTANTES:
        constante = CONSTANTES[args.policy]
        accion: Callable[[np.ndarray], int] = lambda obs: constante
    else:
        if args.config is None or args.checkpoint is None:
            raise SystemExit("una politica entrenada necesita --config y --checkpoint")
        from evaluate_rl import carga_politica
        from train_rl import load_wirings

        cfg = yaml.safe_load(args.config.read_text())
        device = resolve_device(cfg.get("device", "auto"))
        modelo = carga_politica(args.checkpoint, cfg, load_wirings(cfg), device)
        estado = {"h": modelo.initial_state(1, device)}
        reinicia = lambda: estado.update(h=modelo.initial_state(1, device))

        @torch.no_grad()
        def accion(obs: np.ndarray) -> int:
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)[None]
            logits, _, estado["h"] = modelo(tensor, estado["h"])
            if args.sample:
                return int(torch.distributions.Categorical(logits=logits).sample().item())
            return int(logits.argmax(dim=1).item())

    print(f"jugando {args.stage} con '{args.policy}'. Cierra la ventana o Ctrl-C para parar.")
    partida = 0
    try:
        while True:
            partida += 1
            obs = env.reset(args.seed + partida - 1)
            if reinicia is not None:
                reinicia()
            done = False
            reloj = time.monotonic()
            while not done:
                obs, _, done, info = env.step(accion(obs))
                env.env.render()   # sin render_mode="human" esto no pinta nada
                if args.speed > 0:
                    # frame_skip fotogramas por accion, a 60 Hz de NES
                    reloj += env.cfg.frame_skip * FRAME_SECONDS / args.speed
                    espera = reloj - time.monotonic()
                    if espera > 0:
                        time.sleep(espera)
                    else:
                        reloj = time.monotonic()
            final = "META" if info["meta"] else ("MUERTO" if info["muerte"] else "TIEMPO")
            print(f"  partida {partida}: {final}   x {env.x}   progreso {info['progreso']:.1%}")
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\nparado")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
