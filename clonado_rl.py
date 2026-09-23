"""Clonado de conducta de la heuristica: ¿puede cada politica REPRESENTAR una buena conducta?

    python clonado_rl.py --config configs/mario_proto.yaml
    python clonado_rl.py --config configs/mario_proto.yaml --modelos mosca,mlp --pasos 3000

Separa dos diagnosticos de la fase 6 que PPO mezcla: "este modelo no puede expresar una
buena politica" y "PPO no consigue llevarlo hasta ella". Se juegan niveles de
entrenamiento con `scripted_action`, se ajusta cada politica por entropia cruzada contra
esas acciones y se la pone a jugar en codicioso en niveles de validacion.

Sin memoria a proposito: cada paso arranca de `h = 0`. Medido en el checkpoint de la demo
de NES, poner la h a cero cambia la politica 0,006 de variacion total y deja la accion
mas probable igual en el 100 % de los pasos, asi que el nucleo entrenado con PPO tampoco
usa memoria. Solo el entorno propio: la heuristica lee el estado interno del nivel.

El acierto global engana (la clase mayoritaria ya da el 91 %): lo que decide es el recall
del salto y el progreso al jugar, que se informan siempre junto a "siempre derecha" y
"saltar siempre".
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from torch import Tensor, nn

from envs.platformer import (
    N_ACTIONS,
    LevelConfig,
    ObservationConfig,
    PlatformerEnv,
    RewardConfig,
    SeedRanges,
    scripted_action,
)
from models.masked_policy import build_policy
from train_rl import load_wirings

SALTO, DERECHA = 4, 2


def crea_entorno(cfg: dict[str, Any]) -> PlatformerEnv:
    return PlatformerEnv(
        LevelConfig(**cfg.get("level", {})),
        ObservationConfig(**cfg.get("observation", {})),
        RewardConfig(**cfg.get("reward", {})),
        int(cfg.get("max_episode_steps", 1200)),
    )


def demostraciones(cfg: dict[str, Any], semillas: range) -> tuple[Tensor, Tensor]:
    env = crea_entorno(cfg)
    observaciones, acciones = [], []
    for semilla in semillas:
        observacion, done = env.reset(semilla), False
        while not done:
            accion = scripted_action(env)
            observaciones.append(observacion)
            acciones.append(accion)
            observacion, _, done, _ = env.step(accion)
    return torch.tensor(np.stack(observaciones), dtype=torch.float32), torch.tensor(acciones)


def progreso(cfg: dict[str, Any], semillas: range, decide: Callable[[PlatformerEnv, np.ndarray], int]) -> float:
    env = crea_entorno(cfg)
    resultados = []
    for semilla in semillas:
        observacion, done = env.reset(semilla), False
        while not done:
            observacion, _, done, info = env.step(decide(env, observacion))
        resultados.append(info["progreso"])
    return float(np.mean(resultados))


def logits_sin_memoria(politica: nn.Module, x: Tensor) -> Tensor:
    return politica(x, politica.initial_state(x.shape[0], x.device))[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/mario_proto.yaml"))
    parser.add_argument("--modelos", default="mlp,mosca,aleatorio,recableado,denso,sin_nucleo")
    parser.add_argument("--pasos", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lote", type=int, default=256)
    parser.add_argument("--niveles-demo", type=int, default=300)
    parser.add_argument("--niveles-val", type=int, default=60)
    parser.add_argument("--niveles-juego", type=int, default=40)
    parser.add_argument("--semilla", type=int, default=0)
    parser.add_argument("--salida", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    rangos = SeedRanges.from_config(cfg)
    inicio_train, _ = rangos.range("train")
    inicio_val, _ = rangos.range("val")
    # Los niveles donde se juega no son los de la validacion por acciones: asi el progreso
    # no se mide sobre estados que ya se usaron para elegir nada.
    semillas_juego = range(inicio_val + 5000, inicio_val + 5000 + args.niveles_juego)

    x_train, y_train = demostraciones(cfg, range(inicio_train, inicio_train + args.niveles_demo))
    x_val, y_val = demostraciones(cfg, range(inicio_val, inicio_val + args.niveles_val))
    clases = np.bincount(y_train.numpy(), minlength=N_ACTIONS).tolist()
    mayoritaria = float((y_val == y_train.mode().values).float().mean())
    print(f"demostraciones: {len(x_train)} pasos de entrenamiento, {len(x_val)} de validacion,"
          f" clases {clases}; la mayoritaria acierta {mayoritaria:.3f}")

    resultados: dict[str, Any] = {
        "config": cfg, "config_path": str(args.config), "semilla": args.semilla,
        "pasos": args.pasos, "lr": args.lr, "lote": args.lote, "clases_train": clases,
        "acierto_mayoritaria": mayoritaria, "modelos": {},
        "referencias": {
            "heuristica": progreso(cfg, semillas_juego, lambda env, _: scripted_action(env)),
            "siempre_derecha": progreso(cfg, semillas_juego, lambda env, _: DERECHA),
            "saltar_siempre": progreso(cfg, semillas_juego, lambda env, _: SALTO),
        },
    }
    wirings = load_wirings(cfg)
    for nombre in args.modelos.split(","):
        torch.manual_seed(args.semilla)
        politica = build_policy(nombre, cfg, x_train.shape[1], N_ACTIONS, args.semilla, wirings)
        politica.fit_feature_statistics(x_train)
        optimiser = torch.optim.Adam(
            [p for p in politica.parameters() if p.requires_grad], lr=args.lr
        )
        generador = torch.Generator().manual_seed(args.semilla)
        t0 = time.time()
        for _ in range(args.pasos):
            indice = torch.randint(len(x_train), (args.lote,), generator=generador)
            perdida = nn.functional.cross_entropy(
                logits_sin_memoria(politica, x_train[indice]), y_train[indice]
            )
            optimiser.zero_grad(set_to_none=True)
            perdida.backward()
            optimiser.step()

        politica.eval()
        with torch.no_grad():
            pred_train = logits_sin_memoria(politica, x_train).argmax(1)
            pred_val = logits_sin_memoria(politica, x_val).argmax(1)

        def decide(_env: PlatformerEnv, observacion: np.ndarray) -> int:
            with torch.no_grad():
                x = torch.as_tensor(observacion, dtype=torch.float32).unsqueeze(0)
                return int(logits_sin_memoria(politica, x).argmax())

        fila = {
            "perdida_final": float(perdida.item()),
            "acierto_train": float((pred_train == y_train).float().mean()),
            "acierto_val": float((pred_val == y_val).float().mean()),
            "recall_salto_val": float((pred_val[y_val == SALTO] == SALTO).float().mean()),
            "salto_de_mas_val": float((pred_val[y_val != SALTO] == SALTO).float().mean()),
            "progreso": progreso(cfg, semillas_juego, decide),
            "segundos": time.time() - t0,
        }
        resultados["modelos"][nombre] = fila
        print(f"{nombre:11s} acierto val {fila['acierto_val']:.3f}  recall salto"
              f" {fila['recall_salto_val']:.3f}  saltos de mas {fila['salto_de_mas_val']:.3f}"
              f"  progreso {fila['progreso']:.3f}   [{fila['segundos']:.0f} s]", flush=True)

    for nombre, valor in resultados["referencias"].items():
        print(f"{nombre:15s} progreso {valor:.3f}")
    salida = args.salida or Path(cfg["output_dir"]) / f"clonado_s{args.semilla}.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(resultados, indent=1, ensure_ascii=False))
    print(f"\n-> {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
