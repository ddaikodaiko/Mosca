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
    ObservationConfig,
    PlatformerEnv,
    SeedRanges,
    scripted_action,
)
from models.masked_policy import build_policy
from train_rl import load_wirings

SALTO, DERECHA = 4, 2


def crea_entorno(cfg: dict[str, Any], particion: str = "train") -> Any:
    if cfg.get("env", "propio") == "nes":
        from envs.nes_mario import NesConfig, NesMarioEnv

        return NesMarioEnv(particion, ObservationConfig(**cfg.get("observation", {})),
                           NesConfig(**cfg.get("nes", {})))
    return PlatformerEnv.from_config(cfg)


def maestra(cfg: dict[str, Any]) -> Callable[[Any], int]:
    """Una maestra nueva por partida: la del NES recuerda si estaba subiendo."""
    if cfg.get("env", "propio") == "nes":
        from envs.nes_mario import JugadorHeuristico

        return JugadorHeuristico()
    return scripted_action


def demostraciones(
    cfg: dict[str, Any], semillas: range, particion: str = "train", ruido: float = 0.0
) -> tuple[Tensor, Tensor, Tensor]:
    """Con `ruido` > 0 se EJECUTA una accion al azar con esa probabilidad, pero la etiqueta
    sigue siendo la de la maestra: asi la red ve tambien como salir de situaciones a las
    que la maestra nunca llega sola. En el NES hace falta, porque el 1-1 sin ruido da la
    misma partida exacta en todas las semillas."""
    env = crea_entorno(cfg, particion)
    rng = np.random.default_rng(semillas.start)
    observaciones, acciones, partidas = [], [], []
    for partida, semilla in enumerate(semillas):
        observacion, done, decide = env.reset(semilla), False, maestra(cfg)
        while not done:
            accion = decide(env)
            observaciones.append(observacion)
            acciones.append(accion)
            partidas.append(partida)
            if rng.random() < ruido:
                accion = int(rng.integers(N_ACTIONS))
            observacion, _, done, _ = env.step(accion)
    return torch.tensor(np.stack(observaciones), dtype=torch.float32), torch.tensor(acciones), torch.tensor(partidas)


def demos_humanas(
    directorio: Path, progreso_minimo: float = 0.0, fraccion_val: float = 0.15,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Partidas de `grabar_humano.py`, solo de niveles de train. Se separan por PARTIDA, no
    por paso: pasos vecinos de la misma partida son casi iguales y validar con ellos seria
    medir memoria. Las ultimas `fraccion_val` partidas de cada nivel van a validacion."""
    from envs.nes_mario import STAGES
    from grabar_humano import carga_demos

    niveles_train = {f"{w}-{l}" for w, l in STAGES["train"]}
    partidas = carga_demos(directorio, niveles_train, progreso_minimo)
    if not partidas:
        raise SystemExit(f"no hay partidas de niveles de train en {directorio}")
    por_nivel: dict[str, list[dict[str, Any]]] = {}
    for partida in partidas:
        por_nivel.setdefault(partida["resumen"]["nivel"], []).append(partida)
    train, val = [], []
    for grupo in por_nivel.values():
        n_val = int(round(len(grupo) * fraccion_val)) if len(grupo) > 1 else 0
        train += grupo[:len(grupo) - n_val]
        val += grupo[len(grupo) - n_val:]
    if not val:  # muy pocas partidas: se valida sobre la ultima, y se avisa
        print("aviso: una partida por nivel; la validacion por acciones reutiliza la ultima")
        val = train[-1:]

    def apila(grupo: list[dict[str, Any]]) -> tuple[Tensor, Tensor, Tensor]:
        x = torch.tensor(np.concatenate([p["obs"] for p in grupo]), dtype=torch.float32)
        y = torch.tensor(np.concatenate([p["acciones"] for p in grupo]))
        partida = torch.tensor(np.concatenate(
            [np.full(len(p["acciones"]), i) for i, p in enumerate(grupo)]))
        return x, y, partida

    print(f"demos humanas: {len(train)} partidas de entrenamiento, {len(val)} de validacion,"
          f" niveles {sorted(por_nivel)}")
    return (*apila(train), *apila(val))


def progreso(
    cfg: dict[str, Any], semillas: range,
    nueva: Callable[[], Callable[[Any, np.ndarray], int]], particion: str = "val",
) -> float:
    """`nueva()` da la politica de una partida (la maestra del NES tiene estado)."""
    env = crea_entorno(cfg, particion)
    resultados = []
    for semilla in semillas:
        observacion, done, decide = env.reset(semilla), False, nueva()
        while not done:
            observacion, _, done, info = env.step(decide(env, observacion))
        resultados.append(info["progreso"])
    return float(np.mean(resultados))


def logits_sin_memoria(politica: nn.Module, x: Tensor) -> Tensor:
    return politica(x, politica.initial_state(x.shape[0], x.device))[0]


def logits_con_memoria(politica: nn.Module, x: Tensor, partidas: Tensor) -> Tensor:
    """Recorre las demostraciones en orden arrastrando la h, y la pone a cero al empezar
    cada partida: es como juega la red de verdad."""
    salida, h = [], politica.initial_state(1, x.device)
    for t in range(len(x)):
        if t == 0 or partidas[t] != partidas[t - 1]:
            h = politica.initial_state(1, x.device)
        logits, _, h = politica(x[t:t + 1], h)
        salida.append(logits)
    return torch.cat(salida)


def ventanas_validas(partidas: Tensor, ventana: int) -> Tensor:
    """Inicios de tramo de `ventana` pasos que caen enteros dentro de una partida."""
    if len(partidas) < ventana:
        return torch.zeros(0, dtype=torch.long)
    return torch.nonzero(partidas[:len(partidas) - ventana + 1] == partidas[ventana - 1:]).squeeze(1)


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
    parser.add_argument("--ruido", type=float, default=0.0,
                        help="probabilidad de ejecutar una accion al azar en las demostraciones")
    parser.add_argument("--ventana", type=int, default=1,
                        help="1 = sin memoria (h = 0 en cada paso). K > 1 = entrena sobre"
                             " tramos de K pasos seguidos arrastrando la h, y juega con ella")
    parser.add_argument("--demos", type=Path, default=None,
                        help="carpeta de partidas humanas (grabar_humano.py) en vez de la"
                             " heuristica. Solo NES. Se valida por acciones en un 15 %% de las"
                             " partidas y se juega en los niveles de val, que no se han grabado")
    parser.add_argument("--progreso-minimo", type=float, default=0.0,
                        help="con --demos, descarta partidas que no llegaron a esta fraccion")
    parser.add_argument("--salida", type=Path, default=None)
    parser.add_argument("--guarda-pesos", action="store_true",
                        help="guarda cada politica clonada en <output_dir>/clonado/, con el"
                             " formato de train_rl.py, para arrancar PPO desde ella (--desde)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    rangos = SeedRanges.from_config(cfg)
    inicio_train, _ = rangos.range("train")
    inicio_val, _ = rangos.range("val")
    # Los niveles donde se juega no son los de la validacion por acciones: asi el progreso
    # no se mide sobre estados que ya se usaron para elegir nada.
    semillas_juego = range(inicio_val + 5000, inicio_val + 5000 + args.niveles_juego)

    if args.demos is not None:
        x_train, y_train, p_train, x_val, y_val, p_val = demos_humanas(
            args.demos, args.progreso_minimo)
    else:
        x_train, y_train, p_train = demostraciones(
            cfg, range(inicio_train, inicio_train + args.niveles_demo), "train", args.ruido)
        x_val, y_val, p_val = demostraciones(
            cfg, range(inicio_val, inicio_val + args.niveles_val), "val")
    inicios = ventanas_validas(p_train, args.ventana)
    clases = np.bincount(y_train.numpy(), minlength=N_ACTIONS).tolist()
    mayoritaria = float((y_val == y_train.mode().values).float().mean())
    print(f"demostraciones: {len(x_train)} pasos de entrenamiento, {len(x_val)} de validacion,"
          f" clases {clases}; la mayoritaria acierta {mayoritaria:.3f}")

    resultados: dict[str, Any] = {
        "config": cfg, "config_path": str(args.config), "semilla": args.semilla,
        "pasos": args.pasos, "lr": args.lr, "lote": args.lote, "ruido": args.ruido,
        "ventana": args.ventana,
        "demos": str(args.demos) if args.demos else None,
        "progreso_minimo": args.progreso_minimo,
        "niveles_demo": args.niveles_demo, "clases_train": clases,
        "acierto_mayoritaria": mayoritaria, "modelos": {},
        "referencias": {
            "heuristica": progreso(cfg, semillas_juego, lambda: (
                lambda m: lambda env, _: m(env))(maestra(cfg))),
            "siempre_derecha": progreso(cfg, semillas_juego, lambda: lambda env, _: DERECHA),
            "saltar_siempre": progreso(cfg, semillas_juego, lambda: lambda env, _: SALTO),
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
            if args.ventana == 1:
                indice = torch.randint(len(x_train), (args.lote,), generator=generador)
                perdida = nn.functional.cross_entropy(
                    logits_sin_memoria(politica, x_train[indice]), y_train[indice]
                )
            else:
                # El lote son `lote` tramos; la perdida cuenta en todos sus pasos.
                inicio = inicios[torch.randint(len(inicios), (args.lote,), generator=generador)]
                h = politica.initial_state(args.lote, x_train.device)
                perdida = torch.zeros(())
                for t in range(args.ventana):
                    logits, _, h = politica(x_train[inicio + t], h)
                    perdida = perdida + nn.functional.cross_entropy(logits, y_train[inicio + t])
                perdida = perdida / args.ventana
            optimiser.zero_grad(set_to_none=True)
            perdida.backward()
            optimiser.step()

        politica.eval()
        with torch.no_grad():
            if args.ventana == 1:
                pred_train = logits_sin_memoria(politica, x_train).argmax(1)
                pred_val = logits_sin_memoria(politica, x_val).argmax(1)
            else:
                pred_train = logits_con_memoria(politica, x_train, p_train).argmax(1)
                pred_val = logits_con_memoria(politica, x_val, p_val).argmax(1)

        def nueva_partida() -> Callable[[Any, np.ndarray], int]:
            estado = {"h": politica.initial_state(1, torch.device("cpu"))}

            def decide(_env: Any, observacion: np.ndarray) -> int:
                with torch.no_grad():
                    x = torch.as_tensor(observacion, dtype=torch.float32).unsqueeze(0)
                    if args.ventana == 1:
                        return int(logits_sin_memoria(politica, x).argmax())
                    logits, _, estado["h"] = politica(x, estado["h"])
                    return int(logits.argmax())
            return decide

        fila = {
            "perdida_final": float(perdida.item()),
            "acierto_train": float((pred_train == y_train).float().mean()),
            "acierto_val": float((pred_val == y_val).float().mean()),
            "recall_salto_val": float((pred_val[y_val == SALTO] == SALTO).float().mean()),
            "salto_de_mas_val": float((pred_val[y_val != SALTO] == SALTO).float().mean()),
            "progreso": progreso(cfg, semillas_juego, nueva_partida),
            "segundos": time.time() - t0,
        }
        resultados["modelos"][nombre] = fila
        if args.guarda_pesos:
            destino = Path(cfg["output_dir"]) / "clonado" / f"{nombre}_s{args.semilla}.pt"
            destino.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"modelo": nombre, "semilla": args.semilla, "config": cfg,
                        "origen": "clonado_rl.py", "state_dict": politica.state_dict()}, destino)
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
