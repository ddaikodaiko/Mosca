"""Evaluacion de la fase 6: los modelos entrenados contra tres lineas de referencia.

Uso:
    python evaluate_rl.py --config configs/mario.yaml

Cada modelo y cada linea de referencia juega **exactamente los mismos niveles** (las
semillas de test, que son disjuntas de las de entrenamiento y validacion). Eso permite
emparejar por nivel y usar la misma prueba de permutacion pareada de la fase 5: la
varianza entre niveles es enorme comparada con la diferencia entre modelos.

Las tres referencias, sin las cuales el numero de un modelo no se puede leer:

  azar          acciones uniformes. El suelo.
  siempre_dcha  mantener "derecha + carrera" y nada mas: se para en el primer hueco.
  salto_cte     "derecha + salto" sin soltar. **Esta es la que importa.** Es la politica
                degenerada a la que PPO cae solo, y en niveles de 160 casillas ya saca
                0,38 de progreso y llega a la meta el 8 % de las veces. Un modelo que no
                le gane no ha aprendido a jugar, ha aprendido a saltar sin mirar.
  heuristica    correr y saltar lo que hay delante, escrita a mano. Referencia superior,
                como el FNO en fluidos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from torch import nn

from data.spectral import resolve_device
from envs.platformer import (
    N_ACTIONS,
    LevelConfig,
    ObservationConfig,
    PlatformerEnv,
    RewardConfig,
    SeedRanges,
    scripted_action,
)
from evaluate import prueba_de_permutacion
from models.masked_policy import build_policy
from models.masked_rnn import Wiring
from train_rl import load_wirings

METRICAS = ("progreso", "meta", "recompensa")


def es_nes(cfg: dict[str, Any]) -> bool:
    return cfg.get("env", "propio") == "nes"


def niveles_de_test(cfg: dict[str, Any], n_levels: int) -> list[int]:
    """Semillas de test. En el entorno propio son niveles; en el NES eligen nivel y
    espera inicial dentro del conjunto de prueba, que es disjunto del de entrenamiento."""
    if es_nes(cfg):
        return list(range(n_levels))
    low, high = SeedRanges.from_config(cfg).range("test")
    return [low + i % (high - low) for i in range(n_levels)]


def construye_entorno(cfg: dict[str, Any], split: str = "test"):
    if es_nes(cfg):
        from envs.nes_mario import NesConfig, NesMarioEnv

        nes = dict(cfg.get("nes", {}))
        nivel = nes.pop("stage", None)
        env = NesMarioEnv("train" if nivel else split,
                          ObservationConfig(**cfg.get("observation", {})), NesConfig(**nes))
        if nivel is not None:
            # La demo fija un nivel: entrena y mide en el mismo, y lo dice su YAML.
            env.stages = (tuple(nivel),)
        return env
    return PlatformerEnv(
        LevelConfig(**cfg.get("level", {})),
        ObservationConfig(**cfg.get("observation", {})),
        RewardConfig(**cfg.get("reward", {})),
        max_steps=int(cfg.get("max_episode_steps", 1200)),
    )


def juega(
    cfg: dict[str, Any], semillas: list[int], accion: Callable[[Any, np.ndarray], int],
    reinicia: Callable[[], None] | None = None,
) -> dict[str, np.ndarray]:
    """Un episodio por semilla. Devuelve una serie por metrica, alineada con `semillas`."""
    env = construye_entorno(cfg)
    resultados = {metrica: [] for metrica in METRICAS}
    for semilla in semillas:
        observacion = env.reset(semilla)
        if reinicia is not None:
            reinicia()
        total = 0.0
        done = False
        while not done:
            observacion, recompensa, done, info = env.step(accion(env, observacion))
            total += recompensa
        resultados["progreso"].append(info["progreso"])
        resultados["meta"].append(float(info["meta"]))
        resultados["recompensa"].append(total)
    if hasattr(env, "close"):
        env.close()
    return {k: np.asarray(v, dtype=float) for k, v in resultados.items()}


def referencias(cfg: dict[str, Any], semillas: list[int], semilla_azar: int) -> dict[str, dict[str, np.ndarray]]:
    """Las lineas contra las que se lee cualquier modelo.

    Los indices de accion no son los mismos en los dos entornos: en el propio la 2 es
    "derecha + carrera" y la 3 "derecha + salto"; en el NES, SIMPLE_MOVEMENT pone la 1 en
    "derecha", la 2 en "derecha + A" y la 4 en "derecha + A + B". La heuristica escrita a
    mano solo existe en el entorno propio, porque lee su mapa de casillas directamente.
    """
    generador = np.random.default_rng(semilla_azar)
    derecha, salto = (1, 4) if es_nes(cfg) else (2, 3)
    lineas = {
        "azar": juega(cfg, semillas, lambda env, obs: int(generador.integers(N_ACTIONS))),
        "siempre_dcha": juega(cfg, semillas, lambda env, obs: derecha),
        "salto_cte": juega(cfg, semillas, lambda env, obs: salto),
    }
    if not es_nes(cfg):
        lineas["heuristica"] = juega(cfg, semillas, lambda env, obs: scripted_action(env))
    return lineas


def juega_politica(
    cfg: dict[str, Any], semillas: list[int], politica: nn.Module, device: torch.device,
    muestrea: bool = False,
) -> dict[str, np.ndarray]:
    """Estado oculto reiniciado en cada nivel. `muestrea` sortea en vez de coger el maximo.

    Las dos se informan porque no dicen lo mismo: la codiciosa se queda pegada a una
    accion constante y la muestreada conserva la exploracion que la red aprendio.
    """
    estado = {"h": politica.initial_state(1, device)}

    def reinicia() -> None:
        estado["h"] = politica.initial_state(1, device)

    @torch.no_grad()
    def accion(env: Any, observacion: np.ndarray) -> int:
        tensor = torch.as_tensor(observacion, dtype=torch.float32, device=device)[None]
        logits, _, estado["h"] = politica(tensor, estado["h"])
        if muestrea:
            return int(torch.distributions.Categorical(logits=logits).sample().item())
        return int(logits.argmax(dim=1).item())

    return juega(cfg, semillas, accion, reinicia)


def carga_politica(
    ruta: Path, cfg: dict[str, Any], wirings: dict[str, Wiring], device: torch.device
) -> nn.Module:
    checkpoint = torch.load(ruta, map_location=device, weights_only=False)
    entorno = construye_entorno(cfg)
    politica = build_policy(
        checkpoint["modelo"], cfg, entorno.n_features, N_ACTIONS, int(checkpoint["semilla"]), wirings
    )
    faltan, sobran = politica.load_state_dict(checkpoint["state_dict"], strict=False)
    # Los checkpoints anteriores a la normalizacion de la entrada no traen estos buffers;
    # se quedan en media 0 y desviacion 1, que es exactamente como se entrenaron.
    esperadas = {"feature_mean", "feature_std"}
    if set(faltan) - esperadas or sobran:
        raise RuntimeError(f"checkpoint incompatible: faltan {faltan}, sobran {sobran}")
    if faltan:
        print(f"  aviso: {ruta.name} es anterior a la normalizacion de entrada")
    return politica.to(device).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--levels", type=int, default=None, help="niveles de test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--muestrea", action="store_true",
                        help="politica muestreada en vez de codiciosa")
    parser.add_argument("--repeticiones", type=int, default=1,
                        help="partidas por nivel; con politica muestreada hace falta mas de una")
    parser.add_argument("--checkpoint", type=Path, default=None, action="append",
                        help="evaluar estos checkpoints sueltos en vez de los del "
                             "output_dir; se puede repetir para comparar versiones")
    parser.add_argument("--semilla", type=int, default=0,
                        help="semilla del muestreo, para que la evaluacion sea reproducible")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = resolve_device(args.device or cfg.get("device", "auto"))
    # Regla 2 del repo: toda ejecucion registra su semilla. Sin fijarla, dos evaluaciones
    # muestreadas del MISMO checkpoint daban 0,818 y 0,701 de progreso.
    torch.manual_seed(args.semilla)
    np.random.seed(args.semilla)
    if args.muestrea and args.repeticiones == 1:
        print("aviso: politica muestreada con una sola partida por nivel; el ruido del"
              " muestreo es del tamano de las diferencias entre modelos")
    out_dir = Path(cfg["output_dir"])
    n_levels = args.levels or int(cfg.get("test_episodes", 256))
    semillas = niveles_de_test(cfg, n_levels)
    print(f"dispositivo: {device}   niveles de test: {n_levels}"
          f"   politica {'muestreada' if args.muestrea else 'codiciosa'}")

    series: dict[str, dict[str, np.ndarray]] = referencias(cfg, semillas, semilla_azar=0)
    wirings = load_wirings(cfg)
    # Con --checkpoint se comparan versiones de un mismo modelo (por ejemplo el mismo
    # entrenamiento a 300 y a 900 iteraciones) contra las MISMAS lineas de referencia y
    # con la misma semilla de muestreo. Cada checkpoint entra como una fila mas, con el
    # nombre del fichero, asi que la tabla y los contrastes salen igual.
    if args.checkpoint:
        a_evaluar = [(ruta.stem, [ruta]) for ruta in args.checkpoint]
    else:
        a_evaluar = [
            (name, [out_dir / f"{name}_s{seed}.pt" for seed in cfg["seeds"]])
            for name in cfg["models"]
        ]
    for name, rutas in a_evaluar:
        por_semilla = []
        for ruta in rutas:
            if not ruta.exists():
                continue
            politica = carga_politica(ruta, cfg, wirings, device)
            # Varias partidas por nivel y media: la politica muestreada es estocastica y
            # con una sola partida el ruido tapa cualquier diferencia entre cableados.
            partidas = [
                juega_politica(cfg, semillas, politica, device, args.muestrea)
                for _ in range(max(1, args.repeticiones))
            ]
            por_semilla.append({
                metrica: np.mean([p[metrica] for p in partidas], axis=0) for metrica in METRICAS
            })
            print(f"  {ruta.stem}: progreso {por_semilla[-1]['progreso'].mean():.3f}")
        if not por_semilla:
            print(f"  {name}: sin checkpoints en {out_dir}")
            continue
        # Media por nivel sobre las semillas de entrenamiento: la unidad del emparejado
        # sigue siendo el nivel, que es lo que comparten todos los modelos.
        series[name] = {
            metrica: np.mean([s[metrica] for s in por_semilla], axis=0) for metrica in METRICAS
        }

    resumen = {
        nombre: {
            metrica: {
                "media": float(valores[metrica].mean()),
                "desviacion": float(valores[metrica].std(ddof=1)),
            }
            for metrica in METRICAS
        }
        for nombre, valores in series.items()
    }

    permutaciones = int(cfg.get("permutations", 10000))
    contrastes: dict[str, Any] = {}
    # Contra "azar" tambien: un modelo puede ganarle a la politica degenerada y aun asi
    # no distinguirse de apretar botones al azar, que es lo que pasa en el NES a 190
    # iteraciones. Sin este contraste, el primer resultado se lee como si fuera bueno.
    parejas = [("mosca", "aleatorio"), ("mosca", "recableado"), ("mosca", "salto_cte"),
               ("mosca", "azar"), ("mosca", "sin_nucleo"), ("mosca", "mlp"),
               ("mosca", "denso")]
    if args.checkpoint:
        # Comparando versiones: cada una contra las dos lineas que deciden, y las demas
        # contra la primera, que es la referencia de "lo que habia antes".
        nombres = [ruta.stem for ruta in args.checkpoint]
        parejas = [(n, linea) for n in nombres for linea in ("salto_cte", "azar")]
        parejas += [(n, nombres[0]) for n in nombres[1:]]
    for a, b in parejas:
        if a in series and b in series:
            contrastes[f"{a}_vs_{b}"] = {
                metrica: prueba_de_permutacion(
                    series[a][metrica], series[b][metrica], permutaciones, semilla=0
                )
                for metrica in ("progreso", "meta")
            }

    ancho = max((len(n) for n in resumen), default=6)
    print(f"\n{'modelo':<{ancho + 2}}  progreso    meta")
    for nombre, valores in resumen.items():
        print(f"  {nombre:<{ancho}} {valores['progreso']['media']:.3f}     "
              f"{valores['meta']['media']:.3f}")
    for clave, prueba in contrastes.items():
        print(f"  {clave}: progreso dif {prueba['progreso']['diferencia_media']:+.4f}"
              f"  p = {prueba['progreso']['p']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # Con checkpoints sueltos no se pisa el informe de la evaluacion normal.
    destino = out_dir / ("rl_evaluacion_checkpoints.json" if args.checkpoint
                         else "rl_evaluacion.json")
    destino.write_text(json.dumps(
        {"config": cfg, "n_niveles": n_levels, "repeticiones": args.repeticiones,
         "muestreada": bool(args.muestrea), "semilla": args.semilla,
         "checkpoints": [str(r) for r in (args.checkpoint or [])],
         "resumen": resumen, "contrastes": contrastes},
        indent=2, ensure_ascii=False,
    ))
    print(f"\ninforme en {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
