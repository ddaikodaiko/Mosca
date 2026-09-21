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


def niveles_de_test(cfg: dict[str, Any], n_levels: int) -> list[int]:
    low, high = SeedRanges.from_config(cfg).range("test")
    return [low + i % (high - low) for i in range(n_levels)]


def construye_entorno(cfg: dict[str, Any]) -> PlatformerEnv:
    return PlatformerEnv(
        LevelConfig(**cfg.get("level", {})),
        ObservationConfig(**cfg.get("observation", {})),
        RewardConfig(**cfg.get("reward", {})),
        max_steps=int(cfg.get("max_episode_steps", 1200)),
    )


def juega(
    cfg: dict[str, Any], semillas: list[int], accion: Callable[[PlatformerEnv, np.ndarray], int],
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
    return {k: np.asarray(v, dtype=float) for k, v in resultados.items()}


def referencias(cfg: dict[str, Any], semillas: list[int], semilla_azar: int) -> dict[str, dict[str, np.ndarray]]:
    generador = np.random.default_rng(semilla_azar)
    return {
        "azar": juega(cfg, semillas, lambda env, obs: int(generador.integers(N_ACTIONS))),
        "siempre_dcha": juega(cfg, semillas, lambda env, obs: 2),
        "salto_cte": juega(cfg, semillas, lambda env, obs: 3),
        "heuristica": juega(cfg, semillas, lambda env, obs: scripted_action(env)),
    }


def juega_politica(
    cfg: dict[str, Any], semillas: list[int], politica: nn.Module, device: torch.device
) -> dict[str, np.ndarray]:
    """Politica codiciosa, con el estado oculto reiniciado en cada nivel."""
    estado = {"h": politica.initial_state(1, device)}

    def reinicia() -> None:
        estado["h"] = politica.initial_state(1, device)

    @torch.no_grad()
    def accion(env: PlatformerEnv, observacion: np.ndarray) -> int:
        tensor = torch.as_tensor(observacion, dtype=torch.float32, device=device)[None]
        logits, _, estado["h"] = politica(tensor, estado["h"])
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
    politica.load_state_dict(checkpoint["state_dict"])
    return politica.to(device).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--levels", type=int, default=None, help="niveles de test")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = resolve_device(args.device or cfg.get("device", "auto"))
    out_dir = Path(cfg["output_dir"])
    n_levels = args.levels or int(cfg.get("test_episodes", 256))
    semillas = niveles_de_test(cfg, n_levels)
    print(f"dispositivo: {device}   niveles de test: {n_levels}")

    series: dict[str, dict[str, np.ndarray]] = referencias(cfg, semillas, semilla_azar=0)
    wirings = load_wirings(cfg)
    for name in cfg["models"]:
        por_semilla = []
        for seed in cfg["seeds"]:
            ruta = out_dir / f"{name}_s{seed}.pt"
            if not ruta.exists():
                continue
            politica = carga_politica(ruta, cfg, wirings, device)
            por_semilla.append(juega_politica(cfg, semillas, politica, device))
            print(f"  {name} s{seed}: progreso {por_semilla[-1]['progreso'].mean():.3f}")
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
    for a, b in (("mosca", "aleatorio"), ("mosca", "recableado"), ("mosca", "salto_cte"),
                 ("mosca", "sin_nucleo"), ("mosca", "mlp"), ("mosca", "denso")):
        if a in series and b in series:
            contrastes[f"{a}_vs_{b}"] = {
                metrica: prueba_de_permutacion(
                    series[a][metrica], series[b][metrica], permutaciones, semilla=0
                )
                for metrica in ("progreso", "meta")
            }

    print("\nmodelo            progreso    meta")
    for nombre, valores in resumen.items():
        print(f"  {nombre:<16} {valores['progreso']['media']:.3f}     {valores['meta']['media']:.3f}")
    for clave, prueba in contrastes.items():
        print(f"  {clave}: progreso dif {prueba['progreso']['diferencia_media']:+.4f}"
              f"  p = {prueba['progreso']['p']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    destino = out_dir / "rl_evaluacion.json"
    destino.write_text(json.dumps(
        {"config": cfg, "n_niveles": n_levels, "resumen": resumen, "contrastes": contrastes},
        indent=2, ensure_ascii=False,
    ))
    print(f"\ninforme en {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
