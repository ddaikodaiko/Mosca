"""Varianza explicada del critico: el diagnostico estandar de PPO (fase 6).

    python diagnostico_rl.py artifacts/mario_nes_demo/mosca_s0.pt
    python diagnostico_rl.py <checkpoint> --config configs/mario_nes.yaml --episodios 8

PPO mejora la politica siguiendo la VENTAJA, que es el retorno menos el valor predicho.
Si el critico no explica la varianza del retorno, la ventaja es casi todo ruido y la
politica avanza a ciegas por muchas iteraciones que se le echen. Medirlo cuesta unos
episodios y distingue dos diagnosticos que llevan a arreglos distintos: "este cableado no
aprende" y "aqui no aprende nadie porque el critico esta ciego".

Se compara el valor predicho en cada paso con el retorno descontado REAL desde ese paso,
jugando episodios completos con la politica muestreada.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from data.spectral import resolve_device
from evaluate_rl import carga_politica, construye_entorno
from train_rl import load_wirings


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/mario_nes_demo.yaml"))
    parser.add_argument("--episodios", type=int, default=4)
    parser.add_argument("--semilla", type=int, default=101,
                        help="primera semilla de nivel; se usan `episodios` consecutivas")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = resolve_device(args.device or cfg.get("device", "auto"))
    politica = carga_politica(args.checkpoint, cfg, load_wirings(cfg), device)
    gamma = float(cfg["ppo"]["gamma"])
    env = construye_entorno(cfg)

    valores: list[np.ndarray] = []
    retornos_reales: list[np.ndarray] = []
    # Actividad de las neuronas que lee el critico y el actor. Si apenas varia entre
    # estados, ni el valor ni la politica pueden depender de lo que pasa en pantalla,
    # por mucho que se entrene: la lectura es lineal sobre estas motoras.
    motoras: list[np.ndarray] = []
    for semilla in range(args.semilla, args.semilla + args.episodios):
        observacion = env.reset(semilla)
        h = politica.initial_state(1, device)
        predichos: list[float] = []
        recompensas: list[float] = []
        done = False
        while not done:
            with torch.no_grad():
                entrada = torch.as_tensor(observacion, dtype=torch.float32,
                                          device=device).unsqueeze(0)
                logits, valor, h = politica(entrada, h)
                accion = int(torch.distributions.Categorical(logits=logits).sample().item())
            predichos.append(float(valor.item()))
            motoras.append(h[0, politica.motor].cpu().numpy().copy())
            observacion, recompensa, done, info = env.step(accion)
            recompensas.append(float(recompensa))

        # Retorno descontado real desde cada paso, hacia atras.
        retorno = np.zeros(len(recompensas))
        acumulado = 0.0
        for t in range(len(recompensas) - 1, -1, -1):
            acumulado = recompensas[t] + gamma * acumulado
            retorno[t] = acumulado
        valores.append(np.asarray(predichos))
        retornos_reales.append(retorno)
        print(f"  semilla {semilla}: {len(recompensas):4d} pasos, "
              f"retorno inicial {retorno[0]:7.1f}, valor predicho {predichos[0]:7.1f}, "
              f"progreso {info['progreso']:.3f}")

    if hasattr(env, "close"):
        env.close()

    actividad = np.stack(motoras)            # (pasos, n_motoras)
    v = np.concatenate(valores)
    r = np.concatenate(retornos_reales)
    ev = 1 - np.var(r - v) / max(float(np.var(r)), 1e-9)
    print(f"\nvarianza explicada del critico: {ev:+.3f}"
          "   (1 = perfecto, 0 = predice la media, <0 = peor que la media)")
    print(f"retorno real: media {r.mean():.1f} desv {r.std():.1f}"
          f"   |   valor: media {v.mean():.1f} desv {v.std():.1f}")
    if ev < 0.3:
        print("\nEl critico apenas explica el retorno: las ventajas son sobre todo ruido y"
              "\nla politica avanza a ciegas. Antes de culpar al cableado, arreglar esto.")

    # De donde viene: el actor y el critico son lineales sobre las motoras.
    por_neurona = actividad.std(axis=0)
    saturadas = float((np.abs(actividad) > 0.99).mean())
    print(f"\nmotoras ({actividad.shape[1]} neuronas, {actividad.shape[0]} pasos):"
          f"  |h| medio {np.abs(actividad).mean():.3f}"
          f"   desviacion ENTRE pasos: mediana {np.median(por_neurona):.4f},"
          f" maxima {por_neurona.max():.4f}")
    print(f"  neuronas practicamente constantes (desv < 0,01): "
          f"{int((por_neurona < 0.01).sum())} de {actividad.shape[1]}"
          f"   |   saturadas |h|>0,99: {100 * saturadas:.1f} %")
    if np.median(por_neurona) < 0.05:
        print("  La lectura casi no se mueve entre estados: el critico solo puede predecir"
              "\n  una constante, y eso es exactamente lo que hace.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
