"""Visualizador de actividad: el campo de vorticidad y el nucleo encendiendose.

Version sin coordenadas anatomicas (la anatomica es visor.py).
Cada neurona es un pixel de una rejilla, ordenadas por papel y luego por grado:
sensoriales exclusivas, compartidas (sensorial y motora a la vez), motoras exclusivas y
el resto. Sirve de diagnostico: se ve si el nucleo se satura o se apaga.

Fila de arriba: campo real y prediccion desplegada de cada modelo (test).
Fila de abajo: |h| medio a lo largo del tiempo, y la rejilla de actividad de cada modelo,
un fotograma por subpaso interno del nucleo.

    python visualize.py --config configs/ensayo_32.yaml --models mosca aleatorio --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.animation import FFMpegWriter, PillowWriter
from torch import Tensor

from models.masked_rnn import ConnectomeFlowModel
from train import Trajectories, build_model, resolve_device


def neuron_order(model: ConnectomeFlowModel) -> tuple[np.ndarray, list[int]]:
    """Orden de pintado y fronteras entre grupos (sensorial, compartida, motora, resto)."""
    n = model.core.n_neurons
    sensory = set(model.sensory.tolist())
    motor = set(model.motor.tolist())
    edges = model.core.edge_index.cpu().numpy()
    degree = np.bincount(edges.ravel(), minlength=n)
    groups = [
        sorted(sensory - motor),
        sorted(sensory & motor),
        sorted(motor - sensory),
        sorted(set(range(n)) - sensory - motor),
    ]
    order: list[int] = []
    bounds: list[int] = []
    for group in groups:
        order.extend(sorted(group, key=lambda i: -degree[i]))
        bounds.append(len(order))
    return np.array(order), bounds[:-1]


@torch.no_grad()
def record_rollout(model: ConnectomeFlowModel, omega0: Tensor, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Despliegue de una trayectoria: campos (steps, g, g) y actividad (steps*sub, N)."""
    omega, h = omega0[None], None
    fields, activity = [], []
    for _ in range(steps):
        omega, h, trace = model(omega, h, record=True)
        fields.append(omega[0].cpu().numpy())
        activity.extend(state[0].cpu().numpy() for state in trace)
    return np.stack(fields), np.stack(activity)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=["mosca", "aleatorio"])
    parser.add_argument("--seed", type=int, default=0, help="semilla de entrenamiento")
    parser.add_argument("--trajectory", type=int, default=0, help="trayectoria de test")
    parser.add_argument("--start", type=int, default=0, help="instante inicial guardado")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = resolve_device(args.device or cfg.get("device", "auto"))
    out_dir = Path(cfg["output_dir"])
    test = Trajectories.load(Path(cfg["data_dir"]) / "test.npz", device)
    grid = test.grid
    dt_save = float(json.loads((Path(cfg["data_dir"]) / "meta.json").read_text())["dt_save"])
    truth = test.omega[args.trajectory, args.start : args.start + args.steps + 1].cpu().numpy()
    steps = truth.shape[0] - 1

    runs = []
    for name in args.models:
        path = out_dir / "checkpoints" / f"{name}_s{args.seed}.pt"
        if not path.is_file():
            print(f"falta {path}, se salta {name}")
            continue
        model = build_model(name, cfg, grid, args.seed).to(device)
        if not isinstance(model, ConnectomeFlowModel):
            print(f"{name} no tiene nucleo recurrente, se salta")
            continue
        model.load_state_dict(torch.load(path, map_location=device)["state_dict"])
        model.eval()
        fields, activity = record_rollout(model, torch.from_numpy(truth[0]).to(device), steps)
        order, bounds = neuron_order(model)
        side = int(np.ceil(np.sqrt(model.core.n_neurons)))
        # Escala robusta por modelo: |h| es diminuto fuera de las sensoriales y con una
        # escala comun no se veria nada. Se imprime para no enganar.
        scale = float(np.percentile(np.abs(activity), 99)) or 1.0
        runs.append({"name": name, "fields": fields, "activity": activity[:, order],
                     "bounds": bounds, "side": side, "scale": scale})
        print(f"{name}: N={model.core.n_neurons}  |h| medio {np.abs(activity).mean():.4f}"
              f"  escala de color +-{scale:.4f}")
    if not runs:
        print("no hay checkpoints que animar")
        return 1

    substeps = runs[0]["activity"].shape[0] // steps
    vmax = float(np.abs(truth).max())
    n_cols = 1 + len(runs)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6.6), squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    field_images = [axes[0, 0].imshow(truth[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax)]
    axes[0, 0].set_title("real (test)")
    for col, run in enumerate(runs, start=1):
        field_images.append(axes[0, col].imshow(truth[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax))
        axes[0, col].set_title(f"{run['name']} s{args.seed}")

    trace_ax = axes[1, 0]
    trace_ax.set_xticks(np.linspace(0, steps * substeps, 4))
    trace_ax.set_xticklabels([f"{t:.1f}" for t in np.linspace(0, steps * dt_save, 4)])
    trace_ax.set_xlabel("t")
    trace_ax.set_title("|h| medio")
    cursor = trace_ax.axvline(0, color="0.5", lw=0.8)
    for run in runs:
        trace_ax.plot(np.abs(run["activity"]).mean(axis=1), label=run["name"])
    trace_ax.legend(fontsize=7)
    trace_ax.set_yticks([])

    activity_images = []
    for col, run in enumerate(runs, start=1):
        n = run["activity"].shape[1]
        pad = run["side"] ** 2 - n
        frame = np.pad(run["activity"][0], (0, pad), constant_values=np.nan)
        ax = axes[1, col]
        activity_images.append(ax.imshow(
            frame.reshape(run["side"], run["side"]), cmap="RdBu_r",
            vmin=-run["scale"], vmax=run["scale"], interpolation="nearest",
        ))
        for bound in run["bounds"]:  # fronteras entre grupos de neuronas
            row, c = divmod(bound, run["side"])
            ax.plot([c - 0.5, c - 0.5], [row - 0.5, row + 0.5], color="k", lw=1)
            ax.axhline(row + 0.5, xmin=c / run["side"], color="k", lw=0.5)
        ax.set_title(f"actividad (+-{run['scale']:.3f})", fontsize=9)
        ax.set_xlabel("sens | compart | motor | resto", fontsize=7)

    time_text = fig.suptitle("")
    fig.tight_layout()

    def draw(frame: int) -> list:
        step = frame // substeps
        field_images[0].set_data(truth[min(step + 1, steps)])
        for image, run in zip(field_images[1:], runs):
            image.set_data(run["fields"][step])
        for image, run in zip(activity_images, runs):
            n = run["activity"].shape[1]
            values = np.pad(run["activity"][frame], (0, run["side"] ** 2 - n), constant_values=np.nan)
            image.set_data(values.reshape(run["side"], run["side"]))
        cursor.set_xdata([frame, frame])
        time_text.set_text(f"t = {(step + 1) * dt_save:.1f}   subpaso {frame % substeps + 1}/{substeps}")
        return [*field_images, *activity_images, cursor, time_text]

    from matplotlib.animation import FuncAnimation

    animation = FuncAnimation(fig, draw, frames=steps * substeps, blit=False)
    out = args.out or out_dir / "figs" / f"actividad_s{args.seed}_traj{args.trajectory}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps) if out.suffix == ".mp4" else PillowWriter(fps=args.fps)
    animation.save(out, writer=writer, dpi=90)
    print(f"animacion en {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
