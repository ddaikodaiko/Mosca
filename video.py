"""Video para redes: el cerebro de la mosca encendiendose al lado del fluido que predice.

1920x1080, H.264 yuv420p (lo que acepta Twitter/X). Textos en ingles.

Todo lo que se ve sale del modelo, sin adornos inventados:
- gris: las ~138.600 neuronas de FlyWire v783 en su posicion real;
- puntos: las neuronas del nucleo con su activacion `h` (escala real de tanh);
- lineas: las sinapsis por las que pasa mas corriente en ese instante, |w_ij * h_j|,
  con el signo de la presinaptica (naranja excita, azul inhibe);
- derecha: el campo real y la prediccion desplegada sobre sus propias salidas, con la
  correlacion contra la realidad. La interpolacion de los campos es solo de presentacion.

Encuadre: una red cableada con el conectoma de la mosca, no una mosca.

    python video.py --config configs/ensayo_32.yaml --models mosca --seed 0
    python video.py --config configs/ensayo_32.yaml --models mosca aleatorio --seed 0
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
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, zoom

from models.masked_rnn import ConnectomeFlowModel
from train import Trajectories, build_model, resolve_device
from visor import ANATOMICAL, ANNOTATIONS, brain_positions, correlation, run_model

BG = "#05060a"
TEXT = "#f1f3f6"
MUTED = "#8b93a1"
ORANGE = np.array([255, 122, 69]) / 255
BLUE = np.array([77, 170, 255]) / 255
ACCENT = "#ffd166"
LABELS = {"mosca": "fly wiring", "aleatorio": "random wiring", "recableado": "shuffled fly wiring"}
COLORS = {"mosca": "#ffd166", "aleatorio": "#7fd1b9", "recableado": "#c792ea"}
FLUID = LinearSegmentedColormap.from_list(
    "fluid", ["#9be7ff", "#1f7ae0", "#0a1a3a", "#05060a", "#3a0f0a", "#ff5a36", "#ffe29a"]
)
BRAIN = LinearSegmentedColormap.from_list("brain", [BG, "#1a2230", "#52627a", "#9fb2cc"])
TOP_EDGES = 350


def rotate(xyz: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    """FlyWire: x lateral, y dorsal->ventral, z antero-posterior."""
    c, s = np.cos(yaw), np.sin(yaw)
    x, z = xyz[:, 0] * c + xyz[:, 2] * s, -xyz[:, 0] * s + xyz[:, 2] * c
    cp, sp = np.cos(pitch), np.sin(pitch)
    return np.stack([x, xyz[:, 1] * cp - z * sp, xyz[:, 1] * sp + z * cp], axis=1)


def smooth(field: np.ndarray, factor: int) -> np.ndarray:
    """Solo presentacion: interpolacion cubica periodica."""
    padded = np.pad(field, 2, mode="wrap")
    return zoom(padded, factor, order=3)[2 * factor : -2 * factor, 2 * factor : -2 * factor]


def colour(values: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    rgb = np.where(values[:, None] >= 0, ORANGE, BLUE)
    return np.concatenate([rgb, np.clip(alpha, 0, 1)[:, None]], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=["mosca"])
    parser.add_argument("--seed", type=int, default=0, help="semilla de entrenamiento")
    parser.add_argument("--trajectory", type=int, default=0, help="trayectoria de test")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--interp", type=int, default=2, help="fotogramas por subpaso")
    parser.add_argument("--frames", type=int, default=None, help="solo para pruebas")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not ANNOTATIONS.is_file():
        print(f"falta {ANNOTATIONS}: python -m data.download_annotations")
        return 1
    cfg = yaml.safe_load(args.config.read_text())
    device = resolve_device(args.device or cfg.get("device", "auto"))
    out_dir = Path(cfg["output_dir"])
    test = Trajectories.load(Path(cfg["data_dir"]) / "test.npz", device)
    dt_save = float(json.loads((Path(cfg["data_dir"]) / "meta.json").read_text())["dt_save"])
    truth = test.omega[args.trajectory, : args.steps + 1].cpu().numpy()
    steps = truth.shape[0] - 1
    substeps = int(cfg["core"]["n_internal_steps"])

    xyz, has_position = brain_positions()
    centre = np.nanmean(xyz[has_position], axis=0)
    extent = float(np.nanmax(np.abs(xyz[has_position] - centre)))
    background = (xyz[has_position] - centre) / extent
    nodes = np.load(Path(cfg["wiring_dir"]) / "mosca.npz")["nodes"]
    active = np.nan_to_num((xyz[nodes] - centre) / extent)

    runs = []
    for name in args.models:
        path = out_dir / "checkpoints" / f"{name}_s{args.seed}.pt"
        if name not in ANATOMICAL or not path.is_file():
            print(f"se salta {name}: sin checkpoint o sin identidad anatomica")
            continue
        model = build_model(name, cfg, test.grid, args.seed).to(device)
        assert isinstance(model, ConnectomeFlowModel)
        model.load_state_dict(torch.load(path, map_location=device)["state_dict"])
        model.eval()
        out = run_model(model, truth, device)
        with torch.no_grad():
            weights = model.core.weights().cpu().numpy()
        post, pre = model.core.edge_index.cpu().numpy()
        runs.append({"name": name, **out, "corr": correlation(out["fields"], truth[1:]),
                     "weights": weights, "post": post, "pre": pre})
        print(f"{name}: correlacion a t={dt_save:.1f} {runs[-1]['corr'][0]:.3f}, "
              f"a t={steps * dt_save:.1f} {runs[-1]['corr'][-1]:.3f}")
    if not runs:
        return 1
    run = runs[0]  # el cerebro que se pinta

    n_frames = steps * substeps * args.interp
    if args.frames:
        n_frames = min(n_frames, args.frames)
    vmax = float(np.percentile(np.abs(truth), 99.5))
    factor = 12

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    fig.text(0.035, 0.905, "I wired an AI like a fruit fly's brain", color=TEXT, fontsize=40, fontweight="bold")
    fig.text(0.036, 0.848, "and asked it to predict a swirling fluid", color=ACCENT, fontsize=27)

    # ---------- cerebro ----------
    xlim, ylim = 1.02, 0.6
    brain_ax = fig.add_axes([0.0, 0.15, 0.56, 0.66], facecolor=BG)
    brain_ax.set_xlim(-xlim, xlim)
    brain_ax.set_ylim(ylim, -ylim)
    brain_ax.axis("off")
    density = brain_ax.imshow(np.zeros((10, 10)), extent=(-xlim, xlim, ylim, -ylim), cmap=BRAIN,
                              vmin=0, vmax=1, interpolation="bilinear", aspect="auto")
    edges = LineCollection([], linewidths=1.1, zorder=2)
    brain_ax.add_collection(edges)
    halos = brain_ax.scatter(np.zeros(1), np.zeros(1), s=[0], linewidths=0, zorder=3)
    cores = brain_ax.scatter(np.zeros(1), np.zeros(1), s=[0], linewidths=0, zorder=4)
    fig.text(0.035, 0.13, f"{len(nodes)} neurons · their real connections from the FlyWire connectome",
             color=MUTED, fontsize=15)
    fig.text(0.035, 0.1, "lines: synapses carrying the most signal right now  ·  orange excites · blue inhibits",
             color=MUTED, fontsize=13)

    # ---------- fluidos ----------
    panels = [("real fluid", TEXT)] + [(f"AI · {LABELS[r['name']]}", COLORS[r["name"]]) for r in runs]
    n_panels = len(panels)
    right_left, right_width, gap = 0.575, 0.395, 0.02
    width = (right_width - gap * (n_panels - 1)) / n_panels
    height = width * 1920 / 1080
    fields_bottom = 0.8 - height
    field_images, corr_labels = [], []
    for k, (caption, color) in enumerate(panels):
        ax = fig.add_axes([right_left + k * (width + gap), fields_bottom, width, height], facecolor=BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2.5)
        field_images.append(ax.imshow(smooth(truth[1], factor), cmap=FLUID, vmin=-vmax, vmax=vmax,
                                      interpolation="bilinear"))
        ax.set_title(caption, color=color, fontsize=18 if n_panels < 3 else 14, loc="left", pad=8,
                     fontweight="bold")
        if k > 0:
            corr_labels.append(ax.text(0.5, -0.05, "", transform=ax.transAxes, ha="center", va="top",
                                       color=color, fontsize=22 if n_panels < 3 else 17, fontweight="bold"))

    meter_top = fields_bottom - 0.1
    meter_ax = fig.add_axes([right_left + 0.04, 0.1, right_width - 0.04, meter_top - 0.1], facecolor=BG)
    meter_ax.set_xlim(0, steps * dt_save)
    meter_ax.set_ylim(0, 1.02)
    for side in ("top", "right"):
        meter_ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        meter_ax.spines[side].set_color("#2a303a")
    meter_ax.tick_params(colors=MUTED, labelsize=12)
    meter_ax.set_yticks([0, 0.5, 1])
    meter_ax.set_yticklabels(["0%", "50%", "100%"])
    meter_ax.set_xlabel("time", color=MUTED, fontsize=13)
    meter_ax.set_title("match with the real fluid", color=MUTED, fontsize=15, loc="left")
    t_axis = np.arange(1, steps + 1) * dt_save
    lines = []
    for r in runs:
        (line,) = meter_ax.plot([], [], color=COLORS[r["name"]], lw=3.5, label=f"AI · {LABELS[r['name']]}")
        lines.append(line)
    persistence_corr = correlation(np.repeat(truth[:1], steps, 0), truth[1:])
    (persistence_line,) = meter_ax.plot([], [], color="#c9ced6", lw=1.6, ls="--", zorder=5,
                                        label="no AI: freeze the first frame")
    meter_ax.legend(loc="lower left", frameon=False, labelcolor=TEXT, fontsize=12)

    fig.text(0.035, 0.03, "Not a fly: its wiring diagram is the blueprint of an artificial network. "
             "The AI's prediction feeds on itself.", color="#5f6775", fontsize=12)
    clock = fig.text(0.965, 0.03, "", color=MUTED, fontsize=14, ha="right")

    activity = run["activity"]
    weights, post, pre = run["weights"], run["post"], run["pre"]
    n_top = min(TOP_EDGES, weights.size - 1)

    def draw(frame: int) -> None:
        position = frame / args.interp
        sub = min(int(position), steps * substeps - 1)
        nxt = min(sub + 1, steps * substeps - 1)
        weight = position - int(position)
        h = (1 - weight) * activity[sub] + weight * activity[nxt]
        step = min(sub // substeps, steps - 1)

        phase = 2 * np.pi * frame / max(steps * substeps * args.interp, 1)
        yaw, pitch = 0.5 * np.sin(phase), 0.12 * np.sin(2 * phase)
        rotated = rotate(background, yaw, pitch)
        hist, _, _ = np.histogram2d(rotated[:, 1], rotated[:, 0], bins=(300, 520),
                                    range=((-ylim, ylim), (-xlim, xlim)))
        hist = np.log1p(hist)
        glow = gaussian_filter(hist, 6)
        density.set_data(np.clip(0.75 * hist / hist.max() + 0.6 * glow / glow.max(), 0, 1))

        points = rotate(active, yaw, pitch)[:, :2]
        flow = weights * h[pre]  # corriente que llega por cada sinapsis en este subpaso
        top = np.argpartition(-np.abs(flow), n_top)[:n_top]
        strength = np.abs(flow[top]) / (np.abs(flow[top]).max() + 1e-9)
        edges.set_segments(np.stack([points[pre[top]], points[post[top]]], axis=1))
        edges.set_color(colour(flow[top], 0.05 + 0.55 * strength ** 1.5))

        order = np.argsort(np.abs(h))
        magnitude = np.abs(h[order])
        halos.set_offsets(points[order])
        halos.set_sizes(40 + 1600 * magnitude ** 2)
        halos.set_color(colour(h[order], 0.12 * magnitude))
        cores.set_offsets(points[order])
        cores.set_sizes(6 + 160 * magnitude ** 1.5)
        cores.set_color(colour(h[order], 0.35 + 0.65 * magnitude))

        field_images[0].set_data(smooth(truth[step + 1], factor))
        for image, label, r in zip(field_images[1:], corr_labels, runs):
            image.set_data(smooth(r["fields"][step], factor))
            label.set_text(f"{max(r['corr'][step], 0) * 100:.0f}% match")
        for line, r in zip(lines, runs):
            line.set_data(t_axis[: step + 1], r["corr"][: step + 1])
        persistence_line.set_data(t_axis[: step + 1], persistence_corr[: step + 1])
        clock.set_text(f"t = {(step + 1) * dt_save:.1f}")

    # libopenh264: el ffmpeg de Fedora no trae libx264. Sigue siendo H.264, que es lo que
    # acepta Twitter/X; sin -crf, la calidad se fija por bitrate.
    writer = FFMpegWriter(fps=args.fps, codec="libopenh264", bitrate=9000,
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    names = "_".join(r["name"] for r in runs)
    out = args.out or out_dir / "video" / f"fly_fluid_{names}_s{args.seed}_traj{args.trajectory}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(out), dpi=100):
        for frame in range(n_frames):
            draw(frame)
            writer.grab_frame(facecolor=BG)
            if frame % 60 == 0:
                print(f"  fotograma {frame}/{n_frames}", flush=True)
    print(f"video en {out}  ({n_frames / args.fps:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
