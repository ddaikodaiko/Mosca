"""Visor anatomico: el nucleo encendiendose sobre la silueta del cerebro de la mosca.

Genera una pagina HTML autocontenida (se abre sin servidor). Las ~138.600 neuronas de
FlyWire v783 en gris dibujan el cerebro; encima, las neuronas del subgrafo se encienden
con su activacion `h` en cada subpaso del nucleo, al lado del campo de vorticidad real y
del predicho. Arrastrar gira el cerebro.

`mosca` y `recableado` usan esas neuronas con su cableado real o barajado; `aleatorio`
usa las mismas neuronas con aristas al azar, asi que tambien se puede pintar encima. Lo
que se ve es un diagrama de conexiones usado como mascara, no una mosca (seccion 2 del
diseno). `denso` no tiene identidad anatomica y no se pinta.

    python -m data.download_annotations      # una vez, 32 MB
    python visor.py --config configs/ensayo_32.yaml --models mosca aleatorio --seed 0
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from models.masked_rnn import ConnectomeFlowModel
from train import Trajectories, build_model, resolve_device

RAW = Path("data/raw")
ANNOTATIONS = RAW / "Supplemental_file1_neuron_annotations.tsv"
ANATOMICAL = ("mosca", "aleatorio", "recableado")
VOXEL_NM = np.array([4.0, 4.0, 40.0])
TEMPLATE = Path(__file__).resolve().parent / "viz" / "visor_template.html"


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def brain_positions() -> tuple[np.ndarray, np.ndarray]:
    """Posiciones en nm, en el orden de los indices de Shiu, y mascara de las que existen."""
    ids = pd.read_csv(RAW / "Completeness_783.csv", usecols=[0]).iloc[:, 0].to_numpy()
    annotations = pd.read_csv(
        ANNOTATIONS, sep="\t", usecols=["root_id", "pos_x", "pos_y", "pos_z"],
        dtype={"pos_x": "float32", "pos_y": "float32", "pos_z": "float32"},
    )
    positions = annotations.drop_duplicates("root_id").set_index("root_id").reindex(ids)
    xyz = positions.to_numpy(dtype=np.float64) * VOXEL_NM
    return xyz, ~np.isnan(xyz).any(axis=1)


def quantise(values: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.round(values / scale * 127), -127, 127).astype(np.int8)


def quantise16(values: np.ndarray) -> np.ndarray:
    """Activaciones de tanh en int16: resolucion 3e-5. En int8 el interior del nucleo
    (|h| ~ 0,01) quedaria en 0 o +-1 y el visor mentiria."""
    return np.clip(np.round(values * 32767), -32767, 32767).astype(np.int16)


def correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Correlacion de Pearson por instante entre campos (steps, g, g)."""
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))


@torch.no_grad()
def run_model(model: ConnectomeFlowModel, truth: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    """Dos pasadas sobre la misma trayectoria real.

    Despliegue: el modelo se alimenta de su propia prediccion (lo que mide evaluate.py).
    Un paso: en cada instante recibe el campo REAL y se guarda el incremento que predice,
    que es exactamente lo que se entrena a resolver (prediccion residual).
    """
    steps = truth.shape[0] - 1
    omega, h = torch.from_numpy(truth[0])[None].to(device), None
    fields, activity = [], []
    for _ in range(steps):
        omega, h, trace = model(omega, h, record=True)
        fields.append(omega[0].cpu().numpy())
        activity.extend(state[0].cpu().numpy() for state in trace)
    h = None
    deltas = []
    for t in range(steps):
        current = torch.from_numpy(truth[t])[None].to(device)
        prediction, h, _ = model(current, h)
        deltas.append((prediction - current)[0].cpu().numpy())
    return {"fields": np.stack(fields), "activity": np.stack(activity), "deltas": np.stack(deltas)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=["mosca", "aleatorio"])
    parser.add_argument("--seed", type=int, default=0, help="semilla de entrenamiento")
    parser.add_argument("--trajectory", type=int, default=0, help="trayectoria de test")
    parser.add_argument("--steps", type=int, default=50, help="pasos de fluido, hasta 199")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not ANNOTATIONS.is_file():
        print(f"falta {ANNOTATIONS}: python -m data.download_annotations")
        return 1
    cfg = yaml.safe_load(args.config.read_text())
    device = resolve_device(args.device or cfg.get("device", "auto"))
    out_dir = Path(cfg["output_dir"])
    wiring_dir = Path(cfg["wiring_dir"])
    test = Trajectories.load(Path(cfg["data_dir"]) / "test.npz", device)
    meta = json.loads((Path(cfg["data_dir"]) / "meta.json").read_text())
    truth = test.omega[args.trajectory, : args.steps + 1].cpu().numpy()
    steps = truth.shape[0] - 1
    field_scale = float(np.abs(truth).max())

    xyz, has_position = brain_positions()
    centre = np.nanmean(xyz[has_position], axis=0)
    extent = float(np.nanmax(np.abs(xyz[has_position] - centre)))
    background = ((xyz[has_position] - centre) / extent * 32767).astype(np.int16)
    nodes = np.load(wiring_dir / "mosca.npz")["nodes"]
    if not has_position[nodes].all():
        print(f"aviso: {int((~has_position[nodes]).sum())} neuronas del subgrafo sin posicion")
    active = np.nan_to_num((xyz[nodes] - centre) / extent * 32767).astype(np.int16)

    true_delta = truth[1:] - truth[:-1]
    delta_scale = float(np.abs(true_delta).max())
    runs = []
    for name in args.models:
        if name not in ANATOMICAL:
            print(f"{name} no tiene identidad anatomica, se salta")
            continue
        path = out_dir / "checkpoints" / f"{name}_s{args.seed}.pt"
        if not path.is_file():
            print(f"falta {path}, se salta {name}")
            continue
        model = build_model(name, cfg, test.grid, args.seed).to(device)
        assert isinstance(model, ConnectomeFlowModel)
        model.load_state_dict(torch.load(path, map_location=device)["state_dict"])
        model.eval()
        out = run_model(model, truth, device)
        fields, activity, deltas = out["fields"], out["activity"], out["deltas"]
        roles = np.zeros(model.core.n_neurons, dtype=np.int8)
        roles[model.sensory.cpu().numpy()] |= 1
        roles[model.motor.cpu().numpy()] |= 2
        groups = {  # |h| medio por papel: si la senal no llega a las motoras, no hay salida
            "sensoriales": roles == 1, "compartidas": roles == 3,
            "motoras": roles == 2, "interior": roles == 0,
        }
        error = np.linalg.norm(fields - truth[1:], axis=(1, 2)) / np.linalg.norm(truth[1:], axis=(1, 2))
        delta_error = np.linalg.norm(deltas - true_delta, axis=(1, 2)) / np.linalg.norm(true_delta, axis=(1, 2))
        runs.append({
            "name": name,
            "enhance": float(np.percentile(np.abs(activity), 99)) or 1.0,
            "groups": {k: [float(v) for v in np.abs(activity[:, m]).mean(axis=1)] for k, m in groups.items()},
            "error": [float(v) for v in error],
            "corr": [float(v) for v in correlation(fields, truth[1:])],
            "deltaError": [float(v) for v in delta_error],
            "activity": b64(quantise16(activity)),
            "fields": b64(quantise(fields, field_scale)),
            "deltas": b64(quantise(deltas, delta_scale)),
            "roles": b64(roles),
        })
        print(f"{name}: |h| medio {np.abs(activity).mean():.4f}"
              f"  (interior {np.abs(activity[:, groups['interior']]).mean():.4f})"
              f"  error L2 a {steps} pasos {error[-1]:.3f}"
              f"  error del incremento a 1 paso {delta_error.mean():.3f} (persistencia 1,000)")
    if not runs:
        print("no hay checkpoints que pintar")
        return 1

    persistence = np.linalg.norm(truth[0] - truth[1:], axis=(1, 2)) / np.linalg.norm(truth[1:], axis=(1, 2))
    payload = {
        "config": str(args.config),
        "seed": args.seed,
        "trajectory": args.trajectory,
        "grid": test.grid,
        "steps": steps,
        "substeps": int(cfg["core"]["n_internal_steps"]),
        "dtSave": float(meta["dt_save"]),
        "nNeurons": int(nodes.size),
        "nBackground": int(background.shape[0]),
        "background": b64(background),
        "active": b64(active),
        "truth": b64(quantise(truth[1:], field_scale)),
        "trueDelta": b64(quantise(true_delta, delta_scale)),
        "initial": b64(quantise(truth[0], field_scale)),
        "fieldScale": field_scale,
        "deltaScale": delta_scale,
        "persistenceError": [float(v) for v in persistence],
        "persistenceCorr": [float(v) for v in correlation(np.repeat(truth[:1], steps, 0), truth[1:])],
        "threshold": float(cfg.get("evaluation", {}).get("correlation_threshold", 0.8)),
        "runs": runs,
    }
    html = TEMPLATE.read_text().replace("/*__PAYLOAD__*/null", json.dumps(payload))
    out = args.out or out_dir / "figs" / f"visor_s{args.seed}_traj{args.trajectory}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"visor en {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
