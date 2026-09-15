"""Controles del subgrafo: cableado aleatorio y recableado (fase 2).

Uso:
    python -m connectome.controls --config configs/conectoma_4000.yaml

Dos mascaras que igualan a la mosca en todo lo que no es "el cableado real":

- aleatorio:  Erdos-Renyi con el mismo N, el mismo numero de aristas y la misma
              fraccion de aristas inhibidoras. No conserva la distribucion de grados.
- recableado: intercambio de aristas (u->v, x->y) -> (u->y, x->v) sobre el grafo de
              la mosca. Conserva el grado de entrada y de salida de CADA neurona, y
              por tanto tambien la fraccion inhibidora exacta (el signo es de la
              neurona presinaptica y su grado de salida no cambia).

Escribe aleatorio.npz, recableado.npz y controls_stats.json junto a mosca.npz.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from connectome.subgraph import graph_stats


def edge_keys(rows: np.ndarray, cols: np.ndarray, n_nodes: int) -> np.ndarray:
    return rows.astype(np.int64) * n_nodes + cols.astype(np.int64)


def random_mask(
    n_nodes: int,
    n_edges: int,
    target_inhibitory: float,
    synapses: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Erdos-Renyi dirigido sin autoconexiones ni aristas repetidas."""
    n_slots = n_nodes * (n_nodes - 1)
    if n_edges > n_slots:
        raise ValueError("no caben tantas aristas en un grafo simple de ese tamano")

    # Muestreo sobre el indice lineal de la matriz sin diagonal: evita rechazar por
    # duplicados y es exacto en el numero de aristas.
    chosen = rng.choice(n_slots, size=n_edges, replace=False)
    rows = (chosen // (n_nodes - 1)).astype(np.int32)
    offsets = (chosen % (n_nodes - 1)).astype(np.int32)
    cols = offsets + (offsets >= rows)  # salta la diagonal

    # Signo por neurona (ley de Dale): se marcan inhibidoras en orden aleatorio hasta
    # alcanzar la fraccion objetivo de ARISTAS inhibidoras.
    out_degree = np.bincount(rows, minlength=n_nodes)
    order = rng.permutation(n_nodes)
    cumulative = np.cumsum(out_degree[order])
    target_edges = target_inhibitory * n_edges
    n_inhibitory = int(np.argmin(np.abs(cumulative - target_edges))) + 1
    sign = np.ones(n_nodes, dtype=np.int8)
    sign[order[:n_inhibitory]] = -1

    return {
        "rows": rows,
        "cols": cols,
        "synapses": rng.permutation(synapses),
        "sign": sign,
    }


def rewired_mask(
    rows: np.ndarray,
    cols: np.ndarray,
    synapses: np.ndarray,
    n_nodes: int,
    swaps_per_edge: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Intercambio doble de aristas conservando grado de entrada y de salida."""
    rows = rows.copy()
    cols = cols.copy()
    n_edges = int(rows.size)
    present = set(edge_keys(rows, cols, n_nodes).tolist())

    attempts = swaps_per_edge * n_edges
    accepted = 0
    picks = rng.integers(0, n_edges, size=(attempts, 2))
    for i, j in picks:
        if i == j:
            continue
        u, v = int(rows[i]), int(cols[i])
        x, y = int(rows[j]), int(cols[j])
        if u == y or x == v:
            continue  # crearia una autoconexion
        key_uy = u * n_nodes + y
        key_xv = x * n_nodes + v
        if key_uy in present or key_xv in present:
            continue  # ya existe: el grafo seguiria siendo simple pero perderiamos aristas
        present.discard(u * n_nodes + v)
        present.discard(x * n_nodes + y)
        present.add(key_uy)
        present.add(key_xv)
        cols[i], cols[j] = y, v
        synapses[i], synapses[j] = synapses[j], synapses[i]
        accepted += 1

    info = {
        "intentos": int(attempts),
        "intercambios_aceptados": accepted,
        "tasa_aceptacion": round(accepted / attempts, 4) if attempts else 0.0,
    }
    return {"rows": rows, "cols": cols, "synapses": synapses}, info


def degrees(rows: np.ndarray, cols: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    return np.bincount(rows, minlength=n_nodes), np.bincount(cols, minlength=n_nodes)


def compare(reference: dict[str, Any], control: dict[str, Any], tolerance: float) -> dict[str, Any]:
    """Desviacion relativa del control frente a la mosca en densidad y signo."""
    out: dict[str, Any] = {}
    for key in ("densidad", "fraccion_inhibidora_aristas"):
        ref, got = reference[key], control[key]
        out[f"desviacion_{key}"] = float(abs(got - ref) / ref)
    out["dentro_de_tolerancia"] = bool(
        max(out[f"desviacion_{k}"] for k in ("densidad", "fraccion_inhibidora_aristas"))
        <= tolerance
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = Path(cfg["output_dir"])
    fly_path = out_dir / "mosca.npz"
    if not fly_path.is_file():
        raise FileNotFoundError(f"falta {fly_path}. Ejecuta primero connectome.subgraph")

    rng = np.random.default_rng(int(cfg["seed"]) + 1)
    tolerance = float(cfg["controls"]["tolerance"])
    fly = dict(np.load(fly_path))
    n_nodes = int(fly["sign"].size)
    fly_stats = graph_stats(fly["rows"], fly["cols"], fly["sign"], n_nodes)

    report: dict[str, Any] = {"mosca": fly_stats, "semilla": int(cfg["seed"]) + 1}

    t0 = time.perf_counter()
    rand = random_mask(
        n_nodes,
        fly_stats["n_aristas"],
        fly_stats["fraccion_inhibidora_aristas"],
        fly["synapses"],
        rng,
    )
    rand_stats = graph_stats(rand["rows"], rand["cols"], rand["sign"], n_nodes)
    rand_stats.update(compare(fly_stats, rand_stats, tolerance))
    rand_stats["segundos"] = round(time.perf_counter() - t0, 2)
    np.savez_compressed(out_dir / "aleatorio.npz", nodes=fly["nodes"], **rand)
    report["aleatorio"] = rand_stats

    t0 = time.perf_counter()
    rew, swap_info = rewired_mask(
        fly["rows"],
        fly["cols"],
        fly["synapses"].copy(),
        n_nodes,
        int(cfg["controls"]["rewire_swaps_per_edge"]),
        rng,
    )
    rew["sign"] = fly["sign"]  # mismas neuronas, mismos signos
    rew_stats = graph_stats(rew["rows"], rew["cols"], rew["sign"], n_nodes)
    rew_stats.update(compare(fly_stats, rew_stats, tolerance))
    rew_stats.update(swap_info)

    # Verificacion dura: los grados tienen que conservarse EXACTAMENTE.
    out_fly, in_fly = degrees(fly["rows"], fly["cols"], n_nodes)
    out_rew, in_rew = degrees(rew["rows"], rew["cols"], n_nodes)
    rew_stats["grado_salida_conservado"] = bool(np.array_equal(out_fly, out_rew))
    rew_stats["grado_entrada_conservado"] = bool(np.array_equal(in_fly, in_rew))
    changed = int((fly["cols"] != rew["cols"]).sum())
    rew_stats["fraccion_aristas_movidas"] = round(changed / fly_stats["n_aristas"], 4)
    rew_stats["segundos"] = round(time.perf_counter() - t0, 2)
    np.savez_compressed(out_dir / "recableado.npz", nodes=fly["nodes"], **rew)
    report["recableado"] = rew_stats

    (out_dir / "controls_stats.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    ok = True
    for name in ("aleatorio", "recableado"):
        print(f"\n[{name}]")
        for key, value in report[name].items():
            print(f"  {key}: {value}")
        ok &= bool(report[name]["dentro_de_tolerancia"])
    ok &= bool(report["recableado"]["grado_salida_conservado"])
    ok &= bool(report["recableado"]["grado_entrada_conservado"])
    if not ok:
        print("\nCRITERIO NO CUMPLIDO: revisar desviaciones o conservacion de grados")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
