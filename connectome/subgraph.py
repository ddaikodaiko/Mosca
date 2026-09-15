"""Subgrafo de N neuronas del conectoma de FlyWire v783 (fase 2).

Uso:
    python -m connectome.subgraph --config configs/conectoma_4000.yaml

Procedimiento de la seccion 4.1 del diseno: S semillas estratificadas por grado,
crecimiento BFS por manchas hasta N neuronas, componente debilmente conexa mayor,
poda de las neuronas con grado interno 0.

Escribe en el directorio de salida del YAML:
    mosca.npz    nodos, aristas (row/col), sinapsis, signo por neurona
    stats.json   N, aristas, densidad, fraccion inhibidora, grados, componentes
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
import yaml

CONNECTIVITY_FILE = "Connectivity_783.parquet"
COMPLETENESS_FILE = "Completeness_783.csv"
BATCH_ROWS = 2_000_000


@dataclass(frozen=True)
class Connectome:
    """Grafo dirigido completo con dtypes reducidos (el parquet son 101 MB comprimidos y se infla mucho en pandas)."""

    pre: np.ndarray       # int32 (n_edges,)
    post: np.ndarray      # int32 (n_edges,)
    synapses: np.ndarray  # int32 (n_edges,)
    sign: np.ndarray      # int8  (n_neurons,) +1 excitadora, -1 inhibidora, 0 sin salidas
    n_neurons: int

    @property
    def n_edges(self) -> int:
        return int(self.pre.size)


def load_connectome(raw_dir: Path, min_synapses: int = 1) -> Connectome:
    """Lee el parquet por lotes, solo las cuatro columnas necesarias."""
    path = raw_dir / CONNECTIVITY_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"falta {path}. Ejecuta primero: python -m data.download_flywire"
        )
    n_neurons = sum(1 for _ in (raw_dir / COMPLETENESS_FILE).open()) - 1

    pre_parts: list[np.ndarray] = []
    post_parts: list[np.ndarray] = []
    syn_parts: list[np.ndarray] = []
    sign = np.zeros(n_neurons, dtype=np.int8)

    reader = pq.ParquetFile(path)
    columns = ["Presynaptic_Index", "Postsynaptic_Index", "Connectivity", "Excitatory"]
    for batch in reader.iter_batches(batch_size=BATCH_ROWS, columns=columns):
        pre = batch.column(0).to_numpy().astype(np.int32, copy=False)
        post = batch.column(1).to_numpy().astype(np.int32, copy=False)
        syn = batch.column(2).to_numpy().astype(np.int32, copy=False)
        exc = batch.column(3).to_numpy().astype(np.int8, copy=False)
        sign[pre] = exc  # el signo es propiedad de la neurona presinaptica (Dale)
        if min_synapses > 1:
            keep = syn >= min_synapses
            pre, post, syn = pre[keep], post[keep], syn[keep]
        pre_parts.append(pre)
        post_parts.append(post)
        syn_parts.append(syn)

    return Connectome(
        pre=np.concatenate(pre_parts),
        post=np.concatenate(post_parts),
        synapses=np.concatenate(syn_parts),
        sign=sign,
        n_neurons=n_neurons,
    )


def undirected_csr(pre: np.ndarray, post: np.ndarray, n_neurons: int) -> sp.csr_matrix:
    """Adyacencia no dirigida y sin pesos, para el crecimiento BFS."""
    data = np.ones(pre.size * 2, dtype=np.int8)
    rows = np.concatenate([pre, post])
    cols = np.concatenate([post, pre])
    graph = sp.coo_matrix((data, (rows, cols)), shape=(n_neurons, n_neurons)).tocsr()
    graph.data[:] = 1
    return graph


def stratified_seeds(degree: np.ndarray, n_seeds: int, rng: np.random.Generator) -> np.ndarray:
    """Una semilla por estrato de grado, para no sesgar hacia el nucleo denso."""
    candidates = np.flatnonzero(degree > 0)
    order = candidates[np.argsort(degree[candidates], kind="stable")]
    strata = np.array_split(order, n_seeds)
    return np.array(
        [rng.choice(stratum) for stratum in strata if stratum.size > 0], dtype=np.int64
    )


def grow_blobs(
    graph: sp.csr_matrix,
    seeds: np.ndarray,
    n_target: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """BFS simultaneo desde cada semilla, por turnos, hasta repartir n_target nodos.

    Las manchas crecen a la vez y se reparten la cuota: asi ninguna se come el
    presupuesto entera. Si una se queda sin frontera, su cuota pasa a las demas.
    """
    indptr, indices = graph.indptr, graph.indices
    n_blobs = int(seeds.size)
    claimed = np.zeros(graph.shape[0], dtype=bool)
    frontiers: list[list[int]] = [[] for _ in range(n_blobs)]
    selected: list[int] = []

    for blob, seed in enumerate(seeds):
        if not claimed[seed]:
            claimed[seed] = True
            selected.append(int(seed))
            frontiers[blob].append(int(seed))

    heads = [0] * n_blobs
    while len(selected) < n_target:
        progressed = False
        for blob in range(n_blobs):
            if len(selected) >= n_target:
                break
            frontier = frontiers[blob]
            while heads[blob] < len(frontier):
                node = frontier[heads[blob]]
                heads[blob] += 1
                neighbours = indices[indptr[node] : indptr[node + 1]]
                fresh = neighbours[~claimed[neighbours]]
                if fresh.size == 0:
                    continue
                rng.shuffle(fresh)
                room = n_target - len(selected)
                take = fresh[:room]
                claimed[take] = True
                selected.extend(int(x) for x in take)
                frontier.extend(int(x) for x in take)
                progressed = True
                break
        if not progressed:
            break  # todas las manchas agotadas: el grafo no da para mas
    return np.array(sorted(selected), dtype=np.int32)


def induced_edges(
    conn: Connectome, nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aristas del subgrafo inducido, reindexadas a 0..len(nodes)-1."""
    lookup = np.full(conn.n_neurons, -1, dtype=np.int32)
    lookup[nodes] = np.arange(nodes.size, dtype=np.int32)
    pre_local = lookup[conn.pre]
    post_local = lookup[conn.post]
    keep = (pre_local >= 0) & (post_local >= 0)
    return pre_local[keep], post_local[keep], conn.synapses[keep]


def largest_weak_component(
    rows: np.ndarray, cols: np.ndarray, n_nodes: int
) -> tuple[np.ndarray, int]:
    """Devuelve los nodos de la componente debilmente conexa mayor y cuantas hay."""
    graph = sp.coo_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n_nodes, n_nodes)
    ).tocsr()
    n_components, labels = sp.csgraph.connected_components(graph, directed=True, connection="weak")
    sizes = np.bincount(labels, minlength=n_components)
    return np.flatnonzero(labels == int(sizes.argmax())).astype(np.int32), int(n_components)


def graph_stats(
    rows: np.ndarray, cols: np.ndarray, sign: np.ndarray, n_nodes: int
) -> dict[str, Any]:
    out_degree = np.bincount(rows, minlength=n_nodes)
    in_degree = np.bincount(cols, minlength=n_nodes)
    total = out_degree + in_degree
    _, n_components = largest_weak_component(rows, cols, n_nodes)
    inhibitory_edges = int((sign[rows] < 0).sum())
    return {
        "n_neuronas": int(n_nodes),
        "n_aristas": int(rows.size),
        "densidad": float(rows.size / (n_nodes * (n_nodes - 1))),
        "fraccion_inhibidora_aristas": float(inhibitory_edges / rows.size),
        "fraccion_inhibidora_neuronas": float((sign < 0).mean()),
        "grado_medio": float(total.mean()),
        "grado_mediano": float(np.median(total)),
        "grado_entrada_medio": float(in_degree.mean()),
        "grado_salida_medio": float(out_degree.mean()),
        "grado_max": int(total.max()),
        "neuronas_grado_interno_0": int((total == 0).sum()),
        # El signo viene de la neurona presinaptica: una neurona sin ninguna salida en
        # el conectoma completo no tiene signo definido. Su columna de W esta enmascarada
        # a cero, asi que el 0 es inocuo, pero se cuenta para no esconderlo.
        "neuronas_sin_signo": int((sign == 0).sum()),
        "n_componentes_debiles": n_components,
        "autoconexiones": int((rows == cols).sum()),
    }


def build_subgraph(conn: Connectome, cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    sub_cfg = cfg["subgraph"]
    graph = undirected_csr(conn.pre, conn.post, conn.n_neurons)
    degree = np.diff(graph.indptr)

    seeds = stratified_seeds(degree, int(sub_cfg["n_seeds"]), rng)
    nodes = grow_blobs(graph, seeds, int(sub_cfg["n_neurons"]), rng)
    n_grown = int(nodes.size)

    rows, cols, syn = induced_edges(conn, nodes)
    keep_local, n_components = largest_weak_component(rows, cols, n_grown)

    # La poda de grado interno 0 va incluida: un nodo aislado es su propia componente.
    remap = np.full(n_grown, -1, dtype=np.int32)
    remap[keep_local] = np.arange(keep_local.size, dtype=np.int32)
    inside = (remap[rows] >= 0) & (remap[cols] >= 0)
    rows, cols, syn = remap[rows[inside]], remap[cols[inside]], syn[inside]
    nodes = nodes[keep_local]
    sign = conn.sign[nodes]

    stats = graph_stats(rows, cols, sign, int(nodes.size))
    stats["n_semillas"] = int(seeds.size)
    stats["n_crecidas"] = n_grown
    stats["fraccion_en_componente_principal"] = float(nodes.size / n_grown)
    return {"nodes": nodes, "rows": rows, "cols": cols, "synapses": syn, "sign": sign, "stats": stats}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg["seed"]))

    t0 = time.perf_counter()
    conn = load_connectome(Path(cfg["raw_dir"]), int(cfg["graph"]["min_synapses"]))
    t_load = time.perf_counter() - t0
    print(f"conectoma completo: {conn.n_neurons} neuronas, {conn.n_edges} aristas ({t_load:.1f} s)")

    t0 = time.perf_counter()
    result = build_subgraph(conn, cfg, rng)
    t_build = time.perf_counter() - t0

    stats = result["stats"]
    minimo = float(cfg["subgraph"]["min_main_component"])
    stats["criterio_componente_principal"] = minimo
    stats["criterio_cumplido"] = bool(stats["fraccion_en_componente_principal"] >= minimo)
    stats["semilla"] = int(cfg["seed"])
    stats["min_synapses"] = int(cfg["graph"]["min_synapses"])
    stats["segundos_carga"] = round(t_load, 2)
    stats["segundos_construccion"] = round(t_build, 2)
    stats["config"] = cfg

    np.savez_compressed(
        out_dir / "mosca.npz",
        nodes=result["nodes"],
        rows=result["rows"],
        cols=result["cols"],
        synapses=result["synapses"],
        sign=result["sign"],
    )
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    for key, value in stats.items():
        if key != "config":
            print(f"  {key}: {value}")
    if not stats["criterio_cumplido"]:
        print("CRITERIO NO CUMPLIDO: menos del "
              f"{minimo:.0%} de las neuronas en la componente principal")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
