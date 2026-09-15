"""Tests de la fase 2: crecimiento por manchas y equivalencia de los controles.

No tocan data/raw/: trabajan sobre grafos sinteticos pequenos. El unico test que
usa el subgrafo real se salta si todavia no se ha generado.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from connectome.controls import degrees, edge_keys, random_mask, rewired_mask
from connectome.subgraph import (
    Connectome,
    graph_stats,
    grow_blobs,
    induced_edges,
    largest_weak_component,
    stratified_seeds,
    undirected_csr,
)

ARTIFACTS = Path("artifacts/conectoma_4000")


def ring_graph(n: int, extra: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Anillo dirigido con atajos: conexo y con grados heterogeneos."""
    rng = np.random.default_rng(0)
    pre = np.arange(n, dtype=np.int32)
    post = ((pre + 1) % n).astype(np.int32)
    extra_pre = rng.integers(0, n, size=extra * n).astype(np.int32)
    extra_post = rng.integers(0, n, size=extra * n).astype(np.int32)
    keep = extra_pre != extra_post
    pre = np.concatenate([pre, extra_pre[keep]])
    post = np.concatenate([post, extra_post[keep]])
    keys = edge_keys(pre, post, n)
    _, unique = np.unique(keys, return_index=True)
    return pre[unique], post[unique]


def test_grow_blobs_reparte_el_presupuesto_sin_repetir() -> None:
    pre, post = ring_graph(200)
    graph = undirected_csr(pre, post, 200)
    rng = np.random.default_rng(1)
    seeds = stratified_seeds(np.diff(graph.indptr), 5, rng)
    nodes = grow_blobs(graph, seeds, 80, rng)

    assert nodes.size == 80
    assert np.unique(nodes).size == 80
    assert set(seeds.tolist()) <= set(nodes.tolist())


def test_grow_blobs_es_determinista_con_la_misma_semilla() -> None:
    pre, post = ring_graph(200)
    graph = undirected_csr(pre, post, 200)

    def run() -> np.ndarray:
        rng = np.random.default_rng(7)
        return grow_blobs(graph, stratified_seeds(np.diff(graph.indptr), 5, rng), 60, rng)

    assert np.array_equal(run(), run())


def test_componente_principal_poda_los_nodos_aislados() -> None:
    # Un triangulo (0-1-2), un cuadrado (3-4-5-6) y un nodo suelto (7).
    rows = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)
    cols = np.array([1, 2, 0, 4, 5, 6, 3], dtype=np.int32)
    keep, n_components = largest_weak_component(rows, cols, 8)

    assert n_components == 3
    assert np.array_equal(keep, np.array([3, 4, 5, 6], dtype=np.int32))
    assert 7 not in keep.tolist()


def test_induced_edges_reindexa_y_descarta_lo_de_fuera() -> None:
    conn = Connectome(
        pre=np.array([0, 1, 2, 3], dtype=np.int32),
        post=np.array([1, 2, 3, 0], dtype=np.int32),
        synapses=np.array([5, 6, 7, 8], dtype=np.int32),
        sign=np.array([1, -1, 1, -1], dtype=np.int8),
        n_neurons=4,
    )
    rows, cols, syn = induced_edges(conn, np.array([1, 2], dtype=np.int32))

    assert rows.tolist() == [0]      # 1 -> 2 reindexado a 0 -> 1
    assert cols.tolist() == [1]
    assert syn.tolist() == [6]


def test_recableado_conserva_los_grados_exactamente() -> None:
    pre, post = ring_graph(150)
    syn = np.arange(pre.size, dtype=np.int32)
    rng = np.random.default_rng(3)
    rew, info = rewired_mask(pre, post, syn.copy(), 150, 10, rng)

    out_ref, in_ref = degrees(pre, post, 150)
    out_got, in_got = degrees(rew["rows"], rew["cols"], 150)
    assert np.array_equal(out_ref, out_got)
    assert np.array_equal(in_ref, in_got)
    assert info["intercambios_aceptados"] > 0
    assert not np.array_equal(post, rew["cols"])  # algo tiene que haberse movido


def test_recableado_no_crea_autoconexiones_ni_duplicados() -> None:
    pre, post = ring_graph(150)
    rng = np.random.default_rng(4)
    rew, _ = rewired_mask(pre, post, np.ones(pre.size, dtype=np.int32), 150, 10, rng)

    keys = edge_keys(rew["rows"], rew["cols"], 150)
    assert np.unique(keys).size == keys.size
    assert not np.any(rew["rows"] == rew["cols"])


def test_aleatorio_iguala_aristas_y_fraccion_inhibidora() -> None:
    n_nodes, n_edges, target = 300, 6000, 0.43
    rng = np.random.default_rng(5)
    mask = random_mask(n_nodes, n_edges, target, np.ones(n_edges, dtype=np.int32), rng)

    keys = edge_keys(mask["rows"], mask["cols"], n_nodes)
    assert keys.size == n_edges
    assert np.unique(keys).size == n_edges
    assert not np.any(mask["rows"] == mask["cols"])

    stats = graph_stats(mask["rows"], mask["cols"], mask["sign"], n_nodes)
    assert abs(stats["fraccion_inhibidora_aristas"] - target) / target <= 0.01


def test_ley_de_dale_en_los_controles() -> None:
    """El signo es propiedad de la neurona, no de la arista: una sola entrada por neurona."""
    rng = np.random.default_rng(6)
    mask = random_mask(200, 3000, 0.4, np.ones(3000, dtype=np.int32), rng)
    assert mask["sign"].shape == (200,)
    assert set(np.unique(mask["sign"]).tolist()) <= {-1, 1}


@pytest.mark.skipif(not (ARTIFACTS / "mosca.npz").is_file(), reason="subgrafo no generado")
def test_subgrafo_real_cumple_el_criterio_de_aceptacion() -> None:
    stats = json.loads((ARTIFACTS / "stats.json").read_text())
    assert stats["fraccion_en_componente_principal"] >= stats["criterio_componente_principal"]
    assert stats["n_componentes_debiles"] == 1
    assert stats["neuronas_grado_interno_0"] == 0

    fly = np.load(ARTIFACTS / "mosca.npz")
    n_nodes = int(fly["sign"].size)
    assert fly["rows"].max() < n_nodes and fly["cols"].max() < n_nodes
    assert set(np.unique(fly["sign"]).tolist()) <= {-1, 0, 1}
    keys = edge_keys(fly["rows"], fly["cols"], n_nodes)
    assert np.unique(keys).size == keys.size

    # Signo 0 solo se admite en neuronas sin ninguna salida: su columna esta enmascarada.
    sin_signo = fly["sign"] == 0
    assert np.all(np.bincount(fly["rows"], minlength=n_nodes)[sin_signo] == 0)
    assert int(sin_signo.sum()) == stats["neuronas_sin_signo"]
