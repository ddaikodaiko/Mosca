"""Diagnosticos del cableado sin entrenar nada (21-10-2026). EXPLORATORIO.

No forma parte de ningun plan registrado: sirve para ordenar las hipotesis de por que
la mosca queda por debajo del aleatorio en la fase 6d (RESULTADOS.md, seccion del
21-10-2026, y el informe en reports/). Mide sobre cada mascara, con la inicializacion
exacta de `MaskedRecurrentCore` y los puertos exactos de `select_ports`:

  H1  cola de grados y reescalado espectral: radio espectral bruto, factor de escala y
      grueso del espectro (mediana de |autovalor|) tras llevar el maximo a 0,95.
  H2  reciprocidad y Dale: fraccion de aristas con reciproca y signos de esos pares.
  H3  puertos: aristas directas sensorial -> motora, motoras a un salto, distancia media.
  H4  espectro de Dale: entropia normalizada de los valores singulares de W inicial.
  H5  colocacion de los conteos: conteo medio en aristas reciprocas frente al resto, y
      cuota de |W| que sale de los centros (5 % de mas grado de salida).

Y sobre los checkpoints entrenados, si se dan: radio espectral de W y cuanto se ha
movido respecto a la inicializacion.

    python diagnostico_cableado.py --config configs/mario_proto_6c_soltar.yaml \\
        --checkpoints artifacts/mario_proto_6c_soltar_gcp/desde_clonado \\
        --salida artifacts/mario_proto_6c_soltar_gcp/diagnostico_cableado.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.sparse.csgraph import shortest_path
import scipy.sparse as sp

from models.masked_rnn import MaskedRecurrentCore, Wiring, select_ports

CABLEADOS = ("mosca", "aleatorio", "recableado")
SEMILLAS = (0, 1, 2)
CUOTA_CENTROS = 0.05


def matriz_inicial(wiring: Wiring, core: dict[str, Any]) -> np.ndarray:
    """W[post, pre] densa con la inicializacion exacta del nucleo."""
    theta = MaskedRecurrentCore._initial_theta(
        wiring, float(core.get("spectral_radius", 0.95)), str(core.get("normalisation", "max_eig"))
    )
    valores = wiring.sign[wiring.rows].astype(np.float64) * np.log1p(np.exp(theta.astype(np.float64)))
    w = np.zeros((wiring.n_neurons, wiring.n_neurons))
    w[wiring.cols, wiring.rows] = valores
    return w


def matriz_bruta(wiring: Wiring) -> np.ndarray:
    """La misma matriz antes de reescalar: conteos con signo, normalizados a media 1."""
    magnitud = wiring.synapses.astype(np.float64)
    magnitud = magnitud / magnitud.mean()
    w = np.zeros((wiring.n_neurons, wiring.n_neurons))
    w[wiring.cols, wiring.rows] = wiring.sign[wiring.rows] * magnitud
    return w


def entropia_svd(w: np.ndarray) -> float:
    s = np.linalg.svd(w, compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(w.shape[0]))


def reciprocas(wiring: Wiring) -> np.ndarray:
    """Mascara booleana por arista: True si existe la arista en sentido contrario."""
    n = wiring.n_neurons
    claves = wiring.rows.astype(np.int64) * n + wiring.cols
    inversas = wiring.cols.astype(np.int64) * n + wiring.rows
    return np.isin(inversas, claves)


def puertos(wiring: Wiring, encoder: dict[str, Any]) -> dict[str, Any]:
    sensory, motor = select_ports(
        wiring, int(encoder["n_sensory"]), int(encoder["n_motor"]), int(encoder["n_shared"])
    )
    es_s = np.zeros(wiring.n_neurons, dtype=bool)
    es_s[sensory] = True
    es_m = np.zeros(wiring.n_neurons, dtype=bool)
    es_m[motor] = True
    directas = es_s[wiring.rows] & es_m[wiring.cols]
    adyacencia = sp.csr_matrix(
        (np.ones(wiring.n_edges), (wiring.rows, wiring.cols)),
        shape=(wiring.n_neurons, wiring.n_neurons),
    )
    distancias = shortest_path(adyacencia, directed=True, unweighted=True, indices=sensory)
    hasta_motora = distancias.min(axis=0)[motor]
    # Las compartidas estan a distancia 0 de si mismas; se miden aparte.
    exclusivas = hasta_motora[~es_s[motor]]
    alcanzables = exclusivas[np.isfinite(exclusivas)]
    return {
        "aristas_sensorial_a_motora": int(directas.sum()),
        "fraccion_de_aristas_sensorial_a_motora": round(float(directas.mean()), 4),
        "aristas_entre_sensoriales": int((es_s[wiring.rows] & es_s[wiring.cols]).sum()),
        "motoras_exclusivas_a_un_salto": round(float((exclusivas == 1).mean()), 4),
        "motoras_exclusivas_inalcanzables": int((~np.isfinite(exclusivas)).sum()),
        "distancia_media_a_motora_exclusiva": round(float(alcanzables.mean()), 3),
    }


def diagnostico_mascara(wiring: Wiring, cfg: dict[str, Any]) -> dict[str, Any]:
    radio = float(cfg["core"].get("spectral_radius", 0.95))
    bruta = matriz_bruta(wiring)
    inicial = matriz_inicial(wiring, cfg["core"])
    autovalores_brutos = np.abs(np.linalg.eigvals(bruta))
    autovalores = np.abs(np.linalg.eigvals(inicial))

    rec = reciprocas(wiring)
    signo_pre = wiring.sign[wiring.rows]
    signo_post = wiring.sign[wiring.cols]
    # Cada par reciproco aparece dos veces (una por arista); los cocientes no cambian.
    ee = float(((signo_pre > 0) & (signo_post > 0))[rec].mean())
    ii = float(((signo_pre < 0) & (signo_post < 0))[rec].mean())

    salida = np.bincount(wiring.rows, minlength=wiring.n_neurons)
    entrada = np.bincount(wiring.cols, minlength=wiring.n_neurons)
    n_centros = max(1, int(round(CUOTA_CENTROS * wiring.n_neurons)))
    centros = np.argsort(-salida, kind="stable")[:n_centros]
    es_centro = np.zeros(wiring.n_neurons, dtype=bool)
    es_centro[centros] = True
    modulo = np.abs(inicial[wiring.cols, wiring.rows])

    return {
        "n_aristas": wiring.n_edges,
        "grado_max_salida": int(salida.max()),
        "grado_max_entrada": int(entrada.max()),
        "fraccion_inhibidora_aristas": round(float((signo_pre < 0).mean()), 4),
        "H1_radio_bruto": round(float(autovalores_brutos.max()), 3),
        "H1_factor_de_escala": round(radio / float(autovalores_brutos.max()), 4),
        "H1_grueso_del_espectro": round(float(np.median(autovalores)), 4),
        "H1_peso_medio_por_arista": round(float(modulo.mean()), 4),
        "H2_reciprocidad": round(float(rec.mean()), 4),
        "H2_reciprocas_EE": round(ee, 3),
        "H2_reciprocas_II": round(ii, 3),
        "H2_reciprocas_EI": round(1 - ee - ii, 3),
        "H3": puertos(wiring, cfg["encoder"]),
        "H4_entropia_svd": round(entropia_svd(inicial), 4),
        "H5_conteo_medio_reciprocas": round(float(wiring.synapses[rec].mean()), 3),
        "H5_conteo_medio_resto": round(float(wiring.synapses[~rec].mean()), 3),
        "H5_cuota_de_W_que_sale_de_centros": round(
            float(modulo[es_centro[wiring.rows]].sum() / modulo.sum()), 4
        ),
    }


def diagnostico_entrenado(ruta: Path, inicial: np.ndarray) -> dict[str, Any]:
    estado = torch.load(ruta, map_location="cpu", weights_only=False)["state_dict"]
    theta = estado["core.theta"].double()
    indices = estado["core.edge_index"].numpy()
    valores = (estado["core.edge_sign"].double() * torch.nn.functional.softplus(theta)).numpy()
    w = np.zeros_like(inicial)
    w[indices[0], indices[1]] = valores
    previos = inicial[indices[0], indices[1]]
    return {
        "radio_espectral": round(float(np.abs(np.linalg.eigvals(w)).max()), 3),
        "grueso_del_espectro": round(float(np.median(np.abs(np.linalg.eigvals(w)))), 4),
        "cambio_relativo_de_W": round(
            float(np.linalg.norm(valores - previos) / np.linalg.norm(previos)), 3
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--wiring-dir", type=Path, help="por defecto, el `wiring_dir` del YAML")
    parser.add_argument("--checkpoints", type=Path, help="carpeta con <cableado>_s<semilla>.pt")
    parser.add_argument("--salida", type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    wiring_dir = args.wiring_dir or Path(cfg["wiring_dir"])
    informe: dict[str, Any] = {"wiring_dir": str(wiring_dir), "mascaras": {}, "entrenados": {}}
    for nombre in CABLEADOS:
        wiring = Wiring.load(wiring_dir / f"{nombre}.npz")
        informe["mascaras"][nombre] = diagnostico_mascara(wiring, cfg)
        if args.checkpoints is None:
            continue
        inicial = matriz_inicial(wiring, cfg["core"])
        for semilla in SEMILLAS:
            ruta = args.checkpoints / f"{nombre}_s{semilla}.pt"
            if ruta.is_file():
                informe["entrenados"][f"{nombre}_s{semilla}"] = diagnostico_entrenado(ruta, inicial)

    filas = [k for k in informe["mascaras"]["mosca"] if k != "H3"]
    print(f"# Diagnostico del cableado ({wiring_dir})\n")
    print("| medida | " + " | ".join(CABLEADOS) + " |")
    print("|---|" + "---|" * len(CABLEADOS))
    for fila in filas:
        print(f"| {fila} | " + " | ".join(str(informe["mascaras"][c][fila]) for c in CABLEADOS) + " |")
    for fila in informe["mascaras"]["mosca"]["H3"]:
        print(f"| H3_{fila} | " + " | ".join(str(informe["mascaras"][c]["H3"][fila]) for c in CABLEADOS) + " |")
    if informe["entrenados"]:
        print("\n| entrenado | radio espectral | grueso del espectro | cambio relativo de W |")
        print("|---|---|---|---|")
        for nombre, d in informe["entrenados"].items():
            print(f"| {nombre} | {d['radio_espectral']} | {d['grueso_del_espectro']} | {d['cambio_relativo_de_W']} |")
    if args.salida:
        args.salida.write_text(json.dumps(informe, indent=2, ensure_ascii=False))
        print(f"\n-> {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
