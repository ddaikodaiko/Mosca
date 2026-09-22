"""Coste medido de la fase 4 en la maquina donde se va a entrenar.

Mide, por modelo, lo que tarda un paso de Adam de la fase A y un paso de la fase B al
horizonte mas largo del curriculo, y cuanta memoria de GPU pide el pico. Con eso
extrapola el tiempo del entrenamiento completo que declara el YAML. Es para lanzarlo
ANTES de los 21 entrenamientos, no para adivinar despues por que no terminaban.

    python benchmark.py --config configs/entrenamiento_64.yaml
    python benchmark.py --config configs/entrenamiento_64.yaml --pasos 30 --sintetico

Si el split de entrenamiento todavia no esta generado, usa campos aleatorios del mismo
tamano y lo dice en el informe: el tiempo por paso depende de las formas, no del
contenido, pero la extrapolacion no vale como medida fisica de nada mas.

La extrapolacion es un techo optimista: no cuenta las evaluaciones periodicas ni la
carga de datos, y la parada temprana puede recortarla. Sale impresa como cota, no como
promesa.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from data.spectral import resolve_device
from train import Augmenter, Trajectories, build_model, relative_l2, rollout


def carga_datos(cfg: dict[str, Any], device: torch.device, sintetico: bool,
                malla: int | None = None) -> tuple[Trajectories, bool]:
    """El split de entrenamiento real, o campos aleatorios con la misma forma."""
    ruta = Path(cfg["data_dir"]) / "train.npz"
    if not sintetico and ruta.exists():
        limite = cfg.get("max_trajectories", {}).get("train")
        return Trajectories.load(ruta, device, limite), True
    malla = int(malla or cfg.get("grid", 64))
    horizonte = max(int(h) for h in cfg["phase_b"]["curriculum"]) if cfg["phase_b"]["curriculum"] else 1
    n_guardados = horizonte + 2
    omega = torch.randn(8, n_guardados, malla, malla, device=device)
    return Trajectories(omega=omega, seeds=np.arange(8)), False


def crono(funcion, repeticiones: int, device: torch.device) -> float:
    """Segundos por llamada, con calentamiento y sincronizacion de CUDA."""
    for _ in range(3):
        funcion()
    if device.type == "cuda":
        torch.cuda.synchronize()
    inicio = time.perf_counter()
    for _ in range(repeticiones):
        funcion()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - inicio) / repeticiones


def mide_modelo(
    nombre: str, cfg: dict[str, Any], datos: Trajectories, device: torch.device, pasos: int
) -> dict[str, Any]:
    torch.manual_seed(0)
    generador = torch.Generator(device=device)
    generador.manual_seed(0)
    modelo = build_model(nombre, cfg, datos.grid, seed=0).to(device)
    n_parametros = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    if n_parametros == 0:
        return {"modelo": nombre, "parametros": 0, "nota": "no se entrena"}
    if hasattr(modelo, "fit_feature_statistics"):
        muestra, _ = datos.pairs()
        modelo.fit_feature_statistics(muestra[: int(cfg.get("stats_sample", 2048))])

    aumentador = Augmenter.from_config(cfg)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    lote_a = int(cfg["phase_a"]["batch_size"])
    entradas, objetivos = datos.pairs()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=float(cfg["phase_a"]["lr"]))

    def paso_a() -> None:
        indice = torch.randint(0, entradas.shape[0], (lote_a,), generator=generador, device=device)
        x, y = aumentador(entradas[indice], objetivos[indice], generator=generador)
        optimizador.zero_grad()
        salida = modelo(x)
        perdida = relative_l2(salida[0] if isinstance(salida, tuple) else salida, y)
        perdida.backward()
        nn.utils.clip_grad_norm_(modelo.parameters(), float(cfg["grad_clip"]))
        optimizador.step()

    segundos_a = crono(paso_a, pasos, device)

    curriculo = [int(h) for h in cfg["phase_b"]["curriculum"]]
    segundos_b = None
    if curriculo:
        horizonte = max(curriculo)
        lote_b = int(cfg["phase_b"]["batch_size"])

        def paso_b() -> None:
            inicio, objetivo = datos.windows(horizonte, generador, lote_b)
            inicio, objetivo = aumentador(inicio, objetivo, generator=generador)
            optimizador.zero_grad()
            perdida = relative_l2(rollout(modelo, inicio, horizonte).flatten(0, 1),
                                  objetivo.flatten(0, 1))
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), float(cfg["grad_clip"]))
            optimizador.step()

        segundos_b = crono(paso_b, max(pasos // 4, 3), device)

    pico_mb = (torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None

    horas_a = float(cfg["phase_a"]["steps"]) * segundos_a / 3600
    horas_b = 0.0
    if segundos_b is not None:
        # El curriculo corre por etapas; todas cuestan como la mas larga o menos.
        horas_b = len(curriculo) * float(cfg["phase_b"]["steps_per_stage"]) * segundos_b / 3600
    return {
        "modelo": nombre,
        "parametros": int(n_parametros),
        "segundos_paso_a": round(segundos_a, 4),
        "segundos_paso_b": round(segundos_b, 4) if segundos_b is not None else None,
        "pico_memoria_mb": round(pico_mb, 1) if pico_mb is not None else None,
        "horas_fase_a": round(horas_a, 2),
        "horas_fase_b": round(horas_b, 2),
        "horas_una_semilla": round(horas_a + horas_b, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pasos", type=int, default=20, help="pasos cronometrados por fase")
    parser.add_argument("--device", default=None)
    parser.add_argument("--sintetico", action="store_true",
                        help="no cargar el split real aunque exista")
    parser.add_argument("--malla", type=int, default=None,
                        help="tamano de malla para los datos sinteticos (por defecto, 64)")
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = resolve_device(args.device or cfg.get("device", "auto"))
    datos, reales = carga_datos(cfg, device, args.sintetico, args.malla)
    nombre_gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "-"
    print(f"dispositivo: {device}  {nombre_gpu}")
    print(f"datos: {'reales' if reales else 'SINTETICOS'}  {tuple(datos.omega.shape)}\n")

    modelos = args.models or [m for m in cfg["models"] if m != "persistencia"]
    semillas = len(cfg["seeds"])
    filas = [mide_modelo(nombre, cfg, datos, device, args.pasos) for nombre in modelos]

    ancho = max(len(f["modelo"]) for f in filas)
    print(f"{'modelo':<{ancho}}  {'parametros':>11}  {'s/paso A':>9}  {'s/paso B':>9}"
          f"  {'pico MB':>8}  {'h fase A':>8}  {'h fase B':>8}  {'h/semilla':>9}")
    for fila in filas:
        if fila.get("parametros", 0) == 0:
            continue
        b = fila["segundos_paso_b"]
        pico = fila["pico_memoria_mb"]
        print(f"{fila['modelo']:<{ancho}}  {fila['parametros']:>11,}"
              f"  {fila['segundos_paso_a']:>9.3f}  {(f'{b:.3f}' if b else '-'):>9}"
              f"  {(f'{pico:.0f}' if pico else '-'):>8}  {fila['horas_fase_a']:>8.2f}"
              f"  {fila['horas_fase_b']:>8.2f}  {fila['horas_una_semilla']:>9.2f}")

    total = sum(f.get("horas_una_semilla", 0.0) for f in filas) * semillas
    total_b = sum(f.get("horas_fase_b", 0.0) for f in filas) * semillas
    print(f"\ncota para {len(modelos)} modelos x {semillas} semillas: {total:.1f} h"
          "  (sin evaluaciones periodicas; la parada temprana puede recortarlo)")
    if total > 0:
        # Donde esta el tiempo decide donde se recorta. El despliegue de la fase B es
        # secuencial, asi que es el que peor acelera en GPU y el primero que hay que mirar.
        print(f"de las cuales la fase B (despliegue) se lleva {total_b:.1f} h, "
              f"el {100 * total_b / total:.0f} %")

    salida = Path(cfg["output_dir"])
    salida.mkdir(parents=True, exist_ok=True)
    informe = {
        "config": str(args.config),
        "dispositivo": str(device),
        "gpu": nombre_gpu,
        "torch": torch.__version__,
        "datos_reales": reales,
        "pasos_cronometrados": args.pasos,
        "semillas": semillas,
        "horas_totales_cota": round(total, 2),
        "modelos": filas,
    }
    ruta = salida / "benchmark.json"
    ruta.write_text(json.dumps(informe, indent=2, ensure_ascii=False))
    print(f"informe en {ruta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
