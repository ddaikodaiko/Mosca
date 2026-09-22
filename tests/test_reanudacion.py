"""Test de la reanudacion de train.py.

Lo que hay que demostrar no es que "arranca otra vez", sino que reanudar da **exactamente
el mismo modelo** que no haber parado. Si al reanudar cambiara el orden de los lotes, el
entrenamiento dejaria de ser el que la semilla dice que es y la regla de semillas fijas y
registradas se quedaria en adorno. Por eso el estado del generador viaja en el fichero, y
por eso este test compara los pesos bit a bit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from train import (
    Augmenter,
    Reanudacion,
    Trajectories,
    build_model,
    train_phase_a,
    train_phase_b,
)


def configuracion(destino: Path, pasos: int, cada: int) -> dict[str, Any]:
    return {
        "output_dir": str(destino),
        "grad_clip": 1.0,
        "resume_every": cada,
        "core": {"output_scale": 0.1},
        "fno": {"modes": 4, "width": 6, "n_layers": 1},
        "phase_a": {"steps": pasos, "batch_size": 2, "lr": 1.0e-3, "eval_every": 2,
                    "patience": 99},
        "phase_b": {"curriculum": [], "steps_per_stage": 0, "batch_size": 2,
                    "lr": 3.0e-4, "patience": 99},
        "augment": {"enabled": True, "forcing_wavenumber": 4},
    }


def datos(malla: int = 16) -> Trajectories:
    torch.manual_seed(1234)
    return Trajectories(omega=torch.randn(4, 6, malla, malla), seeds=np.arange(4))


def entrena(cfg: dict[str, Any], nombre: str, semilla: int, split: Trajectories,
            modelo_previo: torch.nn.Module | None = None) -> torch.nn.Module:
    torch.manual_seed(semilla)
    modelo = build_model("fno", cfg, split.grid, semilla)
    generador = torch.Generator().manual_seed(semilla)
    reanuda = Reanudacion.para(cfg, nombre, semilla)
    if reanuda.estado is not None:
        modelo.load_state_dict(reanuda.estado["modelo"])
    del modelo_previo
    train_phase_a(modelo, split, split, cfg, generador, Augmenter.from_config(cfg), [], reanuda)
    return modelo


def test_reanudar_da_los_mismos_pesos_que_no_haber_parado(tmp_path: Path) -> None:
    split = datos()

    # Corrida de referencia, sin cortes. Guarda una instantanea en el paso 3 (5 % 3 != 0,
    # asi que el fichero se queda con el paso 3 y no con el final).
    cfg = configuracion(tmp_path / "referencia", pasos=5, cada=3)
    referencia = entrena(cfg, "fno", 0, split)
    instantanea = Path(cfg["output_dir"]) / "checkpoints" / "fno_s0.reanudar.pt"
    assert instantanea.is_file(), "no se guardo ninguna instantanea"
    estado = torch.load(instantanea, map_location="cpu", weights_only=False)
    assert estado["paso"] == 3 and estado["fase"] == "A"

    # Segunda corrida: modelo nuevo que arranca desde esa instantanea y termina los pasos
    # que faltaban. Es lo que pasaria tras un reinicio del PC.
    reanudado = entrena(cfg, "fno", 0, split)

    for (nombre, a), (_, b) in zip(referencia.state_dict().items(),
                                   reanudado.state_dict().items()):
        assert torch.equal(a, b), f"el parametro {nombre} no coincide tras reanudar"


def test_sin_resume_every_no_escribe_nada(tmp_path: Path) -> None:
    """La reanudacion se apaga con resume_every 0, y entonces no toca el disco."""
    cfg = configuracion(tmp_path / "apagada", pasos=3, cada=0)
    entrena(cfg, "fno", 0, datos())
    instantanea = Path(cfg["output_dir"]) / "checkpoints" / "fno_s0.reanudar.pt"
    assert not instantanea.exists()


def test_reanudar_en_la_fase_b_se_salta_las_etapas_ya_hechas(tmp_path: Path) -> None:
    """El curriculo tiene etapas: reanudar en la segunda no puede repetir la primera.

    La fase B se lleva el 80 % del coste del entrenamiento real, asi que es justo la que
    mas falta hace que se pueda reanudar.
    """
    split = datos()
    cfg = configuracion(tmp_path / "curriculo", pasos=0, cada=2)
    cfg["phase_b"] = {"curriculum": [2, 2], "steps_per_stage": 3, "batch_size": 2,
                      "lr": 3.0e-4, "eval_every": 3, "patience": 99}

    def corre() -> torch.nn.Module:
        torch.manual_seed(0)
        modelo = build_model("fno", cfg, split.grid, 0)
        generador = torch.Generator().manual_seed(0)
        reanuda = Reanudacion.para(cfg, "fno", 0)
        if reanuda.estado is not None:
            modelo.load_state_dict(reanuda.estado["modelo"])
        train_phase_b(modelo, split, split, cfg, generador, Augmenter.from_config(cfg),
                      [], reanuda)
        return modelo

    referencia = corre()
    estado = torch.load(Path(cfg["output_dir"]) / "checkpoints" / "fno_s0.reanudar.pt",
                        map_location="cpu", weights_only=False)
    assert estado["fase"] == "B" and estado["etapa"] == 1 and estado["paso"] == 2

    reanudado = corre()
    for (nombre, a), (_, b) in zip(referencia.state_dict().items(),
                                   reanudado.state_dict().items()):
        assert torch.equal(a, b), f"el parametro {nombre} no coincide tras reanudar"
