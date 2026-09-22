"""Tests del medidor de coste (benchmark.py).

No miden velocidad —eso depende de la maquina— sino el contrato: que los datos sinteticos
tengan la forma que pide la config cuando el split real no existe, y que la extrapolacion
a horas use los pasos del YAML y reparta entre las dos fases sin perder tiempo por el
camino. Si la extrapolacion miente, la decision de cuanto cabe en la GPU se toma mal.
"""

from __future__ import annotations

import torch

from benchmark import carga_datos, mide_modelo

CONFIG = {
    "data_dir": "no_existe_este_directorio",
    "output_dir": "no_se_escribe_en_el_test",
    "grad_clip": 1.0,
    "core": {"output_scale": 0.1},
    "fno": {"modes": 4, "width": 4, "n_layers": 1},
    "phase_a": {"steps": 1000, "batch_size": 2, "lr": 1.0e-3},
    "phase_b": {"curriculum": [3], "steps_per_stage": 100, "batch_size": 2, "lr": 3.0e-4},
    "augment": {"enabled": True, "forcing_wavenumber": 4},
    "models": ["persistencia", "fno"],
    "seeds": [0],
}


def test_sin_datos_reales_inventa_campos_con_la_forma_de_la_config() -> None:
    datos, reales = carga_datos(CONFIG, torch.device("cpu"), sintetico=False, malla=16)
    assert not reales
    assert datos.grid == 16
    # Tiene que caber una ventana del horizonte mas largo del curriculo, o la fase B no
    # se podria cronometrar.
    assert datos.omega.shape[1] >= max(CONFIG["phase_b"]["curriculum"]) + 1


def test_la_persistencia_no_se_cronometra_porque_no_se_entrena() -> None:
    datos, _ = carga_datos(CONFIG, torch.device("cpu"), sintetico=True, malla=16)
    fila = mide_modelo("persistencia", CONFIG, datos, torch.device("cpu"), pasos=1)
    assert fila["parametros"] == 0
    assert "horas_una_semilla" not in fila


def test_las_horas_salen_de_los_pasos_del_yaml_y_suman_las_dos_fases() -> None:
    datos, _ = carga_datos(CONFIG, torch.device("cpu"), sintetico=True, malla=16)
    fila = mide_modelo("fno", CONFIG, datos, torch.device("cpu"), pasos=2)
    assert fila["parametros"] > 0
    assert fila["segundos_paso_a"] > 0 and fila["segundos_paso_b"] > 0

    horas_a = CONFIG["phase_a"]["steps"] * fila["segundos_paso_a"] / 3600
    horas_b = (len(CONFIG["phase_b"]["curriculum"]) * CONFIG["phase_b"]["steps_per_stage"]
               * fila["segundos_paso_b"] / 3600)
    assert abs(fila["horas_fase_a"] - horas_a) < 0.01
    assert abs(fila["horas_fase_b"] - horas_b) < 0.01
    assert abs(fila["horas_una_semilla"] - (fila["horas_fase_a"] + fila["horas_fase_b"])) < 0.01
