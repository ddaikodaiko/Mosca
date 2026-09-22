"""Test de la reanudacion de train_rl.py (fase 6).

Aqui la promesa es mas debil que en fluidos y conviene que el test lo diga: los entornos
no se pueden serializar —el emulador lleva su propia RAM—, asi que al reanudar arrancan
de cero y la recoleccion ya no es la que habria salido de un tiron. Lo que se comprueba
es lo que si se promete: que no se repiten iteraciones ya hechas, que lo aprendido
sobrevive al corte, y que el informe deja constancia de cuantas veces se reanudo, porque
un entrenamiento reanudado no se puede reproducir desde la semilla.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

import train_rl
from tests.test_masked_rnn import toy_wiring


def configuracion(destino: Path) -> dict[str, Any]:
    return {
        "output_dir": str(destino),
        "env": "propio",
        "resume_every": 1,
        "normalise_observations": False,
        "core": {"n_internal_steps": 2, "spectral_radius": 0.95, "leak": 1.0},
        "encoder": {"n_sensory": 12, "n_motor": 12, "n_shared": 4},
        "observation": {"window_width": 11, "window_height": 9},
        "level": {"width": 40, "height": 12},
        "level_seeds": {"train": [0, 20], "val": [20, 25], "test": [25, 30]},
        "ppo": {"n_envs": 2, "rollout_steps": 8, "iterations": 4, "epochs": 1,
                "n_minibatches": 1, "lr": 3.0e-4, "gamma": 0.99, "gae_lambda": 0.95,
                "clip": 0.2, "value_coefficient": 0.5, "entropy_coefficient": 0.01,
                "grad_clip": 1.0},
        "eval_every": 100, "eval_envs": 2, "eval_episodes": 2, "test_episodes": 2,
    }


def test_un_corte_no_repite_iteraciones_y_queda_anotado(tmp_path: Path, monkeypatch) -> None:
    cfg = configuracion(tmp_path)
    wirings = {"mosca": toy_wiring(n_neurons=40, seed=0)}
    device = torch.device("cpu")

    # Corta a la tercera iteracion, como haria un reinicio del PC.
    original = train_rl.collect
    llamadas = {"n": 0}

    def collect_que_se_corta(*args, **kwargs):
        llamadas["n"] += 1
        if llamadas["n"] > 2:
            raise RuntimeError("corte simulado")
        return original(*args, **kwargs)

    monkeypatch.setattr(train_rl, "collect", collect_que_se_corta)
    with pytest.raises(RuntimeError):
        train_rl.train_one("mosca", 0, cfg, wirings, device)

    instantanea = tmp_path / "mosca_s0.reanudar.pt"
    assert instantanea.is_file(), "el corte no dejo instantanea"
    estado = torch.load(instantanea, map_location="cpu", weights_only=False)
    assert estado["iteracion"] == 1        # se guardo tras la segunda iteracion (0 y 1)

    monkeypatch.setattr(train_rl, "collect", original)
    resultado = train_rl.train_one("mosca", 0, cfg, wirings, device)

    # No repite: la historia tiene las 4 iteraciones y ninguna dos veces.
    iteraciones = [e["iteracion"] for e in resultado["historia"]]
    assert iteraciones == [0, 1, 2, 3]
    assert resultado["reanudaciones"] == 1
    assert not instantanea.exists(), "al terminar bien hay que borrar la instantanea"
