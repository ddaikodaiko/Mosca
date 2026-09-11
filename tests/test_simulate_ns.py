"""Tests de la fase 1: operadores espectrales y simulador de Navier-Stokes."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml
from pathlib import Path

from data.spectral import (
    cfl_number, energy_spectrum, kinetic_energy, make_grid, random_vorticity,
    resolve_device, rhs, rk4_step, velocity_from_vorticity,
)
from data.simulate_ns import _seeds, simulate_batch, spectrum_slope

N = 32
NU, ALPHA = 1e-3, 0.1


@pytest.fixture(scope="module")
def grid():
    return make_grid(N, device=torch.device("cpu"), dtype=torch.float64)


def test_resolve_device_detecta_cuda():
    """El dispositivo se detecta, nunca se fija a mano."""
    esperado = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto").type == esperado
    assert resolve_device("cpu").type == "cpu"


def test_poisson_recupera_vorticidad_analitica(grid):
    """Para psi = cos(3x)sin(2y), u y v deben salir exactos de w = -lap(psi)."""
    x = torch.linspace(0, 2 * np.pi, N + 1, dtype=torch.float64)[:-1]
    X, Y = torch.meshgrid(x, x, indexing="ij")
    psi = torch.cos(3 * X) * torch.sin(2 * Y)
    w = (3**2 + 2**2) * psi                      # w = -lap(psi)
    u_exacta = 2 * torch.cos(3 * X) * torch.cos(2 * Y)      # d(psi)/dy
    v_exacta = 3 * torch.sin(3 * X) * torch.sin(2 * Y)      # -d(psi)/dx

    u, v = velocity_from_vorticity(torch.fft.rfft2(w), grid)
    assert torch.allclose(u, u_exacta, atol=1e-10)
    assert torch.allclose(v, v_exacta, atol=1e-10)


def test_mascara_desaliasing_2_3(grid):
    """La mascara vale 1 por debajo de n/3 y exactamente 0 por encima."""
    k_cut = N // 3
    dentro = (grid.kx.abs() <= k_cut) & (grid.ky.abs() <= k_cut)
    assert torch.all(grid.dealias[dentro] == 1.0)
    assert torch.all(grid.dealias[~dentro] == 0.0)
    assert grid.dealias.sum() > 0


def test_energia_se_conserva_sin_forzado_ni_disipacion():
    """Euler 2D puro: la energia se conserva. Mide el error del integrador."""
    g = make_grid(N, device=torch.device("cpu"), dtype=torch.float64)
    g_sin_f = type(g)(**{**g.__dict__, "forcing_hat": torch.zeros_like(g.forcing_hat)})
    w_hat = torch.fft.rfft2(torch.from_numpy(random_vorticity(N, seed=7)[None]))

    e0 = float(kinetic_energy(w_hat, g_sin_f))
    for _ in range(100):
        w_hat = rk4_step(w_hat, 1e-3, g_sin_f, nu=0.0, alpha=0.0)
    e1 = float(kinetic_energy(w_hat, g_sin_f))
    assert abs(e1 - e0) / e0 < 1e-6


def test_arrastre_disipa_energia(grid):
    """Sin forzado pero con arrastre, la energia tiene que bajar."""
    g_sin_f = type(grid)(**{**grid.__dict__,
                            "forcing_hat": torch.zeros_like(grid.forcing_hat)})
    w_hat = torch.fft.rfft2(torch.from_numpy(random_vorticity(N, seed=11)[None]))
    e0 = float(kinetic_energy(w_hat, g_sin_f))
    for _ in range(200):
        w_hat = rk4_step(w_hat, 1e-2, g_sin_f, NU, ALPHA)
    assert float(kinetic_energy(w_hat, g_sin_f)) < e0


def test_condicion_inicial_media_cero_y_rms_unidad():
    w = random_vorticity(N, seed=3, rms=1.0)
    assert abs(w.mean()) < 1e-12
    assert abs(np.sqrt((w**2).mean()) - 1.0) < 1e-12


def test_condicion_inicial_es_determinista_e_independiente_de_la_semilla():
    assert np.array_equal(random_vorticity(N, 42), random_vorticity(N, 42))
    assert not np.array_equal(random_vorticity(N, 42), random_vorticity(N, 43))


def test_trayectoria_reproducible_bit_a_bit(grid):
    """Misma semilla, misma trayectoria, hasta el ultimo bit."""
    args = ([1000], grid, 1e-2, NU, ALPHA, 50, 5, 10, 4.0, 1.0)
    a = simulate_batch(*args)["omega"]
    b = simulate_batch(*args)["omega"]
    assert np.array_equal(a, b)


def test_trayectoria_sin_nan_y_con_forma_y_dtype_correctos(grid):
    res = simulate_batch([1000, 1001], grid, 1e-2, NU, ALPHA, 50, 5, 10, 4.0, 1.0)
    assert res["omega"].shape == (2, 5, N, N)
    assert res["omega"].dtype == np.float32
    assert np.isfinite(res["omega"]).all()
    assert float(res["max_cfl"]) < 0.5


def test_cfl_crece_de_forma_lineal_con_dt(grid):
    w_hat = torch.fft.rfft2(torch.from_numpy(random_vorticity(N, seed=5)[None]))
    c1 = float(cfl_number(w_hat, grid, 1e-3))
    c2 = float(cfl_number(w_hat, grid, 2e-3))
    assert c1 > 0 and abs(c2 - 2 * c1) < 1e-12


def test_espectro_es_positivo_y_suma_la_energia_total(grid):
    """La suma de E(k) sobre todos los k debe dar la energia cinetica total."""
    w_hat = torch.fft.rfft2(torch.from_numpy(random_vorticity(N, seed=9)[None]))
    k, e_k = energy_spectrum(w_hat, grid)
    assert (e_k >= 0).all()
    assert k[0] == 0 and len(k) == N // 3 + 1
    # El espectro se trunca en n/3, asi que solo puede acercarse por debajo.
    assert 0.90 < e_k.sum() / float(kinetic_energy(w_hat, grid)) <= 1.0 + 1e-9


def test_ajuste_de_pendiente_recupera_una_ley_de_potencias_conocida():
    k = np.arange(0, 21, dtype=float)
    e_k = np.where(k > 0, np.where(k > 0, k, 1.0) ** -3.0, 0.0)
    assert abs(spectrum_slope(k, e_k, 5, 20) - (-3.0)) < 1e-9


@pytest.mark.parametrize("nombre", ["ns_64", "ns_32_proto", "ns_64_validacion"])
def test_las_semillas_de_train_val_y_test_son_disjuntas(nombre):
    cfg = yaml.safe_load(Path(f"configs/{nombre}.yaml").read_text())
    conjuntos = {s: set(_seeds(c)) for s, c in cfg["splits"].items()}
    assert conjuntos["train"].isdisjoint(conjuntos["val"])
    assert conjuntos["train"].isdisjoint(conjuntos["test"])
    assert conjuntos["val"].isdisjoint(conjuntos["test"])


@pytest.mark.parametrize("nombre", ["ns_64", "ns_32_proto", "ns_64_validacion"])
def test_los_yaml_respetan_los_parametros_fisicos_del_diseno(nombre):
    sim = yaml.safe_load(Path(f"configs/{nombre}.yaml").read_text())["simulation"]
    assert sim["nu"] == 1e-3
    assert sim["alpha"] == 0.1
    assert sim["forcing_wavenumber"] == 4
