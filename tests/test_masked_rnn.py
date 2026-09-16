"""Test obligatorio de la fase 3: la mascara y la ley de Dale se respetan al entrenar.

El criterio del encargo: tras 100 pasos de optimizacion, las entradas fuera de la
mascara siguen siendo EXACTAMENTE cero y ninguna neurona ha cambiado de signo.
"""

from __future__ import annotations

import math

from pathlib import Path

import numpy as np
import pytest
import torch

from models.fno import FNO2d, Persistence
from models.masked_rnn import (
    ConnectomeFlowModel,
    MaskedRecurrentCore,
    Wiring,
    dense_neurons_for_budget,
    low_frequency_modes,
    select_ports,
)

ARTIFACTS = Path("artifacts/conectoma_4000")


def toy_wiring(n_neurons: int = 40, density: float = 0.15, seed: int = 0) -> Wiring:
    """Cableado pequeno pero con la misma estructura que el de la fase 2."""
    rng = np.random.default_rng(seed)
    n_slots = n_neurons * (n_neurons - 1)
    n_edges = int(density * n_slots)
    chosen = rng.choice(n_slots, size=n_edges, replace=False)
    rows = (chosen // (n_neurons - 1)).astype(np.int32)
    offsets = (chosen % (n_neurons - 1)).astype(np.int32)
    cols = (offsets + (offsets >= rows)).astype(np.int32)
    sign = np.where(rng.random(n_neurons) < 0.4, -1, 1).astype(np.int8)
    synapses = rng.integers(1, 50, size=n_edges).astype(np.int32)
    return Wiring(rows=rows, cols=cols, synapses=synapses, sign=sign)


def test_la_mascara_sigue_a_cero_tras_100_pasos_de_optimizacion() -> None:
    wiring = toy_wiring()
    model = ConnectomeFlowModel(wiring, grid=16, n_complex_modes=8, n_sensory=8, n_motor=8, n_shared=2)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)

    outside = model.core.dense_matrix().detach() == 0
    mask = torch.zeros(wiring.n_neurons, wiring.n_neurons, dtype=torch.bool)
    mask[model.core.edge_index[0], model.core.edge_index[1]] = True
    assert torch.equal(outside, ~mask)

    torch.manual_seed(0)
    target = torch.randn(3, 16, 16)
    for _ in range(100):
        optimiser.zero_grad()
        prediction, _, _ = model(torch.randn(3, 16, 16))
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        optimiser.step()

    final = model.core.dense_matrix().detach()
    assert torch.all(final[~mask] == 0.0)          # exactamente cero, no "casi"
    assert torch.count_nonzero(final) <= wiring.n_edges


def test_ninguna_neurona_cambia_de_signo_al_entrenar() -> None:
    wiring = toy_wiring(seed=1)
    model = ConnectomeFlowModel(wiring, grid=16, n_complex_modes=8, n_sensory=8, n_motor=8, n_shared=2)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-1)  # lr agresivo a proposito

    before = torch.sign(model.core.weights().detach())
    torch.manual_seed(1)
    target = torch.randn(3, 16, 16)
    for _ in range(100):
        optimiser.zero_grad()
        prediction, _, _ = model(torch.randn(3, 16, 16))
        torch.nn.functional.mse_loss(prediction, target).backward()
        optimiser.step()

    after = torch.sign(model.core.weights().detach())
    assert torch.equal(before, after)
    expected = torch.from_numpy(wiring.sign[wiring.rows].astype(np.float32))
    assert torch.equal(after, expected)


def test_el_entrenamiento_mueve_de_verdad_los_pesos() -> None:
    """Contrapeso de los dos anteriores: pasarian tambien con gradiente nulo."""
    wiring = toy_wiring(seed=2)
    model = ConnectomeFlowModel(wiring, grid=16, n_complex_modes=8, n_sensory=8, n_motor=8, n_shared=2)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = model.core.theta.detach().clone()

    torch.manual_seed(2)
    target = torch.randn(3, 16, 16)
    for _ in range(100):
        optimiser.zero_grad()
        prediction, _, _ = model(torch.randn(3, 16, 16))
        torch.nn.functional.mse_loss(prediction, target).backward()
        optimiser.step()

    assert not torch.allclose(before, model.core.theta.detach())


def test_radio_espectral_inicial_cerca_del_objetivo() -> None:
    wiring = toy_wiring(seed=3)
    core = MaskedRecurrentCore(wiring, spectral_radius=0.95)
    radius = float(np.max(np.abs(np.linalg.eigvals(core.dense_matrix().detach().numpy()))))
    assert abs(radius - 0.95) < 1e-3


def test_los_puertos_comparten_exactamente_n_shared_neuronas() -> None:
    wiring = toy_wiring(seed=4)
    for n_shared in (0, 3, 8):
        sensory, motor = select_ports(wiring, 8, 8, n_shared)
        assert sensory.size == 8 and motor.size == 8
        assert len(set(sensory.tolist()) & set(motor.tolist())) == n_shared
    out_degree = np.bincount(wiring.rows, minlength=wiring.n_neurons)
    assert out_degree[sensory].min() >= np.delete(out_degree, sensory).max()
    with pytest.raises(ValueError):
        select_ports(wiring, 30, 30, 0)  # 60 > 40 neuronas
    with pytest.raises(ValueError):
        select_ports(wiring, 8, 4, 5)  # mas compartidas que motoras


def test_el_denso_arranca_con_los_puertos_de_la_config_real() -> None:
    """Regresion: 420 neuronas no admitian 256 + 256 puertos disjuntos."""
    wiring = Wiring.dense(dense_neurons_for_budget(175_744), 0.4, seed=0)
    assert wiring.n_neurons == 256 + 256 - 92
    sensory, motor = select_ports(wiring, 256, 256, 92)
    assert len(set(sensory.tolist()) & set(motor.tolist())) == 92


def test_codificar_y_decodificar_es_un_filtro_paso_bajo_idempotente() -> None:
    """Decode(encode(x)) debe ser la proyeccion a los modos bajos: aplicarla dos veces
    tiene que dar lo mismo. Fija el contrato del codificador."""
    wiring = toy_wiring(seed=5)
    model = ConnectomeFlowModel(wiring, grid=16, n_complex_modes=8, n_sensory=8, n_motor=8, n_shared=2)
    x = torch.randn(2, 16, 16)
    once = model.decode(model.encode(x))
    twice = model.decode(model.encode(once))
    assert torch.allclose(once, twice, atol=1e-5)


def test_modos_bajos_ordenados_por_frecuencia() -> None:
    index, mirror_source, mirror_target = low_frequency_modes(16, 8)
    assert index.size == 8
    assert np.array_equal(index, np.unique(index))
    assert 0 in index.tolist()  # el modo k=0 es el de menor frecuencia
    assert mirror_source.size == mirror_target.size
    assert not set(mirror_target.tolist()) & set(index.tolist())  # espejo != independiente


def test_el_codificador_conserva_la_energia_de_los_modos_que_guarda() -> None:
    """Si se pierde el espejo conjugado, la amplitud de esos modos se parte por dos."""
    wiring = toy_wiring(seed=8)
    model = ConnectomeFlowModel(wiring, grid=16, n_complex_modes=8, n_sensory=8, n_motor=8, n_shared=2)
    x = torch.randn(3, 16, 16)
    filtered = model.decode(model.encode(x))
    assert torch.allclose(model.encode(filtered), model.encode(x), atol=1e-5)


def test_prediccion_residual_arranca_cerca_de_la_persistencia() -> None:
    wiring = toy_wiring(seed=6)
    model = ConnectomeFlowModel(wiring, grid=16, n_complex_modes=8, n_sensory=8, n_motor=8, n_shared=2)
    x = torch.randn(4, 16, 16)
    prediction, _, _ = model(x)
    assert float((prediction - x).std()) < float(x.std())


def test_el_denso_iguala_el_presupuesto_de_parametros() -> None:
    n_edges = 175_744
    n_dense = dense_neurons_for_budget(n_edges)
    assert abs(n_dense * (n_dense - 1) - n_edges) / n_edges < 0.01


def test_wiring_denso_no_tiene_autoconexiones() -> None:
    wiring = Wiring.dense(20, 0.4, seed=0)
    assert wiring.n_edges == 20 * 19
    assert not np.any(wiring.rows == wiring.cols)
    assert abs(float((wiring.sign < 0).mean()) - 0.4) < 0.06


def test_traza_de_activaciones_para_el_visualizador() -> None:
    """El visualizador necesita las N activaciones de cada paso interno."""
    wiring = toy_wiring(seed=7)
    model = ConnectomeFlowModel(wiring, grid=16, n_complex_modes=8, n_sensory=8, n_motor=8, n_shared=2)
    _, _, trace = model(torch.randn(2, 16, 16), record=True)
    assert len(trace) == model.core.n_internal_steps
    assert trace[0].shape == (2, wiring.n_neurons)


def test_fno_y_persistencia_comparten_contrato() -> None:
    x = torch.randn(2, 16, 16)
    for model in (FNO2d(modes=4, width=8), Persistence()):
        assert model(x).shape == x.shape
        assert model.rollout(x, 3).shape == (2, 3, 16, 16)
    assert torch.equal(Persistence()(x), x)


@pytest.mark.skipif(not (ARTIFACTS / "mosca.npz").is_file(), reason="subgrafo no generado")
def test_los_tres_cableados_tienen_los_mismos_parametros_entrenables() -> None:
    """Seccion 7 del diseno: los controles solo pueden diferir en la mascara."""
    counts = {}
    for name in ("mosca", "aleatorio", "recableado"):
        wiring = Wiring.load(ARTIFACTS / f"{name}.npz")
        model = ConnectomeFlowModel(wiring, grid=64, n_shared=92)
        counts[name] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert len(set(counts.values())) == 1, counts


def test_la_normalizacion_de_entrada_saca_a_tanh_de_la_saturacion() -> None:
    """Los coeficientes de Fourier de un campo de rms ~2 llegan a valer ~20: sin
    normalizar, las neuronas sensoriales saturan y el gradiente se muere."""
    wiring = toy_wiring(seed=9)
    model = ConnectomeFlowModel(wiring, grid=16, n_complex_modes=8, n_sensory=8, n_motor=8, n_shared=2)
    campo = torch.randn(64, 16, 16) * 2.2

    crudo = model.encode(campo)
    model.fit_feature_statistics(campo)
    normalizado = model.encode(campo)

    assert float(crudo.std()) > 1.5
    assert abs(float(normalizado.std()) - 1.0) < 0.1
    assert float(normalizado.mean().abs()) < 0.1


def test_las_estadisticas_viajan_en_el_state_dict() -> None:
    """Si no van en el checkpoint, el modelo evaluado no es el entrenado."""
    wiring = toy_wiring(seed=10)
    model = ConnectomeFlowModel(wiring, grid=16, n_complex_modes=8, n_sensory=8, n_motor=8, n_shared=2)
    model.fit_feature_statistics(torch.randn(32, 16, 16) * 3.0)
    assert "feature_std" in model.state_dict()

    copia = ConnectomeFlowModel(wiring, grid=16, n_complex_modes=8, n_sensory=8, n_motor=8, n_shared=2)
    copia.load_state_dict(model.state_dict())
    x = torch.randn(4, 16, 16)
    assert torch.allclose(model.encode(x), copia.encode(x))


def test_el_aumento_traslada_entrada_y_objetivo_por_lo_mismo() -> None:
    """Si el par se desplaza distinto, deja de ser una transicion valida del simulador."""
    from train import Augmenter

    augmenter = Augmenter(enabled=True, forcing_wavenumber=4)
    generator = torch.Generator().manual_seed(0)
    campo = torch.randn(8, 16, 16)
    movido, mismo = augmenter(campo, campo.clone(), generator=generator)

    assert torch.equal(movido, mismo)                      # el mismo desplazamiento
    assert not torch.equal(movido, campo)                  # y desplaza de verdad
    assert torch.allclose(movido.std(dim=(1, 2)), campo.std(dim=(1, 2)), atol=1e-5)


def test_el_aumento_produce_transiciones_validas_del_simulador() -> None:
    """El par aumentado tiene que ser exactamente lo que el simulador habria dado.

    Regresion: los ejes estaban cambiados y la traslacion libre caia sobre el eje del
    forzado; los pares aumentados tenian un 4 % de error fisico y ningun test lo veia
    porque el test anterior cometia el mismo error de ejes.
    """
    from data.spectral import make_grid, rk4_step
    from train import Augmenter

    grid = 16
    spectral = make_grid(grid, device=torch.device("cpu"))
    rng = np.random.default_rng(0)
    omega = torch.from_numpy(rng.standard_normal((6, grid, grid)))

    def simulate(field: torch.Tensor) -> torch.Tensor:
        w_hat = torch.fft.rfft2(field)
        for _ in range(5):
            w_hat = rk4_step(w_hat, 0.01, spectral, 1e-3, 0.1)
        return torch.fft.irfft2(w_hat, s=(grid, grid))

    target = simulate(omega)
    augmenter = Augmenter(enabled=True, forcing_wavenumber=4)
    moved, moved_target = augmenter(omega, target, generator=torch.Generator().manual_seed(3))
    assert not torch.equal(moved, omega)
    error = torch.linalg.vector_norm(simulate(moved) - moved_target) / torch.linalg.vector_norm(moved_target)
    assert float(error) < 1e-10

    # Y el eje de las columnas es el forzado: trasladar una casilla ahi NO es simetria.
    broken = torch.roll(omega, 1, dims=-1)
    error = torch.linalg.vector_norm(simulate(broken) - torch.roll(target, 1, dims=-1))
    assert float(error / torch.linalg.vector_norm(target)) > 1e-4


def test_el_aumento_desactivado_no_toca_nada() -> None:
    from train import Augmenter

    augmenter = Augmenter(enabled=False, forcing_wavenumber=4)
    campo = torch.randn(4, 16, 16)
    (igual,) = augmenter(campo, generator=torch.Generator().manual_seed(0))
    assert torch.equal(igual, campo)


def test_la_traslacion_continua_produce_transiciones_validas_del_simulador() -> None:
    """Desplazar fracciones de casilla por fase de Fourier sigue siendo simetria exacta
    en x para campos de banda limitada (el simulador desaliasa por encima de n/3)."""
    from data.spectral import make_grid, rk4_step
    from train import Augmenter

    grid = 16
    spectral = make_grid(grid, device=torch.device("cpu"))
    rng = np.random.default_rng(1)
    omega_hat = torch.fft.rfft2(torch.from_numpy(rng.standard_normal((6, grid, grid))))
    omega = torch.fft.irfft2(omega_hat * spectral.dealias, s=(grid, grid))  # banda limitada

    def simulate(field: torch.Tensor) -> torch.Tensor:
        w_hat = torch.fft.rfft2(field)
        for _ in range(5):
            w_hat = rk4_step(w_hat, 0.01, spectral, 1e-3, 0.1)
        return torch.fft.irfft2(w_hat, s=(grid, grid))

    target = simulate(omega)
    augmenter = Augmenter(enabled=True, forcing_wavenumber=4, continuous_x=True)
    moved, moved_target = augmenter(omega, target, generator=torch.Generator().manual_seed(5))
    assert not torch.equal(moved, omega)
    error = torch.linalg.vector_norm(simulate(moved) - moved_target) / torch.linalg.vector_norm(moved_target)
    assert float(error) < 1e-10


def test_el_orden_de_sorteo_del_aumento_entero_no_cambia() -> None:
    """Regresion de reproducibilidad: con continuous_x apagado, las traslaciones son las
    mismas que antes de anadir la opcion (x primero, y despues, del mismo generador)."""
    from train import Augmenter

    grid, batch = 16, 8
    campo = torch.arange(batch * grid * grid, dtype=torch.float64).reshape(batch, grid, grid)
    (movido,) = Augmenter(enabled=True, forcing_wavenumber=4)(campo, generator=torch.Generator().manual_seed(7))
    generator = torch.Generator().manual_seed(7)
    shift_x = torch.randint(0, grid, (batch,), generator=generator)
    shift_y = torch.randint(0, 4, (batch,), generator=generator) * (grid // 4)
    esperado = torch.stack([torch.roll(campo[i], (int(shift_x[i]), int(shift_y[i])), dims=(0, 1))
                            for i in range(batch)])
    assert torch.equal(movido, esperado)


def test_sin_nucleo_no_tiene_conexiones_y_usa_los_puertos_de_la_mosca() -> None:
    wiring = toy_wiring(seed=9)
    empty = Wiring.without_edges(wiring)
    model = ConnectomeFlowModel(empty, grid=16, n_complex_modes=8, n_sensory=8, n_motor=8,
                                n_shared=2, port_wiring=wiring)
    reference = ConnectomeFlowModel(wiring, grid=16, n_complex_modes=8, n_sensory=8, n_motor=8,
                                    n_shared=2)
    assert model.core.theta.numel() == 0
    assert torch.equal(model.sensory, reference.sensory) and torch.equal(model.motor, reference.motor)
    x = torch.randn(3, 16, 16)
    prediction, h, _ = model(x)
    prediction.pow(2).mean().backward()
    assert torch.isfinite(h).all() and model.encoder.weight.grad is not None
