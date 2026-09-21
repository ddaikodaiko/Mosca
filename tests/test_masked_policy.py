"""Tests de la politica actor-critico sobre el nucleo cableado (fase 6).

El test obligatorio del repo (mascara + ley de Dale) se repite aqui contra el bucle de
PPO, no solo contra el de fluidos: si al entrenar por refuerzo aparecieran sinapsis que
el conectoma no tiene, o una neurona cambiara de signo, el experimento no mediria el
cableado de la mosca sino otra cosa.
"""

from __future__ import annotations

import numpy as np
import torch
import yaml
from pathlib import Path

from envs.platformer import N_ACTIONS
from models.masked_policy import ConnectomePolicy, MLPPolicy, build_policy, hidden_for_budget
from models.masked_rnn import Wiring
from tests.test_masked_rnn import toy_wiring

N_FEATURES = 302   # 3 * 11 * 9 + 5, la observacion por defecto


def small_policy(seed: int = 0, **kwargs) -> ConnectomePolicy:
    torch.manual_seed(seed)
    return ConnectomePolicy(
        toy_wiring(n_neurons=40, seed=seed), N_FEATURES, N_ACTIONS,
        n_shared=4, n_sensory=12, n_motor=12, n_internal_steps=2, **kwargs,
    )


def train_a_few_steps(policy: ConnectomePolicy, steps: int = 30) -> None:
    """Un sustituto barato de PPO: misma ruta de gradiente, sin entorno."""
    optimiser = torch.optim.Adam(policy.parameters(), lr=0.05)
    generator = torch.Generator().manual_seed(0)
    h = policy.initial_state(8, torch.device("cpu"))
    for _ in range(steps):
        observation = torch.randn(8, N_FEATURES, generator=generator)
        logits, value, h = policy(observation, h.detach())
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        loss = -(distribution.log_prob(action) * torch.randn(8, generator=generator)).mean()
        loss = loss + value.pow(2).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()


def test_la_mascara_sigue_a_cero_tras_entrenar_la_politica() -> None:
    policy = small_policy()
    mask = policy.core.dense_matrix() != 0
    train_a_few_steps(policy)
    assert torch.all(policy.core.dense_matrix()[~mask] == 0)


def test_ninguna_neurona_cambia_de_signo_al_entrenar_la_politica() -> None:
    policy = small_policy()
    before = torch.sign(policy.core.weights())
    train_a_few_steps(policy)
    after = torch.sign(policy.core.weights())
    assert torch.equal(before, after)
    assert torch.all(after != 0)


def test_el_entrenamiento_mueve_los_pesos_del_nucleo() -> None:
    policy = small_policy()
    before = policy.core.theta.detach().clone()
    train_a_few_steps(policy)
    assert not torch.allclose(before, policy.core.theta.detach())


def test_el_estado_oculto_se_propaga_y_cambia_la_salida() -> None:
    policy = small_policy()
    observation = torch.randn(4, N_FEATURES)
    h0 = policy.initial_state(4, torch.device("cpu"))
    logits_a, _, h1 = policy(observation, h0)
    logits_b, _, _ = policy(observation, h1)
    assert not torch.allclose(h0, h1)
    assert not torch.allclose(logits_a, logits_b), "la politica ignora su propio estado"


def test_la_politica_arranca_casi_uniforme() -> None:
    policy = small_policy()
    logits, _, _ = policy(torch.randn(32, N_FEATURES), policy.initial_state(32, torch.device("cpu")))
    probabilities = logits.softmax(dim=1)
    assert probabilities.std(dim=1).max() < 0.02, "arranca sesgada hacia una accion"


def test_la_salida_tiene_la_forma_del_espacio_de_acciones() -> None:
    policy = small_policy()
    logits, value, h = policy(torch.zeros(5, N_FEATURES), policy.initial_state(5, torch.device("cpu")))
    assert logits.shape == (5, N_ACTIONS)
    assert value.shape == (5,)
    assert h.shape == (5, policy.n_neurons)


def test_el_mlp_iguala_el_presupuesto_de_parametros() -> None:
    """El MLP no puede ganar por tener mas parametros: se queda en el ancho mas cercano.

    Con un nucleo pequeno, una neurona de ancho ya son cientos de parametros, asi que lo
    exigible no es un porcentaje sino que ningun ancho vecino se acerque mas.
    """
    from models.masked_policy import mlp_parameter_count

    reference = small_policy()
    budget = sum(p.numel() for p in reference.parameters() if p.requires_grad)
    hidden = hidden_for_budget(budget, N_FEATURES, N_ACTIONS)
    mlp = MLPPolicy(N_FEATURES, N_ACTIONS, hidden)
    total = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    assert total == mlp_parameter_count(hidden, N_FEATURES, N_ACTIONS)
    error = abs(total - budget)
    for vecino in (hidden - 1, hidden + 1):
        assert error <= abs(mlp_parameter_count(vecino, N_FEATURES, N_ACTIONS) - budget)

    # A la escala real del experimento el ajuste ya es fino.
    grande = 4_000_000
    hidden = hidden_for_budget(grande, N_FEATURES, N_ACTIONS)
    assert abs(mlp_parameter_count(hidden, N_FEATURES, N_ACTIONS) - grande) / grande < 0.01


def test_build_policy_da_los_seis_modelos_con_los_mismos_puertos() -> None:
    """Mismo n_shared en todos: los atajos entrada->salida no pueden variar por modelo."""
    cfg = yaml.safe_load(Path("configs/mario_proto.yaml").read_text())
    directory = Path(cfg["wiring_dir"])
    if not (directory / "mosca.npz").exists():
        import pytest

        pytest.skip(f"faltan los cableados de {directory}")
    wirings = {n: Wiring.load(directory / f"{n}.npz") for n in ("mosca", "aleatorio", "recableado")}
    n_shared = int(cfg["encoder"]["n_shared"])
    for name in cfg["models"]:
        policy = build_policy(name, cfg, N_FEATURES, N_ACTIONS, seed=0, wirings=wirings)
        if isinstance(policy, MLPPolicy):
            continue
        shared = np.intersect1d(policy.sensory.numpy(), policy.motor.numpy())
        assert shared.size == n_shared, f"{name} comparte {shared.size} neuronas"


def test_la_normalizacion_saca_a_las_sensoriales_de_la_saturacion() -> None:
    """La normalizacion quita el sesgo y la escala de la ENTRADA, y nada mas.

    Medido en el NES: al inicializar no satura nadie (|impulso| medio 0,14 sin normalizar
    y 0,30 con), asi que aqui no arregla ningun problema de arranque; se deja porque es
    lo que pide el hallazgo de fluidos y porque duplica la senal util. El 66 % de
    sensoriales saturadas que se midio en el modelo entrenado viene del CRECIMIENTO de
    los pesos durante PPO, no de la escala de la entrada: eso no lo arregla normalizar.
    """
    torch.manual_seed(0)
    policy = small_policy()
    # Observacion con la forma de la real: casillas encendidas o apagadas, con media
    # lejos de cero, que es lo que descoloca al codificador.
    observaciones = (torch.rand(512, N_FEATURES) < 0.35).float() * 3.0 + 2.0

    def fraccion_saturada() -> float:
        with torch.no_grad():
            normalizada = (observaciones - policy.feature_mean) / policy.feature_std
            return float((policy.encoder(normalizada).abs() > 2.5).float().mean())

    antes = fraccion_saturada()
    policy.fit_feature_statistics(observaciones)
    despues = fraccion_saturada()
    assert antes > 0.2, f"el caso de prueba no satura ({antes:.0%})"
    assert despues < 0.05, f"sigue saturando: {antes:.0%} -> {despues:.0%}"

    # Lo que la normalizacion NO arregla: si el entrenamiento hace crecer los pesos, se
    # vuelve a saturar. Queda escrito para que nadie lea esto como una garantia.
    with torch.no_grad():
        policy.encoder.weight.mul_(8.0)
    assert fraccion_saturada() > 0.3


def test_las_estadisticas_de_entrada_viajan_en_el_checkpoint() -> None:
    policy = small_policy()
    policy.fit_feature_statistics(torch.randn(64, N_FEATURES) * 3.0 + 1.0)
    estado = policy.state_dict()
    assert "feature_mean" in estado and "feature_std" in estado
    copia = small_policy()
    copia.load_state_dict(estado)
    assert torch.equal(copia.feature_mean, policy.feature_mean)
    assert torch.equal(copia.feature_std, policy.feature_std)
    assert float(policy.feature_std.min()) >= 0.1, "el suelo de la desviacion no se aplica"


def test_sin_nucleo_no_tiene_sinapsis_pero_si_puertos() -> None:
    cfg = yaml.safe_load(Path("configs/mario_proto.yaml").read_text())
    directory = Path(cfg["wiring_dir"])
    if not (directory / "mosca.npz").exists():
        import pytest

        pytest.skip(f"faltan los cableados de {directory}")
    wirings = {n: Wiring.load(directory / f"{n}.npz") for n in ("mosca", "aleatorio", "recableado")}
    policy = build_policy("sin_nucleo", cfg, N_FEATURES, N_ACTIONS, 0, wirings)
    mosca = build_policy("mosca", cfg, N_FEATURES, N_ACTIONS, 0, wirings)
    assert policy.core.n_edges == 0
    assert np.array_equal(policy.sensory.numpy(), mosca.sensory.numpy())
    assert np.array_equal(policy.motor.numpy(), mosca.motor.numpy())
