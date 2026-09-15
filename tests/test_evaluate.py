"""Tests de la fase 5: metricas de evaluacion y prueba de permutacion.

Sobre campos sinteticos con la respuesta conocida de antemano: si una metrica no
acierta en un caso que se puede calcular a mano, no vale para comparar modelos.
"""

from __future__ import annotations

import numpy as np
import torch

from evaluate import (
    bandas_por_defecto,
    correlacion_espacial,
    error_espectral,
    k_maximo_del_codificador,
    prueba_de_permutacion,
    tiempo_hasta_descorrelacion,
)
from data.spectral import make_grid


def test_correlacion_de_un_campo_consigo_mismo_es_uno() -> None:
    campo = torch.randn(5, 16, 16)
    assert torch.allclose(correlacion_espacial(campo, campo), torch.ones(5), atol=1e-6)


def test_correlacion_de_un_campo_con_su_opuesto_es_menos_uno() -> None:
    campo = torch.randn(5, 16, 16)
    assert torch.allclose(correlacion_espacial(-campo, campo), -torch.ones(5), atol=1e-6)


def test_la_correlacion_ignora_el_desplazamiento_constante() -> None:
    """Es Pearson: sumar una constante al campo no cambia la correlacion."""
    campo = torch.randn(4, 16, 16)
    assert torch.allclose(
        correlacion_espacial(campo + 7.0, campo), torch.ones(4), atol=1e-6
    )


def test_tiempo_hasta_descorrelacion_encuentra_el_primer_cruce() -> None:
    # Cruza el umbral 0,8 en el indice 2 (tercer paso) -> tiempo 3 * dt.
    curvas = torch.tensor([[1.0, 0.9, 0.7, 0.5]])
    medido = float(tiempo_hasta_descorrelacion(curvas, dt_save=0.1, umbral=0.8)[0])
    assert abs(medido - 0.3) < 1e-9


def test_tiempo_hasta_descorrelacion_censura_si_nunca_cruza() -> None:
    """Si no baja del umbral se devuelve la duracion completa: es un dato censurado."""
    curvas = torch.tensor([[1.0, 0.99, 0.95, 0.9]])
    medido = float(tiempo_hasta_descorrelacion(curvas, dt_save=0.1, umbral=0.8)[0])
    assert abs(medido - 0.4) < 1e-9


def test_error_espectral_de_un_campo_consigo_mismo_es_cero() -> None:
    malla = make_grid(32, device=torch.device("cpu"))
    campo = torch.randn(3, 32, 32).double()
    bandas = bandas_por_defecto(32, 6.3)
    error = error_espectral(campo, campo, malla, bandas)
    assert all(valor < 1e-9 for valor in error.values())


def test_error_espectral_detecta_una_amplitud_escalada() -> None:
    """Multiplicar el campo por 10 multiplica E(k) por 100: log10 da 2 en toda banda."""
    malla = make_grid(32, device=torch.device("cpu"))
    campo = torch.randn(3, 32, 32).double()
    bandas = bandas_por_defecto(32, 6.3)
    error = error_espectral(campo * 10, campo, malla, bandas)
    assert all(abs(valor - 2.0) < 1e-6 for valor in error.values())


def test_permutacion_con_dos_muestras_identicas_no_encuentra_nada() -> None:
    valores = np.random.default_rng(0).normal(size=200)
    prueba = prueba_de_permutacion(valores, valores.copy(), 2000, semilla=1)
    assert prueba["diferencia_media"] == 0.0
    assert prueba["p"] == 1.0


def test_permutacion_detecta_una_diferencia_grande() -> None:
    generador = np.random.default_rng(0)
    a = generador.normal(size=200)
    b = a + 1.0  # desplazamiento enorme comparado con el ruido pareado
    prueba = prueba_de_permutacion(a, b, 2000, semilla=1)
    assert prueba["p"] < 0.01
    assert abs(prueba["diferencia_media"] + 1.0) < 1e-9


def test_permutacion_no_se_asusta_por_ruido() -> None:
    """Dos muestras pareadas sin diferencia real deben dar p alto."""
    generador = np.random.default_rng(3)
    a = generador.normal(size=300)
    b = a + generador.normal(scale=0.01, size=300)
    assert prueba_de_permutacion(a, b, 2000, semilla=1)["p"] > 0.05


def test_las_bandas_cubren_sin_solaparse() -> None:
    bandas = bandas_por_defecto(64, 9.1)
    for anterior, siguiente in zip(bandas, bandas[1:]):
        assert siguiente.k_min == anterior.k_max + 1
    assert bandas[-1].k_max == 64 // 3


def test_k_del_codificador_crece_con_el_numero_de_modos() -> None:
    assert k_maximo_del_codificador(64, 32) < k_maximo_del_codificador(64, 128)
    # Con 128 modos complejos a 64x64 el codificador llega a ~9,1 (medido en la fase 3).
    assert abs(k_maximo_del_codificador(64, 128) - 9.1) < 0.2
