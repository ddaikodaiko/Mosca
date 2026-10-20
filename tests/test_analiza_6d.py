"""Tests del analisis registrado de la fase 6d (`analiza_6d.py`).

Escritos, como el analisis, antes de tener datos. Los informes son sinteticos pero con la
estructura de los de `evaluate_rl.py`, y sus `contrastes` se calculan igual que alli.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from analiza_6c import MODELOS
from analiza_6d import (
    INFORMES,
    N_NIVELES,
    bateria,
    coincide_con_el_informe,
    comprueba,
    filas_de_informe,
    main,
    mejor_constante,
    niveles_en_orden,
)
from evaluate import prueba_de_permutacion

REFERENCIAS = {"azar": 0.20, "siempre_dcha": 0.08, "salto_cte": 0.10, "salto_corriendo": 0.12,
               "heuristica": 0.75}


def _informe(medias: dict[str, list[float]], muestreada: bool = False, ruido: float = 0.05,
             metas: dict[str, float] | None = None, semilla: int = 0) -> dict[str, Any]:
    """Informe con la forma del de `evaluate_rl.py`: progreso por nivel = media del modelo
    y semilla mas ruido por nivel. `medias[m]` da la media de cada semilla."""
    generador = np.random.default_rng(semilla)
    niveles = [str(3_000_000 + i) for i in range(N_NIVELES)]
    dificultad = generador.normal(0, 0.1, N_NIVELES)       # comun a todos: lo que empareja
    metas = metas or {}

    def serie(media: float) -> list[float]:
        return (media + dificultad + generador.normal(0, ruido, N_NIVELES)).tolist()

    def meta(nombre: str) -> list[float]:
        p = metas.get(nombre, 0.0)
        return (generador.random(N_NIVELES) < p).astype(float).tolist()

    series_por_semilla = {
        m: [{"checkpoint": f"{m}_s{s}", "progreso": serie(v), "meta": meta(m)}
            for s, v in enumerate(medias[m])]
        for m in MODELOS
    }
    series_referencias = {r: {"progreso": serie(v), "meta": meta(r)} for r, v in REFERENCIAS.items()}
    # Como en evaluate_rl: media por nivel sobre semillas y permutacion pareada.
    series = {m: np.mean([s["progreso"] for s in v], axis=0) for m, v in series_por_semilla.items()}
    series |= {r: np.asarray(v["progreso"]) for r, v in series_referencias.items()}
    parejas = [("mosca", "aleatorio"), ("mosca", "recableado"), ("mosca", "azar"),
               ("mlp", "azar"), ("mosca", "salto_corriendo")]
    contrastes = {f"{a}_vs_{b}": {"progreso": prueba_de_permutacion(series[a], series[b], 10000, 0)}
                  for a, b in parejas}
    return {
        "config": {"permutations": 10000}, "muestreada": muestreada,
        "partidas": [{"particion": "test", "semilla": int(n), "nivel": n} for n in niveles],
        "series_por_semilla": series_por_semilla, "series_referencias": series_referencias,
        "contrastes": contrastes,
    }


IGUALES = {m: [0.40, 0.42, 0.44] for m in MODELOS}


def _informes(medias: dict[str, list[float]] = IGUALES) -> dict[str, dict[str, Any]]:
    return {nombre: _informe(medias, muestreada=m, semilla=i)
            for i, (nombre, (_, m)) in enumerate(INFORMES.items())}


def test_comprueba_acepta_tres_informes_completos() -> None:
    comprueba(_informes())


def test_comprueba_para_si_falta_una_semilla() -> None:
    informes = _informes()
    informes["afinado"]["series_por_semilla"]["denso"].pop()
    with pytest.raises(SystemExit, match="denso"):
        comprueba(informes)


def test_comprueba_para_si_los_niveles_no_coinciden() -> None:
    informes = _informes()
    informes["clonado"]["partidas"][5]["nivel"] = "9999999"
    with pytest.raises(SystemExit, match="clonado"):
        comprueba(informes)


def test_comprueba_para_si_la_muestreada_no_lo_es() -> None:
    informes = _informes()
    informes["muestreada"]["muestreada"] = False
    with pytest.raises(SystemExit, match="muestreada"):
        comprueba(informes)


def test_las_p_coinciden_con_las_del_informe_y_si_no_para() -> None:
    informe = _informe(IGUALES)
    filas = filas_de_informe(informe, "progreso")
    resultado = bateria(filas, niveles_en_orden(informe), 10000, con_holm=True)
    contrastes = resultado["principales"] + resultado["secundarios"]
    assert coincide_con_el_informe(contrastes, informe) == 5
    informe["contrastes"]["mosca_vs_aleatorio"]["progreso"]["p"] += 0.01
    with pytest.raises(SystemExit, match="mosca vs aleatorio"):
        coincide_con_el_informe(contrastes, informe)


def test_mejor_constante() -> None:
    filas = filas_de_informe(_informe(IGUALES, ruido=0.0), "progreso")
    assert mejor_constante(filas) == "salto_corriendo"


def _escribe(tmp_path: Path, informes: dict[str, dict[str, Any]]) -> tuple[Path, Path]:
    afinado, clonado = tmp_path / "desde_clonado", tmp_path / "clonado"
    afinado.mkdir()
    clonado.mkdir()
    for nombre, (fichero, _) in INFORMES.items():
        destino = clonado if nombre == "clonado" else afinado
        (destino / fichero).write_text(json.dumps(informes[nombre]))
    return afinado, clonado


def _analiza(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
             informes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    afinado, clonado = _escribe(tmp_path, informes)
    salida = tmp_path / "analisis.json"
    monkeypatch.setattr("sys.argv", ["analiza_6d.py", "--dir", str(afinado), "--clonado",
                                     str(clonado), "--salida", str(salida)])
    assert main() == 0
    return json.loads(salida.read_text())


def test_una_mosca_claramente_mejor_gana_con_holm(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    medias = dict(IGUALES) | {"mosca": [0.60, 0.62, 0.64]}
    analisis = _analiza(tmp_path, monkeypatch, _informes(medias))
    principales = {c["b"]: c for c in analisis["afinado"]["principales"]}
    assert principales["aleatorio"]["veredicto"] == "gana"
    assert principales["recableado"]["veredicto"] == "gana"
    assert principales["aleatorio"]["umbral"] in (0.025, 0.05)        # Holm
    assert analisis["p_comprobadas_contra_el_informe"] == 15


@pytest.mark.parametrize("mosca, esperado", [
    # +0,10 con p ~0 en 200 niveles, pero una semilla cae por debajo de las del
    # aleatorio (regla b) y la diferencia supera la dispersion: no resuelto.
    ([0.39, 0.58, 0.60], "no resuelto"),
    # +0,04, mismas semillas mezcladas, y la diferencia cabe en la dispersion entre
    # semillas (0,044): la regla registrada lo llama empate aunque la p sea ~0.
    ([0.39, 0.48, 0.52], "empate"),
])
def test_semillas_mezcladas_no_ganan_aunque_la_p_sea_baja(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mosca: list[float], esperado: str) -> None:
    medias = dict(IGUALES) | {"mosca": mosca}
    analisis = _analiza(tmp_path, monkeypatch, _informes(medias))
    c = next(c for c in analisis["afinado"]["principales"] if c["b"] == "aleatorio")
    assert c["p_nivel"] < 0.025 and not c["separadas"]
    assert c["veredicto"] == esperado


def test_iguales_dan_empate_o_no_resuelto_nunca_gana(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    analisis = _analiza(tmp_path, monkeypatch, _informes())
    for c in analisis["afinado"]["principales"]:
        assert c["veredicto"] in ("empate", "no resuelto")


def test_juega_exige_meta_y_ganar_al_azar_y_a_la_constante() -> None:
    medias = dict(IGUALES) | {"mosca": [0.60, 0.62, 0.64]}
    informe = _informe(medias, metas={"mosca": 0.1})
    filas = filas_de_informe(informe, "progreso")
    resultado = bateria(filas, niveles_en_orden(informe), 10000, con_holm=True)
    from analiza_6d import juega
    filas_juega = juega(filas_de_informe(informe, "meta"), resultado)
    assert filas_juega["mosca"]["juega"]
    assert filas_juega["mosca"]["niveles_con_meta"] > 0
    # el aleatorio gana al azar y a la constante, pero no llega nunca a la meta
    assert filas_juega["aleatorio"]["contra_azar"] == "gana"
    assert not filas_juega["aleatorio"]["juega"]


def test_la_ganancia_por_ppo_se_mide_contra_el_clonado_del_mismo_modelo(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    informes = _informes()
    informes["clonado"] = _informe(dict(IGUALES) | {"mlp": [0.20, 0.22, 0.24]}, semilla=2)
    analisis = _analiza(tmp_path, monkeypatch, informes)
    ganancia = {c["a"]: c for c in analisis["ganancia_por_ppo"]}
    assert ganancia["mlp_afinado"]["b"] == "mlp_clonado"
    assert ganancia["mlp_afinado"]["diferencia"] == pytest.approx(0.20, abs=0.02)
    assert ganancia["mlp_afinado"]["veredicto"] == "gana"
    assert ganancia["mosca_afinado"]["veredicto"] in ("empate", "no resuelto")
