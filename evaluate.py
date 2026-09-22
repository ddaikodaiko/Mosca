"""Evaluacion de los seis modelos sobre el conjunto de test (fase 5).

Uso:
    python evaluate.py --config configs/entrenamiento_32_proto.yaml

Metricas del encargo, todas sobre test y todas con semillas disjuntas de train y val:

  1. error L2 relativo a un paso
  2. error L2 relativo en despliegue de 50 pasos
  3. tiempo hasta que la correlacion espacial cae por debajo de 0,8
  4. error en el espectro de energia, POR BANDAS y en escala logaritmica

La cuarta va por bandas por una razon medida en la fase 1: el 91 % de la energia esta
en k <= 4. En crudo, la metrica espectral la dominarian cuatro modos y no distinguiria
entre modelos. Y hay una banda, k > 9, en la que todos los cableados son persistencia
por construccion, porque el codificador no llega ahi (medido en la fase 3).

Se informa media +- desviacion tipica ENTRE SEMILLAS y una prueba de permutacion de
`mosca` contra cada control. Si la diferencia cae dentro de la variabilidad entre
semillas, se dice explicitamente: es la hipotesis H0 y es el resultado que esperamos.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from data.spectral import energy_spectrum, make_grid, resolve_device
from models.masked_rnn import low_frequency_modes
from train import Trajectories, build_model, relative_l2

# Paleta categorica en orden fijo, nunca ciclada. La persistencia no es una serie:
# es la linea de referencia, y va en gris neutro.
COLORES: dict[str, str] = {
    "mosca": "#2a78d6",
    "aleatorio": "#eb6834",
    "recableado": "#1baf7a",
    "denso": "#eda100",
    "sin_nucleo": "#8a5cd8",
    "fno": "#e87ba4",
}
GRIS_REFERENCIA = "#52514e"
TINTA_PRIMARIA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
SUPERFICIE = "#fcfcfb"
ORDEN = ["mosca", "aleatorio", "recableado", "denso", "sin_nucleo", "fno"]


@dataclass(frozen=True)
class Banda:
    """Una banda de numeros de onda para la metrica espectral."""

    nombre: str
    k_min: int
    k_max: int

    @property
    def etiqueta(self) -> str:
        return f"{self.nombre} (k {self.k_min}-{self.k_max})"


def k_maximo_del_codificador(grid: int, n_complex_modes: int) -> float:
    """Hasta que |k| llega el codificador. Por encima de ahi ningun cableado corrige
    nada: el incremento predicho es cero por construccion y todos son persistencia."""
    modos, _, _ = low_frequency_modes(grid, n_complex_modes)
    n_columnas = grid // 2 + 1
    fila, columna = np.divmod(modos, n_columnas)
    kx = np.fft.fftfreq(grid, d=1.0 / grid)[fila]
    ky = np.fft.rfftfreq(grid, d=1.0 / grid)[columna]
    return float(np.sqrt(kx**2 + ky**2).max())


def bandas_por_defecto(grid: int, k_encoder: float) -> list[Banda]:
    """Inyeccion, banda que el codificador alcanza, y banda que no alcanza."""
    k_cut = grid // 3
    k_alto = int(np.floor(k_encoder))
    return [
        Banda("inyeccion", 1, 4),
        Banda("alcanzable", 5, k_alto),
        Banda("fuera del codificador", k_alto + 1, k_cut),
    ]


def correlacion_espacial(prediccion: torch.Tensor, objetivo: torch.Tensor) -> torch.Tensor:
    """Pearson sobre el campo, por muestra. Forma de entrada (batch, n, n)."""
    a = prediccion.flatten(1) - prediccion.flatten(1).mean(dim=1, keepdim=True)
    b = objetivo.flatten(1) - objetivo.flatten(1).mean(dim=1, keepdim=True)
    numerador = (a * b).sum(dim=1)
    denominador = a.norm(dim=1) * b.norm(dim=1)
    return numerador / denominador.clamp_min(1e-12)


def tiempo_hasta_descorrelacion(
    correlaciones: torch.Tensor, dt_save: float, umbral: float
) -> torch.Tensor:
    """Primer instante en que la correlacion baja del umbral, por trayectoria.

    Si no baja en todo el despliegue se devuelve la duracion completa, y eso hay que
    leerlo como "al menos esto", no como el valor real: es un dato censurado.
    """
    por_debajo = correlaciones < umbral
    hay = por_debajo.any(dim=1)
    primer = torch.where(hay, por_debajo.float().argmax(dim=1), correlaciones.shape[1] - 1)
    return (primer + 1).to(torch.float64) * dt_save


@torch.no_grad()
def error_espectral(
    prediccion: torch.Tensor, objetivo: torch.Tensor, grid_spec: Any, bandas: list[Banda]
) -> dict[str, float]:
    """Desviacion logaritmica media de E(k) por banda: |log10(E_pred / E_real)|."""
    k, e_pred = energy_spectrum(torch.fft.rfft2(prediccion.to(torch.float64)), grid_spec)
    _, e_real = energy_spectrum(torch.fft.rfft2(objetivo.to(torch.float64)), grid_spec)
    salida: dict[str, float] = {}
    for banda in bandas:
        seleccion = (k >= banda.k_min) & (k <= banda.k_max)
        pred = np.clip(e_pred[seleccion], 1e-30, None)
        real = np.clip(e_real[seleccion], 1e-30, None)
        salida[banda.nombre] = float(np.abs(np.log10(pred / real)).mean())
    return salida


@torch.no_grad()
def evalua_modelo(
    modelo: torch.nn.Module,
    test: Trajectories,
    pasos: int,
    umbral: float,
    dt_save: float,
    grid_spec: Any,
    bandas: list[Banda],
    lote: int,
) -> dict[str, Any]:
    modelo.eval()
    entradas, objetivos = test.pairs()

    # Error a un paso PAR A PAR, no solo su media: el contraste entre modelos a un paso es
    # donde se juega el resultado (la mosca le gana a sin_nucleo por 0,0004 en el ensayo),
    # y sin los valores pareados no hay prueba de permutacion que lo juzgue. El orden de
    # los pares es el mismo para todos los modelos, asi que estan pareados por construccion.
    n_pares = entradas.shape[0]
    error_1paso_por_par: list[float] = []
    for inicio in range(0, n_pares, lote):
        trozo = entradas[inicio : inicio + lote]
        objetivo = objetivos[inicio : inicio + lote]
        salida = modelo(trozo)
        salida = salida[0] if isinstance(salida, tuple) else salida
        diferencia = (salida - objetivo).flatten(1).norm(dim=1)
        escala = objetivo.flatten(1).norm(dim=1).clamp_min(1e-12)
        error_1paso_por_par.extend((diferencia / escala).cpu().tolist())
    error_1paso = float(np.mean(error_1paso_por_par))

    l2_por_trayectoria: list[float] = []
    tiempos: list[float] = []
    correlaciones_medias: list[np.ndarray] = []
    espectros: list[dict[str, float]] = []

    inicio_campo = test.omega[:, 0]
    real = test.omega[:, 1 : pasos + 1]
    for comienzo in range(0, inicio_campo.shape[0], lote):
        trozo = slice(comienzo, comienzo + lote)
        predicho = modelo.rollout(inicio_campo[trozo], pasos)
        objetivo = real[trozo]

        error = (predicho - objetivo).flatten(2).norm(dim=2)
        escala = objetivo.flatten(2).norm(dim=2).clamp_min(1e-12)
        l2_por_trayectoria.extend((error / escala).mean(dim=1).cpu().tolist())

        corr = torch.stack(
            [correlacion_espacial(predicho[:, t], objetivo[:, t]) for t in range(pasos)], dim=1
        )
        tiempos.extend(tiempo_hasta_descorrelacion(corr, dt_save, umbral).cpu().tolist())
        correlaciones_medias.append(corr.cpu().numpy())
        espectros.append(error_espectral(predicho[:, -1], objetivo[:, -1], grid_spec, bandas))

    correlacion = np.concatenate(correlaciones_medias, axis=0)
    return {
        "l2_1paso": error_1paso,
        "l2_1paso_por_par": error_1paso_por_par,
        "l2_despliegue": float(np.mean(l2_por_trayectoria)),
        "l2_despliegue_por_trayectoria": l2_por_trayectoria,
        "tiempo_corr": float(np.mean(tiempos)),
        "tiempo_corr_por_trayectoria": tiempos,
        "correlacion_media": correlacion.mean(axis=0).tolist(),
        "espectro": {
            banda.nombre: float(np.mean([e[banda.nombre] for e in espectros])) for banda in bandas
        },
    }


def prueba_de_permutacion(
    a: np.ndarray, b: np.ndarray, n_permutaciones: int, semilla: int
) -> dict[str, float]:
    """Permutacion pareada por trayectoria: se sortea el signo de cada diferencia.

    Pareada porque las dos columnas son el MISMO conjunto de test evaluado por dos
    modelos: la variabilidad entre trayectorias es enorme comparada con la diferencia
    entre modelos, y emparejar la elimina.
    """
    diferencia = a - b
    observado = float(diferencia.mean())
    generador = np.random.default_rng(semilla)
    signos = generador.choice([-1.0, 1.0], size=(n_permutaciones, diferencia.size))
    nulo = (signos * diferencia[None, :]).mean(axis=1)
    p = float((np.abs(nulo) >= abs(observado)).mean())
    return {
        "diferencia_media": observado,
        "p": p,
        "n_permutaciones": n_permutaciones,
    }


def agrega(valores: list[float]) -> dict[str, float]:
    array = np.asarray(valores, dtype=float)
    return {
        "media": float(array.mean()),
        "desviacion": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "n_semillas": int(array.size),
    }


def figura_metricas(resumen: dict[str, Any], bandas: list[Banda], destino: Path) -> None:
    """Puntos con barra de error, no barras desde cero.

    Con barras, un 0,0994 y un 0,1003 se dibujan practicamente iguales, y esa diferencia
    del 0,8 % es justo el resultado del que se discute. Un punto codifica la posicion, asi
    que el eje no tiene que empezar en cero para ser honesto, y ahi si se ve. La linea de
    la persistencia va en la leyenda y no dentro del panel: pegada al eje se solapaba con
    las etiquetas de los valores.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metricas = [
        ("l2_1paso", "Error L2 relativo a un paso"),
        ("l2_despliegue", "Error L2 en despliegue"),
        ("tiempo_corr", "Tiempo hasta correlacion < 0,8"),
    ]
    presentes = [m for m in ORDEN if m in resumen]
    figura, ejes = plt.subplots(1, len(metricas), figsize=(4.2 * len(metricas), 4.6))
    figura.patch.set_facecolor(SUPERFICIE)

    for eje, (clave, titulo) in zip(ejes, metricas):
        eje.set_facecolor(SUPERFICIE)
        alturas = [resumen[m][clave]["media"] for m in presentes]
        errores = [resumen[m][clave]["desviacion"] for m in presentes]
        posiciones = np.arange(len(presentes), dtype=float)
        referencia = resumen["persistencia"][clave]["media"] if "persistencia" in resumen else None

        if referencia is not None:
            eje.axhline(referencia, color=GRIS_REFERENCIA, linestyle="--", linewidth=1.2, zorder=2)
        for x, altura, error, modelo in zip(posiciones, alturas, errores, presentes):
            if referencia is not None:
                # El tallo mide la distancia a la persistencia, que es la comparacion que
                # importa: hacia arriba es peor, hacia abajo es mejor.
                eje.plot([x, x], [referencia, altura], color=COLORES[modelo],
                         linewidth=1.4, alpha=0.55, zorder=3)
            eje.errorbar(x, altura, yerr=error, fmt="o", markersize=7, capsize=3,
                         color=COLORES[modelo], ecolor=COLORES[modelo], zorder=4)

        extremos = [a + e for a, e in zip(alturas, errores)] + [a - e for a, e in zip(alturas, errores)]
        if referencia is not None:
            extremos.append(referencia)
        bajo, alto = min(extremos), max(extremos)
        # Decimales segun lo que hay que distinguir, no segun la magnitud: con la regla
        # anterior los cuatro cableados salian "0.0999" y la diferencia era invisible
        # tambien en el texto, no solo en el dibujo.
        rango = alto - bajo
        decimales = 2 if rango <= 0 else int(np.clip(math.ceil(-math.log10(rango)) + 1, 2, 6))
        margen = (alto - bajo) * 0.18 or max(abs(alto), 1.0) * 0.02
        eje.set_ylim(bajo - margen, alto + margen)
        # Etiqueta directa en cada punto: tres de los colores quedan por debajo de 3:1 de
        # contraste con el fondo, asi que el color no puede ser el unico canal.
        for x, altura, error in zip(posiciones, alturas, errores):
            arriba = referencia is None or altura >= referencia
            eje.annotate(
                f"{altura:.{decimales}f}", (x, altura + (error if arriba else -error)),
                textcoords="offset points", xytext=(0, 7 if arriba else -14),
                ha="center", fontsize=8.5, color=TINTA_PRIMARIA, zorder=5,
            )
        eje.set_xlim(-0.6, len(presentes) - 0.4)
        eje.set_title(titulo, fontsize=10.5, color=TINTA_PRIMARIA)
        eje.set_xticks(posiciones)
        eje.set_xticklabels(presentes, fontsize=9, color=TINTA_SECUNDARIA, rotation=20, ha="right")
        eje.tick_params(axis="y", labelsize=8.5, colors=TINTA_SECUNDARIA, length=0)
        eje.grid(axis="y", color="#e4e3df", linewidth=0.8, zorder=0)
        for lado in ("top", "right", "left"):
            eje.spines[lado].set_visible(False)
        eje.spines["bottom"].set_color("#d8d7d2")

    figura.suptitle(
        "Media +- desviacion tipica entre semillas de entrenamiento",
        fontsize=11, color=TINTA_PRIMARIA, y=0.985,
    )
    if "persistencia" in resumen:
        figura.text(
            0.5, 0.935, "- - -  persistencia (no cambia nada): el eje NO empieza en cero",
            ha="center", fontsize=9, color=TINTA_SECUNDARIA,
        )
    figura.tight_layout(rect=(0, 0, 1, 0.92))
    figura.savefig(destino, dpi=160, facecolor=SUPERFICIE)
    plt.close(figura)


def figura_correlacion(
    resumen: dict[str, Any], dt_save: float, umbral: float, destino: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figura, eje = plt.subplots(figsize=(7.2, 4.6))
    figura.patch.set_facecolor(SUPERFICIE)
    eje.set_facecolor(SUPERFICIE)

    for modelo in ORDEN:
        if modelo not in resumen:
            continue
        curvas = np.asarray(resumen[modelo]["correlacion_curvas"])
        tiempo = np.arange(1, curvas.shape[1] + 1) * dt_save
        media = curvas.mean(axis=0)
        eje.plot(tiempo, media, color=COLORES[modelo], linewidth=2.0, label=modelo, zorder=3)
        if curvas.shape[0] > 1:
            desviacion = curvas.std(axis=0, ddof=1)
            eje.fill_between(
                tiempo, media - desviacion, media + desviacion,
                color=COLORES[modelo], alpha=0.16, linewidth=0, zorder=2,
            )
    if "persistencia" in resumen:
        curvas = np.asarray(resumen["persistencia"]["correlacion_curvas"])
        tiempo = np.arange(1, curvas.shape[1] + 1) * dt_save
        eje.plot(tiempo, curvas.mean(axis=0), color=GRIS_REFERENCIA, linewidth=1.6,
                 linestyle="--", label="persistencia", zorder=3)

    eje.axhline(umbral, color=GRIS_REFERENCIA, linewidth=1.0, linestyle=":", zorder=1)
    eje.text(tiempo[-1], umbral, f" umbral {umbral}", va="bottom", ha="right",
             fontsize=8.5, color=TINTA_SECUNDARIA)
    eje.set_xlabel("tiempo del despliegue", fontsize=9.5, color=TINTA_SECUNDARIA)
    eje.set_ylabel("correlacion espacial con el campo real", fontsize=9.5, color=TINTA_SECUNDARIA)
    eje.set_title("Decaimiento de la correlacion durante el despliegue", fontsize=11,
                  color=TINTA_PRIMARIA)
    eje.tick_params(labelsize=8.5, colors=TINTA_SECUNDARIA, length=0)
    eje.grid(color="#e4e3df", linewidth=0.8, zorder=0)
    for lado in ("top", "right"):
        eje.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        eje.spines[lado].set_color("#d8d7d2")
    eje.legend(frameon=False, fontsize=9, labelcolor=TINTA_SECUNDARIA, ncols=2, loc="lower left")
    figura.tight_layout()
    figura.savefig(destino, dpi=160, facecolor=SUPERFICIE)
    plt.close(figura)


def figura_espectro(
    resumen: dict[str, Any], bandas: list[Banda], destino: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    presentes = [m for m in ORDEN if m in resumen]
    figura, eje = plt.subplots(figsize=(7.2, 4.6))
    figura.patch.set_facecolor(SUPERFICIE)
    eje.set_facecolor(SUPERFICIE)

    ancho = 0.8 / max(len(presentes), 1)
    posiciones = np.arange(len(bandas))
    for i, modelo in enumerate(presentes):
        alturas = [resumen[modelo]["espectro"][b.nombre]["media"] for b in bandas]
        errores = [resumen[modelo]["espectro"][b.nombre]["desviacion"] for b in bandas]
        desplazamiento = (i - (len(presentes) - 1) / 2) * ancho
        barras = eje.bar(
            posiciones + desplazamiento, alturas, width=ancho * 0.88, yerr=errores,
            capsize=2, color=COLORES[modelo], label=modelo, zorder=3,
        )
        for barra, altura in zip(barras, alturas):
            eje.text(barra.get_x() + barra.get_width() / 2, altura, f"{altura:.3f}",
                     ha="center", va="bottom", fontsize=7.5, color=TINTA_PRIMARIA, zorder=4)

    if "persistencia" in resumen:
        # Sin esta linea el panel se lee como "unos cableados distorsionan mas que otros",
        # cuando lo que dice es que TODOS distorsionan un orden de magnitud mas que no
        # hacer nada. Un tramo por banda, no una horizontal: el valor cambia con la banda.
        for x, banda in zip(posiciones, bandas):
            altura = resumen["persistencia"]["espectro"][banda.nombre]["media"]
            eje.plot([x - 0.44, x + 0.44], [altura, altura], color=GRIS_REFERENCIA,
                     linestyle="--", linewidth=1.2, zorder=5,
                     label="persistencia" if x == 0 else None)

    eje.set_xticks(posiciones)
    eje.set_xticklabels([b.etiqueta for b in bandas], fontsize=9, color=TINTA_SECUNDARIA)
    eje.set_ylabel("|log10(E predicho / E real)|", fontsize=9.5, color=TINTA_SECUNDARIA)
    eje.set_title("Error del espectro de energia por bandas, al final del despliegue",
                  fontsize=11, color=TINTA_PRIMARIA)
    eje.tick_params(axis="y", labelsize=8.5, colors=TINTA_SECUNDARIA, length=0)
    eje.grid(axis="y", color="#e4e3df", linewidth=0.8, zorder=0)
    for lado in ("top", "right", "left"):
        eje.spines[lado].set_visible(False)
    eje.spines["bottom"].set_color("#d8d7d2")
    eje.legend(frameon=False, fontsize=9, labelcolor=TINTA_SECUNDARIA, ncols=3)
    figura.tight_layout()
    figura.savefig(destino, dpi=160, facecolor=SUPERFICIE)
    plt.close(figura)


def tabla_markdown(resumen: dict[str, Any], bandas: list[Banda]) -> str:
    cabecera = ["modelo", "L2 1 paso", "L2 despliegue", "t hasta corr<0,8"]
    cabecera += [b.nombre for b in bandas]
    lineas = ["| " + " | ".join(cabecera) + " |",
              "|" + "|".join(["---"] * len(cabecera)) + "|"]
    for modelo in ORDEN + ["persistencia"]:
        if modelo not in resumen:
            continue
        fila = [modelo]
        for clave in ("l2_1paso", "l2_despliegue", "tiempo_corr"):
            dato = resumen[modelo][clave]
            fila.append(f"{dato['media']:.4f} ± {dato['desviacion']:.4f}")
        for banda in bandas:
            dato = resumen[modelo]["espectro"][banda.nombre]
            fila.append(f"{dato['media']:.3f} ± {dato['desviacion']:.3f}")
        lineas.append("| " + " | ".join(fila) + " |")
    return "\n".join(lineas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    evaluacion = cfg.get("evaluation", {})
    pasos = int(evaluacion.get("rollout_steps", 50))
    umbral = float(evaluacion.get("correlation_threshold", 0.8))
    n_permutaciones = int(evaluacion.get("permutations", 10000))

    device = resolve_device(args.device or cfg.get("device", "auto"))
    salida = Path(cfg["output_dir"])
    figuras = salida / "figs"
    figuras.mkdir(parents=True, exist_ok=True)

    test = Trajectories.load(Path(cfg["data_dir"]) / f"{args.split}.npz", device)
    grid = test.grid
    meta = json.loads((Path(cfg["data_dir"]) / "meta.json").read_text())
    dt_save = float(meta["dt_save"])
    grid_spec = make_grid(grid, device=device)

    k_encoder = k_maximo_del_codificador(grid, int(cfg["encoder"]["n_complex_modes"]))
    bandas = bandas_por_defecto(grid, k_encoder)
    print(f"dispositivo {device}  malla {grid}  dt_guardado {dt_save}  "
          f"despliegue {pasos} pasos  el codificador llega a k={k_encoder:.1f}")

    checkpoints = salida / "checkpoints"
    crudos: dict[str, list[dict[str, Any]]] = {}
    for modelo in list(cfg["models"]):
        semillas = [None] if modelo == "persistencia" else list(cfg["seeds"])
        for semilla in semillas:
            red = build_model(modelo, cfg, grid, int(semilla or 0)).to(device)
            if semilla is not None:
                ruta = checkpoints / f"{modelo}_s{semilla}.pt"
                if not ruta.is_file():
                    print(f"  falta {ruta.name}: {modelo} semilla {semilla} sin entrenar, se salta")
                    continue
                red.load_state_dict(torch.load(ruta, map_location=device)["state_dict"])
            metricas = evalua_modelo(red, test, pasos, umbral, dt_save, grid_spec, bandas,
                                     int(cfg["phase_a"]["batch_size"]))
            crudos.setdefault(modelo, []).append(metricas)
            print(f"  {modelo:13s} semilla {semilla}  L2 1 paso {metricas['l2_1paso']:.4f}"
                  f"  despliegue {metricas['l2_despliegue']:.4f}"
                  f"  t_corr {metricas['tiempo_corr']:.2f}")

    resumen: dict[str, Any] = {}
    for modelo, corridas in crudos.items():
        resumen[modelo] = {
            clave: agrega([c[clave] for c in corridas])
            for clave in ("l2_1paso", "l2_despliegue", "tiempo_corr")
        }
        resumen[modelo]["espectro"] = {
            banda.nombre: agrega([c["espectro"][banda.nombre] for c in corridas])
            for banda in bandas
        }
        resumen[modelo]["correlacion_curvas"] = [c["correlacion_media"] for c in corridas]

    # Pruebas de permutacion: mosca contra cada control, pareadas.
    pruebas: dict[str, Any] = {}
    if "mosca" in crudos:
        for clave in ("l2_1paso_por_par", "l2_despliegue_por_trayectoria",
                      "tiempo_corr_por_trayectoria"):
            mosca = np.mean([c[clave] for c in crudos["mosca"]], axis=0)
            # `persistencia` va la primera a proposito: es la puerta. Si el nucleo no le
            # gana a no hacer nada, la comparacion entre cableados no significa nada, y sin
            # el contraste escrito al lado, empatar con el aleatorio se lee como un
            # resultado cuando no lo es (la misma leccion que evaluate_rl.py con el azar).
            for control in ("persistencia", "sin_nucleo", "aleatorio", "recableado",
                            "denso", "fno"):
                if control not in crudos:
                    continue
                otro = np.mean([c[clave] for c in crudos[control]], axis=0)
                prueba = prueba_de_permutacion(mosca, otro, n_permutaciones, semilla=12345)
                base = clave.replace("_por_trayectoria", "").replace("_por_par", "")
                dispersion = max(
                    resumen["mosca"][base]["desviacion"], resumen[control][base]["desviacion"]
                )
                prueba["desviacion_entre_semillas"] = dispersion
                # Con una sola semilla la desviacion es 0 por construccion y el veredicto
                # no significa nada: se marca como no juzgable en vez de mentir.
                n_semillas = min(
                    resumen["mosca"][base]["n_semillas"], resumen[control][base]["n_semillas"]
                )
                prueba["n_semillas"] = n_semillas
                prueba["dentro_de_la_variabilidad_entre_semillas"] = (
                    None if n_semillas < 2 else bool(abs(prueba["diferencia_media"]) <= dispersion)
                )
                pruebas[f"{base}: mosca vs {control}"] = prueba

    figura_metricas(resumen, bandas, figuras / "metricas.png")
    figura_correlacion(resumen, dt_save, umbral, figuras / "correlacion.png")
    figura_espectro(resumen, bandas, figuras / "espectro_por_bandas.png")

    tabla = tabla_markdown(resumen, bandas)
    informe = {
        "config": str(args.config),
        "split": args.split,
        "pasos_despliegue": pasos,
        "umbral_correlacion": umbral,
        "k_maximo_del_codificador": k_encoder,
        "bandas": [{"nombre": b.nombre, "k_min": b.k_min, "k_max": b.k_max} for b in bandas],
        "resumen": resumen,
        "pruebas_de_permutacion": pruebas,
        "tabla_markdown": tabla,
    }
    (salida / "eval_report.json").write_text(json.dumps(informe, indent=2, ensure_ascii=False))

    print("\n" + tabla)
    if pruebas:
        print("\nPruebas de permutacion (pareadas: por par a un paso, por trayectoria en el resto):")
        for nombre, prueba in pruebas.items():
            dentro = prueba["dentro_de_la_variabilidad_entre_semillas"]
            veredicto = (
                f"no juzgable: {prueba['n_semillas']} semilla(s)" if dentro is None
                else "DENTRO de la variabilidad entre semillas" if dentro
                else "por encima de la variabilidad entre semillas"
            )
            print(f"  {nombre}: diferencia {prueba['diferencia_media']:+.4f}  "
                  f"p = {prueba['p']:.4f}  ({veredicto})")
    print(f"\ninforme en {salida / 'eval_report.json'}   figuras en {figuras}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
