"""Evaluacion de la fase 6: los modelos entrenados contra tres lineas de referencia.

Uso:
    python evaluate_rl.py --config configs/mario.yaml

Cada modelo y cada linea de referencia juega **exactamente los mismos niveles** (las
semillas de test, que son disjuntas de las de entrenamiento y validacion). Eso permite
emparejar por nivel y usar la misma prueba de permutacion pareada de la fase 5: la
varianza entre niveles es enorme comparada con la diferencia entre modelos.

Las tres referencias, sin las cuales el numero de un modelo no se puede leer:

  azar          acciones uniformes. El suelo.
  siempre_dcha  mantener "derecha + carrera" y nada mas: se para en el primer hueco.
  salto_cte     "derecha + salto" sin soltar. **Esta es la que importa.** Es la politica
                degenerada a la que PPO cae solo, y en niveles de 160 casillas ya saca
                0,38 de progreso y llega a la meta el 8 % de las veces. Un modelo que no
                le gane no ha aprendido a jugar, ha aprendido a saltar sin mirar.
  salto_corriendo  "derecha + salto + carrera" sin soltar (solo en el entorno propio; en
                el NES `salto_cte` ya lleva B). Anadida el 19-10-2026: es la constante
                mas fuerte del entorno propio (niveles de 60: 0,809 y 67 % de metas,
                contra 0,802 y 61,5 % de `salto_cte`), y la politica a la que llegaron
                la mosca y el MLP del piloto, que la pulsaban en el 93 % de los pasos.
  heuristica    correr y saltar lo que hay delante, escrita a mano. Referencia superior,
                como el FNO en fluidos.
"""

from __future__ import annotations

import argparse
import itertools
import json
from math import comb
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from torch import nn

from data.spectral import resolve_device
from envs.platformer import (
    N_ACTIONS,
    LevelConfig,
    ObservationConfig,
    PlatformerEnv,
    RewardConfig,
    SeedRanges,
    scripted_action,
)
from evaluate import prueba_de_permutacion
from models.masked_policy import build_policy
from models.masked_rnn import Wiring
from train_rl import load_wirings

METRICAS = ("progreso", "meta", "recompensa")

# Una partida = (particion, semilla). La particion hace falta porque en el NES la semilla 0
# de val y la semilla 0 de test son niveles distintos.
Partida = tuple[str, int]


def es_nes(cfg: dict[str, Any]) -> bool:
    return cfg.get("env", "propio") == "nes"


# Niveles que no estan en ninguna particion de `envs.nes_mario.STAGES` y que por tanto la
# red no ha visto nunca. Solo se evaluan (14-10-2026, antes de ver las replicas de la 6c).
# Motivo: de los 7 niveles de val+test, 6 son atleticos (x-3: copas de arbol, setas,
# puentes), un tipo que no hay en entrenamiento, y solo el 8-1 es exterior como los de
# train. Con estos, "nivel nuevo de un tipo conocido" (exteriores: 8-1 + 3-2, 5-2, 6-2,
# 8-2, 8-3) y "tipo nuevo" (atleticos: los seis x-3 + 7-3) tienen 6 y 7 niveles cada uno.
# Se quedan fuera los castillos (x-4), el agua (2-2, 7-2) y el subterraneo 4-2.
NIVELES_EXTRA: tuple[tuple[int, int], ...] = ((3, 2), (5, 2), (6, 2), (7, 3), (8, 2), (8, 3))
TIPO_DE_NIVEL: dict[str, str] = {
    **{f"{m}-3": "atletico" for m in range(1, 8)},
    **{n: "exterior" for n in ("1-1", "2-1", "3-1", "3-2", "4-1", "5-1", "5-2", "6-1",
                                "6-2", "7-1", "8-1", "8-2", "8-3")},
}


def niveles_de_particion(split: str) -> tuple[tuple[int, int], ...]:
    """Los niveles de una particion del NES: las de `STAGES` mas "extra"."""
    from envs.nes_mario import STAGES

    return NIVELES_EXTRA if split == "extra" else STAGES[split]


def nivel_previsto(split: str, semilla: int) -> str:
    """El nivel que `NesMarioEnv.reset(semilla)` va a elegir, sin arrancar el emulador.

    Repite el primer sorteo de `reset`. Si alguien lo cambia alli, `main` lo detecta: compara
    esta prevision con el nivel que de verdad se jugo en cada partida.
    """
    niveles = niveles_de_particion(split)
    mundo, nivel = niveles[int(np.random.default_rng(semilla).integers(len(niveles)))]
    return f"{mundo}-{nivel}"


def partidas_equilibradas(splits: list[str], por_nivel: int) -> list[Partida]:
    """Las primeras `por_nivel` semillas que caen en cada nivel de cada particion.

    Con las semillas 0..15 de test, que es lo que hace `niveles_de_test`, los cuatro niveles
    salen 7/4/3/2 veces y el 8-1 pesa el 44 % de la media. Aqui pesan todos igual, y cada
    nivel se juega con `por_nivel` esperas iniciales distintas.
    """
    partidas: list[Partida] = []
    for split in splits:
        cuenta = {f"{m}-{n}": 0 for m, n in niveles_de_particion(split)}
        semilla = 0
        while min(cuenta.values()) < por_nivel:
            nivel = nivel_previsto(split, semilla)
            if cuenta[nivel] < por_nivel:
                partidas.append((split, semilla))
                cuenta[nivel] += 1
            semilla += 1
    return partidas


def media_por_nivel(valores: np.ndarray, niveles: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Media de cada nivel, en orden alfabetico de nivel."""
    etiquetas = sorted(set(niveles.tolist()))
    return etiquetas, np.array([valores[niveles == e].mean() for e in etiquetas])


def permutacion_exacta_pareada(diferencias: np.ndarray) -> dict[str, float]:
    """Todas las asignaciones de signo (2^n), sin sorteo. Para pocas unidades: niveles.

    Las 16 partidas de test del NES son cuatro niveles con esperas distintas; dentro de un
    nivel se parecen mucho, asi que tratarlas como 16 unidades independientes infla la p.
    Aqui la unidad es el nivel. Con n niveles la p bilateral no puede bajar de 2 / 2^n
    (0,125 con 4, 0,016 con 7): se informa para que una p alta no se lea como "empate".
    """
    n = int(diferencias.size)
    if n > 20:
        raise ValueError(f"{n} unidades: 2^{n} permutaciones no caben, usa la de Monte Carlo")
    observado = float(diferencias.mean())
    signos = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    nulo = (signos * diferencias[None, :]).mean(axis=1)
    p = float((np.abs(nulo) >= abs(observado) - 1e-12).mean())
    return {"diferencia_media": observado, "p": p, "n": n, "p_minima_posible": 2.0 / 2**n}


def permutacion_exacta_entre_grupos(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Diferencia de medias entre dos grupos independientes, con todas las particiones.

    Es la prueba por semilla de entrenamiento: la unidad es el modelo entrenado entero.
    Responde a si la diferencia entre arquitecturas supera la que hay entre dos semillas de
    la misma. Con 3 contra 3 la p bilateral no baja de 0,1, y eso tambien se informa.
    """
    todos = np.concatenate([a, b])
    observado = float(a.mean() - b.mean())
    nulo = []
    for indices in itertools.combinations(range(todos.size), a.size):
        mascara = np.zeros(todos.size, dtype=bool)
        mascara[list(indices)] = True
        nulo.append(todos[mascara].mean() - todos[~mascara].mean())
    p = float((np.abs(np.asarray(nulo)) >= abs(observado) - 1e-12).mean())
    particiones = comb(int(todos.size), int(a.size))
    minima = (2.0 if a.size == b.size else 1.0) / particiones
    return {"diferencia_media": observado, "p": p, "n_a": int(a.size), "n_b": int(b.size),
            "p_minima_posible": minima}


def niveles_de_test(cfg: dict[str, Any], n_levels: int) -> list[int]:
    """Semillas de test. En el entorno propio son niveles; en el NES eligen nivel y
    espera inicial dentro del conjunto de prueba, que es disjunto del de entrenamiento."""
    if es_nes(cfg):
        return list(range(n_levels))
    low, high = SeedRanges.from_config(cfg).range("test")
    return [low + i % (high - low) for i in range(n_levels)]


def construye_entorno(cfg: dict[str, Any], split: str = "test"):
    if es_nes(cfg):
        from envs.nes_mario import NesConfig, NesMarioEnv

        nes = dict(cfg.get("nes", {}))
        nivel = nes.pop("stage", None)
        nes["frontier_probability"] = 0.0   # se mide siempre desde la salida del nivel
        # "extra" no existe en STAGES: es un entorno de test con otros niveles. Asi el
        # codigo del entorno, que es el del entrenamiento, no se toca.
        env = NesMarioEnv("train" if nivel else ("test" if split == "extra" else split),
                          ObservationConfig(**cfg.get("observation", {})), NesConfig(**nes))
        if nivel is not None:
            # La demo fija un nivel: entrena y mide en el mismo, y lo dice su YAML.
            env.stages = (tuple(nivel),)
        elif split == "extra":
            env.stages = NIVELES_EXTRA
        return env
    return PlatformerEnv(
        LevelConfig(**cfg.get("level", {})),
        ObservationConfig(**cfg.get("observation", {})),
        RewardConfig(**cfg.get("reward", {})),
        max_steps=int(cfg.get("max_episode_steps", 1200)),
        release_to_jump=bool(cfg.get("release_to_jump", False)),
    )


def juega(
    cfg: dict[str, Any], partidas: list[Partida], accion: Callable[[Any, np.ndarray], int],
    reinicia: Callable[[], None] | None = None,
) -> dict[str, np.ndarray]:
    """Un episodio por partida. Devuelve una serie por metrica, alineada con `partidas`,
    y el nivel que se jugo en cada una (`nivel`; en el entorno propio, la semilla)."""
    entornos: dict[str, Any] = {}
    resultados = {metrica: [] for metrica in METRICAS}
    niveles: list[str] = []
    for split, semilla in partidas:
        if split not in entornos:
            entornos[split] = construye_entorno(cfg, split)
        env = entornos[split]
        observacion = env.reset(semilla)
        if reinicia is not None:
            reinicia()
        total = 0.0
        done = False
        while not done:
            observacion, recompensa, done, info = env.step(accion(env, observacion))
            total += recompensa
        resultados["progreso"].append(info["progreso"])
        resultados["meta"].append(float(info["meta"]))
        resultados["recompensa"].append(total)
        niveles.append(str(info.get("nivel", semilla)))
    for env in entornos.values():
        if hasattr(env, "close"):
            env.close()
    series = {k: np.asarray(v, dtype=float) for k, v in resultados.items()}
    series["nivel"] = np.asarray(niveles)
    return series


def referencias(cfg: dict[str, Any], semillas: list[Partida], semilla_azar: int) -> dict[str, dict[str, np.ndarray]]:
    """Las lineas contra las que se lee cualquier modelo.

    Los indices de accion no son los mismos en los dos entornos: en el propio la 2 es
    "derecha + carrera" y la 3 "derecha + salto"; en el NES, SIMPLE_MOVEMENT pone la 1 en
    "derecha", la 2 en "derecha + A" y la 4 en "derecha + A + B". La heuristica escrita a
    mano solo existe en el entorno propio, porque lee su mapa de casillas directamente.
    """
    generador = np.random.default_rng(semilla_azar)
    derecha, salto = (1, 4) if es_nes(cfg) else (2, 3)
    lineas = {
        "azar": juega(cfg, semillas, lambda env, obs: int(generador.integers(N_ACTIONS))),
        "siempre_dcha": juega(cfg, semillas, lambda env, obs: derecha),
        "salto_cte": juega(cfg, semillas, lambda env, obs: salto),
    }
    if not es_nes(cfg):
        lineas["salto_corriendo"] = juega(cfg, semillas, lambda env, obs: 4)
        lineas["heuristica"] = juega(cfg, semillas, lambda env, obs: scripted_action(env))
    return lineas


def juega_politica(
    cfg: dict[str, Any], semillas: list[Partida], politica: nn.Module, device: torch.device,
    muestrea: bool = False, contador: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Estado oculto reiniciado en cada nivel. `muestrea` sortea en vez de coger el maximo.

    Las dos se informan porque no dicen lo mismo: la codiciosa se queda pegada a una
    accion constante y la muestreada conserva la exploracion que la red aprendio.
    Si se pasa `contador`, suma en el cuantas veces elige cada accion.
    """
    estado = {"h": politica.initial_state(1, device)}

    def reinicia() -> None:
        estado["h"] = politica.initial_state(1, device)

    @torch.no_grad()
    def accion(env: Any, observacion: np.ndarray) -> int:
        tensor = torch.as_tensor(observacion, dtype=torch.float32, device=device)[None]
        logits, _, estado["h"] = politica(tensor, estado["h"])
        if muestrea:
            elegida = int(torch.distributions.Categorical(logits=logits).sample().item())
        else:
            elegida = int(logits.argmax(dim=1).item())
        if contador is not None:
            contador[elegida] += 1
        return elegida

    return juega(cfg, semillas, accion, reinicia)


def carga_politica(
    ruta: Path, cfg: dict[str, Any], wirings: dict[str, Wiring], device: torch.device
) -> nn.Module:
    checkpoint = torch.load(ruta, map_location=device, weights_only=False)
    entorno = construye_entorno(cfg)
    politica = build_policy(
        checkpoint["modelo"], cfg, entorno.n_features, N_ACTIONS, int(checkpoint["semilla"]), wirings
    )
    faltan, sobran = politica.load_state_dict(checkpoint["state_dict"], strict=False)
    # Los checkpoints anteriores a la normalizacion de la entrada no traen estos buffers;
    # se quedan en media 0 y desviacion 1, que es exactamente como se entrenaron.
    esperadas = {"feature_mean", "feature_std",
                 "normalizador.media", "normalizador.desviacion"}
    if set(faltan) - esperadas or sobran:
        raise RuntimeError(f"checkpoint incompatible: faltan {faltan}, sobran {sobran}")
    if {"feature_mean", "feature_std"} & set(faltan):
        print(f"  aviso: {ruta.name} es anterior a la normalizacion de entrada")
    return politica.to(device).eval()


def evalua_checkpoint(trabajo: dict[str, Any]) -> dict[str, Any]:
    """Un checkpoint en todas las partidas, `repeticiones` veces, y su politica marginal.

    De nivel de modulo para poder repartir los checkpoints entre procesos (`--procesos`).
    Cada repeticion refija la semilla con el mismo valor para todos los modelos, asi que
    el resultado no depende de en que proceso caiga ni en que orden.

    La politica **marginal** juega con las mismas frecuencias de accion que el modelo,
    sorteadas sin mirar la pantalla. Es el control de "¿reacciona a lo que ve?": ganarle al
    azar uniforme no lo demuestra, porque una buena proporcion de botones (mucho derecha y
    salto, pero no siempre) ya recorre mas que el azar y que cualquier accion constante.
    """
    if trabajo["en_proceso"]:
        torch.set_num_threads(1)
    cfg = trabajo["cfg"]
    device = torch.device(trabajo["device"])
    politica = carga_politica(trabajo["ruta"], cfg, load_wirings(cfg), device)
    repeticiones = max(1, int(trabajo["repeticiones"]))
    contador = np.zeros(N_ACTIONS)
    partidas = []
    for repeticion in range(repeticiones):
        torch.manual_seed(trabajo["semilla"] * 1000 + repeticion)
        np.random.seed(trabajo["semilla"] * 1000 + repeticion)
        partidas.append(juega_politica(
            cfg, trabajo["semillas"], politica, device, trabajo["muestrea"], contador))
    frecuencias = contador / contador.sum()
    resultado: dict[str, Any] = {
        "niveles": [p["nivel"].tolist() for p in partidas],
        "series": {m: np.mean([p[m] for p in partidas], axis=0) for m in METRICAS},
        "frecuencias": frecuencias.tolist(),
    }
    if trabajo["marginal"]:
        marginales = []
        for repeticion in range(repeticiones):
            generador = np.random.default_rng(trabajo["semilla"] * 1000 + repeticion)
            marginales.append(juega(
                cfg, trabajo["semillas"],
                lambda env, obs: int(generador.choice(N_ACTIONS, p=frecuencias))))
        resultado["marginal"] = {m: np.mean([p[m] for p in marginales], axis=0) for m in METRICAS}
    return resultado


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--levels", type=int, default=None, help="niveles de test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--muestrea", action="store_true",
                        help="politica muestreada en vez de codiciosa")
    parser.add_argument("--repeticiones", type=int, default=1,
                        help="partidas por nivel; con politica muestreada hace falta mas de una")
    parser.add_argument("--checkpoint", type=Path, default=None, action="append",
                        help="evaluar estos checkpoints sueltos en vez de los del "
                             "output_dir; se puede repetir para comparar versiones")
    parser.add_argument("--semilla", type=int, default=0,
                        help="semilla del muestreo, para que la evaluacion sea reproducible")
    parser.add_argument("--split", default="test",
                        help="niveles a jugar (NES), unidos con +: test, val, extra (p. ej. "
                             "val+test). val es tan limpio como test si el checkpoint es el "
                             "ultimo y no el mejor en val; extra no se usa en ninguna particion")
    parser.add_argument("--por-nivel", type=int, default=None,
                        help="NES: esta cantidad de partidas en cada nivel, en vez de "
                             "las semillas 0..n-1 (que reparten los niveles a desigual)")
    parser.add_argument("--salida", type=Path, default=None,
                        help="nombre del informe dentro de output_dir")
    parser.add_argument("--modelos", nargs="+", default=None,
                        help="en vez de `models` del YAML (que es lo que se entrena con el)")
    parser.add_argument("--semillas", nargs="+", type=int, default=None,
                        help="en vez de `seeds` del YAML")
    parser.add_argument("--marginal", action="store_true",
                        help="ademas, cada modelo contra su politica marginal (mismas "
                             "frecuencias de accion, sin mirar la pantalla)")
    parser.add_argument("--procesos", type=int, default=1,
                        help="checkpoints evaluados a la vez, uno por proceso")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.modelos:
        cfg["models"] = list(args.modelos)
    if args.semillas:
        cfg["seeds"] = list(args.semillas)
    if not es_nes(cfg) and (args.split != "test" or args.por_nivel):
        parser.error("--split y --por-nivel solo existen en el NES")
    if not set(args.split.split("+")) <= {"test", "val", "extra"}:
        parser.error(f"--split {args.split}: las particiones son test, val y extra")
    if args.por_nivel and cfg.get("nes", {}).get("stage") is not None:
        parser.error("--por-nivel no tiene sentido con un nivel fijo (demo)")
    device = resolve_device(args.device or cfg.get("device", "auto"))
    # Regla 2 del repo: toda ejecucion registra su semilla. Sin fijarla, dos evaluaciones
    # muestreadas del MISMO checkpoint daban 0,818 y 0,701 de progreso.
    torch.manual_seed(args.semilla)
    np.random.seed(args.semilla)
    if args.muestrea and args.repeticiones == 1:
        print("aviso: politica muestreada con una sola partida por nivel; el ruido del"
              " muestreo es del tamano de las diferencias entre modelos")
    out_dir = Path(cfg["output_dir"])
    splits = args.split.split("+")
    if args.por_nivel:
        semillas = partidas_equilibradas(splits, args.por_nivel)
    else:
        n_levels = args.levels or int(cfg.get("test_episodes", 256))
        semillas = [(split, s) for split in splits for s in niveles_de_test(cfg, n_levels)]
    n_levels = len(semillas)
    print(f"dispositivo: {device}   partidas: {n_levels} ({args.split}"
          f"{f', {args.por_nivel} por nivel' if args.por_nivel else ''})"
          f"   politica {'muestreada' if args.muestrea else 'codiciosa'}")

    series: dict[str, dict[str, np.ndarray]] = referencias(cfg, semillas, semilla_azar=0)
    niveles = series["azar"]["nivel"]
    if args.por_nivel:
        previstos = [nivel_previsto(split, s) for split, s in semillas]
        if niveles.tolist() != previstos:
            raise RuntimeError("nivel_previsto ya no coincide con NesMarioEnv.reset")
    series_por_semilla: dict[str, list[dict[str, Any]]] = {}
    # Con --checkpoint se comparan versiones de un mismo modelo (por ejemplo el mismo
    # entrenamiento a 300 y a 900 iteraciones) contra las MISMAS lineas de referencia y
    # con la misma semilla de muestreo. Cada checkpoint entra como una fila mas, con el
    # nombre del fichero, asi que la tabla y los contrastes salen igual.
    if args.checkpoint:
        a_evaluar = [(ruta.stem, [ruta]) for ruta in args.checkpoint]
    else:
        a_evaluar = [
            (name, [out_dir / f"{name}_s{seed}.pt" for seed in cfg["seeds"]])
            for name in cfg["models"]
        ]
    # Varias partidas por nivel y media: la politica muestreada es estocastica y con una
    # sola partida el ruido tapa cualquier diferencia entre cableados.
    #
    # La semilla se REFIJA antes de cada repeticion (en `evalua_checkpoint`), y con el
    # mismo valor para todos los modelos: asi la repeticion 3 de la mosca y la repeticion 3
    # del aleatorio arrancan del mismo estado del generador. No hace que jueguen igual
    # —cada politica sortea de su propia distribucion— pero acopla el ruido del muestreo,
    # que es la variabilidad que ya nos dio la vuelta al orden entre modelos una vez.
    # Antes, cada modelo consumia un tramo distinto del generador segun el orden en que se
    # evaluaba. Por lo mismo da igual repartirlos entre procesos.
    trabajos = [(name, ruta) for name, rutas in a_evaluar for ruta in rutas if ruta.exists()]
    comun = {"cfg": cfg, "device": str(device), "semillas": semillas,
             "repeticiones": args.repeticiones, "semilla": args.semilla,
             "muestrea": args.muestrea, "marginal": args.marginal,
             "en_proceso": args.procesos > 1}
    peticiones = [{**comun, "ruta": ruta} for _, ruta in trabajos]
    if args.procesos > 1:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(args.procesos) as pool:
            resultados = pool.map(evalua_checkpoint, peticiones, chunksize=1)
    else:
        resultados = [evalua_checkpoint(peticion) for peticion in peticiones]

    frecuencias: dict[str, list[float]] = {}
    for name, _ in a_evaluar:
        hechos = [(ruta, r) for (n, ruta), r in zip(trabajos, resultados) if n == name]
        if not hechos:
            print(f"  {name}: sin checkpoints en {out_dir}")
            continue
        for ruta, r in hechos:
            if any(jugados != niveles.tolist() for jugados in r["niveles"]):
                raise RuntimeError(f"{ruta.name} no jugo los mismos niveles que las referencias")
            frecuencias[ruta.stem] = r["frecuencias"]
            print(f"  {ruta.stem}: progreso {r['series']['progreso'].mean():.3f}"
                  + (f"  (su marginal {r['marginal']['progreso'].mean():.3f})"
                     if "marginal" in r else ""))
        filas = [(name, "series")] + ([(f"marginal_{name}", "marginal")] if args.marginal else [])
        for fila, clave in filas:
            series_por_semilla[fila] = [
                {"checkpoint": ruta.stem, **{m: r[clave][m].tolist() for m in METRICAS}}
                for ruta, r in hechos
            ]
            # Media por nivel sobre las semillas de entrenamiento: la unidad del emparejado
            # sigue siendo el nivel, que es lo que comparten todos los modelos.
            series[fila] = {
                m: np.mean([r[clave][m] for _, r in hechos], axis=0) for m in METRICAS
            }

    resumen = {
        nombre: {
            metrica: {
                "media": float(valores[metrica].mean()),
                "desviacion": float(valores[metrica].std(ddof=1)),
            }
            for metrica in METRICAS
        }
        for nombre, valores in series.items()
    }

    permutaciones = int(cfg.get("permutations", 10000))
    contrastes: dict[str, Any] = {}
    # Contra "azar" tambien: un modelo puede ganarle a la politica degenerada y aun asi
    # no distinguirse de apretar botones al azar, que es lo que pasa en el NES a 190
    # iteraciones. Sin este contraste, el primer resultado se lee como si fuera bueno.
    parejas = [("mosca", "aleatorio"), ("mosca", "recableado"), ("mosca", "salto_cte"),
               ("mosca", "azar"), ("mosca", "sin_nucleo"), ("mosca", "mlp"),
               ("mosca", "denso")]
    if args.checkpoint:
        # Comparando versiones: cada una contra las dos lineas que deciden, y las demas
        # contra la primera, que es la referencia de "lo que habia antes".
        nombres = [ruta.stem for ruta in args.checkpoint]
        lineas = ("salto_cte", "azar") + (() if es_nes(cfg) else ("salto_corriendo",))
        parejas = [(n, linea) for n in nombres for linea in lineas]
        parejas += [(n, nombres[0]) for n in nombres[1:]]
    if not args.checkpoint:
        parejas += [("mlp", "azar"), ("mlp", "salto_cte")]
        if not es_nes(cfg):
            parejas += [("mosca", "salto_corriendo"), ("mlp", "salto_corriendo")]
    if args.marginal:
        parejas += [(name, f"marginal_{name}") for name, _ in a_evaluar]
    # Tres pruebas por pareja, de la mas optimista a la mas conservadora:
    #   contrastes            por partida (la de siempre). Con varias partidas del mismo
    #                         nivel trata como independientes cosas que no lo son.
    #   contrastes_por_nivel  la unidad es el nivel: exacta sobre las medias por nivel.
    #   contrastes_por_semilla la unidad es el entrenamiento: medias de cada semilla de un
    #                         modelo contra las del otro. Solo entre modelos con >= 2 semillas.
    # Una diferencia entre cableados solo se sostiene si aguanta las dos ultimas.
    contrastes_por_nivel: dict[str, Any] = {}
    contrastes_por_semilla: dict[str, Any] = {}
    # En el entorno propio cada semilla ya es un nivel distinto: alli la de partida es la
    # de nivel, y 2^256 asignaciones no se pueden enumerar.
    pocos_niveles = len(set(niveles.tolist())) <= 20
    for a, b in parejas:
        if a in series and b in series:
            contrastes[f"{a}_vs_{b}"] = {
                metrica: prueba_de_permutacion(
                    series[a][metrica], series[b][metrica], permutaciones, semilla=0
                )
                for metrica in ("progreso", "meta")
            }
            if pocos_niveles:
                contrastes_por_nivel[f"{a}_vs_{b}"] = {
                    metrica: permutacion_exacta_pareada(
                        media_por_nivel(series[a][metrica] - series[b][metrica], niveles)[1]
                    )
                    for metrica in ("progreso", "meta")
                }
            if len(series_por_semilla.get(a, [])) >= 2 and len(series_por_semilla.get(b, [])) >= 2:
                contrastes_por_semilla[f"{a}_vs_{b}"] = {
                    metrica: permutacion_exacta_entre_grupos(
                        np.array([np.mean(s[metrica]) for s in series_por_semilla[a]]),
                        np.array([np.mean(s[metrica]) for s in series_por_semilla[b]]),
                    )
                    for metrica in ("progreso", "meta")
                }

    etiquetas = media_por_nivel(series["azar"]["progreso"], niveles)[0] if pocos_niveles else []
    por_nivel = {
        nombre: {
            metrica: dict(zip(etiquetas, media_por_nivel(valores[metrica], niveles)[1].tolist()))
            for metrica in ("progreso", "meta")
        }
        for nombre, valores in series.items()
    }
    for nombre in resumen:
        # Media con el mismo peso para cada nivel. Con --por-nivel coincide con "media".
        for metrica in ("progreso", "meta"):
            if pocos_niveles:
                resumen[nombre][metrica]["media_niveles"] = float(
                    np.mean(list(por_nivel[nombre][metrica].values())))
        if nombre in series_por_semilla:
            resumen[nombre]["progreso"]["por_semilla"] = [
                float(np.mean(s["progreso"])) for s in series_por_semilla[nombre]]

    ancho = max((len(n) for n in resumen), default=6)
    print(f"\n{'modelo':<{ancho + 2}}  progreso    meta   " + "  ".join(f"{e:>5}" for e in etiquetas))
    for nombre, valores in resumen.items():
        print(f"  {nombre:<{ancho}} {valores['progreso']['media']:.3f}     "
              f"{valores['meta']['media']:.3f}   "
              + "  ".join(f"{por_nivel[nombre]['progreso'][e]:.3f}" for e in etiquetas)
              + (f"   semillas {' '.join(f'{v:.3f}' for v in valores['progreso']['por_semilla'])}"
                 if "por_semilla" in valores["progreso"] else ""))
    for clave, prueba in contrastes.items():
        texto = (f"  {clave}: progreso dif {prueba['progreso']['diferencia_media']:+.4f}"
                 f"  p partida = {prueba['progreso']['p']:.4f}")
        if clave in contrastes_por_nivel:
            nivel = contrastes_por_nivel[clave]["progreso"]
            texto += f"  p nivel = {nivel['p']:.4f} (min {nivel['p_minima_posible']:.3f})"
        if clave in contrastes_por_semilla:
            semilla = contrastes_por_semilla[clave]["progreso"]
            texto += f"  p semilla = {semilla['p']:.3f} (min {semilla['p_minima_posible']:.3f})"
        print(texto)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Con checkpoints sueltos no se pisa el informe de la evaluacion normal.
    destino = out_dir / (args.salida.name if args.salida else
                         "rl_evaluacion_checkpoints.json" if args.checkpoint
                         else "rl_evaluacion.json")
    destino.write_text(json.dumps(
        {"config": cfg, "n_niveles": n_levels, "repeticiones": args.repeticiones,
         "muestreada": bool(args.muestrea), "semilla": args.semilla,
         "split": args.split, "por_nivel": args.por_nivel, "marginal": bool(args.marginal),
         "frecuencias_de_accion": frecuencias,
         "checkpoints": [str(r) for r in (args.checkpoint or [])],
         "partidas": [{"particion": split, "semilla": s, "nivel": n}
                      for (split, s), n in zip(semillas, niveles.tolist())],
         "resumen": resumen, "por_nivel": por_nivel, "contrastes": contrastes,
         "contrastes_por_nivel": contrastes_por_nivel,
         "contrastes_por_semilla": contrastes_por_semilla,
         "series_por_semilla": series_por_semilla,
         "series_referencias": {
             nombre: {m: valores[m].tolist() for m in METRICAS}
             for nombre, valores in series.items() if nombre not in series_por_semilla}},
        indent=2, ensure_ascii=False,
    ))
    print(f"\ninforme en {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
