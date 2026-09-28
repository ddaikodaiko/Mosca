"""Graba partidas de una persona jugando al Super Mario Bros de NES, para que la red aprenda
de ella (fase 6c).

    python grabar_humano.py --config configs/mario_nes_humano.yaml            # niveles de train, al azar
    python grabar_humano.py --config configs/mario_nes_humano.yaml --nivel 1-2
    python grabar_humano.py --config configs/mario_nes_humano.yaml --particion val   # para validar, no entrenar

Controles: flechas izquierda/derecha, X o espacio = saltar (A), Z o mayusculas = correr (B).
Esc sale; la partida a medias se descarta.

Por que: la heuristica de `envs/nes_mario.JugadorHeuristico` solo se pasa el 1-1, asi que
clonarla no ensena a jugar, ensena a repetir un nivel. Una persona sabe pasarse niveles que
la red no ha visto, y eso es lo que tiene que aprender.

Lo que se guarda por partida, en `data/demos_humanas/*.npz`:
    obs       (T, F) float32  la observacion que ve la red, igual que en `NesMarioEnv`
    acciones  (T,)   int8     el boton de SIMPLE_MOVEMENT pulsado al decidir ese paso
    ram       (T, 2048) uint8 la RAM entera en ese momento: si mañana cambia la observacion,
                              se recalcula sin volver a jugar
y el nivel, la semilla, el resultado y la config con la que se grabo.

La red decide cada `frame_skip` fotogramas (15 veces por segundo con 4), asi que la tecla
que cuenta es la que esta pulsada al empezar cada paso; en pantalla se ve a 60 fps.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from envs.nes_mario import STAGES, NesConfig, NesMarioEnv
from envs.platformer import ObservationConfig

DIRECTORIO = Path("data/demos_humanas")
FPS = 60.0
ESCALA = 3


class Salir(Exception):
    """La persona ha pulsado Esc: la partida en curso no se guarda."""


def accion_de_teclas(izquierda: bool, derecha: bool, a: bool, b: bool) -> int:
    """Botones pulsados -> indice de SIMPLE_MOVEMENT (0 nada, 1 derecha, 2 derecha+A,
    3 derecha+B, 4 derecha+A+B, 5 A, 6 izquierda).

    SIMPLE_MOVEMENT no tiene izquierda+A ni izquierda+B: saltar pesa mas que retroceder, asi
    que izquierda+A se graba como A. B sola no hace nada en SMB y se graba como nada.
    """
    if derecha and not izquierda:
        if a and b:
            return 4
        return 2 if a else 3 if b else 1
    if a:
        return 5
    if izquierda:
        return 6
    return 0


def graba_partida(
    env: NesMarioEnv, semilla: int, decide: Callable[[], int], destino: Path,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Juega una partida entera con `decide()` y la guarda en `destino`. Devuelve el resumen."""
    observacion = env.reset(semilla)
    obs, acciones, ram = [], [], []
    done, info = False, {"progreso": 0.0, "meta": False, "muerte": False}
    while not done:
        accion = decide()
        obs.append(observacion)
        acciones.append(accion)
        ram.append(np.asarray(env.env.unwrapped.ram, dtype=np.uint8).copy())
        observacion, _, done, info = env.step(accion)
    resumen = {
        "nivel": f"{env.stage[0]}-{env.stage[1]}", "semilla": semilla,
        "progreso": float(info["progreso"]), "meta": bool(info["meta"]),
        "muerte": bool(info["muerte"]), "pasos": len(acciones),
    }
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destino, obs=np.stack(obs).astype(np.float32), acciones=np.asarray(acciones, np.int8),
        ram=np.stack(ram), resumen=json.dumps(resumen), extra=json.dumps(extra),
    )
    return resumen


def carga_demos(
    directorio: Path, niveles: set[str] | None = None, progreso_minimo: float = 0.0,
) -> list[dict[str, Any]]:
    """Lee las partidas grabadas, en orden de grabacion. Cada una: obs, acciones y resumen."""
    partidas = []
    for fichero in sorted(directorio.glob("*.npz")):
        with np.load(fichero) as datos:
            resumen = json.loads(str(datos["resumen"]))
            if niveles is not None and resumen["nivel"] not in niveles:
                continue
            if resumen["progreso"] < progreso_minimo and not resumen["meta"]:
                continue
            partidas.append({"obs": datos["obs"], "acciones": datos["acciones"].astype(np.int64),
                             "resumen": resumen, "fichero": fichero.name})
    return partidas


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/mario_nes_humano.yaml"))
    parser.add_argument("--particion", default="train", choices=sorted(STAGES))
    parser.add_argument("--nivel", default=None, help="p. ej. 1-2; por defecto, al azar en la particion")
    parser.add_argument("--partidas", type=int, default=1000, help="parar despues de N partidas")
    parser.add_argument("--semilla", type=int, default=None,
                        help="primera semilla; por defecto sale del reloj para no repetir")
    parser.add_argument("--salida", type=Path, default=DIRECTORIO)
    parser.add_argument("--auto", default=None, choices=["corre_salta"],
                        help="sin ventana ni teclado, juega una politica fija: solo para probar")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    nes_cfg = NesConfig(**cfg.get("nes", {}))
    # Una persona espera a veces (una plataforma, un enemigo que pasa): el corte por
    # atasco es para no gastar CPU entrenando, aqui sobra. No entra en la observacion.
    nes_cfg.stuck_steps = nes_cfg.max_steps
    if args.nivel is not None:
        nes_cfg.stage = tuple(int(p) for p in args.nivel.split("-"))
        if args.particion != "train" or nes_cfg.stage not in STAGES["train"]:
            print(f"aviso: {args.nivel} no es de train; no lo uses para entrenar")
    obs_cfg = ObservationConfig(**cfg.get("observation", {}))
    env = NesMarioEnv(args.particion, obs_cfg, nes_cfg)
    extra = {"config_path": str(args.config), "particion": args.particion,
             "nes": asdict(nes_cfg), "observation": asdict(obs_cfg), "fps": FPS}

    decide: Callable[[], int]
    visor: Any = None
    if args.auto == "corre_salta":
        decide = lambda: 4
    else:
        from nes_py._image_viewer import ImageViewer
        from pyglet.window import key

        # El teclado lo lee pyglet, no el visor de nes-py: el suyo revienta si llega el
        # soltar de una tecla pulsada fuera de la ventana (el Enter que lanza el comando).
        visor = ImageViewer("Mosca - grabando (Esc sale)", 240 * ESCALA, 256 * ESCALA)
        visor.open()
        teclado = key.KeyStateHandler()
        visor._window.push_handlers(teclado)
        reloj = {"siguiente": time.perf_counter()}

        def pinta(fotograma: np.ndarray) -> None:
            visor.show(fotograma)
            reloj["siguiente"] += 1.0 / FPS
            espera = reloj["siguiente"] - time.perf_counter()
            if espera > 0:
                time.sleep(espera)
            else:
                reloj["siguiente"] = time.perf_counter()  # sin acumular retraso
            if teclado[key.ESCAPE] or visor._window.has_exit:
                raise Salir

        def decide() -> int:
            return accion_de_teclas(
                teclado[key.LEFT], teclado[key.RIGHT],
                teclado[key.X] or teclado[key.SPACE],
                teclado[key.Z] or teclado[key.LSHIFT] or teclado[key.RSHIFT],
            )

        env.al_fotograma = pinta

    semilla = args.semilla if args.semilla is not None else int(time.time()) % 1_000_000
    pasos_total, metas = 0, 0
    try:
        for n in range(args.partidas):
            sello = datetime.now().strftime("%Y%m%d-%H%M%S")
            destino = args.salida / f"{sello}_{args.particion}_s{semilla}.npz"
            resumen = graba_partida(env, semilla, decide, destino, extra)
            pasos_total += resumen["pasos"]
            metas += resumen["meta"]
            final = "BANDERA" if resumen["meta"] else "muerte" if resumen["muerte"] else "tiempo"
            print(f"[{n + 1}] {resumen['nivel']}  progreso {resumen['progreso']:.2f}  {final:7s}"
                  f"  {resumen['pasos']} pasos   (llevas {pasos_total} pasos, {metas} banderas)",
                  flush=True)
            semilla += 1
            if visor is not None:
                time.sleep(0.8)
    except Salir:
        print("Esc: la partida en curso no se guarda.")
    finally:
        env.close()
        if visor is not None:
            visor.close()
    print(f"-> {args.salida}/  ({pasos_total} pasos grabados en esta sesion)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
