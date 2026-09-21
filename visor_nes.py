"""Visor: la pantalla del juego a un lado y las neuronas encendiendose al otro.

Uso:
    python visor_nes.py --config configs/mario_nes_demo.yaml \
        --checkpoint artifacts/mario_nes_demo/mosca_s0.pt --stage 1-1
    python visor_nes.py --config ... --checkpoint ... --record docs/media/visor.mp4

Es la parte barata del visualizador que pedia el usuario el 12-09-2026 y que quedo
anotada como trabajo futuro: el nucleo expone sus activaciones, asi que se puede ver la
actividad mientras juega. Sirve ademas de diagnostico: si la retina de neuronas se queda
toda saturada o toda apagada, el modelo no esta usando el nucleo.

Por defecto el cerebro gira en 3d (`--vista 3d`): la rotacion y la proyeccion se hacen a
mano sobre las coordenadas, no con mplot3d, porque asi va fluido con 23.000 puntos de
silueta y se puede pintar la profundidad. `--giro` cambia los grados por cuadro y
`--vista frontal|superior|lateral` congela un plano.

**La disposicion es anatomica de verdad.** Cada neurona se dibuja donde esta en el
cerebro, con las coordenadas publicas de FlyWire v783 (Schlegel et al., 2024), las mismas
que usa el visor de fluidos: las 138.625 neuronas del conectoma en gris tenue como
silueta de fondo, y encima las del subgrafo encendiendose. Como el subgrafo se elige por
manchas repartidas, los parches salen distribuidos por todo el cerebro, no agrupados.
Requiere `python -m data.download_annotations` una vez (32 MB, sin registro).

Con `--rejilla` se vuelve a la disposicion abstracta por indice, que es mas legible para
ver saturacion y silencio pero no dice nada de anatomia.

Sin `--checkpoint` usa una politica sin entrenar: se ve la actividad igual, y es util
para comparar "antes" y "despues".
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from data.spectral import resolve_device
from envs.nes_mario import NesConfig, NesMarioEnv
from envs.platformer import N_ACTIONS, ObservationConfig
from visor import ANNOTATIONS, brain_positions

ACCIONES = ("nada", "derecha", "dcha+A", "dcha+B", "dcha+A+B", "A", "izquierda")
FONDO = "#0b0f1a"
TEXTO = "#e6edf3"


def rejilla(n: int) -> tuple[int, int]:
    """Rejilla lo mas cuadrada posible que contenga n neuronas."""
    filas = int(np.floor(np.sqrt(n)))
    while filas > 1 and n % filas:
        filas -= 1
    columnas = n // filas
    if columnas > 4 * filas:      # evita tiras larguisimas cuando n es primo
        filas = int(round(np.sqrt(n)))
        columnas = int(np.ceil(n / filas))
    return filas, columnas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--stage", default="1-1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true", help="muestrear en vez de el maximo")
    parser.add_argument("--record", type=Path, default=None, help="guardar en mp4 en vez de mostrar")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--wait", action="store_true",
                        help="esperar a que aparezca el checkpoint en vez de fallar")
    parser.add_argument("--rejilla", action="store_true",
                        help="disposicion abstracta por indice en vez de la anatomica")
    parser.add_argument("--vista", default="3d",
                        choices=("3d", "frontal", "superior", "lateral"),
                        help="3d gira el cerebro; el resto son planos fijos")
    parser.add_argument("--giro", type=float, default=1.1,
                        help="grados por cuadro que gira la vista 3d")
    parser.add_argument("--puntos", type=int, default=12000,
                        help="neuronas de silueta que se dibujan (menos = mas fluido)")
    parser.add_argument("--ancho", type=float, default=11.0, help="ancho de la ventana en pulgadas")
    parser.add_argument("--dpi", type=int, default=90,
                        help="puntos por pulgada; el coste de pintar va con los pixeles")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg" if args.record else "TkAgg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    cfg = yaml.safe_load(args.config.read_text())
    device = resolve_device(cfg.get("device", "auto"))
    world, level = (int(parte) for parte in args.stage.split("-"))

    nes = dict(cfg.get("nes", {}))
    nes.pop("stage", None)       # el nivel lo manda --stage
    nes["max_steps"] = args.max_steps
    env = NesMarioEnv(
        "train", ObservationConfig(**cfg.get("observation", {})),
        NesConfig(**nes), render_mode=None,
    )
    env.stages = ((world, level),)

    from models.masked_policy import ConnectomePolicy, build_policy
    from train_rl import load_wirings

    if args.checkpoint is not None:
        from evaluate_rl import carga_politica

        # El entrenamiento guarda en cada evaluacion, asi que al principio el fichero
        # todavia no existe. Decirlo claro, y con --wait quedarse esperando.
        if not args.checkpoint.exists():
            hechos = sorted(args.checkpoint.parent.glob("*.pt")) if args.checkpoint.parent.exists() else []
            if not args.wait:
                disponibles = ", ".join(r.name for r in hechos) or "ninguno todavia"
                raise SystemExit(
                    f"no existe {args.checkpoint}\n"
                    f"  checkpoints en {args.checkpoint.parent}: {disponibles}\n"
                    f"  el entrenamiento guarda uno cada `eval_every` iteraciones;"
                    f" usa --wait para esperarlo, o quita --checkpoint para ver el modelo sin entrenar"
                )
            print(f"esperando a {args.checkpoint} ...", flush=True)
            while not args.checkpoint.exists():
                time.sleep(10)
        modelo = carga_politica(args.checkpoint, cfg, load_wirings(cfg), device)
        etiqueta = f"{args.checkpoint.stem}"
    else:
        modelo = build_policy(
            cfg["models"][0], cfg, env.n_features, N_ACTIONS, 0, load_wirings(cfg)
        ).to(device).eval()
        etiqueta = f"{cfg['models'][0]} SIN ENTRENAR"
    if not isinstance(modelo, ConnectomePolicy):
        raise SystemExit("el visor necesita un modelo con nucleo cableado, no el MLP")

    n = modelo.core.n_neurons
    filas, columnas = rejilla(n)

    # Anatomia: posicion real de cada neurona del subgrafo en el cerebro de la mosca.
    anatomia = None
    if not args.rejilla:
        if not ANNOTATIONS.is_file():
            raise SystemExit(
                f"falta {ANNOTATIONS}\n  python -m data.download_annotations  (32 MB, una vez)"
                f"\n  o usa --rejilla para la disposicion abstracta"
            )
        import numpy as _np

        nodos = _np.load(Path(cfg["wiring_dir"]) / "mosca.npz")["nodes"]
        xyz, tiene_posicion = brain_positions()
        sin_posicion = int((~tiene_posicion[nodos]).sum())
        if sin_posicion:
            print(f"aviso: {sin_posicion} neuronas del subgrafo sin coordenadas")
        centro = _np.nanmean(xyz[tiene_posicion], axis=0)
        # Una sola escala para los tres ejes (si no, el cerebro se deforma al girar),
        # pero tomada del percentil 99,5 y no del maximo: cuatro neuronas sueltas en los
        # extremos dejaban el cerebro del tamano de un garbanzo en medio del panel.
        escala = float(_np.nanpercentile(_np.abs(xyz[tiene_posicion] - centro), 99.5))
        fondo3d = (xyz[tiene_posicion] - centro) / escala
        activas3d = _np.nan_to_num((xyz[nodos] - centro) / escala)
        if args.vista != "3d":
            ejes_vista = {"frontal": (0, 1), "superior": (0, 2), "lateral": (2, 1)}[args.vista]
            anatomia = (fondo3d[:, ejes_vista], activas3d[:, ejes_vista])
        else:
            anatomia = (fondo3d, activas3d)
    sensorial = modelo.sensory.cpu().numpy()
    motora = modelo.motor.cpu().numpy()
    print(f"{etiqueta}: {n} neuronas"
          f" ({'rejilla %dx%d' % (filas, columnas) if args.rejilla else 'anatomia FlyWire v783'}),"
          f" {sensorial.size} sensoriales, {motora.size} motoras")

    figura = plt.figure(figsize=(args.ancho, args.ancho * 0.45), dpi=args.dpi, facecolor=FONDO)
    rejas = figura.add_gridspec(3, 2, width_ratios=[1, 1.25], height_ratios=[6, 1, 1],
                                hspace=0.25, wspace=0.08)
    eje_juego = figura.add_subplot(rejas[:, 0])
    eje_cerebro = figura.add_subplot(rejas[0, 1])
    eje_puertos = figura.add_subplot(rejas[1, 1])
    eje_botones = figura.add_subplot(rejas[2, 1])
    for eje in (eje_juego, eje_cerebro, eje_puertos, eje_botones):
        eje.set_facecolor(FONDO)
        eje.set_xticks([]); eje.set_yticks([])
        for lado in eje.spines.values():
            lado.set_color("#26304a")

    env.reset(args.seed)
    # nearest, no el suavizado por defecto: es 6 ms mas barato por cuadro y ademas es lo
    # correcto para pixel art, que con antialiasing sale emborronado.
    imagen_juego = eje_juego.imshow(env.frame, interpolation="nearest")
    gira3d = None
    if anatomia is not None and args.vista == "3d":
        fondo3d, activas3d = anatomia
        # La silueta es lo unico caro del visor: son decenas de miles de puntos que hay
        # que reproyectar y repintar en cada cuadro. Se muestrea a `--puntos` (12.000 por
        # defecto) repartidos por todo el cerebro, no los primeros: el orden del fichero
        # agrupa por region y cogerlos seguidos dejaria medio cerebro sin silueta.
        if fondo3d.shape[0] > args.puntos:
            paso = fondo3d.shape[0] / args.puntos
            fondo3d = fondo3d[(_np.arange(args.puntos) * paso).astype(int)]
        # La silueta va en tres Line2D por banda de profundidad, no en un scatter.
        # Medido: `scatter` con 12.000 puntos cuesta 170 ms por cuadro, porque reconstruye
        # una ruta por punto; tres lineas con marcador cuestan 10 ms y se pierde solo el
        # degradado continuo de profundidad, que con tres tonos se sigue leyendo.
        capas_fondo = [
            eje_cerebro.plot([], [], linestyle="none", marker=".", markersize=tamano,
                             color=color, alpha=alfa, zorder=1)[0]
            for tamano, color, alfa in (
                (1.3, "#93a9d6", 0.55),   # delante
                (1.0, "#5d6f9c", 0.45),
                (0.8, "#3b4a70", 0.35),   # detras
            )
        ]
        nube = eje_cerebro.scatter(
            activas3d[:, 0], -activas3d[:, 1], s=16.0, c=np.zeros(activas3d.shape[0]),
            cmap="inferno", vmin=0.0, vmax=0.8, linewidths=0, zorder=3,
        )
        # El cerebro es ancho y plano: al girar barre x y z, pero en vertical solo ocupa
        # lo que mide en y. Con un cuadrado sobraba la mitad del panel en blanco.
        ancho = float(np.percentile(np.abs(fondo3d[:, [0, 2]]), 99.8)) * 1.08
        alto = float(np.percentile(np.abs(fondo3d[:, 1]), 99.8)) * 1.15
        eje_cerebro.set_xlim(-ancho, ancho)
        eje_cerebro.set_ylim(-alto, alto)
        eje_cerebro.set_aspect("equal")
        imagen_cerebro = None

        estado_giro = {"angulo": 0.0}
        bandas_profundidad = np.percentile(fondo3d[:, 2], [33.0, 66.0])

        def gira3d() -> None:
            """Gira alrededor del eje dorsoventral, que es el giro de toda la vida."""
            estado_giro["angulo"] += np.deg2rad(args.giro)
            coseno, seno = np.cos(estado_giro["angulo"]), np.sin(estado_giro["angulo"])
            x = fondo3d[:, 0] * coseno + fondo3d[:, 2] * seno
            z = -fondo3d[:, 0] * seno + fondo3d[:, 2] * coseno
            banda = np.digitize(z, bandas_profundidad)
            for i, capa in enumerate(capas_fondo):
                dentro = banda == i
                capa.set_data(x[dentro], -fondo3d[dentro, 1])

            x = activas3d[:, 0] * coseno + activas3d[:, 2] * seno
            z = -activas3d[:, 0] * seno + activas3d[:, 2] * coseno
            nube.set_offsets(np.c_[x, -activas3d[:, 1]])
            # Las activas cambian de tamano con la profundidad, que es lo que da volumen.
            nube.set_sizes(7.0 + 11.0 * np.clip((z + 1.0) / 2.0, 0.0, 1.0))
    elif anatomia is not None:
        fondo, activas = anatomia
        eje_cerebro.scatter(fondo[::3, 0], -fondo[::3, 1], s=0.5, c="#33415f",
                            alpha=0.55, linewidths=0, rasterized=True)
        nube = eje_cerebro.scatter(activas[:, 0], -activas[:, 1], s=16.0,
                                   c=np.zeros(activas.shape[0]), cmap="inferno",
                                   vmin=0.0, vmax=0.8, linewidths=0, zorder=3)
        eje_cerebro.set_aspect("equal")
        imagen_cerebro = None
    else:
        nube = None
        lienzo = np.zeros((filas, columnas), dtype=np.float32)
        imagen_cerebro = eje_cerebro.imshow(lienzo, cmap="inferno", vmin=0.0, vmax=1.0,
                                            interpolation="nearest", aspect="auto")
    puertos = np.zeros((2, max(sensorial.size, motora.size)), dtype=np.float32)
    imagen_puertos = eje_puertos.imshow(puertos, cmap="inferno", vmin=0.0, vmax=1.0,
                                        interpolation="nearest", aspect="auto")
    barras = eje_botones.bar(range(N_ACTIONS), np.zeros(N_ACTIONS), color="#f2b632")
    eje_botones.set_ylim(0, 1)
    eje_botones.set_xticks(range(N_ACTIONS))
    eje_botones.set_xticklabels(ACCIONES, color=TEXTO, fontsize=7)
    disposicion = ("rejilla por indice" if args.rejilla else
                   ("cerebro de la mosca, 3d" if args.vista == "3d"
                    else f"cerebro de la mosca, vista {args.vista}"))
    eje_cerebro.set_title(f"actividad del nucleo  |  {etiqueta}  |  {disposicion}",
                          color=TEXTO, fontsize=9)
    eje_puertos.set_title("sensoriales (arriba) y motoras (abajo)", color=TEXTO, fontsize=8)
    marcador = eje_juego.set_title("", color=TEXTO, fontsize=9, family="monospace")

    estado = {"h": modelo.initial_state(1, device)}

    def paso() -> tuple[bool, dict]:
        obs = estado["obs"]
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)[None]
            logits, _, h, traza = modelo(tensor, estado["h"], record=True)
        estado["h"] = h
        probabilidades = logits.softmax(dim=1)[0].cpu().numpy()
        accion = (int(torch.distributions.Categorical(logits=logits).sample().item())
                  if args.sample else int(logits.argmax(dim=1).item()))
        obs, _, done, info = env.step(accion)
        estado["obs"] = obs

        # |h| en [0,1]: el signo no se ve, la intensidad si. tanh ya acota a [-1,1].
        actividad = traza[-1][0].abs().cpu().numpy()
        if nube is not None:
            nube.set_array(actividad)
            if gira3d is not None:
                gira3d()
        else:
            lienzo = np.zeros(filas * columnas, dtype=np.float32)
            lienzo[:n] = actividad[:filas * columnas]
            imagen_cerebro.set_data(lienzo.reshape(filas, columnas))
        puertos[:] = 0.0
        puertos[0, :sensorial.size] = actividad[sensorial]
        puertos[1, :motora.size] = actividad[motora]
        imagen_puertos.set_data(puertos)
        for barra, altura in zip(barras, probabilidades):
            barra.set_height(float(altura))
        imagen_juego.set_data(env.frame)
        marcador.set_text(
            f"{info['nivel']}   x {env.x:4d}   {info['progreso'] * 100:5.1f} %   "
            f"activa {float((actividad > 0.1).mean()) * 100:4.1f} %   {ACCIONES[accion]}"
        )
        return done, info

    estado["obs"] = env.reset(args.seed)
    estado["h"] = modelo.initial_state(1, device)

    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        escritor = FFMpegWriter(fps=args.fps, codec="libopenh264", bitrate=6000)
        with escritor.saving(figura, str(args.record), dpi=100):
            done = False
            while not done:
                done, info = paso()
                escritor.grab_frame()
        print(f"video en {args.record}   ({'META' if info['meta'] else 'fin'},"
              f" progreso {info['progreso']:.1%})")
        return 0

    # Blitting: se dibuja la figura una vez, se guarda el fondo y en cada cuadro solo se
    # repintan los artistas que cambian. Sin esto, matplotlib rehace ejes, titulos y
    # marcos enteros 15 veces por segundo, que es donde se iba el tiempo.
    animados = [imagen_juego, imagen_puertos, marcador, *barras]
    animados += [nube] if nube is not None else [imagen_cerebro]
    if gira3d is not None:
        animados.extend(capas_fondo)
    for artista in animados:
        artista.set_animated(True)

    plt.show(block=False)
    figura.canvas.draw()
    fondo_figura = figura.canvas.copy_from_bbox(figura.bbox)

    def pinta() -> None:
        figura.canvas.restore_region(fondo_figura)
        for artista in animados:
            artista.axes.draw_artist(artista)
        figura.canvas.blit(figura.bbox)
        figura.canvas.flush_events()

    partida = 0
    cuadros_pintados = 0
    arranque = time.monotonic()
    try:
        while True:
            partida += 1
            estado["obs"] = env.reset(args.seed + partida - 1)
            estado["h"] = modelo.initial_state(1, device)
            done = False
            reloj = time.monotonic()
            while not done:
                done, info = paso()
                pinta()
                cuadros_pintados += 1
                if args.speed > 0:
                    reloj += env.cfg.frame_skip / 60.0 / args.speed
                    espera = reloj - time.monotonic()
                    if espera > 0:
                        time.sleep(espera)
                    else:
                        reloj = time.monotonic()
            transcurrido = max(time.monotonic() - arranque, 1e-6)
            print(f"  partida {partida}: {'META' if info['meta'] else 'fin'}"
                  f"   x {env.x}   progreso {info['progreso']:.1%}"
                  f"   {cuadros_pintados / transcurrido:.1f} cuadros/s")
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\nparado")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
