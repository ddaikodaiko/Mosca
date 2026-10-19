"""PPO sobre el entorno de plataformas (fase 6, seccion 13 del diseno).

Uso:
    python train_rl.py --config configs/mario_proto.yaml            # prototipo CPU
    python train_rl.py --config configs/mario.yaml --model mosca --seed 0

Misma pregunta que en fluidos, otra tarea: ¿el cableado de la mosca aporta algo frente a
cableados aleatorios con las mismas estadisticas? Los modelos son los mismos seis mas el
MLP libre, y comparten entorno, semillas de nivel y presupuesto de pasos.

**Recurrencia y gradiente.** El estado oculto persiste entre pasos del entorno durante el
despliegue, pero el gradiente se trunca a UN paso de entorno (que son `n_internal_steps`
pasos de la recurrencia interna): en la actualizacion se recalcula el paso partiendo de la
h almacenada, que entra desacoplada. Es la variante barata de PPO recurrente y la unica
que cabe en CPU con N=4.000. Queda registrado porque limita lo que el nucleo puede
aprender: no puede construir memoria de mas de un paso del juego.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn

from data.spectral import resolve_device
from envs.platformer import (
    N_ACTIONS,
    LevelConfig,
    ObservationConfig,
    RewardConfig,
    SeedRanges,
    VectorPlatformer,
)
from models.masked_policy import build_policy
from models.masked_rnn import Wiring
from train import Reanudacion


@dataclass
class Rollout:
    observations: Tensor    # (T, n_envs, F)
    hidden: Tensor          # (T, n_envs, N) estado ANTES del paso, desacoplado
    actions: Tensor         # (T, n_envs)
    log_probs: Tensor
    values: Tensor
    advantages: Tensor
    returns: Tensor


def build_environments(cfg: dict[str, Any], split: str, n_envs: int):
    """El entorno lo elige la config: `env: propio` o `env: nes`.

    Los dos entregan la MISMA observacion (302 numeros) y las mismas 7 acciones, asi que
    la politica del conectoma es identica en los dos y los resultados son comparables
    entre si en forma, aunque no en dificultad.
    """
    if cfg.get("env", "propio") == "nes":
        from envs.nes_mario import NesConfig, VectorNesMario, VectorNesMarioParalelo

        # Un proceso por emulador (fase 6c): mismas semillas y mismos niveles que en serie.
        vector = VectorNesMarioParalelo if cfg.get("parallel_envs", False) else VectorNesMario
        return vector(
            n_envs=n_envs,
            split=split,
            obs_cfg=ObservationConfig(**cfg.get("observation", {})),
            cfg=NesConfig(**cfg.get("nes", {})),
        )
    return VectorPlatformer(
        n_envs=n_envs,
        split=split,
        seeds=SeedRanges.from_config(cfg),
        level_cfg=LevelConfig(**cfg.get("level", {})),
        obs_cfg=ObservationConfig(**cfg.get("observation", {})),
        reward_cfg=RewardConfig(**cfg.get("reward", {})),
        max_steps=int(cfg.get("max_episode_steps", 1200)),
        release_to_jump=bool(cfg.get("release_to_jump", False)),
    )


def cierra(envs: Any) -> None:
    """Los entornos en paralelo tienen procesos que cerrar; los demas no tienen `close`."""
    if hasattr(envs, "close"):
        envs.close()


@torch.no_grad()
def muestrea_observaciones(envs, state: dict[str, Any], n: int) -> Tensor:
    """Observaciones de una politica al azar sobre los niveles de ENTRENAMIENTO.

    Solo de entrenamiento: las estadisticas son parte del modelo y mirar validacion o
    prueba para calcularlas seria filtrar. Se descarta el estado al terminar, para que el
    entrenamiento arranque en el mismo punto que sin esto.
    """
    muestras = [state["obs"]]
    vistas = state["obs"].shape[0]
    while vistas < n:
        acciones = np.random.randint(0, N_ACTIONS, size=envs.n_envs)
        obs, _, _, _ = envs.step(acciones)
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=state["obs"].device)
        muestras.append(tensor)
        vistas += tensor.shape[0]
    state["obs"] = torch.as_tensor(
        envs.reset(), dtype=torch.float32, device=state["obs"].device
    )
    return torch.cat(muestras, dim=0)


@torch.no_grad()
def collect(
    policy: nn.Module,
    envs: VectorPlatformer,
    state: dict[str, Any],
    steps: int,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> tuple[Rollout, dict[str, float]]:
    """Un lote de experiencia con GAE. `state` lleva obs y h entre llamadas."""
    n_envs = envs.n_envs
    observations = torch.zeros(steps, n_envs, envs.n_features, device=device)
    hidden = torch.zeros(steps, n_envs, policy.n_neurons, device=device)
    actions = torch.zeros(steps, n_envs, dtype=torch.long, device=device)
    log_probs = torch.zeros(steps, n_envs, device=device)
    values = torch.zeros(steps, n_envs, device=device)
    rewards = torch.zeros(steps, n_envs, device=device)
    alive = torch.zeros(steps, n_envs, device=device)   # 1 si el episodio continua

    episodes: list[dict[str, float]] = []
    for t in range(steps):
        observations[t] = state["obs"]
        hidden[t] = state["h"]
        logits, value, h = policy(state["obs"], state["h"])
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        actions[t] = action
        log_probs[t] = distribution.log_prob(action)
        # El critico puede estar prediciendo el retorno normalizado; GAE necesita
        # unidades de recompensa. Con el normalizador apagado esto es la identidad.
        values[t] = policy.normalizador.desnormaliza(value)

        obs, reward, done, infos = envs.step(action.cpu().numpy())
        rewards[t] = torch.as_tensor(reward, device=device)
        done_mask = torch.as_tensor(done, device=device)
        alive[t] = (~done_mask).float()
        for i, finished in enumerate(done):
            if finished:
                episodes.append(infos[i])
        # El episodio siguiente es otro nivel: la h no puede sobrevivir al corte.
        state["h"] = torch.where(done_mask[:, None], torch.zeros_like(h), h)
        state["obs"] = torch.as_tensor(obs, dtype=torch.float32, device=device)

    _, last_value, _ = policy(state["obs"], state["h"])
    last_value = policy.normalizador.desnormaliza(last_value)
    advantages = torch.zeros_like(rewards)
    accumulator = torch.zeros(n_envs, device=device)
    next_value = last_value
    for t in reversed(range(steps)):
        delta = rewards[t] + gamma * next_value * alive[t] - values[t]
        accumulator = delta + gamma * gae_lambda * alive[t] * accumulator
        advantages[t] = accumulator
        next_value = values[t]

    # Los episodios que arrancan en la frontera (fase 6c) llevan el progreso inflado por
    # construccion: el de entrenamiento se informa solo con los que empiezan al principio.
    desde_salida = [e for e in episodes if not e.get("desde_frontera", False)]
    statistics = {
        "episodios": float(len(episodes)),
        "episodios_frontera": float(len(episodes) - len(desde_salida)),
        "recompensa_media": float(rewards.sum().item() / max(len(episodes), 1)),
        "progreso_medio": float(np.mean([e["progreso"] for e in desde_salida])) if desde_salida else float("nan"),
        "tasa_meta": float(np.mean([e["meta"] for e in desde_salida])) if desde_salida else float("nan"),
    }
    return Rollout(
        observations, hidden, actions, log_probs, values, advantages, advantages + values
    ), statistics


def coeficientes(ppo: dict[str, Any], iteration: int) -> dict[str, Any]:
    """`ppo` con el coeficiente de entropia de esta iteracion.

    Con `entropy_final` (fase 6c) baja en linea recta desde `entropy_coefficient` hasta
    ese valor a lo largo del entrenamiento: explora mucho al principio y afina al final.
    Sin la clave es constante, como antes.
    """
    if "entropy_final" not in ppo:
        return ppo
    fraccion = iteration / max(int(ppo["iterations"]) - 1, 1)
    inicial, final = float(ppo["entropy_coefficient"]), float(ppo["entropy_final"])
    return {**ppo, "entropy_coefficient": inicial + (final - inicial) * fraccion}


def ppo_update(
    policy: nn.Module,
    optimiser: torch.optim.Optimizer,
    batch: Rollout,
    cfg: dict[str, Any],
    generator: torch.Generator,
) -> dict[str, float]:
    flat = lambda tensor: tensor.reshape(-1, *tensor.shape[2:])
    observations = flat(batch.observations)
    hidden = flat(batch.hidden)
    actions = flat(batch.actions)
    old_log_probs = flat(batch.log_probs)
    advantages = flat(batch.advantages)
    returns = flat(batch.returns)
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
    # El critico aprende en escala normalizada: con retornos de 30-60 y una lectura lineal
    # sobre neuronas acotadas por tanh, ajustarlos pide pesos enormes y a lr 3e-4 no llega
    # (medido: tras 300 iteraciones predecia 27,7 para cualquier estado, varianza explicada
    # 0,10). Los valores guardados en el lote estan en unidades de recompensa, asi que para
    # el recorte del critico habria que volver a normalizarlos; aqui no se recorta el valor,
    # asi que basta con normalizar el objetivo. Apagado, todo esto es la identidad.
    policy.normalizador.actualiza(returns)
    returns = policy.normalizador.normaliza(returns)

    n_samples = observations.shape[0]
    minibatch = max(1, n_samples // int(cfg["n_minibatches"]))
    losses = {"politica": 0.0, "valor": 0.0, "entropia": 0.0, "kl": 0.0}
    n_updates = 0
    for _ in range(int(cfg["epochs"])):
        order = torch.randperm(n_samples, generator=generator, device=observations.device)
        for start in range(0, n_samples, minibatch):
            index = order[start:start + minibatch]
            logits, value, _ = policy(observations[index], hidden[index])
            distribution = torch.distributions.Categorical(logits=logits)
            log_prob = distribution.log_prob(actions[index])
            ratio = (log_prob - old_log_probs[index]).exp()
            clip = float(cfg["clip"])
            surrogate = torch.min(
                ratio * advantages[index],
                ratio.clamp(1 - clip, 1 + clip) * advantages[index],
            )
            policy_loss = -surrogate.mean()
            value_loss = 0.5 * (value - returns[index]).pow(2).mean()
            entropy = distribution.entropy().mean()
            loss = (
                policy_loss
                + float(cfg["value_coefficient"]) * value_loss
                - float(cfg["entropy_coefficient"]) * entropy
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), float(cfg.get("grad_clip", 1.0)))
            optimiser.step()

            losses["politica"] += float(policy_loss.item())
            losses["valor"] += float(value_loss.item())
            losses["entropia"] += float(entropy.item())
            losses["kl"] += float((old_log_probs[index] - log_prob).mean().item())
            n_updates += 1
    return {k: v / max(n_updates, 1) for k, v in losses.items()}


@torch.no_grad()
def evaluate(
    policy: nn.Module, cfg: dict[str, Any], split: str, n_episodes: int, device: torch.device,
    sample: bool = False,
) -> dict[str, float]:
    """Niveles de `split`, con semillas disjuntas de las de entrenamiento.

    `sample=False` coge el maximo (lo que se publica) y `sample=True` muestrea. Se miden
    las dos porque la codiciosa se atasca: en el prototipo daba el mismo valor exacto en
    todas las evaluaciones mientras la muestreada si se movia, y publicar solo una de las
    dos daria una idea equivocada de lo que ha aprendido la red.
    """
    n_envs = min(int(cfg.get("eval_envs", 8)), n_episodes)
    envs = build_environments(cfg, split, n_envs)
    obs = torch.as_tensor(envs.reset(), dtype=torch.float32, device=device)
    h = policy.initial_state(n_envs, device)
    finished: list[dict[str, float]] = []
    max_steps = int(cfg.get("max_episode_steps", 1200))
    for _ in range(max_steps * (n_episodes // n_envs + 1)):
        if len(finished) >= n_episodes:
            break
        logits, _, h = policy(obs, h)
        action = (torch.distributions.Categorical(logits=logits).sample() if sample
                  else logits.argmax(dim=1))
        next_obs, _, done, infos = envs.step(action.cpu().numpy())
        for i, done_i in enumerate(done):
            if done_i:
                finished.append(infos[i])
        done_mask = torch.as_tensor(done, device=device)
        h = torch.where(done_mask[:, None], torch.zeros_like(h), h)
        obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    finished = finished[:n_episodes]
    cierra(envs)
    return {
        "episodios": float(len(finished)),
        "progreso": float(np.mean([e["progreso"] for e in finished])) if finished else float("nan"),
        "tasa_meta": float(np.mean([e["meta"] for e in finished])) if finished else float("nan"),
        "tasa_muerte": float(np.mean([e["muerte"] for e in finished])) if finished else float("nan"),
        "pisados": float(np.mean([e["pisados"] for e in finished])) if finished else float("nan"),
    }


def train_one(
    name: str, seed: int, cfg: dict[str, Any], wirings: dict[str, Wiring], device: torch.device,
    desde: Path | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    ppo = cfg["ppo"]
    n_envs = int(ppo["n_envs"])
    envs = build_environments(cfg, "train", n_envs)
    n_features = envs.n_features
    policy = build_policy(name, cfg, n_features, N_ACTIONS, seed, wirings).to(device)
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    optimiser = torch.optim.Adam(policy.parameters(), lr=float(ppo["lr"]))

    state = {
        "obs": torch.as_tensor(envs.reset(), dtype=torch.float32, device=device),
        "h": policy.initial_state(n_envs, device),
    }
    total_iterations = int(ppo["iterations"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda it: 1.0 - it / max(total_iterations, 1)
    )

    started = time.time()
    history: list[dict[str, Any]] = []
    best = {"progreso": float("-inf")}
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"{name}_s{seed}.pt"

    # Reanudacion. A diferencia de la de train.py, esta NO es bit a bit: los entornos no
    # se pueden serializar (el emulador lleva su propia RAM), asi que al reanudar arrancan
    # de cero y la recoleccion ya no es la que habria salido de un tiron. Lo que si se
    # conserva es todo lo aprendido: pesos, optimizador, planificador, contador de
    # iteraciones e historial. El numero de reanudaciones se guarda en el informe, porque
    # un entrenamiento reanudado no es reproducible desde la semilla y eso hay que decirlo.
    reanuda = Reanudacion.para(cfg, name, seed, subdirectorio="")
    primera_iteracion, reanudaciones = 0, 0
    if reanuda.estado is not None:
        estado = reanuda.estado
        policy.load_state_dict(estado["modelo"])
        optimiser.load_state_dict(estado["optimizador"])
        scheduler.load_state_dict(estado["planificador"])
        generator.set_state(estado["generador"])
        history = list(estado["historia"])
        best = dict(estado["mejor"])
        primera_iteracion = int(estado["iteracion"]) + 1
        reanudaciones = int(estado["reanudaciones"]) + 1
    elif desde is not None:
        # Arranque desde pesos ya entrenados (p. ej. la politica clonada de la heuristica).
        # Las estadisticas de la entrada viajan en el state_dict y NO se vuelven a medir:
        # cambiarlas moveria la politica de partida. Se evalua antes de la primera
        # iteracion para saber de donde parte PPO.
        policy.load_state_dict(torch.load(desde, map_location=device)["state_dict"])
        episodios = int(cfg.get("eval_episodes", 16))
        history.append({"iteracion": -1, "desde": str(desde),
                        "val": evaluate(policy, cfg, "val", episodios, device),
                        "val_muestreada": evaluate(policy, cfg, "val", episodios, device, sample=True)})
        print(f"  arranque desde {desde}: val progreso {history[-1]['val']['progreso']:.3f}"
              f" (muestreada {history[-1]['val_muestreada']['progreso']:.3f})")
    elif cfg.get("normalise_observations", True):
        # Las estadisticas se miden UNA vez, antes de entrenar; al reanudar vienen ya
        # dentro del checkpoint y volver a medirlas cambiaria el modelo a mitad.
        muestra = muestrea_observaciones(envs, state, int(cfg.get("normalise_samples", 2000)))
        policy.fit_feature_statistics(muestra)

    def guarda() -> None:
        """Se guarda en cada evaluacion, no solo al final: un entrenamiento de horas
        tiene que poder mirarse mientras corre, y cortarse sin perderlo todo."""
        torch.save(
            {"modelo": name, "semilla": seed, "config": cfg,
             "state_dict": policy.state_dict()}, checkpoint,
        )
    print(f"\n[{name} s{seed}] {policy.n_neurons} neuronas, {n_parameters:,} parametros")
    for iteration in range(primera_iteracion, total_iterations):
        t_iteracion = time.time()
        batch, statistics = collect(
            policy, envs, state, int(ppo["rollout_steps"]),
            float(ppo["gamma"]), float(ppo["gae_lambda"]), device,
        )
        t_recoleccion = time.time() - t_iteracion
        t_actualizacion = time.time()
        losses = ppo_update(policy, optimiser, batch, coeficientes(ppo, iteration), generator)
        t_actualizacion = time.time() - t_actualizacion
        scheduler.step()
        if iteration < int(cfg.get("time_first_iterations", 3)):
            # Las primeras iteraciones cronometradas: con eval_every alto, si no, la
            # primera senal de cuanto cuesta esto tarda una hora en aparecer. Sirve para
            # decidir `iterations` en una maquina nueva antes de lanzar la tanda entera.
            total = t_recoleccion + t_actualizacion
            print(f"  iter {iteration + 1:4d}  {total:.1f} s"
                  f"  (recoleccion {t_recoleccion:.1f} s, actualizacion {t_actualizacion:.1f} s)"
                  f"  -> {total * total_iterations / 3600:.1f} h las {total_iterations}")
        entry = {"iteracion": iteration, **statistics, **losses,
                 "segundos_recoleccion": round(t_recoleccion, 2),
                 "segundos_actualizacion": round(t_actualizacion, 2)}
        if (iteration + 1) % int(cfg.get("eval_every", 10)) == 0 or iteration == total_iterations - 1:
            episodios = int(cfg.get("eval_episodes", 16))
            entry["val"] = evaluate(policy, cfg, "val", episodios, device)
            entry["val_muestreada"] = evaluate(policy, cfg, "val", episodios, device, sample=True)
            guarda()
            if entry["val"]["progreso"] > best["progreso"]:
                best = dict(entry["val"], iteracion=iteration)
            print(
                f"  iter {iteration + 1:4d}/{total_iterations}"
                f"  train progreso {statistics['progreso_medio']:.3f}"
                f"  val progreso {entry['val']['progreso']:.3f}"
                f" (muestreada {entry['val_muestreada']['progreso']:.3f})"
                f"  meta {entry['val']['tasa_meta']:.2f}"
                f"  H {losses['entropia']:.3f}"
            )
        history.append(entry)
        if reanuda.toca_guardar(iteration + 1):
            reanuda.guarda(
                donde=f"iteracion {iteration + 1} de {total_iterations}",
                iteracion=iteration, modelo=policy.state_dict(),
                optimizador=optimiser.state_dict(), planificador=scheduler.state_dict(),
                generador=generator.get_state(), historia=history, mejor=best,
                reanudaciones=reanudaciones,
            )

    guarda()
    cierra(envs)
    reanuda.termina()
    episodios_test = int(cfg.get("test_episodes", 32))
    test = evaluate(policy, cfg, "test", episodios_test, device)
    test_muestreada = evaluate(policy, cfg, "test", episodios_test, device, sample=True)
    return {
        "modelo": name,
        "semilla": seed,
        "desde": str(desde) if desde is not None else None,
        "neuronas": int(policy.n_neurons),
        "parametros": int(n_parameters),
        "segundos": time.time() - started,
        # Un entrenamiento reanudado no se puede reproducir desde la semilla: los entornos
        # arrancan de cero en cada reanudacion. Queda escrito en el informe.
        "reanudaciones": reanudaciones,
        "mejor_val": best,
        "test": test,
        "test_muestreada": test_muestreada,
        "historia": history,
    }


def load_wirings(cfg: dict[str, Any]) -> dict[str, Wiring]:
    directory = Path(cfg["wiring_dir"])
    return {name: Wiring.load(directory / f"{name}.npz") for name in ("mosca", "aleatorio", "recableado")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", default=None, help="entrenar solo este modelo")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--iterations", default=None, type=int,
                        help="acorta ppo.iterations; para medir el coste en una maquina nueva")
    parser.add_argument("--desde", default=None, type=Path,
                        help="arranca de estos pesos (formato de checkpoint de train_rl.py);"
                             " la salida va a <output_dir>/desde_<fichero>/. Un solo modelo")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.desde is not None:
        if args.model is None or args.seed is None:
            parser.error("--desde exige --model y --seed")
        cfg["output_dir"] = str(Path(cfg["output_dir"]) / f"desde_{args.desde.parent.name}")
    if args.iterations is not None:
        cfg["ppo"]["iterations"] = args.iterations
        cfg["output_dir"] = str(Path(cfg["output_dir"]) / f"medicion_{args.iterations}it")
    device = resolve_device(args.device or cfg.get("device", "auto"))
    print(f"dispositivo: {device}")
    wirings = load_wirings(cfg)

    models = [args.model] if args.model else list(cfg["models"])
    seeds = [args.seed] if args.seed is not None else list(cfg["seeds"])
    results = [
        train_one(name, int(seed), cfg, wirings, device, args.desde)
        for name in models for seed in seeds
    ]

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "".join(part for part in (
        f"_{args.model}" if args.model else "",
        f"_s{args.seed}" if args.seed is not None else "",
    ))
    report_path = out_dir / f"rl_report{suffix}.json"
    report_path.write_text(json.dumps(
        {"config": cfg, "dispositivo": str(device), "torch": torch.__version__,
         "resultados": results},
        indent=2, ensure_ascii=False,
    ))
    print(f"\ninforme en {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
