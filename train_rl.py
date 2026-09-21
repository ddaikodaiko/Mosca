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
        from envs.nes_mario import NesConfig, VectorNesMario

        return VectorNesMario(
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
    )


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
        values[t] = value

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
    advantages = torch.zeros_like(rewards)
    accumulator = torch.zeros(n_envs, device=device)
    next_value = last_value
    for t in reversed(range(steps)):
        delta = rewards[t] + gamma * next_value * alive[t] - values[t]
        accumulator = delta + gamma * gae_lambda * alive[t] * accumulator
        advantages[t] = accumulator
        next_value = values[t]

    statistics = {
        "episodios": float(len(episodes)),
        "recompensa_media": float(rewards.sum().item() / max(len(episodes), 1)),
        "progreso_medio": float(np.mean([e["progreso"] for e in episodes])) if episodes else float("nan"),
        "tasa_meta": float(np.mean([e["meta"] for e in episodes])) if episodes else float("nan"),
    }
    return Rollout(
        observations, hidden, actions, log_probs, values, advantages, advantages + values
    ), statistics


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
    policy: nn.Module, cfg: dict[str, Any], split: str, n_episodes: int, device: torch.device
) -> dict[str, float]:
    """Politica codiciosa sobre niveles de `split`. Semillas disjuntas de las de train."""
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
        action = logits.argmax(dim=1)
        next_obs, _, done, infos = envs.step(action.cpu().numpy())
        for i, done_i in enumerate(done):
            if done_i:
                finished.append(infos[i])
        done_mask = torch.as_tensor(done, device=device)
        h = torch.where(done_mask[:, None], torch.zeros_like(h), h)
        obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    finished = finished[:n_episodes]
    return {
        "episodios": float(len(finished)),
        "progreso": float(np.mean([e["progreso"] for e in finished])) if finished else float("nan"),
        "tasa_meta": float(np.mean([e["meta"] for e in finished])) if finished else float("nan"),
        "tasa_muerte": float(np.mean([e["muerte"] for e in finished])) if finished else float("nan"),
        "pisados": float(np.mean([e["pisados"] for e in finished])) if finished else float("nan"),
    }


def train_one(
    name: str, seed: int, cfg: dict[str, Any], wirings: dict[str, Wiring], device: torch.device
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
    if cfg.get("normalise_observations", True):
        muestra = muestrea_observaciones(envs, state, int(cfg.get("normalise_samples", 2000)))
        policy.fit_feature_statistics(muestra)
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

    def guarda() -> None:
        """Se guarda en cada evaluacion, no solo al final: un entrenamiento de horas
        tiene que poder mirarse mientras corre, y cortarse sin perderlo todo."""
        torch.save(
            {"modelo": name, "semilla": seed, "config": cfg,
             "state_dict": policy.state_dict()}, checkpoint,
        )
    print(f"\n[{name} s{seed}] {policy.n_neurons} neuronas, {n_parameters:,} parametros")
    for iteration in range(total_iterations):
        batch, statistics = collect(
            policy, envs, state, int(ppo["rollout_steps"]),
            float(ppo["gamma"]), float(ppo["gae_lambda"]), device,
        )
        losses = ppo_update(policy, optimiser, batch, ppo, generator)
        scheduler.step()
        entry = {"iteracion": iteration, **statistics, **losses}
        if (iteration + 1) % int(cfg.get("eval_every", 10)) == 0 or iteration == total_iterations - 1:
            entry["val"] = evaluate(policy, cfg, "val", int(cfg.get("eval_episodes", 16)), device)
            guarda()
            if entry["val"]["progreso"] > best["progreso"]:
                best = dict(entry["val"], iteracion=iteration)
            print(
                f"  iter {iteration + 1:4d}/{total_iterations}"
                f"  train progreso {statistics['progreso_medio']:.3f}"
                f"  val progreso {entry['val']['progreso']:.3f}"
                f"  meta {entry['val']['tasa_meta']:.2f}"
                f"  H {losses['entropia']:.3f}"
            )
        history.append(entry)

    guarda()
    test = evaluate(policy, cfg, "test", int(cfg.get("test_episodes", 32)), device)
    return {
        "modelo": name,
        "semilla": seed,
        "neuronas": int(policy.n_neurons),
        "parametros": int(n_parameters),
        "segundos": time.time() - started,
        "mejor_val": best,
        "test": test,
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
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = resolve_device(args.device or cfg.get("device", "auto"))
    print(f"dispositivo: {device}")
    wirings = load_wirings(cfg)

    models = [args.model] if args.model else list(cfg["models"])
    seeds = [args.seed] if args.seed is not None else list(cfg["seeds"])
    results = [
        train_one(name, int(seed), cfg, wirings, device) for name in models for seed in seeds
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
