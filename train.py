"""Entrenamiento de los seis modelos (fase 4).

Uso:
    python train.py --config configs/entrenamiento_32_proto.yaml
    python train.py --config configs/entrenamiento_64.yaml --model mosca --seed 0

Dos fases, como pide el encargo:

  Fase A  prediccion a un paso, perdida L2 relativa, Adam con decaimiento coseno.
  Fase B  ajuste con despliegue, currículo 5 -> 10 -> 20 pasos, retropropagacion a
          traves del tiempo.

Recorte de gradiente a norma 1,0 y parada temprana por validacion en las dos. Todo lo
que se ejecuta queda registrado en el JSON de salida: semilla, config, metricas por
epoca y tiempos. Sin eso los resultados no valen.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml
from torch import Tensor, nn

from data.spectral import resolve_device
from models.fno import FNO2d, Persistence
from models.masked_rnn import ConnectomeFlowModel, build_wiring


def relative_l2(prediction: Tensor, target: Tensor) -> Tensor:
    """Error L2 relativo por muestra, promediado. Es la perdida y tambien la metrica."""
    difference = (prediction - target).flatten(1).norm(dim=1)
    scale = target.flatten(1).norm(dim=1).clamp_min(1e-12)
    return (difference / scale).mean()


@dataclass
class Augmenter:
    """Aumento de datos por las simetrias EXACTAS del sistema simulado.

    El dominio es [0,2pi]^2 periodico, asi que trasladar en x es una simetria exacta de
    Navier-Stokes. En y no lo es en general, porque el forzado `sin(k_f y)` rompe la
    homogeneidad, pero si lo son las traslaciones de un periodo del forzado: multiplos
    de `grid / k_f` casillas. Trasladar el campo de entrada y el objetivo por lo mismo
    da un par valido que el simulador podria haber producido.

    No es un truco de regularizacion: son datos reales que no habiamos usado. Y se
    aplica identico a los seis modelos, asi que la comparacion sigue siendo justa.
    """

    enabled: bool
    forcing_wavenumber: int
    continuous_x: bool = False

    def __call__(self, *tensors: Tensor, generator: torch.Generator) -> tuple[Tensor, ...]:
        if not self.enabled:
            return tensors
        reference = tensors[0]
        grid = reference.shape[-1]
        batch = reference.shape[0]
        device = reference.device
        step_y = max(1, grid // max(self.forcing_wavenumber, 1))

        # Ejes como en data/spectral.py: kx va en la dimension -2 (filas) y ky, la del
        # forzado sin(k_f y), en la -1 (columnas). Hasta el 16-09-2026 estaban cambiados:
        # la traslacion libre caia sobre el eje forzado y los pares tenian un 4 % de
        # error fisico (verificado con el simulador en tests/test_masked_rnn.py).
        axis = torch.arange(grid, device=device)
        # Mismo orden de sorteo que antes (x y luego y): con continuous_x apagado, las
        # secuencias aleatorias son identicas a las de los entrenamientos ya hechos.
        if self.continuous_x:
            # Traslacion por fase de Fourier, en fracciones de casilla. Exacta para campos
            # de banda limitada; el simulador desaliasa por encima de n/3, asi que lo es.
            shift_x = torch.rand(batch, generator=generator, device=device).double() * grid
            # Fase en float64: en float32 el error de redondeo ya es 1e-7 con saltos enteros.
            k = torch.fft.fftfreq(grid, d=1.0 / grid, device=device, dtype=torch.float64)
            if grid % 2 == 0:
                k[grid // 2] = 0.0  # Nyquist: sin signo definido, y nulo tras desaliasar
            phase = torch.exp(-2j * torch.pi * k[None, :] * shift_x[:, None] / grid)
        else:
            shift_x = torch.randint(0, grid, (batch,), generator=generator, device=device)
            index_x = (axis[None, :] - shift_x[:, None]) % grid
        shift_y = torch.randint(0, self.forcing_wavenumber, (batch,), generator=generator,
                                device=device) * step_y
        index_y = (axis[None, :] - shift_y[:, None]) % grid

        rolled = []
        for tensor in tensors:
            extra = tensor.shape[1:-2]  # el horizonte de la fase B, si lo hay
            view = tensor.reshape(batch, -1, grid, grid)
            if self.continuous_x:
                spectrum = torch.fft.fft(view, dim=2) * phase[:, None, :, None]
                view = torch.fft.ifft(spectrum, dim=2).real.to(tensor.dtype)
            else:
                ix = index_x[:, None, :, None].expand(-1, view.shape[1], -1, grid)
                view = view.gather(2, ix)
            iy = index_y[:, None, None, :].expand(-1, view.shape[1], grid, -1)
            view = view.gather(3, iy)
            rolled.append(view.reshape(batch, *extra, grid, grid))
        return tuple(rolled)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Augmenter":
        section = cfg.get("augment", {})
        return cls(
            enabled=bool(section.get("enabled", False)),
            forcing_wavenumber=int(section.get("forcing_wavenumber", 4)),
            continuous_x=bool(section.get("continuous_x", False)),
        )


@dataclass
class Trajectories:
    """Un split cargado en memoria: (n_traj, n_save, grid, grid)."""

    omega: Tensor
    seeds: np.ndarray

    @classmethod
    def load(cls, path: Path, device: torch.device, max_traj: int | None = None) -> "Trajectories":
        data = np.load(path)
        omega = data["omega"]
        if max_traj is not None:
            omega = omega[:max_traj]
        return cls(
            omega=torch.from_numpy(np.ascontiguousarray(omega)).to(device),
            seeds=data["seeds"][: omega.shape[0]],
        )

    @property
    def grid(self) -> int:
        return int(self.omega.shape[-1])

    def pairs(self) -> tuple[Tensor, Tensor]:
        """Todos los pares (omega_t, omega_{t+1}) aplanados, para la fase A."""
        return self.omega[:, :-1].flatten(0, 1), self.omega[:, 1:].flatten(0, 1)

    def windows(self, horizon: int, generator: torch.Generator, batch: int) -> tuple[Tensor, Tensor]:
        """Un lote de ventanas (inicio, siguientes `horizon` pasos), para la fase B."""
        n_traj, n_save = self.omega.shape[:2]
        traj = torch.randint(0, n_traj, (batch,), generator=generator, device=self.omega.device)
        start = torch.randint(
            0, n_save - horizon, (batch,), generator=generator, device=self.omega.device
        )
        offsets = torch.arange(horizon + 1, device=self.omega.device)
        index = start[:, None] + offsets[None, :]
        window = self.omega[traj[:, None], index]
        return window[:, 0], window[:, 1:]


def build_model(name: str, cfg: dict[str, Any], grid: int, seed: int) -> nn.Module:
    """Los seis modelos de la seccion 7. Solo cambia la mascara (o la arquitectura)."""
    if name == "persistencia":
        return Persistence()
    if name == "fno":
        fno = cfg["fno"]
        return FNO2d(
            modes=int(fno["modes"]),
            width=int(fno["width"]),
            n_layers=int(fno["n_layers"]),
            grid=grid,
            output_scale=float(cfg["core"]["output_scale"]),
        )
    wiring = build_wiring({"wiring": name, "wiring_dir": cfg["wiring_dir"]}, seed)
    port_wiring = (
        build_wiring({"wiring": "mosca", "wiring_dir": cfg["wiring_dir"]}, seed)
        if name == "sin_nucleo" else None
    )
    encoder, core = cfg["encoder"], cfg["core"]
    return ConnectomeFlowModel(
        wiring,
        grid=grid,
        port_wiring=port_wiring,
        n_complex_modes=int(encoder["n_complex_modes"]),
        n_sensory=int(encoder["n_sensory"]),
        n_motor=int(encoder["n_motor"]),
        n_shared=int(encoder["n_shared"]),
        n_internal_steps=int(core["n_internal_steps"]),
        spectral_radius=float(core["spectral_radius"]),
        leak=float(core["leak"]),
        output_scale=float(core["output_scale"]),
        normalisation=str(core.get("normalisation", "max_eig")),
    )


def predict_one_step(model: nn.Module, omega: Tensor) -> Tensor:
    output = model(omega)
    return output[0] if isinstance(output, tuple) else output


def rollout(model: nn.Module, omega: Tensor, steps: int) -> Tensor:
    """Despliegue autoregresivo comun a todos los modelos: (batch, steps, grid, grid)."""
    return model.rollout(omega, steps)


@torch.no_grad()
def evaluate_one_step(model: nn.Module, split: Trajectories, batch: int) -> float:
    model.eval()
    inputs, targets = split.pairs()
    total, count = 0.0, 0
    for start in range(0, inputs.shape[0], batch):
        chunk = slice(start, start + batch)
        value = relative_l2(predict_one_step(model, inputs[chunk]), targets[chunk])
        total += float(value) * (inputs[chunk].shape[0])
        count += inputs[chunk].shape[0]
    model.train()
    return total / max(count, 1)


@torch.no_grad()
def evaluate_rollout(model: nn.Module, split: Trajectories, steps: int, batch: int) -> float:
    model.eval()
    start = split.omega[:, 0]
    target = split.omega[:, 1 : steps + 1]
    total, count = 0.0, 0
    for begin in range(0, start.shape[0], batch):
        chunk = slice(begin, begin + batch)
        predicted = rollout(model, start[chunk], steps)
        value = relative_l2(predicted.flatten(0, 1), target[chunk].flatten(0, 1))
        total += float(value) * predicted.shape[0]
        count += predicted.shape[0]
    model.train()
    return total / max(count, 1)


@dataclass
class EarlyStopping:
    """Se para cuando la validacion deja de mejorar. Guarda el mejor estado."""

    patience: int
    best: float = math.inf
    best_state: dict[str, Tensor] = field(default_factory=dict)
    waited: int = 0

    def update(self, value: float, model: nn.Module) -> bool:
        if value < self.best - 1e-6:
            self.best = value
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            self.waited = 0
            return True
        self.waited += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.waited >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state:
            model.load_state_dict(self.best_state)


@dataclass
class Reanudacion:
    """Guarda el entrenamiento a medias para poder seguirlo tras un corte.

    Un entrenamiento de la fase 4 son horas; un reinicio, una actualizacion o un Ctrl-C
    los tiraba enteros. Se guarda cada `cada_pasos` pasos el estado COMPLETO: pesos,
    optimizador, planificador, la parada temprana con su mejor estado, el registro, y
    **el estado del generador aleatorio**. Eso ultimo no es un detalle: sin el, al
    reanudar cambia el orden de los lotes y el entrenamiento deja de ser el que la
    semilla dice que es. Con el, reanudar da exactamente los mismos pesos que no haber
    parado, y hay un test que lo comprueba bit a bit.

    El fichero se escribe en un temporal y se renombra, asi que un corte en mitad del
    guardado no deja un checkpoint corrupto. Al terminar bien, se borra.
    """

    ruta: Path
    cada_pasos: int
    estado: dict[str, Any] | None = None

    @classmethod
    def para(cls, cfg: dict[str, Any], nombre: str, semilla: int) -> "Reanudacion":
        directorio = Path(cfg["output_dir"]) / "checkpoints"
        directorio.mkdir(parents=True, exist_ok=True)
        ruta = directorio / f"{nombre}_s{semilla}.reanudar.pt"
        cada = int(cfg.get("resume_every", 0))
        estado = None
        if cada > 0 and ruta.is_file():
            estado = torch.load(ruta, map_location="cpu", weights_only=False)
            print(f"    reanudando desde la fase {estado['fase']}, paso {estado['paso']}"
                  f" (etapa {estado['etapa']})")
        return cls(ruta=ruta, cada_pasos=cada, estado=estado)

    @property
    def activa(self) -> bool:
        return self.cada_pasos > 0

    def toca_guardar(self, paso: int) -> bool:
        return self.activa and paso % self.cada_pasos == 0

    def guarda(self, **estado: Any) -> None:
        temporal = self.ruta.with_suffix(".pt.part")
        torch.save(estado, temporal)
        temporal.replace(self.ruta)

    def consume(self, fase: str, etapa: int) -> dict[str, Any] | None:
        """Devuelve el estado si corresponde a esta fase y etapa, y lo marca como usado."""
        if self.estado is None:
            return None
        if self.estado["fase"] != fase or self.estado["etapa"] != etapa:
            return None
        estado, self.estado = self.estado, None
        return estado

    def ya_pasada(self, fase: str, etapa: int) -> bool:
        """True si el estado guardado va por delante de esta fase o etapa: se salta."""
        if self.estado is None:
            return False
        if fase == "A":
            return self.estado["fase"] == "B"
        return self.estado["fase"] == "B" and self.estado["etapa"] > etapa

    def termina(self) -> None:
        self.ruta.unlink(missing_ok=True)


def cosine_schedule(optimiser: torch.optim.Optimizer, total_steps: int) -> Any:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(total_steps, 1))


def instantanea(
    fase: str, etapa: int, paso: int, model: nn.Module, optimiser: torch.optim.Optimizer,
    scheduler: Any, stopper: EarlyStopping, generator: torch.Generator,
    log: list[dict[str, Any]],
) -> dict[str, Any]:
    """Todo lo que hace falta para seguir exactamente donde se dejo."""
    return {
        "fase": fase,
        "etapa": etapa,
        "paso": paso,
        "modelo": model.state_dict(),
        "optimizador": optimiser.state_dict(),
        "planificador": scheduler.state_dict(),
        "parada": {"best": stopper.best, "best_state": stopper.best_state,
                   "waited": stopper.waited},
        "generador": generator.get_state(),
        "log": log,
    }


def restaura(
    model: nn.Module, optimiser: torch.optim.Optimizer, scheduler: Any,
    stopper: EarlyStopping, generator: torch.Generator, log: list[dict[str, Any]],
    estado: dict[str, Any],
) -> int:
    """Deja todo como estaba y devuelve el paso por el que hay que seguir."""
    model.load_state_dict(estado["modelo"])
    optimiser.load_state_dict(estado["optimizador"])
    scheduler.load_state_dict(estado["planificador"])
    stopper.best = estado["parada"]["best"]
    stopper.best_state = estado["parada"]["best_state"]
    stopper.waited = estado["parada"]["waited"]
    generator.set_state(estado["generador"])
    log.clear()
    log.extend(estado["log"])
    return int(estado["paso"]) + 1


def train_phase_a(
    model: nn.Module,
    train: Trajectories,
    val: Trajectories,
    cfg: dict[str, Any],
    generator: torch.Generator,
    augmenter: Augmenter,
    log: list[dict[str, Any]],
    reanuda: Reanudacion | None = None,
) -> None:
    phase = cfg["phase_a"]
    steps = int(phase["steps"])
    batch = int(phase["batch_size"])
    optimiser = torch.optim.Adam(model.parameters(), lr=float(phase["lr"]))
    scheduler = cosine_schedule(optimiser, steps)
    stopper = EarlyStopping(patience=int(phase["patience"]))

    primer_paso = 1
    if reanuda is not None:
        if reanuda.ya_pasada("A", 0):
            return                      # el corte fue ya en la fase B
        guardado = reanuda.consume("A", 0)
        if guardado is not None:
            primer_paso = restaura(model, optimiser, scheduler, stopper, generator, log, guardado)

    inputs, targets = train.pairs()
    every = int(phase["eval_every"])
    t0 = time.perf_counter()
    for step in range(primer_paso, steps + 1):
        index = torch.randint(0, inputs.shape[0], (batch,), generator=generator, device=inputs.device)
        source, target = augmenter(inputs[index], targets[index], generator=generator)
        optimiser.zero_grad()
        loss = relative_l2(predict_one_step(model, source), target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
        optimiser.step()
        scheduler.step()

        if step % every == 0 or step == steps:
            val_error = evaluate_one_step(model, val, batch)
            improved = stopper.update(val_error, model)
            log.append({
                "fase": "A", "paso": step, "perdida": float(loss),
                "val_1paso": val_error, "mejora": improved,
                "lr": scheduler.get_last_lr()[0], "segundos": round(time.perf_counter() - t0, 1),
            })
            print(f"    A {step:5d}/{steps}  perdida {float(loss):.4f}  val {val_error:.4f}"
                  f"{'  *' if improved else ''}")
            if stopper.should_stop:
                print(f"    A parada temprana en el paso {step}")
                break
        if reanuda is not None and reanuda.toca_guardar(step):
            reanuda.guarda(**instantanea("A", 0, step, model, optimiser, scheduler,
                                         stopper, generator, log))
    stopper.restore(model)


def train_phase_b(
    model: nn.Module,
    train: Trajectories,
    val: Trajectories,
    cfg: dict[str, Any],
    generator: torch.Generator,
    augmenter: Augmenter,
    log: list[dict[str, Any]],
    reanuda: Reanudacion | None = None,
) -> None:
    phase = cfg["phase_b"]
    batch = int(phase["batch_size"])
    t0 = time.perf_counter()
    for etapa, horizon in enumerate(phase["curriculum"]):
        horizon = int(horizon)
        steps = int(phase["steps_per_stage"])
        optimiser = torch.optim.Adam(model.parameters(), lr=float(phase["lr"]))
        scheduler = cosine_schedule(optimiser, steps)
        stopper = EarlyStopping(patience=int(phase["patience"]))
        primer_paso = 1
        if reanuda is not None:
            if reanuda.ya_pasada("B", etapa):
                continue                # el corte fue en una etapa posterior
            guardado = reanuda.consume("B", etapa)
            if guardado is not None:
                primer_paso = restaura(model, optimiser, scheduler, stopper, generator,
                                       log, guardado)
        for step in range(primer_paso, steps + 1):
            start, target = train.windows(horizon, generator, batch)
            start, target = augmenter(start, target, generator=generator)
            optimiser.zero_grad()
            predicted = rollout(model, start, horizon)
            loss = relative_l2(predicted.flatten(0, 1), target.flatten(0, 1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optimiser.step()
            scheduler.step()

            if step % int(phase["eval_every"]) == 0 or step == steps:
                val_error = evaluate_rollout(model, val, horizon, batch)
                improved = stopper.update(val_error, model)
                log.append({
                    "fase": "B", "horizonte": horizon, "paso": step, "perdida": float(loss),
                    f"val_despliegue_{horizon}": val_error, "mejora": improved,
                    "segundos": round(time.perf_counter() - t0, 1),
                })
                print(f"    B h={horizon:2d} {step:5d}/{steps}  perdida {float(loss):.4f}"
                      f"  val {val_error:.4f}{'  *' if improved else ''}")
                if stopper.should_stop:
                    print(f"    B h={horizon} parada temprana en el paso {step}")
                    break
            if reanuda is not None and reanuda.toca_guardar(step):
                reanuda.guarda(**instantanea("B", etapa, step, model, optimiser, scheduler,
                                             stopper, generator, log))
        stopper.restore(model)


def train_one(
    name: str, seed: int, cfg: dict[str, Any], splits: dict[str, Trajectories], device: torch.device
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    grid = splits["train"].grid
    model = build_model(name, cfg, grid, seed).to(device)
    if hasattr(model, "fit_feature_statistics"):
        # Normalizacion de la entrada medida SOLO sobre entrenamiento, antes de entrenar.
        # Sin esto el impulso llega a tanh con rms ~2 y las sensoriales saturan de salida.
        sample, _ = splits["train"].pairs()
        model.fit_feature_statistics(sample[: int(cfg.get("stats_sample", 2048))])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [{name} semilla {seed}]  {n_params} parametros entrenables")

    log: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    if n_params > 0:
        augmenter = Augmenter.from_config(cfg)
        reanuda = Reanudacion.para(cfg, name, seed)
        if reanuda.estado is not None:
            # Los pesos del corte, antes de decidir por que fase se sigue.
            model.load_state_dict(reanuda.estado["modelo"])
        train_phase_a(model, splits["train"], splits["val"], cfg, generator, augmenter, log,
                      reanuda)
        if cfg["phase_b"]["curriculum"]:
            train_phase_b(model, splits["train"], splits["val"], cfg, generator, augmenter,
                          log, reanuda)
        reanuda.termina()
    seconds = time.perf_counter() - t0

    batch = int(cfg["phase_a"]["batch_size"])
    result = {
        "modelo": name,
        "semilla": seed,
        "parametros": n_params,
        "val_1paso": evaluate_one_step(model, splits["val"], batch),
        "segundos": round(seconds, 1),
        "log": log,
    }
    for steps in cfg["eval_rollouts"]:
        steps = int(steps)
        if splits["val"].omega.shape[1] > steps:
            result[f"val_despliegue_{steps}"] = evaluate_rollout(model, splits["val"], steps, batch)

    out_dir = Path(cfg["output_dir"])
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if n_params > 0:
        torch.save(
            {"state_dict": model.state_dict(), "modelo": name, "semilla": seed, "config": cfg},
            out_dir / "checkpoints" / f"{name}_s{seed}.pt",
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", default=None, help="entrenar solo este modelo")
    parser.add_argument("--seed", default=None, type=int, help="entrenar solo esta semilla")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = resolve_device(args.device or cfg.get("device", "auto"))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"dispositivo: {device}")

    data_dir = Path(cfg["data_dir"])
    limits = cfg.get("max_trajectories", {})
    splits = {
        split: Trajectories.load(data_dir / f"{split}.npz", device, limits.get(split))
        for split in ("train", "val")
    }
    print(f"datos: train {tuple(splits['train'].omega.shape)}  val {tuple(splits['val'].omega.shape)}")

    models = [args.model] if args.model else list(cfg["models"])
    seeds = [args.seed] if args.seed is not None else list(cfg["seeds"])

    results = []
    for name in models:
        for seed in (seeds if name != "persistencia" else seeds[:1]):
            results.append(train_one(name, int(seed), cfg, splits, device))
            print(f"  -> val 1 paso {results[-1]['val_1paso']:.4f}"
                  f"   ({results[-1]['segundos']:.0f} s)")

    report = {
        "config": cfg,
        "dispositivo": str(device),
        "torch": torch.__version__,
        "resultados": results,
    }
    # Al trocear por modelo o semilla, cada trozo escribe su informe y no pisa los demas.
    suffix = "".join(
        part for part in (
            f"_{args.model}" if args.model else "",
            f"_s{args.seed}" if args.seed is not None else "",
        )
    )
    report_path = out_dir / f"train_report{suffix}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\ninforme en {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
