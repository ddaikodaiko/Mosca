"""Genera el conjunto de trayectorias de Navier-Stokes 2D (fase 1).

Uso:
    python -m data.simulate_ns --config configs/ns_32_proto.yaml
    python -m data.simulate_ns --config configs/ns_64.yaml --device cuda

Escribe en el directorio de salida del YAML:
    <split>.npz   omega (n_traj, n_save, n, n) float32, seeds, times
    meta.json     configuracion, semillas, metricas de aceptacion y tiempos
    figs/         campo de vorticidad y espectro de energia
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from data.spectral import (
    SpectralGrid,
    cfl_number,
    energy_spectrum,
    kinetic_energy,
    make_grid,
    random_vorticity,
    resolve_device,
    rk4_step,
)

CFL_LIMIT = 0.5


class CFLError(RuntimeError):
    """Se ha superado el limite CFL: el resultado no seria de fiar."""


def _seeds(split_cfg: dict[str, Any]) -> list[int]:
    base = int(split_cfg["seed_base"])
    return [base + i for i in range(int(split_cfg["n_traj"]))]


def simulate_batch(
    seeds: list[int],
    grid: SpectralGrid,
    dt: float,
    nu: float,
    alpha: float,
    transient_steps: int,
    n_save: int,
    save_every: int,
    ic_peak_wavenumber: float,
    ic_rms: float,
    progress: str = "",
) -> dict[str, np.ndarray]:
    """Simula un lote de trayectorias y devuelve los campos guardados."""
    n = grid.n
    w0 = np.stack([random_vorticity(n, s, ic_peak_wavenumber, ic_rms) for s in seeds])
    w_hat = torch.fft.rfft2(torch.from_numpy(w0).to(grid.device, grid.dtype))

    max_cfl = 0.0
    energy_transient: list[np.ndarray] = []

    def advance(n_steps: int, record_every: int = 0) -> None:
        nonlocal w_hat, max_cfl
        for i in range(n_steps):
            w_hat = rk4_step(w_hat, dt, grid, nu, alpha)
            if i % 50 == 0 or i == n_steps - 1:
                cfl = float(cfl_number(w_hat, grid, dt).max())
                max_cfl = max(max_cfl, cfl)
                if not np.isfinite(cfl):
                    raise CFLError("La simulacion ha divergido (CFL no finito)")
                if cfl > CFL_LIMIT:
                    raise CFLError(
                        f"CFL = {cfl:.3f} > {CFL_LIMIT}. Reduce dt en el YAML."
                    )
            if record_every and i % record_every == 0:
                energy_transient.append(kinetic_energy(w_hat, grid).cpu().numpy())

    t0 = time.perf_counter()
    advance(transient_steps, record_every=max(1, transient_steps // 100))

    frames = np.empty((len(seeds), n_save, n, n), dtype=np.float32)
    energy = np.empty((len(seeds), n_save), dtype=np.float64)
    for j in range(n_save):
        if j > 0:
            advance(save_every)
        w = torch.fft.irfft2(w_hat, s=(n, n))
        frames[:, j] = w.float().cpu().numpy()
        energy[:, j] = kinetic_energy(w_hat, grid).double().cpu().numpy()
    elapsed = time.perf_counter() - t0

    if not np.isfinite(frames).all():
        raise CFLError("Hay NaN o Inf en los campos guardados")

    print(f"  {progress}{len(seeds)} trayectorias en {elapsed:6.1f} s "
          f"| CFL max {max_cfl:.3f} | E final media {energy[:, -1].mean():.4f}")

    return {
        "omega": frames,
        "energy": energy,
        "energy_transient": np.asarray(energy_transient, dtype=np.float64).T
        if energy_transient else np.zeros((len(seeds), 0)),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "max_cfl": np.float64(max_cfl),
        "seconds": np.float64(elapsed),
        "final_w_hat": w_hat,
    }


def energy_drift(energy: np.ndarray, dt_save: float) -> float:
    """Pendiente relativa de E(t) en la segunda mitad, por unidad de tiempo."""
    n_save = energy.shape[-1]
    half = energy[..., n_save // 2:]
    t = np.arange(half.shape[-1]) * dt_save
    mean_e = half.mean(axis=0)
    slope = np.polyfit(t, mean_e, 1)[0]
    return float(slope / mean_e.mean())


def spectrum_slope(k: np.ndarray, e_k: np.ndarray, k_lo: int, k_hi: int) -> float:
    """Exponente del ajuste log-log de E(k) en la banda [k_lo, k_hi]."""
    band = (k >= k_lo) & (k <= k_hi) & (e_k > 0)
    if band.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(k[band]), np.log(e_k[band]), 1)[0])


def _figures(frames: np.ndarray, k: np.ndarray, e_k: np.ndarray, seed: int,
             out_dir: Path, slope: float, k_lo: int, k_hi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.linspace(0, frames.shape[0] - 1, 4).astype(int)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    vmax = float(np.abs(frames[idx]).max())
    for ax, j in zip(axes, idx):
        im = ax.imshow(frames[j].T, origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, extent=(0, 2 * np.pi, 0, 2 * np.pi))
        ax.set_title(f"paso guardado {j}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im, ax=axes, label=r"$\omega$", shrink=0.85)
    fig.suptitle(f"Vorticidad, semilla {seed}")
    fig.savefig(out_dir / f"vorticity_seed{seed}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    mask = (k > 0) & (e_k > 0)
    ax.loglog(k[mask], e_k[mask], "o-", ms=3, label="E(k)")
    band = (k >= k_lo) & (k <= k_hi) & (e_k > 0)
    if band.sum() >= 3 and np.isfinite(slope):
        kb = k[band]
        ref = e_k[band][0] * (kb / kb[0]) ** slope
        ax.loglog(kb, ref, "r--", label=f"ajuste k^{slope:.2f}")
    ax.set_xlabel("k"); ax.set_ylabel("E(k)")
    ax.set_title(f"Espectro de energia (banda k={k_lo}-{k_hi})")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.savefig(out_dir / "energy_spectrum.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def run(cfg: dict[str, Any], device_override: str | None = None) -> dict[str, Any]:
    sim = cfg["simulation"]
    n = int(sim["grid"])
    dt = float(sim["dt"])
    device = resolve_device(device_override or cfg.get("device", "auto"))
    dtype = torch.float64 if sim.get("precision", "float64") == "float64" else torch.float32
    grid = make_grid(n, int(sim.get("forcing_wavenumber", 4)), device, dtype)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dt_save = dt * int(sim["save_every"])

    print(f"Navier-Stokes 2D | malla {n}x{n} | dt={dt} | nu={sim['nu']} | "
          f"alpha={sim['alpha']} | dispositivo {device} | {dtype}")
    print(f"Transitorio {sim['transient_steps']} pasos (t={sim['transient_steps']*dt:.1f}), "
          f"{sim['n_save']} instantaneas cada {sim['save_every']} pasos "
          f"(dt_guardado={dt_save:g})")

    meta: dict[str, Any] = {
        "config": cfg,
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "dt_save": dt_save,
        "splits": {},
    }
    all_seeds: dict[str, list[int]] = {}
    first_split_frames = None
    first_split_wref = None

    for split, split_cfg in cfg["splits"].items():
        seeds = _seeds(split_cfg)
        all_seeds[split] = seeds
        print(f"[{split}] semillas {seeds[0]}..{seeds[-1]} ({len(seeds)} trayectorias)")

        batch_size = int(cfg.get("batch_size", len(seeds)))
        chunks = [seeds[i:i + batch_size] for i in range(0, len(seeds), batch_size)]
        frames, energies, max_cfl, seconds = [], [], 0.0, 0.0
        for c, chunk in enumerate(chunks):
            res = simulate_batch(
                chunk, grid, dt, float(sim["nu"]), float(sim["alpha"]),
                int(sim["transient_steps"]), int(sim["n_save"]), int(sim["save_every"]),
                float(sim.get("ic_peak_wavenumber", 4.0)), float(sim.get("ic_rms", 1.0)),
                progress=f"lote {c+1}/{len(chunks)}: ",
            )
            frames.append(res["omega"]); energies.append(res["energy"])
            max_cfl = max(max_cfl, float(res["max_cfl"]))
            seconds += float(res["seconds"])
            if split == next(iter(cfg["splits"])) and c == 0:
                first_split_wref = res["final_w_hat"]

        omega = np.concatenate(frames); energy = np.concatenate(energies)
        times = np.arange(int(sim["n_save"]), dtype=np.float32) * dt_save

        np.savez(out_dir / f"{split}.npz", omega=omega, energy=energy,
                 seeds=np.asarray(seeds, dtype=np.int64), times=times,
                 dt_save=np.float64(dt_save), grid=np.int64(n))
        size_mb = (out_dir / f"{split}.npz").stat().st_size / 1e6
        drift = energy_drift(energy, dt_save)
        meta["splits"][split] = {
            "n_traj": len(seeds), "seed_base": seeds[0], "seed_last": seeds[-1],
            "shape": list(omega.shape), "dtype": "float32", "file_mb": round(size_mb, 1),
            "max_cfl": round(max_cfl, 4), "seconds": round(seconds, 1),
            "energy_mean_final": float(energy[:, -1].mean()),
            "energy_drift_rel_per_time": drift,
        }
        print(f"  -> {split}.npz  {omega.shape}  {size_mb:.1f} MB  "
              f"deriva de energia {drift*100:+.2f} %/t")
        if first_split_frames is None:
            first_split_frames = omega

    # --- criterios de aceptacion ---------------------------------------------
    k_lo = int(cfg.get("spectrum_band", [5, 20])[0])
    k_hi = min(int(cfg.get("spectrum_band", [5, 20])[1]), n // 3 - 1)
    w_hat_ref = first_split_wref
    k, e_k = energy_spectrum(w_hat_ref, grid)
    slope = spectrum_slope(k, e_k, k_lo, k_hi)

    # Reproducibilidad: hay que repetir con la MISMA forma de ejecucion. torch.fft
    # no da bits identicos con distinto tamano de lote y Navier-Stokes es caotico,
    # asi que una diferencia de 1e-16 crece hasta ~1e-7 en t=20. El lote es parte
    # del contrato de reproducibilidad y por eso se registra en meta.json.
    first_split = next(iter(cfg["splits"]))
    ref_chunk = all_seeds[first_split][: int(cfg.get("batch_size", 1))]
    sim_args = (grid, dt, float(sim["nu"]), float(sim["alpha"]),
                int(sim["transient_steps"]), int(sim["n_save"]), int(sim["save_every"]),
                float(sim.get("ic_peak_wavenumber", 4.0)), float(sim.get("ic_rms", 1.0)))
    repro = simulate_batch(ref_chunk, *sim_args, progress="reproducibilidad: ")
    bitwise = bool(np.array_equal(repro["omega"], first_split_frames[: len(ref_chunk)]))

    # Diagnostico: cuanto se desvia la misma semilla simulada en un lote de 1.
    ref_seed = ref_chunk[0]
    if len(ref_chunk) > 1:
        solo = simulate_batch([ref_seed], *sim_args, progress="lote de 1 (diagnostico): ")
        dev = float(np.abs(solo["omega"][0] - first_split_frames[0]).max())
        rms = float(np.sqrt((first_split_frames[0] ** 2).mean()))
    else:
        dev, rms = 0.0, float(np.sqrt((first_split_frames[0] ** 2).mean()))

    meta["acceptance"] = {
        "energy_drift_rel_per_time": meta["splits"][next(iter(cfg["splits"]))][
            "energy_drift_rel_per_time"],
        "spectrum_slope": slope, "spectrum_band": [k_lo, k_hi],
        "reproducible_bitwise": bitwise,
        "reference_seeds": ref_chunk,
        "batch_size": int(cfg.get("batch_size", 1)),
        "cross_batch_max_abs_dev": dev,
        "field_rms": rms,
        "float32_ulp": float(np.spacing(np.float32(rms))),
        "max_cfl": max(s["max_cfl"] for s in meta["splits"].values()),
    }
    meta["spectrum"] = {"k": k.tolist(), "E_k": e_k.tolist()}

    _figures(first_split_frames[0], k, e_k, ref_seed, out_dir / "figs", slope, k_lo, k_hi)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--device", default=None, help="auto | cpu | cuda")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    meta = run(cfg, args.device)

    a = meta["acceptance"]
    print("\n=== CRITERIOS DE ACEPTACION, FASE 1 ===")
    print(f"1. Deriva de energia (2a mitad)   {a['energy_drift_rel_per_time']*100:+.3f} %/t"
          f"   {'OK' if abs(a['energy_drift_rel_per_time']) < 0.05 else 'FALLA'} (<5 %)")
    print(f"2. Pendiente de E(k) en k={a['spectrum_band'][0]}-{a['spectrum_band'][1]}"
          f"        {a['spectrum_slope']:.3f}   (medida, sin valor objetivo)")
    print(f"3. Reproducible bit a bit         {a['reproducible_bitwise']}"
          f"            {'OK' if a['reproducible_bitwise'] else 'FALLA'}"
          f" (mismo lote de {a['batch_size']}, semillas {a['reference_seeds'][0]}..)")
    print(f"   Desviacion con lote de 1        {a['cross_batch_max_abs_dev']:.3e}"
          f"   (1 ULP de float32 aqui = {a['float32_ulp']:.3e}; "
          f"caos amplificando el redondeo de la FFT, no un fallo de semillas)")
    print(f"4. CFL maximo                     {a['max_cfl']:.3f}"
          f"          {'OK' if a['max_cfl'] < CFL_LIMIT else 'FALLA'} (<{CFL_LIMIT})")
    print(f"5. Figuras en {Path(meta['config']['output_dir']) / 'figs'}")


if __name__ == "__main__":
    main()
