"""Operadores pseudoespectrales para Navier-Stokes 2D en forma de vorticidad.

Dominio [0, 2*pi]^2 periodico, malla n x n. Los campos se guardan como tensores
reales de forma (..., n, n) con el eje x en la dimension -2 y el eje y en la -1.
Su transformada `rfft2` tiene forma (..., n, n//2 + 1).

Todo esta escrito en torch para que el mismo codigo corra en CPU y en GPU. El
dispositivo nunca se fija a mano: viene de `resolve_device`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resuelve el dispositivo. 'auto' usa cuda si esta disponible, si no cpu."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@dataclass(frozen=True)
class SpectralGrid:
    """Numeros de onda, mascara de desaliasing y forzado, precalculados."""

    n: int
    device: torch.device
    dtype: torch.dtype
    kx: torch.Tensor        # (n, n//2+1) numero de onda en x
    ky: torch.Tensor        # (n, n//2+1) numero de onda en y
    k2: torch.Tensor        # kx^2 + ky^2
    k2_inv: torch.Tensor    # 1/k2 con el modo (0,0) puesto a cero
    dealias: torch.Tensor   # mascara 2/3
    forcing_hat: torch.Tensor  # transformada del forzado de Kolmogorov

    @property
    def dx(self) -> float:
        return 2.0 * np.pi / self.n


def make_grid(
    n: int,
    forcing_wavenumber: int = 4,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> SpectralGrid:
    """Construye la malla espectral con forzado de Kolmogorov f = sin(k_f * y)."""
    device = device or resolve_device()
    # fftfreq * n da numeros de onda enteros en el dominio de longitud 2*pi.
    kx_1d = torch.fft.fftfreq(n, d=1.0 / n, device=device, dtype=dtype)
    ky_1d = torch.fft.rfftfreq(n, d=1.0 / n, device=device, dtype=dtype)
    kx, ky = torch.meshgrid(kx_1d, ky_1d, indexing="ij")

    k2 = kx**2 + ky**2
    k2_inv = torch.where(k2 > 0, 1.0 / torch.where(k2 > 0, k2, torch.ones_like(k2)),
                         torch.zeros_like(k2))

    # Regla de los 2/3: se anula todo por encima de n/3.
    k_cut = n // 3
    dealias = ((kx.abs() <= k_cut) & (ky.abs() <= k_cut)).to(dtype)

    y = torch.linspace(0, 2 * np.pi, n + 1, device=device, dtype=dtype)[:-1]
    forcing = torch.sin(forcing_wavenumber * y).unsqueeze(0).expand(n, n).contiguous()
    forcing_hat = torch.fft.rfft2(forcing)

    return SpectralGrid(n=n, device=device, dtype=dtype, kx=kx, ky=ky, k2=k2,
                        k2_inv=k2_inv, dealias=dealias, forcing_hat=forcing_hat)


def velocity_from_vorticity(w_hat: torch.Tensor, g: SpectralGrid
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Resuelve -lap(psi) = w y devuelve u = d(psi)/dy, v = -d(psi)/dx en real."""
    psi_hat = w_hat * g.k2_inv
    u = torch.fft.irfft2(1j * g.ky * psi_hat, s=(g.n, g.n))
    v = torch.fft.irfft2(-1j * g.kx * psi_hat, s=(g.n, g.n))
    return u, v


def rhs(w_hat: torch.Tensor, g: SpectralGrid, nu: float, alpha: float) -> torch.Tensor:
    """d(w_hat)/dt = -N(w) - nu*k2*w + f - alpha*w, con N desaliasado a 2/3."""
    u, v = velocity_from_vorticity(w_hat, g)
    w_x = torch.fft.irfft2(1j * g.kx * w_hat, s=(g.n, g.n))
    w_y = torch.fft.irfft2(1j * g.ky * w_hat, s=(g.n, g.n))
    nonlinear = torch.fft.rfft2(u * w_x + v * w_y) * g.dealias
    return -nonlinear - nu * g.k2 * w_hat + g.forcing_hat - alpha * w_hat


def rk4_step(w_hat: torch.Tensor, dt: float, g: SpectralGrid, nu: float,
             alpha: float) -> torch.Tensor:
    """Un paso de Runge-Kutta clasico de 4o orden."""
    k1 = rhs(w_hat, g, nu, alpha)
    k2 = rhs(w_hat + 0.5 * dt * k1, g, nu, alpha)
    k3 = rhs(w_hat + 0.5 * dt * k2, g, nu, alpha)
    k4 = rhs(w_hat + dt * k3, g, nu, alpha)
    return w_hat + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def kinetic_energy(w_hat: torch.Tensor, g: SpectralGrid) -> torch.Tensor:
    """Energia cinetica media por unidad de area, E = <|u|^2>/2."""
    u, v = velocity_from_vorticity(w_hat, g)
    return 0.5 * (u**2 + v**2).mean(dim=(-2, -1))


def max_velocity(w_hat: torch.Tensor, g: SpectralGrid) -> torch.Tensor:
    """max(|u|, |v|), para el criterio CFL."""
    u, v = velocity_from_vorticity(w_hat, g)
    return torch.maximum(u.abs().amax(dim=(-2, -1)), v.abs().amax(dim=(-2, -1)))


def cfl_number(w_hat: torch.Tensor, g: SpectralGrid, dt: float) -> torch.Tensor:
    """Numero de Courant max|u| * dt / dx. Debe quedar bastante por debajo de 1."""
    return max_velocity(w_hat, g) * dt / g.dx


def energy_spectrum(w_hat: torch.Tensor, g: SpectralGrid
                    ) -> tuple[np.ndarray, np.ndarray]:
    """E(k) promediado en anillos de |k| entero. Devuelve (k, E(k)) en numpy.

    Se promedia sobre cualquier dimension de lote que traiga w_hat.
    """
    psi_hat = w_hat * g.k2_inv
    # |u_hat|^2 + |v_hat|^2 = k2 * |psi_hat|^2
    e_density = g.k2 * psi_hat.abs() ** 2
    # rfft2 omite la mitad negativa de ky: los modos interiores cuentan doble.
    weight = torch.full_like(g.k2, 2.0)
    weight[:, 0] = 1.0
    if g.n % 2 == 0:
        weight[:, -1] = 1.0
    e_density = e_density * weight / (g.n**4)

    if e_density.ndim > 2:
        e_density = e_density.reshape(-1, *e_density.shape[-2:]).mean(dim=0)

    k_mag = torch.sqrt(g.k2).round().to(torch.int64)
    k_max = int(g.n // 3)
    k_flat = k_mag.flatten().cpu().numpy()
    e_flat = e_density.flatten().double().cpu().numpy()
    spectrum = np.bincount(k_flat, weights=e_flat, minlength=k_max + 1)[: k_max + 1]
    k_axis = np.arange(k_max + 1)
    return k_axis, 0.5 * spectrum


def random_vorticity(n: int, seed: int, peak_wavenumber: float = 4.0,
                     rms: float = 1.0) -> np.ndarray:
    """Condicion inicial aleatoria, filtrada a baja frecuencia y de media cero.

    Se genera con numpy (no con torch) a proposito: el flujo de numeros de
    `np.random.default_rng` es identico en cualquier maquina y dispositivo, asi
    que la misma semilla da la misma trayectoria en el portatil y en la GPU.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, n))
    noise_hat = np.fft.rfft2(noise)

    kx = np.fft.fftfreq(n, d=1.0 / n)[:, None]
    ky = np.fft.rfftfreq(n, d=1.0 / n)[None, :]
    k2 = kx**2 + ky**2
    envelope = np.exp(-k2 / (2.0 * peak_wavenumber**2))
    envelope[0, 0] = 0.0  # vorticidad de media cero

    w = np.fft.irfft2(noise_hat * envelope, s=(n, n))
    w -= w.mean()
    scale = np.sqrt(np.mean(w**2))
    if scale > 0:
        w *= rms / scale
    return w
