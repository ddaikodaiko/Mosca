"""Fourier Neural Operator 2D: el techo de rendimiento (fase 3).

Es el modelo que esperamos que gane a todos los cableados (prediccion registrada:
H0/H1). No tiene restriccion biologica ninguna: opera directamente sobre el campo y
sus modos de Fourier, con la arquitectura estandar de Li et al. (2021).

Comparte contrato con ConnectomeFlowModel: entra omega(t) de forma (batch, n, n), sale
omega(t+dt) por prediccion residual, y expone rollout() para el despliegue.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SpectralConv2d(nn.Module):
    """Convolucion en el espacio de Fourier truncada a los modos bajos."""

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / (in_channels * out_channels)
        # Dos bloques: frecuencias positivas y negativas en x (rfft2 solo trunca en y).
        self.weight_low = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )
        self.weight_high = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )

    @staticmethod
    def _multiply(inputs: Tensor, weight: Tensor) -> Tensor:
        return torch.einsum("bixy,ioxy->boxy", inputs, weight)

    def forward(self, x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        spectrum = torch.fft.rfft2(x, norm="ortho")
        out = torch.zeros(
            batch, self.out_channels, height, width // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        mx, my = self.modes_x, self.modes_y
        out[:, :, :mx, :my] = self._multiply(spectrum[:, :, :mx, :my], self.weight_low)
        out[:, :, -mx:, :my] = self._multiply(spectrum[:, :, -mx:, :my], self.weight_high)
        return torch.fft.irfft2(out, s=(height, width), norm="ortho")


class FNO2d(nn.Module):
    """FNO 2D estandar con prediccion residual, para comparar contra los cableados."""

    def __init__(
        self,
        modes: int = 12,
        width: int = 32,
        n_layers: int = 4,
        grid: int | None = None,
        output_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.width = width
        self.grid = grid
        # Canales de entrada: el campo mas las dos coordenadas periodicas del dominio.
        self.lift = nn.Conv2d(3, width, kernel_size=1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for _ in range(n_layers)]
        )
        self.project = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1), nn.GELU(), nn.Conv2d(128, 1, kernel_size=1)
        )
        with torch.no_grad():  # arranca cerca de la persistencia, como el modelo con mascara
            self.project[-1].weight.mul_(output_scale)
            nn.init.zeros_(self.project[-1].bias)

    @staticmethod
    def _coordinates(omega: Tensor) -> Tensor:
        batch, height, width = omega.shape
        y = torch.linspace(0, 1, height + 1, device=omega.device)[:-1]
        x = torch.linspace(0, 1, width + 1, device=omega.device)[:-1]
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        stacked = torch.stack([grid_y, grid_x])
        return stacked.unsqueeze(0).expand(batch, -1, -1, -1).to(omega.dtype)

    def forward(self, omega: Tensor) -> Tensor:
        x = torch.cat([omega.unsqueeze(1), self._coordinates(omega)], dim=1)
        x = self.lift(x)
        for spectral, pointwise in zip(self.spectral, self.pointwise):
            x = nn.functional.gelu(spectral(x) + pointwise(x))
        return omega + self.project(x).squeeze(1)  # prediccion RESIDUAL

    def rollout(self, omega: Tensor, steps: int) -> Tensor:
        frames = []
        for _ in range(steps):
            omega = self(omega)
            frames.append(omega)
        return torch.stack(frames, dim=1)


class Persistence(nn.Module):
    """El suelo: omega(t+dt) = omega(t). No se entrena."""

    def forward(self, omega: Tensor) -> Tensor:
        return omega

    def rollout(self, omega: Tensor, steps: int) -> Tensor:
        return omega.unsqueeze(1).expand(-1, steps, -1, -1).clone()
