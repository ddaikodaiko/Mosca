"""Nucleo recurrente con mascara del conectoma y ley de Dale (fase 3).

    h <- tanh(W_in @ x + (M * W) @ h + b)     x T_int pasos internos por paso de fluido

Tres restricciones biologicas (seccion 6 del diseno):

1. Conectividad fija. Solo existen las sinapsis de la mascara. Las demas son cero
   SIEMPRE, no solo al inicializar: los pesos viven en un vector de tamano n_aristas,
   asi que las entradas fuera de la mascara ni siquiera existen como parametro.
2. Ley de Dale. W = signo * softplus(theta), con el signo fijado por la neurona
   presinaptica y solo theta entrenable. softplus > 0, luego el signo no puede cambiar.
3. Inicializacion proporcional al numero de sinapsis, reescalada a radio espectral ~0,95.

El nucleo es agnostico a la tarea: para fluidos se envuelve en ConnectomeFlowModel, y
la seccion 13 del diseno (control sensoriomotor) reutilizaria el mismo nucleo con otro
codificador y otro decodificador.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch import Tensor, nn

SOFTPLUS_BETA = 1.0


@dataclass(frozen=True)
class Wiring:
    """Un cableado: aristas, sinapsis y signo por neurona. Lo que produce la fase 2."""

    rows: np.ndarray       # int32 (n_edges,)  neurona presinaptica
    cols: np.ndarray       # int32 (n_edges,)  neurona postsinaptica
    synapses: np.ndarray   # int32 (n_edges,)
    sign: np.ndarray       # int8  (n_neurons,)

    @property
    def n_neurons(self) -> int:
        return int(self.sign.size)

    @property
    def n_edges(self) -> int:
        return int(self.rows.size)

    @classmethod
    def load(cls, path: Path) -> "Wiring":
        data = np.load(path)
        return cls(
            rows=data["rows"].astype(np.int32),
            cols=data["cols"].astype(np.int32),
            synapses=data["synapses"].astype(np.int32),
            sign=data["sign"].astype(np.int8),
        )

    @classmethod
    def dense(cls, n_neurons: int, inhibitory_fraction: float, seed: int) -> "Wiring":
        """Control 'denso': todas contra todas, sin autoconexiones."""
        rng = np.random.default_rng(seed)
        rows, cols = np.meshgrid(
            np.arange(n_neurons, dtype=np.int32), np.arange(n_neurons, dtype=np.int32),
            indexing="ij",
        )
        keep = rows != cols
        sign = np.ones(n_neurons, dtype=np.int8)
        n_inhibitory = int(round(inhibitory_fraction * n_neurons))
        sign[rng.permutation(n_neurons)[:n_inhibitory]] = -1
        return cls(
            rows=rows[keep].copy(),
            cols=cols[keep].copy(),
            synapses=np.ones(int(keep.sum()), dtype=np.int32),
            sign=sign,
        )


def dense_neurons_for_budget(n_edges: int) -> int:
    """Neuronas de una red densa con el mismo numero de parametros entrenables."""
    return max(2, int(round((1 + math.sqrt(1 + 4 * n_edges)) / 2)))


class MaskedRecurrentCore(nn.Module):
    """Nucleo de N neuronas cableado por una mascara fija, con ley de Dale.

    El peso de la sinapsis pre->post vive en theta[e]. La matriz densa nunca se
    materializa: se usa una matriz dispersa de n_aristas valores.
    """

    def __init__(
        self,
        wiring: Wiring,
        n_internal_steps: int = 4,
        spectral_radius: float = 0.95,
        leak: float = 1.0,
        normalisation: str = "max_eig",
    ) -> None:
        super().__init__()
        self.n_neurons = wiring.n_neurons
        self.n_edges = wiring.n_edges
        self.n_internal_steps = n_internal_steps
        self.spectral_radius = spectral_radius
        self.leak = leak

        # W[post, pre]: la fila es quien recibe. Indices fijos, no entrenables.
        indices = np.stack([wiring.cols, wiring.rows]).astype(np.int64)
        self.register_buffer("edge_index", torch.from_numpy(indices))
        # El signo es de la neurona PRESINAPTICA (ley de Dale), no de la sinapsis.
        edge_sign = wiring.sign[wiring.rows].astype(np.float32)
        self.register_buffer("edge_sign", torch.from_numpy(edge_sign))

        self.normalisation = normalisation
        theta = self._initial_theta(wiring, spectral_radius, normalisation)
        self.theta = nn.Parameter(torch.from_numpy(theta))
        self.bias = nn.Parameter(torch.zeros(self.n_neurons))

    @staticmethod
    def _initial_theta(
        wiring: Wiring, spectral_radius: float, normalisation: str = "max_eig"
    ) -> np.ndarray:
        """Magnitud proporcional a las sinapsis, reescalada al radio espectral pedido.

        `normalisation`:
          max_eig  lo que pide el diseno: se escala el autovalor mayor a `spectral_radius`.
                   Medido en la fase 4: el grueso del espectro queda en 0,07 (N=4.000), o
                   sea que un impulso generico se atenua 14 veces por salto.
          row      cada neurona recibe la misma corriente total (se divide por su grado de
                   entrada ponderado) ANTES de escalar el autovalor mayor. Sube el grueso
                   del espectro a 0,55 sin tocar el maximo.
        """
        magnitude = wiring.synapses.astype(np.float64)
        magnitude = magnitude / magnitude.mean()
        if normalisation == "row":
            incoming = np.bincount(wiring.cols, weights=magnitude, minlength=wiring.n_neurons)
            incoming[incoming == 0] = 1.0
            magnitude = magnitude / incoming[wiring.cols]
        elif normalisation != "max_eig":
            raise ValueError(f"normalizacion desconocida: {normalisation}")
        values = wiring.sign[wiring.rows].astype(np.float64) * magnitude
        matrix = sp.coo_matrix(
            (values, (wiring.cols, wiring.rows)),
            shape=(wiring.n_neurons, wiring.n_neurons),
        ).tocsr()
        current = _spectral_radius(matrix)
        if current > 0:
            magnitude *= spectral_radius / current
        # softplus^-1, estable para valores pequenos.
        magnitude = np.clip(magnitude, 1e-8, None)
        return np.log(np.expm1(magnitude)).astype(np.float32)

    def weights(self) -> Tensor:
        """Valores de las sinapsis existentes, con su signo. Siempre > 0 en magnitud."""
        return self.edge_sign * nn.functional.softplus(self.theta, beta=SOFTPLUS_BETA)

    def sparse_matrix(self) -> Tensor:
        return torch.sparse_coo_tensor(
            self.edge_index,
            self.weights(),
            (self.n_neurons, self.n_neurons),
        ).coalesce()

    def dense_matrix(self) -> Tensor:
        """Solo para inspeccion y tests: N x N puede ser grande."""
        matrix = torch.zeros(self.n_neurons, self.n_neurons, device=self.theta.device)
        matrix[self.edge_index[0], self.edge_index[1]] = self.weights()
        return matrix

    def forward(
        self, drive: Tensor, h: Tensor | None = None, record: bool = False
    ) -> tuple[Tensor, list[Tensor]]:
        """drive: (batch, N) corriente de entrada, constante en los pasos internos."""
        if h is None:
            h = torch.zeros(drive.shape[0], self.n_neurons, device=drive.device, dtype=drive.dtype)
        matrix = self.sparse_matrix()
        trace: list[Tensor] = []
        for _ in range(self.n_internal_steps):
            recurrent = torch.sparse.mm(matrix, h.t()).t()
            candidate = torch.tanh(drive + recurrent + self.bias)
            h = candidate if self.leak >= 1.0 else (1 - self.leak) * h + self.leak * candidate
            if record:
                trace.append(h)
        return h, trace


def _spectral_radius(matrix: sp.spmatrix) -> float:
    try:
        value = spla.eigs(matrix.astype(np.float64), k=1, which="LM", return_eigenvectors=False)
        return float(abs(value[0]))
    except (spla.ArpackNoConvergence, ValueError):
        # Grafos pequenos o mal condicionados: cae a la matriz densa.
        return float(np.max(np.abs(np.linalg.eigvals(matrix.toarray()))))


def select_ports(wiring: Wiring, n_sensory: int, n_motor: int) -> tuple[np.ndarray, np.ndarray]:
    """Sensoriales = mas salidas hacia el resto; motoras = mas entradas, y disjuntas."""
    n = wiring.n_neurons
    if n_sensory + n_motor > n:
        raise ValueError("no hay neuronas suficientes para tantos puertos")
    out_degree = np.bincount(wiring.rows, minlength=n)
    in_degree = np.bincount(wiring.cols, minlength=n)

    sensory = np.argsort(-out_degree, kind="stable")[:n_sensory].astype(np.int64)
    taken = np.zeros(n, dtype=bool)
    taken[sensory] = True
    order = np.argsort(-in_degree, kind="stable")
    motor = order[~taken[order]][:n_motor].astype(np.int64)
    return np.sort(sensory), np.sort(motor)


def low_frequency_modes(grid: int, n_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Modos independientes de rfft2 de menor |k|, con sus espejos conjugados.

    El campo es real, asi que su espectro es hermitico: en las columnas ky = 0 y
    ky = N/2 el coeficiente de -kx es el conjugado del de +kx. Si se guardan los dos
    se guarda basura redundante, y si se guarda solo uno y se reconstruye sin poner
    el otro, `irfft2` simetriza y **se pierde la mitad de la amplitud de ese modo**.
    Por eso se eligen solo los independientes y el decodificador rellena los espejos.

    Devuelve (modos, origen_del_espejo, destino_del_espejo), todos indices planos.
    """
    kx = np.fft.fftfreq(grid, d=1.0 / grid)
    ky = np.fft.rfftfreq(grid, d=1.0 / grid)
    kx_grid = np.broadcast_to(kx[:, None], (grid, ky.size))
    ky_grid = np.broadcast_to(ky[None, :], (grid, ky.size))

    self_conjugate_column = (ky_grid == 0) | (ky_grid == grid // 2)
    independent = ~self_conjugate_column | (kx_grid >= 0)

    k2 = (kx_grid ** 2 + ky_grid ** 2).ravel()
    flat_independent = independent.ravel()
    order = np.lexsort((np.arange(k2.size), k2))
    order = order[flat_independent[order]]
    if n_modes > order.size:
        raise ValueError(f"solo hay {order.size} modos independientes en una malla {grid}")
    modes = np.sort(order[:n_modes]).astype(np.int64)

    # Espejo: mismo ky, kx cambiado de signo. Solo en las columnas autoconjugadas y
    # cuando kx no es su propio espejo (kx = 0 y kx = N/2 lo son).
    row, col = np.divmod(modes, ky.size)
    needs_mirror = self_conjugate_column.ravel()[modes] & (row != 0) & (row != grid // 2)
    mirror_source = np.flatnonzero(needs_mirror).astype(np.int64)
    mirror_target = ((grid - row[needs_mirror]) % grid) * ky.size + col[needs_mirror]
    return modes, mirror_source, mirror_target.astype(np.int64)


class ConnectomeFlowModel(nn.Module):
    """Codificador FFT -> nucleo con mascara -> decodificador iFFT, prediccion residual."""

    def __init__(
        self,
        wiring: Wiring,
        grid: int,
        n_complex_modes: int = 128,
        n_sensory: int = 256,
        n_motor: int = 256,
        n_internal_steps: int = 4,
        spectral_radius: float = 0.95,
        leak: float = 1.0,
        output_scale: float = 0.1,
        normalisation: str = "max_eig",
    ) -> None:
        super().__init__()
        self.grid = grid
        self.n_complex_modes = n_complex_modes
        self.core = MaskedRecurrentCore(
            wiring, n_internal_steps, spectral_radius, leak, normalisation
        )

        sensory, motor = select_ports(wiring, n_sensory, n_motor)
        self.register_buffer("sensory", torch.from_numpy(sensory))
        self.register_buffer("motor", torch.from_numpy(motor))
        modes, mirror_source, mirror_target = low_frequency_modes(grid, n_complex_modes)
        self.register_buffer("mode_index", torch.from_numpy(modes))
        self.register_buffer("mirror_source", torch.from_numpy(mirror_source))
        self.register_buffer("mirror_target", torch.from_numpy(mirror_target))

        n_features = 2 * n_complex_modes  # parte real e imaginaria
        # Estadisticas de los modos, medidas sobre el conjunto de entrenamiento. Sin
        # esto el impulso de entrada llega a tanh con rms ~2 y satura de salida: los
        # coeficientes de Fourier de un campo de rms 2,2 llegan a valer 22.
        self.register_buffer("feature_mean", torch.zeros(n_features))
        self.register_buffer("feature_std", torch.ones(n_features))
        self.encoder = nn.Linear(n_features, n_sensory)
        self.decoder = nn.Linear(n_motor, n_features)
        nn.init.zeros_(self.decoder.bias)
        with torch.no_grad():
            self.decoder.weight.mul_(output_scale)  # arranca cerca de la persistencia

    def encode(self, omega: Tensor) -> Tensor:
        spectrum = torch.fft.rfft2(omega, norm="ortho").reshape(omega.shape[0], -1)
        picked = spectrum[:, self.mode_index]
        features = torch.cat([picked.real, picked.imag], dim=1)
        return (features - self.feature_mean) / self.feature_std

    @torch.no_grad()
    def fit_feature_statistics(self, omega: Tensor) -> None:
        """Mide media y desviacion de los modos sobre una muestra de entrenamiento.

        Quedan como buffers, o sea que viajan en el checkpoint y el modelo evaluado es
        exactamente el entrenado. Se llama ANTES de entrenar y nunca despues.
        """
        spectrum = torch.fft.rfft2(omega, norm="ortho").reshape(omega.shape[0], -1)
        picked = spectrum[:, self.mode_index]
        features = torch.cat([picked.real, picked.imag], dim=1)
        self.feature_mean.copy_(features.mean(dim=0))
        self.feature_std.copy_(features.std(dim=0).clamp_min(1e-6))

    def decode(self, features: Tensor) -> Tensor:
        # El decodificador trabaja en unidades normalizadas: se deshace la escala para
        # que el incremento predicho nazca en el orden de magnitud correcto.
        features = features * self.feature_std
        half = features.shape[1] // 2
        picked = torch.complex(features[:, :half], features[:, half:])
        shape = (features.shape[0], self.grid, self.grid // 2 + 1)
        spectrum = torch.zeros(shape, dtype=picked.dtype, device=features.device)
        spectrum = spectrum.reshape(features.shape[0], -1)
        spectrum = spectrum.index_copy(1, self.mode_index, picked)
        mirrored = picked.index_select(1, self.mirror_source).conj()
        spectrum = spectrum.index_copy(1, self.mirror_target, mirrored).reshape(shape)
        return torch.fft.irfft2(spectrum, s=(self.grid, self.grid), norm="ortho")

    def forward(
        self, omega: Tensor, h: Tensor | None = None, record: bool = False
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Devuelve (omega(t+dt), estado oculto, traza de activaciones si record)."""
        features = self.encode(omega)
        drive = torch.zeros(omega.shape[0], self.core.n_neurons, device=omega.device, dtype=omega.dtype)
        drive = drive.index_copy(1, self.sensory, self.encoder(features))
        h, trace = self.core(drive, h, record=record)
        delta = self.decode(self.decoder(h.index_select(1, self.motor)))
        return omega + delta, h, trace  # prediccion RESIDUAL

    def rollout(self, omega: Tensor, steps: int) -> Tensor:
        """Despliegue autoregresivo: (batch, steps, grid, grid)."""
        h: Tensor | None = None
        frames = []
        for _ in range(steps):
            omega, h, _ = self(omega, h)
            frames.append(omega)
        return torch.stack(frames, dim=1)


def build_wiring(cfg: dict[str, Any], seed: int) -> Wiring:
    """Carga el cableado que pide la config: mosca, aleatorio, recableado o denso."""
    name = cfg["wiring"]
    if name == "denso":
        reference = Wiring.load(Path(cfg["wiring_dir"]) / "mosca.npz")
        n_dense = dense_neurons_for_budget(reference.n_edges)
        fraction = float((reference.sign < 0).mean())
        return Wiring.dense(n_dense, fraction, seed)
    return Wiring.load(Path(cfg["wiring_dir"]) / f"{name}.npz")
