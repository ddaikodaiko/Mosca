"""Politica actor-critico con el mismo nucleo cableado que los fluidos (seccion 13).

El nucleo (`MaskedRecurrentCore`) no cambia ni una linea: mascara fija, ley de Dale y
radio espectral, exactamente igual que en `ConnectomeFlowModel`. Lo unico que cambia es
que el codificador ya no es una FFT sino la ventana de casillas del entorno, y que el
decodificador saca logits de botones y un valor, no modos de Fourier.

Diferencia importante con fluidos: **aqui el estado oculto persiste entre pasos del
entorno**. En fluidos cada paso arranca con la h que dejo el anterior dentro de la misma
trayectoria; aqui ademas hay que borrarla cuando un episodio termina, porque el episodio
siguiente es otro nivel.

Los puertos se eligen con `select_ports` y el mismo `n_shared` fijo en todos los modelos,
por la razon de la fase 5b: cada neurona compartida es un atajo entrada -> tanh -> salida
que no pasa por la recurrencia, y con solape libre la mosca ganaria por los atajos.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from models.masked_rnn import MaskedRecurrentCore, Wiring, dense_neurons_for_budget, select_ports


class ConnectomePolicy(nn.Module):
    """observacion -> sensoriales -> nucleo con mascara -> motoras -> (logits, valor)."""

    def __init__(
        self,
        wiring: Wiring,
        n_features: int,
        n_actions: int,
        n_shared: int,
        port_wiring: Wiring | None = None,
        n_sensory: int = 256,
        n_motor: int = 256,
        n_internal_steps: int = 4,
        spectral_radius: float = 0.95,
        leak: float = 1.0,
        normalisation: str = "max_eig",
        policy_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.core = MaskedRecurrentCore(
            wiring, n_internal_steps, spectral_radius, leak, normalisation
        )
        sensory, motor = select_ports(port_wiring or wiring, n_sensory, n_motor, n_shared)
        self.register_buffer("sensory", torch.from_numpy(sensory))
        self.register_buffer("motor", torch.from_numpy(motor))

        self.encoder = nn.Linear(n_features, n_sensory)
        self.actor = nn.Linear(n_motor, n_actions)
        self.critic = nn.Linear(n_motor, 1)
        # Estadisticas de la observacion, medidas SOLO sobre niveles de entrenamiento y
        # guardadas como buffers, asi que viajan en el checkpoint. En el NES no evitan
        # saturacion al arrancar (medido: 0 % con y sin), pero centran la entrada y
        # duplican la senal util, y es lo que pide el hallazgo de fluidos. La saturacion
        # del 66 % medida en un modelo ya entrenado viene de que PPO agranda los pesos
        # del codificador, y eso normalizar la entrada NO lo arregla.
        self.register_buffer("feature_mean", torch.zeros(n_features))
        self.register_buffer("feature_std", torch.ones(n_features))
        # Arranca casi uniforme: con logits grandes al principio, PPO colapsa a una
        # accion antes de que el nucleo haya aprendido nada.
        with torch.no_grad():
            self.actor.weight.mul_(policy_scale)
            nn.init.zeros_(self.actor.bias)

    @property
    def n_neurons(self) -> int:
        return self.core.n_neurons

    def initial_state(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.core.n_neurons, device=device)

    @torch.no_grad()
    def fit_feature_statistics(self, observations: Tensor) -> None:
        """Mide media y desviacion de la observacion. Se llama ANTES de entrenar."""
        self.feature_mean.copy_(observations.mean(dim=0))
        # Muchas casillas casi nunca se encienden: con su desviacion real, normalizar las
        # amplificaria a ruido enorme. El suelo de 0,1 las deja donde estan.
        self.feature_std.copy_(observations.std(dim=0).clamp_min(0.1))

    def forward(
        self, observation: Tensor, h: Tensor, record: bool = False
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, list[Tensor]]:
        """Devuelve (logits, valor, h siguiente), y con `record` la traza interna.

        La traza son las `n_internal_steps` activaciones de dentro de un paso de juego.
        La usa el visor; el entrenamiento nunca la pide, porque guardarla multiplicaria
        la memoria del despliegue por el numero de pasos internos.
        """
        drive = torch.zeros(
            observation.shape[0], self.core.n_neurons,
            device=observation.device, dtype=observation.dtype,
        )
        normalised = (observation - self.feature_mean) / self.feature_std
        drive = drive.index_copy(1, self.sensory, self.encoder(normalised))
        h, trace = self.core(drive, h, record=record)
        readout = h.index_select(1, self.motor)
        logits, value = self.actor(readout), self.critic(readout).squeeze(-1)
        return (logits, value, h, trace) if record else (logits, value, h)



class NormalizadorDeRetorno(nn.Module):
    """Escala con la que el critico aprende, en vez de pelearse con el tamano del retorno.

    El critico es un `Linear` sobre neuronas acotadas por tanh. Si el retorno vale 30 o 60,
    para representarlo necesita pesos enormes, y con lr 3e-4 tarda muchisimo en llegar:
    medido en la demo de NES, tras 300 iteraciones predecia 27,7 para CUALQUIER estado y
    explicaba el 10 % de la varianza del retorno. Con las ventajas hechas casi de ruido, la
    politica avanza a ciegas.

    Aqui el critico predice el retorno NORMALIZADO, y se desnormaliza cuando hace falta en
    unidades de recompensa (el calculo de GAE). La media y la desviacion son medias
    moviles, viajan en el checkpoint como buffers, y con `activo=False` la clase es la
    identidad, asi que lo entrenado antes sigue reproduciendose igual.
    """

    def __init__(self, activo: bool = False, inercia: float = 0.99) -> None:
        super().__init__()
        self.activo = activo
        self.inercia = inercia
        self.register_buffer("media", torch.zeros(()))
        self.register_buffer("desviacion", torch.ones(()))

    @torch.no_grad()
    def actualiza(self, retornos: Tensor) -> None:
        if not self.activo:
            return
        self.media.mul_(self.inercia).add_((1 - self.inercia) * retornos.mean())
        self.desviacion.mul_(self.inercia).add_(
            (1 - self.inercia) * retornos.std().clamp_min(1e-6)
        )

    def normaliza(self, retornos: Tensor) -> Tensor:
        """Retorno en unidades de recompensa -> objetivo del critico."""
        if not self.activo:
            return retornos
        return (retornos - self.media) / self.desviacion.clamp_min(1e-6)

    def desnormaliza(self, valores: Tensor) -> Tensor:
        """Salida del critico -> unidades de recompensa (lo que necesita GAE)."""
        if not self.activo:
            return valores
        return valores * self.desviacion.clamp_min(1e-6) + self.media


class MLPPolicy(nn.Module):
    """Control sin cableado: un MLP libre del mismo presupuesto de parametros.

    En fluidos el MLP libre le gana al nucleo (0,0953 contra 0,0999). Aqui hace el mismo
    papel: marca cuanto de lo que consigue el nucleo es del cableado y cuanto es de tener
    parametros. No es recurrente a proposito; la recurrencia la aporta el nucleo y eso es
    justo lo que se esta midiendo.
    """

    def __init__(self, n_features: int, n_actions: int, hidden: int, policy_scale: float = 0.01):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_features, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)
        self.hidden = hidden
        self.register_buffer("feature_mean", torch.zeros(n_features))
        self.register_buffer("feature_std", torch.ones(n_features))
        with torch.no_grad():
            self.actor.weight.mul_(policy_scale)
            nn.init.zeros_(self.actor.bias)

    @property
    def n_neurons(self) -> int:
        return self.hidden

    def initial_state(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.hidden, device=device)

    @torch.no_grad()
    def fit_feature_statistics(self, observations: Tensor) -> None:
        self.feature_mean.copy_(observations.mean(dim=0))
        self.feature_std.copy_(observations.std(dim=0).clamp_min(0.1))

    def forward(self, observation: Tensor, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.body((observation - self.feature_mean) / self.feature_std)
        return self.actor(features), self.critic(features).squeeze(-1), h


def mlp_parameter_count(hidden: int, n_features: int, n_actions: int) -> int:
    """Parametros entrenables de MLPPolicy con ese ancho."""
    return (
        (n_features + 1) * hidden          # primera capa
        + (hidden + 1) * hidden            # segunda capa
        + (hidden + 1) * (n_actions + 1)   # actor y critico
    )


def hidden_for_budget(n_parameters: int, n_features: int, n_actions: int) -> int:
    """Ancho del MLP que mejor iguala el presupuesto de parametros del nucleo.

    Se resuelve la cuadratica y se queda con el entero mas cercano por arriba o por
    abajo: redondear la raiz sin mas se desviaba un 5 % en los modelos pequenos.
    """
    b = n_features + 2 + n_actions + 1
    c = n_actions + 1 - n_parameters
    root = (-b + (b * b - 4 * c) ** 0.5) / 2
    candidates = [max(8, int(root)), max(8, int(root) + 1)]
    return min(candidates, key=lambda h: abs(mlp_parameter_count(h, n_features, n_actions) - n_parameters))


def build_policy(
    name: str, cfg: dict[str, Any], n_features: int, n_actions: int, seed: int, wirings: dict[str, Wiring]
) -> nn.Module:
    """Construye uno de los modelos del experimento a partir de la config."""
    core_cfg = dict(cfg.get("core", {}))
    encoder_cfg = dict(cfg.get("encoder", {}))
    reference = wirings["mosca"]

    ppo_cfg = dict(cfg.get("ppo", {}))
    normalizador = NormalizadorDeRetorno(
        activo=bool(ppo_cfg.get("normalise_returns", False)),
        inercia=float(ppo_cfg.get("return_momentum", 0.99)),
    )

    if name == "mlp":
        core = ConnectomePolicy(
            reference, n_features, n_actions, n_shared=int(encoder_cfg["n_shared"]),
            n_sensory=int(encoder_cfg["n_sensory"]), n_motor=int(encoder_cfg["n_motor"]),
            **core_cfg,
        )
        budget = sum(p.numel() for p in core.parameters() if p.requires_grad)
        hidden = hidden_for_budget(budget, n_features, n_actions)
        politica = MLPPolicy(n_features, n_actions, hidden)
        politica.normalizador = normalizador
        return politica

    if name == "denso":
        fraction = float((reference.sign < 0).mean())
        wiring = Wiring.dense(dense_neurons_for_budget(reference.n_edges), fraction, seed)
        port_wiring = None
    elif name == "sin_nucleo":
        wiring = Wiring.without_edges(reference)
        port_wiring = reference     # mismos puertos que la mosca, cero conexiones
    else:
        wiring = wirings[name]
        port_wiring = None

    politica = ConnectomePolicy(
        wiring, n_features, n_actions, n_shared=int(encoder_cfg["n_shared"]),
        port_wiring=port_wiring,
        n_sensory=int(encoder_cfg["n_sensory"]), n_motor=int(encoder_cfg["n_motor"]),
        **core_cfg,
    )
    politica.normalizador = normalizador
    return politica
