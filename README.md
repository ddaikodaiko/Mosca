# Mosca — an AI wired like a fruit fly's brain, predicting a fluid

[Español](README.es.md)

<p align="center">
  <img src="docs/media/fly_fluid.gif" alt="Neurons of the fly-wired network lighting up next to the real and predicted vorticity field" width="100%">
</p>

We take the wiring diagram of the *Drosophila* brain (FlyWire connectome, ~139,000 neurons),
cut out a piece, and build an artificial recurrent network that can **only use the
connections that exist in the fly**, keeping which neurons excite and which inhibit (Dale's
law). Then we make it predict a 2D Navier–Stokes flow and play a platform game.

**The question:** does the real wiring help, compared with random wiring that has the same
statistics?

> **Not a fly.** Nothing here is a fly or a model of how a fly thinks. The connectome is used
> as the blueprint of an artificial network.

## In short (21 Oct 2026)

- **Across five tasks, the fly wiring never beats random wiring.** In the only task with
  enough resolution to tell wirings apart (platform levels the model has never seen), it
  **loses** to random wiring, under an analysis written as code before the data existed.
- **We found a design bias that penalises the fly.** To start every
  wiring in the same dynamical regime, the design scales the weights until the largest
  eigenvalue is 0.95. The fly has a core of neuron pairs connected both ways with many
  synapses, which inflates its largest eigenvalue; once that is scaled down, every other
  connection is **27 times weaker than in the random graph** (73 times with 4,000 neurons).
  Its recurrence is nearly switched off, which fits with it tying a control that has no
  connections at all. Equal spectral radius is not equal treatment.
- <!-- 6E --> **But it is not what makes it lose.** Retrained with a normalisation that
  equalises each neuron's input current (the fly's recurrence goes from 27 times more
  switched off to 4), the fly improves a little (0.462 → 0.503 progress) and the random
  wiring improves more (0.568 → 0.653): the gap does not close, it widens. Under the
  analysis written before the data, the rescaling is ruled out as the cause. Why the fly
  loses is still unmeasured. <!-- /6E -->
- **Still missing:** the original full-scale fluid experiment (4,000 neurons, 64×64 grid)
  is at 8 of 18 training runs, paused. With one seed, the fly does beat the no-connection
  control there.

Every number, including the ones that went wrong, is in [`RESULTADOS.md`](RESULTADOS.md)
(Spanish). The comparison with the published literature is in
[`docs/literatura/informe.md`](docs/literatura/informe.md) (Spanish; the underlying notes
are mostly in English).

## Pre-registered prediction

Written down before seeing any result ([design, §3](diseno_mosca_navier_stokes.md)):

| | Hypothesis |
|---|---|
| **H0** | The fly wiring performs the same as random wiring of the same density |
| **H1** | Any advantage is explained by the degree distribution (fly ≈ degree-preserving rewired) |
| **H2** | The specific structure matters (fly > rewired, significant) |

We expect **H0 or H1**, and a Fourier Neural Operator (FNO) to beat every wired network.
A null result is a valid result.

## The models

Only the connectivity mask changes between the wired models; encoder, decoder, training
recipe and parameter budget are shared. Every weight inside the mask is trained; each
neuron's sign comes from the data and never changes.

| Model | Recurrent core |
|---|---|
| `mosca` | FlyWire subgraph, real signs |
| `aleatorio` | Erdős–Rényi, same density and inhibitory fraction |
| `recableado` | Fly graph with edges shuffled, **degrees preserved exactly** |
| `denso` | All-to-all, fewer neurons, same number of trainable weights |
| `sin_nucleo` | Same neurons, signs and ports as the fly, **zero connections** (control) |
| `mlp` | Unmasked network with the same parameter count (Mario only) |
| `fno` | Fourier Neural Operator (upper reference, fluids only) |
| `persistencia` | Predicts "nothing changes" (lower reference, fluids only) |

`sin_nucleo` is the control that says whether there is anything to discuss: if the fly does
not beat it, what it does comes not from its wiring but from the encoder and decoder every
model shares. We have not found any other study that uses it.

## Results, task by task

| Task | Learns by | Fly vs random / rewired | Core vs `sin_nucleo` |
|---|---|---|---|
| Navier–Stokes, rehearsal (500 neurons, 32×32) | supervised | tie / tie | 0.0004 less error: nothing |
| Navier–Stokes, full scale (4,000, 64×64) | supervised | **pending**: 8 of 18 runs | seed 0: −4 % rollout error |
| Own platformer | PPO from scratch | everyone stuck in "jump right"; closed without resolution | — |
| Super Mario Bros (NES), level 1-1 | imitation with memory | tie (fly 0.98, random 0.98, rewired 1.00) | memory decides: 0.98 vs 0.32 |
| NES, unseen levels | PPO | tie / unresolved | tie |
| Procedural platformer, 200 unseen levels | imitation + PPO | **loses** (0.462 vs 0.568) / unresolved | tie |

In the last one, all six models reach the goal on levels they have never seen and beat
random buttons and every constant policy, so the task does discriminate. And in that task
random and rewired wiring **do** beat `sin_nucleo` (+0.091 and +0.049, p < 0.0001):
recurrence helps when it is alive. The one not using it is the fly.

## Equal spectral radius is not equal treatment (but that is not why the fly loses)

| | fly | rewired | random |
|---|---|---|---|
| largest eigenvalue before scaling (500 neurons) | 74.5 | 16.4 | 10.4 |
| bulk of the spectrum after scaling to 0.95 (median \|eigenvalue\|) | **0.013** | 0.102 | 0.356 |
| same with 4,000 neurons | **0.004** | 0.075 | 0.284 |
| fraction of connections with a reciprocal partner | 0.35 | 0.09 | 0.04 |
| progress on unseen levels | 0.462 | 0.527 | 0.568 |
| bulk of the spectrum with row normalisation (`row`) | 0.146 | 0.498 | 0.595 |
| progress with `row` | 0.503 | 0.432 | 0.653 |

The order of the spectral bulk is the order of the results, and training does not undo it:
after PPO the fly is still at 0.013. Measured with
[`diagnostico_cableado.py`](diagnostico_cableado.py), without training anything.

This does not say the connectome is bad. It says that the usual way of putting wirings on
an equal footing (matching the spectral radius, standard practice in reservoir computing)
penalises graphs with a strong reciprocal core, and the fly's has one.

What it does not say is that this explains the defeat. Phase 6e checked it (exploratory,
analysis written before the data): the same batch with `core.normalisation: row`, which
equalises each neuron's input current before scaling. The fly's recurrence wakes up (bulk
0.146) and the fly improves, but the random wiring improves more, and the fly still loses
to it (−0.150, all three of its seeds below). The registered verdict is "no effect"
against both controls: the rescaling is ruled out as the cause.

## The tasks

### Fluids

Pipeline: vorticity field → FFT → lowest Fourier modes → *sensory* neurons (highest
out-degree) → masked recurrent core (4 internal steps) → *motor* neurons (highest in-degree)
→ modes → inverse FFT → **increment** added to the field.

### Mario

The same wired core playing instead of predicting ([design, §13](diseno_mosca_navier_stokes.md)).
Two environments with **the same symbolic observation**, so the policy is identical in
both: a hand-written platformer with procedural levels, and the real NES Super Mario Bros
through an emulator, reading the tile map out of the RAM. The anatomical viewer
(`visor_nes.py`) puts the game on one side and, on the other, the 139,000 connectome neurons
with the subgraph lighting up on top, at their real FlyWire coordinates.

<p align="center">
  <img src="docs/media/mosca_juega_nes.gif" alt="The fly-wired network playing Super Mario Bros, with its neural activity beside it" width="100%">
</p>

This demo (300 PPO iterations, sampled policy) trains and measures on the same level, beats
the constant-jump policy and is **indistinguishable from mashing buttons at random**
(p = 0.25).

Every number is read against the reference policies (random, always right, constant jump,
running jump, and a hand-written heuristic) or not at all. PPO from scratch falls into the
degenerate "always jump right" policy on its own, and so does an unmasked MLP. That is why
the version that separates wirings starts by imitating the heuristic and then fine-tunes
with PPO, and uses the NES rule that the jump button must be released before jumping again,
which sinks every constant policy.

By imitation, the fly-wired network clears NES level 1-1
([video](docs/media/mosca_clonada_1-1.mp4)). That is a demo, not science: same level,
copied from a teacher. With controls, the credit goes to memory, not wiring: random and
rewired cores clone it just as well, and the models without recurrence stall at the first
pipe.

## What we learned on the way

- **A bug in data augmentation**: the free translation was applied along the axis that
  carries the forcing `sin(4y)`, so augmented pairs had 4 % physically wrong content while the
  signal to learn is ~10 % of the field. Caught with a test against the simulator; every
  earlier result with augmentation was discarded.
- **Unfair ports**: letting input and output neurons overlap freely gave the fly 159
  input→output shortcuts and the random graph 18. The overlap is now the same for every
  model.
- **A blind critic**: in Mario, the value estimate saw nothing, because it had to output
  30–60 from small weights. Fixing it was necessary, but it was not what held PPO back:
  exploration was.
- **Evaluation noise**: with one game per level, the ranking between models flipped on its
  own. Now every model plays the same levels with the same random numbers, and tests are run
  per level and per seed.
- **Analysis before data**: since phase 6c, the analysis is written as code and committed
  before the training runs are launched. That is why the fly's loss cannot be reinterpreted
  after the fact.

## Against the literature

As far as we can tell, nobody had predicted a fluid equation with a connectome, nobody uses
a no-connection control, and no study combines all these controls. Published positive
results with fly connectomes live in other regimes (fixed weights with a few parameters per
neuron, or the graph as the only path from input to output) and are mostly 2026 preprints.
The September 2026 wave of "the fly plays X" projects evaluates on the level it trains on,
with almost no controls. Full report, each source labelled by review status:
[`docs/literatura/informe.md`](docs/literatura/informe.md).

## Status

**Full-scale fluids:** 8 of 18 runs done (fly ×3, random ×3, rewired s0, `sin_nucleo` s0),
paused since 23 Sep 2026. About 3 GPU hours left. It is the project's original question and
the only task at 4,000 neurons.

<!-- RESULTS -->

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # on GPU, install the CUDA torch wheel
pytest -q

python -m data.download_flywire           # connectome, 105 MB
python -m data.download_annotations       # neuron positions for the viewer, 32 MB
python -m data.simulate_ns   --config configs/ns_32_proto.yaml
python -m connectome.subgraph --config configs/conectoma_500_proto.yaml
python -m connectome.controls --config configs/conectoma_500_proto.yaml
python benchmark.py --config configs/ensayo_32.yaml          # measured cost before launching
python train.py    --config configs/ensayo_32.yaml
python evaluate.py --config configs/ensayo_32.yaml
python visor.py    --config configs/ensayo_32.yaml --models mosca aleatorio   # interactive HTML
python video.py    --config configs/ensayo_32.yaml --models mosca              # the video above
```

Mario, with the recipe that separates wirings (imitation + PPO, procedural levels):

```bash
C=configs/mario_proto_6c_soltar.yaml
python clonado_rl.py  --config $C --ruido 0.1 --guarda-pesos --semilla 0
python train_rl.py    --config $C --model mosca --seed 0 \
    --desde artifacts/mario_proto_6c_soltar/clonado/mosca_s0.pt
python evaluate_rl.py --config $C --levels 200 \
    --checkpoint artifacts/mario_proto_6c_soltar/desde_clonado/mosca_s0.pt
python diagnostico_cableado.py --config $C        # the spectrum of each wiring
pip install nes-py gym-super-mario-bros           # only for the NES environment
python visor_nes.py   --config configs/mario_nes_demo.yaml     # game + anatomical brain
```

Everything is configured by YAML; seeds and configs are written into every output. The real
runs do not fit on a laptop: the recipe for the GPU machine and for Google Cloud is in
[`TRASPASO.md`](TRASPASO.md) (Spanish) and `scripts/gcp/`.

## Data and credits

- Connectome: FlyWire, materialization v783 — Dorkenwald et al., *Nature* 2024; Schlegel et
  al., *Nature* 2024. Connectivity and signs as packaged by Shiu et al., *Nature* 2024
  ([philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)).
- Neuron positions: [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations).
- Downloaded data is not redistributed in this repository.

MIT licence.
