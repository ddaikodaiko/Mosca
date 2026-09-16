# Mosca — an AI wired like a fruit fly's brain, predicting a fluid

[Español](README.es.md)

<p align="center">
  <img src="docs/media/fly_fluid.gif" alt="Neurons of the fly-wired network lighting up next to the real and predicted vorticity field" width="100%">
</p>

We take the wiring diagram of the *Drosophila* brain (FlyWire connectome, ~139,000 neurons),
cut out a piece, and build an artificial recurrent network that can **only use the
connections that exist in the fly**, keeping which neurons excite and which inhibit (Dale's
law). Then we train it to predict a 2D Navier–Stokes flow one step ahead.

**The question:** does the real wiring help, compared with random wiring that has the same
statistics?

> **Not a fly.** Nothing here is a fly or a model of how a fly thinks. The connectome is used
> as the blueprint of an artificial network.

## Pre-registered prediction

Written down before seeing any result ([design, §3](diseno_mosca_navier_stokes.md)):

| | Hypothesis |
|---|---|
| **H0** | The fly wiring performs the same as random wiring of the same density |
| **H1** | Any advantage is explained by the degree distribution (fly ≈ degree-preserving rewired) |
| **H2** | The specific structure matters (fly > rewired, significant) |

We expect **H0 or H1**, and a Fourier Neural Operator (FNO) to beat every wired network.
A null result is a valid result.

## The six models

Only the connectivity mask changes between the wired models; encoder, decoder, training
recipe and parameter budget are shared.

| Model | Recurrent core |
|---|---|
| `mosca` | FlyWire subgraph, real signs |
| `aleatorio` | Erdős–Rényi, same density and inhibitory fraction |
| `recableado` | Fly graph with edges shuffled, **degrees preserved exactly** |
| `denso` | All-to-all, fewer neurons, same number of trainable weights |
| `fno` | Fourier Neural Operator (upper reference) |
| `persistencia` | Predicts "nothing changes" (lower reference) |

Pipeline: vorticity field → FFT → lowest Fourier modes → *sensory* neurons (highest
out-degree) → masked recurrent core (4 internal steps) → *motor* neurons (highest in-degree)
→ modes → inverse FFT → **increment** added to the field.

## Status

**Work in progress. No final result yet.** The real experiment (N = 4,000 neurons, 64×64
grid, 3 seeds per model) runs on a GPU. What exists so far is a laptop rehearsal (N = 500,
32×32). See [`RESULTADOS.md`](RESULTADOS.md) for every number, including the ones that went
wrong.

<!-- RESULTS -->

## What we learned on the way

- **A bug in data augmentation**: the free translation was applied along the axis that
  carries the forcing `sin(4y)`, so augmented pairs had 4 % physically wrong content while the
  signal to learn is ~10 % of the field. Caught with a test against the simulator; every
  earlier result with augmentation was discarded.
- **Unfair ports**: letting input and output neurons overlap freely gave the fly 159
  input→output shortcuts and the random graph 18. Overlap is now fixed at 92 for every model.
- **The ceiling of the design**: even a perfect core would only lower the one-step error from
  0.105 to 0.083 at 64×64, because the encoder sees only the lowest Fourier modes. There is a
  window to tell wirings apart, but it is narrow.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # on GPU, install the CUDA torch wheel
pytest -q

python -m data.download_flywire           # connectome, 105 MB
python -m data.download_annotations       # neuron positions for the viewer, 32 MB
python -m data.simulate_ns   --config configs/ns_32_proto.yaml
python -m connectome.subgraph --config configs/conectoma_500_proto.yaml
python train.py    --config configs/ensayo_32.yaml
python evaluate.py --config configs/ensayo_32.yaml
python visor.py    --config configs/ensayo_32.yaml --models mosca aleatorio   # interactive HTML
python video.py    --config configs/ensayo_32.yaml --models mosca              # the video above
```

Everything is configured by YAML; seeds and configs are written into every output.

## Data and credits

- Connectome: FlyWire, materialization v783 — Dorkenwald et al., *Nature* 2024; Schlegel et
  al., *Nature* 2024. Connectivity and signs as packaged by Shiu et al., *Nature* 2024
  ([philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)).
- Neuron positions: [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations).
- Downloaded data is not redistributed in this repository.

Project documentation (design and results) is in Spanish. [Leer en castellano](README.es.md).
