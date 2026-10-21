# Trained networks whose wiring comes from a connectome, compared against random, rewired and dense controls (2019-2026)

Scope note. Research done 20-10-2026. Status labels: **[peer-reviewed]**, **[preprint]**, **[conference]** (NeurIPS/ICLR, reviewed), **[abstract only]** (I read only the arXiv abstract page), **[secondary]** (seen only through a blog or summary). "Trained" here means at least some parameters of the connectome-wired network are fitted by gradient descent, evolution or EM. Pure reservoirs, where the wiring is frozen and only a linear readout is fitted, are marked **[reservoir]** and kept because they are the bulk of the "connectome vs random" literature. Three papers the project has already read in full get one line each: Dhiman 2026 (arXiv 2604.04033), Jin et al. FlyGM (arXiv 2602.17997) and Beiran & Litwin-Kumar 2025.

Vocabulary used below, matching Mosca's controls:
- **ER**: random graph with the same number of nodes and edges.
- **DP**: degree-preserving rewiring, e.g. Maslov-Sneppen double-edge swaps.
- **Weight shuffle**: same edges, weights permuted. Any non-DP edge shuffle also destroys degrees.
- **Label permutation**: the graph is kept, but which data neuron maps to which graph node is permuted.
- **Dense**: fully connected, parameter-matched or not.
- **No-core**: zero recurrent connections.

---

## Q1. Fly: flyvis (Lappalainen et al. 2024) and its follow-ups; FlyWire-based RNNs and GNNs; mushroom body and central complex models; any connectome-vs-random comparison with trained weights

### Takeaway
In the fly literature, "trained connectome-constrained network" almost always means **fixed synapse counts and signs, with only per-cell-type or per-synapse-type scalars trained** (flyvis: 734 free parameters; Duan/Dong/Fiete: cell-type gains). Its controls are ablations of biological knowledge, such as a random-parameter model, weight shuffles, sign flips and merged cell types. They are never a DP null with the same training. The only trained-weights-inside-a-FlyWire-mask model on a non-biological task with any wiring control is FLYNN (2026, robot navigation), and its only wiring control is a Watts-Strogatz graph. The studies that ran a strict null or matched baselines (Dhiman 2026 with DP and shared init; Kuklev 2026 with matched baselines) report no wiring advantage, or one that shrinks under the stricter controls. FlyGM's reported win is against unweighted, unit-strength nulls only.

### Cited Findings

**flyvis: Lappalainen et al. 2024, Nature 634:1132-1140 [peer-reviewed]**
- **Fixed from the connectome:** synapse counts (2,355 parameters) and synapse signs (604).
- **Trained:** 65 resting potentials, 65 membrane time constants (shared within each cell type) and 604 non-negative unitary synapse strengths (one per cell-type pair). That is 734 free parameters, plus a decoder.
- **Task and ensemble:** 50 models trained on optic-flow estimation from Sintel video. [Lappalainen 2024, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/)
- **Controls are ablations of biological knowledge, not wiring nulls:**
  - A "random DMN" (random single-cell and synapse parameters under full connectome constraints, decoder only trained) predicted preferred contrast but predicted direction selectivity and preferred direction poorly.
  - "Synapse-optimized" variants with less connectome knowledge: accurate contrast prediction needed the measured signs; direction selectivity could come from cell-type connectivity without synapse counts, but preferred direction could not.
  - Merging cell types into excitatory, inhibitory and mixed types degraded predictions to the level of the random DMN.
  - An unconstrained CNN ensemble (5 networks × 414,666 parameters) was used only as a lower bound on task error.
  - The text does not describe any shuffled, ER or DP connectome control. [Lappalainen 2024, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/)
- **Headline numbers:**
  - The ensemble median got the ON/OFF contrast preference right for 32 of 32 cell types with known selectivity; the single best-task model got 30 of 32.
  - Lower task error correlated with better direction-selectivity predictions: r = −0.60, P = 2.6×10⁻⁶, df = 48. [Lappalainen 2024, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/)
- **Simulation on ground-truth networks (25 per sparsity level, MNIST):**
  - With connectivity alone, the median correlation between recovered and true neurons was 0.85 at 10 % connectivity but only 0.38 at 80 %.
  - With connection-strength estimates added, it exceeded 0.9 at every connectivity.
  - That is, the wiring diagram constrains function mainly when it is sparse. [Lappalainen 2024, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/)
- **Follow-up: Pei, Lappalainen, Turaga, Macke et al., Bernstein 2024 [abstract only].** Connectome-constrained models "can learn many different visual tasks", but only motion-detection tasks reliably yield accurate direction-selectivity predictions. No random or shuffled controls are mentioned. [Pei et al. 2024 poster](https://world-wide.org/conf/bernstein-24/task-choice-influences-single-1waa8x42)
- **Dhiman 2026 (arXiv 2604.04033) [preprint]**, already read. The flyvis "topology advantage" appears under weak nulls and largely vanishes under shared from-scratch initialization plus a DP null. [arXiv 2604.04033](https://arxiv.org/html/2604.04033v1)

**Frozen flyvis on a non-biological task: Kuklev 2026, arXiv 2609.24565 [preprint, single independent author]**
- **Setup:** the flyvis checkpoint is frozen and used as a descriptor (57 cell types × 3 statistics × 8 trajectories = 1,368 dimensions) for one-shot hatch recognition in architectural drawings. The descriptor is matched by nearest-legend cosine similarity, with no learned weights.
- **DEV results:**
  - Fly 0.857.
  - Receptors-only (same input, no connectome) 0.891; fly − receptors = −0.035 [95 % CI −0.049, −0.021].
  - Task-trained CNNs with 5,888 to 92,672 parameters: 0.959-0.966 (3 seeds each; all nine runs beat the fly).
  - Gabor bank 0.802.
- **Controls not run:** random wiring, DP, shuffled connectome, dense, and a parameter-matched fly-sized net. The author lists them as "required future work".
- **Conclusion:** the evidence is "not an advantage of the biological wiring". The fly side uses a single checkpoint (001). [Kuklev 2026](https://arxiv.org/html/2609.24565v1)

**Other frozen-flyvis follow-ups and repos**
- **Dhiman & Panwar 2026, Neuroinformatics [peer-reviewed].** Probes a frozen FlyVis checkpoint for scale-stable direction readout, with task-trained temporal models as controls. These are not wiring nulls. [Springer](https://link.springer.com/article/10.1007/s12021-026-09811-3)
- **"connectome-fidelity" [GitHub repo, unreviewed].** Compares pretrained flyvis ensembles with random baselines using RSA and CKA. [GitHub](https://github.com/michaela10c/connectome-fidelity)

**FLYNN: Wang & Chen 2026, arXiv 2607.00025 v2 [preprint].** The closest published design to Mosca on a non-biological task.
- **Wiring:** FlyWire FAFB v783, 139,255 neurons and 5,342,445 connections (≥5 synapses). The recurrent W is 99.9725 % sparse and non-zero only on connectome edges.
- **Training:** "All model parameters were trainable, including the weights, biases, and per-neuron class leak rates", by DAgger imitation of a privileged planner.
- **Dale:** no sign constraint or Dale's law is mentioned.
- **Controls:**
  - EfficientNet-B0 and MobileNet-v3 (~5 M parameters, GRU heads).
  - "SmallWorldNet": a Watts-Strogatz graph with 139,255 nodes and mean degree 78, matched in synapse count and sparsity. The paper states it was trained without camera dropout.
  - Not run: ER, DP, dense RNN, or a parameter-matched RNN.
- **Seeds and statistics:** number of training seeds not reported; 300 evaluation episodes; no significance tests.
- **Success rates (FLYNN / EfficientNet / MobileNet / SmallWorldNet):**

  | Condition | FLYNN | EfficientNet | MobileNet | SmallWorldNet |
  |---|---|---|---|---|
  | Checkerboard | 92.9 | 90.3 | 91.6 | 89.0 |
  | Out-of-distribution textures | 42.1 | 17.2 | 0.0 | 3.2 |
  | Total blindness | 44.3 | 3.9 | 16.5 | 2.6 |

  [FLYNN v2](https://arxiv.org/html/2607.00025v2)

**FlyGM: Jin et al. 2026, arXiv 2602.17997 [preprint]**, already read. Fixed W over the whole FlyWire graph, imitation learning plus PPO for locomotion. Its rewired and ER baselines are unweighted, unit-strength graphs. [arXiv 2602.17997](https://arxiv.org/pdf/2602.17997)

**Central complex: Duan, Dong & Fiete, NeurIPS 2025 [conference]**
- **Model:** hemibrain head-direction circuit as a rate model. Synapse counts and neurotransmitter signs are fixed; only cell-type-shared gains Z^AB, biases and time constants are trained, with self-supervised and unsupervised objectives. [Duan et al. NeurIPS 2025](https://www.proceedings.com/content/085/085713-4273open.pdf)
- **Perturbations of the initial connectome:**
  - Small sign-preserving Gaussian noise still yields integrators.
  - Sign flips (p = 0.05) degrade performance.
  - Permuting weights within cell-type blocks, or across the whole connectome, "consistently failed to produce functional networks even after extensive training".
  - A single global gain, or separate E/I gains only, also fail.
- **Sweep:** 90 hyperparameter settings × 5 seeds per condition; the best 5 are kept. [Duan et al. NeurIPS 2025](https://www.proceedings.com/content/085/085713-4273open.pdf)
- **What it does not test:** no DP null and no trainable-weights-in-mask variant. The task (angular-velocity integration) is the circuit's native function.

**Larval connectome, frozen operator: Therianos 2026, arXiv 2606.17745 v2 [preprint, single author; nothing is trained]**
- **Setup:** 2,825-neuron strongly connected core run as a leaky-tanh rate operator, compared against a degree-and-weight-matched rewiring ensemble, a random graph and cell-class-preserving nulls.
- **Gross dynamics:** gain and effective dimensionality are "within a few percent of the ensemble".
- **Input routing:** under sparse afferent drive, activity is confined to about 1/5 of the core, versus about 2/3 for the DP ensemble and all of it for a random graph. The confinement survives the cell-class-preserving null.
- **Mushroom body:** it concentrates the leading driving modes beyond the ensemble.
- **Conclusion:** degree and weight statistics set the gross response; exact wiring sets routing and modes. [Therianos 2026](https://arxiv.org/abs/2606.17745)

**FlyWire as an echo-state reservoir: Costi et al. 2025, Biomimetics 10(5):341 [peer-reviewed, reservoir]**
- **Setup:** FlyWire, 104,909 neurons after filtering; subgraphs of N = 50-1,500 plus the full graph. Signs come from the neurotransmitter (GABA negative). Only a ridge readout is trained.
- **Task:** three-body (Earth-Moon) trajectory prediction.
- **Control:** a random sparse matrix with the same size and sparsity, plus hybrids (connectome topology with random weights, and the reverse). No DP null.
- **Statistics:** 50 test cases per configuration; Mann-Whitney tests.
- **Results:**
  - At ridge β = 10⁻⁶ the random control overfits and the connectome wins, increasingly with N.
  - At β = 10⁻³ the full-connectome configuration has the highest error, and the topology-only hybrid matches the control.
  - At spectral radius 0.25 the control is better.
  - The authors conclude that "weights matter more than topology". [Costi et al. 2025, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12109256/)

**Fly-connectome ESN: Morra & Daley 2022, arXiv 2206.02094 [preprint, reservoir, abstract only]**
- A fly-connectome reservoir beats a conventional ESN on Mackey-Glass, Lorenz and Rössler.
- Permuting the positions of non-zero weights, or raising clustering, increased variance and sometimes degraded performance. [arXiv 2206.02094](https://arxiv.org/abs/2206.02094)
- A companion paper: [arXiv 2201.09359](https://arxiv.org/pdf/2201.09359).

**Fly-inspired robot controllers without wiring nulls (2026) [preprints, abstract only]**
- Humanoid controller with 3,609 recurrent states: 61 of 63 conditions completed, versus 62 of 63 for a privileged reference. Resetting the motor state takes success from 19/21 to 0/21. No random or rewired controls in the abstract. [Guan et al. 2026](https://arxiv.org/abs/2609.27001)
- FlyCNS uses the brain-nerve-cord connectome as a "weak prior over communication allocation"; no wiring nulls in the abstract. [Zhang et al. 2026](https://arxiv.org/abs/2609.28816)

### Inferences
- Lappalainen's own simulation suggests that sparse wiring identifies function well (r = 0.85 at 10 % connectivity). That supports using the connectome as a mask only when the target is biological activity. It says nothing about whether the mask helps an alien task such as Navier-Stokes or Mario.
- Across fly studies, the strongest "connectome matters" evidence comes from:
  - a biology-native target (the circuit's own function or recorded activity);
  - very few trained parameters (per type);
  - non-DP shuffles as the null.
- Mosca's design breaks all three:
  - the task is alien;
  - all weights inside the mask are trained;
  - the null is DP.
- A tie in Mosca is therefore what this literature predicts, not a contradiction of it.
- FLYNN's single wiring control (Watts-Strogatz) appears to have been trained without the camera-dropout regime, so its robustness gap is confounded with training regime and cannot be attributed to wiring.
- Duan et al.'s failure under weight permutation is mostly a loss of sign and structure, not a test against a DP or sign-preserving null. Within-block permutation keeps the cell-type block structure but scrambles which neuron connects to which; the paper does not say whether signs survive it.

### Gaps
- I found no FlyWire mushroom-body or central-complex RNN where **all synaptic weights inside the mask are trained** and compared with ER, DP or dense controls. Eschbach et al. 2020 (Nat Neurosci) and Jiang & Litwin-Kumar 2021 (PLOS CB) build connectome-informed MB models, but I could not verify whether they include random-wiring controls. [Jiang & Litwin-Kumar 2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8354444/)
- Not found: any published follow-up retraining flyvis with a DP-rewired connectome at matched training budget, other than Dhiman 2026.
- The Hugging Face "FlyWire olfactory SNN" model card (masked recurrent LIF, weight magnitudes learned) has no paper or controls that I could see. [HF card](https://huggingface.co/MIRE-org/flywire-olfactory-snn)

---

## Q2. C. elegans: Elegans-AI, Neural Circuit Policies, and other worm-connectome deep nets

### Takeaway
The worm literature splits in two:
- **ML-architecture papers** (ElegansNet / Elegans-AI; Schmidgall et al.) report large advantages over random graphs, but with weak or unspecified nulls, unreported seeds and, in Elegans-AI, implausible headline accuracies.
- **Neuroscience papers** (Mi et al. ICLR 2022; Creamer, Leifer & Pillow 2024) find that the connectome helps predict recorded worm activity. There the target is biological.

The one large reservoir benchmark with outdegree-preserving nulls (Reimers et al. 2026) finds that random nulls often win.

### Cited Findings

**ElegansNet: Bardozzo, Terlizzi, Liò & Tagliaferri 2023, arXiv 2304.13538 [preprint, "preliminary report"]**
- **Setup:** the C. elegans connectome is converted into a graph of learnable tensors.
- **Claims:** "quickly reaches the 99.99% of accuracy" on CIFAR-10 validation, "outperforming all 30 models based on randomized small-world graphs" (Watts-Strogatz, ER and Barabási-Albert generators), and 99.8 % on unsupervised MNIST.
- **Caveats:**
  - Watts-Strogatz models perform "slightly similar" to ElegansNet.
  - The report itself states "there are no details on the evolutionary optimization and how it is structured".
  - No seeds or statistics are given. [ElegansNet](https://arxiv.org/pdf/2304.13538)
- In a separate rewiring analysis, the "multi-dyadic" motif signature is lost after "even a slight rewiring". That is a graph-statistics result, not a performance result. [ElegansNet](https://arxiv.org/pdf/2304.13538)

**Elegans-AI: Bardozzo, Terlizzi, Simoncini, Lió & Tagliaferri 2024, Neurocomputing 584:127598 [peer-reviewed; read only abstract and secondary listings]**
- Claims superiority over "randomly rewired artificial connectomes" and "simulated bio-plausible" connectomes.
- Reports top-1 accuracy of 99.99 % on both CIFAR-10 and CIFAR-100, and 99.84 % on unsupervised MNIST.
- The full text was not reachable. [Scholar9 listing](https://scholar9.com/publication-detail/elegans-ai-how-the-connectome-of-a-living-organis--28731); [UNISA IRIS](https://www.iris.unisa.it/handle/11386/4862891)

**Schmidgall, Schuman & Parsa 2022, arXiv 2209.14406 [preprint]**
- **Wiring:** the C. elegans motor circuit (75 motor and 95 body-wall-muscle neurons cited; 3.2 % sparsity).
- **Dale:** enforced as a fixed sign × exp(log-weight).
- **Training:** connectome weights evolved by ES pre-training followed by a genetic algorithm; the encoder and decoder are also trained.
- **Controls:**
  - A "statistics network" with the same neuron and synapse counts and the same E/I ratio, but different wiring. It is not stated to be DP.
  - A fully connected 2-hidden-layer MLP with the same neuron count.
  - A random sparse MLP.
- **Swimmer final scores:** C. elegans statistics network 146.4 ± 3.5, fully connected MLP 96.2 ± 13.6, random sparse MLP 63.7 ± 17.2. The text says "the precise circuitry of the C. Elegans connectome remains superior" to the statistics network.
- **Other tasks:** on HalfCheetah and MNIST the connectome net is worse than the MLP. Performance is "dramatically hindered compared with more densely connected general architectures".
- **Seeds:** number of runs not reported. [Schmidgall et al. 2022 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2209.14406)

**Neural Circuit Policies: Lechner et al. 2020, Nat Mach Intell 2:642-652 [peer-reviewed]**
- A 19-neuron, 253-synapse controller whose wiring is *designed* in the style of the worm (sensory, inter, command and motor layers). It is not a measured connectome.
- Claims better generalizability, interpretability and robustness than much larger black-box models. [Papers with Code](https://paperswithcode.com/paper/neural-circuit-policies-enabling-auditable)
- Its predecessor used the actual tap-withdrawal circuit and found policies that "perform as well as deep neural network policies". [arXiv 1803.08554](https://arxiv.org/abs/1803.08554v1)
- I could not confirm any random-wiring control in either paper.

**Connectome-constrained latent variable model: Mi, Xu, Prakhya, Lin, Shavit, Samuel & Turaga, ICLR 2022 [conference]**
- **Model:** the connectome fixes sparsity ("connectome sparsity"), or also the synapse-count magnitudes ("connectome count", where only the global scales α are trained).
- **Comparisons:** a learned dense model, an L1-sparse model, and a total-count-regularized model. There is no shuffled or rewired-connectome control.
- **Neuron-holdout correlation** (4 random initializations; conductance-based models):

  | Model | Correlation |
  |---|---|
  | Connectome count₂ | 0.319 ± 0.016 |
  | Connectome count₁ | 0.318 |
  | Connectome sparsity | 0.262 |
  | Learned dense | 0.081 |
  | Learned sparse | 0.008 |
  | Learned total count | −0.033 |

- **Worm-holdout correlation:** all 9 variants fall within 0.439-0.474. The appendix says "Our connectome-constrained models do not outperform other unconstrained baselines" on that test.
- **Parameter-count argument:** because L1-sparse loses, the authors argue that "the connectome constraint improves predictability due to its topology, not due to the number of trainable parameters". [Mi et al. ICLR 2022 (Janelia PDF)](https://www.janelia.org/sites/default/files/connectome_constrained_latent.pdf)

**Creamer, Leifer & Pillow 2024, bioRxiv 2024.09.22.614271 [preprint]**
- **Model:** a linear dynamical system whose weights are learned only on connectome pairs (about 9.4 % of pairs), fit by EM to the whole-brain optogenetic perturbation atlas (156 head neurons, 110 animals; 80 train, 30 test).
- **Label-permutation null** (neuron identities shuffled, topology and sparsity kept): "performed poorly" on both causal and correlational measures.
- **Unconstrained model:** did not predict perturbation responses (STAMs) better, though it had a higher test log-likelihood.
- **Fit quality:** STAM correlation 0.15 against a train-test noise ceiling of 0.16, i.e. 0.92 relative.
- **Strength of the null:** label permutation is a strong null for activity prediction, since the graph is identical, but it is not a DP rewiring. [Creamer et al. 2024](https://www.biorxiv.org/content/10.1101/2024.09.22.614271v1.full)

**Reimers, Ramstad, Hubin & Nichele 2026, arXiv 2609.30508 [preprint, reservoir]**
- **Setup:** 22 Witvliet et al. connectomes (8 animals, 7 ages, 3 measurement methods, 180 neurons each) used as ESN reservoirs.
- **Nulls:**
  - wiring randomized with the outdegree distribution kept;
  - input/output nodes randomized;
  - both randomized.
- **Results:**
  - The abstract: randomized nulls "often outperform the original connectomes".
  - Detailed effects (seen only via a blog summary, flagged **[secondary]**): memory capacity is the only task where the connectome beats all nulls (+0.305 / +0.657 / +0.951); random wiring wins on the Hénon map (−1.342 vs the wiring null).
  - Input/output placement matters more than wiring. [arXiv abstract](https://arxiv.org/abs/2609.30508); [blog summary](https://blog.pebblous.ai/blog/celegans-connectome-reservoir-benchmark/en/)

### Inferences
- Elegans-AI's 99.99 % top-1 on CIFAR-100 lies far above any published CIFAR-100 result I know of, including large pretrained models in the mid-90s. Combined with the preprint's admission that the optimization is undocumented, its "beats random rewiring" claim should not be cited as evidence without independent replication. This is my judgement, not a published rebuttal.
- Schmidgall's "statistics network" is the closest worm analogue to Mosca's ER control: same counts and E/I ratio, different wiring. It reports a connectome win on Swimmer, but with no seed count and a GA optimizer, the 146.4 vs "statistics" comparison cannot be assessed statistically. The paper's own non-motor results (connectome worse than MLP on MNIST and HalfCheetah) match Mosca's alien-task tie or loss.
- In the worm, the connectome clearly helps when the target is the worm's own activity (Mi, Creamer). It does not clearly help on generic benchmarks (Reimers).

### Gaps
- Full text of Elegans-AI (Neurocomputing 2024) not accessed: the exact rewiring null (DP or not), seeds, and whether pretrained backbones were used are all unverified.
- NCP 2020: I could not verify whether a random-wiring NCP control exists in the paper.
- I found no worm-connectome network trained on a non-biological task with a DP null and multiple seeds.

---

## Q3. Mammal and human: bio2art (Goulas, Damicelli & Hilgetag 2021) and follow-ups; "connectomes as ANN architecture"; MICrONS as architecture

### Takeaway
The flagship mammalian test, bio2art (Goulas et al. 2021), is an explicit **null result**: macaque, marmoset and human topologies give "no enhancement of performance when compared to randomly wired RNNs". The reservoir follow-up (Damicelli et al. 2022) is also a tie, and adds that some randomness is needed. Positive mammalian results come from:
- a reservoir near criticality (Suárez et al. 2021);
- a statistical, non-EM V1 wiring rule trained under Dale (Chen, Scherr & Maass 2022), against random controls that also change other factors;
- a 2026 MICrONS preprint whose own permuted-initialization control performs about as well as the true one.

I found no study using the MICrONS EM synapse graph as a trainable mask with ER or DP controls.

### Cited Findings

**bio2art: Goulas, Damicelli & Hilgetag 2021, Neural Networks [peer-reviewed; bioRxiv 2021.01.22.427744 read]**
- **Wiring:** marmoset, macaque and human area-level connectomes, expanded to N = 4 neurons per region (between-region connection probability 0.8, within-region all-to-all).
- **Masking:** W_hh = W_bio ⊙ W_init, and "only the non-zero weights are modified" in training, so the mask is effectively fixed. Input and output are all-to-all.
- **Tasks:** sequence memory (L = 3, 5, 10) and N-back (n = 2, 3, 4).
- **Controls:** "r_" networks made by "shuffling the connection weights" of each bio network. DP preservation is not stated.
- **Runs:** 12 hyperparameter combinations × 5 runs; no formal statistics.
- **Result:** "no enhancement of performance when compared to randomly wired RNNs". One mapping (b_w_eq) is worse. The bio topology "does not serve … as an advantageous structural prior". [Goulas et al. 2021](https://www.biorxiv.org/content/10.1101/2021.01.22.427744.full)

**Damicelli, Hilgetag & Goulas 2022, PLOS Comput Biol 18(11):e1010639 [peer-reviewed, reservoir]**
- **Setup:** the same macaque, marmoset and human connectomes as ESNs; only the readout is trained.
- **Nulls:**
  - Bio (no-rank): empirical topology, random weights;
  - Random (density): density-matched random wiring;
  - Random (k = 10);
  - Random (full).
  - No DP null.
- **Runs:** 100 networks per condition (per the figure legends).
- **Results:**
  - Bio (no-rank) ≈ random.
  - Bio (rank), with weights rank-matched to connectome strengths, is the only significantly worse condition, on both memory capacity and sequence recall, for all three connectomes.
- **Conclusion:** biological wiring "can achieve the same performance as the purely random networks", and "a certain level of randomization is actually necessary". [Damicelli et al. 2022, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9710781/)

**Suárez, Richards, Lajoie & Mišić 2021, Nat Mach Intell 3:771-786 [peer-reviewed, reservoir]**
- **Setup:** human diffusion-MRI connectomes as reservoirs, tested against "two null models".
- **Result:** the "underlying topology and modular structure of the brain enhance memory capacity in the context of critical dynamics". The advantage is regime-dependent, not general. [Suárez et al. preprint](https://www.biorxiv.org/content/10.1101/2020.11.10.350876.full.pdf)
- A 2024 CCN abstract by the same group reports a "dramatic difference between connectomes and their rewired variants" for multifunctionality. [CCN 2024](https://2024.ccneuro.org/pdf/288_Paper_authored_CCN.pdf)

**Chen, Scherr & Maass 2022, Science Advances [peer-reviewed]**
- **Model:** the Allen/Billeh mouse V1 model, 51,978 GLIF neurons in 111 types with mean in-degree 278. Connection probabilities depend on cell class, layer and distance, so this is a **statistical wiring rule, not an EM graph**.
- **Training:** recurrent and LGN→V1 weights are trained; "No synaptic connections were added or deleted" and signs are fixed (Dale).
- **Tasks and performance:** five visual tasks, average accuracy 94.28 %.
- **Controls:**
  1. no LGN;
  2. no neuron diversity;
  3. no laminar structure (random connections, same count);
  4. a random recurrent spiking network without Dale;
  5. the same random spiking network with Dale.
- **Control results:**
  - The controls learn more slowly and plateau.
  - Removing laminar structure hurts more than removing neuron diversity.
  - Control 5 is worse than control 4.
- **Seeds and numbers:** 5 runs per model, SEM only. Control accuracies are given only in figures, not in the text. [Chen et al. 2022, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9629744/)

**MICrONS-informed RNNs: Shakiba, Rokni, Mohammadi & Dehghani 2026, arXiv 2606.14975 [preprint]**
- **Data used:** 312-unit RNNs from one MICrONS field. Weights are *initialized* from functional correlation and STTC matrices (W*), real soma coordinates (D*) feed a spatial regularizer, and a communicability regularizer is added (C, or C* from EM connectivity). The EM synapse graph is not used as a mask.
- **Tasks:** One-Choice Inference, Perceptual Decision-Making and Go/NoGo.
- **Statistics:** 20 runs per model, 10 epochs; Holm-corrected Mann-Whitney tests.
- **Accuracy (Task 1 / Task 2 / Task 3):**

  | Model | Task 1 | Task 2 | Task 3 |
  |---|---|---|---|
  | W*D*C* (full model) | 0.985 | 0.880 | 0.951 |
  | Simple-RNN baseline | 0.625 | ≈0.50 | ≈0.50 |
  | W!D*C (**permuted** biological initial weights) | 0.950 | 0.866 | 0.951 |
  | W!D*C* (permuted) | 0.963 | 0.847 | 0.951 |

- **Controls not run:** DP, ER-mask, and explicit parameter matching. [Shakiba et al. 2026](https://arxiv.org/html/2606.14975v1)

**Dale + sparse topology in trained RNNs: Balwani, Wang, Choi & Najafi 2025, bioRxiv 2025.01.09.632231 → Science Advances [peer-reviewed]**
- Dale's backprop with "top-prob" pruning reaches a target sparse connectivity, and "our models match the performance of RNNs trained without any constraints".
- Top-prob pruning beats random pruning.
- It is applied to Allen Visual Behavior data with cell-type connectivity constraints. This is a method for imposing constraints, not a test of a connectome against random wiring. [Balwani et al. bioRxiv](https://www.biorxiv.org/content/10.1101/2025.01.09.632231v1.full); [Sci Adv](https://www.science.org/doi/10.1126/sciadv.adw4970)

**Graph structure, not a specific connectome**
- **You, Leskovec, He & Xie, ICML 2020.** Performance of "relational-graph" networks is a smooth function of clustering and average path length, with a sweet spot whose graph statistics resemble biological networks. These are statistics, not specific wiring. [PMLR](https://proceedings.mlr.press/v119/you20b.html)
- **Talpir & Schneidman 2026, arXiv 2606.23874 [preprint, abstract only].** Across many trained RNNs computing Boolean functions, local 2- and 3-cycles "strongly enhance" computational ability, "outperforming acyclic or reachability-matched controls". A "small set of structural statistics" predicts performance. [Talpir & Schneidman 2026](https://arxiv.org/abs/2606.23874)

### Inferences
- In Shakiba et al., permuting the biological initial weights (W!) recovers nearly all of the gain over the baseline. The benefit therefore appears to come from the weight-magnitude distribution or scale and from the spatial regularizer, not from which MICrONS neuron connects to which.
- The Simple-RNN baseline sits at chance on two of three tasks after 10 epochs. That suggests an under-tuned baseline (orthogonal init, short training), which inflates every "constrained beats baseline" gap. It is the same weak-null pattern Dhiman 2026 diagnosed in flyvis.
- Chen et al.'s "no laminar structure" control keeps the synapse count but redraws connections without the class, layer and distance rule, so it is closer to ER than to DP. Its advantage is about cell-type and spatial structure (which a DP null would also destroy), not about exact wiring.
- Talpir & Schneidman suggest reciprocity (2-cycles) and 3-cycles are the structural features that matter for trained RNN capacity. Standard Maslov-Sneppen DP rewiring does not preserve reciprocity. If the fly subgraph has excess reciprocity, a DP control in Mosca also destroys it. Checking the reciprocity of Mosca's fly and rewired masks would locate where any difference could come from.

### Gaps
- Not found: any network using the **MICrONS EM synapse graph** (cell-to-cell) as a trainable mask with ER, DP or dense controls. The only MICrONS-based trained RNN found uses functional correlations for initialization.
- Not found: human or macaque connectome RNNs trained on cognitive tasks with a DP null and gradient training of all weights in the mask, beyond bio2art (whose shuffled-weight null is not stated to be DP).
- The details of Suárez et al.'s two nulls (DP? spatially constrained?) were not verified from the full text.
- arXiv 2412.14999 (human functional sub-circuit ESNs with a Maslov-Sneppen null) surfaced, but I did not read its results. [arXiv 2412.14999](https://arxiv.org/pdf/2412.14999)

---

## Q4. Does the claimed advantage survive (a) a degree-preserving null, (b) parameter matching, (c) shared initialization, (d) adequate seeds and statistics? Which studies report NULL results?

### Takeaway
No trained connectome-constrained network in this inventory shows an advantage that survives all four tests at once. The studies that apply a DP null and/or shared initialization report:
- a vanishing advantage (Dhiman 2026);
- a small or domain-specific one (Therianos 2026, frozen);
- or random nulls winning (Reimers 2026, reservoir).

Positive claims cluster where nulls are weak (WS/ER, weight shuffles, untuned baselines), seeds are unreported, or the task is the circuit's own biological function. Explicit nulls or ties are common and are the majority among studies that set out to test topology.

### Cited Findings
- **Explicit null or tie results:**
  - **Goulas et al. 2021 (trained RNN, mammalian):** "no enhancement of performance when compared to randomly wired RNNs". [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.01.22.427744.full)
  - **Damicelli et al. 2022 (reservoir):** biological wiring "can achieve the same performance as the purely random networks"; rank-weighted biological is worse. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9710781/)
  - **Reimers et al. 2026 (reservoir, worm, outdegree-preserving null):** nulls "often outperform the original connectomes". [arXiv](https://arxiv.org/abs/2609.30508)
  - **Kuklev 2026 (frozen flyvis, non-biological task):** "not an advantage of the biological wiring"; receptors-only beats the full network by 0.035. [arXiv](https://arxiv.org/html/2609.24565v1)
  - **Mi et al. 2022, worm-holdout test:** "connectome-constrained models do not outperform other unconstrained baselines". [Janelia PDF](https://www.janelia.org/sites/default/files/connectome_constrained_latent.pdf)
  - **Costi et al. 2025:** the connectome advantage disappears at stronger regularization (β = 10⁻³) and at low spectral radius, where the random control wins. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12109256/)
  - **Schmidgall et al. 2022:** the connectome is worse than an MLP on MNIST and HalfCheetah. [ar5iv](https://ar5iv.labs.arxiv.org/html/2209.14406)
  - **Dhiman 2026 (already read):** the advantage largely vanishes under shared init plus a DP null. [arXiv](https://arxiv.org/html/2604.04033v1)
- **(a) Degree-preserving null, applied to a trained or frozen connectome:**
  - Dhiman 2026: the advantage vanishes.
  - Therianos 2026 (frozen, no task): gross dynamics are within a few percent of the DP ensemble; routing and mushroom-body modes differ. [arXiv](https://arxiv.org/abs/2606.17745)
  - Reimers 2026: outdegree-preserving null, mixed or negative.
  - FlyGM: DP-like baselines, but unweighted unit-strength (already read).
  - In every other study listed here, the null was either a weight shuffle, a label permutation, ER, Watts-Strogatz, "same counts" or an unspecified "random rewired" graph.
- **(b) Parameter matching:**
  - FLYNN matches parameter count only against CNN+GRU baselines (~5 M). [FLYNN](https://arxiv.org/html/2607.00025v2)
  - Mi et al. argue topology rather than parameter count, via an L1-sparse model. [Janelia PDF](https://www.janelia.org/sites/default/files/connectome_constrained_latent.pdf)
  - Schmidgall's MLP has the same neuron count but not the same parameter count. [ar5iv](https://ar5iv.labs.arxiv.org/html/2209.14406)
  - Chen et al.'s controls keep the synapse count. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9629744/)
  - No study found matches a dense RNN to the connectome's parameter count, as Mosca's "denso" control (420 neurons) does.
- **(c) Shared initialization:**
  - Only Dhiman 2026 tests it explicitly.
  - Shakiba et al.'s permuted-init control (W!) is a related check, and it shows the true initialization adds little. [arXiv](https://arxiv.org/html/2606.14975v1)
- **(d) Seeds and statistics:**

  | Study | Seeds / runs | Statistics |
  |---|---|---|
  | Shakiba 2026 | 20 runs | Holm-corrected Mann-Whitney |
  | Damicelli 2022 | 100 networks | — |
  | Costi 2025 | 50 test cases | Mann-Whitney |
  | Reimers 2026 | 15 redraws | Beta regression |
  | Duan 2025 | 5 seeds × 90 hyperparameter settings | — |
  | Chen 2022 | 5 runs | SEM only |
  | Mi 2022 | 4 initializations | — |
  | Lappalainen 2024 | 50 models in the main ensemble | — |
  | Goulas 2021 | 5 runs | none |
  | FLYNN | not reported | — |
  | Schmidgall 2022 | not reported | — |
  | ElegansNet / Elegans-AI | not reported | — |
  | Kuklev 2026 | single fly checkpoint | — |

  Sources: [Shakiba](https://arxiv.org/html/2606.14975v1), [Damicelli](https://pmc.ncbi.nlm.nih.gov/articles/PMC9710781/), [Costi](https://pmc.ncbi.nlm.nih.gov/articles/PMC12109256/), [Reimers (secondary)](https://blog.pebblous.ai/blog/celegans-connectome-reservoir-benchmark/en/), [Duan](https://www.proceedings.com/content/085/085713-4273open.pdf), [Chen](https://pmc.ncbi.nlm.nih.gov/articles/PMC9629744/), [Mi](https://www.janelia.org/sites/default/files/connectome_constrained_latent.pdf), [Lappalainen](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/), [Goulas](https://www.biorxiv.org/content/10.1101/2021.01.22.427744.full), [FLYNN](https://arxiv.org/html/2607.00025v2), [Schmidgall](https://ar5iv.labs.arxiv.org/html/2209.14406), [ElegansNet](https://arxiv.org/pdf/2304.13538), [Kuklev](https://arxiv.org/html/2609.24565v1)

### Inferences
- Pattern across the inventory: the stronger the null (WS/ER → weight shuffle → label permutation → DP with shared init and training budget), the smaller the reported connectome advantage. Advantages that survive strong nulls are found only when the target is the circuit's own biological function or activity (Creamer's label-permutation null; Duan's within-block permutation; Lappalainen's knowledge ablations). That is exactly what Mosca's alien tasks lack.
- Mosca's pre-registered H0 (fly ties random and rewired) agrees with every study here that used a DP null or a well-tuned random control.

### Gaps
- No study found reports a power analysis or a minimum detectable effect for a connectome-vs-random comparison, so "tie" results elsewhere are seldom bounded. Mosca's per-level and per-seed permutation tests would be unusual in this literature. This is an inference from absence, not a cited fact.

---

## Q5. Is there any paper with Mosca's exact design (connectome mask + Dale signs from the connectome + all weights trainable inside the mask + ER + degree-preserving + dense parameter-matched + no-core controls) on a non-biological task?

### Takeaway
**None found.** No study in 2019-2026 combines all of Mosca's elements. The closest partial matches each lack at least two of them:
- **FLYNN:** FlyWire mask with trainable weights on a robot task, but no Dale, only a Watts-Strogatz wiring control, no DP, no dense or no-core control, seeds unreported.
- **Schmidgall 2022:** worm mask, Dale, trained weights, a "statistics" network and MLPs, but evolutionary training, no DP, no seeds reported.
- **Goulas 2021:** mammalian mask, trained weights, weight-shuffle null, memory tasks, but no Dale, no DP, no dense or no-core control.
- **Chen 2022:** Dale and trained weights inside fixed wiring with random-wiring controls, but a statistical V1 rule, biological visual tasks, and no DP.
- **Dhiman 2026:** DP plus shared init, but on the flyvis biological task with very short training.
- **Kuklev 2026:** a non-biological task, but frozen, with no wiring nulls.

The **no-core control** (zero recurrent connections), the most diagnostic control in Mosca's results, does not appear in any study found. Kuklev's "receptors-only" baseline is the nearest frozen-network analogue.

### Cited Findings

Inventory of the closest designs (✔ = present, ✗ = absent or not reported, ~ = partial):

| Study | Connectome | Dale from data | Weights in mask trained | ER | DP | Dense / param-matched | No-core | Non-bio task | Seeds |
|---|---|---|---|---|---|---|---|---|---|
| [FLYNN 2026](https://arxiv.org/html/2607.00025v2) | FlyWire v783, whole brain | ✗ | ✔ (plus leaks) | ✗ (WS only) | ✗ | ~ (CNN+GRU ~5 M) | ✗ | ✔ robot navigation | ✗ |
| [Schmidgall 2022](https://ar5iv.labs.arxiv.org/html/2209.14406) | Worm motor circuit | ✔ (sign × exp) | ✔ (GA/ES) | ~ ("statistics net") | ✗ | ~ (MLP, same neuron count) | ✗ | ✔ Swimmer, HalfCheetah, MNIST | ✗ |
| [Goulas 2021](https://www.biorxiv.org/content/10.1101/2021.01.22.427744.full) | Macaque, marmoset, human (areas) | ✗ | ✔ | ~ (weight shuffle) | ✗ | ✗ | ✗ | ✔ sequence memory, N-back | 5 runs |
| [Chen 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9629744/) | Mouse V1 statistical rule | ✔ | ✔ | ✔ (random, same count) | ✗ | ✗ | ✗ | ~ visual tasks incl. MNIST | 5 |
| [Mi 2022](https://www.janelia.org/sites/default/files/connectome_constrained_latent.pdf) | Worm, whole | ✗ (signs unknown) | ✔ (sparsity variant) | ✗ | ✗ | ✔ (learned dense) | ✗ | ✗ (worm activity) | 4 |
| [Creamer 2024](https://www.biorxiv.org/content/10.1101/2024.09.22.614271v1.full) | Worm head | ✗ | ✔ (linear) | ✗ | ~ (label permutation) | ✔ (unconstrained) | ✗ | ✗ (worm perturbations) | 110 animals |
| [Duan 2025](https://www.proceedings.com/content/085/085713-4273open.pdf) | Hemibrain head-direction circuit | ✔ | ✗ (cell-type gains only) | ✗ | ✗ (weight permutation) | ✗ | ✗ | ✗ (head-direction integration) | 5 × 90 |
| [Lappalainen 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/) | Fly optic lobe, 64-65 cell types (EM source not re-verified here) | ✔ | ✗ (734 type-level parameters) | ✗ | ✗ | ~ (CNN lower bound) | ✗ | ✗ (optic flow) | 50 |
| [Shakiba 2026](https://arxiv.org/html/2606.14975v1) | MICrONS (functional init) | ✗ | ✔ (init only, no mask) | ✗ | ✗ | ✗ | ✗ | ~ cognitive tasks | 20 |
| [Kuklev 2026](https://arxiv.org/html/2609.24565v1) | flyvis (frozen) | ✔ (inherited) | ✗ | ✗ | ✗ | ~ (small CNNs) | ~ (receptors-only) | ✔ architectural drawings | 1 checkpoint |
| [Costi 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12109256/) | FlyWire (reservoir) | ✔ | ✗ (readout only) | ✔ (same sparsity) | ✗ | ✗ | ✗ | ✔ three-body prediction | 50 cases |
| [Reimers 2026](https://arxiv.org/abs/2609.30508) | Worm, 22 connectomes (reservoir) | ✗ | ✗ (readout only) | ✗ | ~ (outdegree kept) | ✗ | ✗ | ~ chaotic and cognitive | 15 redraws |
| [Dhiman 2026](https://arxiv.org/html/2604.04033v1) (already read) | flyvis | ✔ | ~ (flyvis parameters) | ✔ | ✔ | ✗ | ✗ | ✗ | 5 rewired graphs |
| [Jin et al. FlyGM 2026](https://arxiv.org/pdf/2602.17997) (already read) | FlyWire, whole brain | — | ✗ (fixed W) | ✔ (unit strength) | ✔ (unit strength) | — | ✗ | ✗ (fly locomotion) | — |

- Searches combining "connectome-constrained", "Dale", "Erdős-Rényi", "degree-preserving" and "dense parameter-matched" (arXiv, bioRxiv, general web, 20-10-2026) returned methodological Dale/sparsity papers ([Balwani et al. 2025](https://www.science.org/doi/10.1126/sciadv.adw4970)) and the Dhiman 2026 preprint, but no study with Mosca's full control set.

### Inferences
- Mosca's control set is stricter than anything published:
  - a single shared subgraph;
  - an ER control with the same density and inhibitory fraction;
  - a DP control;
  - a dense control matched to 420 neurons;
  - fixed port overlap;
  - no-core and MLP controls;
  - multiple training seeds;
  - permutation tests.
- A null result from Mosca would therefore be among the cleanest available tests of "connectome as inductive bias for an alien task". It would extend the null pattern of Goulas 2021, Damicelli 2022 and Dhiman 2026 to:
  - FlyWire;
  - trained weights inside the mask;
  - Dale signs taken from the data;
  - non-biological dynamics (Navier-Stokes) and RL.
- The finding that `sin_nucleo` ties the fly core has no published counterpart. No study found includes a no-recurrence control, so nobody else has checked whether their tasks use the recurrent wiring at all. This is a methodological point Mosca can make. (Inference from absence across the papers read.)

### Gaps
- Full texts of Elegans-AI (2024), NCP (2020), Eschbach 2020 and Jiang & Litwin-Kumar 2021 were not read: Elegans-AI could not be reached, and the other three were not attempted after the abstract-level searches. Their control definitions remain unverified.
- Rate limits (bioRxiv HTTP 429) blocked a second pass on some preprints.
- Several 2026 preprints (FLYNN, FlyCNS, humanoid controller, Talpir & Schneidman) were read at abstract level only, or partially. Their full methods may contain controls not reflected here.
