# Connectomes as reservoirs or fixed recurrent substrates, compared with null graphs, and connectome-based forecasting of chaotic series, PDEs and fluids (2019–2026)

Status legend used below: **[PR]** peer-reviewed journal; **[CONF]** conference paper or abstract; **[PRE]** preprint, not peer-reviewed; **[COMM]** community or hobby project, not peer-reviewed; **[BG]** background, pre-2019 or not connectome-based. Today's date: 2026-10-20.

## Q1. Which papers use connectomes as reservoirs or fixed recurrent networks, what nulls do they use, and who wins (effect sizes, seeds, statistics)?

### Takeaway
There are about 20 academic studies. Advantages for the connectome appear mostly against weak controls: a conventional ESN with a different density or weight distribution, a full weight shuffle, or an ER graph. Against degree-preserving rewiring the picture is mixed. The connectome wins in the mammalian mesoscale memory-capacity work, but only at criticality (Suárez 2021; conn2res 2024). It also wins in Morra's multifunctionality abstract (2024) and in two 2026 fly preprints that train weights or use anatomical I/O (Cho 2026; Jin 2026). It ties or loses in most other cases: the fly mesoscale connectome in conn2res, all 22 *C. elegans* connectomes (Reimers 2026), flyvis (Dhiman 2026), fly hashing (Senge 2026), the McAllister fly subgraphs, and every mesoscale connectome in Damicelli 2022 and Goulas 2021.

### Cited Findings

**Mammalian / mesoscale connectome reservoirs (Mišić group and others)**
- **Suárez, Richards, Lajoie & Mišić 2021, *Nat Mach Intell* 3:771–786 [PR]** ("Learning function from structure in neuromorphic networks") — [journal ref](https://colab.ws/articles/10.1038/s42256-021-00376-1?locale=en); details from the [bioRxiv full text](https://www.biorxiv.org/content/10.1101/2020.11.10.350876v1.full):
  - Data: human diffusion-spectrum MRI from 66 subjects. Consensus connectome of 1,015 nodes (1,000 cortical + 15 subcortical), binary density 2.5%. Replicated at 448+15 nodes.
  - Task: memory capacity (MC). Input through subcortical nodes, readout from the 7 Yeo intrinsic networks. α (spectral radius) swept from 0.3 to 3.5.
  - Nulls: (i) "rewired": edge-pair swaps preserving size, density and degree sequence, 1,000 networks; (ii) spatially constrained spin permutation of the intrinsic-network labels.
  - At criticality (α=1), median MC is 0.92 (empirical) vs 0.80 (rewired), P_rewired < 10⁻⁴, effect size 99%. Against the spin null, 0.92 vs 0.89, P_spin < 10⁻⁴, effect size 88%.
  - The paper also says: "In the stable and chaotic regimes, randomly rewired networks outperform the empiricallly-derived architecture" [sic].
  - Normalised by wiring cost, the empirical connectome beats the rewired networks in all three regimes.
  - At 448+15 nodes: 0.89 vs 0.80 (P = 10⁻⁴, effect size 95%).
- **Suárez et al. 2024, *Nat Commun* 15:656 (conn2res toolbox) [PR]** — [PMC10803782](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10803782/):
  - Human consensus connectome (66 subjects) on MC, compared with "500 rewired nulls that preserve network density and degree sequence" (Maslov–Sneppen). At α=1 the empirical connectome has significantly higher MC.
  - Cross-species MC at peak α:
    - fruit fly: **p = 0.11 at α_peak = 0.9 (not significant)**
    - mouse: p < 0.002 at α_peak = 1.05
    - rat: p < 0.002
    - macaque: p < 0.002 (α_peak = 1.15)
  - The fly network is the mesoscale FlyCircuit reconstruction of Chiang et al. 2011 (Curr Biol), not an EM connectome.
  - Also in this paper: on perceptual decision-making with tanh units, performance falls from stable to chaotic dynamics. On a pattern-discrimination task it rises as dynamics turn chaotic.
- **Damicelli, Hilgetag & Goulas 2022, *PLoS Comput Biol* 18(11):e1010639 [PR]** — [PMC9710781](https://pmc.ncbi.nlm.nih.gov/articles/PMC9710781/):
  - Connectomes: macaque (Markov 2012), marmoset (Majka 2016), human (Betzel & Bassett 2018), scaled ×1–30 into "BioESNs". Reservoir weights uniform in [−1,1], fixed; only the readout is trained.
  - Hyperparameters swept, then fixed at ρ=0.99, ε=1e-5, α=1, b=1.
  - Tasks: memory capacity and sequence recall.
  - Nulls: "Random (density)" (same density), "Random (k)" (k=10 connections per neuron), "Random (full)" (dense); also "Bio (no-rank)" (empirical topology with random weight placement) and "Bio (rank)" (empirical weight ranks).
  - 100 networks per condition; no statistical tests or p-values reported.
  - Result is parity: "biologically inspired networks perform as well as classical echo state networks". Bio (rank) is worse and only catches up when scaled up with heterogeneous interareal weights.
  - Conclusion: "a degeneracy of structure and function".
  - No degree-preserving null.
- **Goulas, Damicelli & Hilgetag 2021, *Neural Networks* 142:608–618 [PR]** ("Bio-instantiated recurrent neural networks"; companion library bio2art) — [bio2art repo](https://github.com/AlGoulas/bio2art), [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.01.22.427744):
  - RNNs instantiated with human and monkey topologies, evaluated on test performance and training speed in working-memory tasks.
  - Abstract: "we observe no enhancement of performance when compared to randomly wired RNNs".
  - bio2art scales macaque (29 regions), marmoset (55), mouse (56), human and fly connectomes up to neuron level.
- **Qian, Yan, Wu, She & Yu 2025, bioRxiv 10.1101/2025.10.29.685101 [PRE]** — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.10.29.685101.full.pdf):
  - Fly, mouse, rat and macaque connectomes taken from conn2res, used as ESN reservoirs; Lorenz forecasting (MSE) and MC.
  - Control: "shuffled controls, in which the connection weights were randomized while preserving the overall weight distribution". This destroys topology: shuffled small-worldness is ~0.2.
  - At α=1, biological networks have lower Lorenz MSE in all species (paired t-test, p < 0.01). n = 10 networks per species.
  - Macaque is best, and the advantage correlates with small-worldness.
  - No degree-preserving null.
- **Guragain, Kakalis & Godino-Llorente 2026, arXiv:2606.09902 [PRE]** — [arXiv](https://arxiv.org/abs/2606.09902):
  - Six species: *C. elegans* 279 neurons, *Drosophila* 49 nodes, mouse 112, rat 73, macaque 29, human 83.
  - Tasks: MC, Lorenz, NARMA-10, Mackey-Glass.
  - Gradient-free optimisation of edge weights. Starting from biological weights beats starting from random weights on the same topology.
  - Up to 17× MC (C. elegans 1.39 → 23.91) and up to 89% NRMSE reduction (Mackey-Glass, human).
  - No topology null (no rewired or ER graphs).
- **Morra 2024, CCN abstract [CONF]** — [CCN 2024 PDF](https://2024.ccneuro.org/pdf/288_Paper_authored_CCN.pdf):
  - 18 MaMI mesoscale connectomes (N = 300) in a continuous-time RC on the "seeing double" multifunctionality task.
  - Nulls: rewired "by randomly swapping edge pairs while retaining the size, sparsity, and degree sequence", with the set of weights preserved.
  - 30 sets of 30 trials. "nearly all connectomes significantly outperform their rewired variants, with the exception of the Nasua3 network" (Wilcoxon signed-rank, p < 0.05).

**Fruit-fly connectome reservoirs**
- **Morra & Daley 2022, IJCNN 2022 [CONF]** (arXiv:2201.09359) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2201.09359):
  - Hemibrain (female), right lateral horn only; synapse counts normalised, diagonal set to 1.
  - Control: one conventional ESN, uniform [0,1) weights, 20% density, fixed seed. No rewired or shuffled controls.
  - Task: Mackey-Glass, with 50 runs per configuration. Spectral radius was not tuned.
  - Validation MSE 95% CIs, fly vs ESN:
    - A=900 variant A: [0.042, 0.043] vs [0.046, 0.051]
    - A=300 variant B: [0.0041, 0.0042] vs [0.0049, 0.0062]
    - some settings overlap (e.g. A=300 variant A: [0.0048, 0.0057] vs [0.0022, 0.0067])
  - "Significance" means outside two standard deviations; no formal test. The "100-fold" variance reduction claim ranges from about 24× to 185× across settings.
- **Morra & Daley 2023, IJCNN 2023 [CONF]** ("Using connectome features to constrain echo state networks", arXiv:2206.02094) — [arXiv PDF](https://arxiv.org/pdf/2206.02094); venue from the [CCN 2024 reference list](https://2024.ccneuro.org/pdf/288_Paper_authored_CCN.pdf):
  - Network: hemibrain right lateral horn, **4,286 neurons**, density 1.3%.
  - Naming is inverted in this paper: the "null model N" is the connectome ESN. "E" is a conventional ESN with uniform [−1,1] weights and ρ = 1.
  - Variants of the connectome:
    - S: densified to 20%
    - D: nonzero weights redrawn from U[−1,1]
    - P: nonzero weights permuted, zero positions kept
    - C: clustering raised from 0.27 to 0.5
  - Extra random baselines: ED (distribution-matched), ES (new seed), E3 and E7 (ρ = 0.3 and 0.7).
  - 30 trials per model; Wilcoxon signed-rank (p < 0.05).
  - Tasks: Mackey-Glass-17, Lorenz, Rössler; multi-step prediction of the second half of a segment.
  - MSE, connectome N vs ESN E:
    - MG-17, TR_IN = 750: 3.3e-2 vs 4.7e-2 (~30% lower)
    - MG-17, TR_IN = 1000: 2.91e-2 vs 2.95e-2
    - Lorenz, TR_IN = 250: **E wins**, 2.1e1 vs 8.5e1
    - Lorenz, TR_IN = 750: 3.4e1 vs 4.6e1
    - Rössler, TR_IN = 250: 7.3e1 vs 9.6e1
    - Rössler, TR_IN = 750: 2.1e1 vs 2.3e1
    - The distribution-matched random ESN (ED) beats N on MG-17 at TR_IN = 250 (1.1e-2 vs 1.6e-2).
  - No degree-preserving null. The authors do not model E/I signs ("we, for example, did not consider excitatory or inhibitory synapses").
- **Morra, Flynn, Amann & Daley 2023 [CONF/PRE]** ("Multifunctionality in a connectome-based reservoir computer", arXiv:2306.01885; IEEE copyright per the [UCC record](https://research.ucc.ie/en/publications/multifunctionality-in-a-connectome-based-reservoir-computer/)) — [arXiv](https://arxiv.org/abs/2306.01885):
  - Fly lateral horn reservoir vs an Erdős–Rényi reservoir on "seeing double".
  - The fly reservoir shows "greater capacity for multifunctionality", works across a broader hyperparameter range, and solves the task at spectral radii beyond the point where the ER reservoir becomes chaotic.
  - No numbers in the abstract.
- **Morra, Fouke, Naumann & Daley 2025, *Natural Computing* 24(3):511–528 [PR]** ("Connectomes inform function…") — [listing](https://scholars.duke.edu/individual/pub1678465):
  - A review of five experiments with fly, larval zebrafish and MaMI connectomes in reservoirs.
  - According to the search summary, it reports better chaotic-input prediction and multifunctionality across hyperparameters, some of them chaotic. Full text not verified (see Gaps).
- **Costi, Hadjiivanov, Dold, Hale & Izzo 2025, *Biomimetics* 10(5):341 [PR]** (ESA ACT) — [PMC12109256](https://pmc.ncbi.nlm.nih.gov/articles/PMC12109256/):
  - FlyWire connectome filtered to 104,909 neurons; subgraphs of N = 50–1,500 ("most connected" or "proportional" selection), plus the full connectome.
  - Weights = FlyWire "connection score". Signs: GABA negative; ACh and glutamate positive; modulatory transmitters excluded. ρ = 0.99.
  - Controls:
    - (1) random topology + random weights at matched sparsity
    - (2) connectome topology + random weights
    - (3) random topology + connectome weights
    - (4) connectome with shuffled weights
  - Task: circular restricted three-body problem (Earth–Moon), horizons of 1/3/7/10 steps.
  - 5 trajectories × 10 reinitialisations = 50 cases; Mann–Whitney–Wilcoxon and Bhattacharyya distance.
  - At ρ = 0.25 "the classical ESN consistently performs better". At ρ = 0.99 the biomimetic reservoirs win.
  - Full connectome: nRMSE "under 2%" at β = 1e-6, "significantly more resilient to overfitting". At β = 1e-3 it shows the highest error.
  - Shuffled weights ≈ random-topology hybrid (p > 0.05).
  - No degree-preserving null. A [review report](https://www.mdpi.com/2313-7673/10/5/341/review_report) flagged that spectral-radius handling was insufficiently addressed (search snippet).
- **McAllister, Wade, Houghton & O'Donnell** (fly larva mushroom body, *C. elegans*, fly subgraphs):
  - 2024 poster "Random Reservoirs Rule! (at Remembering)" [CONF]: "Simpler topologies often perform equally well as more complex networks on prediction tasks" — [Ulster](https://pure.ulster.ac.uk/en/publications/random-and-biological-network-connectivity-for-reservoir-computin/).
  - COSYNE 2025 poster [CONF]: connectome reservoirs have "smaller subsets of highly task-selective neurons" than random ones — [COSYNE 2025](https://world-wide.org/eposter/cosyne-25/sparse-neural-engagement-connectome-ybm245ip).
  - 2026 bioRxiv (10.1101/2026.03.30.715411) [PRE], as summarised by Cho et al.: in small fly-connectome subgraphs raw performance was "generally comparable to random reservoirs", while weight-cost-normalised performance and robustness to neuron loss or parameter variation were higher — [Cho et al. related work](https://arxiv.org/html/2610.10014v1). Primary text not accessed.

**C. elegans connectome reservoirs**
- **Reimers, Ramstad, Hubin & Nichele 2026, arXiv:2609.30508 [PRE]** — [arXiv HTML](https://arxiv.org/html/2609.30508):
  - 22 *C. elegans* connectomes (Witvliet 2021; 8 individuals; synapse count, synapse size and physical contact). Fixed ESN, ρ normalised to 1 and scaled by α. Inputs on sensory neurons, outputs on body-wall muscles.
  - Nulls, 15 samples per original run:
    - (1) rewired preserving the out-degree distribution, I/O drawn from the original groups
    - (2) original wiring with random I/O nodes
    - (3) both randomised
  - Beta-regression coefficients (positive = connectome better), for nulls 1 / 2 / 3:
    - **MC:** +0.305 / +0.657 / +0.951 (all p < .001)
    - **Hénon:** −1.342 / −1.086 / −1.426 (all p < .001)
    - **Mackey-Glass:** −0.347 (p = .001) / +0.009 (p = .939) / −0.373 (p = .003)
    - **Perceptual decision-making:** n.s.
    - **Go/No-go:** +0.282 (p = .065) / −0.386 (p = .010) / −0.547 (p < .001)
  - Conclusion: biological wiring "do[es] not consistently beat null models", and I/O placement matters more than rewiring.
- **Churchland, de Palma Aristides, Garcia-Ojalvo, Ritz, Anderson & Soriano 2026, arXiv:2609.07355 [PRE]** — [arXiv](https://arxiv.org/abs/2609.07355):
  - Perturbs the *C. elegans* connectome's topology, E/I signs, weight magnitudes and placement. Tasks are task-agnostic: MC, IPC, kernel rank, generalisation rank.
  - Finds a performance–robustness trade-off tied to the **raw spectral radius before normalisation**: rescaling every matrix to the same target ρ amplifies low-ρ matrices more, so "architectural differences partly reflect this rescaling effect".
  - No numbers on the abstract page.
- **Pircher et al. 2021, *Sci Rep* 11 [PR]** — [PMC7970964](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7970964/):
  - Compares ANNs with networks built on the *C. elegans* micro-connectome and the human macro-connectome.
  - Cited by Cho et al. 2026 as finding that "random structures performed similarly to biological ones" — [Cho et al.](https://arxiv.org/html/2610.10014v1).
  - Details not verified.

**Generative / meta-analyses**
- **Yu, Liu, Jia, Xiao, Xue, Sun & Chen 2026, arXiv:2609.06093 [PRE]** — [arXiv](https://arxiv.org/abs/2609.06093):
  - Generative latent model of connectome graphs used as RC substrates. Edge-reconstruction AUC up to 0.910; cross-validated R² up to ~0.87 for functional variation across tasks.
  - Memory relates to reciprocal connectivity; prediction and classification relate to spectral properties.
  - Datasets and nulls not stated on the abstract page.

### Inferences
- Most "connectome wins" claims compare against a control that differs in more than wiring:
  - a different density or weight distribution: Morra's conventional ESN is 20% dense vs 1.3% for the connectome;
  - a full shuffle that destroys degree structure (Qian 2025);
  - an ER graph (Morra 2023).
- The clearest positive results against degree-preserving rewiring are from mammalian mesoscale MRI or tract-tracing networks with dozens to ~1,000 nodes, measured on MC at α ≈ 1 (Suárez 2021; conn2res 2024). The one fly network in conn2res is the exception (p = 0.11).
- Statistical practice is uneven. Several papers give no formal test (Damicelli 2022; Morra 2022; Jin 2026, below). Many use one fixed connectome with randomness only in W_in or the readout.
- Mosca's design (seeds, disjoint splits, permutation tests, pre-registered prediction) is stricter than most of this literature.

### Gaps
- Morra et al. 2025 (*Natural Computing*): full text not retrieved, so nulls, numbers and whether a degree-preserving null was used are unverified.
- McAllister et al. 2026: bioRxiv primary not accessible (the API returned "no posts found" for the DOI cited by Cho et al.). Claims come only from Cho et al.'s summary.
- Morra & Daley 2023: the exact density of the conventional ESN "E" is not stated in the text read. 20% is implied only by their convention citations.
- Costi 2025: control-model nRMSE values are in figures only and were not extracted.
- Suárez 2021: numerical medians for the stable and chaotic regimes (where rewired wins) were not extracted.
- Pircher 2021: details not verified.

## Q2. Is the reported advantage robust to degree-preserving nulls, and does it depend on the dynamical regime (spectral radius, edge of chaos)?

### Takeaway
It is not robust in general. When degree-preserving nulls, shared initialisation and controlled I/O placement are used, the advantage usually shrinks to zero or reverses, especially for invertebrate connectomes on tasks alien to the animal. Where it survives, it depends on the regime: Suárez 2021 finds it only at criticality, with rewired nulls winning in the stable and chaotic regimes. Spectral-radius normalisation itself is a confound (Churchland 2026).

### Cited Findings
- **Vanishes or reverses under degree-preserving or I/O-controlled nulls:**
  - **conn2res fly connectome:** p = 0.11 vs 500 degree-preserving nulls on MC — [conn2res](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10803782/).
  - **Reimers 2026, *C. elegans*:** the out-degree-preserving rewired null beats the original on Hénon (β = −1.342) and Mackey-Glass (β = −0.347), while the original wins MC — [arXiv:2609.30508](https://arxiv.org/html/2609.30508).
  - **Dhiman 2026 [PRE] ("Topological sensitivity in connectome-constrained neural networks")** — [arXiv HTML](https://arxiv.org/html/2604.04033):
    - Setup: flyvis scaffold, 45,669 nodes, 1,513,231 edges. 734 trainable and 2,959 fixed recurrent parameters, identical across graphs. MovingEdge direction decoding.
    - Weak controls (checkpoint init vs a naive self-loop-matched random graph): loss 0.514 vs 0.698 after 5 steps.
    - Shared random init, degree-preserving null (directed double-edge swaps, 250,000 accepted swaps): Δloss = +0.0003 at 5 steps.
    - Ensemble (5 graphs × 3 seeds): loss 0.5155 ± 0.0067 vs 0.5172 ± 0.0061.
    - Conclusion: advantages "do not robustly persist under degree-preserving controls".
    - Only 5- and 10-step horizons; no final loss.
  - **Senge 2026 [PRE] ("A connectome test of the fly hashing algorithm")**, using MaleCNS, hemibrain, FlyWire and BANC: measured wiring vs degree-preserving rewiring gives a median retrieval change of −1.6% — "no consistent retrieval advantage" — [arXiv:2610.09114](https://arxiv.org/abs/2610.09114).
  - **McAllister et al. 2026 [PRE]:** fly subgraphs are comparable to random reservoirs in raw performance (via [Cho et al.](https://arxiv.org/html/2610.10014v1)).
  - **Goulas 2021 and Damicelli 2022:** parity with random wiring — [bio2art/Goulas](https://www.biorxiv.org/content/10.1101/2021.01.22.427744), [Damicelli](https://pmc.ncbi.nlm.nih.gov/articles/PMC9710781/).
- **Survives degree-preserving nulls:**
  - **Suárez 2021:** human connectome at α=1, MC 0.92 vs 0.80, effect size 99% — [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.11.10.350876v1.full).
  - **conn2res:** mouse, rat and macaque, p < 0.002 — [conn2res](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10803782/).
  - **Morra CCN 2024:** 17 of 18 mesoscale connectomes beat weight-preserving degree-preserving rewires on multifunctionality — [CCN 2024](https://2024.ccneuro.org/pdf/288_Paper_authored_CCN.pdf).
  - **Cho, Kang & Kim 2026 [PRE] ("A Drosophila whole-connectome network can learn human-designed cognitive tasks")** — [arXiv HTML](https://arxiv.org/html/2610.10014v1):
    - Setup: MaleCNS v1.0, 166,700 nodes, 25.6 M edges. **One trainable scalar per anatomical edge**; I/O maps fixed; no synaptic signs.
    - Null: directed degree-preserving double-edge swaps, 5 accepted swaps per edge.
    - Bounded addition: 92.77% vs 67.93% (+24.84 pp); 4 anatomical runs vs 8 rewired fits; descriptive stats only.
    - Language factorial: 61.59% vs 44.17% (+17.42 pp, 95% crossed-bootstrap CI [11.39, 24.90]).
    - 20-rewire ensemble: 71.35% vs 46.56%; rank fraction 1/21 ≈ 0.048.
    - Dense parameter-matched RNN (5,058 units): 8.72%.
    - No ER control.
    - Sensory shuffle: anatomical 60.94% → 19.27%; rewired 41.80% → 13.02%.
  - **Jin, Zhu, Zhang & Sui 2026 [PRE] (FlyGM, arXiv:2602.17997 v3)** — [arXiv HTML](https://arxiv.org/html/2602.17997v3):
    - Setup: whole-brain FlyWire graph as a **fixed** signed operator W = N_exc − N_inh. Trained parts: per-neuron descriptors, encoder, update MLP and decoder. Deep-RL locomotion of a biomechanical fly.
    - Walking-direction angle error, FlyGM vs degree-preserving rewired vs ER vs MLP: 4.96 vs 7.84 vs 12.11 vs 6.76° at (v=2, ψ=0), and 8.29 vs 13.55 vs 125.36 vs 13.90° at (v=3, ψ=7).
    - Seeds unstated; no statistical tests.
    - Claim: "The structural advantage of the connectome grows with task complexity". The gap to the rewired graph (2.88, 2.20, 3.36, 5.26°) is not strictly monotonic.
- **Regime and spectral-radius dependence:**
  - **Suárez 2021:** "In the stable and chaotic regimes, randomly rewired networks outperform the empiricallly-derived architecture" [sic]. The empirical advantage is specific to α ≈ 1, but cost-normalised performance favours the empirical network in all regimes — [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.11.10.350876v1.full).
  - **Costi 2025:** at ρ = 0.25 the classical ESN "consistently performs better"; at ρ = 0.99 the connectome reservoirs win — [PMC12109256](https://pmc.ncbi.nlm.nih.gov/articles/PMC12109256/).
  - **Morra 2023:** the fly reservoir stays multifunctional past the ρ where the ER reservoir turns chaotic — [arXiv:2306.01885](https://arxiv.org/abs/2306.01885).
  - **Churchland 2026:** differences across *C. elegans* variants partly reflect raw-ρ rescaling — [arXiv:2609.07355](https://arxiv.org/abs/2609.07355).
  - **Qian 2025:** reports its comparison only at α = 1 — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.10.29.685101.full.pdf).
- **I/O placement as the hidden variable:**
  - Reimers 2026 finds random I/O placement changes scores more than rewiring — [arXiv:2609.30508](https://arxiv.org/html/2609.30508).
  - A community project on FlyWire v783 (flyconnectome-nulls, [COMM]) found that column shuffles and degree-preserving swaps overtake the connectome by generation 600 in 10/10 seeds. These nulls "create direct olfactory-to-motor links the real graph almost lacks (0.012% against 10.6–10.7%)". Interior-only randomisation erases the gap — [Fly Brain Hub](https://fly-brain-hub.vercel.app/projects).

### Inferences
- The surviving positive cases share at least one of three features:
  - (a) mammalian mesoscale graphs at criticality;
  - (b) tasks with a natural mapping onto the animal's own sensors and effectors (Cho: language grounded in leg afferents; Jin: fly locomotion);
  - (c) anatomical I/O choices that a rewired graph cannot exploit in the same way.
  Mosca uses alien tasks (2D Navier–Stokes, platformer), an EM fly subgraph, and the same degree-ranked port rule for all models. This combination is where the literature most consistently reports ties, so Mosca's H0/H1 pre-registration matches the prior evidence.
- Two literature findings independently match Mosca's own: degree-preserving rewiring can open input→output shortcuts (community flyconnectome-nulls), and I/O placement dominates (Reimers 2026). This matches Mosca's fixed 92-neuron overlap rule; few academic papers control I/O overlap explicitly.
- Regime dependence (Suárez, Costi, Churchland) implies that a connectome-vs-null comparison at a single gain or spectral-radius setting can produce or erase an effect. For Mosca's trainable masked RNN, the analogue is initial weight scale and how the mask's raw spectral radius differs across fly, ER and rewired graphs at initialisation. This is an inference; no paper tests it for trainable masks.

### Gaps
- No paper found sweeps spectral radius or gain for a **trainable** connectome-masked RNN against degree-preserving nulls. Dhiman 2026 and Cho 2026 do not report such a sweep.
- Jin 2026 gives no seed count or statistical test. Cho 2026's addition comparison uses descriptive means over 4 vs 8 runs.
- I found no study that compares a "no recurrent core" control (zero recurrent edges) against connectome reservoirs in the academic literature. It appears only in community projects (e.g. a "no-recurrence model" in the LegalFly GitHub project, per search summary; the repo URL returned 404 when fetched).

## Q3. Has anyone forecast chaotic time series, PDEs or fluid dynamics (Navier–Stokes, Kuramoto–Sivashinsky, turbulence) with a connectome-derived network?

### Takeaway
Low-dimensional chaotic series: **yes**, about 8 studies. They cover Mackey-Glass, Lorenz, Rössler, Hénon and the restricted three-body problem, all with fixed-weight ESN reservoirs and mostly weak nulls. **PDEs, spatiotemporal chaos and fluid flow: no.** Repeated searches found no paper or preprint that uses a connectome-derived network, fixed or trainable, to forecast Navier–Stokes, Kuramoto–Sivashinsky, turbulence or any PDE. Reservoirs and RNNs on KS and turbulent convection exist, but only with random or engineered graphs.

### Cited Findings
- **Connectome-based forecasting of low-dimensional chaotic systems:**
  - **Morra & Daley 2022 [CONF]:** Mackey-Glass, fly lateral-horn ESN vs conventional ESN — [ar5iv](https://ar5iv.labs.arxiv.org/html/2201.09359).
  - **Morra & Daley 2023 [CONF]:** Mackey-Glass-17, Lorenz (σ=10, r=28, b=8/3), Rössler (a=0.15, b=0.2, c=10); multi-step prediction. Mixed: the conventional ESN wins Lorenz at TR_IN=250 (MSE 21 vs 85) — [arXiv:2206.02094](https://arxiv.org/pdf/2206.02094).
  - **Morra et al. 2025 [PR]:** the summary says it reports better chaotic-input prediction — [listing](https://scholars.duke.edu/individual/pub1678465). Unverified.
  - **Costi et al. 2025 [PR]:** FlyWire reservoir, circular restricted three-body problem, horizons 1–10 steps — [PMC12109256](https://pmc.ncbi.nlm.nih.gov/articles/PMC12109256/).
  - **Qian et al. 2025 [PRE]:** Lorenz with fly, mouse, rat and macaque mesoscale connectomes vs full shuffles — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.10.29.685101.full.pdf).
  - **Guragain et al. 2026 [PRE]:** Lorenz and Mackey-Glass with 6 species' connectomes and optimised weights — [arXiv:2606.09902](https://arxiv.org/abs/2606.09902).
  - **Reimers et al. 2026 [PRE]:** one-step Hénon and Mackey-Glass with 22 *C. elegans* connectomes. The nulls beat the connectome on both — [arXiv:2609.30508](https://arxiv.org/html/2609.30508).
  - **Park, Watabe, Sunada, Yamagami & Uchida 2026 [PRE]:** photonic RC (spatial light modulator) with small-world and scale-free networks and a network built from human brain "connectomes"; one-step prediction of a chaotic series. Small-world performs best — [arXiv:2607.23285](https://arxiv.org/abs/2607.23285). Node count and whether the network is functional or structural are unclear from the abstract.
- **No connectome + PDE or fluid result found.** Search strings, all returning no connectome+PDE work:
  - "connectome reservoir computing Kuramoto-Sivashinsky OR Navier-Stokes OR turbulence"
  - "connectome neural network forecasting Kuramoto-Sivashinsky OR fluid flow OR Navier-Stokes brain-wired recurrent"
  - "C. elegans connectome echo state network Lorenz"
  - "FlyWire echo state network OR reservoir … 2026"
  - Cho et al. 2026's related-work section, a broad 2026 survey of fly-connectome ML, mentions no chaotic, PDE or fluid tasks beyond "Morra et al. 2025 on time-series prediction" — [arXiv:2610.10014](https://arxiv.org/html/2610.10014v1).
- **Background [BG]: PDE and fluid forecasting with random-graph reservoirs and RNNs exists.**
  - Vlachas et al. 2020: RC vs BPTT RNNs on Lorenz-96 and Kuramoto–Sivashinsky — [arXiv:1910.05266](https://arxiv.org/pdf/1910.05266).
  - Pandey & Schumacher 2020: reservoir model of 2D turbulent convection — [arXiv:2001.10280](https://arxiv.org/pdf/2001.10280).
  - Lu, Hunt & Ott 2018 (attractor reconstruction) and Flynn et al. 2023 (Chaos, "seeing double") are the RC lineage that Morra's connectome multifunctionality work builds on — [CCN 2024 refs](https://2024.ccneuro.org/pdf/288_Paper_authored_CCN.pdf), [Flynn 2023](https://ar5iv.labs.arxiv.org/html/2305.05799).
- **Community projects [COMM]:** the Fly Brain Hub catalogue of fly-connectome projects (~290k characters, keyword-searched) has no Navier–Stokes, KS, PDE, Lorenz or Mackey-Glass project — [Fly Brain Hub](https://fly-brain-hub.vercel.app/projects).
  - "flybrain-reservoir" (MaleCNS, 166,700 neurons, ESN on daily market data) vs rewirings preserving each neuron's partner count: "On forecasting there is no edge either way". On MC "the fly comes last, 8–11x behind random wiring". A NARMA-10 benchmark and a FlyWire replication reportedly agree.
  - "Flight-test the fly" uses MIL-spec Dryden turbulence only as a disturbance input to a yaw-damper circuit. It does not forecast flow.

### Inferences
- Mosca's task (a) — 64×64 forced 2D Navier–Stokes vorticity forecasting through a Fourier encoder, with a connectome-masked trainable RNN core — has no precedent in the literature found. This is the clearest genuinely new element. A related "first" claim — that this is the first test of the connectome on spatiotemporal chaos or PDEs — also holds as far as these searches can tell.
- Every chaotic-series connectome study found uses fixed reservoir weights with a trained linear readout. None trains weights inside a connectome mask with Dale's-law signs on a chaotic or PDE target. That is a second novel axis for Mosca.
- The low-dimensional chaotic results that favour connectomes all compare against a conventional ESN, ER or full shuffle, never a degree-preserving null. The only study with degree-preserving-type nulls on a chaotic series (Reimers 2026: Hénon, Mackey-Glass) finds the **nulls win**. A Mosca tie with the rewired control would therefore be consistent with the strongest available evidence, not an anomaly.

### Gaps
- The searches cover web-indexed arXiv, bioRxiv, PMC and publishers. Paywalled conference proceedings (IEEE/ACM), non-English venues and theses were not exhaustively checked, so "no connectome + PDE study exists" is a strong absence-of-evidence statement, not proof.
- Park et al. 2026 (photonic): the abstract does not say whether the "human brain connectome" network is structural or functional, or how large it is. Prediction numbers not extracted.
- Morra et al. 2025 (*Natural Computing*) may contain additional chaotic-series results with other nulls; not verified.

## Q4. Are there 2025–2026 preprints using the full FlyWire (or whole-brain fly) connectome as a reservoir or dynamical substrate for ML benchmarks?

### Takeaway
Yes, several. The truly full-FlyWire ones are Costi 2025 (full 104,909-neuron reservoir; peer-reviewed) and Jin 2026 (FlyGM; whole-brain FlyWire as a fixed graph operator for RL locomotion). Cho 2026 uses the full male CNS (MaleCNS) with trainable per-edge weights. The rest use subgraphs, other fly datasets (flyvis, larva, hemibrain) or hashing. Results against degree-preserving nulls split by design: the two trained or embodied designs (Cho, Jin) report large wins; the fixed-reservoir or alien-task tests (Senge, Dhiman, McAllister, community flybrain-reservoir) report ties.

### Cited Findings
- **Costi et al. 2025 [PR]:** full FlyWire connectome (104,909 neurons after filtering) used as an ESN reservoir; readout of "just over 22×10³" trainable parameters; three-body-problem forecasting; minimal overfitting at β = 1e-6 — [PMC12109256](https://pmc.ncbi.nlm.nih.gov/articles/PMC12109256/).
- **Jin et al. 2026 [PRE] (FlyGM):** whole-brain FlyWire (v783 named in an appendix) as a fixed signed graph; RL locomotion; beats degree-preserving rewired, ER and MLP controls on angle error (table in Q2) — [arXiv:2602.17997](https://arxiv.org/html/2602.17997v3).
- **Cho, Kang & Kim 2026 [PRE]:** MaleCNS v1.0 whole CNS (166,700 nodes); per-edge trainable scalars; addition and grounded-language tasks; beats degree-preserving rewires by 17–25 pp; a parameter-matched dense RNN scores 8.72% — [arXiv:2610.10014](https://arxiv.org/html/2610.10014v1). Closest academic design to Mosca (trainable weights inside the anatomical mask, dense parameter-matched control, degree-preserving null). Differences from Mosca: no Dale signs, no ER, no "no-core" control, tasks mapped onto fly sensory neurons.
- **Dhiman 2026 [PRE]:** flyvis optic-lobe model (45,669 nodes); advantage disappears under shared init and degree-preserving nulls — [arXiv:2604.04033](https://arxiv.org/html/2604.04033).
- **Senge 2026 [PRE]:** FlyWire, hemibrain, MaleCNS and BANC olfactory wiring in a fly-hash LSH; −1.6% median vs degree-preserving rewiring — [arXiv:2610.09114](https://arxiv.org/abs/2610.09114).
- **Yu, Qin, Liu, Xu, Vogelstein, Brown & Vogelstein 2025 [PRE / AGI-2025]** ("Biological Processing Units"):
  - Larval *Drosophila* connectome, ~3,000 neurons, ~65,000 weights, fixed.
  - Results: MNIST 98%, CIFAR-10 58%; GNN-BPU 60% ChessBench move accuracy; CNN-BPU with depth-6 search 91.7%.
  - Controls are size-matched MLPs and Transformers. **No random or rewired graph null** is mentioned.
  - [arXiv:2507.10951](https://arxiv.org/abs/2507.10951)
- **Wang & Chen 2026 [PRE] (FLYNN, arXiv:2607.00025):** RNN wired from the fly connectome for vision-based MuJoCo navigation. Comparable to hand-crafted networks and robust to total vision loss. The summary mentions no random-graph null — [deeplearn.org summary](https://deeplearn.org/arxiv/790121/flynn:-robust-neural-network-for-robot-navigation-using-fly-brain-topology).
- **McAllister et al. 2026 [PRE]:** fly-connectome subgraphs as reservoirs; comparable raw performance to random — [via Cho et al.](https://arxiv.org/html/2610.10014v1).
- **Neuromorphic simulation of FlyWire on Loihi 2 (arXiv:2508.16792)** is a simulation and hardware paper, not an ML benchmark with nulls — [arXiv](https://arxiv.org/pdf/2508.16792).
- **Community [COMM], all on [Fly Brain Hub](https://fly-brain-hub.vercel.app/projects):**
  - **"fly-brain-calculus"** (MaleCNS): fly 0.795 held-out accuracy vs 0.311 control. Permuting E/I labels beats the real graph, degree-preserving rewiring beats that, and "a plain random graph with the right edge count scores 0.980". Only shuffling synapse counts hurts, by moving the network out of its operating regime.
  - **"SmolFly":** degree-preserving null costs 0.0781 nats (~1.58%) inside an LLM.
  - **"fly-cartpole":** shuffled 390.3 vs real 392.5 (p = 0.48).
  - **"flybrain-reservoir":** ties on forecasting; fly worst on MC.

### Inferences
- A practical rule emerges from the 2025–2026 fly work. Where the connectome is embodied or task-matched (locomotion, leg-afferent-grounded language), it beats degree-preserving rewiring. Where it is a generic reservoir for alien tasks (hashing, market data, calculus, MNIST-style), it ties random wiring or loses to it. Mosca sits in the second category by construction, and its ties match this pattern.
- Mosca's control set is broader than any single academic paper found: ER at matched density and inhibitory fraction, degree-preserving rewired, dense parameter-matched, no-core, free MLP, and an FNO upper reference. Its pre-registered null prediction is also unusual. Among academic papers, Cho 2026 comes closest (rewired + dense parameter-matched + sensory-shuffle). Explicit pre-registration appears only in community projects ("five pre-registration protocols" in flyconnectome-nulls; BioReservoir).
- Mosca also enforces Dale's law with signs from the connectome. Among the fly papers, signs are used by Costi 2025 (neurotransmitter-based; glutamate treated as positive) and Jin 2026 (net polarised counts), but in fixed-weight settings. Cho 2026 omits signs. No paper found trains Dale-constrained weights inside a FlyWire mask against ER, rewired and no-core controls on a physics task.

### Gaps
- Jin 2026: neuron and edge counts and seed counts not stated, so "whole-brain" scale is from the abstract wording only.
- FLYNN: no full text read; controls unknown.
- Community projects are self-reported and unreviewed. They are included only as signal, not evidence.
- The "Zhang et al. 2026" whole-brain communication model and the "Wang et al. 2026 (BrainTrace)" study cited by Cho et al. were not checked.
