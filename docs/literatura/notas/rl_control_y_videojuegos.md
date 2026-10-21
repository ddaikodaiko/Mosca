# Connectomes as controllers: RL, embodied control and video games (2019-2026)

Scope note. Research done 20-10-2026. Status labels used below: **[peer-reviewed]**, **[preprint]**, **[hobby repo]** (GitHub README, self-reported, unreviewed), **[press]**, **[company post]**. Three things must be kept apart:
(i) **connectome-wired networks**: an artificial network whose graph comes from an EM connectome (FlyWire v783, MaleCNS v1.0, C. elegans). This is Mosca's category;
(ii) **living neurons in a dish**: DishBrain, Cortical Labs CL1. No connectome involved;
(iii) **brain-inspired architectures**: NCP/LTC, NCAP, NEAT/MarI/O. These use a circuit motif or none, not a measured wiring diagram.

Context that shapes everything below: the MaleCNS v1.0 connectome (adult male, ~166,700 neurons) was released as data on 8 June 2026 and announced with the *Cell* paper on 3 September 2026 ([nfly README](https://github.com/nzxhuong/nfly); [Gizmodo, 8 Sep 2026](https://gizmodo.com/google-mapped-a-fruit-flys-brain-now-its-playing-doom-and-super-mario-64-2000808616); [awesome-fly](https://github.com/rustyorb/awesome-fly)). Within days it set off a viral wave of hobby projects in which a "fly plays game X". Almost every game instance found dates from September-October 2026 and is a hobby repo.

---

## Q1. Connectome-based controllers in RL / embodied control (flybody, NeuroMechFly v2, Eon, C. elegans robots, NCPs, Atari/Pong/Doom agents): what controls did they use?

### Takeaway
The flagship embodied-fly platforms do not use the connectome as the controller with random-wiring controls. flybody's controllers are MLPs. NeuroMechFly v2 plugs in a connectome-constrained *visual* model with no wiring nulls reported. Eon's "uploaded fly" (March 2026) is a fixed LIF connectome with a hand-picked readout and no controls. Among academic RL works, only FlyGM (and a few 2026 preprints) ran degree-preserving nulls. Most games-with-connectome agents are hobby repos from September 2026 and are mostly frozen-reservoir-plus-trained-readout designs. A minority include shuffled or rewired controls, and those often come out null.

### Cited Findings

**Fly: academic / company platforms**
- **flybody** (Vaxenburg et al., *Nature* 2025; bioRxiv March 2024) **[peer-reviewed]**: a whole-body MuJoCo fly whose controllers are trained by "data-driven end-to-end reinforcement learning" and include a vision-guided flight controller. The abstract mentions no connectome-constrained component ([bioRxiv 2024.03.11.584515](https://www.biorxiv.org/content/10.1101/2024.03.11.584515v1); [Nature](https://www.nature.com/articles/s41586-025-09029-4)). A curated list states plainly: "Controllers are MLPs, not connectome graphs" ([awesome-fruit-fly](https://github.com/townie/awesome-fruit-fly)). No wiring controls apply.
- **NeuroMechFly v2** (Wang-Chen et al., *Nature Methods* 2024) **[peer-reviewed]**: adds vision, olfaction, ascending feedback and complex terrain. In a closed-loop fly-following demo the input passes through the connectome-constrained **flyvis** visual network, and the detected object position modulates a turning signal for a separately engineered walking controller ([bioRxiv 2023.09.18.556649](https://www.biorxiv.org/content/10.1101/2023.09.18.556649.full.pdf); [EPFL postprint](https://epfl.ch/labs/ramdya-lab/wp-content/uploads/2024/08/NMF2_postprint.pdf)). None of the sources reached report random or rewired visual-network controls.
- **flyvis / Lappalainen et al.**, *Nature* 2024 **[peer-reviewed]**: connectome-constrained models of the fly visual system ([Nature](https://www.nature.com/articles/s41586-024-07939-3)). It is not an RL controller. Dhiman 2026 later re-examined flyvis topology claims with stricter nulls (see Q3).
- **FlyGM** (Jin et al., arXiv 2602.17997) **[preprint]**, in one line: the whole FlyWire graph is used as a message-passing policy trained by imitation plus PPO on flybody locomotion. It beats a degree-preserving rewired graph, and the gap widens with task difficulty ([arXiv](https://arxiv.org/pdf/2602.17997)). (Already read by the project. Its rewired/ER baselines were unweighted.)
- **Eon Systems "We've Uploaded a Fruit Fly"** (7, 8 and 10 March 2026) **[company post]**: the FlyWire connectome with "a simple neuron model" (the Shiu et al. 2024 LIF model) controls a MuJoCo NeuroMechFly body. Eon claims it is "the world's first embodiment of a whole-brain emulation that produces multiple behaviors" ([Eon updates](https://eon.systems/updates)). Press reports grooming and sugar feeding, and $3.5M seed funding led by Protocol Labs. The funding is unconfirmed by a primary source ([startbase](https://www.startbase.com/news/startup-eon-systems-demonstriert-erste-multi-verhaltensfaehige-gehirn-emulation/); [cybernews](https://cybernews.com/ai-news/digital-fruit-fly-brain-simulation/)). **No random/shuffled controls are reported** and nothing is trained ([Eon updates](https://eon.systems/updates)).
  - Critique (Columbia Science Review, 9 Sep 2026) **[student press]**: only a small hand-picked set of neurons is read out and mapped to high-level commands (walk, turn, groom, feed). The NeuroMechFly body executes pre-programmed movements, there is no modelled nerve cord, and Eon's own blog calls the video's activity "somewhat decorative" ([Columbia Science Review](https://www.thecolumbiasciencereview.com/online-articles/scientists-just-downloaded-a-fly-brain-into-a-computer-or-did-they)).
- **Antennal grooming** network derived from the connectome in a MuJoCo body (NeLy-EPFL, listed as *Nature Communications* 2026) ([awesome-fruit-fly](https://github.com/townie/awesome-fruit-fly)). Controls were not checked.
- **FLYNN** (Wang & Chen, arXiv 2607.00025, June/July 2026) **[preprint]**: a fly-connectome-topology network for vision-based MuJoCo robot navigation. It performs comparably to hand-crafted networks of similar parameter count, is more robust out of distribution, and still works under total vision loss. **The abstract mentions no rewired or random-wiring control** ([arXiv](https://arxiv.org/abs/2607.00025)). A curated list says it was trained with DAgger ([awesome-fruit-fly](https://github.com/townie/awesome-fruit-fly)). Not verified in the paper.

**C. elegans**
- **"Worm brain in a Lego robot"** (Timothy Busbice, 2014; pre-window precedent) **[hobby/press]**: sensors drive simulated sensory neurons and motor-neuron outputs drive the wheels. The behaviour is hand-built spreading activation with thresholds, with no learning ([Hackaday 2014](https://hackaday.com/2014/12/15/gift-your-next-robot-with-the-brain-of-a-roundworm/); [GoPiGo tutorial](https://www.dexterindustries.com/GoPiGo/projects/python-examples-for-the-raspberry-pi/simulate-a-worm-brain/)).
- Valencia Urbina et al., "Emergent dynamics in a robotic model based on the C. elegans connectome" (arXiv 2020; PMC 2022) **[peer-reviewed]**: analyses that robot's dynamics and finds worm-like global dynamics appear spontaneously. No training and no wiring nulls ([PMC9868850](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9868850/); [arXiv 2011.09057](https://arxiv.org/pdf/2011.09057)).
- **BAAIWorm / MetaWorm** (Peking Univ., *Nature Computational Science*, Dec 2024) **[peer-reviewed]**: a biophysically detailed model of all 302 neurons coupled to a 96-muscle body in a 3D fluid environment ([PKU NBIC](https://nbic.pku.edu.cn/en/Achievements/d57de79476bf4b279d32e0c285da6405.htm); [MetaWorm bioRxiv](https://www.biorxiv.org/content/10.1101/2024.02.22.581686.full.pdf)). It is a closed-loop emulation, not an RL policy. I could not confirm its perturbation and null experiments.
- **Connectome-based digital twin worm** (*Mathematics* 2023) **[peer-reviewed]**: a connectome-based network trained **offline on data from a PID controller**, reproducing chemotaxis in a simulated body (search snippet only; the page returned 403) ([MDPI doi](https://doi.org/10.3390/math11112442)).
- **Schmidgall, Schuman & Parsa, arXiv 2209.14406 (2022)** **[preprint]**: translates the C. elegans motor circuit into ANNs. The locomotion circuit gives a strong inductive bias for locomotion but can hurt unrelated tasks (visual classification). The **exact wiring need not be preserved**, because the architectural statistics alone are a useful prior ([arXiv](https://arxiv.org/abs/2209.14406)).
- **NCAP**, Bhattasali, Zador & Engel, NeurIPS 2022 **[peer-reviewed]**: C. elegans locomotion circuits used as a policy for the Swimmer agent. Performance is comparable to MLPs with "dramatically improving data efficiency" and orders of magnitude fewer parameters, and it transfers to new bodies. Constrained E/I is crucial. **The baseline is a generic MLP, with no rewired control** ([arXiv 2201.05242](https://arxiv.org/abs/2201.05242)). This is a hand-translated motif, not a full connectome.
- **NCPs** (Lechner, Hasani et al., *Nature Machine Intelligence* 2020) **[peer-reviewed, brain-inspired, not a connectome]**: a 19-neuron, 253-synapse controller inspired by the C. elegans tap-withdrawal circuit, mapping 32 input features to steering ([ISTA record](https://research-explorer.ista.ac.at/record/8679)). The ICML 2020 follow-up ("Natural Lottery Ticket Winner: RL with Ordinary Neural Circuits") reportedly shows these circuits realise higher max-flow than "arbitrarily wired" networks (secondary summary; baseline construction not verified) ([sotaverified summary](https://sotaverified.org/papers/the-natural-lottery-ticket-winner)).
- A GA-trained full worm connectome (~3,680 synapse weights) predates the window (2015, ScienceDirect). It uses evolution, not RL ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231215008279)).

**Connectome agents in Atari / Pong / Doom and similar (all hobby unless noted)**
- **flywire-pong** **[hobby repo]**: 1,230 states mapped to selected FlyWire FAFB783 neurons, 8,515 edges, trained by "supervised initialization followed by PPO". It wins 43/60 clean-Pong games against the Atari CPU. **No random or rewired controls**, and the README says it does not establish a causal advantage of fly topology ([GitHub](https://github.com/zonzujiro/flywire-pong)).
- **nfly** **[hobby repo]**: MaleCNS v1.0 visual subnetwork (138,743 neurons, 8.4M edges) with fixed signs (Dale's law) and per-edge trainable gains, trained with PPO/A2C/RLlib. **RL from pixels on Pong fails** (−20.5 after ~916k steps; a CNN with PPO reaches +19). RL from a cloned head reaches −14 to −16. A frozen fly with an MLP head cloned from a CNN averages +8.0. CartPole: RL peaks at 228 and only ~1 in 3 seeds learns; cloning from a heuristic gives 500/500. **No rewired or random control.** The author states that at equal parameter count "the fly is not cheaper and does not score higher" ([GitHub](https://github.com/nzxhuong/nfly)).
- **Doom**: DOOMFLY (Alex Wormuth, 6 Sep 2026): MaleCNS on ViZDoom, with damage sending a signal to PPL101 dopamine cells. Its own docs say "learned survival is not demonstrated" ([awesome-fruit-fly](https://github.com/townie/awesome-fruit-fly); [Gizmodo](https://gizmodo.com/google-mapped-a-fruit-flys-brain-now-its-playing-doom-and-super-mario-64-2000808616); [HotHardware](https://hothardware.com/news/google-mapped-a-fruit-flys-brain-so-engineers-taught-it-to-play-doom)). **FlyDoom** (eganeganegan) is described as comparing a MaleCNS-constrained sparse recurrent controller against **rewired graphs and conventional nets** on ViZDoom ([awesome-fly](https://github.com/rustyorb/awesome-fly)). The repo returned 404, so its results are unverified. flydoom (mutkuoz) uses FAFB-v783 ([awesome-fruit-fly](https://github.com/townie/awesome-fruit-fly)). Fly Marksman also targets ViZDoom (same list).
- Other Atari and arcade titles (all hobby, from the curated lists): MaleCNS **Asteroids** (escape/pursuit readouts, lesion tests); **FlyPong** (negative results for dopamine plasticity); **fly-craftax** (PPO-trained descending-neuron readout with baselines); **Fly Dino** (80-neuron circuit with a 243-parameter CEM readout, held-out benchmarks and silencing controls); **FightingICE**; **osu!mania** via DAgger ([awesome-fly](https://github.com/rustyorb/awesome-fly); [awesome-fruit-fly](https://github.com/townie/awesome-fruit-fly)).
- Wiring controls inside this hobby wave (self-reported):
  - **fly-cartpole**: a frozen 5,459-neuron flight circuit with 5 tuned gains. Untuned: 275.6 steps; tuned: 499.9 on held-out seeds; a "random" condition: 22.1 ([awesome-fly](https://github.com/rustyorb/awesome-fly)).
  - **Connectocopter**: rewired graphs "fail every flight" ([awesome-fruit-fly](https://github.com/townie/awesome-fruit-fly)).
  - **fly-brain-zero-shot**: intact beats scrambled ([awesome-fruit-fly](https://github.com/townie/awesome-fruit-fly)).
  - **20fly8** (2048, mushroom-body TD): trained 6,986 vs shuffled synapses 1,287 vs random policy 1,103 ([awesome-fly](https://github.com/rustyorb/awesome-fly)).
- Non-connectome comparators, to keep separate:
  - **DishBrain** (Kagan et al., *Neuron*, Oct 2022) **[peer-reviewed, living neurons]**: ~800k mouse and human neurons on an MEA played Pong ([UCL](https://www.ucl.ac.uk/news/2022/oct/human-brain-cells-dish-learn-play-pong); [Axios](https://www.axios.com/2022/10/16/neurons-learn-play-pong)).
  - **Cortical Labs CL1 plays Doom** (~200k human neurons, ~March 2026) **[press/company]**: weak, beginner-level play ([Decrypt](https://decrypt.co/359697/human-brain-cells-play-doom-cortical-labs-experiment?amp=1); [military.com](https://secure.military.com/feature/2026/03/10/scientists-teach-human-brain-cells-play-doom-sci-fi-experiment.html)).
  - Neither involves a connectome.

### Inferences
- The *Nature*-tier embodied-fly work (flybody, NeuroMechFly v2) never put the full connectome in the policy seat with trainable weights against wiring nulls. FlyGM is the only academic RL-locomotion paper found that does both, and it is still a preprint.
- In the hobby wave, the dominant design is a **frozen connectome reservoir plus a trained readout**, which differs from Mosca's trainable masked RNN with Dale's law. Where end-to-end RL on a connectome network was tried (nfly Pong and CartPole), it **failed or was unstable without cloning**. This matches Mosca's finding that PPO from scratch stays in a degenerate policy and cloning is needed.
- Eon's demo shows closed-loop LIF emulation, not control competence. Its behaviours depend on engineered command mappings, so it does not bear on Mosca's question of whether the wiring helps a trainable policy.

### Gaps
- Could not access full texts of flybody (*Nature*), NeuroMechFly v2 (*Nature Methods*), BAAIWorm or the NCP papers to confirm the absence of shuffled-wiring ablations. Absence is inferred from abstracts and secondary summaries.
- FlyDoom (eganeganegan) returned 404. Its rewired-vs-connectome ViZDoom results are unknown.
- "Deep Reinforcement Learning for Neural Control" (arXiv 2006.07352) surfaced in search. It appears to be RL that *stimulates* a C. elegans connectome model rather than using it as a policy, but it was not read.
- Lappalainen 2024's own controls were not re-checked.

---

## Q2. Has anyone put a fly or worm connectome into Super Mario Bros or a platformer?

### Takeaway
**Yes for the fly, as hobby projects only, all from September-October 2026.** At least four public GitHub repos wire a fly connectome (MaleCNS v1.0) to **NES Super Mario Bros**, plus one to Super Mario 64 and several platformer-like games (Doodle Jump clone, Chrome Dino). **None is peer-reviewed.**
- Three of the four NES repos use a **frozen** connectome and train only a readout. The fourth is a connectome-*modelled* layered SNN with no reported results.
- Only one (fly-plays-nes) reports a shuffled-wiring control: 5 runs, with the shuffle method unspecified.
- **None uses disjoint train/test levels.** All evaluate on 1-1 or unspecified levels.
- **No C. elegans connectome Mario or platformer was found.**

### Cited Findings

**NES Super Mario Bros (connectome-wired, hobby)**
- **fly-plays-nes** (arpanguria68-ui) **[hobby repo, v0.1.0, 6 commits]**:
  - Model: MaleCNS v1.0 (166,700 neurons, 25.6M connections) simulated live at 20 ticks/s. Frames go to T4/T5 and to LC10a/LPLC2/LC4/LPLC1, and descending neurons press buttons. The connectome is **frozen** ("no training inside it"), and a small learner on top is trained from editable event rewards.
  - SMB result: real connectome **3,214 px** of new ground vs shuffled **563 px**, 5 runs each, Mann-Whitney **p = 0.03**. "Hold right, no brain" gets 0 px. The shuffle procedure is not described, and there is no train/test level split.
  - Galaga learner: ~3× score (p = 0.003) but it "plateaus" ([GitHub](https://github.com/arpanguria68-ui/fly-plays-nes)).
- **fly-mario** (ksanjeev284) **[hobby repo, 1 commit]**:
  - Model: MaleCNS v1.0 with CUDA LIF dynamics; the wiring is "strictly frozen". A 3,038-feature regularised logistic readout (1,314 descending neurons, 97 MBONs, etc. **plus 192 raw grid tiles**) is trained on **teacher-policy demonstrations**, refit online with a "MistakeBuffer" ("dopaminergic" credit on failures), with optional **DAgger**. Reflex controllers handle looms, items and obstacles.
  - Claims "Flagpole Reached" (best x 3,161 = teacher). No completion rates, seeds, controls or held-out levels ([GitHub](https://github.com/ksanjeev284/fly-mario); also listed in [awesome-fruit-fly](https://github.com/townie/awesome-fruit-fly)).
- **fly-games** (shantanugoel) **[hobby repo]**:
  - Model: frozen MaleCNS LIF. Game state is reduced to 4 channels (loom→LPLC2, threat→LC4, shot→LPLC1, chase→LC10a). Twelve descending command groups feed a **ridge-regression readout fit to a scripted policy's labels** (behaviour cloning). Effectively, "escape-dominant firing → jump, quiet → walk".
  - Held-out-*episode* decision accuracy 90.0% vs a 72.5% majority baseline. In closed loop the cloned fly **dies around decision 27**, while the hand rule survives to 120. The author attributes this to compounding BC error. **No shuffled control and no held-out levels.** The same repo also does Kung Fu and Doom ([GitHub](https://github.com/shantanugoel/fly-games)).
- **super-fly** (EthanThatOneKid) **[hobby repo]**:
  - Model: a PyTorch SNN "modeled on" the fly connectome as 4 layers (ommatidia → optic lobe → CX/MB → thoracic ganglion). The connectome data source is not stated.
  - Learning: dopamine-modulated STDP (PAM reward, PPL1 punishment). There is **supervised pretraining on a recorded 1-1 teacher trajectory**, plus a RIGHT+JUMP bootstrap curriculum for the first 20 episodes. Evaluation is frozen and seeded, judged by `completion_rate`. **No results reported**, no controls, only 1-1 ([GitHub](https://github.com/EthanThatOneKid/super-fly)).
  - Issue #10 proposes "pre-train connectome weights on offline SMB playthrough trajectories" — a proposal, not a result ([issue #10](https://github.com/EthanThatOneKid/super-fly/issues/10)).

**Other Mario and platformers**
- **Fly64** (Jessica Paquette, ornata/fly) **[hobby repo]**: MaleCNS drives a decompiled **Super Mario 64**. No training or reward in the published loop, and Mario repeatedly walks into walls. The author called it "100% vibe coded" ([Gizmodo, 8 Sep 2026](https://gizmodo.com/google-mapped-a-fruit-flys-brain-now-its-playing-doom-and-super-mario-64-2000808616); [awesome-fruit-fly](https://github.com/townie/awesome-fruit-fly); [Tom's Hardware](https://www.tomshardware.com/software/programming/google-maps-entire-brain-and-central-nervous-system-of-adult-male-fruit-fly-software-engineers-immediately-make-it-run-doom-ai-powered-3d-model-of-over-166-000-neurons-can-also-play-super-mario-64)).
- **Doodle Fly** (Doodle Jump-style platformer) **[hobby repo]**: full FlyWire v783 with hand-made mappings and nothing trained. Control: 0-1 falls with intact eyes vs 41-44 with swapped eyes. This is a sensory-lesion control, not a wiring null ([awesome-fly](https://github.com/rustyorb/awesome-fly)).
- **Fly Dino** (Chrome Dino runner) **[hobby repo]**: an 80-neuron MaleCNS circuit with a CEM-trained 243-parameter readout, held-out benchmarks and silencing controls ([awesome-fly](https://github.com/rustyorb/awesome-fly)).

**Press and social coverage of the wave**
- Gizmodo (8 Sep 2026); Tom's Hardware; HotHardware; 404 Media ("a digital fly brain has taken over the internet"); a Minecraft demo by a Georgia Tech student on 4 Sep with millions of views; X-only clips of Beat Saber, Deadlock, Smash Bros and parallel parking without repos ([Gizmodo](https://gizmodo.com/google-mapped-a-fruit-flys-brain-now-its-playing-doom-and-super-mario-64-2000808616); [404 Media](https://404media.co/a-digital-fly-brain-has-taken-over-the-internet); [AI Weekly](https://aiweekly.co/alerts/janelia-google-male-fly-connectome-goes-viral-on-minecraft-demo); [awesome-fruit-fly](https://github.com/townie/awesome-fruit-fly)).
- One analysis calls the Doom demo a simulation test case, not evidence of intelligence ([Shelly Palmer](https://shellypalmer.com/2026/09/166700-simulated-neurons-are-playing-doom/)). A critical blog post exists on "a fly brain wired to Mario" (not read in full) ([Fleagle](https://thejasonfleagle.com/fly-brain-mario-connectomics-ai/)).

**Worm**
- The search "C. elegans connectome plays video game" found only games *about* worms: nematode.farm uses worm simulations as opponent AI, and the WormWorx educational app has players tune synapses. **No worm connectome plays Mario or a platformer** ([nematode_farm](https://github.com/nategri/nematode_farm); [WormWorx](https://skich.app/games/wormworx)).

**Distinguishing categories**
- Living neurons: DishBrain played Pong (2022) and CL1 played Doom (2026). Neither plays Mario, and neither uses a connectome ([UCL](https://www.ucl.ac.uk/news/2022/oct/human-brain-cells-dish-learn-play-pong); [Decrypt](https://decrypt.co/359697/human-brain-cells-play-doom-cortical-labs-experiment?amp=1)).
- Brain-inspired: MarI/O (NEAT, 2015) is the best-known "evolved net plays Mario". It is not connectome-based and was not re-sourced here.
- A diffractive optical network has also been reported playing Mario ([arXiv 2212.11278](https://arxiv.org/pdf/2212.11278)). Not a connectome.

### Inferences
- Mosca's 6b/6c setup has no public equivalent found. It combines: the connectome as a **trainable mask** with Dale's law; real NES SMB with a symbolic RAM tile observation; PPO, then BC plus PPO; **disjoint train/val/test levels**; and **six matched controls** (ER, degree-preserving rewired, dense, no-core, free MLP, plus random/constant-action baselines).
- The hobby NES projects differ on every axis: frozen reservoir, a readout that often sees **raw tiles directly** (fly-mario adds 192 grid tiles to the readout, which is a non-connectome shortcut), training and evaluation on 1-1, and at most 5-run shuffle controls.
- fly-games reproduces, at small scale, Mosca's experience that **behaviour cloning without memory or DAgger compounds errors** in SMB.
- fly-plays-nes is the only Mario repo claiming real > shuffled (p = 0.03, n = 5, frozen wiring, unspecified shuffle). It is not comparable to Mosca's tie between *trained* wirings, and it is vulnerable to the shortcut artefact described in Q3: a shuffle can make or break direct sensory→descending paths.

### Gaps
- YouTube, Reddit and X were not searched directly; only press coverage and two curated GitHub lists. The curated lists may miss repos created after mid-October 2026.
- No Hacker News thread was returned by search ("Hacker News fruit fly connectome Doom MaleCNS").
- The super-fly PR #1 and issue #10 bodies were not read in full.
- Whether fly-plays-nes "shuffled" preserves degrees is unknown.

Searches run for "does not exist" claims:
- "connectome plays Super Mario Bros"
- "fruit fly brain connectome Super Mario Bros NES video"
- "C. elegans connectome plays video game"
- "connectome neural network Atari Pong agent"
- "C. elegans connectome controller reinforcement learning robot"
- "Hacker News fruit fly connectome Doom MaleCNS"
- full reads of awesome-fly and awesome-fruit-fly

---

## Q3. 2024-2026 results on whether connectome wiring improves sample efficiency or generalization vs degree-preserving controls (positive and null)

### Takeaway
The evidence is split and very recent. All of it is preprints or hobby repos except NCAP (2022, MLP baseline only).
- **Positive vs degree-preserving nulls:**
  - FlyGM (RL locomotion)
  - KAIST arXiv 2610.10014 (supervised cognitive tasks on MaleCNS: +17 to +25 pp over rewires)
  - arXiv 2609.24565 (frozen fly vision: rewiring costs 0.27-0.36 accuracy)
- **Null or confounded:**
  - Dhiman arXiv 2604.04033: the advantage vanishes under shared initialization and degree-preserving nulls
  - Park's flyconnectome-nulls: standard nulls *beat* the connectome only because they create sensory→motor shortcuts; with boundary-preserving nulls the two are equivalent
  - many hobby nulls

No RL **game** study with degree-preserving controls, multiple seeds and held-out levels was found.

### Cited Findings

**Positive**
- **FlyGM** (arXiv 2602.17997) **[preprint]**: beats degree-preserving rewired. The rewired graph loses orientation control as difficulty rises ([arXiv](https://arxiv.org/pdf/2602.17997)).
- **Cho, Kang & Kim (KAIST), "A Drosophila Whole-Connectome Network Can Learn Human-Designed Cognitive Tasks", arXiv 2610.10014, 7 Oct 2026** **[preprint]**:
  - Model: MaleCNS v1.0 (166,700 nodes, 25.6M edges) as a fixed mask on a leaky tanh RNN. Only edge scalars are trained; input and readout are fixed.
  - Addition test: 92.77% vs rewired 67.93%.
  - Grounded language: 61.59% vs 44.17% (95% CI +11.4 to +24.9 pp). Rank 1/21 against 20 degree-preserving rewires.
  - A parameter-matched dense RNN (5,058 units) scores 8.72%, below the 16.67% chance level. No Erdős-Rényi control. Supervised, not RL ([arXiv](https://arxiv.org/html/2610.10014v1)).
- **Kuklev, arXiv 2609.24565 (21 Sep 2026)** **[preprint]**: a frozen connectome-constrained fly-vision model used for architectural drawings. Degree- and type-preserving rewiring costs 0.271-0.356 accuracy (3 seeds), so the exact wiring is "load-bearing".
  - However, the circuit does not beat its own receptor-only input on clean data.
  - Task-trained CNNs dominate on clean data (0.959 vs 0.857).
  - The fly pipeline degrades less under noise ([arXiv](https://arxiv.org/abs/2609.24565)).
- **NCAP** (NeurIPS 2022): data efficiency and parameter efficiency vs MLP. **No degree-preserving control** ([arXiv](https://arxiv.org/abs/2201.05242)).
- **Fly Self Driving** (suanmiao, Sep 2026) **[hobby repo]**:
  - Model: full MaleCNS (165,122 neurons) as an RNN; only per-synapse gains and per-neuron leaks are trained (25.7M params). BC from a pure-pursuit expert, then 3 DAgger rounds.
  - **58/60 on three held-out sets of unseen streets** with traffic. Randomly rewired graph 16/20; MLP 18/20; linear 12/20; fly with BC only 12/20.
  - **One seed per condition**, and it is unclear whether the rewired graph got DAgger ([GitHub](https://github.com/suanmiao/fly-self-driving)).

**Null, negative or confounded**
- **Dhiman, "Topological Sensitivity in Connectome-Constrained Neural Networks", arXiv 2604.04033 (5 Apr 2026)** **[preprint]**: in a flyvis-based model, the connectome looks much better against weak controls (a self-loop-matched random graph, recovered from a connectome-trained checkpoint). The early-loss advantage **disappears with shared random initialization**, and the activity advantage **disappears against a degree-preserving null** (5-sample ensemble). Conclusion: the advantages "largely stem from initialization and null-model confounds" ([arXiv](https://arxiv.org/abs/2604.04033)).
- **Park, flyconnectome-nulls (2026; Zenodo 10.5281/zenodo.22871090)** **[hobby/preprint, pre-registered]**:
  - Setup: FlyWire v783 compressed to 512 groups plus 1,000 KCs, used by agents evolved for foraging (neuroevolution, not RL).
  - Standard nulls (column shuffle, degree-preserving swaps) overtake the connectome by the last probe (10/10 seeds, p_holm = 0.0059).
  - Cause: these nulls create **direct olfactory→descending-motor shortcuts**, 10.6-10.7% of olfactory output vs 0.012% in the connectome. Transplanting the shortcuts into the connectome adds +0.44 fitness, and removing them from the null removes its lead.
  - Boundary-preserving nulls (only the interior randomised) are **equivalent** to the connectome within ±0.10.
  - Scrambling the interior of evolved agents loses 46-55% of fitness.
  - Includes a reported failed prediction ([GitHub](https://github.com/gyujeongion/flyconnectome-nulls)).
- **arXiv 2606.17745** (larval connectome, frozen rate operator) **[preprint]**: the title states "degree and weight govern the gross response, exact wiring governs input routing and mushroom-body modes". Per search snippet, it argues degree-preserving nulls destroy community structure and proposes block-preserving nulls ([arXiv](https://arxiv.org/pdf/2606.17745)). Not read in full.
- **Schmidgall et al. 2022**: the exact wiring need not be preserved for the benefit; architectural statistics suffice ([arXiv](https://arxiv.org/abs/2209.14406)).
- **nfly**: at equal parameter count "the fly is not cheaper and does not score higher". The sparse product costs ~4× the arithmetic and 35-57× the CPU time of an equal MLP ([GitHub](https://github.com/nzxhuong/nfly)).
- **Hobby nulls** (self-reported, from curated lists):
  - FlyAim (pre-registered): wiring adds nothing over a degree-preserving shuffle.
  - Fly Tic-Tac-Toe: degree-preserving rewired scores about the same.
  - ChessFly: real ≈ shuffled.
  - The Fly Hears: 70.0% vs 69.7% scrambled.
  - ENSOphilia: 85/100 rewired graphs match or beat the circuit.
  - fly-arithmetic: real wiring 48% vs random network 98% on held-out sums.
  - therealfly: a scrambled graph produces the same rhythm.
  - Flybrain gate course: a presynaptic shuffle also solves it.
  - Boxhead: survival level with a no-brain baseline.
  - FLYT3: fly classifier 60.8% vs pooled-pixel MLP 85.7%.
  - alpha-fly: the README says the wiring is "probably not special for chess".

  Sources: [awesome-fruit-fly](https://github.com/townie/awesome-fruit-fly); [awesome-fly](https://github.com/rustyorb/awesome-fly).
- A protocol repo explicitly frames Dhiman (advantage disappears) against FlyGM (wiring wins) as the open contradiction ([AayanShabbir/fly-brain-connectome-null-model](https://github.com/AayanShabbir/fly-brain-connectome-null-model)). A PR titled "A fly connectome as a fixed reservoir, its null models, and the negative result they produced" also exists ([selfplay-lab PR #1](https://github.com/zompreacher/selfplay-lab/pull/1)). Not read.

**Adjacent, non-RL**
- Biological Processing Units, arXiv 2507.10951: the larval connectome unmodified as a substrate for image recognition and chess puzzles ([arXiv](https://arxiv.org/pdf/2507.10951)).
- Adult fly connectome as an ESN reservoir for chaotic time series, reported more resistant to overfitting (*Biomimetics* 2025) ([PMC12109256](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12109256/)).
- MICrONS-derived RNN inductive biases, arXiv 2606.14975 ([arXiv](https://arxiv.org/pdf/2606.14975)).
- McAllister et al., "Non-random brain connectome wiring enables robust and efficient neural network function under high sparsity" (bioRxiv 2026.03.30.715411): cited by 2610.10014 but not located, so its claims are unverified ([citation in 2610.10014](https://arxiv.org/html/2610.10014v1)).

### Inferences
- Park's shortcut artefact is the **same mechanism Mosca identified and controlled** with the fixed `n_shared = 92` port overlap: randomised graphs or ports create direct sensory→motor paths that bypass recurrence. Park found it in nulls, Mosca in port selection. This is independent corroboration that comparisons can be decided by I/O shortcuts rather than interior wiring.
  - Corollary: any positive real-vs-shuffled result that does not control the sensory-motor boundary (fly-plays-nes, Connectocopter, 20fly8, and possibly FlyGM) is suspect in either direction.
- Dhiman's shared-initialization point supports Mosca's practice of training all six models from the same recipe and initialisation scheme.
- Positive results cluster where the wiring is **frozen or only gain-tuned** and the task touches the circuit's native function (vision, locomotion), or where the readout and interface are fixed (2610.10014). Nulls cluster in arbitrary game and cognitive tasks with trained readouts. Mosca (trained weights inside the mask, a non-native task, a tie between wirings) fits the null cluster.
- No published RL result shows **sample-efficiency** gains vs degree-preserving controls in games. FlyGM is the only RL positive, and it is locomotion, native to the fly.

### Gaps
- Not read in full: FlyGM's seed counts and effect sizes, 2606.17745, the McAllister bioRxiv, the "Yu et al. 2026 larval reservoir chess" paper cited by 2610.10014, and "Morra et al. 2025 Natural Computing" / "Zhang et al. 2026 Fundamental Research" (also cited by 2610.10014).
- No peer-reviewed 2024-2026 RL paper with degree-preserving controls was found. All are preprints.

---

## Q4. Imitation + RL fine-tuning with biological wiring; disjoint test levels

### Takeaway
There are precedents, but almost all are 2026 preprints or hobby repos:
- **FlyGM** (imitation + PPO, academic preprint)
- **flywire-pong** (supervised init → PPO; no controls)
- **fly-fpv** (BC from a reflex pilot → PPO with connectome/shuffled/ER/MLP arms, but **no results posted**)
- **nfly** (RL from a cloned head; failed on Pong)

Pure imitation or DAgger with connectome wiring is more common (Fly Self Driving, fly-mario, fly-games, fly-mania, FLYNN, the C. elegans digital twin). **Disjoint held-out environments** exist in some (unseen streets, held-out seeds, held-out benchmarks), but **no one combines a connectome policy with BC + RL fine-tuning on disjoint game levels**, and none does it on SMB.

### Cited Findings
- **FlyGM**: imitation then PPO with a whole-FlyWire graph policy on flybody ([arXiv](https://arxiv.org/pdf/2602.17997)).
- **flybody** itself trains MLP controllers via RL to track real fly trajectories, with no connectome ([bioRxiv](https://www.biorxiv.org/content/10.1101/2024.03.11.584515v1); [awesome-fruit-fly](https://github.com/townie/awesome-fruit-fly)). The "imitation via RL" scaffold that FlyGM reuses.
- **flywire-pong**: "supervised initialization followed by PPO". The supervision target is not stated, there are no controls, and it wins 43/60 against the Atari CPU ([GitHub](https://github.com/zonzujiro/flywire-pong)).
- **fly-fpv / flypv** **[hobby repo, 90 commits]**: closest in design to Mosca.
  - Model: the MaleCNS mask (absent edges stay zero) with learned weights and signs (no Dale's law), 6 shared-mask residual iterations per decision, and no persistent state.
  - Training: BC from a geometric reflex pilot, then PPO with a curriculum.
  - Arms: `connectome`, `shuffled` (degree-preserving), `erdos`, and a parameter-matched `mlp`, plus `--no-vision` and frozen-random-edge ablations.
  - **No results and no held-out tracks reported** ([GitHub](https://github.com/joey-david/fly-fpv)).
- **nfly**: RL started from a cloned MLP head reaches only −14 to −16 on Pong. A cloned frozen head reaches +8.0, and cloning solves CartPole. Fine-tuning by RL did not improve on cloning ([GitHub](https://github.com/nzxhuong/nfly)).
- **Imitation / DAgger only, connectome wiring**:
  - **Fly Self Driving**: BC + 3 DAgger rounds; 58/60 on held-out streets vs rewired 16/20 (one seed) ([GitHub](https://github.com/suanmiao/fly-self-driving)).
  - **fly-mario**: teacher demos + DAgger, readout only ([GitHub](https://github.com/ksanjeev284/fly-mario)).
  - **fly-games**: ridge readout cloned from a scripted policy; fails in closed loop through compounding error ([GitHub](https://github.com/shantanugoel/fly-games)).
  - **fly-mania**: frozen subgraph + DAgger on osu!mania ([awesome-fruit-fly](https://github.com/townie/awesome-fruit-fly)).
  - **FLYNN**: DAgger per the curated list; not stated in the abstract ([awesome-fruit-fly](https://github.com/townie/awesome-fruit-fly); [arXiv](https://arxiv.org/abs/2607.00025)).
  - **super-fly**: teacher-trajectory pretraining, then dopamine-STDP, on SMB 1-1 with no results ([GitHub](https://github.com/EthanThatOneKid/super-fly)).
  - **C. elegans digital twin** (2023): supervised on PID-controller data ([MDPI doi](https://doi.org/10.3390/math11112442)).
- **Brain-inspired (not connectome)**: NCP-based "Brain-Inspired Deep Imitation Learning for Autonomous Driving" (2021) reports better generalisation on unseen data (Udacity, comma.ai) ([ar5iv 2107.14654](https://ar5iv.labs.arxiv.org/html/2107.14654)). The NCP *NMI* 2020 driving controller learns steering from data ([ISTA](https://research-explorer.ista.ac.at/record/8679)). Whether that was imitation was not confirmed from the abstract.
- **Held-out evaluation in connectome control**:
  - Fly Self Driving: three sets of 20 unseen streets ([GitHub](https://github.com/suanmiao/fly-self-driving)).
  - fly-cartpole: held-out seeds ([awesome-fly](https://github.com/rustyorb/awesome-fly)).
  - Fly Dino: held-out benchmarks (same list).
  - 2609.24565: a preregistered held-out split ([arXiv](https://arxiv.org/abs/2609.24565)).
  - FLYNN: out-of-distribution evaluation ([arXiv](https://arxiv.org/abs/2607.00025)).
  - **No Mario or platformer connectome project reports disjoint levels.** fly-plays-nes, fly-mario, fly-games and super-fly are all on 1-1 or unspecified levels.

### Inferences
- Mosca's 6c pilot findings match the only hobby datapoints available:
  - From scratch, RL fails: Mosca's degenerate "always jump right" policy; nfly's Pong at −20.5.
  - Cloning gets most of the way: Mosca's clone completes 1-1; nfly's CartPole 500/500; Fly Self Driving.
  - RL fine-tuning on top adds little or something modest: nfly −14 to −16; Mosca +0.137 in its procedural pilot.
- The design Mosca is running (trainable masked RNN with Dale's law + BC + PPO + degree-preserving/ER/dense/no-core/MLP controls + disjoint NES levels + multiple seeds) **does not appear to have been done or published elsewhere**. fly-fpv has the closest planned design (same arms, BC→PPO), but it has no results, no Dale's law, no memory and no held-out tracks.
- FlyGM is the only academic precedent for imitation + RL with full-connectome wiring, and it is locomotion, not games, with weaker (unweighted) nulls than Mosca's.

### Gaps
- fly-fpv may publish results later. It should be re-checked before any write-up, because it is the nearest competitor design.
- The FlyGM imitation details (dataset, how much RL adds over imitation) were not re-read, by instruction.
- Whether any 2026 conference paper (CoRL, NeurIPS 2026 decisions) combines connectome policies with BC + RL on held-out game levels could not be checked. None surfaced on arXiv in these searches.
