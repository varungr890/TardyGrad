# TardyGrad — Delayed Reward

*Biologically-plausible credit assignment and local learning rules in artificial and
spiking neural networks.*

> **Why "TardyGrad"?** A tardigrade that carries its gradient late — *tardy grad*ient.
> Late but helpful; no backprop, no problem. A [Neuromatch Academy](https://neuromatch.io/)
> group project.

## The question

Backpropagation is the workhorse of modern deep learning, but it is not biologically
plausible: it requires a symmetric backward pass that reuses the exact forward weights,
and it assigns credit with information the brain does not obviously have — especially when
the reward or feedback signal is **delayed**. How, then, does a physical neural system
assign credit for an outcome back to the synapses responsible for it?

This project studies **local, biologically-plausible alternatives to backprop** — rules
that update a synapse using only signals available at that synapse — and asks how close
they come to backprop's performance, and at what cost. The learning rules in scope:

- **Hebbian / STDP** — "fire together, wire together"; no error signal at all.
- **Node perturbation** — estimate the gradient by random search; a three-factor,
  reward-modulated Hebbian rule.
- **Predictive coding** — represent error explicitly in dedicated neurons and let the
  network relax to transport it; provably approximates backprop.
- **Feedback alignment / Kolen–Pollack** *(planned)* — break weight symmetry while still
  delivering a useful teaching signal.

alongside **spiking neural networks (SNNs)**, where time and delayed reward are intrinsic
rather than bolted on.

## Status

This is an active course project. What is in the repository today is a solid foundation;
the broader delayed-reward agenda is the direction of travel, not a finished result.

| Area | Status | Where |
|---|---|---|
| **Q1 — Learning-rule comparison** (backprop vs. Hebbian vs. node perturbation vs. predictive coding, on MNIST) | **Implemented** | `Q1_Model_Comparison/` (on `main`) |
| **Q2 — Spiking networks vs. MLPs** (snnTorch, leaky integrate-and-fire) | **In progress** | `Microlearning_SNN.ipynb`; more on the `q2` branch |
| Feedback alignment, Kolen–Pollack | Planned | — |
| Delayed-reward tasks — Pong, rhythmic-timing, Sequential MNIST | Planned | — |
| Eligibility traces / BTSP for delayed credit | Background reading | see References |

## What's implemented

### Q1 — Do biologically-plausible rules approach backprop?

`Q1_Model_Comparison/Q1a_combined.ipynb` puts four learning rules behind a single common
interface (a `Learner` exposing `probs()` and `train_batch()`) and trains them on an
identical MNIST task, so they are genuinely swappable and scored with the same yardstick.
The analysis sections are the payoff: they show predictive coding's weight updates match
the true backprop gradient (cosine similarity ≈ 0.99998) without ever computing a
gradient, and they diagnose *why* the plain Hebbian rule collapses — and how a one-line
fix roughly doubles its accuracy.

`Q1_Model_Comparison/Node_Perturbation/` benchmarks node perturbation separately on the
full 10-class MNIST task, with a four-stage hyperparameter search and a confusion-matrix
analysis. See the READMEs inside `Q1_Model_Comparison/` for the headline numbers and
their caveats (notably: the 3-class combined run and the 10-class node-perturbation run
are **different tasks and should not be compared head-to-head**).

### Q2 — Spiking networks

`Microlearning_SNN.ipynb` is the Neuromatch tutorial backbone the rest of the project
builds on (it defines the shared MLP, optimizer, and Hebbian machinery) and introduces a
leaky integrate-and-fire `SNN_MLP` via [snnTorch](https://snntorch.readthedocs.io/). Work
comparing spiking and non-spiking architectures continues on the `q2` branch.

## Repository at a glance

- `Q1_Model_Comparison/` — the canonical Q1 work: the combined four-rule notebook, the
  node-perturbation sub-study, and their READMEs.
- `Microlearning_SNN.ipynb` — Neuromatch tutorial base + the Q2 spiking starter.
- `MLP_with_*.ipynb` and the root `Q1a_combined.ipynb` — earlier single-rule drafts,
  **superseded** by `Q1_Model_Comparison/`; kept for history.
- Branches: `main` (Q1), `q2` (spiking work, not yet merged).

## Team

A Neuromatch Academy pod. GitHub handles filled in where known — members, please add yours.

| Member | GitHub |
|---|---|
| Varun Chokshi | [@varungr890](https://github.com/varungr890) |
| Dylan Picart | [@dylanpicart](https://github.com/dylanpicart) |
| Alexandre Gomes Caldeira | @— |
| Lukas Valenzuela | @— |
| Bruno Bustos | [@BrunoBustos96](https://github.com/BrunoBustos96) |
| Sean Afridi | @— |
| Raylynn Zhou | @— |
| Josh Reback | [@joshreback](https://github.com/joshreback) |
| Joel Tabak | @— |

**Mentor:** Mostafa — [@Miandari](https://github.com/Miandari)

## References

**Predictive coding & the free-energy principle**
- Whittington & Bogacz (2017), *An approximation of the error backpropagation algorithm in
  a predictive coding network with local Hebbian plasticity* — https://pmc.ncbi.nlm.nih.gov/articles/PMC5467749/
- *Predictive coding is energetically efficient* — https://pmc.ncbi.nlm.nih.gov/articles/PMC9768680/
- Bogacz, *A tutorial on the free-energy framework* — https://www.sciencedirect.com/science/article/pii/S0022249615000759
- Reference implementation — https://github.com/alec-tschantz/predcoding

**Node perturbation**
- Fiete & Seung (2006), *Gradient learning in spiking networks by dynamic perturbation of
  conductances*, PRL 97(4):048104.
- Hiratani, Mehta, Lillicrap & Latham (2022), *On the stability and scalability of node
  perturbation learning*, NeurIPS.

**Bio-plausible learning, benchmarks & spiking**
- *Comparison of bio-plausible neural nets on MNIST* — https://www.science.org/doi/epdf/10.1126/sciadv.adn6076
- *Spiking neural network + backpropagation + Pong* — https://www.nature.com/articles/s41467-020-17236-y
- snnTorch tutorials — https://snntorch.readthedocs.io/en/latest/tutorials/index.html

**Delayed reward, eligibility traces & timing**
- Sutton & Barto, *Reinforcement Learning* — eligibility traces — http://www.incompleteideas.net/book/ebook/node72.html
- *Eligibility traces in neural circuits* — https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2018.00053/full
- Behavioral timescale synaptic plasticity (BTSP) — https://www.annualreviews.org/content/journals/10.1146/annurev-neuro-090919-022842
- *Neural state trajectories underlie the tempo of rhythmic tapping*, PLOS Biology — https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000054

**Course**
- Neuromatch NeuroAI foundations — https://neuroai.neuromatch.io/w2d3-intro/

---
*Educational research project. Not affiliated with or endorsed by any institution beyond
the Neuromatch Academy program.*
