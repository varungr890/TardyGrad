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
- **Feedback alignment / Kolen–Pollack** *(explored)* — break weight symmetry while still
  delivering a useful teaching signal.

alongside **spiking neural networks (SNNs)**, where time and delayed reward are intrinsic
rather than bolted on.

## Status

This is an active course project from a Neuromatch Academy NeuroAI pod. The repository
now follows the project from static MNIST comparisons to spiking networks and delayed
feedback experiments. Results are intentionally reported with their task, architecture,
and experimental caveats because the notebooks do not all use the same benchmark.

| Area | Status | Where |
|---|---|---|
| **Q1 — Learning-rule comparison** (backprop vs. Hebbian vs. node perturbation vs. predictive coding, on MNIST) | **Implemented** | `Q1_Model_Comparison/` |
| **Q2 — Learning rules in spiking networks** (snnTorch, leaky integrate-and-fire) | **Implemented / being consolidated** | `Combined_SNN_LearningRules.ipynb` |
| Feedback alignment and Kolen–Pollack | **Explored** | `Bio_Plausible_SNN.ipynb`; `Flexible_SNN_v_MLP_Comparison.ipynb` |
| **Q3 — Delayed and probabilistic feedback** | **Exploratory** | `Flexible_SNN_v_MLP_Comparison.ipynb`; `Acc_v_Feedback_Delay_Prob.ipynb` |
| **Q3 — Delayed feedback with sequence memory** | **Exploratory** | `RSNN_on_Probabilistic_Delayed_Feedback_MNIST*.ipynb` |
| Eligibility traces / BTSP for delayed credit | Background reading | see References |

## Questions and findings

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

The Q1 combined experiment uses a reduced **3-class MNIST** task. In the reported run,
predictive coding reached 99.17% accuracy, backpropagation 98.86%, node perturbation
98.60%, and Hebbian learning 36.07%. The predictive-coding result is especially useful
because its update direction has cosine similarity ≈ 0.99998 with the true backprop
gradient without explicitly computing that gradient.

Node perturbation was also benchmarked separately on **full 10-class MNIST** across five
seeds: 86.70% ± 0.73%, compared with 92.28% ± 0.08% for backpropagation and 10.95% for
the existing Hebbian configuration. These are different tasks and should not be
compared head-to-head. Node perturbation received substantially more hyperparameter
tuning than the other rules, which is another important caveat.

### Q2 — Does the substrate change the comparison?

`Microlearning_SNN.ipynb` is the Neuromatch tutorial backbone. The newer
`Combined_SNN_LearningRules.ipynb` ports the comparison to a two-layer LIF network
(`784 → 100 → 10`) and evaluates accuracy from spike counts. Its reported final
accuracies are approximately 95.1% for backpropagation, 89.7% for predictive coding,
52.1% for node perturbation, and 11.7% for Hebbian learning. The qualitative ordering
from Q1 mostly remains, but node perturbation is much less stable after being ported to
spikes. The predictive-coding implementation is experimental rather than a strict
spiking predictive-coding formulation.

`Bio_Plausible_SNN.ipynb` provides a compact SNN baseline for backpropagation versus
feedback alignment. `Flexible_SNN_v_MLP_Comparison.ipynb` studies how deterministic
delay and probabilistic loss feedback affect an SNN and an MLP. Kolen–Pollack is also
explored there, but its reported 90.91% versus 84.16% comparison is a separate
preliminary run, not part of the main controlled factorial experiment.

### Q3 — What happens when feedback is delayed?

The feedforward experiments vary delay and the probability that a learning signal is
received. Delay alone degrades the SNN more strongly than the MLP control, while the SNN
is surprisingly robust when feedback is randomly dropped. When low feedback probability
and long delay are combined, that advantage disappears: the reported worst case is about
80.68% for the SNN and 82.88% for the MLP.

`Acc_v_Feedback_Delay_Prob.ipynb` and
`RSNN_on_Probabilistic_Delayed_Feedback_MNIST_latest.ipynb` move closer to a true delayed-
credit task by comparing a memoryless MLP, an RNN, and a recurrent SNN on delayed MNIST
sequences. These notebooks remain exploratory: class selections vary between runs,
some results are stored mainly in plots, and the most ambitious run was interrupted.
They should be treated as preliminary evidence until one fixed configuration is rerun
to completion and its numerical results are recorded.

### Working hypothesis

Across the project, the central hypothesis is that biologically plausible credit
assignment can approach backpropagation, but the cost of locality, temporal dynamics,
and delayed feedback is not uniform across learning rules. The current evidence supports
that qualified claim: predictive coding is closest to backpropagation in Q1, while node
perturbation and spiking dynamics expose substantial stability and compute trade-offs.

## Presentation storyline

The notebooks answer the project questions in this order:

1. `Q1a_combined.ipynb` — compare learning rules on a common MNIST task.
2. `Combined_SNN_LearningRules.ipynb` — port the rules to a spiking substrate.
3. `Flexible_SNN_v_MLP_Comparison.ipynb` — test delayed and probabilistic feedback.
4. `Acc_v_Feedback_Delay_Prob.ipynb` — extend the feedback-delay sweep.
5. `RSNN_on_Probabilistic_Delayed_Feedback_MNIST_latest.ipynb` — test memory in a delayed sequence task.

For the final presentation, keep the story concise: motivate the biological constraint,
state Q1–Q3, show the controlled Q1/Q2 results, then present Q3 as an exploratory test
of temporal credit assignment. Do not combine the 3-class and 10-class accuracies into
one ranking, and label the sequence-task results as preliminary until the interrupted
run has been completed.

## Repository at a glance

- `Q1_Model_Comparison/` — the canonical Q1 work: the combined four-rule notebook, the
  node-perturbation sub-study, and their READMEs.
- `Microlearning_SNN.ipynb` — Neuromatch tutorial base + the Q2 spiking starter.
- `Bio_Plausible_SNN.ipynb`, `Combined_SNN_LearningRules.ipynb`, and
  `Flexible_SNN_v_MLP_Comparison.ipynb` — SNN baselines, learning-rule comparisons,
  and delayed/probabilistic feedback experiments.
- `Acc_v_Feedback_Delay_Prob.ipynb` and `RSNN_on_Probabilistic_Delayed_Feedback_MNIST*.ipynb`
  — exploratory sequence-level delayed-feedback experiments.
- `MLP_with_*.ipynb` and the root `Q1a_combined.ipynb` — earlier single-rule drafts,
  **superseded** by `Q1_Model_Comparison/`; kept for history.
- `Q3_LSNN_accuracy_pixel/` — generated LSNN accuracy and error figures.
- Branches: `main` is the consolidated project branch; `q1a` is an older Q1 snapshot.

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
