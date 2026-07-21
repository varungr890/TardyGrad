# Q1: Does PC Significantly Improve HL Performance Compared to BP?

## Research Question

> Backpropagation (BP) is the main gradient descent method in current Machine Learning literature. However, the forward and backward passes use symmetric weights during learning for error-based updates, which is not biologically plausible. Recent neuromorphic computing methods propose variants of Hebbian Learning (HL) implementations as alternatives to backpropagation such as Weight Perturbation, Node Perturbation, Feedback Alignment, Kolen-Pollack and Predictive Coding (PC) algorithms. For our first question, we ask whether PC improves the performance of simpler HL methods towards convergence as fast as BP.

In short: BP is the accuracy benchmark, but it's biologically implausible. PC is the group's focal candidate for closing that gap — this directory tests it directly against a simpler HL baseline (Node Perturbation) and against BP, to see whether PC's added complexity actually earns its keep.

## Directory Structure

```
Q1_Model_Comparison/
├── Node_Perturbation/
│   ├── Node_Perturbation_copy_of_MLP_with_backprop_and_Hebbian.ipynb
│   ├── MLP_with_node_perturbation.ipynb
│   └── README.md
└── Q1a_combined.ipynb
```

## `Node_Perturbation/`

The simpler HL baseline PC is measured against. Implemented, tuned across four stages, and benchmarked separately against BP and standard Hebbian learning on full **10-class** MNIST. See the subdirectory's own README for details; headline numbers from that separate 10-class run:

| Rule | Accuracy (5 seeds, 10-class) |
|---|---|
| Backprop | 92.28% ± 0.08% |
| Node perturbation | 86.70% ± 0.73% |
| Hebbian | 10.95% (collapsed to a single predicted class) |

## `Q1a_combined.ipynb`

The actual side-by-side comparison this question asks for: BP, Hebbian, Node Perturbation, and Predictive Coding, all trained on the same **3-class**, 10-epoch task, sharing a single interface (`Learner` → `train_batch()`/`probs()`) so every rule is trained and evaluated identically. Note this uses a different, easier task (3 classes, not 10) than `Node_Perturbation/`'s standalone benchmark above — the two shouldn't be compared directly against each other.

**Current results** (single run, seed=0, all four rules):

| Rule | Final loss | Final accuracy | Notes |
|---|---|---|---|
| **Predictive coding** | 0.5674 | **99.17%** | Loss is inflated by a known scale artifact (trained against squared error on a linear output) — accuracy, not loss, is the fair read of its decision quality |
| Backprop | 0.0406 | 98.86% | |
| Node perturbation | 0.0525 | 98.60% | Simpler HL baseline PC is measured against |
| Hebbian | 3.0925 | 36.07% | Peaks near 54% around epoch 4, then diverges |

**Answer to Q1, from this run:** predictive coding does converge, and it does narrowly hold the top accuracy of the four rules (99.17%). But the margin over the simplest alternative tested — node perturbation, a rule with no inference dynamics at all — is under a point (0.57), and PC pays a real structural cost to get there: its inference step relaxes over 20 timesteps *per batch* before every single weight update, far more computation per training step than node perturbation's two forward passes or backprop's one forward-and-backward pass. Its raw loss number looks like the worst of the three working rules, but that's the scale artifact above, not worse learning — accuracy is what should be read for PC specifically. Taken together, the honest version of Q1's answer is: **PC converges and edges out the simpler HL baseline here, but the margin is narrow enough, and the per-step cost high enough, that its added complexity is not yet clearly justified by this result.** Hebbian, at the settings tested, does not converge on this task at all, so it isn't a meaningful comparison point for PC either way.

**Caveat — single seed, and an open question specifically for PC.** This table comes from one run per rule (`seed=0`). Node perturbation's own reliability across seeds has been separately confirmed on this same 3-class task (98.0% ± 0.2% across 5 seeds), so its number here is known to be stable, not a lucky draw. That same check has not been run for PC. This cuts both ways for reading PC's result: if PC is running at a first-working configuration rather than a tuned one, its already-narrow edge might have real headroom to grow with tuning — or it might be more fragile than a single run suggests, since it hasn't been stress-tested the way node perturbation has. Worth resolving with whoever owns the PC implementation before treating either reading as the team's final answer.
