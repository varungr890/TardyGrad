# Q1: Does PC Significantly Improve HL Performance Compared to BP?
---

## Research Question

> Backpropagation (BP) is the main gradient descent method in current Machine Learning literature. However, the forward and backward passes use symmetric weights during learning for error-based updates, which is not biologically plausible. Recent neuromorphic computing methods propose variants of Hebbian Learning (HL) implementations as alternatives to backpropagation such as Weight Perturbation, Node Perturbation, Feedback Alignment, Kolen-Pollack and Predictive Coding (PC) algorithms. For our first question, we ask whether PC improves the performance of simpler HL methods towards convergence as fast as BP.

In short: BP is the accuracy benchmark, but it's biologically implausible. Several bio-plausible alternatives exist at different levels of sophistication — this directory compares them, with Predictive Coding (PC) tested specifically against a simpler HL variant (Node Perturbation) and against BP, to see whether PC's added complexity actually buys anything.

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

The simpler HL variant named in the question above, developed and tuned separately before being folded into the combined comparison below. Implemented, tuned across four stages, and benchmarked directly against BP and standard Hebbian learning on full **10-class** MNIST. See the subdirectory's own README for details; headline numbers from that separate 10-class run:

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
| Backprop | 0.0406 | 98.86% | |
| Node perturbation | 0.0525 | 98.60% | Loss is directly comparable to backprop's — same NLL scale |
| Predictive coding | 0.5674 | 99.17% | Loss is inflated by a known scale artifact (trained against squared error on a linear output); accuracy is the fairer read here |
| Hebbian | 3.0925 | 36.07% | Peaks near 54% around epoch 4, then diverges — loss climbs for the rest of training even as predictions stop improving. Diagnosed in the notebook's own section on Hebbian's failure mode. |

**Answer to Q1, from this run:** PC does not show a decisive edge over the simpler node perturbation rule here — node perturbation's accuracy sits within 0.6 points of PC's, and its loss doesn't need PC's calibration caveat to be read fairly. Hebbian, at the settings tested, does not converge on this task.

**Caveat — single seed.** This table comes from one run per rule (`seed=0`). Node perturbation's own reliability across seeds has been separately confirmed on this same 3-class task — 98.0% ± 0.2% across 5 seeds, in `Node_Perturbation/`'s own notebook — so its 98.60% here is consistent with a stable result, not a lucky draw. That check hasn't been run for backprop, Hebbian, or PC in this notebook; their single-run numbers above should be read as a first pass, not a final word on how stable each rule is across seeds.

## Note on comparing across contributors

A comparison is only as fair as the tuning effort behind each entry. Node perturbation's settings (`noise_std=0.15, lr=0.01`) reflect four rounds of hyperparameter sweeps in a companion notebook; it isn't clear from `Q1a_combined.ipynb` alone whether PC's and Hebbian's settings received a comparable pass, or whether they're running at first-working defaults. If the latter, PC's and Hebbian's results here may understate what those rules are actually capable of — worth checking with whoever owns each implementation before treating this table as Q1's final answer.
