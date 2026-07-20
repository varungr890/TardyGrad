# Q1: Does PC Significantly Improve HL Performance Compared to BP?

## Research Question

> Backpropagation (BP) is the main gradient descent method in current Machine Learning literature. However, the forward and backward passes use symmetric weights during learning for error-based updates, which is not biologically plausible. Recent neuromorphic computing methods propose variants of Hebbian Learning (HL) implementations as alternatives to backpropagation such as Weight Perturbation, Node Perturbation, Feedback Alignment, Kolen-Pollack and Predictive Coding (PC) algorithms. For our first question, we ask whether PC improves the performance of simpler HL methods towards convergence as fast as BP.

In short: BP is the accuracy benchmark, but it's biologically implausible. Several bio-plausible alternatives exist at different levels of sophistication — this directory compares them, with Predictive Coding (PC) tested specifically against the simpler HL variants and against BP, to see whether the added complexity of PC actually buys anything.

## Directory Structure

```
Q1/
├── node_perturbation/
│   ├── Node_Perturbation_copy_of_MLP_with_backprop_and_Hebbian.ipynb
│   ├── MLP_with_Node_Perturbation.ipynb
│   └── README.md
└── Q1a_combined.ipynb
```

## `node_perturbation/`

One of the simpler HL variants named in the question above. Implemented, tuned across four stages, and benchmarked directly against BP and standard Hebbian learning on full 10-class MNIST. See the subdirectory's own README for details; headline numbers:

| Rule | Accuracy (5 seeds) |
|---|---|
| Backprop | 92.28% ± 0.08% |
| Node perturbation | 86.70% ± 0.73% |
| Hebbian | 10.95% (collapsed to a single predicted class) |

This gives Q1 a concrete data point on the "simpler HL methods" side of the question: node perturbation, once tuned, converges reliably and learns all ten classes — a meaningfully higher bar than plain Hebbian learning cleared at the settings tested here.

## `Q1a_combined.ipynb`

The aggregate comparison this question is actually asking for: BP, the HL variants (node perturbation, and whichever of weight perturbation / feedback alignment / Kolen-Pollack the team implements), and PC, evaluated side by side on the same task. This is where Q1 gets its final answer — whether PC's added complexity measurably closes the gap to BP faster than the simpler HL methods do, or whether it performs comparably to something like node perturbation without the extra machinery.

**Status:** node perturbation's numbers above are ready to drop into this comparison. PC and any additional HL variants still need their own tuned results before Q1 can be answered in full — this file should be updated once those land, following the same format (mean ± std across matched seeds, same task, same epoch count) so the comparison stays fair across contributors.

## Note on comparing across contributors

Same caveat that applies inside `node_perturbation/`, worth repeating at this level since more rules are being added here: a comparison is only as fair as the tuning effort behind each entry. If PC or another HL variant is reported at its first working configuration while node perturbation reflects four rounds of sweeps, the comparison will make PC look worse than it might actually be, not because it *is* worse. Before finalizing Q1's answer, confirm each rule in `hl_performance_model_comparisons.md` got a comparable tuning pass — or note explicitly, per rule, when it didn't.
