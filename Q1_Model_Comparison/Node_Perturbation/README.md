# Node Perturbation

## Overview

Node perturbation is a biologically plausible alternative to backpropagation for training neural networks — one of several such rules explored in this project alongside Hebbian learning. Rather than computing exact gradients via the chain rule, node perturbation estimates them by randomly perturbing neuron activity and correlating the resulting change in loss with the perturbation itself. This makes it a zeroth-order, finite-difference-style gradient estimator — the same family as SPSA and REINFORCE — and formally equivalent to a reward-modulated, three-factor Hebbian plasticity rule.

Because node perturbation needs a single global signal (whether the network's overall loss improved) rather than local, layer-by-layer information, it can't be implemented as a per-layer custom autograd function the way Hebbian learning was in this project. This implementation instead runs a clean and a perturbed forward pass, computes the resulting weight update by hand, and sets `.grad` directly, so it still works with the project's existing SGD-style optimizer unmodified. One deliberate improvement over the course's reference implementation: output-layer noise is injected into the pre-softmax logits rather than the post-softmax probabilities, keeping every perturbed output a valid probability distribution.

## Key Results

- **Tuned across a four-stage hyperparameter search** (`noise_std` and learning rate) — first on an easy 3-class subset, then re-verified on the full 10-class task, since the easier task's tuning didn't fully transfer. Final settings: `noise_std=0.15, lr=0.01`.
- **86.70% ± 0.73% accuracy** on full 10-class MNIST across 5 seeds, reliably learning all ten digit classes — behind backprop (92.28% ± 0.08%), but functioning where Hebbian learning collapsed entirely (10.95%, always predicting a single class) at the team's existing settings.
- **Confusion matrix analysis** showed errors landing in intuitively sensible places (e.g. digit 3 misread as 5, digit 4 misread as 9) rather than randomly, suggesting the model learned genuine digit structure rather than a rule-specific artifact.

## Caveat on the comparison

This comparison is fair in *procedure* — same task, same seeds, same epoch count for all three rules — but not in *tuning effort*. Node perturbation went through four rounds of hyperparameter sweeps; Hebbian and backprop ran at their existing/default settings, never independently retuned for the harder 10-class task. The defensible claim is that node perturbation, tuned, outperformed Hebbian and approached backprop *at their existing settings* — not that it is the better rule in general.

## Notebooks

| File | Contents |
|---|---|
| `Node_Perturbation_copy_of_MLP_with_backprop_and_Hebbian.ipynb` | Full working notebook — implementation, all four tuning stages, and the statistical comparison against backprop and Hebbian. |
| `MLP_with_Node_Perturbation.ipynb` | Clean, minimal notebook: final implementation, the optimized model, and its results. Tuning process summarized as text rather than re-run. |

## References

- Fiete, I. R., & Seung, H. S. (2006). Gradient learning in spiking neural networks by dynamic perturbation of conductances. *Physical Review Letters*, 97(4), 048104.
- Fiete, I. R., Fee, M. S., & Seung, H. S. (2007). Model of birdsong learning based on gradient estimation by dynamic perturbation of neural conductances. *Journal of Neurophysiology*, 98(4), 2038–2057.
- Miconi, T. (2017). Biologically plausible learning in recurrent neural networks reproduces neural dynamics observed during cognitive tasks. *eLife*, 6, e20899.
- Hiratani, N., Mehta, Y., Lillicrap, T. P., & Latham, P. E. (2022). On the stability and scalability of node perturbation learning. *NeurIPS*.
