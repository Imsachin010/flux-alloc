# Third-Party Software and Licenses

FluxAlloc depends on and incorporates open-source software developed by third parties. This document records the third-party software packages used by the project and their respective licenses.

---

## 1. Stable-Baselines3

* **Usage:** Used for the core Reinforcement Learning algorithms and neural policy infrastructure (PPO).
* **Project:** Stable-Baselines3
* **Repository:** https://github.com/DLR-RM/stable-baselines3
* **License:** MIT License

## 2. SB3-Contrib

* **Usage:** Used for the `MaskablePPO` implementation enabling action-masking over valid memory blocks.
* **Project:** Stable-Baselines3 Contrib
* **Repository:** https://github.com/Stable-Baselines-Team/stable-baselines3-contrib
* **License:** MIT License

## 3. Gymnasium

* **Usage:** Used to implement the custom Markov Decision Process (MDP) simulation environment for dynamic memory placement.
* **Project:** Farama Gymnasium
* **Repository:** https://github.com/Farama-Foundation/Gymnasium
* **License:** MIT License

## 4. PyTorch

* **Usage:** Deep learning tensor library used for training and evaluating Transformer and MLP Neural Ranker models.
* **Project:** PyTorch
* **Repository:** https://github.com/pytorch/pytorch
* **License:** BSD-3-Clause License

## 5. NumPy

* **Usage:** Vectorized numerical operations, heap state arrays, and mathematical computations.
* **Project:** NumPy
* **Repository:** https://github.com/numpy/numpy
* **License:** BSD-3-Clause License

## 6. SciPy

* **Usage:** Statistical hypothesis testing, paired two-sided Wilcoxon signed-rank tests, and rank correlation.
* **Project:** SciPy
* **Repository:** https://github.com/scipy/scipy
* **License:** BSD-3-Clause License

## 7. Matplotlib

* **Usage:** Publication-quality plotting and visualization generation for memory fragmentation and latency tradeoffs.
* **Project:** Matplotlib
* **Repository:** https://github.com/matplotlib/matplotlib
* **License:** PSF / BSD-compatible License

## 8. Pandas

* **Usage:** Dataframe manipulation, log parsing, and structured tabular results logging.
* **Project:** Pandas
* **Repository:** https://github.com/pandas-dev/pandas
* **License:** BSD-3-Clause License

## 9. TensorBoard

* **Usage:** Experiment tracking and training telemetry visualization for PPO agents.
* **Project:** TensorBoard
* **Repository:** https://github.com/tensorflow/tensorboard
* **License:** Apache License 2.0

---

## Dependency Notice

The complete runtime dependency set is specified in [`requirements.txt`](requirements.txt). All third-party software remains subject to its respective upstream license terms. This repository's license applies solely to the original source code, models, and documentation created by the authors.
