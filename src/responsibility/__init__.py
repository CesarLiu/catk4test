"""Counterfactual-responsibility interpretability framework.

Implements Hsu, Leung, Chen, Fisac & Pavone, "Interpretable Trajectory
Prediction for Autonomous Vehicles via Counterfactual Responsibility"
(IROS 2023), on top of this repository's SMART tokenized traffic model
trained on the Waymo Open Motion Dataset.

Submodules
----------
geometry        Dg safety-margin metric, collision/road-deviation costs.
risk            CVaR (conditional value-at-risk).
motion_set      Counterfactual motion-set synthesis via SMART rollouts.
metrics         Safety (Eq. 3) and courtesy (Eq. 4/7) responsibility.
hmm             Gaussian HMM responsibility-level abstraction (Sec IV-A).
reward_model    Per-level IRL-style reward networks (Sec IV-B).
dataset_builder Responsibility-augmented dataset construction.
selection       Online responsibility-aware trajectory selection.
pipeline        End-to-end orchestration used by scripts/responsibility/*.
"""
