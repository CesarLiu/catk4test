"""End-to-end orchestration glueing the responsibility modules together,
used by the CLI scripts in ``scripts/responsibility/``.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, read_write
from torch_geometric.loader import DataLoader

from src.responsibility.dataset_builder import (
    ResponsibilityObservation,
    assign_levels,
    build_reward_training_features,
    compute_scenario_responsibility,
    group_sequences,
    sequences_as_arrays,
)
from src.responsibility.hmm import GaussianHMM, ModelSelectionResult, fit_hmm_with_model_selection
from src.responsibility.metrics import ResponsibilityConfig
from src.responsibility.motion_set import MotionSetSynthesizer
from src.responsibility.reward_model import RewardNetBundle, train_reward_networks_per_level
from src.smart.datamodules.target_builder import WaymoTargetBuilderVal
from src.smart.datasets.scalable_dataset import MultiDataset
from src.smart.model.smart import SMART


def compose_responsibility_config(overrides: Optional[List[str]] = None) -> DictConfig:
    """Composes this repo's Hydra config tree (model, data, paths, trainer,
    hydra groups -- skipping callbacks/logger/experiment, which loading a
    checkpoint for inference does not need) via
    ``configs/responsibility/eval.yaml``, and activates the global
    ``HydraConfig`` singleton so ``SMART.__init__`` (which reads
    ``HydraConfig.get().runtime.output_dir``) works outside of a real
    ``hydra.main``-launched run.

    A real Hydra job resolves ``hydra.runtime.output_dir`` from
    ``hydra.run.dir`` (which itself depends on ``${hydra:runtime.cwd}``,
    i.e. on ``HydraConfig`` already being set) as part of launching that
    job; a bare ``hydra.compose(..., return_hydra_config=True)`` leaves
    ``runtime.output_dir`` as an unresolved mandatory value. We reproduce
    that bootstrap in two steps: set the (still output-dir-less) config
    first so ``run.dir``'s interpolations can resolve, then fill in and
    re-set ``runtime.output_dir``.
    """
    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        cfg = hydra.compose(
            config_name="responsibility/eval",
            overrides=overrides or [],
            return_hydra_config=True,
        )
    HydraConfig.instance().set_config(cfg)
    resolved_output_dir = str(cfg.hydra.run.dir)
    with read_write(cfg.hydra.runtime):
        cfg.hydra.runtime.output_dir = resolved_output_dir
    HydraConfig.instance().set_config(cfg)
    return cfg


def load_smart_model(model_config: DictConfig, ckpt_path: str, device: str = "cpu") -> SMART:
    """Loads a SMART checkpoint for inference-only use (no Trainer needed),
    mirroring the weight-loading call in ``src/run.py``'s ``finetune`` action."""
    model = SMART(model_config)
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state["state_dict"] if "state_dict" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        print(f"[load_smart_model] missing keys: {missing}")
    if len(unexpected) > 0:
        print(f"[load_smart_model] unexpected keys: {unexpected}")
    model.eval()
    return model.to(device)


def default_sampling_scheme(num_k: int = 2048, temp: float = 1.0) -> DictConfig:
    """``topk_prob`` with a large ``num_k`` approximates genuine sampling
    from the full model policy pi_theta (Eq. 1a), rather than the
    tightly-restricted top-k used for closed-loop training/eval rollouts."""
    return OmegaConf.create({"criterium": "topk_prob", "num_k": num_k, "temp": temp})


def build_dataloader(raw_dir: str, tfrecord_dir: Optional[str] = None) -> DataLoader:
    """batch_size=1 loader so every batch is exactly one scenario with PyG's
    usual ``.batch``/``.num_graphs`` collation fields attached (required by
    ``TokenProcessor``), matching how ``local_val.sh`` drives the model."""
    dataset = MultiDataset(raw_dir, WaymoTargetBuilderVal(), tfrecord_dir=tfrecord_dir)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def compute_dataset_responsibility(
    model: SMART,
    dataloader: DataLoader,
    cfg: ResponsibilityConfig,
    sampling_scheme: Optional[DictConfig] = None,
    agent_type_filter: Optional[int] = 0,
    max_agents_per_scenario: Optional[int] = None,
    max_scenarios: Optional[int] = None,
    keep_data: bool = True,
) -> Tuple[List[ResponsibilityObservation], Dict[str, object]]:
    """Runs Sec. III's counterfactual responsibility pipeline over every
    scenario in ``dataloader``. Returns the flat observation list and (if
    ``keep_data``) a ``scenario_id -> data`` dict for reuse when building
    the reward-network training features later."""
    synthesizer = MotionSetSynthesizer(model, sampling_scheme or default_sampling_scheme())
    observations: List[ResponsibilityObservation] = []
    data_by_scenario: Dict[str, object] = {}

    for i, data in enumerate(dataloader):
        if max_scenarios is not None and i >= max_scenarios:
            break
        scenario_id = str(data["scenario_id"][0]) if "scenario_id" in data else str(i)
        obs = compute_scenario_responsibility(
            synthesizer,
            data,
            cfg,
            agent_type_filter=agent_type_filter,
            max_agents=max_agents_per_scenario,
        )
        for o in obs:
            o.scenario_id = scenario_id
        observations.extend(obs)
        if keep_data:
            data_by_scenario[scenario_id] = data

    return observations, data_by_scenario


def fit_hmm_stage(
    observations: List[ResponsibilityObservation],
    n_states_grid=(3, 5, 7, 9, 11),
    n_iter: int = 100,
    seed: int = 0,
    verbose: bool = False,
) -> ModelSelectionResult:
    grouped = group_sequences(observations)
    sequences = sequences_as_arrays(grouped)
    return fit_hmm_with_model_selection(
        sequences, n_states_grid=n_states_grid, n_features=2, n_iter=n_iter, seed=seed, verbose=verbose
    )


def label_observations_with_levels(
    observations: List[ResponsibilityObservation], hmm: GaussianHMM
) -> None:
    grouped = group_sequences(observations)
    assign_levels(grouped, hmm)


def train_reward_stage(
    model: SMART,
    observations: List[ResponsibilityObservation],
    data_by_scenario: Dict[str, object],
    cfg: ResponsibilityConfig,
    n_aux: int = 24,
    sampling_scheme: Optional[DictConfig] = None,
    hidden_dim: int = 32,
    n_epochs: int = 300,
    lr: float = 1e-3,
) -> Dict[int, RewardNetBundle]:
    synthesizer = MotionSetSynthesizer(model, sampling_scheme or default_sampling_scheme())
    level_features = build_reward_training_features(
        synthesizer,
        data_by_scenario,
        observations,
        cfg,
        n_aux=n_aux,
        sampling_scheme=sampling_scheme,
    )
    features_by_level = {
        level: (fs.real, fs.fake) for level, fs in level_features.items()
    }
    return train_reward_networks_per_level(
        features_by_level, hidden_dim=hidden_dim, n_epochs=n_epochs, lr=lr
    )


def save_observations(observations: List[ResponsibilityObservation], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(observations, f)


def load_observations(path: str) -> List[ResponsibilityObservation]:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_hmm(hmm: GaussianHMM, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(hmm, f)


def load_hmm(path: str) -> GaussianHMM:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_reward_nets(bundles: Dict[int, RewardNetBundle], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({level: b.state_dict() for level, b in bundles.items()}, path)


def load_reward_nets(path: str, hidden_dim: int = 32) -> Dict[int, RewardNetBundle]:
    states = torch.load(path, map_location="cpu")
    return {level: RewardNetBundle.load_state_dict(state, hidden_dim=hidden_dim) for level, state in states.items()}
