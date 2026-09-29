"""End-to-end orchestration glueing the responsibility modules together,
used by the CLI scripts in ``scripts/responsibility/``.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, read_write
from torch_geometric.loader import DataLoader

from src.responsibility.dataset_builder import (
    ResponsibilityObservation,
    assign_levels,
    compute_scenario_responsibility,
    group_sequences,
    sequences_as_arrays,
    stream_reward_training_features,
    window_steps,
)
from src.responsibility.hmm import GaussianHMM, ModelSelectionResult, fit_hmm_with_model_selection
from src.responsibility.metrics import ResponsibilityConfig
from src.responsibility.motion_set import MotionSetSynthesizer
from src.responsibility.results_io import (  # noqa: F401  (re-exported for scripts)
    atomic_pickle,
    check_run_config,
    load_hmm,
    load_observations,
    load_reward_nets,
    save_hmm,
    save_observations,
    save_reward_nets,
    scenario_files,
)
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
    mirroring the weight-loading call in ``src/run.py``'s ``finetune`` action.

    Fails loudly rather than returning a partly random model: every
    parameter must come from the checkpoint, and the decoder hyper-
    parameters it was trained with must match ``model_config`` (radii and
    step counts do not show up as shape mismatches but change behaviour).
    """
    # Lightning checkpoints carry DictConfig hyper-parameters, which the
    # weights_only default of torch>=2.6 refuses to unpickle.
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    trained = state.get("hyper_parameters", {}).get("model_config", None)
    if trained is not None and "decoder" in trained:
        want = OmegaConf.to_container(model_config.decoder, resolve=True)
        got = OmegaConf.to_container(trained["decoder"], resolve=True)
        mismatched = {k: (got.get(k), want.get(k)) for k in want if got.get(k) != want.get(k)}
        if mismatched:
            raise ValueError(
                f"{ckpt_path} was trained with a different decoder config "
                f"(checkpoint, requested): {mismatched}. Pick the matching configs/model/*.yaml."
            )

    model = SMART(model_config)
    state_dict = state["state_dict"] if "state_dict" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    param_names = {name for name, _ in model.named_parameters()}
    missing_params = [k for k in missing if k in param_names]
    if missing_params:
        raise ValueError(f"{ckpt_path} does not provide these parameters: {missing_params}")
    if unexpected:
        print(f"[load_smart_model] ignoring unexpected keys: {unexpected}")
    model.eval()
    return model.to(device)


def default_sampling_scheme(num_k: int = 2048, temp: float = 1.0) -> DictConfig:
    """``topk_prob`` with a large ``num_k`` approximates genuine sampling
    from the full model policy pi_theta (Eq. 1a), rather than the
    tightly-restricted top-k used for closed-loop training/eval rollouts."""
    return OmegaConf.create({"criterium": "topk_prob", "num_k": num_k, "temp": temp})


def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_dataloader(
    raw_dir: str,
    tfrecord_dir: Optional[str] = None,
    indices: Optional[Sequence[int]] = None,
) -> DataLoader:
    """batch_size=1 loader so every batch is exactly one scenario with PyG's
    usual ``.batch``/``.num_graphs`` collation fields attached (required by
    ``TokenProcessor``), matching how ``local_val.sh`` drives the model.
    ``indices`` restricts it to a subset of the (sorted) scenario files."""
    dataset = MultiDataset(raw_dir, WaymoTargetBuilderVal(), tfrecord_dir=tfrecord_dir)
    if indices is not None:
        dataset = dataset[list(indices)]
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def scenario_id_of(data, fallback: str = "unknown") -> str:
    if "scenario_id" not in data:
        return fallback
    raw = data["scenario_id"]
    return str(raw[0]) if isinstance(raw, (list, tuple)) else str(raw)


def count_scenarios(raw_dir: str) -> int:
    return len(MultiDataset(raw_dir, WaymoTargetBuilderVal()))


def iter_scenarios(
    raw_dir: str,
    device: str = "cpu",
    indices: Optional[Sequence[int]] = None,
) -> Iterator[Tuple[int, str, object]]:
    """Yields ``(dataset_index, scenario_id, data)`` with ``data`` already on
    ``device``. The model's buffers live on the device it was loaded to, so
    scenarios must be moved there before tokenization."""
    order = list(indices) if indices is not None else range(count_scenarios(raw_dir))
    loader = build_dataloader(raw_dir, indices=list(order) if indices is not None else None)
    for dataset_index, data in zip(order, loader):
        yield dataset_index, scenario_id_of(data, str(dataset_index)), data.to(device)


def load_scenario(raw_dir: str, index: int, device: str = "cpu"):
    for _, _, data in iter_scenarios(raw_dir, device, indices=[index]):
        return data
    raise IndexError(f"no scenario {index} in {raw_dir}")


def compute_dataset_responsibility(
    model: SMART,
    scenarios: Iterable[Tuple[int, str, object]],
    cfg: ResponsibilityConfig,
    sampling_scheme: Optional[DictConfig] = None,
    agent_type_filter: Optional[int] = 0,
    max_agents_per_scenario: Optional[int] = None,
    max_scenarios: Optional[int] = None,
    keep_data: bool = True,
    rollout_batch_size: Optional[int] = None,
    sweep_windows: bool = True,
) -> Tuple[List[ResponsibilityObservation], Dict[str, object]]:
    """Runs Sec. III's counterfactual responsibility pipeline over every
    scenario yielded by ``scenarios`` (see :func:`iter_scenarios`). Returns the flat observation list and (if
    ``keep_data``) a ``scenario_id -> data`` dict for reuse when building
    the reward-network training features later.

    With ``sweep_windows`` (the default) each clip is evaluated at several
    current steps spaced ``cfg.window_stride`` apart, so every agent yields
    a beta *sequence* and the HMM can learn transitions; otherwise only the
    clip's native current step is used."""
    synthesizer = MotionSetSynthesizer(
        model, sampling_scheme or default_sampling_scheme(), rollout_batch_size=rollout_batch_size
    )
    observations: List[ResponsibilityObservation] = []
    data_by_scenario: Dict[str, object] = {}

    for i, (_, scenario_id, data) in enumerate(scenarios):
        if max_scenarios is not None and i >= max_scenarios:
            break
        steps = (
            window_steps(data, cfg, synthesizer, cfg.window_stride)
            if sweep_windows
            else None
        )
        obs = compute_scenario_responsibility(
            synthesizer,
            data,
            cfg,
            agent_type_filter=agent_type_filter,
            max_agents=max_agents_per_scenario,
            steps=steps,
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
    scenarios: Iterable[Tuple[str, object]],
    cfg: ResponsibilityConfig,
    n_aux: int = 24,
    sampling_scheme: Optional[DictConfig] = None,
    hidden_dim: int = 32,
    n_epochs: int = 300,
    lr: float = 1e-3,
    rollout_batch_size: Optional[int] = None,
) -> Dict[int, RewardNetBundle]:
    """``scenarios`` yields ``(scenario_id, data)`` and is consumed one
    scenario at a time (see ``stream_reward_training_features``)."""
    synthesizer = MotionSetSynthesizer(
        model, sampling_scheme or default_sampling_scheme(), rollout_batch_size=rollout_batch_size
    )
    level_features = stream_reward_training_features(
        synthesizer,
        scenarios,
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


# ---------------------------------------------------------------------------
# Resumable, shardable stage 1
# ---------------------------------------------------------------------------


def run_responsibility_to_dir(
    model: SMART,
    data_root: str,
    out_dir: str,
    cfg: ResponsibilityConfig,
    run_config: Dict,
    device: str = "cpu",
    sampling_scheme: Optional[DictConfig] = None,
    agent_type_filter: Optional[int] = 0,
    max_agents_per_scenario: Optional[int] = None,
    max_scenarios: Optional[int] = None,
    sweep_windows: bool = True,
    rollout_batch_size: Optional[int] = None,
    num_shards: int = 1,
    shard_index: int = 0,
    allow_config_mismatch: bool = False,
    log=None,
) -> Dict[str, int]:
    """Stage 1 over a dataset, one result file per scenario under
    ``out_dir/obs/``. Scenarios that already have a result are skipped, so an
    interrupted run resumes where it stopped, and ``num_shards`` processes
    with distinct ``shard_index`` can fill the same directory in parallel."""
    import time

    if log is None:  # unbuffered: this is usually redirected to a log file
        def log(message):
            print(message, flush=True)

    out = Path(out_dir)
    (out / "obs").mkdir(parents=True, exist_ok=True)
    # settings added after a directory was started still match at their default
    defaults = {"responsibility": asdict(ResponsibilityConfig(num_historical_steps=cfg.num_historical_steps))}
    check_run_config(out, run_config, allow_config_mismatch, defaults=defaults)

    files = scenario_files(data_root)
    mine = [i for i in range(len(files)) if i % num_shards == shard_index]
    if max_scenarios is not None:
        mine = mine[:max_scenarios]
    todo = [i for i in mine if not (out / "obs" / f"{files[i].stem}.pkl").exists()]
    log(f"[shard {shard_index}/{num_shards}] {len(mine)} scenarios assigned, "
        f"{len(mine) - len(todo)} already done, {len(todo)} to go")

    synthesizer = MotionSetSynthesizer(
        model, sampling_scheme or default_sampling_scheme(), rollout_batch_size=rollout_batch_size
    )
    totals = {"scenarios": 0, "observations": 0, "rollouts": 0, "windows": 0}
    start = time.perf_counter()
    for n_done, (index, scenario_id, data) in enumerate(iter_scenarios(data_root, device, indices=todo), 1):
        t0 = time.perf_counter()
        stats: Dict[str, int] = {}
        steps = window_steps(data, cfg, synthesizer, cfg.window_stride) if sweep_windows else None
        obs = compute_scenario_responsibility(
            synthesizer, data, cfg, agent_type_filter=agent_type_filter,
            max_agents=max_agents_per_scenario, steps=steps, stats=stats,
        )
        for o in obs:
            o.scenario_id = scenario_id
        atomic_pickle(obs, out / "obs" / f"{files[index].stem}.pkl")

        dt = time.perf_counter() - t0
        totals["scenarios"] += 1
        totals["observations"] += len(obs)
        totals["rollouts"] += stats.get("rollouts", 0)
        totals["windows"] += stats.get("windows", 0)
        elapsed = time.perf_counter() - start
        eta = elapsed / n_done * (len(todo) - n_done)
        log(f"[shard {shard_index}] {n_done}/{len(todo)} {scenario_id}: {len(obs)} obs, "
            f"{stats.get('rollouts', 0)} rollouts in {dt:.1f} s "
            f"({stats.get('rollouts', 0) / max(dt, 1e-9):.0f}/s); eta {eta / 3600:.1f} h")
    return totals
