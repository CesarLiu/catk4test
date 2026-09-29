"""Reading and writing responsibility results. Kept free of the SMART /
torch_geometric imports so it can be used (and tested) anywhere."""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List

import torch


def scenario_files(raw_dir: str) -> List[Path]:
    """The cached scenario files in dataset-index order (MultiDataset sorts
    them the same way); each is named after its scenario id."""
    return [Path(p) for p in sorted(Path(raw_dir).glob("*"))]


def atomic_pickle(obj, path: Path) -> None:
    """Writes via a temporary file and a rename, so a killed process never
    leaves a truncated result that a restart would mistake for finished."""
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


_MISSING = object()


def config_differences(old, new, prefix: str = "", defaults=None) -> List[str]:
    """Leaf-level differences between two JSON-like configurations. A
    setting absent from ``old`` (written before the setting existed) does
    not count when ``new`` holds its default from ``defaults``: the default
    reproduces the behaviour from before it was added."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key in sorted(set(old) | set(new)):
            sub_defaults = defaults.get(key, _MISSING) if isinstance(defaults, dict) else _MISSING
            out += config_differences(old.get(key, _MISSING), new.get(key, _MISSING), f"{prefix}{key}.", sub_defaults)
        return out
    if old is _MISSING and defaults is not _MISSING and new == defaults:
        return []
    if old == new:
        return []
    shown = lambda v: "(absent)" if v is _MISSING else repr(v)  # noqa: E731
    return [f"{prefix[:-1]}: {shown(old)} -> {shown(new)}"]


def _encode(config):
    return json.loads(json.dumps(config, default=str))


def check_run_config(out_dir: Path, run_config: Dict, allow_mismatch: bool = False, defaults: Dict = None) -> None:
    """Records the configuration of a results directory on first use and
    refuses to add results computed under a different one. ``defaults``
    (same nesting as ``run_config``) lets directories written before a
    setting existed accept runs that leave it at its default."""
    path = Path(out_dir) / "config.json"
    encoded = _encode(run_config)
    if path.exists():
        existing = json.loads(path.read_text())
        changed = config_differences(existing, encoded, defaults=_encode(defaults) if defaults else _MISSING)
        if changed and not allow_mismatch:
            raise ValueError(
                f"{out_dir} holds results computed with a different configuration:\n  "
                + "\n  ".join(changed)
                + "\nUse a new --out-dir, or --allow-config-mismatch."
            )
        return
    tmp = path.with_name(f"config.json.tmp{os.getpid()}")
    tmp.write_text(json.dumps(encoded, indent=2, sort_keys=True))
    os.replace(tmp, path)


def save_observations(observations, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(observations, f)


def load_observations(path: str) -> list:
    """Loads observations from a single .pkl, or from a results directory
    written by ``run_responsibility_to_dir`` (every ``obs/*.pkl``)."""
    p = Path(path)
    if p.is_dir():
        folder = p / "obs" if (p / "obs").is_dir() else p
        observations = []
        for f in sorted(folder.glob("*.pkl")):
            with open(f, "rb") as handle:
                observations.extend(pickle.load(handle))
        return observations
    with open(p, "rb") as f:
        return pickle.load(f)


def save_hmm(hmm, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(hmm, f)


def load_hmm(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_reward_nets(bundles: Dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({level: b.state_dict() for level, b in bundles.items()}, path)


def load_reward_nets(path: str, hidden_dim: int = 32) -> Dict:
    from src.responsibility.reward_model import RewardNetBundle

    states = torch.load(path, map_location="cpu", weights_only=False)
    return {level: RewardNetBundle.load_state_dict(state, hidden_dim=hidden_dim) for level, state in states.items()}
