"""Records: everything needed to redraw one agent's responsibility through a
clip without the model or the data -- the scene, the settings, and for every
window the beta values with the samples they were computed from.

scripts/responsibility/visualize_responsibility.py renders them; it and
compute_responsibility.py --keep-records write them.
"""

from dataclasses import asdict
from typing import Dict, List

# torch and the pipeline are imported where needed, so that rendering saved
# records needs neither

RECORD_VERSION = 1


def to_numpy(x):
    import torch

    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_numpy(v) for v in x)
    return x


def scene_snapshot(data) -> Dict:
    agent = data["agent"]
    return to_numpy({
        "position": agent["position"][..., :2],
        "heading": agent["heading"],
        "velocity": agent["velocity"],
        "valid": agent["valid_mask"],
        "shape": agent["shape"],
        "type": agent["type"],
        "id": agent["id"],
        "is_sdc": agent["role"][:, 0],
        "is_ooi": agent["role"][:, 1],
        "map_pos": data["map_save"]["traj_pos"],
        "map_type": data["pt_token"]["pl_type"],
    })


def record_frames(data, observations, details: Dict, agent: int) -> List[Dict]:
    """One frame per window in which ``agent`` was queried toward its own
    neighbours, from compute_scenario_responsibility's observations and the
    ``details`` it filled."""
    from src.responsibility.dataset_builder import SCOPE_NEIGHBOURS

    agent_id = data["agent"]["id"]
    frames = []
    for obs in observations:
        if obs.agent_idx != agent or obs.scope != SCOPE_NEIGHBOURS:
            continue
        d = to_numpy(details[(obs.k, agent)])
        frames.append({
            "k": obs.k,
            "beta": obs.beta.numpy(),
            "neighbours": d["neighbours"],
            "per_neighbour": {b: obs.per_neighbour[int(agent_id[b])] for b in d["neighbours"]},
            "interaction": {b: obs.interaction[int(agent_id[b])] for b in d["neighbours"]},
            "safety": d["safety"],
            "courtesy": d["courtesy"],
        })
    return frames


def make_record(data, frames: List[Dict], cfg, sampling: Dict, ckpt: str, scenario_id: str,
                scenario_index: int, scenario_file: str, agent: int) -> Dict:
    return {
        "version": RECORD_VERSION,
        "scenario_id": scenario_id,
        "scenario_index": scenario_index,
        "scenario_file": scenario_file,
        "agent": agent,
        "agent_id": int(data["agent"]["id"][agent]),
        "config": asdict(cfg),
        "sampling": sampling,
        "ckpt": ckpt,
        "scene": scene_snapshot(data),
        "frames": frames,
    }
