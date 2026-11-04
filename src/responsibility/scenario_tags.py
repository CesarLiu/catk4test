"""Rule-based tags for an interacting pair of agents in a cached WOMD
scenario: what kind of interaction it is, what kind of place it happens in,
and when it happens.

Everything is read from the log and the map of the cached scenario (the
whole 9.1 s clip, future included) -- no model calls. It is meant for
slicing responsibility results offline ("which levels show up at
unprotected left turns?"), not for anything causal.

Interaction type (``classify_interaction``), from the two logged paths:

  following               same lane, same direction, one behind the other
  lane_change             paths converge at a small angle, neither turns
                          (one moves over into the other's lane)
  merge                   paths converge at a small angle, one of them turns
                          into the other's lane (on-ramp, turning onto a road)
  crossing                paths cross at an angle, the two started out
                          neither head-on nor side by side
  left_turn_across_path   paths cross; the two approached from opposite
                          directions (unprotected left turn against oncoming)
  turn_across_same_dir    paths cross; the two started in the same direction
                          and one turns across the other (e.g. a right hook)
  head_on                 the two occupy the same place heading towards each
                          other
  adjacent_lane           never on the same path, but side by side
  oncoming_pass           never on the same path, but pass close, opposite
                          directions
  proximity               never on the same path, but close at another angle
  none                    none of the above

With a pedestrian (type 1) or a cyclist (type 2) in the pair, the
pair's ``category`` says so (``vehicle_pedestrian`` etc.). The type is the
same geometric rule, except that a pedestrian walking across a vehicle's
path is ``pedestrian_crossing`` and one walking along it is ``pedestrian_along``.

Scene type (``classify_scene``), from the map within ``scene_radius`` of
the conflict point (or of the pair's closest approach):

  signalized_intersection / stop_intersection / uncontrolled_intersection
      vehicle-lane centerlines cross each other at an angle there; the
      control is read from the lanes' traffic-light state at the current
      step and from stop-sign lanes
  highway        mostly freeway lanes, no crossing lanes
  urban_road     surface-street lanes, no crossing lanes
  no_lanes       no vehicle lane nearby (parking lots, driveways)

plus flags (signal, stop sign, crosswalk/speed bump nearby, freeway share,
number of crossing lane pairs). The cached map keeps no lane connectivity,
so intersections are found geometrically, and roundabouts are not told
apart from other intersections.

Interaction window (``interaction_window``): the steps at which the pair is
interacting, i.e. both are logged and at least one of

  - their footprints are within ``proximity_gap`` of each other,
  - one of them is at a place the other occupies within ``time_gap``
    seconds (before or after) -- a local post-encroachment time, which is
    the time headway when following and the PET around a crossing,
  - both will reach their conflict point within ``approach_time`` seconds,

holds. The longest such run (gaps up to ``merge_gap`` steps are bridged) is
the window; its peak is the step with the smallest local PET.
"""

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from src.responsibility.geometry import circle_centers, circle_decomposition

DT = 0.1  # seconds per 10Hz step

VEHICLE, PEDESTRIAN, CYCLIST = 0, 1, 2
TYPE_NAMES = {VEHICLE: "vehicle", PEDESTRIAN: "pedestrian", CYCLIST: "cyclist"}

# cached map (src/smart/utils/preprocess.py::preprocess_map): polygon types
PL_LANE, PL_CROSSWALK = 0, 3
# point types (src/data_preprocess.py): 0 freeway, 1 surface street, 2 stop-sign lane
PT_FREEWAY, PT_SURFACE, PT_STOP = 0, 1, 2
VEHICLE_LANE_TYPES = (PT_FREEWAY, PT_SURFACE, PT_STOP)
# light types: 0 NO_LANE_STATE, 1 UNKNOWN, 2 STOP, 3 GO, 4 CAUTION
LIGHT_NONE = 0

INTERACTION_TYPES = (
    "following", "lane_change", "merge", "crossing", "left_turn_across_path",
    "turn_across_same_dir", "head_on", "adjacent_lane", "oncoming_pass", "proximity",
    "pedestrian_crossing", "pedestrian_along", "none",
)
SCENE_TYPES = (
    "signalized_intersection", "stop_intersection", "uncontrolled_intersection",
    "highway", "urban_road", "no_lanes",
)


@dataclass
class TagConfig:
    path_tolerance: float = 2.0  # m: two path points are "the same place"
    place_slack: float = 0.3  # m: points this close to the nearest one count as nearest too
    same_dir_angle: float = 30.0  # deg: below this two headings are "the same direction"
    opposite_angle: float = 135.0  # deg: above this two agents approach head-on
    same_lane_offset: float = 1.5  # m: lateral offset of two cars in one lane
    side_by_side_gap: float = 3.0  # m: footprint gap of a close pass without a shared path
    turn_angle: float = 20.0  # deg of heading change that makes a maneuver a turn
    u_turn_angle: float = 150.0
    stationary_distance: float = 2.0  # m travelled over the clip
    lane_change_offset: float = 2.5  # m of lateral drift without turning
    proximity_gap: float = 5.0  # m, window criterion
    time_gap: float = 3.0  # s, window criterion (local PET / time headway)
    approach_time: float = 3.0  # s before the later agent reaches the conflict point
    merge_gap: int = 5  # steps of inactivity bridged inside one window
    yield_slowdown: float = 0.3  # relative speed drop that counts as yielding
    scene_radius: float = 30.0  # m around the interaction point
    crossing_lane_angle: float = 25.0  # deg between two lane centerlines that cross
    min_lane_crossings: int = 2  # crossing lane-segment pairs that make an intersection
    freeway_share: float = 0.5


# ---------------------------------------------------------------------------
# per-agent motion
# ---------------------------------------------------------------------------


def wrap_angle(x: Tensor) -> Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi


def _valid_track(data, i: int) -> Tuple[Tensor, Tensor, Tensor]:
    agent = data["agent"]
    steps = agent["valid_mask"][i].nonzero(as_tuple=True)[0]
    return steps, agent["position"][i, steps, :2], agent["heading"][i, steps]


def agent_maneuver(data, i: int, cfg: TagConfig) -> Dict:
    """stationary / straight / lane_change_left|right / left / right / u_turn
    over the agent's logged steps, with the total heading change (deg,
    positive = left) and the distance travelled (m)."""
    steps, pos, head = _valid_track(data, i)
    if steps.numel() < 2:
        return {"maneuver": "stationary", "heading_change": 0.0, "distance": 0.0}
    distance = float(torch.linalg.norm(pos[1:] - pos[:-1], dim=-1).sum())
    turn = float(torch.rad2deg(wrap_angle(head[1:] - head[:-1]).sum()))
    if distance < cfg.stationary_distance:
        maneuver = "stationary"
    elif abs(turn) >= cfg.u_turn_angle:
        maneuver = "u_turn"
    elif abs(turn) >= cfg.turn_angle:
        maneuver = "left" if turn > 0 else "right"
    else:
        # lateral drift in the frame of the initial heading
        h0 = head[0]
        rel = pos[-1] - pos[0]
        lateral = float(-math.sin(h0) * rel[0] + math.cos(h0) * rel[1])
        if abs(lateral) >= cfg.lane_change_offset:
            maneuver = "lane_change_left" if lateral > 0 else "lane_change_right"
        else:
            maneuver = "straight"
    return {"maneuver": maneuver, "heading_change": turn, "distance": distance}


# ---------------------------------------------------------------------------
# pair geometry
# ---------------------------------------------------------------------------


def _angle_deg(h1: Tensor, h2: Tensor) -> float:
    return float(torch.rad2deg(wrap_angle(h1 - h2).abs()))


def footprint_gap_series(data, a: int, b: int) -> Tensor:  # [T], inf where either is unlogged
    agent = data["agent"]
    pos, head, shape = agent["position"][[a, b], :, :2], agent["heading"][[a, b]], agent["shape"][[a, b]]
    off, r = circle_decomposition(shape[:, 0], shape[:, 1], 3)
    centres = circle_centers(pos, head, off)  # [2, T, C, 2]
    d = torch.linalg.norm(centres[0][:, :, None, :] - centres[1][:, None, :, :], dim=-1)
    gap = (d.amin(dim=(-2, -1)) - r[0] - r[1]).clamp(min=0.0)
    both = agent["valid_mask"][a] & agent["valid_mask"][b]
    return gap.masked_fill(~both, float("inf"))


def _place_matches(data, a: int, b: int, cfg: TagConfig) -> Tensor:
    """[T_a, T_b] bool: b at step s is at the place a occupies at step t.

    Taken from a's side: for each t, the steps s at which b comes closest to
    a's position (within ``place_slack`` of that closest distance), as long
    as that is within ``path_tolerance``. Matching the *closest* point, not
    any point within the tolerance, keeps a following car's headway and a
    crossing's PET from being shortened by tolerance / speed."""
    agent = data["agent"]
    pos = agent["position"][:, :, :2]
    va, vb = agent["valid_mask"][a], agent["valid_mask"][b]
    d = torch.cdist(pos[a], pos[b]).masked_fill(~(va[:, None] & vb[None, :]), float("inf"))
    nearest, at = d.min(dim=1, keepdim=True)
    # A nearest point at either end of b's log only says that a is beyond
    # where b was logged (e.g. behind a car whose log starts ahead of it),
    # unless a is right on top of it.
    logged = vb.nonzero(as_tuple=True)[0]
    tol = torch.full_like(nearest, cfg.path_tolerance)
    if logged.numel():
        at_end = (at == logged[0]) | (at == logged[-1])
        tol = torch.where(at_end, torch.full_like(tol, cfg.place_slack), tol)
    return (d <= nearest + cfg.place_slack) & (nearest < tol)


def conflict_point(data, a: int, b: int, cfg: TagConfig) -> Optional[Dict]:
    """Where the two logged paths meet (within ``path_tolerance``), choosing
    the meeting with the smallest time difference: the steps t_a, t_b at
    which each was there, their headings there and the point itself. None
    if the paths never meet."""
    agent = data["agent"]
    pos = agent["position"][:, :, :2]
    match = _place_matches(data, a, b, cfg) | _place_matches(data, b, a, cfg).T
    if not bool(match.any()):
        return None
    steps = torch.arange(pos.shape[1], device=pos.device)
    dt = (steps[:, None] - steps[None, :]).abs().float()
    dt = dt.masked_fill(~match, float("inf"))
    flat = int(dt.argmin())
    ta, tb = flat // dt.shape[1], flat % dt.shape[1]
    return {
        "t_a": ta,
        "t_b": tb,
        "pet": abs(ta - tb) * DT,
        "point": ((pos[a, ta] + pos[b, tb]) / 2),
        "angle": _angle_deg(agent["heading"][a, ta], agent["heading"][b, tb]),
    }


def local_pet_series(data, a: int, b: int, cfg: TagConfig) -> Tensor:  # [T] seconds
    """At each step t, the smallest |t - s| such that one agent at t is
    where the other is at s: the time headway when one follows the other,
    the PET around a crossing; inf where neither is on the other's path."""
    agent = data["agent"]
    n_step = agent["position"].shape[1]
    va, vb = agent["valid_mask"][a], agent["valid_mask"][b]
    steps = torch.arange(n_step, device=va.device)
    dt = (steps[:, None] - steps[None, :]).abs().float() * DT
    inf = torch.tensor(float("inf"), device=va.device)
    a_on_b = torch.where(_place_matches(data, a, b, cfg), dt, inf).amin(dim=1)  # a at t, b at some s
    b_on_a = torch.where(_place_matches(data, b, a, cfg), dt, inf).amin(dim=1)  # b at t, a at some s
    out = torch.minimum(a_on_b, b_on_a)
    return out.masked_fill(~(va & vb), float("inf"))


def _lateral_offset(data, a: int, b: int, t: int) -> Tuple[float, float]:
    """b's (longitudinal, lateral) offset in a's frame at step t."""
    agent = data["agent"]
    h = agent["heading"][a, t]
    rel = agent["position"][b, t, :2] - agent["position"][a, t, :2]
    lon = float(math.cos(h) * rel[0] + math.sin(h) * rel[1])
    lat = float(-math.sin(h) * rel[0] + math.cos(h) * rel[1])
    return lon, lat


def _first_common_step(data, a: int, b: int) -> Optional[int]:
    both = data["agent"]["valid_mask"][a] & data["agent"]["valid_mask"][b]
    idx = both.nonzero(as_tuple=True)[0]
    return int(idx[0]) if idx.numel() else None


# ---------------------------------------------------------------------------
# interaction type
# ---------------------------------------------------------------------------


def pair_category(data, a: int, b: int) -> str:
    types = sorted(TYPE_NAMES.get(int(data["agent"]["type"][i]), "other") for i in (a, b))
    order = ["vehicle", "cyclist", "pedestrian", "other"]
    types.sort(key=order.index)
    return f"{types[0]}_{types[1]}"


def classify_interaction(data, a: int, b: int, cfg: Optional[TagConfig] = None) -> Dict:
    """Interaction type of the pair (see the module docstring), with the
    evidence it rests on: both maneuvers, the conflict point, initial and
    conflict relative headings, who passes the conflict point first and
    whether the other one yielded."""
    cfg = cfg or TagConfig()
    agent = data["agent"]
    man_a, man_b = agent_maneuver(data, a, cfg), agent_maneuver(data, b, cfg)
    t0 = _first_common_step(data, a, b)
    gap = footprint_gap_series(data, a, b)
    out = {
        "category": pair_category(data, a, b),
        "maneuver_a": man_a["maneuver"],
        "maneuver_b": man_b["maneuver"],
        "heading_change_a": man_a["heading_change"],
        "heading_change_b": man_b["heading_change"],
        "min_gap": float(gap.min()) if torch.isfinite(gap).any() else float("inf"),
        "min_gap_step": int(gap.argmin()) if torch.isfinite(gap).any() else -1,
        "initial_angle": float("nan"),
        "conflict_angle": float("nan"),
        "pet": float("inf"),
        "conflict_step_a": -1,
        "conflict_step_b": -1,
        "first_through": "",
        "yielder": "",
        "interaction_type": "none",
    }
    if t0 is None:
        return out
    out["initial_angle"] = _angle_deg(agent["heading"][a, t0], agent["heading"][b, t0])
    conflict = conflict_point(data, a, b, cfg)
    is_ped = int(agent["type"][a]) == PEDESTRIAN or int(agent["type"][b]) == PEDESTRIAN

    if conflict is None:
        if out["min_gap"] <= cfg.side_by_side_gap:
            t = out["min_gap_step"]
            ang = _angle_deg(agent["heading"][a, t], agent["heading"][b, t])
            _, lat = _lateral_offset(data, a, b, t)
            if ang <= cfg.same_dir_angle and abs(lat) <= cfg.same_lane_offset and not is_ped:
                # one right behind the other without ever reaching its spot,
                # e.g. a queue behind a stopped car
                out["interaction_type"] = "following"
            elif ang <= cfg.same_dir_angle:
                out["interaction_type"] = "adjacent_lane"
            elif ang >= cfg.opposite_angle:
                out["interaction_type"] = "oncoming_pass"
            else:
                out["interaction_type"] = "proximity"
        return out

    out.update(
        conflict_angle=conflict["angle"], pet=conflict["pet"],
        conflict_step_a=conflict["t_a"], conflict_step_b=conflict["t_b"],
    )
    if conflict["t_a"] != conflict["t_b"]:
        first, later = ("a", b) if conflict["t_a"] < conflict["t_b"] else ("b", a)
        out["first_through"] = first
        if _yielded(data, later, max(conflict["t_a"], conflict["t_b"]), t0, cfg):
            out["yielder"] = "b" if first == "a" else "a"

    same_dir = conflict["angle"] <= cfg.same_dir_angle
    if is_ped:
        out["interaction_type"] = "pedestrian_along" if same_dir else "pedestrian_crossing"
        return out

    turning = {"left", "right", "u_turn"}
    any_turn = man_a["maneuver"] in turning or man_b["maneuver"] in turning
    if same_dir:
        _, lat0 = _lateral_offset(data, a, b, t0)
        if out["initial_angle"] <= cfg.same_dir_angle and abs(lat0) <= cfg.same_lane_offset:
            out["interaction_type"] = "following"
        elif any_turn:
            out["interaction_type"] = "merge"
        else:
            out["interaction_type"] = "lane_change"
    elif conflict["angle"] >= 180.0 - cfg.same_dir_angle:
        out["interaction_type"] = "head_on"
    elif out["initial_angle"] >= cfg.opposite_angle:
        out["interaction_type"] = "left_turn_across_path"
    elif out["initial_angle"] <= cfg.same_dir_angle and any_turn:
        out["interaction_type"] = "turn_across_same_dir"
    else:
        out["interaction_type"] = "crossing"
    return out


def _yielded(data, i: int, conflict_step: int, t0: int, cfg: TagConfig) -> bool:
    """Whether agent i slowed down markedly (or stopped) before reaching the
    conflict point after the other one had passed it."""
    agent = data["agent"]
    lo = max(t0, conflict_step - int(round(cfg.approach_time / DT)) * 2)
    seg = slice(lo, conflict_step + 1)
    valid = agent["valid_mask"][i, seg]
    if int(valid.sum()) < 3:
        return False
    speed = torch.linalg.norm(agent["velocity"][i, seg, :2], dim=-1)[valid]
    start = float(speed[0])
    return start > 1.0 and float(speed.min()) <= (1.0 - cfg.yield_slowdown) * start


# ---------------------------------------------------------------------------
# interaction window
# ---------------------------------------------------------------------------


def _runs(mask: Tensor, bridge: int) -> List[Tuple[int, int]]:
    idx = mask.nonzero(as_tuple=True)[0].tolist()
    runs: List[List[int]] = []
    for t in idx:
        if runs and t - runs[-1][1] <= bridge + 1:
            runs[-1][1] = t
        else:
            runs.append([t, t])
    return [(s, e) for s, e in runs]


def interaction_window(data, a: int, b: int, cfg: Optional[TagConfig] = None,
                       conflict: Optional[Dict] = None) -> Dict:
    """Start, end and peak of the pair's interaction, as clip steps and
    seconds from the clip start (see the module docstring for the rule).
    Start/end are -1 when the pair never interacts."""
    cfg = cfg or TagConfig()
    if conflict is None:
        conflict = conflict_point(data, a, b, cfg)
    gap = footprint_gap_series(data, a, b)
    lpet = local_pet_series(data, a, b, cfg)
    active = (gap <= cfg.proximity_gap) | (lpet <= cfg.time_gap)
    if conflict is not None:
        later = max(conflict["t_a"], conflict["t_b"])
        if conflict["pet"] <= cfg.time_gap:
            lo = max(0, later - int(round(cfg.approach_time / DT)))
            approach = torch.zeros_like(active)
            approach[lo : later + 1] = True
            active |= approach & torch.isfinite(gap)
    runs = _runs(active, cfg.merge_gap)
    if not runs:
        return {"start_step": -1, "end_step": -1, "peak_step": -1,
                "start_s": float("nan"), "end_s": float("nan"), "peak_s": float("nan"), "duration_s": 0.0,
                "n_windows": 0}
    start, end = max(runs, key=lambda r: (r[1] - r[0], -r[0]))
    span = slice(start, end + 1)
    # peak: smallest local PET inside the window, ties (and no PET) by smallest gap
    key = torch.where(torch.isfinite(lpet[span]), lpet[span], torch.full_like(lpet[span], 1e6)) * 1e3 + gap[span].clamp(max=1e3)
    peak = start + int(key.argmin())
    return {
        "start_step": start, "end_step": end, "peak_step": peak,
        "start_s": start * DT, "end_s": end * DT, "peak_s": peak * DT,
        "duration_s": (end - start) * DT, "n_windows": len(runs),
    }


# ---------------------------------------------------------------------------
# scene type
# ---------------------------------------------------------------------------


def _segments_cross(p1: Tensor, p2: Tensor, q1: Tensor, q2: Tensor) -> Tensor:
    """[N, M] whether segment p1p2 (N of them) properly intersects q1q2 (M)."""
    def cross(o, u, v):  # z of (u - o) x (v - o), broadcast
        return (u[..., 0] - o[..., 0]) * (v[..., 1] - o[..., 1]) - (u[..., 1] - o[..., 1]) * (v[..., 0] - o[..., 0])

    P1, P2 = p1[:, None], p2[:, None]
    Q1, Q2 = q1[None, :], q2[None, :]
    d1, d2 = cross(Q1, Q2, P1), cross(Q1, Q2, P2)
    d3, d4 = cross(P1, P2, Q1), cross(P1, P2, Q2)
    # touching counts: lane segments often end exactly on another lane
    return (d1 * d2 <= 0) & (d3 * d4 <= 0)


def classify_scene(data, center: Tensor, cfg: Optional[TagConfig] = None) -> Dict:
    """Scene type around ``center`` [2] (see the module docstring)."""
    cfg = cfg or TagConfig()
    out = {"scene_type": "no_lanes", "n_lane_crossings": 0, "signalized": False, "stop_sign": False,
           "crosswalk_nearby": False, "freeway_share": 0.0, "n_lane_segments": 0}
    try:
        traj = data["map_save"]["traj_pos"]  # [n_pl, 3, 2]
        pt_type = data["pt_token"]["type"].long()
        pl_type = data["pt_token"]["pl_type"].long()
        light = data["pt_token"]["light_type"].long()
    except (KeyError, IndexError):
        return out
    near = (torch.linalg.norm(traj - center[None, None, :], dim=-1) <= cfg.scene_radius).any(-1)
    lane = near & (pl_type == PL_LANE)
    vlane = torch.zeros_like(lane)
    for t in VEHICLE_LANE_TYPES:
        vlane |= lane & (pt_type == t)
    out["crosswalk_nearby"] = bool((near & (pl_type == PL_CROSSWALK)).any())
    out["signalized"] = bool((vlane & (light != LIGHT_NONE)).any())
    out["stop_sign"] = bool((vlane & (pt_type == PT_STOP)).any())
    n_lane = int(vlane.sum())
    out["n_lane_segments"] = n_lane
    if n_lane == 0:
        return out
    out["freeway_share"] = float((vlane & (pt_type == PT_FREEWAY)).sum()) / n_lane

    seg = traj[vlane]
    p1, p2 = seg[:, 0], seg[:, -1]
    direction = torch.atan2(p2[:, 1] - p1[:, 1], p2[:, 0] - p1[:, 0])
    crosses = _segments_cross(p1, p2, p1, p2)
    angle = torch.rad2deg(wrap_angle(direction[:, None] - direction[None, :]).abs())
    crosses &= angle >= cfg.crossing_lane_angle
    n_cross = int(torch.triu(crosses, diagonal=1).sum())
    out["n_lane_crossings"] = n_cross

    if n_cross >= cfg.min_lane_crossings:
        if out["signalized"]:
            out["scene_type"] = "signalized_intersection"
        elif out["stop_sign"]:
            out["scene_type"] = "stop_intersection"
        else:
            out["scene_type"] = "uncontrolled_intersection"
    elif out["freeway_share"] >= cfg.freeway_share:
        out["scene_type"] = "highway"
    else:
        out["scene_type"] = "urban_road"
    return out


# ---------------------------------------------------------------------------
# pairs and the full tag
# ---------------------------------------------------------------------------


@dataclass
class PairTag:
    scenario_id: str
    agent_a: int  # index in the cached scenario
    agent_b: int
    id_a: int  # WOMD object id
    id_b: int
    pair_source: str
    category: str
    interaction_type: str
    scene_type: str
    start_s: float
    end_s: float
    peak_s: float
    duration_s: float
    start_step: int
    end_step: int
    peak_step: int
    maneuver_a: str
    maneuver_b: str
    initial_angle: float
    conflict_angle: float
    pet: float
    min_gap: float
    first_through: str
    yielder: str
    signalized: bool
    stop_sign: bool
    crosswalk_nearby: bool
    freeway_share: float
    n_lane_crossings: int

    def as_row(self) -> Dict:
        row = asdict(self)
        for k, v in row.items():
            if isinstance(v, float) and math.isfinite(v):
                row[k] = round(v, 1 if k.endswith("_s") else 3)
        return row


def tag_pair(data, a: int, b: int, scenario_id: str = "", pair_source: str = "",
             cfg: Optional[TagConfig] = None) -> PairTag:
    cfg = cfg or TagConfig()
    inter = classify_interaction(data, a, b, cfg)
    conflict = conflict_point(data, a, b, cfg)
    window = interaction_window(data, a, b, cfg, conflict=conflict)
    agent = data["agent"]
    if conflict is not None:
        center = conflict["point"]
    else:
        t = window["peak_step"] if window["peak_step"] >= 0 else max(inter["min_gap_step"], 0)
        center = (agent["position"][a, t, :2] + agent["position"][b, t, :2]) / 2
    scene = classify_scene(data, center, cfg)
    ids = agent.get("id") if hasattr(agent, "get") else None
    id_a = int(ids[a]) if ids is not None else -1
    id_b = int(ids[b]) if ids is not None else -1
    return PairTag(
        scenario_id=scenario_id, agent_a=a, agent_b=b, id_a=id_a, id_b=id_b, pair_source=pair_source,
        category=inter["category"], interaction_type=inter["interaction_type"], scene_type=scene["scene_type"],
        start_s=window["start_s"], end_s=window["end_s"], peak_s=window["peak_s"],
        duration_s=window["duration_s"], start_step=window["start_step"], end_step=window["end_step"],
        peak_step=window["peak_step"], maneuver_a=inter["maneuver_a"], maneuver_b=inter["maneuver_b"],
        initial_angle=inter["initial_angle"], conflict_angle=inter["conflict_angle"], pet=inter["pet"],
        min_gap=inter["min_gap"], first_through=inter["first_through"], yielder=inter["yielder"],
        signalized=scene["signalized"], stop_sign=scene["stop_sign"],
        crosswalk_nearby=scene["crosswalk_nearby"], freeway_share=scene["freeway_share"],
        n_lane_crossings=scene["n_lane_crossings"],
    )


PAIRS_AUTO = "auto"
PAIRS_INTEREST = "interest"
PAIRS_SDC = "sdc"
PAIRS_ALL = "all"
PAIR_MODES = (PAIRS_AUTO, PAIRS_INTEREST, PAIRS_SDC, PAIRS_ALL)


def _present(data, min_presence: float) -> Tensor:
    return data["agent"]["valid_mask"].float().mean(-1) >= min_presence


def interacting_pairs(data, mode: str = PAIRS_AUTO, cfg: Optional[TagConfig] = None,
                      min_presence: float = 0.3, max_pairs: int = 10,
                      prefilter_radius: float = 50.0) -> List[Tuple[int, int, str]]:
    """Pairs to tag, as (a, b, source):

    - ``interest``: the two WOMD objects of interest (``role[:, 1]``), the
      annotated interactive pair of the validation_interactive split
    - ``sdc``: the self-driving car and the agent it interacts with most
      tightly over the clip (smallest PET, then closest gap)
    - ``auto``: ``interest`` when exactly two objects of interest are
      marked, ``sdc`` otherwise
    - ``all``: every pair (up to ``max_pairs``, tightest first) whose paths
      meet within ``time_gap`` seconds or whose footprints come within
      ``proximity_gap``
    """
    if mode not in PAIR_MODES:
        raise ValueError(f"pair mode must be one of {PAIR_MODES}, got {mode!r}")
    cfg = cfg or TagConfig()
    agent = data["agent"]
    role = agent.get("role") if hasattr(agent, "get") else None
    if mode in (PAIRS_AUTO, PAIRS_INTEREST) and role is not None:
        interest = role[:, 1].nonzero(as_tuple=True)[0].tolist()
        if len(interest) == 2:
            return [(interest[0], interest[1], PAIRS_INTEREST)]
        if mode == PAIRS_INTEREST:
            return []
    present = _present(data, min_presence)
    if mode in (PAIRS_AUTO, PAIRS_SDC):
        if role is None or not bool(role[:, 0].any()):
            return []
        sdc = int(role[:, 0].nonzero(as_tuple=True)[0][0])
        best = _tightest_partner(data, sdc, [i for i in present.nonzero(as_tuple=True)[0].tolist() if i != sdc], cfg)
        return [(sdc, best, PAIRS_SDC)] if best is not None else []

    idx = present.nonzero(as_tuple=True)[0].tolist()
    scored = []
    for i, a in enumerate(idx):
        for b in idx[i + 1 :]:
            if not _near_ever(data, a, b, prefilter_radius):
                continue
            s = _pair_strength(data, a, b, cfg)
            if s is not None:
                scored.append((s, a, b))
    scored.sort()
    return [(a, b, PAIRS_ALL) for _, a, b in scored[:max_pairs]]


def _near_ever(data, a: int, b: int, radius: float) -> bool:
    agent = data["agent"]
    both = agent["valid_mask"][a] & agent["valid_mask"][b]
    if not bool(both.any()):
        return False
    d = torch.linalg.norm(agent["position"][a, both, :2] - agent["position"][b, both, :2], dim=-1)
    return bool(d.min() <= radius)


def _pair_strength(data, a: int, b: int, cfg: TagConfig) -> Optional[Tuple[float, float]]:
    lpet = local_pet_series(data, a, b, cfg)
    gap = footprint_gap_series(data, a, b)
    pet, g = float(lpet.min()), float(gap.min())
    if pet <= cfg.time_gap or g <= cfg.proximity_gap:
        return (pet, g)
    return None


def _tightest_partner(data, a: int, candidates: Sequence[int], cfg: TagConfig) -> Optional[int]:
    best, best_key = None, None
    for b in candidates:
        if not _near_ever(data, a, b, 50.0):
            continue
        conflict = conflict_point(data, a, b, cfg)
        pet = conflict["pet"] if conflict is not None else float("inf")
        gap = float(footprint_gap_series(data, a, b).min())
        key = (pet, gap)
        if best_key is None or key < best_key:
            best, best_key = b, key
    return best
