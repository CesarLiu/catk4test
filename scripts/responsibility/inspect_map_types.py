"""Checks the road-deviation reference on real data.

The road-deviation cost penalises distance beyond --drivable-half-width to
the nearest reference map point (``dataset_builder.get_map_points``: the
vehicle-lane centerlines by default). Logged vehicles drive on the road, so
for them that cost should almost always be zero; this script reports how
often it is, against using every map point, plus the map type histograms.

Example:
    python -m scripts.responsibility.inspect_map_types --data-root data/womd_cache/validation --n 20
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.responsibility.dataset_builder import get_map_points  # noqa: E402
from src.responsibility.pipeline import iter_scenarios  # noqa: E402

POLYGON_NAMES = ["lane", "road_edge", "road_line", "crosswalk"]
POINT_NAMES = {0: "freeway lane", 1: "surface-street lane", 2: "stop sign", 3: "bike lane",
               4: "road edge (boundary)", 5: "road edge (median)", 6: "broken line",
               7: "solid line", 8: "double line", 9: "crosswalk"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--n", type=int, default=20, help="Number of scenarios to inspect.")
    p.add_argument("--drivable-half-width", type=float, default=3.0)
    p.add_argument("--moving-speed", type=float, default=1.0, help="m/s separating moving from stationary.")
    return p.parse_args()


def main():
    args = parse_args()
    polygons, points = Counter(), Counter()
    groups = ("moving", "stationary")
    dist = {(g, name): [] for g in groups for name in ("vehicle lanes", "all map points")}
    for i, _, data in iter_scenarios(args.data_root):
        if i >= args.n:
            break
        polygons.update(data["pt_token"]["pl_type"].tolist())
        points.update(data["pt_token"]["type"].tolist())
        agent = data["agent"]
        vehicle = agent["type"] == 0
        valid = agent["valid_mask"][vehicle]
        pos = agent["position"][vehicle][..., :2][valid]  # every logged vehicle pose
        speed = torch.linalg.norm(agent["velocity"][vehicle], dim=-1)[valid]
        # parked cars sit off the lane network by design; the cost matters for moving ones
        split = {"moving": speed > args.moving_speed, "stationary": speed <= args.moving_speed}
        for name, lanes_only in (("vehicle lanes", True), ("all map points", False)):
            nearest = torch.cdist(pos, get_map_points(data, vehicle_lanes_only=lanes_only)).min(dim=-1).values
            for g in groups:
                dist[(g, name)].append(nearest[split[g]])

    print("polygon types:", {POLYGON_NAMES[k]: v for k, v in sorted(polygons.items())})
    print("point types:  ", {POINT_NAMES.get(k, k): v for k, v in sorted(points.items())})
    print(f"\nlogged vehicle poses vs the nearest reference point "
          f"(cost is zero within {args.drivable_half_width} m):")
    for (group, name), parts in dist.items():
        d = torch.cat(parts)
        if d.numel() == 0:
            continue
        within = (d <= args.drivable_half_width).float().mean().item()
        print(f"  {group:10s} {name:15s}  median {d.median():5.2f} m   p95 {d.quantile(0.95):5.2f} m   "
              f"zero cost for {100 * within:5.1f}% of {d.numel()} poses")


if __name__ == "__main__":
    main()
