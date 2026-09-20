"""Prints the distribution of ``pt_token.pl_type`` (map-polygon type) values
found in a cached scenario, so you can decide which type ids correspond to
lanes/road edges (i.e. "drivable") in your particular cached WOMD build and
tighten :func:`src.responsibility.geometry.road_deviation_cost`'s
``map_points`` argument accordingly (see that function's docstring).

Example:
    python -m scripts.responsibility.inspect_map_types --data-root /path/to/cache/validation --n 5
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.responsibility.pipeline import build_dataloader  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--n", type=int, default=5, help="Number of scenarios to inspect.")
    return p.parse_args()


def main():
    args = parse_args()
    dataloader = build_dataloader(args.data_root)
    counts = Counter()
    for i, data in enumerate(dataloader):
        if i >= args.n:
            break
        if "pt_token" not in data or "pl_type" not in data["pt_token"]:
            print(f"scenario {i}: no pt_token.pl_type field found.")
            continue
        values = data["pt_token"]["pl_type"].tolist()
        counts.update(values)
        print(f"scenario {i}: pl_type histogram = {Counter(values)}")

    print("\nAggregate pl_type histogram across inspected scenarios:")
    for value, count in sorted(counts.items()):
        print(f"  {value}: {count}")
    print(
        "\nCross-reference these ids with your WOMD preprocessing "
        "(see src/data_preprocess.py / src/smart/utils/preprocess.py) to "
        "identify which correspond to lanes / road edges."
    )


if __name__ == "__main__":
    main()
