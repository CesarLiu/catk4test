"""Screen WOMD-style scenes for right-of-way violations at intersections without a working signal.

For every scene it takes the self-driving car (SDC) and the other object of interest (OOI), or a
pair you name, finds where their paths cross, decides who had the right of way under a
jurisdiction profile (``jurisdictions/<id>.json``) and writes one row per scene to a CSV: who went
first, who held priority, and so whether the SDC or the OOI violated the other's right of way.

A violation is a *candidate*: the reviewer still looks at the scene (see SKILL.md). Cut-ins,
tailgating and close passes are not screened here (no right of way is involved); judge them with
rubric.md.

Input is a folder of ScenarioNet/MetaDrive scene pickles (CAT's ``raw_scenes_500/*.pkl``), or one
such file. It reads them with the ``responsibility`` package of the CAT repository (the scene
loader, the lane graph, the conflict finder), found through ``--cat-root`` or $CAT_ROOT, else by
walking up from here. Convert a WOMD tfrecord with CAT's ``convert_womd_split.py`` first.

    python screen_crossing.py --scenes raw_scenes_500 --profile us-california --out screen.csv
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jurisdiction  # noqa: E402

DT = 0.1
SIGNAL_CASES_PASSED_THROUGH = ("following", "same-direction", "driveway", "oncoming", "signal-unknown",
                               "no-conflict", "too-short", "not-seen-together")


def find_cat_root(explicit=None):
    candidates = [explicit, os.environ.get("CAT_ROOT")]
    here = Path(__file__).resolve()
    candidates += [p for p in here.parents]
    candidates += [p / "cat" for p in here.parents]
    for c in candidates:
        if c and (Path(c) / "responsibility" / "right_of_way.py").exists():
            return Path(c)
    sys.exit("cannot find the CAT repository (responsibility/right_of_way.py): pass --cat-root or set CAT_ROOT")


def scene_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    files = list(path.glob("*.pkl"))
    return sorted(files, key=lambda p: (int(p.stem) if p.stem.isdigit() else float("inf"), p.stem))


def arc_and_speed(scene, i):
    idx = np.flatnonzero(scene.valid[i])
    pos = scene.position[i, idx, :2].astype(float)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
    return idx, arc, np.linalg.norm(scene.velocity[i, idx], axis=1)


def time_at(idx, arc, s):
    """Time (s) the agent has covered s metres of its path; NaN if never within the clip."""
    return float(idx[np.argmax(arc >= s)]) * DT if (arc >= s).any() else float("nan")


def stop_line_arc(R, scene, graph, stop_lanes, agent, path):
    """Arc length at which the agent's path meets the stop line of the stop-controlled lane it uses."""
    best = None
    for lane in stop_lanes:
        try:
            s, dist = path.project(graph.points[lane][0])
        except KeyError:
            continue
        if dist < 2.5 and (best is None or dist < best[1]):
            best = (s, dist)
    return None if best is None else best[0]


def screen_pair(R, lanes_mod, scene, a, o, profile, tie_steps):
    params = R.RightOfWayParams()
    pr = R.priority(scene, a, o, scene.n_steps - 1, params)
    row = {"case": pr.case, "holder": "none", "went_first": "", "pet_s": "", "sdc_violation": 0,
           "ooi_violation": 0, "confidence": "", "flags": ""}
    if pr.conflict is None or pr.approaches is None:
        return row
    A, B = pr.approaches
    row.update(conflict_x=round(float(pr.conflict.point[0]), 1), conflict_y=round(float(pr.conflict.point[1]), 1),
               conflict=pr.conflict.kind, angle_deg=round(math.degrees(pr.conflict.angle), 1),
               sdc_control=A.control, ooi_control=B.control, sdc_turn=A.turn, ooi_turn=B.turn)
    holder = {a: "sdc", o: "ooi", None: "none"}[pr.holder]
    case = pr.case
    if pr.case not in SIGNAL_CASES_PASSED_THROUGH and jurisdiction.handles(A, B):
        h, case = jurisdiction.decide(profile, A, B, tie_steps)
        holder = {"a": "sdc", "b": "ooi", None: "none"}[h]
    row["case"], row["holder"] = case, holder

    ia, arc_a, v_a = arc_and_speed(scene, a)
    io, arc_o, v_o = arc_and_speed(scene, o)
    t_a, t_o = time_at(ia, arc_a, pr.conflict.s[0]), time_at(io, arc_o, pr.conflict.s[1])
    flags = []
    if t_a == t_a and t_o == t_o:
        row["went_first"] = "sdc" if t_a < t_o else "ooi" if t_o < t_a else "tie"
        row["pet_s"] = round(abs(t_o - t_a), 1)
    elif t_a == t_a or t_o == t_o:
        row["went_first"] = "sdc" if t_a == t_a else "ooi"
        flags.append("other_never_crossed_in_clip")
    sdc_viol = holder == "ooi" and row["went_first"] == "sdc"
    ooi_viol = holder == "sdc" and row["went_first"] == "ooi"
    row["sdc_violation"], row["ooi_violation"] = int(sdc_viol), int(ooi_viol)

    if "order-unknown" in case:
        flags.append("both_waiting_at_clip_start")
    if A.control == "stop" and B.control == "stop" and (A.arrived == -1 or B.arrived == -1):
        flags.append("arrival_censored")
    graph = lanes_mod.lane_graph(scene)
    stop_lanes, _ = R._stop_signs(scene)
    for role, agent, arc, v in (("sdc", a, arc_a, v_a), ("ooi", o, arc_o, v_o)):
        s_line = stop_line_arc(R, scene, graph, stop_lanes, agent, R._Path(scene, agent, scene.n_steps - 1, params.extend))
        if s_line is None:
            continue
        k = int(np.argmax(arc >= s_line)) if (arc >= s_line).any() else None
        if k is None:
            continue
        flags.append("%s_v_at_step0=%.1f" % (role, v[0]))
        if s_line > 3.0 and v[:k + 1].min() >= 1.5:  # an observed approach that never slowed
            flags.append("%s_rolled_stop_observed" % role)
    row["flags"] = ";".join(flags)
    if sdc_viol or ooi_viol:
        row["confidence"] = ("low" if "other_never_crossed_in_clip" in flags
                             else "medium" if "both_waiting_at_clip_start" in flags else "high")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", required=True, help="folder of scene pickles, or one file")
    ap.add_argument("--profile", default="us-california", help="profile id or path: %s" % ", ".join(jurisdiction.list_profiles()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cat-root")
    ap.add_argument("--agent", help="track id of the first agent (default: the SDC); needs --other")
    ap.add_argument("--other", help="track id of the second agent (default: the other object of interest)")
    ap.add_argument("--tie-seconds", type=float, default=1.0, help="arrivals this close count as simultaneous")
    args = ap.parse_args()

    root = find_cat_root(args.cat_root)
    sys.path.insert(0, str(root))
    from responsibility import right_of_way as R  # noqa: E402
    from responsibility import lanes as lanes_mod  # noqa: E402
    from responsibility.scene import Scene  # noqa: E402

    profile = jurisdiction.load_profile(args.profile)
    if profile["status"].startswith("draft"):
        print("WARNING: profile %s is a draft: %s" % (profile["id"], profile["status"]), file=sys.stderr)
    tie_steps = int(round(args.tie_seconds / DT))
    rows, skipped = [], 0
    for path in scene_files(args.scenes):
        try:
            scene = Scene.load(path)
            if args.agent:
                a, o = scene.index(args.agent), scene.index(args.other)
            else:
                others = [i for i in scene.objects_of_interest if i != scene.sdc]
                if not others:
                    skipped += 1
                    continue
                a, o = scene.sdc, others[0]
            row = screen_pair(R, lanes_mod, scene, a, o, profile, tie_steps)
        except Exception as exc:  # one bad scene must not stop a batch
            print("skip %s: %s: %s" % (path.name, type(exc).__name__, exc), file=sys.stderr)
            skipped += 1
            continue
        row.update(scene=path.stem, scenario_id=scene.scenario_id, jurisdiction=profile["id"])
        rows.append(row)

    import csv
    cols = ["scene", "scenario_id", "jurisdiction", "case", "holder", "went_first", "pet_s", "sdc_violation",
            "ooi_violation", "confidence", "conflict", "angle_deg", "conflict_x", "conflict_y", "sdc_control",
            "ooi_control", "sdc_turn", "ooi_turn", "flags"]
    # the same recording cut at different times gives the same conflict point and angle
    seen = {}
    for r in rows:
        if "conflict_x" in r:
            seen.setdefault((r["conflict_x"], r["conflict_y"], r["angle_deg"]), []).append(r["scene"])
    for r in rows:
        if "conflict_x" in r:
            same = [s for s in seen[(r["conflict_x"], r["conflict_y"], r["angle_deg"])] if s != r["scene"]]
            if same:
                r["flags"] = (r["flags"] + ";" if r["flags"] else "") + "same_conflict_as=" + "/".join(same)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    n_s, n_o = sum(r["sdc_violation"] for r in rows), sum(r["ooi_violation"] for r in rows)
    print("%d scenes screened (%d skipped); SDC violation candidates %d, OOI %d -> %s"
          % (len(rows), skipped, n_s, n_o, args.out))


if __name__ == "__main__":
    main()
