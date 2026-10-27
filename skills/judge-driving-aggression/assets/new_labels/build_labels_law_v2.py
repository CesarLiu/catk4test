"""Build labels_law_v2.csv from labels.csv (the first labelling) and screen_law_us.csv (the law screen).

Rules, as agreed:
* Same-direction kinds the first labeller found (cut_in, tailgate, close_pass, signal) are kept as
  labelled; labels.csv is not touched.
* Crossing and turning conflicts at intersections without a working signal are decided by the law
  screen (skills/judge-driving-aggression, profile us-california): the vehicle that went first
  although the other held the right of way is aggressive.
* High and medium confidence candidates are labelled 1. Low confidence candidates stay 0 in the main
  file (source "law_pending") and are listed in labels_law_v2_low_confidence.csv for review.
* Scenes the human marked in human_sdc_aggressive.csv as excluded_lawful or unverifiable keep
  their law verdict and carry that status in eval_status.

    python build_labels_law_v2.py
"""

import csv
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PHYSICAL = {"cut_in", "tailgate", "close_pass", "signal"}

LAW = {
    "all-way-stop/first-in": "the other arrived at its stop line first (CVC 21800(a))",
    "all-way-stop/order-unknown-yield_right": "both were already waiting when the clip began, so they count as a tie, and the vehicle on the left yields to the one on the right (CVC 21800(b))",
    "all-way-stop/order-unknown-oncoming_left_yields": "both were already waiting when the clip began, so they count as a tie, and a left turn yields to oncoming traffic (CVC 21801(a))",
    "all-way-stop/yield_right": "they arrived within 1 s of each other and the vehicle on the left yields to the one on the right (CVC 21800(b))",
    "all-way-stop/yield-right": "they arrived within 1 s of each other and the vehicle on the left yields to the one on the right (CVC 21800(b))",
    "all-way-stop/oncoming_left_yields": "they arrived within 1 s of each other and a left turn yields to oncoming traffic (CVC 21801(a))",
    "all-way-stop/left-turn": "they arrived within 1 s of each other and a left turn yields to oncoming traffic (CVC 21801(a))",
    "stop-sign": "it faced a stop sign and the other did not, so it had to yield to the cross traffic (CVC 21802)",
    "through-road": "its road ends at the junction and has to yield to the road that continues (CVC 21800(c))",
    "driveway": "it entered from off the mapped road (driveway or car park) and has to yield to the road (CVC 21804(a))",
    "uncontrolled/oncoming_left_yields": "no control is mapped and a left turn yields to oncoming traffic (CVC 21801(a))",
    "red-light": "it entered on red while the other had a green light (CVC 21453)",
}
CAVEAT = {
    "both_waiting_at_clip_start": "the order before the clip is unseen",
    "other_never_crossed_in_clip": "the other vehicle never crossed within the 9 s, so going first is extrapolated",
}


def law_reason(role, row):
    me, other = ("SDC", "OOI") if role == "sdc" else ("OOI", "SDC")
    turn = lambda r: row["%s_turn" % r]
    pet = "" if pd.isna(row["pet_s"]) else " (PET %.1f s)" % row["pet_s"]
    basis = LAW.get(row["case"], row["case"])
    notes = [CAVEAT[f] for f in str(row["flags"]).split(";") if f in CAVEAT]
    text = "%s (%s) went through the conflict first%s although the %s (%s) had the right of way: %s." % (
        me, turn(role), pet, other, turn("ooi" if role == "sdc" else "sdc"), basis)
    if notes:
        text = text[:-1] + "; " + "; ".join(notes) + "."
    return text


def main():
    labels = pd.read_csv(HERE / "labels.csv").set_index("scene")
    screen = pd.read_csv(HERE / "screen_law_us.csv").set_index("scene")
    human = pd.read_csv(HERE / "human_sdc_aggressive.csv").set_index("scene")
    rows, low = [], []
    for scene in labels.index:
        lab, scr = labels.loc[scene], screen.loc[scene]
        out = {"scene": scene, "scenario_id": lab.scenario_id}
        for role in ("sdc", "ooi"):
            agg, kind, reason = int(lab["%s_aggressive" % role]), lab["%s_kind" % role], lab["%s_reason" % role]
            conf, src = lab.confidence, "original"
            if kind in PHYSICAL:  # kept as labelled
                if scr["%s_violation" % role] == 1 and scr.confidence in ("high", "medium"):
                    reason += " [law screen also flags a right-of-way violation: %s]" % law_reason(role, scr)
            elif scr["%s_violation" % role] == 1:
                text = law_reason(role, scr)
                if scr.confidence in ("high", "medium"):
                    agg, kind, reason, conf, src = 1, "right_of_way", text, scr.confidence, "law"
                else:
                    kind, src = "", "law_pending"
                    low.append({"scene": scene, "scenario_id": lab.scenario_id, "agent": role.upper(),
                                "case": scr["case"], "pet_s": scr["pet_s"], "flags": scr["flags"],
                                "candidate_reason": text})
                    conf = "low"
            out.update({"%s_aggressive" % role: agg, "%s_kind" % role: kind if isinstance(kind, str) else "",
                        "%s_reason" % role: reason, "%s_confidence" % role: conf, "%s_source" % role: src})
        status = ""
        if scene in human.index and human.loc[scene, "status"] in ("excluded_lawful", "unverifiable"):
            status = human.loc[scene, "status"]
            out["sdc_source"] = "human_" + status if out["sdc_source"] == "original" else out["sdc_source"]
        out["eval_status"] = status
        flags = [f for f in str(scr["flags"]).split(";") if f.startswith("same_conflict_as=")]
        out["same_event_as"] = flags[0].split("=")[1] if flags else ""
        rows.append(out)
    cols = ["scene", "scenario_id", "sdc_aggressive", "sdc_kind", "sdc_reason", "sdc_confidence", "sdc_source",
            "ooi_aggressive", "ooi_kind", "ooi_reason", "ooi_confidence", "ooi_source", "eval_status", "same_event_as"]
    with open(HERE / "labels_law_v2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    pd.DataFrame(low).to_csv(HERE / "labels_law_v2_low_confidence.csv", index=False, encoding="utf-8")
    d = pd.DataFrame(rows)
    print("scenes:", len(d))
    for role in ("sdc", "ooi"):
        print(role, "aggressive:", int(d[role + "_aggressive"].sum()), d[role + "_source"].value_counts().to_dict(),
              "| by kind:", d[d[role + "_aggressive"] == 1][role + "_kind"].value_counts().to_dict())
    print("low-confidence candidates:", len(low))


if __name__ == "__main__":
    main()
