---
name: judge-driving-aggression
description: Decide whether an agent's driving toward another agent is aggressive in WOMD-style scenes (Waymo Open Motion Dataset / ScenarioNet scene files), judging crossing and turning conflicts at intersections by the traffic law of a chosen country or region, and cut-ins, tailgating and close passes by physical rules. Use this whenever the user wants to label, audit or score aggressive driving of the SDC or objects of interest in motion-prediction or driving-log scenes, check or explain human aggression labels, find who violated the right of way, or adapt the aggression criteria to another country (left-hand traffic, China, Germany, other traffic codes), even if they only say "is the SDC aggressive here", "who had the right of way", or "label these scenes".
---

# Judging aggressive driving in WOMD-style scenes

Aggressive means: X's own action shifted risk or effort onto Y beyond what the situation required.
For conflicts at intersections the traffic law decides what X owed Y, so the criteria depend on
the country. For same-direction traffic (cut-in, tailgating, close pass) they are physical.

The method came out of labelling CAT's 500 WOMD scenes and comparing with a person's labels. Its
main lesson: a first labeller that judged by braking and PET alone missed every crossing
violation, and a person who judged by the law found that most "aggressive" crossings were lawful.
`references/validation-us-california.md` has the details; read it before trusting a result.

## Workflow

1. **Check the input.** The screening script reads ScenarioNet / MetaDrive scene pickles (one
   per scene, as CAT's `raw_scenes_500/*.pkl`) through the `responsibility` package of the CAT
   repository. It finds the repo by `--cat-root`, `$CAT_ROOT`, or by walking up from the skill;
   ask the user where it is if it fails. A WOMD tfrecord must be converted first with CAT's
   `scripts/responsibility/convert_womd_split.py`. Each scene needs an SDC and a second object of
   interest (or pass `--agent` and `--other` track ids for one scene).

2. **Pick the jurisdiction.** Ask the user if they have not said. WOMD is entirely US cities, so
   `us-california` is the default there; other datasets need the profile of where they were
   recorded. `python scripts/jurisdiction.py` lists the profiles. Only `us-california` has been
   validated; the others are drafts (the script warns). If the user needs a country with no
   profile, follow "Adding a jurisdiction" in `references/jurisdictions.md` and say plainly that
   the result is unvalidated until a local expert has checked it.

3. **Screen the crossing conflicts.**

   ```bash
   python scripts/screen_crossing.py --scenes <folder-or-file> --profile us-california --out screen.csv
   ```

   One row per scene: who held priority (`holder`), who `went_first` through the conflict, `pet_s`,
   `sdc_violation` and `ooi_violation` (the other vehicle held priority and this one went first),
   a `confidence`, and `flags` that say what the verdict rests on (`both_waiting_at_clip_start`,
   `other_never_crossed_in_clip`, `same_conflict_as=<scene>` for duplicates, ...). These are
   candidates, not labels.

4. **Review every candidate and every scene the screen cannot see.** Open the tracks
   (`responsibility.scene.Scene.load(path)`: `position`, `velocity`, `heading`, `valid`, 10 Hz,
   91 steps; `sdc`, `objects_of_interest`) and check against `references/rubric.md`:
   - who arrived at the stop line first and whether the vehicle that went first really had to
     yield (the clip's first second is the SDC already at its line; a flagged tie rests on that);
   - whether the other vehicle actually crossed in the clip (`other_never_crossed_in_clip` rows
     are low confidence);
   - for same-direction scenes, which vehicle changed lanes and what the headway and braking were.
   The screen does not find cut-ins, tailgating or close passes; apply the rubric's numbers to the
   tracks directly.

5. **Write the labels.** One row per scene: `scene, sdc_aggressive, sdc_reason, ooi_aggressive,
   ooi_reason, confidence`. Each reason is one English sentence with the deciding rule and
   numbers ("SDC left-turned across the oncoming OOI at 3.6 s while both had been waiting since
   before the clip; CVC 21801(a) gives the OOI priority"). Name the jurisdiction. Merge duplicate
   scenes into one event when you count.

## Rules of judgement, and why

- **A lawful first arrival is not aggressive**, even if the other vehicle braked or waited.
  Every vehicle has to slow for an intersection, and in the CAT scenes lawful first arrivals
  caused the same braking as ones first labelled aggressive, so braking cannot tell them apart.
- **Do not label "failed to stop" from a clip.** A stop before step 0 cannot be seen, so a vehicle
  creeping at its line at step 0 proves nothing; among 25 SDCs whose approach was observed none
  rolled through.
- **State the assumption behind a tie.** Two vehicles already waiting at step 0 are treated as
  simultaneous arrivals, so the left-turn and yield-right rules decide. That is a legal fiction;
  say so in the reason and give it medium confidence.
- **Both directions.** Judge the SDC toward the OOI and the OOI toward the SDC; usually at most
  one is aggressive. If the other vehicle was the actor, the victim is not aggressive.
- **Never present an unvalidated profile's result as a finding.** A draft profile is a starting
  point for a conversation with someone who knows the local code.

## Files

- `scripts/screen_crossing.py`: the screen. `scripts/jurisdiction.py`: the profile engine
  (`python scripts/jurisdiction.py` self-tests it).
- `jurisdictions/*.json`: the profiles. `references/jurisdictions.md`: schema, rule catalogue, how
  to add a country.
- `references/rubric.md`: the definition, the physical rules for same-direction traffic, failure
  modes. `references/validation-us-california.md`: what was checked and its limits.
- `assets/human_sdc_aggressive_cat500.csv`: the person's SDC labels on CAT's 500 scenes with the
  decision for each (kept, excluded as lawful, unverifiable), usable as a small test set.
