# What counts as aggressive

Each scene has two objects of interest: the self-driving car (SDC) and its partner (OOI). Judge
**each one toward the other only** and ignore how either treats third vehicles.

X is aggressive toward Y when **X's own action shifted risk or effort onto Y beyond what the
situation required**. Having the right of way does not excuse a compressed safety margin; being
the one who must yield and yielding is never aggressive.

## Two families, judged differently

**A. Conflicts the traffic law settles** (crossing and turning paths at an intersection without a
working signal). Decide who had the right of way from the jurisdiction profile
(`jurisdictions/<id>.json`, run `scripts/screen_crossing.py`). X is aggressive when it **went
through the conflict first although Y had the right of way**. No threshold on braking, PET or gap
is used here: a lawful first arrival is not aggressive even if Y had to brake or wait, and every
vehicle has to slow for an intersection. In the CAT scenes lawful first arrivals caused the same
braking (2.5 to 3.8 m/s²) as the ones people first labelled aggressive, so braking cannot
separate them.

**B. Conflicts with no right of way involved** (same-direction traffic). These are physical, and
the numbers are the original rubric's, not a statute's:

- **Cut-in**: X moved laterally into Y's lane ahead of Y (X did the lane change, not Y), leaving Y
  a time headway below about 1 s, or making Y brake harder than about 2 m/s².
- **Tailgating**: X followed Y in the same lane at 5 m/s or more with a time headway below about
  0.9 s for a second or longer, by its own choice (not in the first ~3 s after Y cut in ahead of
  X), or closed in on Y with a time to collision below 2 s.
- **Close pass** that X caused: the boxes came within about 0.5 m at more than 3 m/s relative
  speed and X closed in laterally.
- Anything clearly of the same kind (X forced a merge, accelerated to deny a gap Y was already
  taking).

Cut-in was the only kind the first labeller found reliably; tailgating and close pass were rare.
Screens for them are not in this skill: read the tracks (`Scene.position`, `.velocity`,
`.heading`, `.valid`, 10 Hz) and check the numbers above yourself.

## Not aggressive

- No real interaction (far apart, paths never conflict in time).
- Y was the actor (Y cut in, Y violated X's right of way, Y tailgated X): X is the victim, label 0.
- X yielded, waited, followed at a normal headway, or took a gap Y did not have to brake for.
- Creeping in behind a car that has already left the conflict, with a gap of about a metre or more.
- Slow queues (walking pace) and parking manoeuvres with ample room.

Both can be aggressive in one scene; usually at most one is.

## Failure modes to check by eye

- **The clip starts mid-approach.** Anything before step 0 is unseen. At an all-way stop the SDC is
  almost always already at its stop line (arrival time unknown, treated as earliest); a vehicle
  that has been waiting since before the clip is treated as tied with the other (profile
  `unknown_order`). Say which assumption the verdict rests on.
- **Do not label "failed to stop" from a clip.** A stop before step 0 cannot be seen, and among
  the 25 SDCs in the CAT scenes whose approach to a stop line was observed none drove through
  without slowing. Judge by who arrived first and by the yield rules.
- **Who moved.** A cut-in needs X itself to change lanes. Lane matching can treat adjacent lanes
  as one, so a "tailgate" may be two cars side by side in neighbouring lanes.
- **Duplicates.** The same recording cut at different times appears as two scenes (CAT scenes 282
  and 400; 342 and 482). The screen flags them (`same_conflict_as`). Count them as one event.
- **Signal maps.** At a signalised junction the screen does not apply the profile: it keeps the
  light-based verdict of the CAT code (`red-light`, `protected-arrow`, ...), and a junction whose
  light for this movement was not observed (`signal-unknown`) gets no verdict.
- **Other never crossed.** If Y never reached the conflict within the 9 s, "X went first" rests on
  an extrapolation: keep the candidate but mark it low confidence.
