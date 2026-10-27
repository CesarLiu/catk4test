# What the California profile was checked against

Source: CAT's 500 WOMD scenes (`raw_scenes_500`), labelled first by a rubric-and-evidence pass and
then compared with 13 scenes a person marked "SDC aggressive". Counts below are from that work.

## What the first labeller missed

It found only the two cut-ins (scenes 342 and 482, one event). Eleven human positives were
missed, mostly all-way-stop crossings, for these reasons:

- its right-of-way rules gave the SDC the priority (arrival time is cut off at the clip start, so
  the SDC counted as first in 45 of 48 all-way-stop scenes) or no priority (both waiting at the
  start), and a violation needs the other vehicle to hold priority;
- its margin test needs the other vehicle to have braked hard (a_req > 4 m/s²), but the other
  vehicle was usually already stopped;
- it only used a PET below 2 s when the other vehicle held priority.

## Decision by traffic law

The person's rule: judge unsignalized intersections by US traffic law, with no numeric thresholds.
Applying it:

- Treating "both already waiting at the start" as a tie (left turn yields to oncoming, then
  yield-right) recovers scenes 137, 282/400 and 423, with no change in the SDC-first all-way-stop
  scenes that were not labelled.
- Scenes 76, 207, 318, 431 and 480: the SDC reached its stop line first, so going first is lawful.
  The person judged these labels an error, as for 436 (SDC on green against a creeping right
  turner).
- Scene 197: the SDC was first but already moving at 1.9 m/s, 0.8 m before the line at step 0, so
  whether it stopped cannot be seen. A screen over all 500 scenes finds 10 SDCs like this and none
  among the 25 whose approach was observed. Marked `unverifiable`.

The full table is `assets/human_sdc_aggressive_cat500.csv`.

## Limits

- Only 3 SDC events agree with the profile (137, 282/400, 423), plus 1 cut-in. This is a sanity
  check, not an accuracy figure.
- The human set marked positives only; unflagged scenes were not independently labelled, so
  precision is unknown. The profile also flags scenes nobody marked (for example 187, 439 and 448,
  where the other vehicle never crossed within the clip), which need a person's look.
- The person decided that braking does not count in the law-settled family; other uses may differ.
