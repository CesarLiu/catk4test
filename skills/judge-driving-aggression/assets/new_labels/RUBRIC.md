# Labelling the two objects of interest: aggressive or not

Each WOMD scene has two objects of interest (OOI): the self-driving car (SDC, blue) and its
partner (OOI, orange). Label **each of the two, toward the other only**, as aggressive (1) or
not (0). Ignore how either behaves toward third vehicles (grey).

## Definition

X is aggressive toward Y when **X's own action shifted risk or effort onto Y beyond what the
situation required**. Holding the right of way does **not** excuse a compressed safety margin.

Aggressive (any one suffices, and it must be X's doing):

1. **Took the right of way**: Y held priority (traffic light, stop sign, the through road, the
   left-turn rule, first at an all-way stop, ...) and X went through the conflict first, so that
   Y had to brake or wait, or with a post-encroachment time (PET) below about 2 s.
2. **Compressed the margin while holding the right of way, or with none**: X entered in front of a Y
   that was visibly not yielding (not slowing) and would have needed hard braking (a_req > 4 m/s²) to stop,
   or X pressed past a Y that was already encroaching on its path at well under 1 m.
3. **Cut in**: X moved laterally into Y's lane ahead of Y (X did the lane change, not Y), leaving Y a
   time headway below about 1 s or making Y brake (> 2 m/s²).
4. **Tailgated**: X followed Y in the same lane at ≥ 5 m/s with a time headway below 0.9 s for a
   second or longer, by its own choice (not in the first ~3 s after Y cut in ahead of X), or closed
   in on Y with a time to collision below 2 s.
5. **Close pass** that X caused: the boxes came within ~0.5 m at > 3 m/s relative speed, and it was X
   that closed in laterally (same direction), or X went second through a crossing squeezing behind Y,
   or X went first past a Y already in its path.
6. Anything else clearly of the same kind (e.g. X forced a merge, accelerated to deny Y a gap Y was
   already taking, ran a red light / stop sign in conflict with Y).

### Unsignalized intersections: judge by US traffic law (California Vehicle Code)

For a crossing or turning conflict at an intersection without a working signal, X is aggressive
toward Y when X **violated Y's right of way** and went through the conflict first. Decide who
had the right of way from the law, with no numeric threshold on braking, PET or gap:

- all-way stop (21800(a)): the first vehicle to arrive and stop goes first;
- simultaneous arrival, or both already waiting when the clip begins (the order is unknown): a
  left turn yields to oncoming traffic (21801(a)), otherwise the vehicle on the left yields to
  the one on the right (21800(b));
- two-way stop: the vehicle at the stop sign yields to cross traffic (21802);
- T junction without control: the road that ends yields to the one that continues (21800(c));
- entering from a driveway or car park: yield to the road (21804(a)).

A stop sign also requires a **complete stop** (21800(a), 21802), but the clip starts mid-approach,
so a stop before step 0 usually cannot be seen. Do **not** use "failure to stop" as the basis
for labelling the SDC aggressive: in the 500 scenes none of the 25 SDCs whose approach to a stop
line is observed (more than 3 m short of it at step 0) drove through without slowing, and the 10
that never slowed below 1.5 m/s were all within 1.6 m of the line at step 0, where a stop may
have happened earlier. Judge such a scene by who arrived first and by the yield rules above.
Scene 197 is the known example (1.9 m/s, 0.8 m before the line at 0 s, accelerating) and is
`unverifiable` in `human_sdc_aggressive.csv`.

Apart from that, **a lawful first arrival is not aggressive**, even if the other vehicle had to brake or wait,
and a vehicle that merely rolled in on a gap the other did not need is not aggressive either.
Do not infer aggression from "the other braked hard at the intersection": every vehicle has to
slow for an intersection, and in the logged scenes lawful first arrivals caused the same braking
(2.5 to 3.8 m/s²) as the ones once labelled aggressive.

In `evidence.csv`, `rule`/`holder` are right_of_way.py's verdict. All-way stops that both were
already waiting at (`all-way-stop/order-unknown`) now count as ties (`order_unknown_tie=True`),
so the left-turn and yield-right rules decide (`all-way-stop/order-unknown-left-turn`,
`all-way-stop/order-unknown-yield-right`). A holder of `ooi` with `sdc_pet` > 0 is a violation
candidate for the SDC. The reviewer still checks the figure: with `sdc_pet` based on an OOI that
never crossed within the 9 s, treat the verdict as low confidence.

Human SDC labels that the law does not support are listed in `human_sdc_aggressive.csv` with
`status=excluded_lawful` (scenes 76, 207, 318, 431, 436, 480): keep them out of the SDC
positives when scoring a labeller, or score them as lawful (0). Scene 197 is `unverifiable`:
leave it out of the SDC evaluation, positive or negative. Scenes 282 and 400 are one
recording, as are 342 and 482, so count each pair as one event.

Not aggressive:

- no real interaction between the two (far apart, paths never conflict in time);
- Y was the actor (Y cut in, Y took the right of way, Y tailgated X): then X is the victim, label X 0;
- X yielded, waited, followed at a normal headway, or took a gap that Y did not have to brake for;
- creeping in behind a car that has already left the conflict (e.g. a left-turner rolling off
  after the oncoming car passed) with a gap of about a metre or more;
- slow queues (walking pace), parking manoeuvres with ample room.

Both can be aggressive in one scene; usually at most one is.

## The evidence

`evidence.csv` has one row per scene (column `scene`). Shared columns: `rule` and `holder`
(right_of_way.py's rule and who held priority: sdc, ooi or none), `conflict`, `angle` (deg between
paths at the first crossing), `sdc_control`/`ooi_control` (GO, RED, STOP, ARROW, NONE, UNKNOWN),
`*_turn`, `*_changed_lane`, `min_gap` (m, exact box gap), `t_min_gap` (s), `rel_speed_at_min`.
Then `sdc_*` (the SDC toward the OOI) and `ooi_*` (the OOI toward the SDC): `pet` (s, positive: this
one went first), `demand` (a_req, m/s², braking this one's entry asked of the other; 20 means the other
was already in its zone), `follow_thw`, `follow_close_s`, `follow_ttc`, `cut_in_*`, `other_brake_after`,
`candidate` (1: a rule fired) and `fired` (which rule, with numbers).

`figures/<scene>.png`:

- left: the map at the event step (the first rule that fired, else the closest approach). Blue
  SDC and orange OOI boxes, both full tracks with a dot and the time (s) every second, other agents
  grey, traffic-light stop points coloured by their state at that step (red/yellow/green squares,
  grey unknown), stop signs as dark-red octagons, the first crossing of the two tracks as a red star.
  Lane centrelines light grey, road edges dark grey, crosswalks yellow.
- right, top: each one's distance past the conflict point (m) over time, its conflict zone shaded.
- right, middle: both speeds (m/s).
- right, bottom: the box gap (m, black) and the time headways (dashed, right axis: blue the SDC
  behind the OOI, orange the OOI behind the SDC; dotted line 0.9 s). Dotted vertical line: the event.

The rules are **candidates only**. Known failure modes to check by eye:

- right_of_way.py misreads some maps (wrong holder, "too-short", "signal-unknown"): judge
  priority yourself from the map, lights and stop signs where you can;
- lane matching can treat adjacent lanes as one, so "tailgate" or "cut_in" may be two cars side by
  side in neighbouring lanes;
- who moved: a cut-in needs X itself to change lanes; check the tracks;
- a_req assumes a 1 s reaction, so for a slow, already-braking Y it overstates.

If the figure is not enough, you may load the scene: `~/venvs/cat39/bin/python` from
`/home/Cesar/catspond-cr` with `from responsibility.scene import Scene; s = Scene.load("raw_scenes_500/<scene>.pkl")`
(`s.sdc`, `s.objects_of_interest`, `s.position[i, t, :2]`, `s.velocity`, `s.heading`, `s.valid`; 10 Hz, 91 steps).

## Output

One CSV row per scene: `scene,sdc_aggressive,sdc_reason,ooi_aggressive,ooi_reason,confidence`.
Reasons in English, one sentence each with the deciding numbers and times (e.g. "OOI changed lanes
into the SDC's lane at 6.4 s, 0.24 s ahead of it"; "no conflict: paths separate, closest 14 m").
`confidence`: high, medium or low (low when priority or who moved is unclear).
